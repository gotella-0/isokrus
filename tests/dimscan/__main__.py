"""Запуск всех вариантов подсветки размерных отрезков.

    python tests/dimscan/__main__.py                 # все варианты, все листы
    python tests/dimscan/__main__.py --pages 1,2     # только два листа
    python tests/dimscan/__main__.py --only ghost    # один вариант
    python tests/dimscan/__main__.py --check         # сверка с эталоном
    python tests/dimscan/__main__.py --dump-json     # + выгрузка отрезков

Ни одного обращения к LLM: всё считается из PDF, поэтому прогон
воспроизводим и не зависит от наличия ключей в ``.env``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from isokrus import dimscan as detect  # noqa: E402
from tests.dimscan import reference, render  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"

VARIANTS = {
    "vector": render.render_vector,
    "ghost": render.render_ghost,
    "nodes": render.render_nodes,
}


def parse_pages(raw: str | None, total: int) -> list[int]:
    if not raw or raw.lower() == "all":
        return list(range(1, total + 1))
    out: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, _, b = chunk.partition("-")
            out.update(range(int(a), int(b) + 1))
        elif chunk:
            out.add(int(chunk))
    return sorted(p for p in out if 1 <= p <= total)


def report(results: list[detect.DimResult]) -> None:
    """Сводка по листам: что нашлось и сколько из этого в сумме."""
    print()
    print("  лист  размеров  выносок  без подписи  сумма как есть  "
          "сумма без повторов  толщина  масштаб")
    for result in results:
        s = result.summary()
        chain = result.chain_total()
        print(f"  {s['page']:>4}  {s['segments']:>7}  {s['pointers']:>6}  "
              f"{s['segments'] - s['labelled']:>10}  {chain['sum_all']:>13.0f}  "
              f"{chain['sum_without_duplicates']:>19.0f}  "
              f"{s['width_used']:7.2f}  {result.stats['scale']:6.3f}")
    total_segments = sum(len(r.lines) for r in results)
    sum_all = sum(r.chain_total()["sum_all"] for r in results)
    sum_unique = sum(r.chain_total()["sum_without_duplicates"] for r in results)
    print(f"  итого: размеров {total_segments}, "
          f"сумма как есть {sum_all:.0f}, без повторов {sum_unique:.0f} "
          f"(наивный счёт завышал на {sum_all - sum_unique:.0f})")

    scales = {round(r.stats["scale"], 3) for r in results}
    if len(scales) > 1:
        # Пороги приводятся к масштабу каждого листа, поэтому разные
        # множители — это норма, а не повод для тревоги. Но если комплект
        # смешанный (часть листов в другом масштабе экспорта), об этом лучше
        # знать сразу, чем гадать потом, откуда взялась разница в суммах.
        print(f"  масштаб листов различается: {sorted(scales)} — "
              f"пороги подстроены под каждый, но проверьте, что комплект однороден")

    orphans = [
        (r.page_number, n.text) for r in results for n in r.orphan_numbers
    ]
    if orphans:
        # Число, к которому не привязался ни один размер, — это либо
        # регрессия, либо неучтённый вид размера. Молчать об этом нельзя:
        # на листе выглядит так, будто размер просто нет.
        print()
        print("  числа без размера: " + ", ".join(f"{p}:{t}" for p, t in orphans))

    mismatched = [
        (r.page_number, o)
        for r in results
        for o in r.chain_total()["overall_segments"]
        if not o["arithmetic_ok"]
    ]
    if mismatched:
        # Не ошибка, а факт чертежа: обобщённый размер меряет всю трассу, а
        # звенья под ним нарисованы не все. Сообщаем, чтобы сумма не
        # читалась как сбой детектора.
        print()
        print("  обобщённые размеры, чьи звенья не дают их сумму "
              "(на чертеже отрисованы не все участки):")
        for page, o in mismatched:
            print(f"    лист {page}: {o['label']} при сумме звеньев {o['children_sum']:.0f}")


def check(results: list[detect.DimResult]) -> None:
    """Сверка с эталонными длинами трасс.

    Два независимых вопроса, и их важно не путать — отчёт разводит их
    в разные колонки именно поэтому.

    **Есть ли все размеры** — набирается ли эталонная сумма набором
    найденных чисел. Это проверка поиска: пропущен размер или за размер
    принято чужое число. Да на всех листах.

    **Сколько в сумме** — что даёт нынешняя иерархия «обобщённый /
    вложенный». Это другой вопрос и другим слоем не решается: рисунок
    нарисован не в масштабе (на листе 5 от 1.34 до 20.95 мм на пункт), а
    ряды размеров разных ветвей перекрываются по проекции, поэтому по
    геометрии размерной линии родителя не определить. Нужна топология
    трассы — она ищется в `research/exp_04` по координатам привязки.

    Итог по слоям такой: поиск полон, иерархия не доделана.
    """
    print()
    print("  1. ПОИСК: есть ли все размеры (эталон набирается найденными числами)")
    print("  2. ИЕРАРХИЯ: сколько даёт текущее разбиение на обобщённые и вложенные")
    print()
    print("  лист  размеров  сумма как есть  эталон  все размеры найдены  "
          "не вошли в эталон")
    print("  ----  -------  ---------------  ------  ------------------  "
          "------------------")
    bad: list[str] = []
    for result in results:
        page = result.page_number
        chain = result.chain_total()
        values = [l.value for l in result.lines if l.value is not None]
        target = reference.EXPECTED.get(page)
        subset = reference.reachable(values, target) if target else None
        if target is None:
            mark = "—"
        elif subset is not None:
            mark = "да"
        else:
            mark = "НЕТ"
            bad.append(
                f"лист {page}: эталон {target:.0f} не набирается найденными числами"
            )
        print(
            f"  {page:>4}  {len(values):>7}  {chain['sum_all']:>15.0f}  "
            f"{target:>6.0f}  {mark:>18}  {_not_counted(values, subset)}"
        )

    print()
    print("  лист  без повторов  эталон  разница   что не сошлось")
    for result in results:
        page = result.page_number
        chain = result.chain_total()
        target = reference.EXPECTED.get(page)
        got = chain["sum_without_duplicates"]
        if target is None:
            continue
        print(
            f"  {page:>4}  {got:>13.0f}  {target:>6.0f}  {got - target:>+8.0f}   "
            f"{'—' if abs(got - target) < 1 else 'не все вложенные сняты'}"
        )
    same = sum(
        1 for r in results
        if r.page_number in reference.EXPECTED
        and abs(r.chain_total()["sum_without_duplicates"]
                - reference.EXPECTED[r.page_number]) < 1
    )
    print()
    if bad:
        print("  ПОИСК НЕПОЛОН: " + "; ".join(bad))
        print("  (не хватает размера или за размер принято чужое число)")
    else:
        print(f"  1. Поиск полон на всех {len(results)} листах.")
    print(f"  2. Иерархия сходится с эталоном на {same} листах из {len(results)}.")
    print("     Рисунок нарисован не в масштабе, и по геометрии размерной")
    print("     линии нельзя понять, какой размер обобщённый. Сумма вложенных")
    print("     участков при этом не обязана равняться обобщённому.")


def _not_counted(values: Sequence[float], subset: Sequence[float] | None) -> str:
    """Числа, не вошедшие в набор, — по кратности, а не по значению.

    «6000» на листе 8 встречается четыре раза, и проверка ``v not in subset``
    по значению выбросила бы все четыре, даже если в набор вошло одно.
    """
    if subset is None:
        return "-"
    left = Counter(subset)
    out: list[float] = []
    for value in values:
        if left[value] > 0:
            left[value] -= 1
        else:
            out.append(value)
    return str(sorted(round(v) for v in out))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Подсветка размерных отрезков")
    parser.add_argument("--pdf", default="02_Изометрии_10_листов.pdf")
    parser.add_argument("--pages", default=None, help="1,2 или 1-4 или all")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--only", default=None,
        choices=[*VARIANTS, "all"],
        help="ограничить одним вариантом (по умолчанию — все)",
    )
    parser.add_argument(
        "--min-length", type=float, default=detect.DimParams().min_length,
        help="минимальная длина размерной линии, в пунктах эталонного листа",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="сверить суммы с эталонными длинами трасс",
    )
    parser.add_argument("--dump-json", action="store_true")
    args = parser.parse_args(argv)

    pdf = Path(args.pdf)
    if not pdf.is_file():
        print(f"PDF не найден: {pdf}", file=sys.stderr)
        return 1

    import pymupdf

    with pymupdf.open(pdf) as document:
        total = document.page_count
    pages = parse_pages(args.pages, total)

    params = detect.DimParams(min_length=args.min_length)
    results = detect.detect_pdf(str(pdf), pages, params)
    lines_by_page = {r.page_number: r.lines for r in results}

    print(f"{pdf.name}: листов {total}, обработано {len(results)}")
    print(f"толщина размерных линий: {results[0].width_used:.2f} pt "
          f"({'подобрана' if results[0].stats.get('width_guessed') else 'задана'}), "
          f"масштаб листов {results[0].stats.get('scale_source', '?')}")
    report(results)
    if args.check:
        check(results)

    wanted = [k for k in VARIANTS if args.only in (None, "all", k)]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for name in wanted:
        VARIANTS[name](str(pdf), out_root, lines_by_page, dpi=args.dpi)
        print(f"  -> {name}: {out_root}")

    if args.dump_json:
        payload = {
            "pdf": pdf.name,
            "min_length_pt": params.min_length,
            "reference_sheet": {
                "text_size_pt": params.ref_text_size,
                "page_side_pt": params.ref_page_side,
            },
            "totals": {
                "sum_all": sum(r.chain_total()["sum_all"] for r in results),
                "sum_without_duplicates": sum(
                    r.chain_total()["sum_without_duplicates"] for r in results
                ),
            },
            "pages": [
                {
                    "summary": r.summary(),
                    "chain": r.chain_total(),
                    "hierarchy": [
                        # Имена P1..Pn, чтобы id в выгрузке и надпись на
                        # картинке совпадали буква в букву.
                        {
                            "id": n.line.name,
                            "kind": n.line.kind,
                            "label": n.line.label,
                            "value": n.line.value,
                            "parent": None if n.parent is None else f"P{n.parent}",
                            "children": [f"P{i}" for i in n.children],
                        }
                        for n in r.hierarchy()
                    ],
                    "numbers": [
                        {"id": f"N{n.index}", "text": n.text,
                         "bbox": [round(v, 2) for v in n.box]}
                        for n in r.numbers
                    ],
                    "orphan_numbers": [
                        {"text": n.text, "bbox": [round(v, 2) for v in n.box]}
                        for n in r.orphan_numbers
                    ],
                    "segments": [l.as_dict() for l in r.lines],
                    "stats": r.stats,
                }
                for r in results
            ],
        }
        path = out_root / "segments.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> json: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
