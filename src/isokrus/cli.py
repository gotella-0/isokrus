"""Консольный интерфейс.

    python -m isokrus run 02_Изометрии_10_листов.pdf --experiment exp_01
    python -m isokrus experiments
    python -m isokrus schema exp_02
    python -m isokrus extract 02_Изометрии_10_листов.pdf --pages 1-10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from isokrus import config
from isokrus.annotate import annotate_results, collect_regions
from isokrus.dimscan import format_dimensions_block
from isokrus.errors import IsokrusError
from isokrus.experiments import Experiment, list_experiments, load_experiment
from isokrus.extract import extract_pages
from isokrus.pipeline import analyze_pages, run, write_json_report
from isokrus.report import build_rows, build_stats, write_csv, write_xlsx
from isokrus.schema import model_to_strict_schema
from isokrus.textmap import TextIndex, format_text_block


def _use_utf8_console() -> None:
    """Консоль Windows по умолчанию cp1251 и роняет вывод на любом символе.

    Меняем кодировку на UTF-8 и отключаем замену непечатаемых символов:
    иначе блок извлечённого текста (а в нём есть «Ø», «∠», кириллица)
    нельзя ни посмотреть, ни перенаправить в файл.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # поток уже перенаправлен в файл
            pass


def _split_pages(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None
    parts: list[str] = []
    for value in values:
        parts.extend(p.strip() for p in value.split(",") if p.strip())
    return parts or None


def _progress(done: int, total: int, page) -> None:
    status = "ok" if done <= total else ""
    size = f"{page.width}x{page.height}"
    if page.crop_box:
        # Полей на листе не осталось — показываем, сколько срезали, иначе
        # непонятно, откуда взялись такие размеры.
        size = f"{size} (срезано {page.area_ratio:.0%})"
    print(
        f"  [{done}/{total}] лист {page.page_number} "
        f"({size}, {page.size_bytes // 1024} КБ) {status}",
        flush=True,
    )


def _share(part: float, whole: float) -> str:
    """Доля part от whole в процентах; пустая строка, если знаменатель нулевой."""
    return f"{part / whole:.0%}" if whole else ""


def _summarize(run_result, created: dict) -> None:
    stats = build_stats(run_result)
    print()
    print("Итоги")
    print(f"  листов:            {stats['pages_ok']} из {stats['pages_total']}")
    print(f"  время:             {stats['duration_seconds']} с "
          f"({stats['seconds_per_page']} с/лист)")
    print(f"  обращений к LLM:   {stats['calls_per_page']} на лист "
          f"(всего {run_result.usage.calls}, повторов {stats['retries']})")
    print(f"  токены:            {stats['prompt_tokens']} prompt / "
          f"{stats['completion_tokens']} completion "
          f"(из них reasoning {stats['reasoning_tokens']}, "
          f"видимых {stats['visible_completion_tokens']})")
    if stats["cost_total"]:
        cur = stats["currency"]
        print(f"  стоимость:         {stats['cost_total']:.4f} {cur} "
              f"({stats['cost_per_page']:.4f} за лист)")
        print(f"    из них reasoning {stats['cost_reasoning']:.4f} {cur} "
              f"({_share(stats['cost_reasoning'], stats['cost_total'])})")
    for error in stats.get("errors", [])[:5]:
        print(f"  ! лист {error['page']}: {error['error']}")

    print("Файлы")
    for name, path in created.items():
        if path is not None:
            print(f"  {name:<6} {path}")

    annotated = list((run_result.output_dir / "annotated").glob("*.png"))
    if annotated:
        print(f"  разметка: {len(annotated)} файл(ов) в {run_result.output_dir / 'annotated'}")


def _dims(args) -> list:
    """Извлечь страницы с разбором размеров и вернуть результат."""
    return extract_pages(
        args.pdf,
        dpi=args.dpi,
        pages=_split_pages(args.pages),
        trim_margins=not args.no_trim,
        with_dimensions=not args.no_dimscan,
        with_overlay=getattr(args, "overlay", False) or None,
        with_clean=_clean_flag(args),
    )


def _clean_flag(args):
    """Что выбрано флагами зачистки.

    ``--clean`` и ``--no-clean`` противоположны, и без разбора порядка при
    включённых обоих побеждал бы молчаливый порядок регистрации флагов.
    """
    if getattr(args, "no_clean", False):
        return False
    return getattr(args, "clean", False) or None


def _experiment(args) -> Experiment:
    """Загрузить эксперимент и применить переопределения модели/размышления.

    Подменяем поля в самом объекте, а не передаём их отдельно вниз: модель
    и глубину размышления читают несколько мест — шапка прогона, счётчики
    токенов, ``stats.json``. Отдельные параметры разъезжались бы с тем, что
    реально ушло в API, а отчёт должен говорить правду.
    """
    experiment = load_experiment(args.experiment)
    changes = {}
    if getattr(args, "model", None):
        changes["model"] = args.model
    if getattr(args, "reasoning", None):
        changes["reasoning_effort"] = args.reasoning
    if changes:
        experiment = replace(experiment, **changes)
    return experiment


def cmd_run(args: argparse.Namespace) -> int:
    experiment = _experiment(args)
    trim_margins = not args.no_trim
    want_dims = not args.no_dimscan
    print(f"Эксперимент: {experiment.name} ({experiment.path})")
    print(f"  модель:    {experiment.model or config.LLM_MODEL}")
    print(f"  reasoning: {experiment.reasoning_effort or config.LLM_REASONING_EFFORT}")
    print(f"  параллельность: {args.parallel}")
    print(f"  обрезка полей: {'да' if trim_margins else 'нет'}")
    print(f"  разбор размеров: {'да' if want_dims else 'нет'}")

    print(f"Извлекаю страницы из {args.pdf} ...")
    print(f"Обрабатываю (до {args.parallel} параллельных вызовов):")

    run_result = run(
        pdf_path=args.pdf,
        experiment=experiment,
        output_dir=args.output,
        pages=_split_pages(args.pages),
        dpi=args.dpi,
        max_parallel=args.parallel,
        save_page_images=not args.no_images,
        progress=None if args.quiet else _progress,
        trim_margins=trim_margins,
        with_dimensions=want_dims,
        with_clean=_clean_flag(args),
    )

    created: dict = {}
    rows = build_rows(run_result.pages)
    out = run_result.output_dir
    created["csv"] = write_csv(rows, out / "result.csv")
    created["xlsx"] = write_xlsx(rows, out / "result.xlsx")
    created["json"] = write_json_report(run_result)
    stats = build_stats(run_result)
    created["stats"] = out / "stats.json"
    created["stats"].write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not args.no_annotate:
        annotate_results(run_result, out)

    if not args.quiet:
        _summarize(run_result, created)
    else:
        print(created["json"])

    return 0 if not run_result.failed else 1


def cmd_extract(args: argparse.Namespace) -> int:
    """Только извлечение: полезно, чтобы проверить рендер до трат на API."""
    result = _dims(args)
    out_dir = Path(args.output) if args.output else config.OUTPUT_DIR / "pages"
    for page in result.pages:
        saved = page.save(out_dir)
        crop = f", срезано {page.area_ratio:.0%}" if page.crop_box else ""
        print(f"{saved} {page.width}x{page.height} {page.size_bytes // 1024} КБ"
              f"{crop} (подписей в текстовом слое: {len(page.text_items)}, "
              f"размеров: {len(page.dimensions)})")
    print(f"Страниц в PDF: {result.page_count}, извлечено: {len(result)}")
    return 0


def cmd_dims(args: argparse.Namespace) -> int:
    """Показать блок разбора размеров — ровно то, что уходит в системный промт.

    Нужен, чтобы не гадать, что модель видит: если числа нет в блоке, модель
    его и не выпишет, а если число помечено как «не размер», она не должна
    тянуть его в расчёт. Всё считается из вектора, к API не ходит.
    """
    result = _dims(args)
    for page in result.pages:
        if args.which and page.page_number not in args.which:
            continue
        info = page.meta.get("dimscan") or {}
        error = page.meta.get("dimscan_error")
        if error:
            print(f"# лист {page.page_number} — разбор размеров не удался: {error}")
            continue
        print(f"# лист {page.page_number} — {page.width}x{page.height} px, "
              f"размеров {len(page.dimensions)}, "
              f"масштаб {info.get('scale')} ({info.get('scale_source')})")
        print(format_dimensions_block(
            page.dimensions, page.rejected_numbers,
            with_rejected=not args.no_rejected,
        ))
        print()
    return 0


def cmd_text(args: argparse.Namespace) -> int:
    """Показать блок извлечённого текста — ровно то, что уходит в системный промт.

    Нужен, чтобы не гадать, видит ли модель нужные подписи: если числа нет в
    блоке, модель их и не выпишет, и разметка ничего не обведёт.
    """
    result = _dims(args)
    for page in result.pages:
        if args.which and page.page_number not in args.which:
            continue
        text = format_text_block(page.text_items, page.width, page.height)
        if args.numbers_only:
            index = TextIndex(page.text_items)
            print(f"# лист {page.page_number}: {len(index)} подписей, "
                  f"{len(index.numbers())} числовых")
            for item in index.numbers():
                print(item.as_line())
            continue
        print(f"# лист {page.page_number} — {page.width}x{page.height} px, "
              f"{len(page.text_items)} подписей")
        print(text)
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    names = list_experiments()
    if not names:
        print(f"Эксперименты не найдены в {config.RESEARCH_DIR}")
        return 1
    print(f"Папка экспериментов: {config.RESEARCH_DIR}")
    for name in names:
        try:
            experiment = load_experiment(name)
        except IsokrusError as exc:
            print(f"  {name}: ОШИБКА — {exc}")
            continue
        marker = " (по умолчанию)" if name == config.DEFAULT_EXPERIMENT else ""
        print(f"\n  {name}{marker}")
        print(f"    модель:    {experiment.model or config.LLM_MODEL}")
        print(f"    reasoning: {experiment.reasoning_effort or config.LLM_REASONING_EFFORT}")
        print(f"    файлы:     {', '.join(experiment.files)}")
        print(f"    поля:      {', '.join(experiment.response_model.model_fields)}")
        if experiment.notes:
            print(f"    заметки:   {experiment.notes}")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.experiment)
    schema = model_to_strict_schema(experiment.response_model)
    text = json.dumps(schema, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Схема {experiment.name} сохранена в {args.output}")
    else:
        print(text)
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.experiment)
    if args.part == "system":
        print(experiment.system)
    else:
        print(experiment.user or "(user.md отсутствует — используется текст по умолчанию)")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Анализ без сохранения: показать сырой ответ модели для отладки промта."""
    experiment = _experiment(args)
    result = _dims(args)
    page_numbers = [int(p) for p in (args.which or [])] or [p.page_number for p in result.pages]
    selected = [p for p in result.pages if p.page_number in page_numbers]

    print(f"Эксперимент {experiment.name}, листов: "
          f"{', '.join(str(p.page_number) for p in selected)}")
    results, usage = analyze_pages(selected, experiment, max_parallel=args.parallel)
    for item in results:
        print(f"\n--- лист {item.page.page_number} ({item.elapsed:.1f} с) ---")
        if item.error:
            print(f"ОШИБКА: {item.error}")
            continue
        print(json.dumps(item.response, ensure_ascii=False, indent=2))
        if args.regions:
            for region in collect_regions(item.response, item.page):
                print(f"  [{region.kind}] {region.label} "
                      f"({region.x0:.0f},{region.y0:.0f})-({region.x1:.0f},{region.y1:.0f})")
    print(f"\nОбращений: {usage.calls}, токены: {usage.prompt_tokens}/"
          f"{usage.completion_tokens}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isokrus",
        description="Расчёт длины трубопроводов по изометрическим чертежам (PDF -> LLM -> CSV).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--experiment", "-e", default=config.DEFAULT_EXPERIMENT,
            help=f"эксперимент из research/ (по умолчанию {config.DEFAULT_EXPERIMENT})",
        )
        sub.add_argument(
            "--parallel", "-p", type=int, default=config.MAX_PARALLEL,
            help=f"максимум параллельных вызовов LLM (по умолчанию {config.MAX_PARALLEL})",
        )
        # Модель и глубину размышления можно сменить, не трогая папку
        # эксперимента: meta.json — это зафиксированная версия опыта, а
        # сравнение «эта модель против той на том же промте» требует менять
        # только модель. Переопределение подставляется в тот же объект
        # эксперимента, поэтому в stats.json и в шапке прогона пишется ровно
        # та модель, которая реально работала.
        sub.add_argument(
            "--model", default=None,
            help="модель вместо указанной в meta.json (или LLM_MODEL)",
        )
        sub.add_argument(
            "--reasoning", default=None,
            help="глубина размышления: minimal, low, medium, high "
                 "(перекрывает meta.json и LLM_REASONING_EFFORT)",
        )

    def add_pages(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("pdf", help="путь к PDF с изометриями")
        sub.add_argument(
            "--pages", nargs="+", default=None,
            help="страницы: 1-10, 1 3 7 или 1,3,7 (по умолчанию все)",
        )
        sub.add_argument(
            "--dpi", type=int, default=config.RENDER_DPI,
            help=f"плотность рендера (по умолчанию {config.RENDER_DPI})",
        )
        sub.add_argument(
            "--no-trim", action="store_true",
            help="не обрезать пустые поля листа (по умолчанию обрезаются)",
        )
        sub.add_argument(
            "--no-dimscan", action="store_true",
            help="не искать размеры в векторе (по умолчанию ищутся)",
        )
        sub.add_argument(
            "--overlay", action="store_true",
            help="пометить на листе найденные размеры и отправить в модель "
                 "подсвеченный снимок (по умолчанию выключено: точность "
                 "с подсветкой ниже, см. README)",
        )
        sub.add_argument(
            "--clean", action="store_true",
            help="вырезать из PDF всё, что размером не является: координаты, "
                 "номера узлов, штампы, словесный шум (по умолчанию выключено)",
        )
        sub.add_argument(
            "--no-clean", action="store_true",
            help="то же, что без --clean: слать в модель исходный лист",
        )

    run_parser = subparsers.add_parser("run", help="полный цикл обработки PDF")
    add_pages(run_parser)
    add_common(run_parser)
    run_parser.add_argument("--output", "-o", default=None, help="каталог для результатов")
    run_parser.add_argument("--no-images", action="store_true", help="не сохранять исходные PNG")
    run_parser.add_argument("--no-annotate", action="store_true", help="не строить разметку")
    run_parser.add_argument("--quiet", "-q", action="store_true", help="только итоговый путь")
    run_parser.set_defaults(func=cmd_run)

    extract_parser = subparsers.add_parser("extract", help="только извлечь страницы в PNG")
    add_pages(extract_parser)
    extract_parser.add_argument("--output", "-o", default=None, help="каталог для PNG")
    extract_parser.set_defaults(func=cmd_extract)

    text_parser = subparsers.add_parser(
        "text", help="извлечённый из PDF текст с координатами (то, что уходит в промт)"
    )
    add_pages(text_parser)
    text_parser.add_argument("--which", nargs="+", type=int, default=None,
                              help="номера листов (по умолчанию все выбранные)")
    text_parser.add_argument("--numbers-only", action="store_true",
                             help="только числовые подписи — вероятные размеры")
    text_parser.set_defaults(func=cmd_text)

    dims_parser = subparsers.add_parser(
        "dims", help="разобранные размеры листа (то, что уходит в промт)"
    )
    add_pages(dims_parser)
    dims_parser.add_argument("--which", nargs="+", type=int, default=None,
                             help="номера листов (по умолчанию все выбранные)")
    dims_parser.add_argument(
        "--no-rejected", action="store_true",
        help="не печатать числа, отсеянные как «не размер»",
    )
    dims_parser.set_defaults(func=cmd_dims)

    analyze_parser = subparsers.add_parser("analyze", help="вызвать LLM и показать сырой JSON")
    add_pages(analyze_parser)
    add_common(analyze_parser)
    analyze_parser.add_argument("--which", nargs="+", default=None, help="номера листов")
    analyze_parser.add_argument("--regions", action="store_true", help="показать найденные bbox")
    analyze_parser.set_defaults(func=cmd_analyze)

    experiments_parser = subparsers.add_parser("experiments", help="список экспериментов")
    experiments_parser.set_defaults(func=cmd_experiments)

    schema_parser = subparsers.add_parser("schema", help="JSON-схема ответа эксперимента")
    schema_parser.add_argument("experiment", nargs="?", default=config.DEFAULT_EXPERIMENT)
    schema_parser.add_argument("--output", "-o", default=None, help="сохранить в файл")
    schema_parser.set_defaults(func=cmd_schema)

    prompt_parser = subparsers.add_parser("prompt", help="показать промт эксперимента")
    prompt_parser.add_argument("experiment", nargs="?", default=config.DEFAULT_EXPERIMENT)
    prompt_parser.add_argument(
        "--part", choices=("system", "user"), default="system", help="какой промт показать"
    )
    prompt_parser.set_defaults(func=cmd_prompt)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _use_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except IsokrusError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nПрервано пользователем", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
