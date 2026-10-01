"""Проверки поиска размеров (``isokrus.dimscan``).

Запуск из корня проекта, без pytest::

    python tests/test_dimscan_detect.py

Здесь ловится ровно то, что ломалось молча. Обе ошибки изначально выглядели
как «детектор работает», и обе попадали в сумму:

* **число в квадрате** — позиционный знак «9» на листе 2 обведён рамкой и
  приклеился к размеру. Числа в рамках, кругах и соседях по строке
  (``X 267000``, ``50 mm PE``, ``LT 4116``) размером не являются;
* **выноска без второго наконечника** — размер «13» на листе 3 начертан
  одной стрелкой, указывающей на зазор между фланцами. Раньше он молча
  пропадал, потому что искали только отрезки с наконечниками на обоих концах.

Плюс сквозная сверка с эталонными длинами трасс: если эталон не набирается
найденными числами, поиск неполон — где-то пропущен размер или за размер
принято чужое число.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isokrus import dimscan as detect
from tests.dimscan import reference

FAILURES: list[str] = []

# Числа, которые на листах нарисованы, но размером не являются.
# Позиционные знаки и номера опор — в квадратах, координаты привязки, марки
# труб и номера приборов — в строке с другими словами или в кружке.
NOT_DIMENSIONS: dict[int, set[str]] = {
    2: {"9", "3", "2", "1"},                  # позиционные знаки в квадратах
    3: {"11", "10", "12", "7", "5", "4", "1", "3"},
    4: {"7", "8", "4", "3", "2", "1"},
    10: {"9", "8", "7", "5", "4", "3", "2"},
}
COORDINATE_LIKE = {
    "267000", "119850", "11136", "266350", "121000", "14434", "121457",
    "14750", "120800", "15014", "268700", "117730", "117000", "15024",
    "258500", "110670", "15400", "16750", "98630", "135000", "7920",
    "77650", "10695", "101175", "9695", "102884", "132500", "9100",
    "102689", "132990", "8925", "104000", "131389", "130650", "130326",
    "131384", "8843", "8373", "48600", "41640", "35946", "48300", "43800",
    "37008", "48100", "37329", "29549", "83585", "55400", "85700", "29716",
    "54550", "29750", "89205", "65549", "63216", "63250", "57350", "85600",
    "22341", "22863", "86380", "86474", "54400", "78500",
    "50", "40", "4116",  # марки труб и номер прибора в кружке
}

# Размер, начертанный **только** выноской: отрезка с двумя наконечниками
# рядом нет, есть одна стрелка, указывающая на зазор между фланцами. Его
# отсутствие было ошибкой, а не особенностью листа.
POINTERS: dict[int, set[str]] = {3: {"13"}}

# Размер, у которого отрезок есть, а подпись отведена в сторону выноской.
# На листе 2 его подпись «224» терялась из-за того, что рядом стоял
# позиционный знак «9» в квадрате и перехватывал отрезок.
DISPLACED: dict[int, set[str]] = {2: {"224"}}


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  OK    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")


def _stamp_masks(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Прямоугольники, закрытые белой заливкой: поля и штамп.

    Именно ими на листе отрезано поле чертежа от штампа и от пустых полей,
    поэтому проверка «подпись не в штампе» получается без знания координат:
    рамки берутся из самого PDF.
    """
    out: list[pymupdf.Rect] = []
    for path in page.get_drawings():
        fill = path.get("fill")
        if not fill or max(fill) < 0.99:
            continue
        rect = path["rect"]
        if rect.get_area() > 0.9 * page.rect.get_area():
            continue  # весь лист белый — не заливка поля
        out.append(rect)
    return out


def _hits(rect: pymupdf.Rect, box: tuple[float, float, float, float]) -> bool:
    return not (
        rect.x1 <= box[0] or rect.x0 >= box[2]
        or rect.y1 <= box[1] or rect.y0 >= box[3]
    )


pdfs = sorted(glob.glob("02_*.pdf"))
if not pdfs:
    print("  пропуск: 02_*.pdf не найден")
    raise SystemExit(1)

print("1. числа в рамках, кругах и соседях по строке — не размеры")
doc = pymupdf.open(pdfs[0])
results = [
    detect.detect_page(doc[i], detect.DimParams()) for i in range(doc.page_count)
]
by_page = {r.page_number: r for r in results}

for page, unwanted in sorted(NOT_DIMENSIONS.items()):
    labels = {l.label for l in by_page[page].lines}
    leaked = sorted(unwanted & labels)
    check(
        f"лист {page:2d}: в размерах нет {sorted(unwanted)}",
        not leaked,
        f"просочились {leaked}",
    )

for page, result in sorted(by_page.items()):
    leaked = sorted(
        {l.label for l in result.lines} & COORDINATE_LIKE
    )
    check(
        f"лист {page:2d}: координаты и марки не попали в размеры",
        not leaked,
        f"просочились {leaked}",
    )

print("1b. ни один размер не лежит в штампе и за рамкой чертежа")
for page, result in sorted(by_page.items()):
    masks = _stamp_masks(doc[page - 1])
    leaked = [
        l.name for l in result.lines
        if l.label_bbox and any(_hits(r, l.label_bbox) for r in masks)
    ]
    check(
        f"лист {page:2d}: подписи размеров вне штампа",
        not leaked,
        f"в штампе: {leaked}",
    )

print("2. размеры, начертанные выноской, находятся")
for page, wanted in sorted(POINTERS.items()):
    lines = {l.label: l for l in by_page[page].lines}
    missing = sorted(set(wanted) - set(lines))
    check(
        f"лист {page:2d}: найдены размеры-выноски {sorted(wanted)}",
        not missing,
        f"не найдены {missing}",
    )
    for label in sorted(wanted & set(lines)):
        line = lines[label]
        check(
            f"  лист {page}: «{label}» помечен как указатель, а не как отрезок",
            line.kind == "pointer" and not line.measured,
            f"kind={line.kind}, measured={line.measured}",
        )

for page, wanted in sorted(DISPLACED.items()):
    lines = {l.label: l for l in by_page[page].lines}
    missing = sorted(set(wanted) - set(lines))
    check(
        f"лист {page:2d}: найдены размеры с отведённой подписью {sorted(wanted)}",
        not missing,
        f"не найдены {missing}",
    )
    for label in sorted(wanted & set(lines)):
        line = lines[label]
        check(
            f"  лист {page}: «{label}» — отрезок с выноской к подписи",
            line.measured and line.leader_end is not None,
            f"measured={line.measured}, leader_end={line.leader_end}",
        )

print("3. ни одного числа без размера и ни одного размера без числа")
for page, result in sorted(by_page.items()):
    check(
        f"лист {page:2d}: все числа привязаны к размерам",
        not result.orphan_numbers,
        ", ".join(n.text for n in result.orphan_numbers),
    )
    check(
        f"лист {page:2d}: у всех размеров есть подпись",
        all(l.value is not None for l in result.lines),
        ", ".join(l.name for l in result.lines if l.value is None),
    )

print("4. одно число не подписывает два размера")
for page, result in sorted(by_page.items()):
    labels = [l.label for l in result.lines]
    boxes = [tuple(l.label_bbox or ()) for l in result.lines]
    repeated = sorted({b for b in boxes if boxes.count(b) > 1})
    check(
        f"лист {page:2d}: каждое число досталось ровно одному размеру",
        not repeated,
        f"использованы дважды: {repeated}",
    )
    check(
        f"лист {page:2d}: длина списка равна числу подписей ({len(labels)})",
        len(labels) == len(result.lines),
    )

print("5. сумма без повторов не превышает наивную")
for page, result in sorted(by_page.items()):
    chain = result.chain_total()
    check(
        f"лист {page:2d}: без повторов {chain['sum_without_duplicates']:.0f} "
        f"<= как есть {chain['sum_all']:.0f}",
        chain["sum_without_duplicates"] <= chain["sum_all"] + 0.5,
    )

print("6. эталонная длина трассы набирается найденными числами")
for page, result in sorted(by_page.items()):
    target = reference.EXPECTED.get(result.page_number)
    if target is None:
        continue
    values = [l.value for l in result.lines if l.value is not None]
    subset = reference.reachable(values, target)
    check(
        f"лист {page:2d}: эталон {target:.0f} набирается из {len(values)} чисел",
        subset is not None,
        f"найдено {sorted(round(v) for v in values)}",
    )

print("7. отсев «не размер» виден в статистике")
for page, result in sorted(by_page.items()):
    rejected = result.stats["numbers_rejected"]
    check(
        f"лист {page:2d}: числовых слов {rejected['numeric_words']}, "
        f"размеров {result.summary()['segments']}",
        rejected["numeric_words"] > result.summary()["segments"],
        f"{rejected}",
    )

print("8. пороги следуют за масштабом листа")
# Тот же чертёж, вписанный в лист другого размера. Раньше пороги были
# заданы в пунктах PDF «как на эталонном листе», и на таком листе они
# разъезжались: на x2 площадь наконечника уходила за порог, стрелки не
# находились, и лист давал ноль размеров; на x0.5 «9» в квадрате снова
# проскакивало в размеры. Теперь масштаб измеряется по кеглю подписей.
base = pymupdf.open(pdfs[0])
baseline = [
    sorted((l.label for l in by_page[i + 1].lines if l.value is not None), key=float)
    for i in range(base.page_count)
]
for factor in (3.0, 2.0, 0.5, 0.25):
    src = pymupdf.open(pdfs[0])
    scaled = pymupdf.open()
    for i in range(src.page_count):
        source = src[i]
        target = scaled.new_page(
            width=source.rect.width * factor,
            height=source.rect.height * factor,
        )
        target.show_pdf_page(target.rect, src, i)
    bad: list[str] = []
    for i in range(scaled.page_count):
        result = detect.detect_page(scaled[i], detect.DimParams())
        got = sorted(
            (l.label for l in result.lines if l.value is not None), key=float
        )
        if got != baseline[i]:
            bad.append(f"{i + 1}: ждали {baseline[i]}, получили {got}")
    scaled.close()
    src.close()
    check(
        f"весь комплект в x{factor:g}: набор размеров тот же",
        not bad,
        "; ".join(bad[:2]),
    )
base.close()

scales = {
    round(r.stats["scale"], 2) for r in results
}
check(
    f"на исходном комплекте масштаб около 1.0 (получено {sorted(scales)})",
    all(0.95 <= s <= 1.05 for s in scales),
)

print()
if FAILURES:
    print(f"ПРОВАЛЕНО: {', '.join(FAILURES)}")
    raise SystemExit(1)
print("Все проверки пройдены.")
