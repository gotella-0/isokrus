"""Проверка зачистки листа (без обращения к API):

    python tests/test_clean.py

Зачистка уходит в модель вместо снимка листа, поэтому ошибка в ней стоит
дорого и заметить её трудно: картинка остаётся похожей, просто неверной —
или, наоборот, из листа выпадает размер, которого модель не увидит.

Проверяется то, что должно быть верно всегда, на всех десяти листах:

* ни одна подпись найденного размера не попадает под закрашивание;
* каждая область, задевшая размер, **не закрашивается вовсе**;
* знаки в квадратах и опорные знаки на листе есть и стираются;
* модель получает зачищенный лист, на диск и в meta — исходный;
* при отключённой зачистке в модель уходит ровно исходный байт.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf

from isokrus import clean, dimscan
from isokrus.extract import extract_pages

PDF = "02_Изометрии_10_листов.pdf"
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  OK    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")


def analyse(sheet: str):
    """Зачистка листа тем же путём, каким идёт конвейер, но без его обвязки."""
    import isokrus.extract as extract

    result = extract.extract_pages(PDF, pages=[sheet], with_clean=True)
    return result.pages[0]


print("1. зачистка на всех десяти листах не трогает размеры")


def ink_count(pixmap, box, threshold: int = 128) -> int:
    """Сколько тёмных пикселей в рамке — мера «подпись на месте»."""
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1 = min(pixmap.width - 1, int(box[2]))
    y1 = min(pixmap.height - 1, int(box[3]))
    samples = pixmap.samples
    total = 0
    for y in range(y0, y1 + 1):
        base = y * pixmap.stride
        for x in range(x0, x1 + 1):
            if samples[base + x * pixmap.n] < threshold:
                total += 1
    return total


total_erased = 0
touched: list[str] = []
erased_by_sheet: dict[str, int] = {}
for sheet in range(1, 11):
    page = analyse(str(sheet))
    info = page.meta.get("clean") or {}
    total_erased += info.get("erased", 0)
    erased_by_sheet[str(sheet)] = info.get("erased", 0)
    check(f"лист {sheet}: зачистка отработала", info.get("erased", 0) > 0,
          f"стёрто {info.get('erased', 0)}")
    check(f"лист {sheet}: ни один размер не задет",
          info.get("kept") == 0,
          f"затронуто областей: {info.get('kept')} — {info.get('by_kind')}")
    check(f"лист {sheet}: нет ошибки зачистки", not page.meta.get("clean_error"),
          str(page.meta.get("clean_error")))
    # Подпись размера не должна лежать в закрашенной области. Проверяется по
    # самим байтам и по количеству чернил, а не по одной точке: рамка подписи
    # — это em-бокс текста, её середина попадает в просвет между цифрами и
    # там и без всякой зачистки белым цветом.
    raw = pymupdf.Pixmap(page.data)
    cleaned = pymupdf.Pixmap(page.model_image())
    for mark in page.dimensions:
        before = ink_count(raw, mark.bbox)
        after = ink_count(cleaned, mark.bbox)
        if after < before:
            touched.append(
                f"лист {sheet} {mark.line_id}={mark.label} "
                f"({before}->{after})"
            )
check("ни одна подпись размера не потеряла чернила", not touched,
      "; ".join(touched[:6]))
check("зачистка что-то делает", total_erased > 100, f"стёрто {total_erased}")

print("\n2. закрашивание не выходит за размеры листа и не меняет его размера")
page = analyse("8")
raw = pymupdf.Pixmap(page.data)
out = pymupdf.Pixmap(page.model_image())
check("размер картинки тот же",
      (raw.width, raw.height) == (out.width, out.height),
      f"{out.width}x{out.height} против {raw.width}x{raw.height}")
check("зачищенный лист меньше исходного",
      len(page.model_image()) < len(page.data),
      f"{len(page.model_image())} против {len(page.data)}")

print("\n3. модель получает зачистку, на диск — исходник")
check("model_image() это зачистка", page.model_image() == page.cleaned)
check("исходник остался нетронутым", page.data != page.cleaned)
check("save() кладёт исходник, не зачистку", page.data != page.model_image())
check("data_url() не пустой", page.data_url().startswith("data:image/png;base64,"))

print("\n4. отключение зачистки не меняет картинку для модели")
plain = extract_pages(PDF, pages=["8"], with_clean=False).pages[0]
check("зачистки нет", not plain.cleaned)
check("модель получает ровно исходник", plain.model_image() == plain.data)
check("размеры нашлись всё равно", len(plain.dimensions) == 12)
check("в meta видно, что зачистки не было", not (plain.meta.get("clean") or {}))

print("\n5. классы элементов на листе 8 различимы и учтены")
info = page.meta.get("clean") or {}
kinds = info.get("by_kind", {})
check("на листе есть знаки в квадратах", kinds.get("signs", 0) > 0, str(kinds))
check("на листе есть опорные знаки", kinds.get("notes", 0) > 0, str(kinds))
check("в meta записаны применённые фильтры",
      tuple(info.get("filters", ())) == clean.FILTERS,
      str(info.get("filters")))

print("\n6. фильтры включаются по отдельности")
only_signs = extract_pages(
    PDF, pages=["8"], with_clean=True, clean_filters=("signs",),
).pages[0]
only_kinds = (only_signs.meta.get("clean") or {}).get("by_kind", {})
check("при filters=(signs,) стёрты только знаки",
      set(only_kinds) <= {"signs"}, str(only_kinds))
check("signs без rejected — узкая зачистка",
      (only_signs.meta.get("clean") or {}).get("erased", 0)
      < (page.meta.get("clean") or {}).get("erased", 0))
nothing = extract_pages(
    PDF, pages=["8"], with_clean=True, clean_filters=(),
).pages[0]
check("пустой список фильтров — лист не трогаем",
      nothing.model_image() == nothing.data)

print("\n7. геометрия зачистки сходится с подписями")
document = pymupdf.open(PDF)
pdf_page = document.load_page(7)
params = dimscan.DimParams()
scan = dimscan.detect_page(pdf_page, params)
check("разбор отдал контуры зачистке", bool(scan.contours),
      f"контуров {len(scan.contours)}")
check("разбор отдал пороги зачистке", scan.limits is not None)
box = page.pixel_box((0.0, 0.0, 72.0, 72.0))
check("перевод пунктов в пиксели похож на масштаб рендера",
      abs(box[2] - box[0] - page.pt_to_px(72.0)) < 1.0,
      f"{box[2] - box[0]:.1f} против {page.pt_to_px(72.0):.1f}")
document.close()

print()
if FAILURES:
    print(f"ПРОВАЛЕНО {len(FAILURES)}: {', '.join(FAILURES[:12])}")
    sys.exit(1)
print("Все проверки пройдены.")