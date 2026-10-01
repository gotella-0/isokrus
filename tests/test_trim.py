"""Проверки обрезки пустых полей (``isokrus.trim``).

Запуск из корня проекта, без pytest::

    python tests/test_trim.py

Проверяются случаи, которых нет в ``02_*.pdf``, но которые ломают наивную
реализацию «сравнить пиксели с белым и обрезать»:

1. полностью белый лист — обрезка не должна выродиться в несколько пикселей;
2. серый скан (фон 240) — фон берётся из угла, а не из константы 255;
3. серый скан с шумом ±6 — шум не должен отменять обрезку;
4. одиночная тёмная точка в поле — при ``min_ink=1`` обрезку отменяет
   (безопасный выбор: не режем ничего похожего на содержимое), при
   ``min_ink=8`` сохраняет;
5. бледная линия (250) — ожидаемо НЕ находится: это осознанный компромисс
   порога ``noise``, и тест фиксирует именно такое поведение;
6. содержимое меньше ``min_side`` — обрезка отменяется целиком;
7. на реальном листе подписи после сдвига попадают на нарисованные пиксели,
   а обрезанный лист не меньше половины исходного.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from isokrus import config
from isokrus.extract import extract_pages
from isokrus.trim import content_box, expand, trim

W, H = 1600, 1200
FAILURES: list[str] = []


def make_page(draw, background: int = 255, noise: int = 0) -> pymupdf.Pixmap:
    """Лист заданного фона с шумом, на который сверху что-то нарисовано."""
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    page.draw_rect(pymupdf.Rect(0, 0, W, H), color=None, fill=(background / 255,) * 3)
    if noise:
        # Лоскутное пятно: соседние участки бумаги отличаются на уровень-два,
        # как у реального скана. Важно не красота, а чтобы в каждой строке и
        # каждом столбце фона были пиксели темнее фона.
        step = 20
        for index, y in enumerate(range(0, H, step)):
            for x in range(0, W, step):
                shift = ((index * 7 + x // step * 13) % (2 * noise + 1)) - noise
                level = (background + shift) / 255
                page.draw_rect(pymupdf.Rect(x, y, x + step, y + step), color=None,
                               fill=(level,) * 3)
    if draw:
        draw(page)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1, 1), alpha=False)
    doc.close()
    return pix


def check(name: str, got, want, tol: int = 8) -> None:
    if all(abs(g - e) <= tol for g, e in zip(got, want)):
        print(f"  OK    {name}: {got}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}: получено {got}, ожидалось {want}")


# --- содержимое: рамка 400..1200 по X, 300..900 по Y ------------------------


def drawing(page: pymupdf.Page) -> None:
    page.draw_rect(pymupdf.Rect(400, 300, 1200, 900), color=(0, 0, 0), width=2)
    page.draw_line(pymupdf.Point(500, 500), pymupdf.Point(1100, 700), width=1)


def speck(page: pymupdf.Page) -> None:
    drawing(page)
    page.draw_rect(pymupdf.Rect(120, 600, 121, 601), color=(0, 0, 0), fill=(0, 0, 0))


def faint(page: pymupdf.Page) -> None:
    drawing(page)
    page.draw_line(pymupdf.Point(100, 100), pymupdf.Point(100, 1100), width=1,
                   color=(250 / 255,) * 3)


def box_of(pixmap, **kwargs) -> tuple[int, int, int, int]:
    box, _ = content_box(pixmap, **kwargs)
    return box


def _labels_on_ink(pixmap, items, background: int) -> int:
    """Сколько подписей содержат нарисованный пиксель.

    Подпись в текстовом слое и её начертание на картинке — одно и то же
    место. Если сдвиг рамок верен, то внутри каждой рамки есть пиксель цвета
    чернил. Это и есть проверка, что обрезка не разъехала текст с картинкой.
    """
    found = 0
    for item in items:
        # Рамку подписи подрезаем по картинке: за кадром всё равно нечего
        # искать, а pixel() за пределами растра бросает исключение.
        x0 = max(0, min(round(item.x0), pixmap.width - 1))
        y0 = max(0, min(round(item.y0), pixmap.height - 1))
        x1 = max(x0 + 1, min(round(item.x1), pixmap.width))
        y1 = max(y0 + 1, min(round(item.y1), pixmap.height))
        for y in range(y0, y1):
            for x in range(x0, x1):
                if min(pixmap.pixel(x, y)) < background - 8:
                    found += 1
                    break
            else:
                continue
            break
    return found


DEFAULTS = {
    "probe_side": config.TRIM_PROBE_SIDE,
    "min_run": config.TRIM_MIN_RUN,
    "noise": config.TRIM_NOISE,
    "min_ink": config.TRIM_MIN_INK,
}

print("1. белый лист без содержимого")
check("рамка = весь лист", box_of(make_page(None), **DEFAULTS), (0, 0, W, H), tol=0)

print("2. белый лист с чертежом")
check("рамка = чертёж", box_of(make_page(drawing), **DEFAULTS), (400, 300, 1200, 900))

print("3. серый скан (фон 240)")
check("рамка = чертёж", box_of(make_page(drawing, background=240), **DEFAULTS),
      (400, 300, 1200, 900))

print("4. серый скан с шумом ±6")
check("рамка = чертёж", box_of(make_page(drawing, background=240, noise=6), **DEFAULTS),
      (400, 300, 1200, 900))

print("5. одиночная тёмная точка в поле")
check("min_ink=1: обрезка отменена (ничего не режем зря)",
      box_of(make_page(speck), **DEFAULTS), (100, 300, 1200, 900), tol=24)
check("min_ink=8: обрезка сохранена",
      box_of(make_page(speck), **{**DEFAULTS, "min_ink": 8}), (400, 300, 1200, 900))

print("6. бледная линия 250 в поле (осознанно не ловится)")
check("рамка = чертёж, бледная линия проигнорирована",
      box_of(make_page(faint), **DEFAULTS), (400, 300, 1200, 900))

print("7. содержимое меньше min-side")
tiny = make_page(lambda p: p.draw_rect(pymupdf.Rect(700, 600, 740, 640),
                                       color=(0, 0, 0), width=2))
box, _ = content_box(tiny, **DEFAULTS)
check("min-side отменяет обрезку",
      expand(box, pad=12, width=W, height=H, min_side=96), (0, 0, W, H), tol=0)

print("8. trim() на пустом листе возвращает исходный растр")
result = trim(make_page(None))
check("размер не изменился", (result.pixmap.width, result.pixmap.height), (W, H), tol=0)
if result.skipped:
    print("  OK    пустой лист помечен как skipped")
else:
    FAILURES.append("пустой лист не skipped")
    print("  FAIL  пустой лист не помечен как skipped")

print("9. реальный лист: подписи после обрезки попадают на картинку")
pdfs = sorted(glob.glob("02_*.pdf"))
if not pdfs:
    print("  пропуск: 02_*.pdf не найден")
else:
    page = extract_pages(pdfs[0], pages=["1"], trim_margins=True).pages[0]
    info = page.meta["trim"]
    pixmap = pymupdf.Pixmap(page.data)
    check("лист обрезан", (int(page.crop_box is not None), int(page.area_ratio < 0.8)),
          (1, 1), tol=0)
    if page.crop_box:
        on_ink = _labels_on_ink(pixmap, page.text_items, info["background"])
        total = len(page.text_items)
        if on_ink == total and total:
            print(f"  OK    все подписи на месте: {on_ink}/{total}, "
                  f"размер {pixmap.width}x{pixmap.height}, "
                  f"площадь {info['area_ratio']:.0%}, "
                  f"подписей вне среза: {page.meta['text_items_dropped']}")
        else:
            FAILURES.append("подписи не на картинке")
            print(f"  FAIL  подписи на месте только {on_ink}/{total}")

print()
if FAILURES:
    print(f"ПРОВАЛЕНО: {', '.join(FAILURES)}")
    raise SystemExit(1)
print("Все проверки пройдены.")
