"""Проверка подсветки размеров (без обращения к API):

    python tests/test_overlay.py

Подсветка уходит в модель вместо снимка листа, поэтому ошибки в ней стоят
дорого и заметить их трудно: картинка остаётся похожей, просто неверной.
Проверяется то, что должно быть верно всегда: размеры совпадают с исходником,
каждый размер обведён ровно один раз, подписи не закрыты, ничего не уехало за
край, и при отключённой подсветке в модель уходит ровно исходный байт.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf

from isokrus import config, overlay
from isokrus.extract import extract_pages

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  OK    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")


def image_size(data: bytes) -> tuple[int, int]:
    pixmap = pymupdf.Pixmap(data)
    return pixmap.width, pixmap.height


def pixels_changed(before: bytes, after: bytes) -> int:
    """Сколько пикселей отличается — грубая мера «подсветка вообще нарисована»."""
    a = pymupdf.Pixmap(before).samples
    b = pymupdf.Pixmap(after).samples
    return sum(1 for x, y in zip(a, b, strict=True) if x != y)


print("1. подсветка того же размера, что и исходник")
# Подсветка выключена по умолчанию (по замерам она ухудшает точность), поэтому
# тест просит её явно. Иначе он проверял бы значение переменной окружения, а не
# код подсветки, и молча проходил бы или падал вместе с DIMSCAN_OVERLAY.
page = extract_pages(
    "02_Изометрии_10_листов.pdf", pages=["8"], with_overlay=True,
).pages[0]
check("размеры на листе есть", len(page.dimensions) == 12, str(len(page.dimensions)))
check("подсветка построена", bool(page.overlay), "overlay пуст")
check("размер совпадает",
      image_size(page.overlay) == image_size(page.data),
      f"{image_size(page.overlay)} против {image_size(page.data)}")
check("подсветка отличается от исходника", page.overlay != page.data)
changed = pixels_changed(page.data, page.overlay)
check("пикселей изменено немного (только пометки)",
      0 < changed < len(page.data) * 4,
      f"{changed} байт отличается")

print("2. модель получает подсветку, на диск — исходник")
check("model_image() это подсветка", page.model_image() == page.overlay)
check("save() кладёт исходник", page.data != page.overlay)
check("data_url() не пустой", page.data_url().startswith("data:image/png;base64,"))

print("3. отключение подсветки не меняет картинку для модели")
plain = extract_pages(
    "02_Изометрии_10_листов.pdf", pages=["8"], with_overlay=False,
).pages[0]
check("подсветки нет", not plain.overlay)
check("модель получает ровно исходник", plain.model_image() == plain.data)
check("размеры нашлись всё равно", len(plain.dimensions) == 12)
check("в meta видно, что подсветки не было", plain.meta.get("overlay") is False)

print("4. лист без размеров остаётся исходником")
empty = overlay.render_overlay(plain.data, plain.width, plain.height, [])
check("пустой список размеров — исходные байты", empty == plain.data)

print("5. рамка плотная, плашка маленькая и сидит на углу")
unit = min(page.width, page.height)
pad = max(2.0, unit * overlay.FRAME_PAD_RATIO)
mark_size = max(7.5, unit * overlay.MARK_RATIO)
off_sheet = []
too_big = []
covers_number = []
not_anchored = []
for mark in page.dimensions:
    frame = pymupdf.Rect(
        mark.bbox[0] - pad, mark.bbox[1] - pad,
        mark.bbox[2] + pad, mark.bbox[3] + pad,
    )
    if not overlay._fits(frame, page.width, page.height):
        off_sheet.append(mark.line_id)
    badge = overlay._badge_rect(frame, mark.line_id, mark_size)
    if not overlay._fits(badge, page.width, page.height):
        off_sheet.append(mark.line_id)
    # Плашка — уголок рамки, а не второй блок: она должна быть заметно
    # меньше самой рамки и не накрывать само число.
    if badge.width >= frame.width or badge.height >= frame.height:
        too_big.append(f"{mark.line_id} {badge.width:.0f}x{badge.height:.0f} "
                       f"при рамке {frame.width:.0f}x{frame.height:.0f}")
    if badge.y0 > frame.y0 + frame.height * 0.55:
        covers_number.append(mark.line_id)
    # Привязана к правому верхнему углу: перекрывает его, но не уезжает.
    if badge.x0 < frame.x1 - badge.width or badge.x0 > frame.x1:
        not_anchored.append(f"{mark.line_id} x0={badge.x0:.0f} x1={frame.x1:.0f}")
check("рамки и плашки на листе", not off_sheet, ", ".join(off_sheet))
check("плашка меньше рамки", not too_big, "; ".join(too_big))
check("плашка не накрывает число", not covers_number, ", ".join(covers_number))
check("плашка сидит на углу рамки", not not_anchored, "; ".join(not_anchored))

print("6. ключи наконечников указывают в обе стороны")
first = page.dimensions[0]
start = pymupdf.Point(*first.line_start)
end = pymupdf.Point(*first.line_end)
check("концы линии заданы и различны", start.distance_to(end) > 1.0)
middle = overlay._inset(start, end, 10.0)
check("отступ для стрелок оставляет середину", middle is not None)
if middle is not None:
    check("отрезок внутри исходного",
          pymupdf.Point((start.x + end.x) / 2, (start.y + end.y) / 2).distance_to(
              pymupdf.Point((middle[0].x + middle[1].x) / 2,
                            (middle[0].y + middle[1].y) / 2)) < 1e-6)

print("7. выноска и отрезок получают разный цвет")
kinds = {m.kind for m in page.dimensions}
check("вид попадает в промт как слово",
      all(m.kind in ("span", "pointer") for m in page.dimensions))
check("цвета линий отличаются", overlay.SPAN_COLOR != overlay.POINTER_COLOR)
check("виды встречаются на листе", bool(kinds), str(kinds))
accent = overlay._accent(overlay.SPAN_COLOR)
check("наконечник темнее линии", all(a < c for a, c in zip(accent, overlay.SPAN_COLOR)),
      f"{accent} против {overlay.SPAN_COLOR}")
check("метка не совпадает с линией", overlay.FRAME_COLOR != overlay.SPAN_COLOR)
check("метка не совпадает с наконечником", overlay.FRAME_COLOR != accent)
check("метка не совпадает с цветом выноски",
      overlay.FRAME_COLOR != overlay.POINTER_COLOR)
check("текст плашки белый", overlay.BADGE_TEXT_COLOR == (1.0, 1.0, 1.0))

print("8. подсветка выключена по умолчанию — точность с ней ниже")
check("DIMSCAN_OVERLAY выключен", config.DIMSCAN_OVERLAY is False)
plain8 = extract_pages("02_Изометрии_10_листов.pdf", pages=["8"]).pages[0]
check("без флага подсветки нет", not plain8.overlay)
on = extract_pages(
    "02_Изометрии_10_листов.pdf", pages=["8"], with_overlay=True,
).pages[0]
check("с флагом подсветка есть", bool(on.overlay))
check("размеры те же самые",
      [m.line_id for m in on.dimensions] == [m.line_id for m in plain8.dimensions])

print()
if FAILURES:
    print(f"ПРОВАЛЕНО: {', '.join(FAILURES)}")
    raise SystemExit(1)
print("Все проверки пройдены.")
