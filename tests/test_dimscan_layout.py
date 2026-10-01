"""Проверки раскладки имён отрезков (``tests.dimscan.layout``).

Запуск из корня проекта, без pytest::

    python tests/test_dimscan_layout.py

Проверяется то, что ломается молча: если правила размещения откатятся к
«поставить по нормали от середины», на двух-трёх листах это будет выглядеть
нормально, и заметить можно только глазами на остальных. Тест ловит это
сразу и говорит, на каком листе и каким именем.

1. ни одно имя не пересекает текст листа, подписи размеров и другие имена;
2. ни одно имя не выходит за пределы листа;
3. ни одно имя не встаёт на отрезок или на его выноску;
4. на реальном комплекте не остаётся вынужденных размещений — то есть
   свободного места хватает всем;
5. раскладка не зависит от того, в каком порядке подали отрезки, — иначе
   P3 и P7 менялись бы местами между запусками.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isokrus import dimscan as detect
from tests.dimscan import layout as lay

FONT_SIZE = 10.9  # как в render: _line_width_for(страница A1, 2.2)
FAILURES: list[str] = []


def width_of(text: str, size: float) -> float:
    return pymupdf.get_text_length(text, fontname="hebo", fontsize=size)


def overlap(a: pymupdf.Rect, b: pymupdf.Rect, pad: float = 0.0) -> bool:
    return not (
        a.x1 <= b.x0 + pad
        or a.x0 >= b.x1 - pad
        or a.y1 <= b.y0 + pad
        or a.y0 >= b.y1 - pad
    )


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  OK    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")


def text_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    out = []
    for x0, y0, x1, y1, *_ in page.get_text("words"):
        if (y1 - y0) >= 2.0:
            out.append(pymupdf.Rect(x0, y0, x1, y1))
    return out


print("1. раскладка на реальном комплекте")
pdfs = sorted(glob.glob("02_*.pdf"))
if not pdfs:
    print("  пропуск: 02_*.pdf не найден")
    raise SystemExit(1)

doc = pymupdf.open(pdfs[0])
total_labels = 0
forced_all: list[str] = []
for number in range(doc.page_count):
    page = doc[number]
    result = detect.detect_page(page, detect.DimParams())
    layout = lay.layout_for_page(page, result.lines, FONT_SIZE, width_of)
    rects = text_rects(page)

    problems: list[str] = []
    for placement in layout.placements:
        # 1) не наезжает на текст листа и на другие имена
        for other in rects:
            if overlap(placement.rect, other, pad=lay.CLEARANCE):
                problems.append(f"{placement.name} на текст {tuple(round(v) for v in other)}")
                break
        for other in layout.placements:
            if other is placement:
                continue
            if overlap(placement.rect, other.rect, pad=lay.CLEARANCE):
                problems.append(f"{placement.name} на имя {other.name}")
                break
        # 2) внутри листа
        r = placement.rect
        if not (r.x0 >= page.rect.x0 and r.x1 <= page.rect.x1
                and r.y0 >= page.rect.y0 and r.y1 <= page.rect.y1):
            problems.append(f"{placement.name} за пределами листа")
        # 3) не на отрезке и не на выноске
        cx = (r.x0 + r.x1) / 2.0
        cy = (r.y0 + r.y1) / 2.0
        radius = max(r.width, r.height) * 0.35 + 1.0
        for line in result.lines:
            if lay.dist_point_seg((cx, cy), line.a, line.b) <= radius:
                problems.append(f"{placement.name} на отрезок {line.name}")
                break
            if line.leader_end is not None and lay.dist_point_seg(
                (cx, cy), line.midpoint, line.leader_end
            ) <= radius:
                problems.append(f"{placement.name} на выноску {line.name}")
                break

    forced = [p.name for p in layout.placements if p.forced]
    forced_all.extend(f"{number + 1}:{n}" for n in forced)
    total_labels += len(layout.placements)

    check(
        f"лист {number + 1:2d}: {len(layout.placements):2d} имён без наложений",
        not problems,
        "; ".join(problems[:4]),
    )

print(f"2. всего имён {total_labels}, вынужденных размещений {len(forced_all)}")
check("на всём комплекте нашлось чистое место каждому имени", not forced_all,
      ", ".join(forced_all[:8]))

print("3. раскладка не зависит от порядка подачи отрезков")
page = doc[7]
result = detect.detect_page(page, detect.DimParams())
straight = lay.layout_for_page(page, result.lines, FONT_SIZE, width_of)
shuffled = list(reversed(result.lines))
mixed = lay.layout_for_page(page, shuffled, FONT_SIZE, width_of)
same = all(
    abs(a.rect.x0 - b.rect.x0) < 1e-6 and abs(a.rect.y0 - b.rect.y0) < 1e-6
    for a, b in zip(sorted(straight.placements, key=lambda p: p.line_index),
                    sorted(mixed.placements, key=lambda p: p.line_index))
)
check("порядок подачи не меняет результат", same)

print("4. вырожденный случай: отрезок в углу листа")
tiny = pymupdf.open()
small = tiny.new_page(width=90, height=70)
tiny_page = tiny[0]
box = lay.Layout(page_rect=tiny_page.rect)
box.add_text((2, 2, 88, 10))
placement = box.place(1, "P1", (45.0, 35.0), 0.0, 20.0, 8.0)
check("имя находит место на крошечном листе", placement is not None)
check(
    "имя не вылезло за лист",
    placement.rect.x0 >= tiny_page.rect.x0 - 0.01
    and placement.rect.x1 <= tiny_page.rect.x1 + 0.01
    and placement.rect.y0 >= tiny_page.rect.y0 - 0.01
    and placement.rect.y1 <= tiny_page.rect.y1 + 0.01,
    str(placement.rect),
)

print("5. детектор и раскладка согласованы по именам")
result = detect.detect_page(doc[0], detect.DimParams())
names_in_json = {line.name for line in result.lines}
layout = lay.layout_for_page(doc[0], result.lines, FONT_SIZE, width_of)
names_drawn = {p.name for p in layout.placements}
check("имена на картинке совпадают с именами в выгрузке",
      names_in_json == names_drawn, f"{names_in_json ^ names_drawn}")

print()
if FAILURES:
    print(f"ПРОВАЛЕНО: {', '.join(FAILURES)}")
    raise SystemExit(1)
print("Все проверки пройдены.")