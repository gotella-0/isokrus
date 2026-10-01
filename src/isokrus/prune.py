"""Удаление лишнего с листа: текст вырезается, линии закрашиваются.

Два разных способа для двух разных вещей, и это не выбор вкуса, а следствие
того, что видно на листе.

**Текст вырезается** из PDF: ``apply_redactions`` убирает подпись из
содержимого страницы, и в рендере её нет вообще — ни серого ореола от
сглаживания, ни обрезка буквы.

**Линии закрашиваются**, а не вырезаются. Причина — пересечения. Лист
пересекается: выноска опорного знака упирается в трубу, стрелка примыкает к
размерной линии, и на пересечении стоят два элемента, из которых удаляется
один. Вырезание по режиму «задетая графика» уносит оба и оставляет после
себя обрывок трубы или размерной линии без наконечника. Заливка «наглухо»
тоже не годится: закрасив пересечение целиком, срезаешь кусок нужной линии.

Поэтому закрашивание идёт **по пикселям, голосованием**. Для каждого пикселя
смотрится, что его держит:

* если хоть один элемент, который нужно оставить, проходит через пиксель —
  пиксель остаётся чёрным;
* если через него проходят только удаляемые элементы — закрашивается;
* два удаляемых элемента, пересекшиеся в пикселе, закрашиваются вместе:
  «очистка тут уже была» означает, что и второй элемент удаляем.

Для голосования нужны две карты: где лежит удаляемое и где лежит оставляемое.
Обе строятся прямо из векторов — отрисовкой на чистых страницах того же
формата, а **не** вырезанием из копии листа. Причина в том, что
``apply_redactions`` смотрит на bbox элемента, а не на его форму: у огрызка
рамки bbox шире, чем прямоугольник по геометрии самого огрызка, и элемент не
вырезается никогда — сколько прямоугольник ни наращивай. Карта, полученная
вырезанием, отличалась от исходной картинки на 208 пикселей при наращивании
от 0 до 12 пунктов, то есть ровно ни на что, и закрашивать по ней было нечем.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pymupdf

from . import classify
from .raster import as_rgb, flush
from .redact import _segment_hits

# На сколько прямоугольник выреза шире подписи: рамка знака нарисована на
# пару пунктов шире текста внутри неё.
FRAME_PAD = 2.4
# На сколько область удаления шире элемента при решении, что элемент удаляемый.
# Запас нужен на обводку: отрезок тоньше своего прямоугольника, и без запаса
# он в него не помещается.
ITEM_PAD = 1.2
# Толщина линии, которой рисуются карты. Тоньше она не рисуется вовсе, а
# толще — голосование становится слишком щедрым и оставляет огрызки.
STROKE = 0.5


@dataclass(frozen=True)
class Plan:
    """Что и как убираем с листа."""
    page_number: int
    # Подписи: вырезаются из содержимого страницы.
    text: tuple[tuple[float, float, float, float], ...] = ()
    texts: tuple[str, ...] = ()
    # Графика: только закрашивается, из PDF не вырезается.
    lines: tuple[tuple[float, float, float, float], ...] = ()
    # Размерные линии и их выноски: закрашивать нельзя.
    keep: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()

    @property
    def total(self) -> int:
        return len(self.text) + len(self.lines)

    def counts(self) -> dict[str, int]:
        return {"подписи": len(self.text), "графика": len(self.lines)}


def _keep_geometry(scan) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Что нельзя закрашивать: размерные линии и их выноски."""
    out: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for line in scan.lines:
        out.append(((line.a[0], line.a[1]), (line.b[0], line.b[1])))
        if line.leader_end is not None:
            out.append(((line.a[0], line.a[1]),
                        (line.leader_end[0], line.leader_end[1])))
    return tuple(out)


def build_plan(pdf_page: pymupdf.Page, scan, only_digits: bool = True) -> Plan:
    """План удаления по ролям, назначенным регуляркой и разбором.

    Прямоугольник подписи расширяется до рамки, в которую она попала: рамка
    опорного знака нарисована шире текста, и по одному тексту она в
    прямоугольник не помещается — остаётся огрызок в виде скобки.
    """
    elements = classify.assign_roles(
        classify.elements_of(pdf_page, only_with_digits=only_digits),
        [tuple(line.label_bbox) for line in scan.lines if line.label_bbox],
    )
    frames = [pymupdf.Rect(c.rect) for c in scan.contours]

    texts: list[tuple[float, float, float, float]] = []
    names: list[str] = []
    for element in elements:
        if element.role not in classify.CUT_ROLES:
            continue
        rect = pymupdf.Rect(element.rect)
        for frame in frames:
            grown = pymupdf.Rect(frame.x0 - 1, frame.y0 - 1,
                                 frame.x1 + 1, frame.y1 + 1)
            if grown.contains(rect):
                rect |= frame
                break
        rect = pymupdf.Rect(rect.x0 - FRAME_PAD, rect.y0 - FRAME_PAD,
                            rect.x1 + FRAME_PAD, rect.y1 + FRAME_PAD)
        texts.append(tuple(rect))
        names.append(element.role)

    return Plan(
        page_number=pdf_page.number + 1,
        text=tuple(texts), texts=tuple(names),
        lines=tuple(_removable_lines(pdf_page, scan, texts)),
        keep=_keep_geometry(scan),
    )


def _touches_keep(rect: pymupdf.Rect, keep: Sequence[tuple], pad: float = 0.6) -> bool:
    """Проходит ли через прямоугольник хоть одна размерная линия."""
    grown = pymupdf.Rect(rect.x0 - pad, rect.y0 - pad,
                         rect.x1 + pad, rect.y1 + pad)
    return any(_segment_hits(pymupdf.Point(a), pymupdf.Point(b), grown)
               for a, b in keep)


def _removable_lines(
    pdf_page: pymupdf.Page, scan, texts: Sequence[tuple[float, ...]],
) -> list[tuple[float, float, float, float]]:
    """Графика под закрашивание: рамки знаков и выноски без подписи.

    Рамка — контур, в который попала удаляемая подпись. Выноска — тонкий
    отрезок, упирающийся в удаляемую рамку; если отрезок совпадает с
    размерной линией, закрашивать его нельзя, и он отбрасывается.
    """
    from .clean import _segments  # локально: clean -> prune

    keep = _keep_geometry(scan)
    boxes = [pymupdf.Rect(t) for t in texts]
    out: list[tuple[float, float, float, float]] = []
    seen: set[tuple[int, int, int, int]] = set()

    def add(rect: pymupdf.Rect) -> None:
        key = tuple(int(v) for v in rect)
        if key in seen or _touches_keep(rect, keep):
            return
        seen.add(key)
        out.append(tuple(rect))

    for contour in scan.contours:
        frame = pymupdf.Rect(contour.rect)
        grown = pymupdf.Rect(frame.x0 - 1, frame.y0 - 1,
                             frame.x1 + 1, frame.y1 + 1)
        if any(grown.contains(box) for box in boxes):
            add(frame)

    for a, b in _segments(pdf_page.get_drawings(), scan.limits):
        pa, pb = pymupdf.Point(a), pymupdf.Point(b)
        if abs(pa.x - pb.x) + abs(pa.y - pb.y) < 3.0:
            continue
        span = pymupdf.Rect(pa, pb)
        if not any(box.contains(pa) or box.contains(pb) for box in boxes):
            continue
        add(span)

    return out


# --- Карты для голосования ------------------------------------------------


def _item_rect(item) -> pymupdf.Rect | None:
    """Прямоугольник, покрывающий один элемент рисунка."""
    kind = item[0]
    if kind == "l":
        return pymupdf.Rect(item[1], item[2])
    if kind == "re":
        return pymupdf.Rect(item[1])
    if kind == "qu":
        return pymupdf.Rect(item[1].rect)
    if kind == "c":
        return pymupdf.Rect(item[1]) | pymupdf.Rect(item[4])
    return None


def _paint_item(page: pymupdf.Page, item, width: float) -> None:
    """Нарисовать один элемент рисунка чёрным, заливкой или обводкой."""
    kind = item[0]
    ink = (0, 0, 0)
    line = max(width, STROKE)
    if kind == "l":
        page.draw_line(item[1], item[2], color=ink, width=line)
    elif kind == "re":
        page.draw_rect(item[1], color=None, fill=ink, width=0)
    elif kind == "qu":
        page.draw_quad(item[1], color=None, fill=ink, width=0)
    elif kind == "c":
        page.draw_bezier(item[1], item[2], item[3], item[4],
                         color=ink, width=line)


def _blank_like(pdf_page: pymupdf.Page) -> tuple[pymupdf.Document, pymupdf.Page]:
    """Пустая страница того же размера, что и лист."""
    document = pymupdf.open()
    page = document.new_page(width=pdf_page.rect.width,
                             height=pdf_page.rect.height)
    return document, page


def _rasterize(page: pymupdf.Page, dpi: float) -> pymupdf.Pixmap:
    return page.get_pixmap(dpi=dpi, alpha=False)


def _inside(rect: pymupdf.Rect, areas: Sequence[pymupdf.Rect]) -> bool:
    """Помещается ли элемент целиком в одну из областей удаления."""
    return any(area.contains(rect) for area in areas)


def maps(
    pdf_page: pymupdf.Page,
    plan: Plan,
    dpi: float,
) -> tuple[pymupdf.Pixmap, pymupdf.Pixmap]:
    """Две карты листа: где удаляемое, где оставляемое.

    Обе рисуются заново из векторов, поэтому они не зависят от того, как
    PyMuPDF вырезает графику, и остаются верными там, где вырезание молчит.

    Подписи, которые остаются на листе, тоже попадают в карту оставляемого
    как залитые прямоугольники: текста в ``get_drawings()`` нет, а закрасить
    подпись размера нельзя. Подписи из плана удаления, наоборот, в карту
    оставляемого не идут — они и так вырезаны из страницы.
    """
    areas = [pymupdf.Rect(r) for r in plan.lines]
    grown = [pymupdf.Rect(a.x0 - ITEM_PAD, a.y0 - ITEM_PAD,
                          a.x1 + ITEM_PAD, a.y1 + ITEM_PAD) for a in areas]
    dropped = [pymupdf.Rect(t) for t in plan.text]

    cut_doc, cut = _blank_like(pdf_page)
    keep_doc, stay = _blank_like(pdf_page)

    for path in pdf_page.get_drawings():
        width = float(path.get("width") or 0.0)
        for item in path["items"]:
            box = _item_rect(item)
            if box is None or box.is_empty:
                continue
            if _inside(box, grown):
                _paint_item(cut, item, width)
            else:
                _paint_item(stay, item, width)

    for block in pdf_page.get_text("blocks"):
        box = pymupdf.Rect(block[:4])
        if box.is_empty or _inside(box, grown + dropped):
            continue
        stay.draw_rect(box, color=None, fill=(0, 0, 0), width=0)

    cut_pixmap = _rasterize(cut, dpi)
    keep_pixmap = _rasterize(stay, dpi)
    cut_doc.close()
    keep_doc.close()
    return cut_pixmap, keep_pixmap


def _overlay(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Привести чужую карту к размеру рабочей, обрезав или дополнив."""
    if mask.shape == (height, width):
        return mask
    out = np.zeros((height, width), dtype=bool)
    rows = min(height, mask.shape[0])
    cols = min(width, mask.shape[1])
    out[:rows, :cols] = mask[:rows, :cols]
    return out


def paint(
    pixmap: pymupdf.Pixmap,
    cut: pymupdf.Pixmap,
    stay: pymupdf.Pixmap,
    threshold: int = 250,
) -> int:
    """Закрасить лишнее голосованием по пикселям.

    ``cut`` — карта удаляемого, ``stay`` — карта оставляемого. Пиксель
    закрашивается, только если удаляемое в нём есть, а оставляемого нет:
    чёрное в ``stay`` означает, что здесь проходит нужный элемент. Два
    удаляемых элемента, сошедшиеся в пикселе, закрашиваются вместе — оба
    в ``stay`` отсутствуют.

    Считается целиком в numpy, тремя операциями над массивами. По пикселям из
    Python тот же результат на порядки медленнее: на листе при 600 dpi это
    десятки миллионов итераций, и каждая обращается к буферу растра, который
    копируется при каждом чтении.
    """
    image = as_rgb(pixmap)
    height, width = image.shape[:2]
    removable = _overlay(as_rgb(cut)[:, :, 0] < threshold, height, width)
    needed = _overlay(as_rgb(stay)[:, :, 0] < threshold, height, width)

    target = removable & (image[:, :, 0] < threshold) & ~needed
    painted = int(np.count_nonzero(target))
    if painted:
        image[target] = 255
        flush(pixmap, image)
    return painted


def redact_page(pdf_page: pymupdf.Page, plan: Plan) -> int:
    """Вырезать подписи из содержимого страницы.

    Только текст. Графика закрашивается на снимке, и вырезать её из PDF не
    нужно: вырезание не различает, чья это линия, и уносит вместе с нужной.

    Все подписи вырезаются **одним** вызовом. Это требование PyMuPDF, а не
    аккуратность: после ``apply_redactions`` страница переписывается, и
    следующий вызов по той же странице текст уже не находит.
    """
    if not plan.text:
        return 0
    for rect in plan.text:
        pdf_page.add_redact_annot(pymupdf.Rect(rect))
    pdf_page.apply_redactions(graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                              text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    return len(plan.text)


def redact_text(pdf_page: pymupdf.Page, plan: Plan) -> int:
    """Совместимое имя прежней функции."""
    return redact_page(pdf_page, plan)
