"""Лист с подсвеченными размерами — то, что уходит в модель вместо фото.

Зачем. Модель читает числа на картинке и читает их неверно: на пяти листах из
десяти и ``glm-5.3-flash``, и ``gpt-6-luna`` независимо писали в примечаниях
«22100» вместо ``24100``. Промах в одну цифру уезжает прямо в сумму, и разбор
вектора, который нашёл правильное число, не помогает — модель всё равно
смотрит на картинку своим глазом.

Что делает подсветка:

* обводит рамкой **уже нарисованное** число — своё значение туда не вписывается,
  иначе на листе окажется рядом два разных числа, а подпись закрыется;
* подсвечивает саму размерную линию, отступив от концов, чтобы исходные
  наконечники стрелок остались видны, и дорисовывает наконечники поверх: две
  стрелки означают отмеренный промежуток, одна — выноску, указывающую на зазор;
* подписывает у рамки метку ``P7`` — она отвечает на вопрос «какая строка блока
  разбора про это число».

Как рисуем. Поверх **тех же** пикселей, что ушли бы в модель: новая страница
размером с обрезанную картинка, в неё вставляется исходный PNG и поверх
рисуется подсветка. Кадрирование, DPI и обрезка поэтому гарантированно те же,
и рамки из блока разбора попадают на свои места без пересчётки.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import pymupdf

if TYPE_CHECKING:  # цикл: extract -> overlay -> extract
    from .dimscan import DimensionMark

# Цвета подсветки. Отдельный для выносок: те измеряют зазор между фланцами и в
# разборе трассы не участвуют, и на листе их должно быть видно иначе, чем
# обычные участки.
SPAN_COLOR = (0.85, 0.1, 0.55)
POINTER_COLOR = (0.05, 0.5, 0.9)

# Цвет рамки и плашки с меткой. Общий для обоих видов линий: плашка отвечает
# на вопрос «какая это цифра», а не «какого она вида», и путать её с цветом
# линии незачем. Зелёный выбран потому, что на белом листе с чёрными линиями он
# контрастен и не совпадает ни с одним из цветов подсветки.
FRAME_COLOR = (0.0, 0.6, 0.25)
BADGE_TEXT_COLOR = (1.0, 1.0, 1.0)

# Доли от меньшей стороны картинки. Толщина линии и кегль метки заданы от
# нативного текста чертежа: подсветка — пометка поверх чертежа, а не замена
# ему, поэтому крупнее собственного текста рисовать её нельзя.
STROKE_RATIO = 0.0030
MARK_RATIO = 0.0080
ARROW_RATIO = 0.012
FRAME_PAD_RATIO = 0.0022


def _accent(color: tuple[float, float, float]) -> tuple[float, float, float]:
    """Затемнённый вариант цвета линии — для наконечников стрелок.

    Наконечник лежит на самой линии, поэтому в её собственном цвете он
    сливается: тот же розовый на розовом читается как утолщение, а не как
    стрелка. Затемнение даёт контраст, не вводя второго оттенка розового —
    иначе на листе появятся «разные» размеры, которых на чертеже нет.
    """
    return tuple(max(0.0, c * 0.48) for c in color)  # type: ignore[return-value]


def _away(mid: pymupdf.Point, end: pymupdf.Point) -> pymupdf.Point:
    """Точка за концом отрезка по направлению от его середины.

    На неё смотрит наконечник: у отрезка со стрелками это оба конца, у выноски
    — тот, где стрелка одна.
    """
    return pymupdf.Point(mid.x * 2 - end.x, mid.y * 2 - end.y)


def _inset(
    start: pymupdf.Point, end: pymupdf.Point, trim: float,
) -> tuple[pymupdf.Point, pymupdf.Point] | None:
    """Отступить оба конца на ``trim``, чтобы не закрыть наконечники стрелок."""
    dx, dy = end.x - start.x, end.y - start.y
    length = (dx * dx + dy * dy) ** 0.5
    if length <= 2 * trim:
        return None
    ux, uy = dx / length, dy / length
    return (
        pymupdf.Point(start.x + ux * trim, start.y + uy * trim),
        pymupdf.Point(end.x - ux * trim, end.y - uy * trim),
    )


def _draw_arrow(
    sheet: pymupdf.Page,
    tip: pymupdf.Point,
    outward: pymupdf.Point,
    size: float,
    color: tuple[float, float, float],
) -> None:
    """Наконечник стрелки, смотрящий от ``tip`` в сторону ``outward``."""
    dx, dy = outward.x - tip.x, outward.y - tip.y
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / length, dy / length
    back = pymupdf.Point(tip.x - ux * size, tip.y - uy * size)
    left = pymupdf.Point(back.x - uy * size * 0.42, back.y + ux * size * 0.42)
    right = pymupdf.Point(back.x + uy * size * 0.42, back.y - ux * size * 0.42)
    sheet.draw_polyline(
        [tip, left, right, tip], color=color, fill=color, width=0.5,
    )


def _intersects(a: pymupdf.Rect, b: pymupdf.Rect, pad: float = 0.0) -> bool:
    return not (
        a.x1 + pad < b.x0
        or a.x0 - pad > b.x1
        or a.y1 + pad < b.y0
        or a.y0 - pad > b.y1
    )


def _fits(rect: pymupdf.Rect, width: int, height: int) -> bool:
    return rect.x0 >= 0 and rect.y0 >= 0 and rect.x1 <= width and rect.y1 <= height


def _plate(width: float, height: float, x: float, y: float) -> pymupdf.Rect:
    return pymupdf.Rect(x, y, x + width, y + height)


def _badge_rect(
    frame: pymupdf.Rect, text: str, mark_size: float,
) -> pymupdf.Rect:
    """Габариты плашки с номером размера на верхнем правом углу рамки.

    Метка намеренно перекрывает рамку и частично выходит за её пределы, а не
    стоит рядом: на чертеже рядом с числом обычно сама выносная линия или
    подпись другой метки, и отдельно стоящий ярлык сбивает с толку, к какой
    рамке он относится. Угол — единственное место, где это читается однозначно.

    Плашка заметно меньше самой цифры: две соседние цифры стоят в миллиметрах
    друг от друга, и крупная плашка перекрывает соседнюю — тогда номер
    перестаёт соответствовать своей рамке. Ширина под жирный шрифт: «P12» в
    helvetica-bold заметно шире, чем в обычном, и при узкой плашке последняя
    цифра упирается в край.
    """
    height = max(mark_size * 1.15, frame.height * 0.52)
    width = max(height * 1.05, len(text) * height * 0.66)
    return pymupdf.Rect(
        frame.x1 - width * 0.28,
        frame.y0 - height * 0.34,
        frame.x1 - width * 0.28 + width,
        frame.y0 - height * 0.34 + height,
    )


def _draw_badge(
    sheet: pymupdf.Page,
    frame: pymupdf.Rect,
    text: str,
    mark_size: float,
) -> pymupdf.Rect:
    """Плашка с номером размера: заливка, номер белым по центру."""
    rect = _badge_rect(frame, text, mark_size)
    sheet.draw_rect(rect, color=FRAME_COLOR, fill=FRAME_COLOR, width=0.5)
    sheet.insert_text(
        (
            rect.x0 + (rect.width - len(text) * rect.height * 0.40) / 2,
            rect.y0 + rect.height * 0.5 + rect.height * 0.34,
        ),
        text,
        fontsize=rect.height * 0.70,
        fontname="hebo",
        color=BADGE_TEXT_COLOR,
    )
    return rect


# Единственный режим подсветки. Концы размерных отрезков обводятся рамками
# поверх уже нарисованных наконечников и засечек — ничего нового не рисуется,
# иначе поверх листа появляется вторая размерная линия, которой на чертеже нет.
MODE_FULL = 2
MODE_OFF = 0


def _normal(start: pymupdf.Point, end: pymupdf.Point) -> pymupdf.Point:
    """Единичная нормаль к оси отрезка.

    Рамка на конце отрезка строится по ней, поэтому у отрезка под 45° получается
    ромб, а не ровный квадрат. Это сознательно: рамка должна быть соразмерна
    наконечнику, а не подстраиваться под ось.
    """
    dx, dy = end.x - start.x, end.y - start.y
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return pymupdf.Point(1.0, 0.0)
    return pymupdf.Point(-dy / length, dx / length)


def _outline_end(
    sheet: pymupdf.Page,
    tip: pymupdf.Point,
    normal: pymupdf.Point,
    side: float,
    color: tuple[float, float, float],
    width: float,
) -> None:
    """Обвести наконечник или засечку, уже нарисованные на чертеже.

    Именно обвести, а не дорисовать. Засечки и стрелки на листе есть, и
    повторная отрисовка своих наконечников поверх дала бы вторую размерную
    линию, которой на чертеже нет: модель посчитала бы лишний размер, а
    читающему человек документ это просто неверно. Рамка поверх — единственный
    способ показать «вот этот конец замера», не вмешиваясь в геометрию.
    """
    nx, ny = normal.x, normal.y
    sheet.draw_rect(
        pymupdf.Rect(
            tip.x + nx * side, tip.y + ny * side,
            tip.x - nx * side, tip.y - ny * side,
        ),
        color=color,
        width=width,
    )


def render_overlay(
    data: bytes,
    width: int,
    height: int,
    marks: Sequence["DimensionMark"],
    mode: int = MODE_FULL,
) -> bytes:
    """PNG того же размера, что и исходный, с пометками найденных размеров.

    Принимает байты, а не страницу: подсветка строится до того, как объект
    страницы собран, и тянуть его ради трёх полей незачем.

    Без размеров (или в режиме ``MODE_OFF``) возвращается исходная картинка —
    вызывающий должен понимать, что подсветки не будет, и не считать, что она
    отрисовалась «в никуда».
    """
    if not marks or mode == MODE_OFF:
        return data

    unit = min(width, height)
    stroke = max(1.4, unit * STROKE_RATIO)
    mark_size = max(7.5, unit * MARK_RATIO)
    arrow = max(3.5, unit * ARROW_RATIO)
    pad = max(2.0, unit * FRAME_PAD_RATIO)

    document = pymupdf.open()
    try:
        sheet = document.new_page(width=width, height=height)
        sheet.insert_image(sheet.rect, stream=data)

        # Занятые места — рамки подписей размеров (накрывать их нельзя) и уже
        # поставленные метки.
        taken: list[pymupdf.Rect] = []

        for mark in marks:
            color = POINTER_COLOR if mark.kind != "span" else SPAN_COLOR
            taken.append(pymupdf.Rect(*mark.bbox))

            start = pymupdf.Point(*mark.line_start)
            end = pymupdf.Point(*mark.line_end)
            if start.distance_to(end) < 1.0:
                # Вырожденная линия — рисуем по рамке подписи, иначе на листе
                # будет метка без единого пикселя подсветки.
                x0, y0, x1, y1 = mark.bbox
                start = pymupdf.Point(x0, (y0 + y1) / 2)
                end = pymupdf.Point(x1, (y0 + y1) / 2)

            middle = _inset(start, end, arrow)
            head = _accent(color)
            if middle is None:
                sheet.draw_line(start, end, color=color, width=stroke)
                middle = (start, end)
            else:
                sheet.draw_line(middle[0], middle[1], color=color, width=stroke)
            mid = pymupdf.Point((start.x + end.x) / 2, (start.y + end.y) / 2)
            _draw_arrow(sheet, middle[0], _away(mid, start), arrow, head)
            if mark.kind == "span":
                _draw_arrow(sheet, middle[1], _away(mid, end), arrow, head)

            # Границы замера: обводим наконечники, которые на чертеже уже есть.
            # У выноски обводим только конец, который указывает на зазор.
            edge = arrow * 1.15
            normal = _normal(start, end)
            _outline_end(sheet, start, normal, edge, head,
                         max(1.0, stroke * 0.6))
            _outline_end(sheet, end, normal, edge, head,
                         max(1.0, stroke * 0.6))

            # Число на листе уже нарисовано — обводим его рамкой и ничего в
            # рамку не кладём. Зазор маленький: рамка должна обводить число
            # плотно, иначе непонятно, к какой подписи она относится.
            frame = pymupdf.Rect(
                mark.bbox[0] - pad, mark.bbox[1] - pad,
                mark.bbox[2] + pad, mark.bbox[3] + pad,
            )
            sheet.draw_rect(frame, color=FRAME_COLOR, width=max(1.0, stroke * 0.6))
            taken.append(frame)
            taken.append(
                _draw_badge(sheet, frame, mark.line_id, mark_size)
            )

        return sheet.get_pixmap(alpha=False).tobytes("png")
    finally:
        document.close()
