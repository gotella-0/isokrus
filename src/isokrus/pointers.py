"""Указатели на листе и их обозначения.

Зачем. Пока модели не видно, к какой линии относится подпись, вырезать нечего:
линия, на которую смотрит опорный знак, и размерная линия на картинке
неразличимы, и снося первую вместе со знаком можно снести вторую вместе с
наконечником. Поэтому указатели сначала находят и **обозначают буквами**, и
только потом модель смотрит на вырезы и говорит, чья это подпись.

Что такое указатель. Отрезок с **одним** наконечником на конце, у которого
свободный конец ни с чем не соединён. Это и есть выноска: знак или номер узла
стоит у её хвоста, наконечник упирается в то, на что знак указывает. У
размерной линии наконечников два, по краям, поэтому в указатели она не попадает
и указ��ателем быть не может.

Почему буквы, а не координаты. Модель не умеет считать пункты и не должна этим
заниматься: на вырезе нужна метка, которую можно назвать словами. Метка
рисуется прямо на картинке рядом с указателем, и ответ читается прямо: «эта
подпись у стрелки B2». Проверять потом нечего — указ��атель с меткой и есть
помеченный.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pymupdf

from . import config, dimscan

# Буквы обозначений. Сорок восемь — с запасом на лист с сотней указателей,
# но не больше: двузначная метка читается на вырезе уже плохо.
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _name(index: int) -> str:
    """Метка указателя по порядку: A, B, C, ..., AA."""
    if index < len(_ALPHABET):
        return _ALPHABET[index]
    return _ALPHABET[index // len(_ALPHABET) - 1] + _ALPHABET[index % len(_ALPHABET)]


@dataclass(frozen=True)
class Pointer:
    """Указатель на листе: отрезок с одним наконечником."""

    name: str                    # "B2" — метка, нарисованная на вырезе
    tip: tuple[float, float]     # конец с наконечником: куда указывает
    tail: tuple[float, float]    # свободный конец: где стоит подпись
    width: float

    def segment(self) -> tuple[pymupdf.Point, pymupdf.Point]:
        return pymupdf.Point(self.tip), pymupdf.Point(self.tail)

    def rect(self, pad: float = 2.0) -> pymupdf.Rect:
        a, b = self.segment()
        return pymupdf.Rect(min(a.x, b.x) - pad, min(a.y, b.y) - pad,
                            max(a.x, b.x) + pad, max(a.y, b.y) + pad)


def _triangles(drawings, max_area: float) -> list[tuple[tuple[float, float],
                                                        tuple[float, float],
                                                        tuple[float, float]]]:
    """Залитые треугольники как (остриё, крыло, крыло).

    Отличие от :func:`dimscan._arrow_tips` в том, что возвращается **весь**
    треугольник, а не только остриё. Это ровно то, что нужно, чтобы понять,
    чей это наконечник: у части линий треугольник нарисован поверх конца
    отрезка и остриё уходит вперёд, а у части — линия доходит до острия, и
    основание треугольника осталось позади. Сверять с одним остриём можно
    только во втором случае: на четвёртом листе двадцать восемь выносок не
    находились именно потому, что остриё стояло в 3.1 пункта от конца линии,
    а допуск ``arrow_gap`` равен одному пункту.
    """
    out = []
    for path in drawings:
        if not path.get("fill"):
            continue
        segments = list(dimscan._path_segments(path))
        if len(segments) != 3:
            continue
        if dimscan._signed_area(segments) > max_area:
            continue
        corners = {s[0] for s in segments} | {s[1] for s in segments}
        if len(corners) != 3:
            continue
        tip = max(corners,
                  key=lambda q: sum(dimscan.dist(q, other)
                                    for other in corners if other != q))
        wings = tuple(sorted(corners - {tip}))
        out.append((tip, wings[0], wings[1]))
    return out


def _own_end(triangles, end, gap: float) -> tuple[tuple[float, float], ...] | None:
    """Чей это конец отрезка: треугольник, стоящий на нём.

    Проверяется расстояние до **всей** площади треугольника — до острия и до
    обоих крыльев. Иначе наконечник, нарисованный поверх конца линии, не
    находится никогда, и выноска теряется целиком вместе со знаком, который
    она держит.
    """
    best: tuple[float, tuple[tuple[float, float], ...]] | None = None
    for tip, left, right in triangles:
        distance = min(dimscan.dist(end, tip), dimscan.dist(end, left),
                       dimscan.dist(end, right))
        if distance <= gap and (best is None or distance < best[0]):
            best = (distance, (tip, left, right))
    return best[1] if best else None


def of_page(page: pymupdf.Page, scan) -> tuple[Pointer, ...]:
    """Указатели листа с метками, в порядке сверху вниз.

    Допуск до наконечника берётся с запасом относительно ``arrow_gap``: сам
    порог отвечает на вопрос «наконечник ли это», а здесь нужно «стоит ли
    наконечник на этом конце», и треугольник в три пункта длиной отстоит от
    конца линии, нарисованной под ним.

    Порядок тот же, что у размеров в :func:`dimscan.detect_page` — по
    вертикали, — иначе номера на картинке и в отчёте не совпали бы с
    порядком чтения. Метки не зависят от номера листа: они относятся к
    вырезу, и в вырезе видны все, к которым этот элемент близок.
    """
    limits = scan.limits
    drawings = page.get_drawings()
    triangles = _triangles(drawings, limits.arrow_max_area)
    stroked = dimscan._stroked_segments(drawings, limits)
    # Наконечник выходит за конец линии на длину своего треугольника, и длина
    # эта у всех стрелок комплекта одна — в пределах полутора миллиметра.
    gap = limits.arrow_gap * config.POINTER_ARROW_SLACK

    # ``_stroked_segments`` отдаёт концы парами ``(x, y)``, а не точками:
    # это его собственный формат, и приводить всё к pymupdf.Point заранее
    # незачем — точки нужны только в конце, когда указатель уже признан.
    found: list[tuple[float, float, float, float, float, float]] = []
    for a, b, width in stroked:
        if dimscan.dist(a, b) < limits.leader_min_length:
            continue
        at_a = _own_end(triangles, a, gap) is not None
        at_b = _own_end(triangles, b, gap) is not None
        # Наконечник ровно один. Два — размерная линия, ноль — отрезок без
        # смысла: она не указывает ни на что.
        if at_a == at_b:
            continue
        tip, tail = (a, b) if at_a else (b, a)
        found.append((tail[0] + tail[1], tip[0], tip[1], tail[0], tail[1],
                      width))

    found.sort(key=lambda row: row[0])
    return tuple(
        Pointer(name=_name(index), tip=(row[1], row[2]),
                tail=(row[3], row[4]), width=row[5])
        for index, row in enumerate(found)
    )


def near(point: pymupdf.Rect, pointers: Sequence[Pointer],
         limit: float) -> tuple[Pointer, ...]:
    """Указатели, хвост которых стоит рядом с прямоугольником.

    ``limit`` — расстояние в пунктах от прямоугольника до хвоста. Подпись
    может стоять не у самого хвоста, а рядом, поэтому сравнивается с
    прямоугольником, а не с точкой.
    """
    out: list[tuple[float, Pointer]] = []
    for pointer in pointers:
        tail = pymupdf.Point(*pointer.tail)
        distance = _gap(point, tail)
        if distance <= limit:
            out.append((distance, pointer))
    out.sort(key=lambda pair: pair[0])
    return tuple(pointer for _, pointer in out)


def _gap(box: pymupdf.Rect, point: pymupdf.Point) -> float:
    """Расстояние от точки до прямоугольника, ноль внутри."""
    dx = max(box.x0 - point.x, 0.0, point.x - box.x1)
    dy = max(box.y0 - point.y, 0.0, point.y - box.y1)
    return (dx * dx + dy * dy) ** 0.5


# --- Обозначения на картинке -----------------------------------------------

# Цвет метки. Красный: на чёрно-белом листе выделенного цвета нет, а серый
# сливается с тонкой графикой и с антисглаженными краями букв.
INK = (0.85, 0.1, 0.1)


def labelled_page(
    page: pymupdf.Page,
    pointers: Sequence[Pointer],
    document: pymupdf.Document,
    size: float = 7.0,
) -> tuple[pymupdf.Page, pymupdf.Document]:
    """Копия страницы с нарисованными метками указателей.

    Возвращается и страница, и документ-владелец: страница без документа —
    это обращение к освобождённой памяти, и рисунок рассыпается на ровном
    месте. Вызывающий обязан держать пару до конца работы с вырезами.

    Исходный документ передаётся явно, потому что ``Page`` его не хранит: у
    страницы есть только ``xref``, и обратно к документу по нему не дойти.
    Принадлежность страницы документу приходится знать вызывающему.

    Копия нужна, а не сама страница: метки — подсветка для модели, и в
    содержимом PDF им не место. Всё остальное на листе остаётся нетронутым,
    включая подписи и линии, — иначе модель увидит не тот чертёж.
    """
    owner = pymupdf.open()
    copy = owner.new_page(width=page.rect.width, height=page.rect.height)
    copy.show_pdf_page(copy.rect, document, page.number)

    for pointer in pointers:
        tail = pymupdf.Point(*pointer.tail)
        tip = pymupdf.Point(*pointer.tip)
        # Метка ставится у хвоста и сдвинута от линии: на самом хвосте она
        # закрыла бы подпись, ради которой и вырезается.
        dx, dy = tail.x - tip.x, tail.y - tip.y
        span = max((dx * dx + dy * dy) ** 0.5, 1.0)
        at = pymupdf.Point(tail.x + dx / span * 2.5, tail.y + dy / span * 2.5)
        _write(copy, pointer.name, at, size)

    return copy, owner


def _write(page: pymupdf.Page, text: str, at: pymupdf.Point, size: float) -> None:
    """Подписать точку буквами с белой подложкой.

    Подложка обязательна: подпись размера может оказаться прямо под меткой, и
    без белого прямоугольника буквы метки смешаются с цифрами — модель решит,
    что перед ней «N214», и ответ будет про несуществующий элемент.
    """
    width = size * 0.62 * len(text)
    box = pymupdf.Rect(at.x - 1.0, at.y - size * 0.85,
                       at.x + width + 1.0, at.y + size * 0.35)
    page.draw_rect(box, color=None, fill=(1, 1, 1), width=0)
    page.insert_text(
        (at.x, at.y), text, fontsize=size, fontname="hebo",
        color=INK,
    )
