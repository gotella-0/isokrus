"""Раскладка имён отрезков (P1, P14) без пересечений с текстом и друг с другом.

Зачем отдельный модуль: имена нужны всем трём вариантам отрисовки, а
правила их размещения — это не рисование, а поиск свободного места. Держать
это в ``render.py`` означало бы смешать геометрию с проверками, и при
изменении правил пришлось бы трогать всё сразу.

Что считается препятствием:

* **текст листа** — все слова из текстового слоя PDF. Подпись размера,
  координаты привязки, номера узлов в рамках: всё это нарисовано на
  чертеже и наезжать на него нельзя;
* **уже размещённые имена** — чтобы P3 не встал на P7;
* **сами отрезки и их выноски** — имя, перечёркивающее цветную линию,
  читается хуже, чем имя сбоку.

Как ищется место: у каждого отрезка есть опорная точка (середина) и
направление. Перебираются позиции по сетке «расстояние × угол» вокруг этой
точки, от ближайшей к самой далёкой, и берётся первая свободная. Углы
перебираются не по кругу равномерно, а в порядке предпочтения: сначала
перпендикуляр к отрезку (имя сбоку — самое читаемое), потом вдоль, потом
остальные. Это даёт предсказуемый результат: имена встают аккуратными
рядами по краям размерных линий, а не разбросаны по листу.

Если свободного места не нашлось даже на самом дальнем радиусе, имя
размещается в наименее плохом месте — с минимальной суммой пересечений.
Молча пропустить имя хуже: на листе безымянный отрезок выглядит так же,
как ненайденный.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import pymupdf

Pt = tuple[float, float]
Box = tuple[float, float, float, float]


# Насколько расширяем препятствие, чтобы имя не липло к букве впритык.
CLEARANCE = 1.6

# Радиусы поиска, кратные размеру шрифта. Первые три — имя рядом с
# отрезком, дальше — «в край», если рядом всё занято.
RADIUS_STEPS = (1.2, 2.0, 3.0, 4.5, 6.5, 9.0, 13.0)


@dataclass
class Placement:
    """Куда встало имя отрезка."""

    line_index: int
    name: str
    rect: pymupdf.Rect
    radius_step: float
    forced: bool = False  # не нашли чистого места, выбрали наименее плохое

    @property
    def origin(self) -> Pt:
        return (self.rect.x0, self.rect.y1)  # базовая линия, левый нижний угол


def _norm(dx: float, dy: float) -> Pt:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return (1.0, 0.0)
    return (dx / length, dy / length)


def _angles_for(angle: float) -> list[Pt]:
    """Направления смещения в порядке убывания привлекательности.

    Отрезок под углом ``angle`` (градусы, Y вниз). Перпендикуляр к нему —
    это куда имя встаёт по умолчанию: сбоку от линии, а не на ней. Дальше
    идут направления вдоль отрезка и диагонали между ними.
    """
    a = math.radians(angle)
    ux, uy = math.cos(a), math.sin(a)
    # Перпендикуляры к отрезку: Y вниз, поэтому «вправо-вверх» и «влево-вниз».
    nx, ny = uy, -ux
    return [
        (nx, ny),
        (-nx, -ny),
        (ux, uy),
        (-ux, -uy),
        _norm(nx + ux, ny + uy),
        _norm(nx - ux, ny - uy),
        _norm(-nx + ux, -ny + uy),
        _norm(-nx - ux, -ny - uy),
    ]


@dataclass
class Layout:
    """Раскладка имён по листу. Наполняется по одному листу."""

    page_rect: pymupdf.Rect
    # Препятствия-подписи: текст листа и уже поставленные имена.
    boxes: list[pymupdf.Rect] = field(default_factory=list)
    # Препятствия-линии: отрезки и выноски. Их пересекать нельзя, но
    # «пересечение» тут — расстояние от центра имени до линии.
    strokes: list[tuple[Pt, Pt]] = field(default_factory=list)
    # Что поставили, по порядку размещения.
    placements: list[Placement] = field(default_factory=list)

    # -- наполнение ---------------------------------------------------------

    def add_text(self, box: Box) -> None:
        """Подпись листа — препятствие наравне с уже поставленными именами."""
        self.boxes.append(_sized(box))

    def add_placed(self, rect: pymupdf.Rect) -> None:
        """Запомнить поставленное имя — чтобы следующее его обходило."""
        self.boxes.append(rect)

    def add_stroke(self, a: Pt, b: Pt) -> None:
        if dist(a, b) > 0.5:
            self.strokes.append((a, b))

    # -- проверки ------------------------------------------------------------

    def _inside_page(self, rect: pymupdf.Rect) -> bool:
        return (
            rect.x0 >= self.page_rect.x0
            and rect.x1 <= self.page_rect.x1
            and rect.y0 >= self.page_rect.y0
            and rect.y1 <= self.page_rect.y1
        )

    def _clear_of_text(self, rect: pymupdf.Rect) -> bool:
        grown = pymupdf.Rect(
            rect.x0 - CLEARANCE, rect.y0 - CLEARANCE,
            rect.x1 + CLEARANCE, rect.y1 + CLEARANCE,
        )
        return not any(_overlaps(grown, other) for other in self.boxes)

    def _clear_of_strokes(self, rect: pymupdf.Rect) -> bool:
        """Не наезжает ли имя на отрезок или выноску.

        Проверяется расстояние от центра имени до линии: имя — короткая
        надпись, и «пересечение» с линией толщиной в полпикселя читается
        именно как расстояние до неё, а не как пересечение прямоугольников.
        """
        cx = (rect.x0 + rect.x1) / 2.0
        cy = (rect.y0 + rect.y1) / 2.0
        radius = max(rect.width, rect.height) * 0.35 + 1.0
        return all(
            dist_point_seg((cx, cy), a, b) > radius for a, b in self.strokes
        )

    def is_free(self, rect: pymupdf.Rect) -> bool:
        return self._inside_page(rect) and self._clear_of_text(rect) and self._clear_of_strokes(rect)

    def penalty(self, rect: pymupdf.Rect) -> float:
        """Насколько имя неудобно стоит — меньше лучше.

        Считаем и налезание на текст, и выход за лист, и близость к линиям.
        Нужно только для последнего рубежа: когда свободных мест не осталось,
        выбираем наименее плохое, а не бросаем имя и не рисуем поверх текста.
        """
        grown = pymupdf.Rect(
            rect.x0 - CLEARANCE, rect.y0 - CLEARANCE,
            rect.x1 + CLEARANCE, rect.y1 + CLEARANCE,
        )
        score = 40.0 * sum(1 for other in self.boxes if _overlaps(grown, other))
        if not self._inside_page(rect):
            score += 400.0
        cx = (rect.x0 + rect.x1) / 2.0
        cy = (rect.y0 + rect.y1) / 2.0
        for a, b in self.strokes:
            d = dist_point_seg((cx, cy), a, b)
            if d <= max(rect.width, rect.height) * 0.35 + 1.0:
                score += 15.0
        return score

    # -- размещение ----------------------------------------------------------

    def place(
        self,
        line_index: int,
        name: str,
        anchor: Pt,
        angle: float,
        width: float,
        font_size: float,
    ) -> Placement:
        """Поставить имя отрезка в ближайшее свободное место.

        Возвращает размещение всегда: если свободного места не нашлось,
        выбирается наименее плохое и ставится ``forced=True``. На листе без
        свободного места имя всё равно нужно — иначе отрезок выглядит так,
        будто его не нашли.
        """
        best: tuple[float, pymupdf.Rect] | None = None
        for step in RADIUS_STEPS:
            radius = font_size * step
            for dx, dy in _angles_for(angle):
                rect = _place_box(anchor, dx * radius, dy * radius, width, font_size)
                if self.is_free(rect):
                    placement = Placement(line_index, name, rect, step)
                    self.add_placed(rect)
                    self.placements.append(placement)
                    return placement
                score = self.penalty(rect)
                if best is None or score < best[0]:
                    best = (score, rect)

        assert best is not None  # RADIUS_STEPS и _angles_for не пусты
        placement = Placement(
            line_index, name, best[1], RADIUS_STEPS[-1], forced=True
        )
        self.add_placed(best[1])
        self.placements.append(placement)
        return placement


# ---------------------------------------------------------------------------
# Геометрия
# ---------------------------------------------------------------------------


def dist(a: Pt, b: Pt) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def dist_point_seg(point: Pt, a: Pt, b: Pt) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return dist(point, a)
    t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy))


def _overlaps(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
    return not (
        a.x1 <= b.x0 or a.x0 >= b.x1 or a.y1 <= b.y0 or a.y0 >= b.y1
    )


def _sized(box: Box) -> pymupdf.Rect:
    x0, y0, x1, y1 = box
    rect = pymupdf.Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    return rect


def _place_box(
    anchor: Pt, dx: float, dy: float, width: float, font_size: float
) -> pymupdf.Rect:
    """Рамка имени, смещённая от опорной точки.

    Смещение отсчитывается от **левого нижнего угла** текста, а не от его
    центра: иначе при разных направлениях смещения имя смещалось бы на
    половину своей ширины и «рядом с отрезком» переставало быть рядом.
    """
    x0 = anchor[0] + dx
    y1 = anchor[1] + dy
    return pymupdf.Rect(x0, y1 - font_size, x0 + width, y1)


def layout_for_page(
    page: pymupdf.Page,
    lines: Sequence,
    font_size: float,
    width_of,
) -> Layout:
    """Собрать раскладку для листа: препятствия из текста и сами линии.

    Отрезки добавляются в порядке убывания длины. Длинные идут первыми не
    из-за важности, а потому что им объективно труднее найти место: они
    занимают больше места, и у них меньше соседей. Короткие размещаются
    остатками.
    """
    layout = Layout(page_rect=page.rect)
    for x0, y0, x1, y1, *_ in page.get_text("words"):
        if (y1 - y0) >= 2.0:  # служебные штрихи в 1 pt — не препятствие
            layout.add_text((x0, y0, x1, y1))

    for line in lines:
        layout.add_stroke(line.a, line.b)
        if line.leader_end is not None:
            layout.add_stroke(line.midpoint, line.leader_end)

    for line in sorted(lines, key=lambda l: -l.length_pt):
        layout.place(
            line_index=line.index,
            name=line.name,
            anchor=line.midpoint,
            angle=line.angle,
            width=width_of(line.name, font_size),
            font_size=font_size,
        )
    # Возвращаем в порядке номеров, а не в порядке размещения: длинные
    # ставились первыми ради мест, но читать выгрузку и сверять с картинкой
    # удобнее по возрастанию P1, P2, ...
    layout.placements.sort(key=lambda p: p.line_index)
    return layout