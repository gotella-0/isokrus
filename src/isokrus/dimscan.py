"""Поиск размеров на изометрии: сначала числа, потом геометрия вокруг них.

Модуль живёт в пакете, а не в ``tests``, потому что он больше не только
инструмент разработчика: ``extract.py`` зовёт его на каждом листе **до**
рендера и обрезки, и найденный список уходит в промт модели. Пакет не должен
зависеть от каталога ``tests`` — при установке он туда не попадает.

Порядок в конвейере (``PDF -> dimscan -> trim -> LLM``) выбран не по привычке:

* детектор работает с **вектором** PDF в пунктах, поэтому обрезка полей на нём
  не сказывается никак, и ставить его можно где угодно до рендера;
* но держать его **до** рендера всё же правильно: это единственный шаг, где
  страница ещё открыта, и он даёт результат раньше, чем появится хоть один
  пиксель картинки — на тяжёлом листе это экономит заметное время;
* в промт и в разметку размеры попадают уже **в пикселях обрезанной картинки**,
  в той же системе координат, что и текстовый слой (см. :mod:`.extract`),
  иначе модель получит рамки, мимо которых сама не поведёт.

Зачем перевёрнут порядок внутри модуля. Поиск идёт от чисел, а не от
геометрии: «нашёл отрезок с наконечниками на обоих концах — подбери к нему
число». На листе 2 это дало две ошибки, и обе молчаливые:

* число ``9``, обведённое квадратом (позиционный знак), приклеилось к размеру
  и попало в сумму;
* число ``13`` на листе 3 пропало совсем, потому что рядом с ним нет отрезка с
  двумя наконечниками — есть только **выноска**: одна стрелка, указывающая на
  узкий зазор между фланцами. Такой размер начертежён, но не измерен линией.

Порядок «сначала число» снимает обе. Число на листе — это и есть размер;
всё остальное (рамка, позиционный знак, марка прибора, координата привязки) —
тоже числа, и их надо отсечь явно и по геометрическим признакам, а не надеясь,
что рядом случайно окажется отрезок.

Отсев «не размер» — три независимых правила, каждое ловит свой класс
насекомых:

1. **строка из одного слова.** Подпись размера стоит на листе сама по себе.
   Рядом с координатой (``X 267000``), маркой трубы (``50 mm PE``), названием
   прибора или строкой штампа всегда есть ещё слова — ``get_text("words")``
   отдаёт их одной строкой, и число среди них не размер.
2. **замкнутый контур.** Позиционный знак, номер опоры и марка прибора
   нарисованы квадратом или окружностью. Контур ищется в геометрии: связная
   компонента пути, у которой все концы замкнуты в петлю (квадрат, нередко
   нарисованный одним путём вместе с выноской), либо фигура, у которой нет
   ни одной точки внутри выпуклой оболочки (круг, в том числе пунктирный).
3. **высота шрифта.** Служебные цифры под позиционными знаками набраны
   кеглем втрое меньше размерных.

Дальше число ищет, что рядом:

* **отрезок** — тонкая линия с наконечниками на обоих концах. Её длина и есть
  размер; число стоит сбоку от середины;
* **выноска-указатель** — тонкая линия ровно с одним наконечником, свободный
  конец которой упирается в число. Так нарисован размер зазора, который
  измерить линией нельзя; на листе 3 это ``13``;
* **выноска при отрезке** — тот же вид линии, но её наконечник стоит на
  середине отрезка: это не отдельный размер, а подпись, отведённая в сторону,
  чтобы не мешать другим размерам (на листе 2 так подписаны ``224``).
  Различаются эти два случая именно тем, где стоит наконечник, поэтому порядок
  такой: сначала выноски, которые ни на какой отрезок не указывают, и только
  потом — остальные.

Число не может обслуживать два размера, и размер не может остаться с двумя
числами: всё раздаётся один-к-одному, ближайшим. Всё, что не досталось ни
одному размеру, попадает в ``orphan_numbers`` — их видно в статистике, молча
терять числа нельзя.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator, Sequence

import pymupdf

Pt = tuple[float, float]
Box = tuple[float, float, float, float]

# Толщина размерной линии в пунктах. Задаётся параметром, а не константой:
# у другого экспортированного комплекта толщина другая, но подбирается
# автоматически (:func:`_score_width`) по тому, у какой толщины больше всего
# отрезков со стрелками на обоих концах.
DEFAULT_WIDTH = 0.48

_NUM_RE = re.compile(r"^[<>]?\s*\d+(?:[.,]\d+)?$")


# ---------------------------------------------------------------------------
# Параметры поиска
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimParams:
    """Пороги отбора размеров, заданные для **эталонного** листа.

    Все длины — в пунктах PDF, и все они относятся к листу размером
    ``ref_glyph_height`` по высоте цифры подписи. На другом масштабе экспорта
    эти пункты не годятся напрямую: тот же чертёж, вписанный в лист вдвое
    крупнее, имел бы цифры высотой 16.8 pt, наконечники площадью 60 pt² и
    зазоры до подписей вдвое больше — и все пороги молча перестали бы
    подходить. Поэтому :func:`detect_page` пересчитывает их в
    :class:`Thresholds` по фактическому масштабу листа, а отсюда наружу
    отдаются уже приведённые к масштабу значения.

    Множитель один на весь лист и не зависит от того, какая это трасса:
    подписи, наконечники и выноски масштабируются вместе.
    """

    width: float | None = DEFAULT_WIDTH
    width_tol: float = 0.05
    # 12 pt, а не 20: на листе 10 есть размер 134 длиной 17 pt — зазор между
    # двумя фланцами, — и при пороге 20 он молча терялся, а на листе его
    # подпись «134» осталась без отрезка. Проверено: понижение до 8 pt не
    # добавляет ни одного ложного срабатывания, потому что отсекает уже
    # не длина, а наличие наконечников на обоих концах.
    min_length: float = 12.0
    # 0,2 pt, а не 1,0: кончик наконечника нарисован в той же точке, где
    # кончается размерная линия, — у всех размеров с наконечниками с двух
    # концов касание равно нулю. При пороге 1,0 штрих, задвший наконечник
    # чужой размерной линии, проходил за выноску и забирал чужое число, а
    # сама размерная линия оставалась без числа и выпадала из разбора.
    # Проверено: понижение до 0,05 не меняет ничего.
    arrow_gap: float = 0.2
    # 16 pt, а не 12: на листе 3 число «210» стоит в 14.7 pt от своего
    # короткого размера — размер с подписью впритык не помещается, CAD
    # отодвигает текст наружу. Порог берётся с запасом, а лишние числа
    # отсекаются тем, что каждое достаётся ровно одному размеру.
    label_gap: float = 16.0
    # 8 pt, а не 12: выноска-указатель на листе 3 («13») длиной 64 pt, но
    # на листе 5 номер «34» подписан выноской короче 20 pt, и при пороге 20
    # он переставал находиться. Проверено на всём комплекте: в диапазоне
    # 8..16 pt набор найденных размеров не меняется ни на одном листе,
    # а за его границами начинают теряться размеры.
    leader_min_length: float = 8.0
    # Свободный конец выноски должен доходить до числа вплотную. 12 pt — с
    # запасом на повёрнутый шрифт; на этих листах реальный зазор 0-2 pt.
    leader_label_gap: float = 12.0
    # Насколько конец выноски может отстоять от середины отрезка, чтобы
    # считаться его подписью, а не отдельным размером.
    leader_anchor_gap: float = 2.5
    # Площадь наконечника-треугольника. Единственная величина, которая
    # масштабируется не в линию, а в квадрат.
    arrow_max_area: float = 30.0
    min_label_height: float = 4.0
    # Соседнее слово на расстоянии меньше этого — часть той же подписи.
    # Размер нарисован сам по себе; «X 267000» и «50 mm PE» — нет.
    line_gap: float = 8.0
    # Контур вокруг числа: допуск, на который подпись может вылезти за рамку
    # (у текста свой em-бокс, он чуть больше нарисованной рамки), и запас от
    # её края, чтобы рамка не «ловила» число, стоящее рядом.
    frame_overshoot: float = 1.2
    frame_margin: float = 0.3
    frame_min_side: float = 5.0
    frame_max_side: float = 60.0
    # Круг и пунктирный круг не являются петлёй из связных отрезков: у них
    # много коротких дуг. Для них отдельное правило — «у фигуры нет точек
    # внутри выпуклой оболочки», и два порога: минимум отрезков, чтобы
    # треугольник-наконечник сюда не попал, и минимальное заполнение рамки,
    # чтобы треугольник не прошёл и по форме. Оба безразмерные.
    ring_min_segments: int = 8
    ring_min_fill: float = 0.55

    # Масштаб, к которому приведены все пороги выше.
    auto_scale: bool = True
    # Кегль подписи размера на эталонном листе, pt. Это и есть «единица»
    # чертежа: подписи, наконечники и выноски нарисованы в одном масштабе.
    #
    # Именно кегль, а не высота рамки слова: у повёрнутой подписи рамка
    # неповоротный прямоугольник и выше кегля в полтора раза, а на листе 8
    # половина размеров повёрнута на 30° — медиана рамок там 13.6 pt против
    # 8.4 pt на остальных листах, и по ней масштаб определился бы как 1.66.
    # Кегль от поворота не зависит: на всех десяти листах он 7.43..7.44 pt.
    ref_text_size: float = 7.43
    # Запасной якорь на случай, когда на листе почти нет подписей с цифрами
    # и медиана по ним неустойчива: меньшая сторона листа эталонного формата.
    ref_page_side: float = 841.89
    # Ниже этого множителя масштаб не доверяем: на листе, обрезанном до
    # куска трассы, медиана кеглей ещё осмысленна, а размер страницы — нет.
    min_scale: float = 0.2
    max_scale: float = 8.0


@dataclass(frozen=True)
class Thresholds:
    """Пороги, уже приведённые к масштабу конкретного листа.

    Отдельный тип нужен, чтобы нельзя было забыть про ``* scale`` в одном из
    мест: все внутренние функции принимают ``Thresholds`` и читают пороги
    напрямую, а пересчёт живёт в единственной функции :func:`_rescaled`.
    """

    width: float
    width_tol: float
    min_length: float
    arrow_gap: float
    label_gap: float
    leader_min_length: float
    leader_label_gap: float
    leader_anchor_gap: float
    arrow_max_area: float
    min_label_height: float
    line_gap: float
    frame_overshoot: float
    frame_margin: float
    frame_min_side: float
    frame_max_side: float
    ring_min_segments: int
    ring_min_fill: float

    def close_width(self, value: float) -> bool:
        return abs(value - self.width) <= self.width_tol


def sheet_scale(page: pymupdf.Page, params: DimParams) -> tuple[float, str]:
    """Во сколько раз лист крупнее эталонного, и откуда это взято.

    Основной якорь — медианный **кегль** подписей с цифрами: подписи,
    наконечники и выноски масштабируются вместе, и кегль отражает масштаб
    точнее, чем размер страницы (он меняется и от обрезки полей, и при
    экспорте на лист другого размера при той же аннотации).

    Кегль берётся из шрифтов, а не из рамок слов: ``get_text("dict")`` отдаёт
    ``size`` независимо от поворота, а неповоротная рамка повёрнутой подписи
    на 30° выше кегля в полтора раза. На листе 8, где половина размеров
    повёрнута, медиана рамок дала бы масштаб 1.66 вместо 1.0.

    Запасной якорь — меньшая сторона страницы. Он нужен на листе, где
    подписей с цифрами почти нет и медиана скачет: там лучше ошибиться в
    масштабе, чем потерять все размеры.

    Возвращается и источник: он попадает в статистику, потому что «почему
    множитель такой» — вопрос, на который иначе приходится отвечать
    догадками.
    """
    if not params.auto_scale:
        return 1.0, "задан вручную"
    sizes = sorted(
        round(float(span.get("size", 0.0)), 3)
        for block in page.get_text("dict")["blocks"] if block.get("type") == 0
        for line in block.get("lines", []) for span in line.get("spans", [])
        if span.get("size") and _NUM_RE.search(span.get("text", ""))
    )
    if len(sizes) >= 5 and params.ref_text_size > 0:
        median = sizes[len(sizes) // 2]
        return _clamp(median / params.ref_text_size, params, "кегль подписи")
    side = min(page.rect.width, page.rect.height)
    if params.ref_page_side > 0:
        return _clamp(side / params.ref_page_side, params, "сторона листа")
    return 1.0, "не определён"


def _clamp(value: float, params: DimParams, source: str) -> tuple[float, str]:
    if value < params.min_scale:
        return params.min_scale, f"{source} (обрезан снизу)"
    if value > params.max_scale:
        return params.max_scale, f"{source} (обрезан сверху)"
    return value, source


def _rescaled(params: DimParams, scale: float, width_pt: float) -> Thresholds:
    """Единственное место, где пороги умножаются на масштаб листа.

    Длины — в линию, площадь наконечника — в квадрат, безразмерные
    (заполнение рамки, число отрезков) не трогаются. ``width_pt`` передаётся
    уже в пунктах **этого** листа: толщину линий находят измерением, а не
    масштабированием, и пересчитывать её здесь было бы вторым домножением.
    """
    return Thresholds(
        width=width_pt,
        width_tol=params.width_tol * scale,
        min_length=params.min_length * scale,
        arrow_gap=params.arrow_gap * scale,
        label_gap=params.label_gap * scale,
        leader_min_length=params.leader_min_length * scale,
        leader_label_gap=params.leader_label_gap * scale,
        leader_anchor_gap=params.leader_anchor_gap * scale,
        arrow_max_area=params.arrow_max_area * scale * scale,
        min_label_height=params.min_label_height * scale,
        line_gap=params.line_gap * scale,
        frame_overshoot=params.frame_overshoot * scale,
        frame_margin=params.frame_margin * scale,
        frame_min_side=params.frame_min_side * scale,
        frame_max_side=params.frame_max_side * scale,
        ring_min_segments=params.ring_min_segments,
        ring_min_fill=params.ring_min_fill,
    )


# ---------------------------------------------------------------------------
# Что нашли
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumberMark:
    """Число-подпись, прошедшая отсев «не размер»."""

    text: str
    box: Box
    index: int = 0  # порядковый номер среди чисел листа

    @property
    def center(self) -> Pt:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    @property
    def value(self) -> float | None:
        try:
            return float(self.text.strip().lstrip("<>").replace(",", "."))
        except ValueError:
            return None


@dataclass(frozen=True)
class Contour:
    """Замкнутый контур: квадрат позиционного знака или круг марки прибора."""

    rect: pymupdf.Rect

    def wraps(self, box: Box, overshoot: float, margin: float) -> bool:
        """Находится ли подпись с рамкой ``box`` внутри контура.

        Два допуска, и оба нужны. ``margin`` отступает от края контура, чтобы
        рамка не «ловила» число, стоящее рядом с ней, а ``overshoot``
        разрешает подписи чуть вылезти за контур: у текста свой em-бокс, он на
        полпункта-два больше нарисованной рамки, и строгое попадание внутрь
        теряло бы половину позиционных знаков.
        """
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        r = self.rect
        if not (r.x0 + margin <= cx <= r.x1 - margin):
            return False
        if not (r.y0 + margin <= cy <= r.y1 - margin):
            return False
        return (
            r.x0 - overshoot <= box[0] and r.y0 - overshoot <= box[1]
            and r.x1 + overshoot >= box[2] and r.y1 + overshoot >= box[3]
        )


@dataclass(frozen=True)
class DimLine:
    """Один размер: либо отрезок, либо выноска-указатель.

    ``kind`` различает два вида, которые на листе выглядят одинаково, но
    значат разное:

    ``span``     линия с наконечниками на обоих концах; ``length_pt`` —
                 измеренная длина, ей соответствует значение;
    ``pointer``  выноска с одним наконечником; она **указывает** на зазор, а не
                 измеряет его, поэтому ``length_pt`` — длина нарисованной
                 линии, и никакого отношения к ``value`` не имеет.
    """

    page_number: int
    a: Pt
    b: Pt
    length_pt: float
    width: float
    angle: float  # градусы, 0 — горизонтально, Y вниз (как в PDF)
    label: str = ""
    label_bbox: Box | None = None
    index: int = 0
    kind: str = "span"
    # Конец выноски, если число отведено от отрезка. Нужен при отрисовке:
    # номер нельзя ставить на выноску, и на листе 10 номер 14 вставал ровно
    # на неё — вертикальный отрезок, а нормаль к нему горизонтальная, то
    # есть уходит вправо, туда же, куда уходит выноска.
    leader_end: Pt | None = None

    @property
    def midpoint(self) -> Pt:
        return ((self.a[0] + self.b[0]) / 2.0, (self.a[1] + self.b[1]) / 2.0)

    @property
    def measured(self) -> bool:
        """Верно ли, что длина линии — это сам размер."""
        return self.kind == "span"

    @property
    def value(self) -> float | None:
        """Число из подписи, если она числовая. Иначе ``None``."""
        if not _NUM_RE.match(self.label):
            return None
        try:
            return float(self.label.strip().lstrip("<>").replace(",", "."))
        except ValueError:
            return None

    @property
    def angle_rotated(self) -> bool:
        """Подпись/линия повёрнуты: на изометрии вертикальные размеры — не редкость."""
        return abs(math.sin(math.radians(self.angle))) > 0.5

    @property
    def name(self) -> str:
        """Имя отрезка для картинки и для JSON: ``P1``, ``P14``.

        Буква нужна, чтобы подпись отрезка не путать с числовой подписью
        размера на самом чертеже: на листе рядом стоят «P14» и «134», и без
        префикса обе выглядели бы как числа.
        """
        return f"P{self.index}"

    def as_dict(self) -> dict:
        return {
            "id": self.name,
            "page": self.page_number,
            "kind": self.kind,
            "measured": self.measured,
            "label": self.label,
            "value": self.value,
            "length_pt": round(self.length_pt, 2),
            "angle": round(self.angle, 1),
            "rotated": self.angle_rotated,
            "a": [round(self.a[0], 2), round(self.a[1], 2)],
            "b": [round(self.b[0], 2), round(self.b[1], 2)],
            "label_bbox": (
                [round(v, 2) for v in self.label_bbox] if self.label_bbox else None
            ),
            "leader_end": (
                [round(v, 2) for v in self.leader_end] if self.leader_end else None
            ),
        }


@dataclass
class DimNode:
    """Размер и его место в иерархии: что он суммирует и что суммирует его."""

    line: DimLine
    children: list[int] = field(default_factory=list)  # индексы отрезков внутри
    parent: int | None = None

    @property
    def is_overall(self) -> bool:
        """Обобщённый размер — тот, что идёт поверх остальных и включает их."""
        return bool(self.children)


@dataclass
class DimResult:
    """Всё, что нашли на листе: размеры, отсеянные числа, статистика."""

    page_number: int
    lines: list[DimLine] = field(default_factory=list)
    numbers: list[NumberMark] = field(default_factory=list)
    orphan_numbers: list[NumberMark] = field(default_factory=list)
    # Числа, отсеянные как «не размер», — значениями, в порядке чтения.
    # Уходят в промт модели: по картинке их отличить нечем.
    rejected_numbers: list[str] = field(default_factory=list)
    # Толщина размерных линий в пунктах **этого** листа, а не эталонного.
    width_used: float | None = None
    # Контуры (знаки в квадратах, опорные знаки) и пороги, приведённые к
    # масштабу листа. Отдаются наружу для зачистки (``clean.py``): она ищет
    # ровно те же рамки и обязана мерить их теми же порогами, иначе на листе
    # другого масштаба знаки найдутся не все. Считать второй раз незачем.
    contours: tuple["Contour", ...] = ()
    limits: "Thresholds | None" = None
    stats: dict = field(default_factory=dict)

    @property
    def values(self) -> list[float]:
        return [v for v in (l.value for l in self.lines) if v is not None]

    def hierarchy(self) -> list[DimNode]:
        """Кто из размеров обобщённый, а кто — его составные части.

        **Из рисунка это не определяется, и метод даёт лишь грубое
        приближение.** Причина измерена: лист нарисован не в масштабе — на
        листе 5 у соседних размеров от 1.34 до 20.95 мм на пункт, в 15 раз
        разница, — поэтому пропорцией длины проверить вложенность нельзя.
        Второе препятствие: ряды размеров, относящиеся к разным ветвям
        трассы, перекрываются по проекции, и на листе 5 размер «285» попадает
        внутрь проекции «2000», хотя человек считает их разными ветвями.

        Что здесь всё же работает — **два ряда одной цепочки**: обобщённый
        размер нарисован поверх звеньев, они параллельны и сдвинуты наружу.
        На листе 8 это 6000 + 6000 + 6000 + 4100 = 24100 и
        3800 + 5200 + 5000 + 6000 + 1850 = 23525. Сложив все 12 чисел листа,
        получим двойной счёт — ровно та ошибка, ради которой подсчёт и
        делают без повторов.

        Остальное — работа с картинкой и координатами привязки
        (``research/exp_04``): какой физической линии принадлежит размер и
        где проходят узлы. Проверка ``--check`` это разделяет: колонка
        «все размеры найдены» про поиск, колонка «разница» — про иерархию.

        **Сумма звеньев не обязана равняться обобщённому размеру**, и это не
        ошибка детектора: обобщённый размер меряет всю трассу целиком, а
        вложенные участки нарисованы не все. Поэтому ``arithmetic_ok`` —
        предупреждение для человека, а не критерий правильности разбора.

        Выноски-указатели (``kind="pointer"``) в иерархию не входят: они не
        измеряют, а указывают на зазор, и «накрывать» ими обобщённый размер
        нельзя. В сумму без повторов их значение входит самостоятельно.
        """
        measured = [line for line in self.lines if line.measured]
        nodes = [DimNode(line=line) for line in self.lines]
        for big in measured:
            kids = [s for s in measured if s is not big and _spans_over(big, s)]
            if len(kids) < 2:
                continue  # одиночное перекрытие — не разложение, а соседний размер
            big_node = nodes[big.index - 1]
            big_node.children = sorted(s.index for s in kids)
            for kid in kids:
                nodes[kid.index - 1].parent = big.index
        return nodes

    def chain_total(self) -> dict:
        """Суммы длин: как есть и без повторов.

        Расхождение между ними и есть величина ошибки наивного сложения всех
        чисел на листе.
        """
        nodes = self.hierarchy()
        covered: set[int] = set()
        for node in nodes:
            covered.update(node.children)
        unique = [n.line for n in nodes if n.line.index not in covered]
        inside = [n.line for n in nodes if n.parent is None]

        def total(items: list[DimLine]) -> float:
            return sum(v for v in (l.value for l in items) if v is not None)

        by_index = {n.line.index: n.line for n in nodes}
        overall = [
            {
                # id — имя P1..Pn, children — те же имена. Внутри иерархии
                # по-прежнему индексы: они 1-based и совпадают с номером,
                # а имена для человека и для выгрузки.
                "id": n.line.name,
                "label": n.line.label,
                "value": n.line.value,
                "children": [by_index[i].name for i in n.children],
                "children_sum": total([by_index[i] for i in n.children]),
                "arithmetic_ok": (
                    n.line.value is not None
                    and abs(total([by_index[i] for i in n.children]) - n.line.value) < 1
                ),
            }
            for n in nodes if n.is_overall
        ]
        return {
            "sum_all": total(self.lines),
            "sum_without_duplicates": total(unique),
            "sum_internals": total(inside),
            "overall_segments": overall,
        }

    def summary(self) -> dict:
        return {
            "page": self.page_number,
            "segments": len(self.lines),
            "pointers": sum(1 for l in self.lines if not l.measured),
            "labelled": sum(1 for l in self.lines if l.value is not None),
            "rotated": sum(1 for l in self.lines if l.angle_rotated),
            "numbers": len(self.numbers),
            "rejected_numbers": len(self.rejected_numbers),
            "orphan_numbers": [n.text for n in self.orphan_numbers],
            "width_used": self.width_used,
            "sum_values": sum(self.values),
            "max_length_pt": round(max((l.length_pt for l in self.lines), default=0.0), 2),
        }


def _spans_over(big: DimLine, small: DimLine) -> bool:
    """Проходит ли ``small`` рядом с ``big``, целиком внутри его проекции.

    Три условия, и все три нужны:
      * прямые **параллельны** — иначе это пересечение двух разных размеров
        на изометрии, а не звено цепочки;
      * лежат на **разных** прямых (сдвиг больше нуля) — звенья одной
        цепочки идут в линию, и их сдвиг равен нулю;
      * проекция ``small`` **внутри** ``big`` — звено не выходит за общий
        размер, иначе это соседний, не входящий в сумму.
    """
    length = dist(big.a, big.b)
    if length < 1e-6 or small.length_pt > length:
        return False
    ux = (big.b[0] - big.a[0]) / length
    uy = (big.b[1] - big.a[1]) / length
    projections: list[float] = []
    offsets: list[float] = []
    for point in (small.a, small.b):
        dx = point[0] - big.a[0]
        dy = point[1] - big.a[1]
        projections.append(dx * ux + dy * uy)
        offsets.append(-dx * uy + dy * ux)
    if abs(offsets[0] - offsets[1]) > 0.8:  # не параллельны
        return False
    if abs(sum(offsets) / 2.0) < 0.5:       # лежат на одной прямой
        return False
    lo, hi = min(projections), max(projections)
    return lo >= -1.0 and hi <= length + 1.0


# ---------------------------------------------------------------------------
# Геометрия
# ---------------------------------------------------------------------------


def _path_segments(path: dict) -> Iterator[tuple[Pt, Pt]]:
    """Сегменты одного пути: и прямые линии, и стороны прямоугольника."""
    for item in path["items"]:
        kind = item[0]
        if kind == "l":
            a, b = item[1], item[2]
            yield (a.x, a.y), (b.x, b.y)
        elif kind == "re":
            r = item[1]
            corners = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
            for i in range(4):
                yield corners[i], corners[(i + 1) % 4]
        # кривые Безье и прочее пропускаем: на чертёже их роль не размерная


def _signed_area(segments: Sequence[tuple[Pt, Pt]]) -> float:
    return 0.5 * abs(
        sum(a[0] * b[1] - b[0] * a[1] for a, b in segments)
    )


def dist(a: Pt, b: Pt) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def dist_point_segment(point: Pt, a: Pt, b: Pt) -> float:
    """Расстояние от точки до отрезка (не до бесконечной прямой).

    Именно до отрезка: подпись размера стоит сбоку от середины, и проекция
    на прямую ещё не гарантирует, что подпись относится к этому размеру,
    а не к соседнему.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return dist(point, a)
    t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy))


def _on_segment_fraction(point: Pt, a: Pt, b: Pt) -> float:
    """Доля пути от ``a`` до ``b``, в которую проектируется точка."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return 0.0
    t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_sq
    return max(0.0, min(1.0, t))


def _arrow_tips(
    drawings: Sequence[dict], params: Thresholds
) -> list[Pt]:
    """Острия залитых треугольников — это наконечники стрелок.

    Берём именно вершину, а не центр: у наконечника стрелки расстояния от
    центра до вершины и до крыльев разные, а для отбора нужно попадание
    ровно в конец отрезка.
    """
    tips: list[Pt] = []
    for path in drawings:
        if not path.get("fill"):
            continue
        segments = list(_path_segments(path))
        if len(segments) != 3:
            continue
        if _signed_area(segments) > params.arrow_max_area:
            continue
        corners = {s[0] for s in segments} | {s[1] for s in segments}
        if len(corners) != 3:
            continue
        tip = max(
            corners,
            key=lambda q: sum(dist(q, other) for other in corners if other != q),
        )
        tips.append(tip)
    return tips


def _stroked_segments(
    drawings: Sequence[dict], params: Thresholds
) -> list[tuple[Pt, Pt, float]]:
    """Обводочные сегменты заданной толщины (без заливок)."""
    out: list[tuple[Pt, Pt, float]] = []
    for path in drawings:
        if path.get("fill"):
            continue
        width = float(path.get("width") or 0.0)
        if not params.close_width(width):
            continue
        out.extend((a, b, width) for a, b in _path_segments(path))
    return out


def _tip_at(point: Pt, tips: Iterable[Pt], gap: float) -> bool:
    return any(dist(point, tip) <= gap for tip in tips)


def _has_arrows_at_both_ends(
    a: Pt, b: Pt, tips: Sequence[Pt], gap: float
) -> bool:
    """Стрелка должна быть у обоих концов и с разных сторон.

    Проверка «с разных сторон» отсекает случай, когда оба конца отрезка
    случайно попали в один и тот же наконечник: у выносной линии со
    стрелкой на одном конце оба конца иногда стоят рядом с треугольником.
    """
    if not _tip_at(a, tips, gap) or not _tip_at(b, tips, gap):
        return False
    return any(
        dist(a, tip) <= gap and dist(b, tip) > gap for tip in tips
    ) and any(
        dist(b, tip) <= gap and dist(a, tip) > gap for tip in tips
    )


def _components(segments: Sequence[tuple[Pt, Pt]]) -> list[list[tuple[Pt, Pt]]]:
    """Разбить сегменты на связные по концам группы.

    Нужен для поиска замкнутых контуров: на этих листах позиционный знак
    нарисован одним путём вместе с выноской и её наконечником, и квадрат
    виден только внутри него. По концам отрезков такая петля отделяется от
    хвоста, а по ``path["items"]`` — нет.
    """
    adjacency: dict[Pt, list[int]] = {}
    for i, (a, b) in enumerate(segments):
        adjacency.setdefault(a, []).append(i)
        adjacency.setdefault(b, []).append(i)
    seen_points: set[Pt] = set()
    seen_edges: set[int] = set()
    out: list[list[tuple[Pt, Pt]]] = []
    for start in adjacency:
        if start in seen_points:
            continue
        stack = [start]
        seen_points.add(start)
        group: list[tuple[Pt, Pt]] = []
        while stack:
            point = stack.pop()
            for i in adjacency[point]:
                edge = segments[i]
                if i not in seen_edges:
                    seen_edges.add(i)
                    group.append(edge)
                other = edge[1] if edge[0] == point else edge[0]
                if other not in seen_points:
                    seen_points.add(other)
                    stack.append(other)
        out.append(group)
    return out


def _is_closed_loop(segments: Sequence[tuple[Pt, Pt]]) -> bool:
    """Петля ли из отрезков: у каждой вершины ровно два конца.

    Отрезок не петля, у него две вершины степени один; ломаная с хвостом —
    тоже. А вот квадрат позиционного знака, круг марки прибора и залитый
    треугольник-наконечник — все петли, и отличать их придётся по размеру.
    """
    degree: dict[Pt, int] = {}
    for a, b in segments:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    return len(segments) >= 3 and all(v == 2 for v in degree.values())


def _bounds(segments: Sequence[tuple[Pt, Pt]]) -> pymupdf.Rect:
    xs = [p[0] for s in segments for p in s]
    ys = [p[1] for s in segments for p in s]
    return pymupdf.Rect(min(xs), min(ys), max(xs), max(ys))


def _convex_hull(points: Sequence[Pt]) -> list[Pt]:
    """Выпуклая оболочка (монотонный метод Эндрю)."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def half(seq: Sequence[Pt]) -> list[Pt]:
        chain: list[Pt] = []
        for q in seq:
            while len(chain) >= 2:
                (ax, ay), (bx, by) = chain[-2], chain[-1]
                cross = (bx - ax) * (q[1] - ay) - (by - ay) * (q[0] - ax)
                if cross > 1e-9:
                    chain.pop()
                else:
                    break
            chain.append(q)
        return chain

    return half(pts)[:-1] + half(reversed(pts))[:-1]


def _polygon_area(polygon: Sequence[Pt]) -> float:
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % len(polygon)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def _is_ring(segments: Sequence[tuple[Pt, Pt]], min_fill: float) -> bool:
    """Кольцо ли это: путь, у которого нет ни одной точки внутри оболочки.

    Признак нужен для кругов и **пунктирных** кругов: у них десятки
    несвязанных дуг, ни одна не замкнута, и по связным компонентам они не
    находятся. Но у кольца все точки лежат на выпуклой оболочке — это
    отличает круг от ломаной с хвостом (у «стрелки направления» и её
    выноски часть точек уходит внутрь) и от оси трубы (у неё оболочка
    вырождена в отрезок, и заполнение рамки равно нулю).

    Треугольник-наконечник под правило формально тоже подходит — заполнение
    ровно 0.5, — но его порог не проходит, а число туда и не влезает.
    """
    points = [p for s in segments for p in s]
    hull = _convex_hull(points)
    if len(hull) < 3:
        return False
    on_hull = set(hull)
    for point in points:
        if point not in on_hull and _distance_to_polygon(point, hull) > 0.35:
            return False  # точка ушла внутрь оболочки — это фигура с хвостом
    rect = _bounds(segments)
    return _polygon_area(hull) >= min_fill * rect.width * rect.height


def _distance_to_polygon(point: Pt, polygon: Sequence[Pt]) -> float:
    """Расстояние от точки до границы многоугольника."""
    return min(
        dist_point_segment(point, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    )


def _contours(drawings: Sequence[dict], params: Thresholds) -> list[Contour]:
    """Замкнутые контуры листа: рамки позиционных знаков и круги марок.

    Два независимых признака, потому что фигуры нарисованы по-разному:
    квадрат — одной петлёй из четырёх отрезков (часто вперемешку с выноской
    в одном пути), круг — десятками дуг, и ни одна из них не замкнута.
    """
    out: list[Contour] = []

    def add(segments: Sequence[tuple[Pt, Pt]]) -> None:
        rect = _bounds(segments)
        if not (
            params.frame_min_side <= rect.width <= params.frame_max_side
            and params.frame_min_side <= rect.height <= params.frame_max_side
        ):
            return
        out.append(Contour(rect=rect))

    for path in drawings:
        if path.get("fill"):
            continue  # заливка — это наконечник стрелки, а не рамка
        segments = list(_path_segments(path))
        if not segments:
            continue
        for group in _components(segments):
            if _is_closed_loop(group):
                add(group)
        if (
            len(segments) >= params.ring_min_segments
            and _is_ring(segments, params.ring_min_fill)
        ):
            add(segments)
    return out


# ---------------------------------------------------------------------------
# Числа: отсев «не размер»
# ---------------------------------------------------------------------------


def _text_rows(page: pymupdf.Page) -> list[list[tuple[str, Box]]]:
    """Слова листа, разложенные по строкам текстового слоя.

    ``get_text("words")`` вместо ``rawdict`` — сознательно: он уже отдаёт
    номер строки, а на изометрии одна размерная цепочка («24100») может
    лежать в одном спа́не вместе с соседней подписью. Группировка по строке
    — это и есть проверка «число стоит само по себе».
    """
    rows: dict[tuple[int, int], list[tuple[str, Box]]] = {}
    for x0, y0, x1, y1, word, block, line, _ in page.get_text("words"):
        rows.setdefault((block, line), []).append((word, (x0, y0, x1, y1)))
    return [sorted(items, key=lambda it: it[1][0]) for items in rows.values()]


def _alone_in_row(box: Box, row: Sequence[tuple[str, Box]], gap: float) -> bool:
    """Нет ли рядом, на той же строке, других слов.

    Подпись размера начертана одна. Всё остальное числовое на листе —
    часть чужой подписи: ``X 267000`` и ``Z+ 14850`` (координата привязки),
    ``50 mm PE`` (марка трубы), ``Лист 2 из 10`` (штамп), ``LT 4116``
    (две строки внутри круга — их ловит уже проверка контура).
    """
    left, right = box[0], box[2]
    for _, other in row:
        if other == box:
            continue
        if other[2] < left and left - other[2] < gap:
            return False
        if other[0] > right and other[0] - right < gap:
            return False
        if other[0] <= left and other[2] >= right:
            return False  # соседнее слово шире — перекрытие, строка общая
    return True


def _numbers(
    page: pymupdf.Page, params: Thresholds, contours: Sequence[Contour]
) -> tuple[list[NumberMark], dict, list[str]]:
    """Числа листа, которые могут быть размерами.

    Порядок чтения как на чертеже: сверху вниз, в строке слева направо.
    Номера в ``NumberMark.index`` идут в том же порядке, что и строки
    ``segments.json``, — сверять выгрузку с картинкой удобнее по порядку,
    а не по координатам.

    Возвращается и разбор отсева: молча выкинутые числа — это либо
    регрессия, либо чертёж не того формата, и по цифре счётчика одно
    отличить от другого нельзя. Сами отсеянные значения возвращаются
    отдельно: в промт модели уходит именно список «что здесь нарисовано, но
    размером не является», потому что по картинке «9» в квадрате от «9» как
    размера не отличить, и модель лишнее берёт в расчёт.
    """
    marks: list[NumberMark] = []
    seen = 0
    rejected = {"in_row": 0, "in_contour": 0, "too_small": 0}
    dropped: list[str] = []
    for row in _text_rows(page):
        for text, box in row:
            if not _NUM_RE.match(text):
                continue
            seen += 1
            if (box[3] - box[1]) < params.min_label_height:
                rejected["too_small"] += 1
                continue
            if not _alone_in_row(box, row, params.line_gap):
                rejected["in_row"] += 1
                dropped.append(text)
                continue
            if any(c.wraps(box, params.frame_overshoot, params.frame_margin)
                   for c in contours):
                rejected["in_contour"] += 1
                dropped.append(text)
                continue
            marks.append(NumberMark(text=text, box=box, index=len(marks) + 1))
    return marks, {"numeric_words": seen, **rejected}, dropped


# ---------------------------------------------------------------------------
# Привязка чисел к геометрии
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Leader:
    """Тонкая линия ровно с одним наконечником.

    ``tip`` — остриё стрелки (куда указывает), ``tail`` — свободный конец.
    Если ``tail`` упирается в число, а ``tip`` не стоит на отрезке, это
    размер-зазор: ``pointer``. Если ``tip`` стоит на середине отрезка, это
    подпись, отведённая от него в сторону, и размером остаётся отрезок.
    """

    tip: Pt
    tail: Pt
    width: float

    @property
    def length_pt(self) -> float:
        return dist(self.tip, self.tail)

    @property
    def angle(self) -> float:
        return math.degrees(math.atan2(self.tail[1] - self.tip[1],
                                       self.tail[0] - self.tip[0]))


def _spans(
    segments: Sequence[tuple[Pt, Pt, float]],
    tips: Sequence[Pt],
    params: Thresholds,
) -> list[tuple[Pt, Pt, float]]:
    """Отрезки с наконечниками на обоих концах — их длина измерена."""
    out: list[tuple[Pt, Pt, float]] = []
    for a, b, width in segments:
        if dist(a, b) < params.min_length:
            continue
        if not _has_arrows_at_both_ends(a, b, tips, params.arrow_gap):
            continue
        out.append((a, b, width))
    return out


def _leaders(
    segments: Sequence[tuple[Pt, Pt, float]],
    tips: Sequence[Pt],
    params: Thresholds,
) -> list[_Leader]:
    """Выноски: тонкие линии, у которых ровно один конец свободен.

    Свободный — значит не упирается в остриё наконечника. У настоящей
    размерной линии оба конца заняты треугольниками, а у выноски их обычно
    один (со стороны примечания) или ни одного. Проверка на «ровно один»
    вместо «хотя бы один» отсекает сами размерные линии: они короче
    ``leader_min_length`` и сюда не попадают, но их удобно отсечь и явно.
    """
    out: list[_Leader] = []
    for a, b, width in segments:
        if dist(a, b) < params.leader_min_length:
            continue
        at_a = any(dist(a, t) <= params.arrow_gap for t in tips)
        at_b = any(dist(b, t) <= params.arrow_gap for t in tips)
        if at_a == at_b:
            continue
        tip, tail = (a, b) if at_a else (b, a)
        out.append(_Leader(tip=tip, tail=tail, width=width))
    return out


def _anchored_leaders(
    leaders: Sequence[_Leader], spans: Sequence[tuple[Pt, Pt, float]], params: Thresholds
) -> dict[int, int]:
    """Выноски, указывающие на отрезок: ``индекс выноски -> индекс отрезка``.

    Признак — наконечник стоит на **середине** отрезка, а не на чём-то ещё.
    Всё остальное (выноска к позиционному знаку, к рамке «О6», к прибору)
    сюда не попадает: там наконечник упирается в объект, у которого
    середины отрезка нет.
    """
    out: dict[int, int] = {}
    for i, leader in enumerate(leaders):
        for j, (a, b, _) in enumerate(spans):
            if dist(leader.tip, _mid(a, b)) <= params.leader_anchor_gap:
                out[i] = j
                break
    return out


def _mid(a: Pt, b: Pt) -> Pt:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _tail_to_box(point: Pt, box: Box) -> float:
    """Расстояние от точки до рамки подписи (нулевое, если точка внутри)."""
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


def _attach(
    numbers: Sequence[NumberMark],
    spans: Sequence[tuple[Pt, Pt, float]],
    leaders: Sequence[_Leader],
    anchored: dict[int, int],
    params: Thresholds,
) -> tuple[list[DimLine], list[NumberMark]]:
    """Раздать размеры и числа один-к-одному.

    Порядок не произвольный:

    1. **указатели.** Число, к которому примыкает свободный конец выноски,
       не указывающей на отрезок, — это размер-зазор. Такая привязка
       локальная и однозначная, её невозможно перепутать, поэтому она
       раздаётся первой: иначе отрезок рядом успеет забрать то же число.
    2. **отрезки.** Каждому отрезку достаётся ближайшее из оставшихся чисел.
    3. **подписи, отведённые выноской.** Отрезок без числа рядом получает
       число со своей выноски — так начерчены короткие размеры на листе 9.

    Число достаётся ровно одному размеру: иначе «6000» на листе 8 подписало
    бы сразу четыре отрезка, а сумма разъехалась бы вчетверо.
    """
    lines: list[DimLine] = []
    used_numbers: set[int] = set()
    used_leaders: set[int] = set()

    # 1. размеры-зазоры, начертанные выноской
    for i, number in enumerate(numbers):
        best: tuple[float, int] | None = None
        for j, leader in enumerate(leaders):
            if j in used_leaders or j in anchored:
                continue
            gap = _tail_to_box(leader.tail, number.box)
            if gap > params.leader_label_gap:
                continue
            if best is None or gap < best[0]:
                best = (gap, j)
        if best is None:
            continue
        j = best[1]
        used_numbers.add(i)
        used_leaders.add(j)
        lines.append(
            _make_line(number, leaders[j].tip, leaders[j].tail, leaders[j].width,
                       "pointer")
        )

    # 2. размеры, у которых есть отрезок с наконечниками
    for j, (a, b, width) in enumerate(spans):
        number, _ = _nearest_to_span(numbers, used_numbers, a, b, params)
        if number is not None:
            used_numbers.add(number)
            lines.append(_make_line(numbers[number], a, b, width, "span"))
            continue
        # 3. число отведено в сторону — идём по выноске от середины отрезка
        for li, target in anchored.items():
            if target != j or li in used_leaders:
                continue
            tail = leaders[li].tail
            k = _nearest_to_point(
                numbers, used_numbers, tail, params.leader_label_gap
            )
            if k is None:
                continue
            used_numbers.add(k)
            used_leaders.add(li)
            lines.append(
                _make_line(numbers[k], a, b, width, "span", leader_end=tail)
            )
            break

    orphans = [n for i, n in enumerate(numbers) if i not in used_numbers]
    return lines, orphans


def _make_line(
    number: NumberMark,
    a: Pt,
    b: Pt,
    width: float,
    kind: str,
    leader_end: Pt | None = None,
) -> DimLine:
    return DimLine(
        page_number=0,  # проставляется в detect_page
        a=a,
        b=b,
        length_pt=dist(a, b),
        width=width,
        angle=math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])),
        label=number.text,
        label_bbox=number.box,
        kind=kind,
        leader_end=leader_end,
    )


def _nearest_to_span(
    numbers: Sequence[NumberMark],
    used: set[int],
    a: Pt,
    b: Pt,
    params: Thresholds,
) -> tuple[int | None, float]:
    """Ближайшее свободное число к отрезку, по «удобству» подписи.

    Расстояние берётся до **отрезка**, а не до середины: подпись длинного
    размера CAD сдвигает к концу, и требовать её ровно посередине значило бы
    терять половину размеров. Штраф за уход от середины (0.35 длины) нужен
    для обратного: на изометрии размеры идут параллельными рядами, и число
    соседнего ряда отстоит от прямой так же, как своё — без штрафа длинный
    отрезок перехватил бы подпись соседа.
    """
    length = dist(a, b)
    best: tuple[float, int] | None = None
    for i, number in enumerate(numbers):
        if i in used:
            continue
        d = dist_point_segment(number.center, a, b)
        if d > params.label_gap:
            continue
        t = _on_segment_fraction(number.center, a, b)
        score = d + 0.35 * abs(t - 0.5) * length
        if best is None or score < best[0]:
            best = (score, i)
    return (best[1], best[0]) if best is not None else (None, 0.0)


def _nearest_to_point(
    numbers: Sequence[NumberMark], used: set[int], point: Pt, gap: float
) -> int | None:
    """Ближайшее свободное число к точке — концу выноски."""
    best: tuple[float, int] | None = None
    for i, number in enumerate(numbers):
        if i in used:
            continue
        d = _tail_to_box(point, number.box)
        if d > gap:
            continue
        if best is None or d < best[0]:
            best = (d, i)
    return best[1] if best is not None else None


# ---------------------------------------------------------------------------
# Поиск
# ---------------------------------------------------------------------------


def _any_width(drawings: Sequence[dict], params: Thresholds) -> list[tuple[Pt, Pt, float]]:
    """Обводочные сегменты **любой** толщины — для подбора толщины размеров.

    Отдельная копия с бесконечным допуском: :func:`_score_width` ещё не знает,
    какая толщина искомой, и не может искать её самой толщиной.
    """
    loose = Thresholds(**{**params.__dict__, "width": 0.0, "width_tol": 1e9})
    return _stroked_segments(drawings, loose)


def _score_width(drawings: Sequence[dict], params: Thresholds) -> float:
    """Толщина линий с наконечниками на обоих концах — в пунктах листа.

    Пороги, по которым идёт отбор, к этому моменту уже приведены к листу, а
    значит отбор работает на настоящих размерах наконечников: тот же чертёж,
    вписанный вдвое крупнее, даст здесь 0.96 pt вместо 0.48 pt, и оба раза
    ответ будет верным.
    """
    tips = _arrow_tips(drawings, params)
    buckets: dict[float, int] = {}
    for a, b, width in _any_width(drawings, params):
        if dist(a, b) < params.min_length:
            continue
        if not _has_arrows_at_both_ends(a, b, tips, params.arrow_gap):
            continue
        key = round(width, 2)
        buckets[key] = buckets.get(key, 0) + 1
    if not buckets:
        return params.width
    return max(sorted(buckets), key=lambda k: buckets[k])


def detect_page(
    page: pymupdf.Page, params: DimParams | None = None
) -> DimResult:
    """Размеры одного листа.

    Читается как задача: сначала собираются **числа**, потом отсеиваются те,
    что размером не являются, потом к каждому оставшемуся ищется то, рядом с
    чем оно нарисовано.

    Пороги заданы для эталонного листа, а лист бывает крупнее или мельче, чем
    он, поэтому первым делом измеряется масштаб и все пороги приводятся к
    нему (:func:`_rescaled`). Дальше по коду идут уже числа этого листа.
    """
    params = params or DimParams()
    drawings = page.get_drawings()
    scale, scale_source = sheet_scale(page, params)

    # Толщина искомых линий: либо задана, либо подобрана по наконечникам.
    # Подбор идёт при порогах, уже приведённых к листу, и возвращает
    # величину в эталонных пунктах, поэтому масштаб применяется один раз.
    if params.width is None:
        probe = _rescaled(params, scale, DEFAULT_WIDTH)  # толщина тут не нужна
        width_pt = _score_width(drawings, probe)
    else:
        width_pt = params.width * scale
    limits = _rescaled(params, scale, width_pt)

    contours = _contours(drawings, limits)
    numbers, rejected, dropped = _numbers(page, limits, contours)
    tips = _arrow_tips(drawings, limits)
    stroked = _stroked_segments(drawings, limits)
    spans = _spans(stroked, tips, limits)
    leaders = _leaders(stroked, tips, limits)
    anchored = _anchored_leaders(leaders, spans, limits)

    lines, orphans = _attach(numbers, spans, leaders, anchored, limits)
    page_number = page.number + 1
    lines = [
        DimLine(**{**line.__dict__, "page_number": page_number})
        for line in lines
    ]
    lines.sort(key=lambda l: (round(l.midpoint[1], 1), l.midpoint[0]))
    lines = [
        DimLine(**{**line.__dict__, "index": i})
        for i, line in enumerate(lines, start=1)
    ]
    # Нумерация присваивается по порядку чтения сверху вниз, а рисунок
    # нумерует по ней же — иначе «номер 3» на картинке и в segments.json
    # означали бы разные размеры.

    result = DimResult(
        page_number=page_number,
        lines=lines,
        numbers=list(numbers),
        orphan_numbers=orphans,
        rejected_numbers=dropped,
        width_used=width_pt,
        contours=tuple(contours),
        limits=limits,
    )
    result.stats = {
        "paths": len(drawings),
        "arrows": len(tips),
        "contours": len(contours),
        "spans": len(spans),
        "leaders": len(leaders),
        "leaders_anchored": len(anchored),
        "width_guessed": params.width is None,
        # Масштаб, к которому приведены пороги, и откуда он взят. Без этого
        # в отчёте нельзя понять, почему на листе другого формата получились
        # другие абсолютные зазоры.
        "scale": round(scale, 4),
        "scale_source": scale_source,
        "numbers_rejected": rejected,
        "candidates_by_width": len(
            [1 for a, b, _ in stroked if dist(a, b) >= limits.min_length]
        ),
    }
    return result


def detect_pdf(
    pdf_path: str,
    pages: Sequence[int] | None = None,
    params: DimParams | None = None,
) -> list[DimResult]:
    """Размеры листов PDF. ``pages`` — 1-based номера, ``None`` — все."""
    with pymupdf.open(pdf_path) as document:
        wanted = list(range(document.page_count)) if not pages else [p - 1 for p in pages]
        return [detect_page(document.load_page(i), params) for i in wanted]


# ---------------------------------------------------------------------------
# Слой для конвейера: тот же результат, но в пикселях картинки
# ---------------------------------------------------------------------------

# Токен в system.md, на место которого подставляется блок размеров.
DIMENSIONS_TOKEN = "{{DIMENSIONS}}"

# Как подписан вид линии в блоке для модели. Слова, а не коды: блок читает
# человек при разборе промта, и «выноска» понятнее, чем ``pointer``.
_KIND_WORDS = {"span": "отрезок", "pointer": "выноска"}


@dataclass(frozen=True)
class DimensionMark:
    """Размер, готовый к отправке модели.

    ``bbox`` — рамка подписи размера в **пикселях той же картинки**, что ушла
    в модель, поэтому рамку можно показывать в разметке и накладывать на
    изображение без всякой пересчётки. ``line_bbox`` — рамка самой линии
    (по концам отрезка или выноски): она нужна, чтобы различать размеры,
    когда подписи соседних участков стоят рядом.
    """

    value: float
    label: str
    kind: str
    bbox: tuple[float, float, float, float]
    line_bbox: tuple[float, float, float, float]
    line_id: str
    # Направление размерной линии: градусы, как их посчитал разбор, и номер
    # оси изометрической сетки (0–5, по 30° начиная с горизонтали).
    #
    # Зачем это в промте. Правило «вложенным может быть только размер вдоль
    # той же оси» проверяется лишь сравнением углов, и на словах модели не
    # работает: на листе 5 размер 34 мм под -29.5° модель объявляла вложенным
    # то в 3505 (+29.9°), то в 1600 (+29.7°) — то есть исключала его, каждый
    # раз подбирая другое ложное основание. Угол по картинке глазом читается
    # ненадёжно, а передать его можно: он уже посчитан, стоит ноль усилий и
    # снимает целый класс правдоподобных, но неверных объяснений.
    angle: float = 0.0
    axis: int = 0
    # Концы размерной линии в пикселях картинки. Нужны, чтобы подсветить
    # саму линию на листе (см. overlay.py): рамки ``line_bbox`` для повёрнутого
    # на 30° отрезка заметно шире самого отрезка, и рисовать по ней — значит
    # закрасить лишнее и закрыть геометрию, ради которой всё затевается.
    line_start: tuple[float, float] = (0.0, 0.0)
    line_end: tuple[float, float] = (0.0, 0.0)

    @property
    def kind_word(self) -> str:
        return _KIND_WORDS.get(self.kind, self.kind)

    def as_line(self) -> str:
        """Строка для промта: индекс, значение, рамки, вид, ось.

        Индекс обязателен, а не украшение: на листе 9 есть два размера по 123,
        на листе 10 — по два 220, 244, 341 и 134. Списком значений их не
        различить, и модель, выписывающая числа, вынуждена угадывать, какой
        из двух «341» имеется в виду. Ссылка по индексу P7 снимает вопрос.

        Ось дописывается последней, и она единственное поле в строке, которое
        модель не может вывести из картинки надёжно: направление линии глазом
        не измерить, а вложенность определяется именно им.
        """
        return (
            f"{self.line_id} \"{self.label}\" {_box(self.bbox)} "
            f"линия {_box(self.line_bbox)} {self.kind_word} "
            f"ось {self.axis} ({self.angle:.0f}°)"
        )


def _box(box: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.0f}" for v in box) + "]"


# Шаг сетки изометрии. Изометрическая сетка состоит из шести направлений по
# 30° начиная с горизонтали; на трубопроводах горизонталь почти не встречается,
# но исключать её из шага нельзя — иначе 0° и 180° разъедутся.
AXIS_STEP = 30.0


def axis_of(angle: float) -> int:
    """Номер оси изометрической сетки, ближайшей к углу (0–5).

    Нужен, чтобы сравнивать направления целыми числами. Углы с листа приходят
    с плавающей точкой и со знаком: одна и та же диагональ читается как +29.9
    и -29.8 градусов, и наивное сравнение чисел считает их разными. Приведение
    к номеру оси эту разницу снимает, и две линии на одной диагонали получают
    один номер независимо от того, с какого конца посчитан угол.
    """
    folded = angle % 180.0
    return int(round(folded / AXIS_STEP)) % 6


def marks_in_pixels(
    result: DimResult, to_pixels
) -> tuple[DimensionMark, ...]:
    """Перевести найденные размеры в пиксели картинки.

    ``to_pixels(box) -> (x0, y0, x1, y1)`` — функция пересчёта рамки из
    пунктов PDF в пиксели; её задаёт вызывающий, потому что матрица рендера
    и сдвиг обрезки известны только там (:mod:`.extract`).

    Порядок — как на листе: сверху вниз, а при равной высоте слева направо.
    Модель читает лист сверху вниз, и разнобой в порядке заставляет её
    перечитывать блок целиком.

    Метки перенумеровываются **по порядку показа**: собственные номера разбора
    идут в порядке создания линий, а не чтения, и в блоке получалось
    ``P1, P2, P4, P3, P5, P6, P8, P9, P7`` — модель, читающая сверху вниз,
    вынуждена держать в голове эту перестановку и ошибается в атрибуции.
    Здесь важно только, чтобы метка была стабильной и шла подряд, поэтому
    ``line.name`` разбора не используется.
    """
    marks = [
        DimensionMark(
            value=float(line.value or 0.0),
            label=line.label,
            kind=line.kind,
            bbox=tuple(to_pixels(line.label_bbox or (0, 0, 0, 0))),
            line_bbox=tuple(
                to_pixels(
                    (
                        min(line.a[0], line.b[0]),
                        min(line.a[1], line.b[1]),
                        max(line.a[0], line.b[0]),
                        max(line.a[1], line.b[1]),
                    )
                )
            ),
            line_id=line.name,
            angle=float(line.angle),
            axis=axis_of(line.angle),
            # to_pixels отдаёт рамку из четырёх чисел, а конец точки — пара,
            # поэтому берём левый верхний угол вырожденной рамки.
            line_start=tuple(
                to_pixels((line.a[0], line.a[1], line.a[0], line.a[1]))[:2]
            ),
            line_end=tuple(
                to_pixels((line.b[0], line.b[1], line.b[0], line.b[1]))[:2]
            ),
        )
        for line in result.lines
        if line.value is not None
    ]
    marks.sort(key=lambda m: (round(m.bbox[1], 1), m.bbox[0]))
    return tuple(
        replace(mark, line_id=f"P{index}")
        for index, mark in enumerate(marks, start=1)
    )


def format_dimensions_block(
    marks: Sequence[DimensionMark],
    rejected: Sequence[str] = (),
    *,
    with_rejected: bool = True,
) -> str:
    """Блок для системного промта: что на листе является размером.

    Главное в блоке — не сами числа, а **границы**: список отсеянных чисел
    отвечает на вопрос, который у модели возникает чаще всего и на который
    она отвечает хуже всего всего. По картинке нельзя отличить размер «9» от
    позиционного знака «9» в квадрате, и модель стабильно берёт лишнее в
    расчёт; здесь сказано прямым текстом, что перечисленное — не размер.

    Формат строки повторяет блок текстового слоя (``textmap``), только
    добавлены рамка линии, вид и ось: по рамке подписи размер, который
    подписан через выноску, визуально неотличим от любого числа на листе, а
    ось — единственный признак, по которому вложенность вообще определяется,
    и с картинки он не читается (см. ``DimensionMark.axis``).
    """
    header = [
        "# Размеры, найденные на листе разбором вектора",
        "",
        f"Изображение этого листа показано ниже. Найдено размеров: {len(marks)}.",
        "",
        "Формат: P7 \"значение\" [рамка подписи] линия [рамка линии] вид "
        "линии ось N (градусы).",
        "Рамки — в пикселях того же изображения, координаты как у текстового слоя.",
        "P7 — метка размера. Одинаковые значения встречаются по два и более "
        "раза, и по значению их не различить, поэтому ссылаться на размер "
        "нужно по метке.",
        "Ось — номер направления изометрической сетки от 0 до 5, шаг 30°, "
        "считая с горизонтали; всего шесть осей, и 0° это та же ось, что "
        "180°. Две размерные линии лежат на одной линии трубы только если у "
        "них одна ось. Разные оси — это разные линии трассы, и размер на "
        "одной оси не может быть вложен в размер на другой, как бы они ни "
        "пересекались на рисунке.",
        "",
    ]
    if marks:
        header.extend(m.as_line() for m in marks)
    else:
        header.append(
            "Размеров на листе не найдено. Если на изображении они есть — "
            "это дефект разбора, а не отсутствие размеров: сообщи об этом "
            "в notes и верни пустой список."
        )
    if with_rejected and rejected:
        header += [
            "",
            f"Числа, которые на листе нарисованы, но размерами НЕ являются ({len(rejected)}):",
            "Это позиционные знаки и номера опор в квадратах, координаты "
            "привязки X/Y/Z+, марки труб («50 mm PE»), номера приборов и "
            "страница штампа. Не включай их в расчёт и не выводи их как "
            "участки трассы.",
            "",
            " ".join(rejected),
        ]
    return "\n".join(header)
