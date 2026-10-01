"""Сопоставление подписи с её указателем: обрезки, метки, запрос к модели.

Зачем. Лист не читается, пока подпись не связана со своей линией. Разбор
умеет находить размерные линии, но не умеет сказать, что выноска опорного
знака и подпись рядом с ней — одно и то же, а выносок на листе десятки и
размеров среди них меньше. Пока связь не установлена, вырезать нечего:
снос��и выноску вместе со знаком можно снести размерную линию, а она
отличается от выноски только наконечником на втором конце.

Порядок работы жёсткий, и он не переставляется.

1. **Отбор.** Берутся элементы, не прошедшие фильтр. Размеры в список не
   входят: их вырезать нельзя, и правила фильтра это уже обеспечивают.
2. **Обрезки.** По одному вырезу на элемент, 2–3 выреза на запрос.
3. **Метки внутри выреза.** Указатели подписываются буквами **на самой
   картинке выреза**, а не на листе. Метка на листе в вырез не попадает: у
   выреза свой кадр, и буква, нарисованная в тридцати пунктах от его края,
   модели не видна. Рисовать метку нужно там, где её увидят.
4. **Буквы сквозные в пределах запроса.** Первый вырез — A, B, C, второй —
   D, E. Обнулять буквы нельзя: модель отвечает один раз на несколько
   вырезов, и одинаковые буквы на разных картинках в одном ответе значат
   разные стрелки. Соответствие «буква → где стоит» хранится у нас.
5. **Запрос.** Флеш-модель, ``reasoning: low``, короткий вопрос по одному
   вырезу. Нужен один ответ на один вырез, а не решение задачи.
6. **Удаление — после того, как размечены все.** Порядок тот же, что и в
   задании: сначала разметка целиком, потом вырезание. Обрезка, снятая по
   ходу, показала бы, что состояние листа меняется на глазах у того, кто
   по нему потом ходит.

Модуль ничего не стирает. Он говорит, что и каким способом: подпись — вырезом
из PDF, линию — закрашиванием по пикселям. Стирает :mod:`.redact` и
:mod:`.prune`, и там же проверка, что размер не задет.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Sequence

import pymupdf
from pydantic import BaseModel, Field

from . import classify, config, llm, placing, pointers, raster

# Ширина буквы кеглем и высота строки подложки, доли. Буквы жирного шрифта
# шире среднего, и подложка по этим долям с запасом накрывает всё нарисованное.
LABEL_WIDTH = 0.62
LABEL_HEIGHT = 0.78

# Буквы меток. Двузначные читаются на вырезе плохо, поэтому запас небольшой,
# а лишние метки лучше не показать, чем спрятать за обрезанным краем.
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Что модель отвечает про линию рядом с подписью.
LINE_POINTER = "указатель"      # выноска знака: наконечник один, упирается в трубу
LINE_DIMENSION = "размерная"     # наконечников два, по краям
LINE_NONE = "нет"

SYSTEM = """\
Ты читаешь изометрический чертёж трубопровода. Тебе показывают несколько \
вырезов. В каждом вырезе красным подписана **одна** подпись — её метка стоит \
рядом с ней. Указатели на вырезе тоже подписаны красными буквами.

Твоя задача: для каждой подписи сказать, **чей это указатель** — из \
нарисованных на её вырезе или ничей.

## Что означает указатель

Указатель — это линия, которая **начинается** у подписи и идёт к трубе, \
заканчиваясь наконечником. Так показывают опоры, задвижки, изоляцию, \
штуцары: «РВБЛ», «ОБОГРЕВ», «ШТУРВАЛ», «DN80».

Наконечник — жирный чёрный треугольник на конце линии, он смотрит на трубу.

У **размерной линии** наконечников два, по краям отрезка. Такие линии на \
вырезе тоже бывают, и для них ответ `нет`: они не указатели.

## Как определить, чей указатель

Указатель принадлежит подписи, если он **выходит из рамки или из-под самой \
подписи**: линия начинается прямо там. Проверь начало линии, а не её конец.

Соседние подписи на вырезе — не твои. Указатель, выходящий из подписи «2» или \
«DN25X25», не относится к подписи «<8>», даже если стоит рядом.

Если в кадре несколько указателей, выбери тот, что выходит из подписи. Если \
ни один не выходит из подписи — ответь `нет`.

## Правила ответа

1. Смотри на картинку. Текст подписи не решает ничего: рядом могут стоять \
чужие линии, и по подписи их не отличить.
2. Красную метку подписи и буквы указателей на картинке видно — используй их.
3. Не выдумывай букву, которой нет на этом вырезе.
4. На каждый вырез — ровно один элемент в ответе, в том же порядке.
"""


class Verdict(BaseModel):
    """Ответ модели по **всем** вырезам одного запроса сразу."""

    verdicts: list[Element] = Field(
        description="По одному ответу на каждый вырез вопроса, в том же порядке"
    )


class Element(BaseModel):
    """Ответ по одному вырезу."""

    mark: str = Field(description="Метка подписи из вопроса, копией. Не выдумывай.")
    line: str = Field(
        description=f"Что за линия рядом с подписью: {LINE_POINTER} — выноска "
                    f"знака, наконечник один и упирается в трубу; "
                    f"{LINE_DIMENSION} — размерная линия, наконечников два; "
                    f"{LINE_NONE} — линии рядом нет"
    )
    pointer: str = Field(
        description="Буква указателя из картинки, если он относится к этой "
                    "подписи. Иначе пустая строка."
    )
    role: str = Field(
        description="Что за подпись: размер, координата привязки, номер узла, "
                    "опорный знак, обозначение трубы, штамп, прочее"
    )
    confidence: float = Field(description="Насколько уверен, от 0 до 1")


@dataclass(frozen=True)
class Marked:
    """Один вырез с подписью и подписанными указателями."""

    mark: str                    # "N7" — метка подписи, рисуется красным
    text: str                    # что написано, для подсказки модели
    rect: tuple[float, float, float, float]   # элемент в пунктах PDF
    area: tuple[float, float, float, float]   # вырез в пунктах PDF
    image: bytes                 # PNG выреза с метками
    letters: tuple[str, ...]     # буквы, видимые в этом вырезе


@dataclass
class Batch:
    """Запрос к модели: несколько вырезов и общий алфавит меток."""

    page_number: int
    marks: list[Marked] = field(default_factory=list)
    # Буква → указатель. Сквозной алфавит листа: одна буква — один указатель
    # на весь лист, и позволяет по ответу модели вернуться к геометрии, не
    # полагаясь на память картинки.
    registry: dict[str, pointers.Pointer] = field(default_factory=dict)
    # Сколько букв занято на этом листе.
    used: int = 0

    @property
    def images(self) -> list[bytes]:
        return [mark.image for mark in self.marks]

    def pointer(self, letter: str) -> pointers.Pointer | None:
        return self.registry.get(letter.strip().upper())


@dataclass
class Answer:
    """Разобранный ответ модели по одному вырезу."""

    mark: str
    role: str
    line: str
    pointer: pointers.Pointer | None
    confidence: float
    reason: str = ""

    @property
    def has_pointer(self) -> bool:
        """Надо ли удалять вместе с подписью её указатель.

        Удаляются оба: подпись вырезом из PDF, линия закрашиванием. Разные
        способы, один список — иначе вынос��а останется на листе одна, без
        знака, и станет для модели вторым отрезком.
        """
        return self.pointer is not None


# --- Отбор и обрезки -------------------------------------------------------


def candidates(pdf_page: pymupdf.Page, scan) -> list[classify.Element]:
    """Элементы листа, подлежащие удалению.

    Берутся те, чья роль не размер. Правила фильтра дают этот список по
    построению, и размеры в него не попадают — поэтому указатели у этих
    элементов заведомо не размерные, и гадать о том, размер это или нет,
    второй раз не нужно.
    """
    return [element for element in classify.elements_of(pdf_page)
            if element.role != classify.ROLE_DIMENSION]


def _letter(index: int) -> str:
    if index < len(_ALPHABET):
        return _ALPHABET[index]
    return _ALPHABET[index // len(_ALPHABET) - 1] + _ALPHABET[index % len(_ALPHABET)]


def _draw_labels(
    gray: pymupdf.Pixmap,
    area: pymupdf.Rect,
    labelled: Sequence[tuple[str, pymupdf.Point]],
    focus: pymupdf.Rect | None = None,
    mark: str = "",
    size: float = config.LABEL_SIZE_PT,
) -> bytes:
    """Подписать вырез: буквы указателей и метку самой подписи.

    Вырез собирается заново как отдельная страница того же размера в пунктах
    PDF: так буквы попадают в кадр вместе с чертёжом, а не рядом с ним. Метка
    на листе, снятая после обрезки, в вырез не попадает по построению — у
    выреза своя рамка.

    Метки соединены с объектами тонкой красной линией, и это не украшение.
    Буква, просто стоящая рядом с объектом, не читается: на вырезе подписи «<8>»
    буква «A» оказывалась между тремя линиями, и даже человек не мог сказать,
    к какой она относится. Модель отвечала на такой кадр «указателя нет» с
    уверенностью 0.9 — не потому что её зрение плохое (она называет все надписи
    на вырезе), а потому что принадлежность на картинке не выражена. Линия от
    буквы к наконечнику выражает её прямо.
    """
    document = pymupdf.open()
    try:
        page = document.new_page(width=area.width, height=area.height)
        page.insert_image(page.rect, stream=gray.tobytes("png"))

        for letter, at in labelled:
            tip = _aims.get(letter)
            if tip is not None:
                _lead(page, at, tip, area, size)
            _write(page, letter, pymupdf.Point(at.x - area.x0,
                                               at.y - area.y0), size)

        if focus is not None and mark:
            # Рамка вокруг подписи: показывает, о какой именно надписи вопрос,
            # когда в кадре их несколько.
            box = pymupdf.Rect(focus.x0 - area.x0, focus.y0 - area.y0,
                               focus.x1 - area.x0, focus.y1 - area.y0)
            page.draw_rect(box, color=pointers.INK, width=0.4)
            spot = _free_spot(gray, area, focus, mark, size,
                              config.PRUNE_LINK_DPI / 72.0)
            _lead(page, spot,
                  pymupdf.Point((box.x0 + box.x1) / 2, box.y0), area, size)
            _write(page, mark, pymupdf.Point(spot.x - area.x0,
                                             spot.y - area.y0), size)

        return page.get_pixmap(dpi=config.PRUNE_LINK_DPI,
                               alpha=False).tobytes("png")
    finally:
        document.close()


def _lead(page: pymupdf.Page, at: pymupdf.Point, to: pymupdf.Point,
          area: pymupdf.Rect, size: float) -> None:
    """Тонкая линия от метки к тому, что она помечает.

    Идёт от края подложки метки, а не из её центра: из центра она ныряет под
    белую подложку и не видна сразу за буквой. Толщина минимальная — линия
    должна читаться как связь, а не как часть чертежа.
    """
    start = pymupdf.Point(at.x - area.x0 + size * 0.3, at.y - area.y0)
    end = pymupdf.Point(to.x - area.x0, to.y - area.y0)
    if abs(start.x - end.x) + abs(start.y - end.y) < 2.0:
        return
    page.draw_line(start, end, color=pointers.INK, width=0.35)


def _above(box: pymupdf.Rect, area: pymupdf.Rect) -> pymupdf.Point:
    """Где написать метку подписи — вне поля зрения.

    Заглушка: место ищется в :func:`_free_spot` по карте расстояний, и сюда
    попадает только тогда, когда карта построить не удалось.
    """
    return pymupdf.Point(box.x0 - area.x0, box.y0 - area.y0 - 1.0)


def _free_spot(
    gray: pymupdf.Pixmap,
    area: pymupdf.Rect,
    box: pymupdf.Rect,
    text: str,
    size: float,
    scale: float,
) -> pymupdf.Point:
    """Найти для метки место, где под ней нет чертежа.

    Метка на вырезе обязана быть: без неё модель не знает, к какой из нескольких
    подписей в кадре вопрос. Но подложка метки одноцветна, и линия под ней
    пропала бы бесследно — а на вырезе видно ту самую графику, ради которой он
    и сделан. Поэтому место ищется по карте расстояний, а не задаётся
    смещением от подписи: фиксированные смещения на десяти листах накрывали
    137 822 пикселя чертежа, а поиск по карте — ноль на всех 573 вырезах.
    """
    width = int(len(text) * size * LABEL_WIDTH * scale) + 2
    height = int(size * scale * LABEL_HEIGHT) + 1
    view = raster.as_rgb(gray)
    free = placing.free_map(view[:, :, 0], buffer_px=config.LABEL_BUFFER_PX)
    # Подпись — единственное место, куда метке заходить нельзя ни при каких
    # условиях: без этого она встаёт прямо на неё, и модель видит обрывки
    # цифр вместо текста. Рамка знака для метки чертёж, а сама надпись —
    # цель вопроса, и это разные вещи.
    placing.forbid(free, box, area, scale, pad=1.0)
    anchor = (int((box.x0 - area.x0) * scale), int((box.y0 - area.y0) * scale))
    spot = placing.place(free, anchor, (width, height),
                         search=config.LABEL_SEARCH_PX)
    # Метка не должна выйти за кадр: поиск идёт по радиусу от подписи, а подпись
    # у самого края листа, и место находится за пределами выреза. Зажатая
    # метка наезжает на границу, но хотя бы видна.
    x = min(max(spot.x, 1), max(1, view.shape[1] - width - 1))
    y = min(max(spot.y, 1), max(1, view.shape[0] - height - 1))
    return pymupdf.Point(area.x0 + x / scale, area.y0 + y / scale)


def _write(page: pymupdf.Page, text: str, at: pymupdf.Point,
           size: float = config.LABEL_SIZE_PT) -> None:
    """Написать метку с белой подложкой.

    Подложка обязательна: рядом может стоять подпись размера, и без белого
    прямоугольника буквы сольются с цифрами — модель решит, что перед ней
    «A214», и ответ будет про несуществующий элемент.
    """
    width = size * 0.62 * len(text)
    box = pymupdf.Rect(at.x - 1.0, at.y - size * 0.9,
                       at.x + width + 1.0, at.y + size * 0.35)
    page.draw_rect(box, color=None, fill=(1, 1, 1), width=0)
    page.insert_text((at.x, at.y), text, fontsize=size, fontname="hebo",
                     color=pointers.INK)


def _tail_label_at(tail: tuple[float, float],
                   tip: tuple[float, float]) -> pymupdf.Point:
    """Куда поставить букву: у хвоста и в сторону от линии.

    На самом хвосте буква закрыла бы подпись, ради которой всё и затевалось.
    """
    dx, dy = tail[0] - tip[0], tail[1] - tip[1]
    span = max((dx * dx + dy * dy) ** 0.5, 1.0)
    return pymupdf.Point(tail[0] + dx / span * 3.0, tail[1] + dy / span * 3.0)


def _gap(box: pymupdf.Rect, point: pymupdf.Point) -> float:
    """Расстояние от точки до прямоугольника, ноль внутри."""
    dx = max(box.x0 - point.x, 0.0, point.x - box.x1)
    dy = max(box.y0 - point.y, 0.0, point.y - box.y1)
    return (dx * dx + dy * dy) ** 0.5


def in_view(
    area: pymupdf.Rect,
    pointers_found: Sequence[pointers.Pointer],
) -> tuple[pointers.Pointer, ...]:
    """Указатели, у которых в кадр попали **и хвост, и наконечник**.

    Оба конца обязательны. По одному хвосту указатель помечался, когда стрелка
    уходила за край выреза: соединитель от буквы тогда тянулся за пределы кадра
    и обрывался, то есть прямо на картинке показывал, что это не тот наконечник,
    о котором спрашивают. Модель и человек читали такую метку одинаково — не
    понимая ничего.

    Расстояние до подписи как признак принадлежности тоже не годится: на листе
    4 у подписи «1» в квадрате ближайшим хвостом оказывался конец выноски,
    ведущей к «2 / DN25X25», — по расстоянию это «её» стрелка, а по смыслу
    чужая. Различить их может только модель, глядя на картинку, и выбирать ей
    не из чего, если показать одну метку.
    """
    return tuple(
        pointer for pointer in pointers_found
        if area.contains(pymupdf.Point(*pointer.tail))
        and area.contains(pymupdf.Point(*pointer.tip))
    )


def make_batch(
    pdf_page: pymupdf.Page,
    elements: Sequence[classify.Element],
    pointers_found: Sequence[pointers.Pointer],
    registry: dict[str, pointers.Pointer],
    size: int = config.PRUNE_BATCH_SIZE,
) -> Batch:
    """Обрезки элементов с подписанными указателями, 2–3 на запрос.

    ``registry`` — сквозной алфавит листа: буква → указатель. Он передаётся
    извне и пополняется на месте, потому что буквы не сбрасываются между
    запросами: «A» на первой картинке и «A» на третьей в одном ответе модели
    должны означать одну и ту же стрелку. Собственный счётчик в функции дал бы
    на листе 4 двадцать пять букв вместо двадцати и переиспользовал бы их —
    а по букве после этого не восстановить, где стояла стрелка.
    """
    batch = Batch(page_number=pdf_page.number + 1)
    scale = config.PRUNE_LINK_DPI / 72.0

    for element in elements[:size]:
        box = pymupdf.Rect(element.rect)
        area = classify.crop_rect(box, pdf_page.rect)
        inside = _clip(pdf_page, area)

        labelled: list[tuple[str, pymupdf.Point]] = []
        aims: dict[str, pymupdf.Point] = {}
        for pointer in in_view(inside, pointers_found):
            # Помечаются все указатели в кадре: выбирать, чья это стрелка,
            # будет модель — по расстоянию до подписи это не решается, см.
            # :func:`in_view`. Буква у указателя одна на весь лист, и в других
            # вырезах она повторяется: по букве после ответа модели надо вернуться
            # к геометрии, а не полагаться на то, что модель помнит картинку.
            letter = _letter_of(pointer, registry)
            labelled.append((letter, _tail_label_at(pointer.tail, pointer.tip)))
            aims[letter] = _aim_of(pointer)

        # Серый вырез нужен дважды: из него строится карта расстояний для
        # метки подписи, и на его фоне она рисуется. Рендер один на вырез.
        gray = _render_gray(pdf_page, inside, scale)
        _aims.clear()
        _aims.update(aims)
        image = _draw_labels(gray, area, labelled, focus=box,
                             mark=element.mark)
        batch.marks.append(Marked(
            mark=element.mark, text=element.text, rect=element.rect,
            area=(inside.x0, inside.y0, inside.x1, inside.y1),
            image=image, letters=tuple(letter for letter, _ in labelled),
        ))

    batch.registry = registry
    batch.used = len(registry)
    return batch


# Буква → точка, к которой ведёт соединительная линия. Заполняется на время
# отрисовки одного выреза. Модуль не хранит состояние между вырезами: буквы
# сквозные на лист, и оставшаяся в модуле карта от предыдущего выреза сделала
# бы соединители чужими.
_aims: dict[str, pymupdf.Point] = {}


def _aim_of(pointer: pointers.Pointer) -> pymupdf.Point:
    """Куда ведёт соединитель от буквы: в наконечник, а не в хвост.

    Буква ставится у хвоста — рядом с подписью, от которой идёт линия, — но
    соединитель должен дотягиваться до наконечника: именно он и есть тот самый
    указатель, о котором спрашиваем. Соединитель к хвосту указал бы на
    подпись, а не на стрелку.
    """
    return pymupdf.Point(*pointer.tip)


def _letter_of(pointer: pointers.Pointer,
               registry: dict[str, pointers.Pointer]) -> str:
    """Буква указателя: уже выдана или следующая свободная.

    Поиск идёт по **имени** указателя, а не по букве: имя на листе уникально,
    буква — нет.
    """
    for letter, known in registry.items():
        if known.name == pointer.name:
            return letter
    letter = _letter(len(registry))
    registry[letter] = pointer
    return letter


def batches_of_page(
    pdf_page: pymupdf.Page,
    scan,
    elements: Sequence[classify.Element],
    pointers_found: Sequence[pointers.Pointer],
    size: int = config.PRUNE_BATCH_SIZE,
) -> list[Batch]:
    """Все запросы листа: обрезки, нарезанные по ``size`` штук.

    Алфавит один на лист и раздаётся по ходу: буква выдаётся при первом
    появлении указателя и дальше повторяется, поэтому порядок нарезки важен
    только для того, чтобы буквы шли подряд.
    """
    registry: dict[str, pointers.Pointer] = {}
    out: list[Batch] = []
    for start in range(0, len(elements), size):
        out.append(make_batch(pdf_page, elements[start:start + size],
                              pointers_found, registry, size))
    return out


def _question(batch: Batch) -> str:
    """Текст запроса: по строке на вырез, с перечислением видимых букв.

    Буквы перечисляются явно, потому что модель видит их на картинке, но не
    обязана их перечислить сама, а ответ без буквы бесполезен: по букве
    находится геометрия стрелки, и без неё сопоставление не восстановить.
    """
    lines = [
        "Ниже перечислены вырезы. Ответь по каждому, в том же порядке.",
        "",
    ]
    for number, mark in enumerate(batch.marks, start=1):
        letters = ", ".join(mark.letters) if mark.letters else "нет"
        lines.append(
            f"Вырез {number}: подпись {mark.mark}, написано «{mark.text}». "
            f"Буквы указателей на этом вырезе: {letters}."
        )
    lines += [
        "",
        "Для каждого выреза верни один элемент в verdicts:",
        "  mark — метка подписи из вопроса;",
        "  pointer — буква того указателя, который выходит из этой подписи, "
        "либо пустая строка, если такого нет;",
        "  line — «указатель», если указатель нашёлся, иначе «нет»;",
        "  role — что за подпись: размер, координата привязки, номер узла, "
        "опорный знак, обозначение трубы, штамп, прочее;",
        "  confidence — насколько уверен, от 0 до 1.",
    ]
    return "\n".join(lines)


def ask(
    batch: Batch,
    model: str | None = None,
    usage=None,
) -> list[Answer]:
    """Спросить модель про один запрос из двух-трёх вырезов.

    Все вырезы идут одним вызовом, а не по одному: буквы указателей сквозные на
    лист, и отдельный запрос на каждый вырез разорвал бы связь между буквой на
    картинке и её геометрией — модель отвечала бы на картинку, где этой буквы
    может не быть.
    """
    urls = [
        "data:image/png;base64," + base64.b64encode(mark.image).decode("ascii")
        for mark in batch.marks
    ]
    verdict = llm.call_structured(
        system=SYSTEM,
        user=_question(batch),
        response_model=Verdict,
        model=model or config.PRUNE_JUDGE_MODEL,
        reasoning_effort=config.PRUNE_JUDGE_REASONING,
        image_data_urls=urls,
        usage=usage,
    )
    return parse(verdict, batch)


def parse(verdict: Verdict, batch: Batch) -> list[Answer]:
    """Разобрать ответ: сопоставить букву с геометрией, отсеять чужое.

    Две проверки обязательны, и обе отвечают на одну и ту же ошибку — модель
    отвечает не про тот вырез, который ей показывали. Метка не совпала: ответ
    не нашёлся, удалять нечего. Буквы нет на картинке: модель выдумала её,
    брать нечего. В обоих случаях подпись уходит на удаление как есть — по
    правилам фильтра, а её указатель остаётся нетронутым.
    """
    asked = {mark.mark: mark for mark in batch.marks}
    out: list[Answer] = []
    for item in verdict.verdicts:
        letter = item.pointer.strip().upper()
        known = asked.get(item.mark.strip())
        pointer = None
        reason = ""

        if not letter:
            reason = "указателя нет"
        elif letter not in known.letters if known else True:
            reason = f"буквы {letter} на вырезе нет"
        else:
            pointer = batch.registry.get(letter)
            if pointer is None:
                reason = f"буквы {letter} нет в реестре листа"

        out.append(Answer(
            mark=item.mark.strip(),
            role=_role(item.role),
            line=_line(item.line),
            pointer=pointer,
            confidence=max(0.0, min(1.0, float(item.confidence))),
            reason=reason,
        ))
    return out


_ROLE_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (classify.ROLE_DIMENSION, ("размер", "длина")),
    (classify.ROLE_COORDINATE, ("координат", "привязк")),
    (classify.ROLE_NODE, ("узел", "позицион")),
    (classify.ROLE_SUPPORT, ("опор", "штуцар", "задвижк")),
    (classify.ROLE_TITLE, ("штамп", "лист ", "чертёж", "чертеж", "комплект")),
    (classify.ROLE_SPEC, ("труб", "dn", "материал", "обозначение")),
)


def _role(text: str) -> str:
    """Роль из свободного ответа модели.

    Модель отвечает своими словами и не обязана повторять формулировки из
    подсказки. Неопознанное считается «прочее» — подпись уйдёт на удаление, и
    это верно: список удаляемого уже собран правилами фильтра, роль от модели
    нужна только для отчёта.
    """
    low = text.strip().lower()
    for role, words in _ROLE_WORDS:
        if any(word in low for word in words):
            return role
    return classify.ROLE_OTHER


def _line(text: str) -> str:
    low = text.strip().lower()
    if "указател" in low or "вынос" in low:
        return LINE_POINTER
    if "размер" in low:
        return LINE_DIMENSION
    return LINE_NONE


def _clip(pdf_page: pymupdf.Page, area: pymupdf.Rect) -> pymupdf.Rect:
    """Обрезок, вписанный в лист: за краем листа выреза нет."""
    return pymupdf.Rect(area) & pymupdf.Rect(pdf_page.rect)


def _render(pdf_page: pymupdf.Page, area: pymupdf.Rect, scale: float) -> bytes:
    """PNG участка листа в заданном разрешении."""
    return pdf_page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                               clip=area, alpha=False).tobytes("png")


def _render_gray(pdf_page: pymupdf.Page, area: pymupdf.Rect,
                 scale: float) -> pymupdf.Pixmap:
    """Растр выреза: нужен и под метку, и для карты расстояний.

    Пиксель возвращается живым, а не байтами: :func:`placing.free_map` читает его
    как массив, и копия на 90 мегабайт на лист тут ни к чему.
    """
    return pdf_page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                               clip=area, alpha=False, colorspace=pymupdf.csGRAY)
