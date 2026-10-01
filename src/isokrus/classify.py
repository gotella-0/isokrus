"""Элементы листа, которые надо понять глазами, а не разбором.

Зачем. Разбор отвечает на вопрос «нарисован ли здесь отрезок с
наконечниками», но не отвечает на вопрос «что это за число» — и именно
второе решает, можно ли его вырезать. На восьмом листе один и тот же
текстовый элемент может быть координатой привязки, номером узла в квадрате,
опорным знаком или размером, и отличаются они только тем, как нарисованы
вокруг них линии.

Что делает модуль. Собирает **все** текстовые элементы, в которых есть
цифры, и для каждого режет вырез из того же снимка листа, что уходит в
модель. Модель смотрит на вырез и на текст элемента и отвечает, что это:
связку «Z+ -> 9695» она получает уже готовой, потому что элемент целиком.
Ответ нужен один на элемент, поэтому модель можно взять дешёвую и без
глубоких размышлений.

Чего модуль не делает: не решает, что элемент размер. Он только готовит
вырезы и разбирает ответ. Вырезать — работа :mod:`.redact`, и там же
проверка, что размер не задет.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import pymupdf

from . import config
from .raster import as_rgb, crop_bytes

# Роли, которые различает модель. Порядок значим: первая подходящая роль и
# есть ответ, ``default`` стоит последним.
ROLE_DIMENSION = "размер"
ROLE_COORDINATE = "координата привязки"
ROLE_NODE = "номер узла в квадрате"
ROLE_SUPPORT = "опорный знак"
ROLE_TITLE = "штамп или номер листа"
ROLE_SPEC = "обозначение трубы или материала"
ROLE_OTHER = "прочее"
ROLES = (ROLE_DIMENSION, ROLE_COORDINATE, ROLE_NODE, ROLE_SUPPORT,
         ROLE_TITLE, ROLE_SPEC, ROLE_OTHER)

_DIGITS = re.compile(r"\d")

# Порядок правил значим: первое подошедшее и объясняет элемент.
#
# Правила ловят только то, что **несёт в себе признак сам в себе**:
# угловые скобки у позиционного знака, букву оси у координаты, «DN» у
# трубы. Всё это видно в тексте элемента, и для этого хватает регулярки.
#
# Голые числа сюда намеренно не попали: «2» — это и номер узла в квадрате,
# и кусок размера, и номер страницы, и по тексту они неразличимы. Их решает
# разбор, а не регулярка, — см. :func:`role_of_number`.
_KEEP, _CUT = True, False

PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"^<\s*\d+\s*>$"), ROLE_NODE, "позиционный знак <N>"),
    # Пробелов между буквой и числом может быть несколько: span-ы склеиваются с
    # их исходным разделением, и «X  48300» с двумя пробелами — тот же
    # координатный блок, что и «X 48300». На листе 1 из-за двойного пробела
    # блок «X  48300 / DN50X50» не узнавался и уходил с листа целиком.
    (re.compile(r"^[XYZxyz]\s{0,3}[+±-]?\s{0,3}\d+"), ROLE_COORDINATE,
     "координата привязки"),
    (re.compile(r"^[ОOoККk]\s{0,3}\d+"), ROLE_SUPPORT, "опорный знак"),
    (re.compile(r"^\d+\s*/\s*\d+$"), ROLE_SUPPORT, "опорный знак «N/M»"),
    (re.compile(r"^[Сс][Мм]\.?\s*\d"), ROLE_COORDINATE,
     "привязка к другому листу"),
    (re.compile(r"^\d{2,4}[_-]\d{2,3}[_-]\w+"), ROLE_SPEC, "номер чертежа"),
    # «DN50X50» — обозначение трубы с размерами, без пробела между буквами и
    # числом. Прежнее правило требовало пробел и такой блок не узнавало.
    (re.compile(r"^[Dd][Nn]\s{0,3}\d+\s{0,3}[XxХх]\s{0,3}\d+"), ROLE_SPEC,
     "обозначение трубы DN с размерами"),
    (re.compile(r"^[Dd][Nn]\s{0,3}\d"), ROLE_SPEC, "обозначение трубы DN"),
    (re.compile(r"^[SW]{1,2}[_-]?\d", re.IGNORECASE), ROLE_TITLE,
     "обозначение комплекта"),
    (re.compile(r"^Лист\b", re.IGNORECASE), ROLE_TITLE, "номер листа"),
    (re.compile(r"^Страница\b", re.IGNORECASE), ROLE_TITLE, "штамп"),
    # Словесный шум без цифр. Без этих правил он уходил с листа целиком: отбор
    # идёт по элементам с цифрами, а в «ТЕСТОВОЕ ЗАДАНИЕ», «N» и «НПЕ» их нет.
    # Роль ROLE_OTHER — единственная из не-размерных, что означает «вырезать».
    (re.compile(r"ТЕСТОВОЕ\s*ЗАДАНИЕ", re.IGNORECASE), ROLE_OTHER,
     "водяной знак комплекта"),
    (re.compile(r"^[Рр][Вв][Бб][Лл]\b"), ROLE_SUPPORT, "опорный знак РВБЛ"),
    (re.compile(r"^[Нн][Ии][Пп][Ее]\b"), ROLE_OTHER, "текстовый знак НПЕ"),
    # Стрелка севера и служебные буквы у её основания: без цифр и без рамки.
    (re.compile(r"^[Сс]\.?$"), ROLE_OTHER, "буква при стрелке севера"),
    (re.compile(r"^[Nn]$"), ROLE_OTHER, "буква при стрелке севера"),
    # Привязка к другому листу без номера следом: «СМ.» стоит отдельной
    # строкой над блоком, и в склеенном элементе цифры за ним может не быть.
    (re.compile(r"^[Сс][Мм]\.?\s*$"), ROLE_COORDINATE, "привязка к другому листу"),
    (re.compile(r"ПОДКЛЮЧЕНИЕ", re.IGNORECASE), ROLE_OTHER,
     "словесная примечание на листе"),
    (re.compile(r"^[A-Za-zА-Яа-я]{1,3}-[\d/]+$"), ROLE_OTHER,
     "обозначение точки подключения"),
    (re.compile(r"^\d+\s*mm\b", re.IGNORECASE), ROLE_SPEC,
     "диаметр в миллиметрах"),
    (re.compile(r"^(Н\.)?[ОOo]\.?\s*[А-Яа-яA-Za-z]"), ROLE_OTHER,
     "текстовый знак"),
    (re.compile(r"^[ОДНОДНА]{1,2}\s*$", re.IGNORECASE), ROLE_OTHER,
     "слово «одно»"),
    (re.compile(r"^ШТУРВАЛ\b", re.IGNORECASE), ROLE_OTHER, "обозначение штуцаров"),
)

# Роли, которые заведомо не размер. Регулярка может ошибиться и назвать
# размер чем-то иным; страховка — проверка ниже по рамке и разбору.
CUT_ROLES = frozenset({
    ROLE_COORDINATE, ROLE_NODE, ROLE_SUPPORT, ROLE_TITLE, ROLE_SPEC, ROLE_OTHER,
})


@dataclass
class Element:
    """Текстовый элемент с цифрами и его вырез из снимка листа."""

    mark: str                     # "N1", "N2" — метка для ответа модели
    text: str                     # что написано, для подсказки модели
    rect: tuple[float, float, float, float]        # элемент в пунктах PDF
    # Вырез появляется только когда он действительно рисуется: план удаления
    # обходится без картинок, и рендер их был бы лишней работой.
    crop: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    image: bytes = b""            # PNG выреза
    accepted: bool = False        # разбор счёл это размером
    role: str = ROLE_OTHER        # что это, по регулярке или разбору
    reason: str = ""              # почему такая роль

    @property
    def keep(self) -> bool:
        return self.role == ROLE_DIMENSION


@dataclass
class ElementBatch:
    """Вырезы одного листа."""

    page_number: int
    elements: list[Element] = field(default_factory=list)

    def by_mark(self) -> dict[str, Element]:
        return {e.mark: e for e in self.elements}


def _digits_present(text: str) -> bool:
    return bool(_DIGITS.search(text))


def spans_of(pdf_page: pymupdf.Page) -> list[tuple[str, pymupdf.Rect]]:
    """Текстовые элементы листа с их рамками, в пунктах PDF.

    Берётся ``rawdict``, а не ``dict``: ``dict`` режет строку по пробелам, и
    связка ``Z+ 9695`` распадается на ``Z+`` и ``9695``. ``rawdict`` отдаёт
    span-ы как есть, и в одном элементе остаётся всё, что нарисовано рядом.
    """
    raw = pdf_page.get_text("rawdict")
    out: list[tuple[str, pymupdf.Rect]] = []
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = "".join(ch["c"] for ch in span["chars"]).strip()
                if not text:
                    continue
                box = pymupdf.Rect(span["bbox"])
                # Пробелы-разделители между span-ами одной строки — часть
                # подписи, а не отдельный элемент: «X  78500» приходит двумя
                # span-ами, и модель должна увидеть связку целиком.
                out.append((text, box))
    return out


def group_spans(
    spans: Sequence[tuple[str, pymupdf.Rect]],
) -> list[tuple[str, pymupdf.Rect]]:
    """Склеить span-ы одной строки в элементы.

    Склеиваются только те, чьи рамки стоят на одной высоте и почти
    соприкасаются: ``X`` + ``78500`` становятся ``X 78500``. Разрыв
    условный — он должен быть заметно меньше ширины цифры, иначе склеились бы
    соседние подписи, стоящие рядом.
    """
    ordered = sorted(spans, key=lambda s: (round(s[1].y0, 1), s[1].x0))
    merged: list[tuple[list[str], pymupdf.Rect]] = []
    for text, box in ordered:
        if merged:
            parts, current = merged[-1]
            same_line = abs(box.y0 - current.y0) <= max(
                1.5, 0.35 * min(box.height, current.height)
            )
            gap = box.x0 - current.x1
            if same_line and -0.5 <= gap <= max(2.0, 0.6 * box.height):
                joined = " ".join(parts + [text])
                merged[-1] = (parts + [text], current | box)
                del joined
                continue
        merged.append(([text], pymupdf.Rect(box)))
    return [(" ".join(parts), box) for parts, box in merged]


def crop_rect(
    box: pymupdf.Rect, page: pymupdf.Rect,
    ratio: float | None = None,
    pad_ratio: float | None = None,
    min_ratio: float | None = None,
) -> pymupdf.Rect:
    """Вырез вокруг элемента, в пунктах PDF.

    Размер задаётся долей меньшей стороны листа, а не числом пикселей: при
    смене ``RENDER_DPI`` вырез остаётся тем же куском чертежа, но в большем
    разрешении. Число пикселей в задании не годится — оно заставляет либо
    рендерить вырезы в чужом разрешении, либо получать кашу на краях.
    """
    share = config.PRUNE_CROP_RATIO if ratio is None else ratio
    pad = config.PRUNE_CROP_PAD_RATIO if pad_ratio is None else pad_ratio
    least = config.PRUNE_CROP_MIN_RATIO if min_ratio is None else min_ratio

    least_side = min(page.width, page.height)
    side = share * least_side
    side = max(side, box.width * least + 2 * pad * least_side)
    side = max(side, box.height * least + 2 * pad * least_side)
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    return pymupdf.Rect(cx - side / 2, cy - side / 2,
                        cx + side / 2, cy + side / 2)


def _page_scale(pdf_page: pymupdf.Page, dpi: float, max_side: int) -> float:
    """Пикселей на пункт с учётом ограничения по стороне."""
    scale = dpi / 72.0
    longest = max(pdf_page.rect.width, pdf_page.rect.height) * scale
    return scale if longest <= max_side else max_side / longest


def _crop_pixmap(pixmap: pymupdf.Pixmap, rect: pymupdf.Rect) -> pymupdf.Pixmap:
    """Вырезать кусок растра (совместимость; основной путь — ``crop_bytes``)."""
    from .raster import crop_bytes

    array = as_rgb(pixmap)
    data = crop_bytes(pixmap, array, rect)
    if not data:
        return pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 0, 0), False)
    return pymupdf.Pixmap(data)


def render_crops(
    pdf_page: pymupdf.Page,
    elements: Sequence[tuple[str, pymupdf.Rect]],
    dpi: float | None = None,
    max_side: int = 0,
) -> list[tuple[str, pymupdf.Rect, bytes, pymupdf.Rect]]:
    """Вырезы из одного рендера листа.

    Лист рендерится один раз, а не по разу на элемент: на восьмом листе
    элементов больше сорока, и сорок рендеров страницы дороже всей остальной
    подготовки. ``max_side`` по умолчанию не ограничивает: вырезы маленькие,
    им нужен максимум разрешения, а экономия памяти на всю страницу тут
    неуместна — лишние два мегабайта никто не заметит, а ступеньки на буквах
    модель заметит.
    """
    if not elements:
        return []
    dpi = config.PRUNE_CROP_DPI if dpi is None else dpi
    scale = _page_scale(pdf_page, dpi, max_side) if max_side else dpi / 72.0
    pixmap = pdf_page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                                 alpha=False)
    array = as_rgb(pixmap)
    origin_x, origin_y = pixmap.irect[0], pixmap.irect[1]

    out: list[tuple[str, pymupdf.Rect, bytes, pymupdf.Rect]] = []
    for text, box in elements:
        area = crop_rect(box, pdf_page.rect)
        cx, cy = (area.x0 + area.x1) / 2, (area.y0 + area.y1) / 2
        half = max(area.width, area.height) / 2
        rect_px = pymupdf.Rect(
            (cx - half) * scale + origin_x, (cy - half) * scale + origin_y,
            (cx + half) * scale + origin_x, (cy + half) * scale + origin_y,
        )
        image = crop_bytes(pixmap, array, rect_px)
        crop_pt = pymupdf.Rect(
            rect_px.x0 / scale + origin_x, rect_px.y0 / scale + origin_y,
            rect_px.x1 / scale + origin_x, rect_px.y1 / scale + origin_y,
        )
        out.append((text, box, image, crop_pt))
    return out


def role_of_text(text: str) -> tuple[str, str] | None:
    """Роль элемента по его собственному тексту, без картинки.

    Возвращает ``None``, если текст ничего не даёт: такому элементу роль
    назначает разбор, а не правило.
    """
    normalised = " ".join(text.split())
    for pattern, role, reason in PATTERNS:
        if pattern.match(normalised):
            return role, reason
    return None


def role_of_number(accepted: bool) -> tuple[str, str]:
    """Роль голого числа — по решению разбора, а не по тексту.

    Голое число неразличимо само в себе: «2» бывает номером узла в квадрате
    и куском размера. Разбор уже решил это по наличию размерной линии, и его
    решение здесь единственное, чем можно опереться.
    """
    if accepted:
        return ROLE_DIMENSION, "разбор нашёл для него размерную линию"
    return ROLE_OTHER, "разбор не нашёл для него размерной линии"


def assign_roles(
    elements: Sequence[Element],
    accepted_boxes: Sequence[tuple[float, float, float, float]] = (),
    slack: float = 1.0,
) -> list[Element]:
    """Проставить роли элементам.

    Логика от обратного: **размер — это то, что разбор назвал размером, всё
    остальное вырезается**. Регулярка не решает, что оставить, — она только
    объясняет в отчёте, что именно было вырезано.

    Признак один, и он абсолютный: рамка элемента совпала с рамкой подписи
    размера из разбора. Совпадение точное — на десяти листах 115 подписей
    размеров и 115 совпадений, ни одного размера без своего элемента и ни
    одного элемента-размера лишнего. Проверка на повторяющихся текстах
    показала, что это работает и там, где по тексту не различить: четыре
    подписи «159» на листе 4 разошлись с размерами #3, #9, #16, #17 рамка к
    рамке, хотя расстояния до линий различаются меньше чем на пункт.

    Раньше роль по умолчанию была «прочее» = не трогать, и с листа уходило
    188 чужих подписей. Обратная логика пробовалась и с обоснованием, что рамки
    «расходятся на доли пункта всегда»; на деле расходятся, но не с тем
    элементом: сравнение шло со списком целиком, а не с нужной подписью.
    """
    out: list[Element] = []
    for element in elements:
        box = tuple(round(v, 1) for v in element.rect)
        accepted = any(
            abs(box[0] - ref[0]) <= slack and abs(box[1] - ref[1]) <= slack
            and abs(box[2] - ref[2]) <= slack and abs(box[3] - ref[3]) <= slack
            for ref in accepted_boxes
        )
        verdict = role_of_text(element.text)
        if accepted:
            role, reason = ROLE_DIMENSION, "разбор нашёл для него размерную линию"
        elif verdict is not None:
            role, reason = verdict[0], verdict[1]
        else:
            role, reason = ROLE_OTHER, "разбор не нашёл для него размерной линии"
        out.append(
            Element(
                mark=element.mark, text=element.text, rect=element.rect,
                crop=element.crop, image=element.image,
                accepted=accepted, role=role, reason=reason,
            )
        )
    return out


def elements_of(
    pdf_page: pymupdf.Page,
    only_with_digits: bool = True,
) -> list[Element]:
    """Элементы листа с их ролями, **без** вырезов.

    Отделено от :func:`build_batch` не для удобства: рендер вырезов стоит
    секунды на лист, а план удаления нуждается только в тексте и рамках.
    Считать вырезы там, где они не показываются модели, — чистая потеря
    времени, и на десяти листах она превращалась в минуты.
    """
    chosen = [
        (text, box) for text, box in group_spans(spans_of(pdf_page))
        if not only_with_digits or _digits_present(text)
    ]
    return [
        Element(
            mark=f"N{index}", text=text,
            rect=(box.x0, box.y0, box.x1, box.y1),
        )
        for index, (text, box) in enumerate(chosen, start=1)
    ]


def build_batch(
    pdf_page: pymupdf.Page,
    dpi: float | None = None,
    max_side: int = 0,
    only_with_digits: bool = True,
) -> ElementBatch:
    """Вырезы всех подозрительных элементов листа.

    Берутся **все** элементы, где есть цифры, а не только отвергнутые
    разбором: разбор может ошибиться в обе стороны, и отзыв у модели есть
    только на элементы, которые ей показали.
    """
    elements = elements_of(pdf_page, only_with_digits)
    crops = render_crops(
        pdf_page, [(e.text, pymupdf.Rect(e.rect)) for e in elements],
        dpi, max_side,
    )
    for element, (text, box, image, area) in zip(elements, crops, strict=True):
        element.image = image
        element.crop = (area.x0, area.y0, area.x1, area.y1)
    return ElementBatch(page_number=pdf_page.number + 1, elements=elements)