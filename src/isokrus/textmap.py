"""Текстовый слой PDF в координатах отрендеренной картинки.

Зачем модуль: подписи на изометрии (размеры, номера узлов, координаты привязки)
рисуются шрифтом, поэтому у них есть точные координаты в PDF. Раньше модель
угадывала положение подписи глазами по картинке — почти всегда мимо. Теперь мы
отдаём ей те же координаты, что и в пикселях отрендеренного PNG, и по тем же
же числам рисуем разметку: обе стороны гарантированно считаются в одной
системе координат.

Ключевое условие: рендер страницы (``extract.py``) и извлечение текста
должны использовать **одну и ту же** матрицу, и применять её целиком. Тогда
координаты из ``get_text("rawdict")`` (пункты PDF) попадают ровно в пиксели
картинки, ушедшей в модель.

Если картинку обрезали (``trim.py``), матрица сдвигается на величину среза
(см. :func:`shifted_matrix`): тогда подписи извлекаются уже в координатах
обрезанной картинки и двигать их вручную не нужно — а значит, нельзя
забыть.

Формат подсказки для промта — по одной подписи в строке:

    24100 [2208,1452,2256,1476]
    X 78500 [1160,204,1176,222]

Слева — текст, справа — рамка в пикселях изображения (начало в левом верхнем
углу). Ничего больше: ни шрифт, ни кернинг, ни технические блоки.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import pymupdf

# Токен, который в system.md эксперимента заменяется блоком извлечённого текста.
PDF_TEXT_TOKEN = "{{PDF_TEXT}}"

_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"^[\d.,]+$")

# Во сколько раз координата в выбранных единицах больше пункта PDF (1 pt = 1/72").
# Единственное определение на проект: standalone-скрипт extract_text.py и
# конвейер обязаны считать одинаково, иначе блок для промта и разметка
# разъедутся по координатам.
UNIT_SCALE: dict[str, float | None] = {
    "pt": 1.0, "px": None, "in": 1 / 72, "mm": 25.4 / 72,
}


def unit_scale(units: str = "pt", dpi: float = 72.0) -> float:
    """Множитель «пункты PDF -> выбранные единицы»."""
    if units not in UNIT_SCALE:
        raise ValueError(f"Неизвестные единицы координат: {units!r}")
    scale = UNIT_SCALE[units]
    return dpi / 72.0 if scale is None else scale


def shifted_matrix(
    matrix: pymupdf.Matrix,
    dx: float,
    dy: float,
) -> pymupdf.Matrix:
    """Матрица рендера, сдвинутая на ``dx``/``dy`` пикселей растра.

    Нужна после обрезки листа: масштаб остаётся прежним, а начало координат
    переезжает в угол обрезанной картинки. Благодаря этому подписи, извлечённые
    с такой матрицей, сразу лежат в координатах той картинки, что ушла в
    модель, — и промт, и разметка читают одни и те же числа.

    Сдвиг задаётся в пикселях результата, поэтому идёт в ``e``/``f`` как есть:
    ``x' = zoom * x - dx``.
    """
    zoom = float(matrix[0]) or 1.0
    return pymupdf.Matrix(zoom, 0.0, 0.0, zoom, -dx, -dy)


def to_units(
    box: Sequence[float],
    scale: float,
    page_height: float,
    origin: str = "top-left",
) -> list[float]:
    """Пересчитать рамку из пунктов PDF в нужные единицы.

    ``origin`` — ``top-left`` как в PDF (Y вниз) либо ``bottom-left`` как в
    CAD (Y вверх). Для переворота нужен только размер страницы по Y: ось X
    в обеих системах направлена одинаково.
    """
    x0, y0, x1, y1 = box
    if origin == "bottom-left":
        y0, y1 = page_height - y1, page_height - y0
    return [round(v * scale, 2) for v in (x0, y0, x1, y1)]


def box_to_pixels(
    box: Sequence[float], matrix: pymupdf.Matrix
) -> tuple[float, float, float, float]:
    """Рамка из пунктов PDF в пиксели картинки по той же матрице.

    Единственный путь, которым любые данные с листа попадают в промт и в
    разметку: модель видит картинку, обрезанную по пустым полям, поэтому и
    рамки обязаны быть в её системе координат. Матрица после обрезки уже
    сдвинута на величину среза (см. :func:`shifted_matrix`), и рамка уезжает
    вместе с картинкой сама — двигать координаты руками не нужно, а значит,
    негде забыть.

    Обрабатываются все четыре угла, а не левый верхний: при повёрнутой оси
    (такие есть на чертёже) порядок углов не гарантирован, а вернуть надо
    упорядоченный прямоугольник.
    """
    a, b, c, d, e, f = _matrix_parts(matrix)
    xs: list[float] = []
    ys: list[float] = []
    for px, py in ((box[0], box[1]), (box[2], box[1]),
                   (box[2], box[3]), (box[0], box[3])):
        xs.append(a * px + c * py + e)
        ys.append(b * px + d * py + f)
    return (min(xs), min(ys), max(xs), max(ys))


# --------------------------------------------------------------------------
# Элемент текста
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TextItem:
    """Одна подпись страницы в пикселях рендера.

    ``bbox`` — уже в системе координат картинки (0,0 — левый верхний угол),
    поэтому ни модель, ни разметка ничего не пересчитывают.
    """

    text: str
    bbox: tuple[float, float, float, float]
    size: float = 0.0  # высота шрифта в пикселях рендера
    angle: float = 0.0  # градусы, 0 — горизонтально (Y вниз, как в PDF)

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    def as_line(self) -> str:
        """Строка для промта: ``"текст" [x0,y0,x1,y1]`` (+ ``∠`` для повёрнутых)."""
        box = ",".join(f"{v:.0f}" for v in self.bbox)
        mark = f" ∠{self.angle:.0f}" if abs(self.angle) > 5 else ""
        return f'"{self.text}" [{box}]{mark}'


def norm_key(text: str) -> str:
    """Ключ для сопоставления подписей: без регистра и лишних пробелов."""
    return _WS_RE.sub(" ", text.replace("\u00a0", " ").replace("\u2007", " ")).strip().lower()


# --------------------------------------------------------------------------
# Извлечение
# --------------------------------------------------------------------------


def _matrix_parts(matrix: pymupdf.Matrix) -> tuple[float, ...]:
    """Матрица как шесть чисел: ``x' = a*x + c*y + e``, ``y' = b*x + d*y + f``."""
    return tuple(float(v) for v in matrix)


def _token_bbox(
    chars: list[dict],
    m: tuple[float, ...],
) -> tuple[float, float, float, float] | None:
    """Рамка группы символов, переведённая в пиксели картинки.

    Применяется **вся** матрица, а не только масштаб: у матрицы после обрезки
    листа есть сдвиг, и рамка должна уехать вместе с картинкой. Для осей,
    не совпадающих с экранными, берутся все четыре угла.
    """
    a, b, c, d, e, f = m
    straight = not b and not c
    x0 = y0 = x1 = y1 = None
    for char in chars:
        box = char.get("bbox")
        if not box:
            continue
        if straight:
            xs = (a * box[0] + e, a * box[2] + e)
            ys = (d * box[1] + f, d * box[3] + f)
        else:
            xs, ys = [], []
            for px, py in ((box[0], box[1]), (box[2], box[1]),
                           (box[0], box[3]), (box[2], box[3])):
                xs.append(a * px + c * py + e)
                ys.append(b * px + d * py + f)
        lo_x, hi_x = min(xs), max(xs)
        lo_y, hi_y = min(ys), max(ys)
        x0 = lo_x if x0 is None else min(x0, lo_x)
        y0 = lo_y if y0 is None else min(y0, lo_y)
        x1 = hi_x if x1 is None else max(x1, hi_x)
        y1 = hi_y if y1 is None else max(y1, hi_y)
    if x0 is None:
        return None
    return (x0, y0, x1, y1)


def _split_span(span: dict) -> list[list[dict]]:
    """Разбить спан на токены по пробельным символам, сохраняя сами символы.

    ``span["text"]`` для разбивки не годится: на изометрии в один спан
    попадает вся размерная цепочка («24100»), и сопоставить её с конкретным
    числом уже невозможно. Рамку токена собираем из рамок его символов.
    """
    tokens: list[list[dict]] = []
    current: list[dict] = []
    for char in span.get("chars", []):
        if char.get("c", "").isspace():
            if current:
                tokens.append(current)
                current = []
            continue
        current.append(char)
    if current:
        tokens.append(current)
    return tokens


def extract_text_items(
    page: pymupdf.Page,
    matrix: pymupdf.Matrix,
    min_height_px: float = 0.0,
    max_items: int | None = None,
) -> list[TextItem]:
    """Подписи страницы в пикселях картинки, отсортированные как читают лист.

    ``matrix`` — та же матрица, что и в ``page.get_pixmap()`` (у обрезанного
    листа — сдвинутая на срез, см. :func:`shifted_matrix`). Иначе координаты
    разойдутся с картинкой, которую видит модель.
    """
    zoom = float(matrix[0]) or 1.0
    parts = _matrix_parts(matrix)
    items: list[TextItem] = []
    seen: set[tuple] = set()

    for block in page.get_text("rawdict").get("blocks", []):
        if block.get("type") != 0:  # 1 = изображение
            continue
        for line in block.get("lines", []):
            dx, dy = line.get("dir", (1.0, 0.0))
            angle = round(math.degrees(math.atan2(dy, dx)), 1)
            for span in line.get("spans", []):
                span_size = float(span.get("size", 0.0)) * zoom
                if min_height_px and span_size < min_height_px:
                    continue
                for chars in _split_span(span):
                    text = "".join(c.get("c", "") for c in chars).strip()
                    bbox = _token_bbox(chars, parts)
                    if not text or bbox is None:
                        continue
                    key = (round(bbox[0]), round(bbox[1]), round(bbox[2]), round(bbox[3]), text)
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append(
                        TextItem(
                            text=text,
                            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                            size=span_size,
                            angle=angle,
                        )
                    )

    # Порядок чтения: сверху вниз, в строке слева направо.
    items.sort(key=lambda it: (round(it.y0, 1), it.x0))

    if max_items and len(items) > max_items:
        # Обрезаем равномерно, а не «первые N»: нужен весь лист, а не его угол.
        step = len(items) / float(max_items)
        items = [items[int(i * step)] for i in range(max_items)]
    return items


def inside_box(
    items: Sequence[TextItem],
    box: Sequence[float],
) -> tuple[list[TextItem], int]:
    """Оставить подписи, центр которых попал в рамку.

    Рамка найдена по нарисованным пикселям, поэтому подпись вне неё — это
    либо бледная графика, которую обрезка не увидела, либо артефакт слоя.
    Такие подписи убираем: в промте координата за пределами картинки модели
    бесполезна, а число из неё всё равно не найдётся при разметке.

    ``box`` задаётся в той же системе, что и ``items``. Для обрезанного листа
    это размеры самой картинки, ``(0, 0, width, height)``.

    Возвращает ``(оставшиеся, отброшено)`` — второе число попадает в
    ``PageImage.meta``, чтобы молчаливая потеря подписей была видна.
    """
    x0, y0, x1, y1 = box
    kept: list[TextItem] = []
    dropped = 0
    for item in items:
        cx = (item.x0 + item.x1) / 2.0
        cy = (item.y0 + item.y1) / 2.0
        if x0 <= cx < x1 and y0 <= cy < y1:
            kept.append(item)
        else:
            dropped += 1
    return kept, dropped


# --------------------------------------------------------------------------
# Поиск по подписям
# --------------------------------------------------------------------------


class TextIndex:
    """Подписи листа, разложенные по тексту: ``occurrences("24100")`` -> позиции.

    Нужен, чтобы по числу, названному моделью, найти его начертание на листе.
    Одно и то же число на изометрии встречается многократно («6000» четыре
    раза), поэтому отдаются все вхождения, а раздаёт их вызывающий код.
    """

    def __init__(self, items: Iterable[TextItem]) -> None:
        self.items: list[TextItem] = list(items)
        self._by_text: dict[str, list[int]] = {}
        for index, item in enumerate(self.items):
            self._by_text.setdefault(norm_key(item.text), []).append(index)

    def __len__(self) -> int:
        return len(self.items)

    def occurrences(self, key: str) -> list[int]:
        """Индексы всех вхождений уже нормализованного ключа ``key``."""
        return list(self._by_text.get(key, ()))

    def numbers(self) -> list[TextItem]:
        """Только числовые подписи — вероятные размеры."""
        return [it for it in self.items if _NUM_RE.match(norm_key(it.text).replace(" ", ""))]


# --------------------------------------------------------------------------
# Блок для системного промта
# --------------------------------------------------------------------------


def format_text_block(
    items: Sequence[TextItem],
    page_width: int,
    page_height: int,
    title: str = "Текстовый слой листа, извлечённый из PDF",
) -> str:
    """Подсказка для модели: подписи + их рамки в пикселях той же картинки."""
    if not items:
        return (
            f"# {title}\n\n"
            "Текстовый слой листа ПУСТ — это скан без встроенного текста. "
            "Ориентируйся только по картинке."
        )

    lines = [
        f"# {title}",
        "",
        f"Изображение этого листа: {page_width}x{page_height} px.",
        (
            'Формат строки: "текст подписи" [x0,y0,x1,y1] — рамка подписи '
            "в пикселях изображения, начало координат в левом верхнем углу, "
            "Y растёт вниз."
        ),
        (
            "Координаты точные, из того же PDF, что и картинка. Чтобы указать, "
            "где находятся длины и координаты привязки, бери рамки прямо отсюда."
        ),
        "",
    ]
    lines.extend(item.as_line() for item in items)
    return "\n".join(lines)
