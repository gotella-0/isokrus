"""Разметка чертежей: где на листе находятся прочитанные моделью числа.

Координаты больше не приходят «на глаз» из ответа модели. Есть два источника,
и оба считаются в одном пространстве — в пикселях отрендеренной картинки:

1. **Текстовый слой PDF** (``textmap.py``) — основной. Подписи на изометрии
   нарисованы шрифтом, поэтому их рамки известны точно. Модель возвращает
   *числа* («24100», «6000»), программа находит их начертания на листе и
   обводит именно их, попиксельно.
2. **``bbox`` из ответа модели** — только если текстового слоя нет (скан без
   встроенного текста). Раньше он был основным, и рамки уезжали мимо размеров;
   рисовать его поверх точных рамок только запутывало бы картинку.

Раньше система координат угадывалась по величине числа («< 1000 — значит
нормализованные, иначе пиксели»). На картинке шире 1000 px это ломалось, и
рамки уезжали. Теперь пространство задаётся явно
(``config.ANNOTATE_COORD_SPACE``), а рамки проверяются на попадание в лист.

Порядок ответа модели значения не важен: итоговую длину собирает программа
(см. ``report.py``), а здесь мы только показываем, где что на листе.

Лист уже обрезан по пустым полям (``trim.py``), поэтому «поля, на которых
лежит легенда» больше нет: она переехала в правый нижний угол.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import pymupdf

from . import config
from .extract import PageImage
from .indices import resolve_dimension_indices
from .textmap import TextIndex, TextItem, norm_key

if TYPE_CHECKING:  # только для аннотаций, без циклического импорта
    from .pipeline import RunResult


# --- Палитра ---------------------------------------------------------------
# Цвет несёт смысл, а не украшает: инженер должен за секунду отличить
# обобщённую длину от дочернего участка.

WHITE: tuple[float, float, float] = (1.0, 1.0, 1.0)
BLACK: tuple[float, float, float] = (0.05, 0.05, 0.05)
LEGEND_BG: tuple[float, float, float] = (1.0, 1.0, 1.0)
LEGEND_EDGE: tuple[float, float, float] = (0.35, 0.35, 0.35)

PARENT_COLOR: tuple[float, float, float] = (0.05, 0.42, 0.85)  # синий — обобщённая
CHILD_COLOR: tuple[float, float, float] = (0.00, 0.58, 0.20)  # зелёный — участок
MODEL_COLOR: tuple[float, float, float] = (0.90, 0.52, 0.05)  # оранжевый — bbox модели
ERROR_COLOR: tuple[float, float, float] = (0.82, 0.12, 0.12)  # красный — ошибка

COLORS: dict[str, tuple[float, float, float]] = {
    "included": CHILD_COLOR,
    "main": CHILD_COLOR,
    "used": CHILD_COLOR,
    "child": CHILD_COLOR,
    "parent": PARENT_COLOR,
    "branch": PARENT_COLOR,
    "total": PARENT_COLOR,
    "excluded": (0.65, 0.65, 0.68),  # серый — не использовано
    "reference": (0.45, 0.45, 0.50),
    "ambiguous": MODEL_COLOR,  # оранжевый — требует проверки
    "warning": MODEL_COLOR,
    "error": ERROR_COLOR,
    "failed": ERROR_COLOR,
    "note": (0.42, 0.28, 0.68),
    "model_bbox": MODEL_COLOR,
}
DEFAULT_COLOR = (0.15, 0.15, 0.15)

# Служебные числа ответа: на листе их нет, искать их бессмысленно.
# Без фильтра разметка обводила бы номера участков, количество повторов и
# координаты рамок вместо размеров, а в легенде сыпались бы ложные
# «нет на листе».
META_KEYS = (
    "index", "order", "seq", "number", "count", "n", "repeat", "repeats",
    "repeat_count", "confidence", "status", "sheet", "page", "list", "layer",
    "priority", "id", "code",
)
# Границы рамки — координаты на картинке, а не миллиметры трассы.
FRAME_KEYS = ("x0", "y0", "x1", "y1", "left", "top", "right", "bottom",
              "width", "height", "cx", "cy")
# Блоки самопроверки модели — это арифметика, а не то, что нарисовано.
SKIP_CONTEXTS = ("verification", "self_check", "selfcheck", "check", "stats", "metadata")

_BOX_KEYS = (("x0", "y0", "x1", "y1"), ("left", "top", "right", "bottom"))
_LABEL_KEYS = (
    "label", "text", "raw_text", "designation", "line_designation",
    "comment", "description", "name", "title", "note",
)
_ROLE_KEYS = ("status", "role", "kind", "type", "category", "state", "class")

# Встроенный шрифт PDF (Helvetica) не умеет кириллицу, поэтому нечитаемый
# текст молча превращался в квадраты. Всё, что не latin1, заменяем заранее.
_REPLACEMENTS = {
    "°": " deg", "Ø": "D", "ø": "d", "×": "x", "✕": "x", "✖": "x",
    "—": "-", "–": "-", "−": "-", "…": "...", "·": "-",
    "≤": "<=", "≥": ">=", "±": "+-", "²": "2",
    "⟨": "<", "⟩": ">", "⟶": "->",
}
_NON_LATIN_RE = re.compile(r"[^ -~]")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Модель разметки
# ---------------------------------------------------------------------------


@dataclass
class Region:
    """Найденный на чертеже элемент. Рамка — в пикселях картинки."""

    bbox: tuple[float, float, float, float]
    label: str
    color: tuple[float, float, float]
    kind: str
    origin: str = "pdf_text"  # pdf_text | model_bbox

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

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


@dataclass
class Candidates:
    """Числа, которые модель назвала, и что с ними удалось сделать на листе."""

    regions: list[Region]
    found: int = 0
    duplicated: int = 0
    missing: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Приведение типов
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ".").strip())
        except ValueError:
            return None
    return None


def _number_text(value: float) -> list[str]:
    """Как число может быть написано на чертеже: «24100», «24100.0», «24 100»."""
    forms = [f"{value:g}"]
    if value.is_integer():
        forms.append(str(int(value)))
    else:
        forms.append(f"{value:.1f}")
    out: list[str] = []
    for form in forms:
        out.append(form)
        spaced = f"{int(value):,}".replace(",", " ") if value.is_integer() else form
        if spaced != form:
            out.append(spaced)
    return out


def _pick(node: dict, keys: Sequence[str]) -> str:
    for key in keys:
        value = node.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, list) and value:
            parts = [str(v) for v in value if isinstance(v, (str, int, float))]
            if parts:
                return ", ".join(parts[:2])
    return ""


def _color_for(node: dict, context: str = "") -> tuple[tuple[float, float, float], str]:
    """Цвет по статусу/роли узла, затем по имени ключа-контейнера."""
    candidates = (_pick(node, _ROLE_KEYS).lower(), context.lower())
    for raw in candidates:
        for key, color in COLORS.items():
            if key in raw:
                return color, key
    return DEFAULT_COLOR, "default"


# ---------------------------------------------------------------------------
# Координаты
# ---------------------------------------------------------------------------


def _extract_box(node: dict) -> tuple[float, float, float, float] | None:
    """Рамка из объекта: явные x0/y0/x1/y1 либо список bbox."""
    for keys in _BOX_KEYS:
        if all(k in node for k in keys):
            values = [_as_float(node[k]) for k in keys]
            if all(v is not None for v in values):
                x0, y0, x1, y1 = values  # type: ignore[misc]
                return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    for key in ("bbox", "box", "rect"):
        bbox = node.get(key)
        if isinstance(bbox, dict):
            found = _extract_box(bbox)
            if found:
                return found
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            values = [_as_float(v) for v in bbox[:4]]
            if all(v is not None for v in values):
                x0, y0, x1, y1 = values  # type: ignore[misc]
                return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    return None


def to_pixels(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    space: str | None = None,
) -> tuple[float, float, float, float] | None:
    """Перевести рамку в пиксели картинки.

    ``space`` — ``"px"`` (уже пиксели) или ``"norm1000"`` (0..1000 от левого
    верхнего угла). Пространство выбирается явно, а не угадывается по величине
    числа: картинка шире 1000 px, и старая эвристика «< 1000 значит
    нормализовано» работала только на части листов.
    """
    space = space or config.ANNOTATE_COORD_SPACE
    x0, y0, x1, y1 = box
    if space == "px":
        scale_x = scale_y = 1.0
    else:
        scale_x = width / 1000.0
        scale_y = height / 1000.0
    rect = pymupdf.Rect(x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
    rect.normalize()
    # Модель ошибается в обе стороны: рамка может уехать за край листа.
    rect = rect & pymupdf.Rect(0, 0, width, height)
    if rect.is_empty or rect.width < 1 or rect.height < 1:
        return None
    return (rect.x0, rect.y0, rect.x1, rect.y1)


def _grow(box: tuple[float, float, float, float], pad: float) -> tuple[float, float, float, float]:
    """Подпись в 8 pt высотой — рамку надо расширить, иначе она слипается с текстом."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return (
        x0 - pad,
        y0 - pad,
        x1 + pad + max(0.0, pad * 2 - w),
        y1 + pad + max(0.0, pad * 2 - h),
    )


# ---------------------------------------------------------------------------
# Обход ответа
# ---------------------------------------------------------------------------


def _walk(node: Any, context: str = "", depth: int = 0) -> Iterator[tuple[dict, str]]:
    if depth > 12:
        return
    if isinstance(node, dict):
        yield node, context
        for key, value in node.items():
            yield from _walk(value, key if isinstance(value, (dict, list)) else context, depth + 1)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item, context, depth + 1)


def model_regions(
    response: dict | None,
    width: int,
    height: int,
    space: str | None = None,
) -> list[Region]:
    """Рамки, которые вернула сама модель (запасной источник)."""
    if not isinstance(response, dict):
        return []

    best: dict[tuple, Region] = {}
    order: list[tuple] = []

    for node, context in _walk(response):
        box = _extract_box(node)
        if box is None:
            continue
        pixels = to_pixels(box, width, height, space)
        if pixels is None:
            continue
        color, kind = _color_for(node, context)
        label = _pick(node, _LABEL_KEYS) or kind
        region = Region(pixels, label=label, color=color, kind=kind, origin="model_bbox")

        key = tuple(round(v) for v in pixels)
        if key not in best:
            best[key] = region
            order.append(key)
        elif _specificity(region) > _specificity(best[key]):
            best[key] = region

    return [best[k] for k in order]


def _specificity(region: Region) -> tuple[int, int]:
    """Насколько узел информативен: осмысленная подпись + известная роль."""
    return (
        1 if region.label and region.label != "default" else 0,
        1 if region.kind != "default" else 0,
    )


# ---------------------------------------------------------------------------
# Привязка чисел к подписям листа
# ---------------------------------------------------------------------------


def _numbers_of(node: dict) -> Iterator[tuple[str, float]]:
    """Числа узла с указанием поля, из которого они взяты.

    Строковые числа пропускаются намеренно: в ``label`` модели почти всегда
    пишет размер словами или с единицами («2160 мм»), и такие значения не
    найти в текстовом слое — только засорить легенду «нет на листе».
    """
    for key, value in node.items():
        if key in _LABEL_KEYS or key in _ROLE_KEYS or key in META_KEYS or key in FRAME_KEYS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            yield key, float(value)


def _is_parent_role(key: str) -> bool:
    """Обобщённая длина ветви: ``parent_length``, ``total_length_mm``, ...

    Сопоставление по подстроке, а не по точному списку имён: набор полей
    задаёт эксперимент, и жёсткий перечень развалился бы на следующей же
    новой схеме ответа.
    """
    key = key.lower()
    return any(name in key for name in ("parent", "total", "overall", "summary", "общ"))


def _is_child_role(key: str) -> bool:
    """Массив дочерних участков: ``child_segments``, ``segments``, ``parts``..."""
    key = key.lower()
    return any(name in key for name in ("child", "segment", "part", "section", "участок"))


def _is_excluded_context(context: str) -> bool:
    """Число лежит в списке исключений ответа, а не во взятых.

    Имена полей у экспериментов разные (``excluded``, ``dropped``, ``skipped``),
    поэтому ищем по подстроке, как в :func:`_is_parent_role`.
    """
    key = context.lower()
    return any(
        name in key for name in ("exclud", "drop", "skipp", "not_counted", "nested")
    )


def collect_candidates(response: dict | None) -> list[tuple[float, str, str | None]]:
    """Все числа ответа в порядке появления, роль каждого и его метка.

    Порядок важен: повторяющиеся размеры («6000» на листе встречается четыре
    раза) раздаются подписям по очереди, иначе четыре сегмента попадут в одну
    рамку. Модель перечисляет ветви в порядке чтения чертежа, поэтому этот
    порядок — лучшее из доступных приближений к ходу трассы.

    Метка (``P8``) нужна, чтобы подписать рамку ею, а не порядковым номером:
    ответ по меткам содержит чисел нет вовсе, и без метки на разметке не видно,
    что модель сочла участком, а что исключила из расчёта. В ответах без меток остаётся
    ``None``, и рамка подписывается номером, как раньше.

    Числа внутри объектов-сегментов уже собраны обходом ``_walk`` (он входит
    в любой вложенный dict), поэтому массовые массивы обрабатываются здесь
    только когда это числа, а не объекты.
    """
    if not isinstance(response, dict):
        return []

    found: list[tuple[float, str, str | None]] = []

    def add(value: float, role: str, mark: str | None = None) -> None:
        if abs(value) > 1e-9:
            found.append((value, role, mark))

    def mark_of(node: dict) -> str | None:
        """Метка размера рядом с числом: ``mark``/``line_id``/``index``."""
        for key in ("mark", "line_id", "dimension_mark"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    for node, context in _walk(response):
        if any(name in context.lower() for name in SKIP_CONTEXTS):
            continue  # блок самопроверки: арифметика модели, не чертёж

        # Метка берётся из узла целиком: у сегмента она лежит рядом с числом
        # в том же объекте, а не в отдельном поле ответа.
        mark = mark_of(node)

        # Роль целиком из того поля ответа, где число лежит. Число из
        # ``excluded`` — это размер, который модель исключила из расчёта
        # (обычно как вложенный), и рисовать его тем же цветом, что взятый,
        # незачем: ради различия и делается разметка.
        excluded = _is_excluded_context(context)
        role = "excluded" if excluded else ""

        # 1) Скалярные числа узла: length_mm, value, parent_length, ...
        for key, value in _numbers_of(node):
            if excluded:
                add(value, "excluded", mark)
            elif _is_parent_role(key):
                add(value, "parent", mark)
            else:
                add(value, "child", mark)

        # 2) Плоские массивы сегментов: child_segments: [6000, 6000, 4100]
        for key, value in node.items():
            if not isinstance(value, list) or not value:
                continue
            if not (_is_child_role(key) or _is_child_role(context)):
                continue
            for item in value:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    continue  # объекты-сегменты уже разобраны обходом выше
                add(float(item), role or "child", mark)

    return found


def _locate(
    index: TextIndex,
    value: float,
    used: dict[str, set[int]],
) -> tuple[TextItem, bool] | None:
    """Найти подпись со значением ``value``; ``None`` — значения на листе нет.

    Второй элемент ответа — предупреждение: значение на листе есть, но все его
    вхождения уже заняты (например «6000» встречается три раза, а модель
    сослалась на него четырежды). Такой случай честно помечается, а не
    молча рисуется вторым прямоугольником поверх первого.
    """
    for form in _number_text(value):
        key = norm_key(form)
        pool = index.occurrences(key)
        if not pool:
            continue
        busy = used.setdefault(key, set())
        free = [i for i in pool if i not in busy]
        pick = free[0] if free else pool[0]
        busy.add(pick)
        return index.items[pick], not free
    return None


def text_regions(
    page: PageImage,
    response: dict | None,
    color_overrides: dict[str, tuple[float, float, float]] | None = None,
) -> Candidates:
    """Разметить числа ответа по их реальным подписям на листе.

    Возвращает и неразмеченные значения — их полезно показать в легенде:
    значит, модель сослалась на число, которого на листе нет.

    Роль приходит из ответа, а не угадывается по цвету: у ответа по меткам
    есть ``excluded``, и размер, который модель исключила из расчёта, должен
    на разметке отличаться от взятого. Раньше исключённые и взятые выглядели
    одинаково — проверить ответ по картинке было нечем.
    """
    candidates = collect_candidates(response)
    if not candidates:
        return Candidates(regions=[])

    index = TextIndex(page.text_items)
    if not len(index):
        return Candidates(regions=[], found=0, duplicated=0,
                          missing=[_fmt(v) for v, _, _ in candidates])

    used: dict[str, set[int]] = {}
    regions: list[Region] = []
    missing: list[str] = []
    duplicates = 0
    hit = 0

    for value, role, mark in candidates:
        located = _locate(index, value, used)
        if located is None:
            missing.append(_fmt(value))
            continue
        item, duplicated = located
        hit += 1
        duplicates += int(duplicated)
        regions.append(
            Region(
                bbox=_grow(item.bbox, 2.0),
                label=mark or item.text,
                color=(color_overrides or {}).get(
                    role, COLORS.get(role, DEFAULT_COLOR)
                ),
                kind=role,
                origin="pdf_text",
            )
        )

    return Candidates(regions=regions, found=hit, duplicated=duplicates, missing=missing)


def _fmt(value: float) -> str:
    return f"{value:g}"


def _dedupe(regions: list[Region]) -> list[Region]:
    """Слить рамки, попавшие в одну подпись: рисуем один раз, но ярче."""
    merged: dict[tuple, Region] = {}
    order: list[tuple] = []
    for region in regions:
        key = tuple(round(v / 6.0) for v in region.bbox)  # ~6 px — терпимость
        if key in merged:
            merged[key] = region
            continue
        merged[key] = region
        order.append(key)
    return [merged[k] for k in order]


def collect_regions(
    response: dict | None,
    page: PageImage | None = None,
    space: str | None = None,
) -> list[Region]:
    """Все элементы для разметки.

    Текстовый слой — источник правды: рамки из него попиксельно точные.
    ``bbox`` модели используется **только** если текстового слоя нет (скан без
    встроенного текста). Иначе модельные рамки рисовались бы вторым слоем
    поверх точных и делали картинку только запутаннее — на листе 8 девять
    размеров находились по тексту, а модель отметила те же девять мимо.
    """
    if page is None:
        return []

    # Ответ со ссылками на метки размеров (P7) не содержит чисел — только
    # метки. Разворачиваем их в значения, иначе разметка не найдёт на листе
    # ничего и нарисует пустую картинку с «0 размечено».
    resolved = resolve_dimension_indices(response, page)
    if resolved is not None:
        response = resolved

    if page.text_items:
        return _dedupe(text_regions(page, response).regions)

    regions = model_regions(response, page.width, page.height, space)
    occupied = [r.bbox for r in regions]
    for region in regions:
        if any(_overlaps(region.bbox, other, tol=0.5) for other in occupied):
            continue
        occupied.append(region.bbox)
    return regions


def _overlaps(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    tol: float = 0.0,
) -> bool:
    return not (
        a[2] < b[0] - tol
        or a[0] > b[2] + tol
        or a[3] < b[1] - tol
        or a[1] > b[3] + tol
    )


# ---------------------------------------------------------------------------
# Рисование
# ---------------------------------------------------------------------------


_FONT = "helv"

# Встроенные в PDF шрифты (Base-14) живут в WinAnsi, где нет кириллицы:
# русский текст заголовка и легенды превращался в квадраты. Поэтому ищем
# системный TTF с полным набором символов; нет его — рисуем латиницей.
_FONT_FILES = (
    "arial.ttf", "segoeui.ttf", "tahoma.ttf", "verdana.ttf", "calibri.ttf",
    "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf",
)
_FONT_DIRS = (
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/liberation"),
    Path("/usr/share/fonts/TTF"),
    Path.home() / ".fonts",
)


@lru_cache(maxsize=1)
def _font() -> tuple[pymupdf.Font, Path] | None:
    """Системный шрифт с кириллицей либо ``None``, если подходящего нет.

    Возвращается пара ``(шрифт, путь)``: объект ``Font`` не помнит, из какого
    файла он загружен, а ``insert_text`` требует путь для встраивания в PDF.
    """
    for folder in _FONT_DIRS:
        for name in _FONT_FILES:
            path = folder / name
            if not path.is_file():
                continue
            try:
                return pymupdf.Font(fontfile=str(path)), path
            except Exception:  # файл повреждён или нечитаем — берём следующий
                continue
    return None


def _text_width(text: str, size: float) -> float:
    loaded = _font()
    if loaded is not None:
        return loaded[0].text_length(text, size)
    return pymupdf.get_text_length(text, fontname=_FONT, fontsize=size)


def _draw_text(
    pdf_page: pymupdf.Page,
    at: tuple[float, float],
    text: str,
    size: float,
    color: tuple[float, float, float],
) -> None:
    """Написать текст, по возможности шрифтом с кириллицей."""
    loaded = _font()
    kwargs: dict = {"fontname": _FONT}
    if loaded is not None:
        _font_obj, path = loaded
        # insert_font не принимает пробелов и точек в имени, а PostScript-имя
        # шрифта вроде «Arial-BoldMT» или «DejaVu Sans» содержит и то, и другое.
        kwargs = {"fontname": "IsokrusSans", "fontfile": str(path)}
    pdf_page.insert_text(at, text, fontsize=size, color=color, **kwargs)


def _plain(text: str) -> str:
    """Привести текст к тому, что умеет нарисовать выбранный шрифт.

    С системным TTF (Arial, Segoe UI) проходит любая кириллица, поэтому
    транслитерация нужна только для запасного пути на Base-14.
    """
    if _font() is not None:
        return _WS_RE.sub(" ", text).strip()
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = _NON_LATIN_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _text_size(page: PageImage) -> float:
    """Кегль подписей разметки, от ширины листа.

    Берётся ширина полного рендера: разметка рисуется на нём, и кегль
    должен быть одинаковым независимо от того, насколько срезали полей.
    """
    width = _png_size(page.full_data)[0]
    return max(11.0, min(26.0, width / 240.0))


def _png_size(png: bytes) -> tuple[int, int]:
    """``(ширина, высота)`` PNG из его заголовка (IHDR), без зависимостей.

    Разметка строится на полном рендере, а его размеры не лежат в
    ``PageImage`` — читать их из заголовка дешевле, чем тащить ещё один
    экземпляр байтов через конвейер.
    """
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Ожидался PNG-поток")
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


def _place_chip(anchor: pymupdf.Rect, size: float, page_rect: pymupdf.Rect) -> pymupdf.Rect:
    """Где поставить плашку с номером — сбоку от рамки, а не поверх цифр.

    Раньше плашка ложилась сверху и закрывала сам размер, ради которого её
    и рисовали. Порядок проб: справа, слева, сверху, снизу; берём первый
    вариант, целиком помещающийся на лист.
    """
    pad = size * 0.3
    w = size * 0.7 + 2 * pad  # хватает на номер до 3 знаков
    h = size * 1.45
    gap = 2.0
    tries = (
        pymupdf.Rect(anchor.x1 + gap, anchor.y0, anchor.x1 + gap + w, anchor.y0 + h),
        pymupdf.Rect(anchor.x0 - gap - w, anchor.y0, anchor.x0 - gap, anchor.y0 + h),
        pymupdf.Rect(anchor.x0, anchor.y0 - gap - h, anchor.x0 + w, anchor.y0 - gap),
        pymupdf.Rect(anchor.x0, anchor.y1 + gap, anchor.x0 + w, anchor.y1 + gap + h),
    )
    for chip in tries:
        if (
            chip.x0 >= page_rect.x0 and chip.x1 <= page_rect.x1
            and chip.y0 >= page_rect.y0 and chip.y1 <= page_rect.y1
        ):
            return chip
    return tries[0] & page_rect  # нигде не поместилось — обрезаем по листу


def _chip(
    pdf_page: pymupdf.Page,
    anchor: pymupdf.Rect,
    number: int,
    color: tuple[float, float, float],
    size: float,
    text: str | None = None,
) -> None:
    """Плашка на рамке: связать её глазами с веткой в таблице.

    Подписью по умолчанию остаётся порядковый номер — у ответов с числами он
    и есть идентификатор ветви. Но в ответе по меткам (``P8``) чисел нет вовсе,
    и порядковый номер не говорит ничего: метка ``P8`` в ответе и на листе одна
    и та же, поэтому она и подписывает рамку, когда она известна.
    """
    chip = _place_chip(anchor, size, pdf_page.rect)
    if chip.is_empty:
        return
    pdf_page.draw_rect(chip, color=None, fill=color)
    caption = text or str(number)
    _draw_text(
        pdf_page,
        (chip.x0 + (chip.width - _text_width(caption, size)) / 2,
         chip.y1 - size * 0.32),
        caption,
        size,
        WHITE,
    )


def _legend(
    pdf_page: pymupdf.Page,
    lines: list[tuple[tuple[float, float, float], str]],
    size: float,
) -> None:
    """Легенда в правом нижнем углу листа.

    Раньше она лежала в левом верхнем углу — «на листах всегда есть поля».
    Обрезка пустых полей (``trim.py``) эти поля убрала, и непрозрачная
    плашка легла бы прямо на чертёж. Правый нижний угол — то место, где на
    изометрии пусто чаще всего, а если не пусто, то прикрывает самый край
    листа, а не начало трассы.
    """
    if not lines:
        return
    pad = size * 0.45
    row = size * 1.55
    width = max(_text_width(_plain(text), size) for _, text in lines) + 2 * pad + size * 2.0
    height = row * len(lines) + 2 * pad
    page_rect = pdf_page.rect
    box = pymupdf.Rect(
        page_rect.x1 - pad - width,
        page_rect.y1 - pad - height,
        page_rect.x1 - pad,
        page_rect.y1 - pad,
    )
    if box.x0 < page_rect.x0 or box.y0 < page_rect.y0:
        # Легенда не помещается в правый нижний угол (мелкий лист или много
        # строк): прижимаем к левому верхнему, но остаёмся внутри страницы.
        box = pymupdf.Rect(
            page_rect.x0, page_rect.y0,
            page_rect.x0 + width, page_rect.y0 + height,
        ) & page_rect
    if box.is_empty:
        return

    pdf_page.draw_rect(box, color=LEGEND_EDGE, fill=LEGEND_BG, width=0.9)
    y = box.y0 + pad + size
    for color, text in lines:
        swatch = pymupdf.Rect(
            box.x0 + pad, y - size * 0.8,
            box.x0 + pad + size * 1.2, y - size * 0.05,
        )
        pdf_page.draw_rect(swatch, color=None, fill=color)
        _draw_text(pdf_page, (swatch.x1 + size * 0.5, y), _plain(text), size, BLACK)
        y += row


def annotate_page(
    page: PageImage,
    response: dict | None,
    output_path: str | Path,
    regions: list[Region] | None = None,
    title: str = "",
    space: str | None = None,
) -> Path:
    """Наложить разметку на изображение листа и сохранить PNG.

    Основой берётся **полный** рендер страницы (``page.full_data``), а не
    обрезанный: разметка смотрится в контексте полей листа, как на бумаге.
    Рамки при этом живут в пикселях обрезанной картинки (текстовый слой
    сдвинут на срез), поэтому на полный лист они переносятся со сдвигом
    на рамку обрезки.

    Изображение встраивается в PDF-страницу размером ровно в пиксели картинки
    (1 pt = 1 px), поэтому координаты подписей из текстового слоя ложатся на
    растр без всякой поправки на масштаб.
    """
    if regions is None:
        regions = collect_regions(response, page, space)

    base = page.full_data
    # Сдвиг рамок: пиксели обрезанной картинки -> пиксели полного листа.
    trim_info = page.meta.get("trim")
    if trim_info and base is not page.data:
        off_x, off_y = trim_info["box"][0], trim_info["box"][1]
        regions = [
            Region(
                bbox=(r.bbox[0] + off_x, r.bbox[1] + off_y,
                      r.bbox[2] + off_x, r.bbox[3] + off_y),
                label=r.label, color=r.color, kind=r.kind, origin=r.origin,
            )
            for r in regions
        ]
    base_width, base_height = _png_size(base)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open()
    try:
        pdf_page = document.new_page(width=base_width, height=base_height)
        pdf_page.insert_image(pdf_page.rect, stream=base)

        size = _text_size(page)
        for number, region in enumerate(regions, start=1):
            rect = pymupdf.Rect(*region.bbox) & pdf_page.rect
            if rect.is_empty:
                continue
            pdf_page.draw_rect(rect, color=region.color, width=max(1.5, size * 0.16))
            _chip(pdf_page, rect, number, region.color, size, region.label)

        counts: dict[tuple, int] = {}
        for region in regions:
            counts[region.color] = counts.get(region.color, 0) + 1

        legend: list[tuple[tuple[float, float, float], str]] = []
        for color, name in (
            (PARENT_COLOR, "обобщённая длина (родитель)"),
            (CHILD_COLOR, "дочерний участок"),
            (COLORS["excluded"], "исключено из расчёта моделью (вложенные)"),
            (MODEL_COLOR, "рамка по bbox модели"),
        ):
            if color in counts:
                legend.append((color, f"{counts[color]} — {name}"))
        if title:
            legend.insert(0, (LEGEND_EDGE, title))
        _legend(pdf_page, legend, size)

        pdf_page.get_pixmap(alpha=False).save(target)
    finally:
        document.close()

    return target


def annotate_results(
    results: "RunResult", output_dir: str | Path, suffix: str = "_annotated"
) -> list[Path]:
    """Разметить все листы прогона. Возвращает список созданных файлов."""
    from .pipeline import RunResult  # локальный импорт: pipeline импортирует llm/extract

    if not isinstance(results, RunResult):
        raise TypeError("Ожидается RunResult")

    out = Path(output_dir) / "annotated"
    created: list[Path] = []
    for result in results.pages:
        if not result.ok:
            continue
        created.append(
            annotate_page(
                result.page,
                result.response,
                out / f"{result.page.label.rsplit('.', 1)[0]}{suffix}.png",
                title=_sheet_title(result),
            )
        )
    return created


def _sheet_title(result: "PageResult") -> str:
    """Заголовок листа разметки: что именно размечено и чего не нашлось.

    Пропуски в заголовке — не украшение: если модель сослалась на число,
    которого нет в текстовом слое листа, это сразу видно на картинке, а не
    выясняется спустя час при сверке сметы.
    """
    stats = annotation_stats(result.page, result.response)
    title = f"Лист {result.page.page_number} · размечено {stats['found']} из {stats['total']}"
    notes: list[str] = []
    if stats.get("excluded"):
        notes.append(f"исключено из расчёта: {stats['excluded']}")
    if stats["duplicated"]:
        notes.append(f"повторов на листе не хватило: {stats['duplicated']}")
    if stats["missing"]:
        notes.append(f"нет на листе: {stats['missing']}")
    if not result.page.text_items:
        notes.append("текстовый слой PDF пуст — размечено по bbox модели")
    return f"{title} · {' · '.join(notes)}" if notes else title


def annotation_stats(page: PageImage, response: dict | None) -> dict:
    """Сводка по листу: сколько чисел размечено точным текстом, сколько нет.

    Ответ приводится к тому же виду, что и при отрисовке. Раньше здесь брался
    сырой ответ, а в ответе по метках чисел нет вовсе — только ``P7``, — и
    заголовок писал «размечено 0 из 0» под картинкой с семнадцатью рамками.
    """
    resolved = resolve_dimension_indices(response, page)
    if resolved is not None:
        response = resolved
    candidates = collect_candidates(response)
    located = text_regions(page, response)
    parents = sum(1 for r in located.regions if r.kind == "parent")
    dropped = sum(1 for r in located.regions if r.kind == "excluded")
    return {
        "total": len(candidates),
        "found": located.found,
        "parent": parents,
        "child": len(located.regions) - parents - dropped,
        "excluded": dropped,
        "from_text": sum(1 for r in located.regions if r.origin == "pdf_text"),
        "duplicated": located.duplicated,
        "missing": len(located.missing),
        "text_items": len(page.text_items),
    }


if TYPE_CHECKING:
    from .pipeline import PageResult
