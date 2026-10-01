"""Удаление с листа того, что размером не является.

Зачем. Модель получает лист и список размеров одновременно, и всё, что на
картинке похоже на размер, тянет её в расчёт: на девятом листе верны все пять
размеров, но рядом лежат квадраты с цифрами «1» и «2», и модель их тоже видит.
Разбор такие числа уже отверг — иначе они не попали бы в список, — но на
картинке они остались.

Что удаляем, по классам, которые видны в векторе:

* ``signs`` — рамка, внутри которой только цифры: номер узла или позиционное
  обозначение. Сюда же входит ``примечание в рамке`` с текстом внутри;
* ``leaders`` — линия, уходящая из такой рамки: она принадлежит знаку, а не
  размеру, и на картинке выглядит как ещё один отрезок с наконечником;
* ``notes`` — опорный знак с буквой и цифрой («О3», «4/3») и рамка примечания;
* ``near_text`` — подписи рядом со знаком в квадрате, которые разбор отверг:
  это и есть «блок со знаком и доп текстом» из второго примера.

О чём модуль не спорит. Попытка отличить данные привязки («X 78500»,
«Y 101175», «Z+ 9695») от размеров по одной геометрии текста не работает:
зазор между префиксом и числом не является признаком. По всем десяти листам
встречаются ``Z+ 10695`` в 5 px (мусор) и ``Z+ 1900`` в 55 px (размер),
``DN50X50 174`` в 75 px (размер) и ``DN50X50 2`` в −105 px (мусор). Единственный
надёжный признак — разбор уже отверг это число как размер. Поэтому ``near_text``
использует именно отказ разбора, а не сходство подписей.

Чего модуль не делает: не трогает рамки подписей найденных размеров. Проверка
на всех десяти листах показала, что центры ни одного размера не попадают
в стираемые области; единственное касание — лист 4, размер ``P16=171``, и там
запас снизу делает область больше подписи, а не накрывающей её.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence

import pymupdf

if TYPE_CHECKING:
    from .dimscan import Contour, Thresholds
    from .extract import PageImage

# Пороги заданы для эталонного листа и приводятся к масштабу каждого листа
# тем же способом, что и пороги разбора (см. ``dimscan._rescaled``).
MAX_SIGN_SIDE_PT = 22.0     # крупнее — уже не знак, а рамка примечания
MIN_SIGN_SIDE_PT = 6.0      # мельче — мусор разбора, а не знак
NOTE_MAX_PT = 260.0         # крупнее — рамка листа, её не трогаем
LEADER_REACH_PT = 3.0       # насколько близко конец линии должен быть к рамке
LEADER_MIN_PT = 2.5         # короче — это штрих, а не выноска
LEADER_MAX_PX = 400.0       # длиннее — это разрезка трубы, а не выноска знака
MARGIN_PT = 2.0             # запас вокруг стираемого, чтобы следы рамки не мыли

FILTERS = ("signs", "leaders", "notes", "rejected")

# Что́ назовём при слиянии перекрывающихся областей. Область рамки со знаком и
# область отвергнутого числа описывают одни и те же пиксели — цифра внутри
# квадрата подходит под оба признака, — и без приоритета метка достаётся той,
# что пришла позже, а счёт «сколько знаков стёрто» становится неправдой.
# Рамка приоритетнее: выноска и число — приложение к ней, а не наоборот.
KIND_PRIORITY = ("signs", "notes", "leaders", "rejected")


def _better_kind(a: str, b: str) -> str:
    """Метка, которая точнее описывает область."""
    left = KIND_PRIORITY.index(a) if a in KIND_PRIORITY else len(KIND_PRIORITY)
    right = KIND_PRIORITY.index(b) if b in KIND_PRIORITY else len(KIND_PRIORITY)
    return a if left <= right else b


@dataclass(frozen=True)
class EraseBox:
    """Область, которую надо закрасить белым, с объяснением почему."""

    box: tuple[float, float, float, float]
    kind: str
    reason: str


@dataclass(frozen=True)
class CleanReport:
    """Что и сколько стёрто — попадает в статистику прогона."""

    boxes: tuple[EraseBox, ...] = ()
    conflicts: tuple[str, ...] = ()

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for item in self.boxes:
            out[item.kind] = out.get(item.kind, 0) + 1
        return out

    @property
    def total(self) -> int:
        return len(self.boxes)


@dataclass
class _Acc:
    """Накопитель областей со списком знаков, из которых они выросли."""

    boxes: list[EraseBox] = field(default_factory=list)
    signs: list[tuple[float, float, float, float]] = field(default_factory=list)


def _grown(box: Sequence[float], pad: float) -> tuple[float, float, float, float]:
    return (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)


def _touches(a: Sequence[float], b: Sequence[float], pad: float) -> bool:
    return not (
        a[2] + pad < b[0] or a[0] - pad > b[2]
        or a[3] + pad < b[1] or a[1] - pad > b[3]
    )


def _covers(box: Sequence[float], item, pad: float = 5.0) -> bool:
    return (
        box[0] - pad <= item.x0 and item.x1 <= box[2] + pad
        and box[1] - pad <= item.y0 and item.y1 <= box[3] + pad
    )


def _center_inside(box: Sequence[float], other: Sequence[float]) -> bool:
    """Лежит ли центр рамки ``other`` внутри ``box``.

    Принимает и подпись текстового слоя, и кортеж координат: рамка размера
    хранится кортежем, а подпись — объектом, и заводить ради одного сравнения
    ещё один тип не хочется.
    """
    x0 = getattr(other, "x0", None)
    target = (x0, other.y0, other.x1, other.y1) if x0 is not None \
        else tuple(other)[:4]
    cx = (target[0] + target[2]) / 2
    cy = (target[1] + target[3]) / 2
    return _center_within(box, cx, cy)


def _center_within(box: Sequence[float], cx: float, cy: float) -> bool:
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _text_inside(box: Sequence[float], items: Iterable, pad: float = 5.0) -> list[str]:
    return [item.text for item in items if _covers(box, item, pad)]


def _kind_of(inner: Sequence[str]) -> str | None:
    """Класс знака по тому, что стоит внутри рамки.

    Только цифры — номер узла или позиционное обозначение; цифра с буквой —
    опорный знак; текст — примечание в рамке. Пустую рамку не трогаем: скорее
    всего это часть графики, а не подпись.
    """
    joined = "".join(inner).strip()
    if not joined:
        return None
    if joined.isdigit():
        return "signs"
    if any(ch.isdigit() for ch in joined):
        return "notes"
    if any(ch.isalpha() for ch in joined):
        return "notes"
    return None


def _segments(
    drawings: Sequence[dict], limits: "Thresholds",
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Отрезки вектора — кандидаты в выноски знаков.

    Допуск по толщине берётся от толщины размерной линии, а не от нуля: иначе в
    кандидаты попадёт вся мелкая обвязка листа. Множитель не 1, а 4, потому что
    выноска знака рисуется заметно жирнее размерной линии — на восьмом листе
    стрелки от опорных знаков вдвое толще размерных, и при строгом допуске они
    отсекались, а вместе с ними оставались на листе стрелки без знаков.
    """
    out: list[tuple[tuple[float, float], tuple[float, float]]] = []
    max_width = 4.0 * (limits.width * limits.width_tol + limits.width)
    for path in drawings:
        width = float(path.get("width") or 0.0)
        if width > max_width:
            continue
        for item in path.get("items") or []:
            if item[0] == "l":
                out.append(((item[1].x, item[1].y), (item[2].x, item[2].y)))
    return out


def _leaders(
    page: "PageImage",
    sign: Sequence[float],
    segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    used: set[int],
    dim_lines: Sequence[Sequence[float]],
) -> list[EraseBox]:
    """Выноски знака: отрезки, упирающиеся одним из концов в рамку.

    Выноска принадлежит знаку, а не размеру, и на картинке выглядит как ещё
    один отрезок с наконечником. Стирать надо и знак, и выноску вместе: если
    стереть только знак, на листе остаётся стрелка без подписи.

    Отбираются отрицанием: выноска — это отрезок у знака, который **не
    совпадает ни с одной размерной линией**. Отбор по толщине не годится: на
    восьмом листе у всех отрезков у знака толщина 0.72 pt, но такую же толщину
    имеют ещё 689 отрезков листа, а размерные линии бывают и 0.48, и 0.96 pt.
    Отбор по длине тоже не годится: у знаков встречаются и 33 px, и 292 px, и
    обе полосы накрывают настоящие размерные линии — при первой попытке стирания
    с листа пропали наконечники всех размеров. Разбор уже знает, где размерные
    линии, и спорить с ним тут незачем.

    Ищутся выноски и у знаков в квадрате, и у опорных: на восьмом листе стрелки
    идут именно от «О3»/«О4».
    """
    to_pixels = page.pixel_box
    reach = page.pt_to_px(LEADER_REACH_PT)
    minimum = page.pt_to_px(LEADER_MIN_PT)
    outer = _grown(sign, reach)
    out: list[EraseBox] = []
    for index, (a, b) in enumerate(segments):
        if index in used:
            continue
        pa = to_pixels((a[0], a[1], a[0], a[1]))
        pc = to_pixels((b[0], b[1], b[0], b[1]))
        length = ((pc[2] - pa[0]) ** 2 + (pc[3] - pa[1]) ** 2) ** 0.5
        if length < minimum or length > LEADER_MAX_PX:
            continue
        if not any(
            outer[0] - reach <= x <= outer[2] + reach
            and outer[1] - reach <= y <= outer[3] + reach
            for x, y in ((pa[0], pa[1]), (pc[2], pc[3]))
        ):
            continue
        span = (
            min(pa[0], pc[2]), min(pa[1], pc[3]),
            max(pa[0], pc[2]), max(pa[1], pc[3]),
        )
        if _overlaps_line(span, dim_lines, page.pt_to_px(2.0)):
            continue
        out.append(
            EraseBox(_grown(span, page.pt_to_px(1.0)), "leaders", "выноска знака")
        )
        used.add(index)
    return out


def _overlaps_line(
    span: Sequence[float], lines: Sequence[Sequence[float]], pad: float,
) -> bool:
    """Лежит ли отрезок на размерной линии.

    Проверяется не совпадение концов, а общая площадь: отрезок бывает задан
    половиной размерной линии или, наоборот, чуть выходит за неё, и сравнение
    концов такое пропускает.
    """
    return any(
        _shared(span, _grown(line, pad)) > 0.3 * _area(span)
        for line in lines if _area(line) > 0
    )


def find_erasures(
    page: "PageImage",
    contours: Sequence["Contour"],
    segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    limits: "Thresholds",
    filters: Sequence[str] = FILTERS,
) -> CleanReport:
    """Области к закрашиванию, в пикселях картинки.

    Работает в координатах PDF (контуры и отрезки там), а отдаёт в пикселях
    обрезанной картинки — в тех же, что и подписи текстового слоя, иначе
    закрасится не то место.
    """
    to_pixels = page.pixel_box
    pad = page.pt_to_px(MARGIN_PT)
    dim_boxes = [mark.bbox for mark in page.dimensions]
    dim_lines = [mark.line_bbox for mark in page.dimensions]
    acc = _Acc()
    used_segments: set[int] = set()

    for contour in contours:
        rect = contour.rect
        side_pt = max(rect.width, rect.height)
        if not (MIN_SIGN_SIDE_PT <= side_pt <= MAX_SIGN_SIDE_PT) and \
                not (NOTE_MAX_PT > side_pt > MAX_SIGN_SIDE_PT):
            continue
        px = to_pixels((rect.x0, rect.y0, rect.x1, rect.y1))
        inner = _text_inside(px, page.text_items)
        kind = _kind_of(inner)
        if kind is None or kind not in filters:
            continue
        acc.boxes.append(
            EraseBox(
                _grown(px, pad), kind,
                f"{'цифра' if kind == 'signs' else 'знак'} "
                f"{' '.join(inner) or 'пусто'} в рамке",
            )
        )
        if kind == "signs":
            acc.signs.append(px)
        if "leaders" in filters:
            acc.boxes.extend(
                _leaders(page, px, segments, used_segments, dim_lines)
            )

    if "rejected" in filters:
        acc.boxes.extend(_rejected_numbers(page, dim_boxes, pad))

    # Последний рубеж: область, под которую попал центр подписи размера, не
    # закрашивается вовсе. Проверка идёт по исходным областям, а не по слитым:
    # слияние склеивает соседние знаки в один прямоугольник, и проверка на нём
    # нашла бы конфликт там, где ни одна отдельная область размер не трогает.
    kept: list[EraseBox] = []
    conflicts: list[str] = []
    for box in acc.boxes:
        hit = next(
            (f"{mark.line_id}={mark.label}" for mark in page.dimensions
             if _center_inside(box.box, mark.bbox)),
            None,
        )
        if hit is None:
            kept.append(box)
        else:
            conflicts.append(f"{hit} под «{box.kind}»")
    return CleanReport(
        tuple(_merge(kept)), tuple(dict.fromkeys(conflicts)),
    )


def _rejected_numbers(
    page: "PageImage",
    dim_boxes: Sequence[Sequence[float]],
    pad: float,
) -> list[EraseBox]:
    """Числа, которые разбор отверг как размеры.

    Вместе со знаками в квадратах это и есть «блоки с квадратиками и доп
    текстом»: «X 78500», «Y 101175», «Z+ 9695» из блока привязки, «78500» и
    «101175» рядом с номером чертежа. На листе они выглядят ровно так же, как
    размеры, и модель тянет их в сумму.

    Стирать их безопасно именно потому, что разбор их отверг: иначе они были бы
    в списке размеров, и на десяти листах эталонная сумма достигается подмножеством
    именно принятых чисел. Чего не хватает в списке — того не нужно и считать.
    Это единственное правило отсева, у которого есть проверяемое основание:
    попытка отличить «Z+ 10695» от «Z+ 1900» по расстоянию или по буквам не
    работает, зазоры у них пересекаются.

    Подписи с буквами («DN50X50», «4137_14-005-SW-1001») не трогаются: они не
    бывают размерами, но и на ответ модели не влияют, а стирание длинной строки
    задевает соседнюю графику.
    """
    out: list[EraseBox] = []
    for item in page.text_items:
        if not item.text.isdigit():
            continue
        if any(_covers(dimension, item, pad=6.0) for dimension in dim_boxes):
            continue
        out.append(
            EraseBox(
                _grown((item.x0, item.y0, item.x1, item.y1), pad),
                "rejected", f"число {item.text} отвергнуто разбором",
            )
        )
    return out


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _shared(a: Sequence[float], b: Sequence[float]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, width) * max(0.0, height)


def _merge(boxes: Sequence[EraseBox]) -> list[EraseBox]:
    """Слить сильно перекрывающиеся прямоугольники, сохранив их разметку.

    Склеиваются не просто соприкасающиеся, а те, у которых общая площадь — не
    меньше половины меньшей из них. Иначе слияние перескакизывает с одного знака
    на соседний: четыре подписи рядом дают в цепочке прямоугольник во много раз
    больше любой из них, и такой прямоугольник уже накрывает размеры, которых ни
    одна исходная область не касалась.

    При объединении остаётся метка, которая точнее описывает область: см.
    :func:`_better_kind`.
    """
    result: list[EraseBox] = []
    for box in boxes:
        current = box
        joined = True
        while joined:
            joined = False
            for other in list(result):
                if _shared(current.box, other.box) < 0.5 * min(
                    _area(current.box), _area(other.box)
                ):
                    continue
                current = EraseBox(
                    (
                        min(current.box[0], other.box[0]),
                        min(current.box[1], other.box[1]),
                        max(current.box[2], other.box[2]),
                        max(current.box[3], other.box[3]),
                    ),
                    _better_kind(current.kind, other.kind),
                    current.reason if current.kind == _better_kind(
                        current.kind, other.kind) else other.reason,
                )
                result.remove(other)
                joined = True
                break
        result.append(current)
    return result


def apply_erasures(
    data: bytes, width: int, height: int, boxes: Sequence[Sequence[float]],
) -> bytes:
    """Закрасить области белым, вернуть PNG того же размера.

    Закрашивание идёт через новую страницу с той же картинкой: так же, как
    строилось примечание, и по той же причине — пиксели и координаты остаются
    от одной матрицы.
    """
    if not boxes:
        return data
    document = pymupdf.open()
    try:
        sheet = document.new_page(width=width, height=height)
        sheet.insert_image(sheet.rect, stream=data)
        for box in boxes:
            sheet.draw_rect(pymupdf.Rect(*box), color=None, fill=(1, 1, 1), width=0)
        return sheet.get_pixmap(alpha=False).tobytes("png")
    finally:
        document.close()