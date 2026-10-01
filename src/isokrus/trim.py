"""Обрезка пустых полей листа — аналог ``str.strip()`` для растров.

Зачем модуль: лист изометрии печатается на формате шире и выше, чем сама
трасса, и по краям остаётся чистая бумага. На листах из ``02_*.pdf`` это
47–63% площади. Vision-модель всё равно подгоняет картинку под свой лимит
входа, поэтому пустое поле не стоит ничего, а съедает разрешение: на
обрезанном листе чертёж получает примерно на 25% больше пикселей на
миллиметр, и размерные подписи читаются.

Как ищется рамка: никаких заранее заданных полей. Берётся уменьшенная копия
растра, цвет пустого места определяется по левому верхнему углу страницы (у
белого листа это 255, у серого скана — серый), затем строки и столбцы
сканируются от краёв внутрь до первого нарисованного. Столбцы проверяются
только внутри найденных строк, поэтому проверка точная, а не эвристическая.

Координаты текстового слоя после обрезки не двигаются вручную: они
извлекаются матрицей, уже сдвинутой на величину среза (см.
``textmap.shifted_matrix``). Поэтому подпись в промте и её начертание на
картинке остаются в одной системе координат.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pymupdf

from . import config

# Полуоткрытая рамка в пикселях: (x0, y0, x1, y1), правая и нижняя границы
# не входят в рамку — как в ``pymupdf.IRect``.
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class TrimResult:
    """Что получилось после обрезки и почему."""

    pixmap: pymupdf.Pixmap
    box: Box
    full_size: tuple[int, int]  # размер листа до обрезки
    background: int
    factor: int
    empty: bool = False       # на листе вообще ничего не нарисовано
    skipped: bool = False     # результат вышел меньше min_side — лист оставлен целиком
    seconds: float = 0.0

    @property
    def area_ratio(self) -> float:
        """Доля площади листа, оставшаяся после обрезки."""
        full_w, full_h = self.full_size
        if self.skipped or not full_w or not full_h:
            return 1.0
        box_w = self.box[2] - self.box[0]
        box_h = self.box[3] - self.box[1]
        return (box_w * box_h) / (full_w * full_h)

    def as_dict(self) -> dict:
        return {
            "box": list(self.box),
            "from": list(self.full_size),
            "to": [self.pixmap.width, self.pixmap.height],
            "area_ratio": round(self.area_ratio, 4),
            "background": self.background,
            "probe_factor": self.factor,
            "empty": self.empty,
            "skipped": self.skipped,
        }


def _probe(pixmap: pymupdf.Pixmap, probe_side: int) -> tuple[pymupdf.Pixmap, int]:
    """Уменьшенная копия растра и коэффициент уменьшения.

    Копия нужна обязательно: ``shrink`` сжимает растр на месте, а исходный
    пиксель листа ещё понадобится для точной обрезки.
    """
    small = pymupdf.Pixmap(pixmap.colorspace, pixmap.irect, False)
    small.copy(pixmap, pixmap.irect)
    factor = 1
    while small.width > probe_side:
        small.shrink(1)
        factor *= 2
    return small, factor


def _ink_table(ink_level: int) -> bytes:
    """Таблица ``bytes.translate``: 1 для «нарисованного», 0 для фона."""
    return bytes(1 if value < ink_level else 0 for value in range(256))


def content_box(
    pixmap: pymupdf.Pixmap,
    probe_side: int,
    min_run: int,
    noise: int,
    min_ink: int,
) -> tuple[Box, dict]:
    """Рамка нарисованного содержимого в координатах исходного растра.

    Пустой лист даёт рамку во весь лист — обрезать нечего.

    Пороги и их цена:

    ``noise``
        на сколько уровней пиксель должен быть темнее фона, чтобы считаться
        нарисованным. У серого скана фон гуляет на несколько уровней, без
        запаса обрезка не сработала бы вовсе.
    ``min_ink``
        сколько тёмных пикселей должно быть в строке или столбце, чтобы они
        считались содержимым. При ``1`` мы не режем ничего, что хоть чем-то
        похоже на содержимое; больше единицы нужно для сканов, где одиночная
        тёмная пылинка иначе отменяет обрезку всего листа.
    ``min_run``
        поля тоньше этого числа клеток зонда — это край, а не поле.

    Обратная сторона ``noise``: графика светлее ``фон - noise`` (248..254 на
    белом фоне) считается пустотой. На векторных чертежах CAD её нет.
    """
    small, factor = _probe(pixmap, probe_side)
    background = min(small.pixel(0, 0))
    ink_level = background - noise
    table = _ink_table(ink_level)
    # bytes, а не memoryview: у memoryview нет translate, а считать тёмные
    # пиксели нужно на скорости C, а не циклом на Python.
    buf = bytes(small.samples_mv)
    n, stride, w, h = small.n, small.stride, small.width, small.height
    row_bytes = w * n  # без хвоста выравнивания: он не часть картинки

    def row_blank(y: int) -> bool:
        chunk = buf[y * stride : y * stride + row_bytes]
        if min(chunk) >= ink_level:  # частый случай — строка пустая
            return True
        return chunk.translate(table).count(1) < min_ink

    def col_blank(x: int, y0: int, y1: int) -> bool:
        found = 0
        for y in range(y0, y1):
            off = y * stride + x * n
            if min(buf[off : off + n]) < ink_level:
                found += 1
                if found >= min_ink:
                    return False
        return True

    # 1) Строки: сверху и снизу, до первой «нарисованной».
    top = 0
    while top < h and row_blank(top):
        top += 1
    if top >= h:
        # Пустой лист: без этой проверки получилась бы вырожденная рамка
        # в углу, и модель получила бы картинку в несколько пикселей.
        return (0, 0, pixmap.width, pixmap.height), {
            "background": background,
            "factor": factor,
            "empty": True,
        }

    bottom = h - 1
    while bottom > top and row_blank(bottom):
        bottom -= 1
    y0, y1 = top, bottom + 1  # полуоткрытый интервал строк с содержимым

    # 2) Столбцы: только внутри найденных строк. Вне них содержимого нет по
    #    построению, поэтому проверка точная, а не эвристическая.
    left = 0
    while left < w and col_blank(left, y0, y1):
        left += 1
    right = w - 1
    while right > left and col_blank(right, y0, y1):
        right -= 1
    x0, x1 = left, right + 1

    # 3) Клетки зонда накрывают factor x factor пикселей и лежат в рамке
    #    целиком, поэтому нарисованный пиксель обрезан быть не может, даже
    #    если усреднение при уменьшении сделал его бледнее порога.
    x0, y0 = max(0, x0 - min_run + 1), max(0, y0 - min_run + 1)
    x1, y1 = min(w, x1 + min_run - 1), min(h, y1 + min_run - 1)
    box = (
        x0 * factor,
        y0 * factor,
        min(pixmap.width, x1 * factor),
        min(pixmap.height, y1 * factor),
    )
    return box, {"background": background, "factor": factor, "empty": False}


def expand(
    box: Box,
    pad: int,
    width: int,
    height: int,
    min_side: int,
) -> Box:
    """Рамка + белый запас. Слишком мелкий результат — лист целиком.

    Запас нужен, чтобы у края среза не оказалась вплотную графика: модель
    (и человек на разметке) должны видеть, где кончается лист.
    """
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return (0, 0, width, height)
    return (x0, y0, x1, y1)


def trim(
    pixmap: pymupdf.Pixmap,
    pad: int | None = None,
    min_side: int | None = None,
    noise: int | None = None,
    min_ink: int | None = None,
    min_run: int | None = None,
    probe_side: int | None = None,
) -> TrimResult:
    """Обрезать пустые поля. ``None`` означает «взять из ``config``»."""
    pad = config.TRIM_PAD_PX if pad is None else pad
    min_side = config.TRIM_MIN_SIDE if min_side is None else min_side
    noise = config.TRIM_NOISE if noise is None else noise
    min_ink = config.TRIM_MIN_INK if min_ink is None else min_ink
    min_run = config.TRIM_MIN_RUN if min_run is None else min_run
    probe_side = config.TRIM_PROBE_SIDE if probe_side is None else probe_side

    started = time.perf_counter()
    box, info = content_box(pixmap, probe_side, min_run, noise, min_ink)
    full_size = (pixmap.width, pixmap.height)
    full_box = expand(box, pad, pixmap.width, pixmap.height, min_side)
    skipped = full_box == (0, 0, pixmap.width, pixmap.height)
    if skipped:
        return TrimResult(
            pixmap=pixmap,
            box=full_box,
            full_size=full_size,
            background=info["background"],
            factor=info["factor"],
            empty=info["empty"],
            skipped=True,
            seconds=time.perf_counter() - started,
        )

    irect = pymupdf.IRect(*full_box)
    cropped = pymupdf.Pixmap(pixmap.colorspace, irect, False)
    cropped.copy(pixmap, irect)
    return TrimResult(
        pixmap=cropped,
        box=full_box,
        full_size=full_size,
        background=info["background"],
        factor=info["factor"],
        empty=False,
        skipped=False,
        seconds=time.perf_counter() - started,
    )
