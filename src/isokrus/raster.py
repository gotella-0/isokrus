"""Растры как массивы, без копий.

Почему этот модуль есть. ``Pixmap.samples`` в PyMuPDF **копирует весь буфер
при каждом обращении**: на листе при 200 dpi это 23 МБ и 16 мс на одно
свойство. Код, который читает ``samples`` в цикле по строкам, а строк в
вырезе сотни, нарезает полмиллиона копий — и десять листов проходят минутами
вместо секунд. Именно так и вышло: один прогон съел 1124 секунды CPU.

``samples_mv`` — тот же буфер без копии, и ``numpy.frombuffer`` поверх него
не выделяет ничего нового. Дальше сравнения и закрашивание идут как обычно,
по массиву, а не по пикселям из Python.
"""

from __future__ import annotations

import numpy as np
import pymupdf


class Raster(np.ndarray):
    """Представление буфера пикселя, которое держит сам пиксель живым.

    Массив смотрит в память, принадлежащую :class:`pymupdf.Pixmap`, и ничего
    её не удерживает. Если пиксель собрал мусор, массив остаётся верным на вид
    и читает освобождённую память — а на следующей же строке процесс падает с
    ``ACCESS_VIOLATION`` без всякой связи с правкой картинки. Поэтому пиксель
    хранится прямо в объекте: ``page.get_pixmap()`` можно передать в
    :func:`as_rgb` без временной переменной, и это будет безопасно.
    """

    pixmap: pymupdf.Pixmap

    def __new__(cls, pixmap: pymupdf.Pixmap) -> Raster:
        view = np.frombuffer(pixmap.samples_mv, dtype=np.uint8)
        if pixmap.stride == pixmap.width * pixmap.n:
            shaped = view.reshape(pixmap.height, pixmap.width, pixmap.n)
        else:
            # У растра с альфой или с выравниванием строки между строками есть
            # заполнитель, и без среза массив не разложится.
            rows = view.reshape(pixmap.height, pixmap.stride)
            shaped = rows[:, :pixmap.width * pixmap.n]
            shaped = shaped.reshape(pixmap.height, pixmap.width, pixmap.n)
        self = shaped.view(cls)
        self.pixmap = pixmap
        return self

    def __array_finalize__(self, source) -> None:
        # Срез или транспонирование — по-прежнему тот же пиксель.
        self.pixmap = getattr(source, "pixmap", None)


def as_rgb(pixmap: pymupdf.Pixmap) -> Raster:
    """Растр как массив ``(высота, ширина, каналы)`` без копии.

    Возвращается **представление** буфера, а не копия: правки массива видны в
    пикселе. Поэтому писать в него можно только через :func:`flush`.
    """
    return Raster(pixmap)


def flush(pixmap: pymupdf.Pixmap, array: np.ndarray) -> None:
    """Записать массив обратно в пиксель."""
    pixmap.samples_mv[:] = np.ascontiguousarray(array).tobytes()


def crop_bytes(
    pixmap: pymupdf.Pixmap,
    array: np.ndarray,
    rect: pymupdf.Rect,
) -> bytes:
    """PNG выреза из уже загруженного растра.

    Вырез берётся срезом массива, а не построчным копированием: набор строк
    в Python — это ровно то, ради чего модуль и написан.
    """
    box = pymupdf.Rect(rect)
    height, width = array.shape[:2]
    x0, y0 = max(0, int(box.x0)), max(0, int(box.y0))
    x1, y1 = min(width, -(-int(box.x1) // 1)), min(height, -(-int(box.y1) // 1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return b""
    piece = np.ascontiguousarray(array[y0:y1, x0:x1])
    return pymupdf.Pixmap(
        pixmap.colorspace, piece.shape[1], piece.shape[0], piece.tobytes(), False
    ).tobytes("png")
