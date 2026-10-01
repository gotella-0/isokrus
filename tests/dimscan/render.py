"""Подсветка размерных отрезков: три варианта одной картинки.

Все три рисуют одно и то же — отрезки, найденные в ``detect.py``, — и
отличаются только тем, насколько сильно трогают исходный чертёж. Векторная
геометрия тут единственный источник правды: растровый путь (top-hat по
тонким линиям) был отброшен, и вот почему. На изометрии осевая линия трубы
пунктиром нарисована той же толщиной, что и размерная, — в растре они
неразличимы, и top-hat честно подсвечивал осевую вместо размеров. Различить
их в пикселях можно только по наконечникам-треугольникам, но на этом листе
размерная линия в нескольких местах смыкается со своими же продолжениями и
выносными, сливаясь с ними в одну компоненту связности; на разборе такой
компоненты на отрезки результат зависел от порогов. В PDF же тот же признак
читается точно: наконечник — отдельный залитый путь, а не пиксельная бугорка.

``vector``  Чертёж как есть, поверх — жирные цветные отрезки, обведённые
            светлой подложкой, с точками на концах и рамками вокруг чисел.
            Контрольный вариант: если подсветка легла точно на «2160»,
            геометрия прочитана верно.
``ghost``   Чертёж обесцвечен до светло-серого, размеры остаются цветными.
            Убирает шум координат и диаметров — остаётся ровно то, о чём
            спор: «сумма этих отрезков».
``nodes``   Как ``ghost``, плюс номер каждого отрезка плашкой. Нужен, чтобы
            глазами сопоставить отрезок и запись в ``segments.json``.

Все варианты пишут PNG в одной системе координат — пункты PDF, — поэтому
их можно сравнивать друг с другом попиксельно.

Выноски-указатели во всех трёх рисуются **вдвое тоньше** отрезков и без
точек на концах: длина выноски не является размером, и видеть её наравне с
измеренными отрезками нельзя. На листе 3 это «13» — зазор между фланцами,
начертанный одной стрелкой.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pymupdf

from isokrus.dimscan import DimLine
from .layout import Layout, Placement, layout_for_page

RGB = tuple[float, float, float]
Pt = tuple[float, float]

# Палитра: максимально различимые между собой и не совпадающие с чёрным
# чертежом. Порядок фиксирован — один и тот же отрезок получает один и тот
# же цвет на всех листах, иначе сравнивать листы глазами бессмысленно.
PALETTE: list[RGB] = [
    (0.90, 0.10, 0.10),  # красный
    (0.05, 0.45, 0.90),  # синий
    (0.00, 0.65, 0.25),  # зелёный
    (0.95, 0.55, 0.00),  # оранжевый
    (0.60, 0.15, 0.80),  # фиолетовый
    (0.00, 0.70, 0.70),  # бирюзовый
    (0.85, 0.20, 0.60),  # маджента
    (0.45, 0.35, 0.10),  # охра
    (0.20, 0.60, 0.15),  # оливковый
    (0.55, 0.60, 0.10),  # лайм
]


def color_for(index: int) -> RGB:
    return PALETTE[(index - 1) % len(PALETTE)]


# Во сколько раз выноска-указатель тоньше отрезка. Меньше единицы, а не
# больше: на листе 3 выноска «13» сама по себе тонкая, и утолщать её не
# надо — надо, чтобы она не читалась как измеренный отрезок.
_POINTER_WIDTH = 0.45


def _tint(color: RGB, amount: float) -> RGB:
    """Подмешать цвет к белому. ``amount=0`` — чистый цвет, ``1`` — белый."""
    return tuple(c + (1.0 - c) * amount for c in color)  # type: ignore[return-value]


def _line_width_for(page: pymupdf.Page, mult: float) -> float:
    """Толщина подсветки в пунктах — по размеру листа, а не константой.

    Один и тот же пункт на листе A1 и на листе, вписанном в A4, даёт
    разный охват картинки. Считаем от меньшей стороны листа.
    """
    return max(0.8, min(page.rect.width, page.rect.height) / 170.0) * mult


def _stroke_for(page: pymupdf.Page, line: DimLine, mult: float) -> float:
    """Толщина линии конкретного отрезка — с поправкой на его длину.

    Постоянная толщина годится для длинных размеров и ломает короткие:
    на листе 10 отрезок «134» длиной 26 pt при толщине 5 pt превращался в
    сплошной прямоугольник, в котором не видно ни самого отрезка, ни его
    числа. Поэтому толщина ограничена долей длины: короткий отрезок
    рисуется тонко и остаётся отрезком.
    """
    return min(_line_width_for(page, mult), line.length_pt * 0.14)


# ---------------------------------------------------------------------------
# Подготовка страницы
# ---------------------------------------------------------------------------


def _open_pdf(pdf_path: str) -> pymupdf.Document:
    return pymupdf.open(pdf_path)


def _scratch_page(source: pymupdf.Page, out: pymupdf.Document) -> pymupdf.Page:
    """Чистая страница того же размера в пунктах — под рисуемую подсветку.

    Заводится в **отдельном** документе, а не в исходном: ``new_page`` в
    исходнике сдвигает нумерацию, а ``delete_page`` инвалидирует уже
    загруженные объекты страниц — после первой итерации ``source` становится
    битым и рендер падает на «invalid null reference». Отдельный документ
    держит исходник нетронутым, что заодно позволяет не бояться, что
    перерисовка испортит файл на диске.
    """
    return out.new_page(width=source.rect.width, height=source.rect.height)


def _draw_ghost(page: pymupdf.Page, target: pymupdf.Page, dpi: int, strength: float) -> None:
    """Перенести исходный лист на новую страницу, обесцветив его.

    ``strength`` — насколько сильно гасим чертёж: 1.0 оставляет всё как
    есть, 0.75 уводит линии в светло-серый. Гасим растр, а не вектор:
    иначе пришлось бы перебирать сотни путей и менять цвета у каждого.
    """
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    # Заливка белым, пропорциональное уменьшение «чернил» до серого.
    faded = 255.0 - (255.0 - image[:, :, :3].astype(np.float32)) * strength
    faded = np.clip(faded, 0, 255).astype(np.uint8)
    rgba = pymupdf.Pixmap(pymupdf.csRGB, faded.shape[1], faded.shape[0], faded.tobytes(), False)
    # Растр в пунктах листа, а не в пикселях: иначе картинка легла бы в угол.
    target.insert_image(target.rect, pixmap=rgba, keep_proportion=False)


def _draw_segments(
    target: pymupdf.Page,
    lines: Sequence[DimLine],
    *,
    color: RGB | None,
    width_mult: float,
) -> None:
    """Нарисовать отрезки. ``color=None`` — цвет по порядковому номеру.

    Рисуем в два прохода: сначала **все** подложки, потом **все** линии. Если
    идти по отрезкам целиком, подложка одного закрывает линию соседнего, а на
    листе 2 размеры идут внахлёст — и подсветка рассыпалась.

    Толщина линии берётся из :func:`_stroke_for` — с поправкой на длину
    отрезка. Постоянная толщина годится для длинных размеров и ломает
    короткие: «134» на листе 10 при толщине 5 pt превращался в сплошной
    прямоугольник, и не было видно ни отрезка, ни его числа.

    **Выноски-указатели** (``kind="pointer"``) рисуются вдвое тонжее и без
    точек на концах. Это не украшение: выноска не измеряет, а указывает на
    зазор, и её длина ничего не значит. На картинке она должна читаться как
    «здесь 13 мм», а не как «отрезок 13 мм», иначе следующий человек снова
    решит, что длина выноски и есть размер.
    """
    for line in lines:
        target.draw_line(
            pymupdf.Point(*line.a), pymupdf.Point(*line.b),
            color=color or color_for(line.index),
            width=_stroke_for(target, line, width_mult)
            * (1.0 if line.measured else _POINTER_WIDTH),
        )


def _draw_endpoint_dots(
    target: pymupdf.Page, lines: Sequence[DimLine], *, radius_mult: float = 0.7
) -> None:
    """Точки на концах отрезков.

    На изометрии отрезок без концов читается как часть трубы: конец упирается
    в фланец, и глаз не видит, где он кончился. Точка фиксирует границу.

    Радиус — доля от длины отрезка, а не константа. На листе 10 отрезок «134»
    длиной 26 pt при постоянном радиусе 4.4 pt превращался в две жирные
    кляксы, между которыми не оставалось ни линии, ни её числа: выглядело
    как «толстые указатели и короткая линия».

    У выносок точек нет: наконечник у них уже нарисован на чертеже, а
    свободный конец упирается в подпись — точка попала бы прямо на цифру.
    """
    base = _line_width_for(target, radius_mult)
    for line in lines:
        if not line.measured:
            continue
        radius = min(base, line.length_pt * 0.08)
        # Круг, а не прямоугольник: острие стрелки на чертёже упирается в
        # конец отрезка под углом, и квадратная точка читалась бы как
        # элемент конструкции, а не как конец подсвеченного отрезка.
        for point in (line.a, line.b):
            target.draw_circle(
                pymupdf.Point(*point), radius,
                color=color_for(line.index), fill=color_for(line.index), width=0,
            )


def _draw_value_marks(
    target: pymupdf.Page, lines: Sequence[DimLine], *, font_size: float
) -> None:
    """Подсветить само число размера — как маркером в Word.

    Именно маркером, а не плашкой: заливка цветом закрывала напечатанную
    цифру (исходный лист приходит растром, и цифры уже впечатаны в картинку),
    и на листе 10 рядом с отрезком «134» оставалась пустая рамка. Здесь
    рисуется полупрозрачная полоса **под** цифрами, начертание сохраняется и
    остаётся читаемым.

    Прозрачность задаётся через ``fill_opacity`` — и это единственный
    непрозрачный примитив, который работает у прямоугольника в PyMuPDF.
    Раньше он применялся к линии, а не к заливке, из-за чего подложка
    выходила тремя параллельными полосами.

    Рамка берётся из текстового слоя PDF, поэтому подсветка гарантированно
    попадает на нужное число, а не на соседнее «X 48300».
    """
    for line in lines:
        if not line.label_bbox:
            continue  # подписи нет — подсвечивать нечего
        x0, y0, x1, y1 = line.label_bbox
        pad = font_size * 0.16
        target.draw_rect(
            pymupdf.Rect(x0 - pad, y0 - pad * 0.5, x1 + pad, y1 + pad * 0.5),
            color=None, fill=_tint(color_for(line.index), 0.55),
            fill_opacity=0.5,
        )


def _draw_numbers(
    target: pymupdf.Page,
    source: pymupdf.Page,
    lines: Sequence[DimLine],
    *,
    font_size: float,
) -> list[Placement]:
    """Имена отрезков P1, P14 — связь с ``segments.json``.

    Пишется с буквой: рядом стоит подпись размера «134», и без префикса обе
    надписи выглядели бы как числа — именно из-за этого размер 134 на
    листе 10 однажды казался неразмеченным.

    Текстом с белым контуром, а не залитой плашкой: плашка была вдвое
    крупнее самой надписи и закрывала чертёж, а на листе 8 номера вообще
    перекрывали подписи соседних размеров.

    Позиции считает :mod:`layout`: имена не должны наезжать ни на текст
    листа, ни на подписи размеров, ни друг на друга. Раньше здесь стояло
    четыре попытки сдвига по нормали и вдоль отрезка — их не хватало, на
    листе 10 имена всё равно налезали на текст и друг на друга.
    """
    layout = layout_for_page(source, lines, font_size, _text_length)
    for placement in layout.placements:
        base_x, base_y = placement.origin
        # Сначала белый текст чуть крупнее, затем цветной поверх: получается
        # тонкий белый контур, и надпись читается на любом фоне, не закрывая
        # ничего вокруг.
        target.insert_text(
            (base_x, base_y), placement.name,
            fontsize=font_size * 1.12, color=(1, 1, 1), fontname="hebo",
        )
        target.insert_text(
            (base_x, base_y), placement.name,
            fontsize=font_size, color=color_for(placement.line_index), fontname="hebo",
        )
    return layout.placements


def _text_length(text: str, size: float) -> float:
    """Ширина текста. Вынесена, потому что нужна и при раскладке, и при сдвиге."""
    return pymupdf.get_text_length(text, fontname="hebo", fontsize=size)


def _finish(target: pymupdf.Page, dpi: int) -> pymupdf.Pixmap:
    return target.get_pixmap(dpi=dpi, alpha=False)


# ---------------------------------------------------------------------------
# Варианты
# ---------------------------------------------------------------------------


def render_vector(
    pdf_path: str,
    out_dir: str | Path,
    lines_by_page: dict[int, Sequence[DimLine]],
    *,
    dpi: int = 200,
    width_mult: float = 1.0,
    endpoints: bool = True,
) -> Path:
    """Вариант 1: исходный лист целиком, поверх — отрезки, маркер и номера.

    Исходник не гасится: это контрольный вариант. Если подсветка легла
    точно на «2160», значит геометрия из PDF прочитана верно, и дальше
    вопрос уже к подаче, а не к детектору.

    Отрезки рисуются так же, как в ``ghost``, — без подложки и без рамок.
    Раньше под каждым отрезком была широкая светлая полоса «в тень», и на
    листе 2 эти полосы накрывали собой цифры соседних размеров: значение
    становилось не видно ровно там, ради чего всё и затевалось.

    Значение размера подсвечивается маркером, а номер ставится аккуратным
    текстом с белым контуром рядом с отрезком.
    """
    return _render_faded(
        pdf_path, out_dir, lines_by_page, dpi, fade=1.0, width_mult=width_mult,
        endpoints=endpoints, suffix="vector", marks=True, numbers=True,
    )


def render_ghost(
    pdf_path: str,
    out_dir: str | Path,
    lines_by_page: dict[int, Sequence[DimLine]],
    *,
    dpi: int = 200,
    fade: float = 0.22,
    width_mult: float = 1.0,
    endpoints: bool = True,
) -> Path:
    """Вариант 2: чертёж приглушён, отрезки цветные, цифры не тронуты."""
    return _render_faded(
        pdf_path, out_dir, lines_by_page, dpi, fade, width_mult,
        endpoints=endpoints, suffix="ghost",
    )


def render_nodes(
    pdf_path: str,
    out_dir: str | Path,
    lines_by_page: dict[int, Sequence[DimLine]],
    *,
    dpi: int = 200,
    fade: float = 0.22,
    width_mult: float = 1.0,
) -> Path:
    """Вариант 3: приглушённый чертёж, отрезки, маркер на цифрах и номера.

    Отличие от ``vector`` — только приглушение исходника. Маркер на цифрах
    и номера в обоих: по ``segments.json`` отрезок N известен, и на листе
    он же помечен N, иначе список и картинку приходится сопоставлять вручную.
    """
    return _render_faded(
        pdf_path, out_dir, lines_by_page, dpi, fade, width_mult,
        endpoints=True, suffix="nodes", marks=True, numbers=True,
    )


def _render_faded(
    pdf_path: str,
    out_dir: str | Path,
    lines_by_page: dict[int, Sequence[DimLine]],
    dpi: int,
    fade: float,
    width_mult: float,
    *,
    endpoints: bool,
    suffix: str,
    marks: bool = False,
    numbers: bool = False,
) -> Path:
    """Общая часть всех трёх вариантов.

    Они делают одно и то же и расходятся только флагами, поэтому держать
    три почти одинаковые копии означало бы, что правку в геометрии
    придётся вносить трижды — и однажды забыть.

    Порядок отрисовки фиксирован и важен: маркер на цифрах ложится **до**
    имён и точек, иначе имя перекрыло бы подсвеченное значение.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    document = _open_pdf(pdf_path)
    scratch_doc = pymupdf.open()
    try:
        for number, lines in sorted(lines_by_page.items()):
            source = document.load_page(number - 1)
            scratch = _scratch_page(source, scratch_doc)
            _draw_ghost(source, scratch, dpi, strength=fade)
            _draw_segments(scratch, lines, color=None, width_mult=width_mult)
            if marks:
                _draw_value_marks(scratch, lines, font_size=_line_width_for(scratch, 2.6))
            if endpoints:
                _draw_endpoint_dots(scratch, lines)
            if numbers:
                # Раскладке нужен исходный лист: из его текстового слоя
                # берутся подписи, которых имена обязаны избегать.
                _draw_numbers(
                    scratch, source, lines, font_size=_line_width_for(scratch, 2.2)
                )
            _finish(scratch, dpi).save(out_root / f"p{number:02d}_{suffix}.png")
            scratch_doc.delete_page(scratch.number)
    finally:
        scratch_doc.close()
        document.close()
    return out_root
