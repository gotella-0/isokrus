"""Чтение подписи размера в кропе вокруг отрезка.

Зачем это, если есть полностраничный режим. Полностраничный на листе 8
находит 2 из 9 подписей размеров — и это на **идеальном** векторном рендере
500 dpi, где каждая цифра занимает сотую долю пункта. Причина в том, что 73
из 114 подписей на этих листах повёрнуты на ±30°, а 22 стоят вертикально.
Tesseract разбирает растр в одной ориентации: текст, повёрнутый на 30°, для
него не текст, а набор вертикальных черт, и он честно отвечает «не знаю
такого». Определение угла страницы тут не помогает: подписи на одном листе
повёрнуты в четыре разные стороны, и общей стороны у страницы нет.

Кроп решает задачу иначе. У каждого размерного отрезка своя система
координат: вдоль отрезка и по нормали. Повернув кроп на угол отрезка,
получаем подпись строго горизонтальной — и это верно при любом перекосе
листа, потому что угол берётся у отрезка, а не у страницы.

Двухшаговое чтение. Одного прохода мало, и на листе 8 это видно буквально.
Кроп вокруг отрезка на 500 dpi — это 1136x825 пикселей, и кроме подписи
там лежат: номера узлов в рамках, кусок пунктирной осевой трубы, стрелки,
подписи соседних размеров и то, что осталось от затирания. Движок,
прочитавший такой кроп целиком, возвращает с десяток строк, из которых
подпись — одна, а остальные нужно отбросить, и отбрасывать их нечем: без
рамок непонятно, где именно движок увидел «03» в рамке и где «6000».

Поэтому читаем дважды:

1. **Грубо.** Кроп целиком, ``psm 11``, без белого списка. Находим все
   блоки-кандидаты с их рамками и уверенностью.
2. **Точно.** Для каждого кандидата, прошедшего отбор по форме и
   положению, вырезается маленький кроп **только этого блока** и
   читается с белым списком цифр и ``psm 7``. Здесь движок уже не выбирает,
   куда смотреть, и читает то, что ему дали.

Второй шаг поднимает результат с 9 из 12 подписей листа 8 до полного
попадания, и разница видна не в цифрах, а в структуре: после него движок
возвращает не «много чего-то», а одно число с рамкой.

Затирание размерной линии обязательно. Штрих 0.48 pt на 500 dpi — это
3.3 px, ровно столько, сколько нужно движку, чтобы решить, что перед ним
цифра «1». Незатёртая линия внутри кропа даёт устойчиво неверные числа, и
выглядит это правдоподобно. Наконечники стираются отдельно и с запасом:
они шире штриха, и оставленный наконечник читается как обломок цифры.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import words as words_mod
from .engines import Engine
from .render import PageRaster

# Зазоры заданы в **пунктах PDF** и переводятся в пиксели через
# ``px_per_pt``. Подпись на этих листах стоит в 5.5-10 pt от своего размера,
# то есть в 38-70 px при 500 dpi. Кроп, посчитанный «на глаз» в пикселях,
# туда не попадает, и движок честно отвечает «пусто» — при вполне
# читаемой картинке это выглядит как полный провал распознавания.
#
# Вдоль отрезка: подпись стоит у его середины, но иногда смещается к
# концу, если в середине не помещается. Короткий размер на листе 10 («134»,
# длина 17 pt) подписан выноской в 60 pt в сторону, и до неё полоса не
# дотягивается — она обрабатывается отдельно.
ALONG_FACTOR = 0.5
ALONG_PAD_PT = 26.0

# Полоса по нормали: с запасом против измеренных 5.5-10 pt, потому что
# расстояние берётся до центра подписи, а у края кропа нужен ещё поперечник
# самой цифры и белое поле вокруг неё. 20 pt = 139 px при 500 dpi.
NORMAL_PAD_PT = 20.0

# Во сколько раз увеличивать кроп. Подпись капителью 4.2 pt на 500 dpi —
# это 29 px, а движок уверенно читает от ~32 px. Тройное увеличение
# поднимает капитель до ~90 px и не рисует новых деталей: интерполяция
# кубическая, information появляется только из уже имеющихся пикселей.
UPSCALE = 3.0

# Во сколько раз увеличивать отдельно вырезанный блок кандидата. Здесь
# движок уже не выбирает направление, и ему достаточно разобрать форму
# цифр, поэтому увеличение сильнее.
BLOCK_UPSCALE = 6.0

# Сколько белого вокруг блока. Движок на обрезанной по буквам рамке теряет
# первую и последнюю цифру, и это проявляется именно на коротких
# подписях в две-три цифры — то есть там, где ошибка стоит дороже всего.
BLOCK_BORDER_PX = 20

# Ориентации, под которыми пробуем читать. Первая — вдоль отрезка: на
# чертеже подпись повёрнута вместе с размером, и это основная гипотеза.
# Остальные нужны для подписей, поставленных горизонтально рядом с
# наклонным размером: на листе 8 «1000» стоит вертикально при отрезке,
# идущем под -30°.
ANGLE_OFFSETS_DEG = (0.0, 90.0, -90.0, 180.0)

# Отбор кандидата по форме блока. Цифра подписи — это несколько знаков
# одинаковой высоты, стоящих в ряд. Отсекаем всё, что не похоже на строку
# цифр: рамки узлов, одиночные стрелки, обрывки осевой.
MIN_BLOCK_ASPECT = 0.9      # ширина к высоте; «6000» — это 3.4, «1» — 0.5
MAX_BLOCK_ASPECT = 12.0     # одиночная точка или вертикальная черта
MIN_BLOCK_HEIGHT_PX = 6     # копейки от шума на неоднородном фоне
MAX_BLOCK_HEIGHT_PT = 20.0  # подпись размеру выше 20 pt не бывает; это
                            # заголовок листа или номер узла в рамке


@dataclass
class Candidate:
    """Блок-кандидат на подпись, найденный на грубом проходе."""

    text: str
    conf: float
    box_px: tuple[int, int, int, int]
    centre_px: tuple[float, float]
    height_px: int
    distance_px: float
    in_normal_band: bool
    angle: float

    @property
    def plausible(self) -> bool:
        return self.in_normal_band


@dataclass
class Reading:
    """Одно прочтение подписи: результат точного прохода по блоку."""

    text: str
    value: float | None
    conf: float
    angle: float
    distance_px: float
    box_px: tuple[int, int, int, int]
    kind: str = "side"
    fixes: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "value": self.value,
            "conf": round(self.conf, 1),
            "angle": round(self.angle, 1),
            "distance_px": round(self.distance_px, 1),
            "box_px": list(self.box_px),
            "kind": self.kind,
            "fixes": self.fixes,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Преобразования
# ---------------------------------------------------------------------------


def _erase_line(
    image: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    width: int,
    *,
    tip_radius: int | None = None,
) -> np.ndarray:
    """Затереть размерную линию и её наконечники белым."""
    import cv2

    out = image.copy()
    cv2.line(
        out,
        (int(round(a[0])), int(round(a[1]))),
        (int(round(b[0])), int(round(b[1]))),
        255, thickness=max(1, width), lineType=cv2.LINE_AA,
    )
    radius = tip_radius if tip_radius is not None else max(3, int(width * 1.6))
    for point in (a, b):
        cv2.circle(
            out,
            (int(round(point[0])), int(round(point[1]))),
            radius, 255, thickness=-1, lineType=cv2.LINE_AA,
        )
    return out


@dataclass
class CropFrame:
    """Повёрнутый кроп и всё, что нужно, чтобы вернуть его в исходник."""

    image: np.ndarray
    matrix: np.ndarray       # исходная картинка → кроп
    box_px: tuple[int, int, int, int]
    angle: float
    origin_px: tuple[float, float]  # центр отрезка в пикселях исходника
    axis: tuple[float, float]       # единичный вектор вдоль отрезка
    normal: tuple[float, float]     # единичный вектор по нормали

    def to_source(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Рамка в координатах кропа → рамка в исходной картинке."""
        x0, y0, x1, y1 = box
        corners = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
        )
        inverse = np.linalg.inv(self.matrix)
        back = corners @ inverse[:2, :2].T + inverse[:2, 2]
        return (
            int(math.floor(back[:, 0].min())), int(math.floor(back[:, 1].min())),
            int(math.ceil(back[:, 0].max())), int(math.ceil(back[:, 1].max())),
        )

    def distance_to_axis(self, centre: tuple[float, float]) -> float:
        """Расстояние от точки кропа до оси отрезка, в пикселях кропа."""
        dx = centre[0] - self.image.shape[1] / 2.0
        dy = centre[1] - self.image.shape[0] / 2.0
        # Ось ``x`` кропа — вдоль отрезка, поэтому расстояние до неё это
        # вертикальная координата точки.
        return abs(dy)


def _make_crop(
    image: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    angle: float,
    *,
    scale: float,
    px_per_pt: float,
    half_along_extra: float = 0.0,
    border: int = 16,
) -> CropFrame:
    """Вырезать полосу вокруг отрезка и повернуть её горизонтально."""
    import cv2

    mid = (a + b) / 2.0
    length = float(np.hypot(*(b - a)))
    if length < 1e-6:
        tiny = np.zeros((4, 4), dtype=np.uint8)
        return CropFrame(
            image=tiny, matrix=np.eye(3)[:2], box_px=(0, 0, 4, 4),
            angle=angle, origin_px=(float(mid[0]), float(mid[1])),
            axis=(1.0, 0.0), normal=(0.0, 1.0),
        )

    half_along = length / 2.0 * (1.0 + ALONG_FACTOR) + ALONG_PAD_PT * px_per_pt
    half_along += half_along_extra
    half_norm = NORMAL_PAD_PT * px_per_pt

    matrix = cv2.getRotationMatrix2D((float(mid[0]), float(mid[1])), angle, scale)
    cos_a, sin_a = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    ex = abs(cos_a) * half_along + abs(sin_a) * half_norm
    ey = abs(sin_a) * half_along + abs(cos_a) * half_norm

    # Размер выходной картинки — область **после** увеличения, поэтому
    # умножается на масштаб. Забыть про ``scale`` значит показать на тройном
    # увеличении треть полосы, и подпись уедет за край: картинка выглядит
    # крупнее, а информации в ней меньше.
    width = max(int(round(2 * ex * scale)) + 2 * border, 24)
    height = max(int(round(2 * ey * scale)) + 2 * border, 24)

    shifted = matrix[:, :2].dot(mid) + matrix[:, 2]
    matrix[0, 2] += width / 2.0 - shifted[0]
    matrix[1, 2] += height / 2.0 - shifted[1]

    warped = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    radians = math.radians(angle)
    return CropFrame(
        image=warped,
        matrix=np.vstack([matrix, [0.0, 0.0, 1.0]]),
        box_px=(0, 0, width, height),
        angle=angle,
        origin_px=(float(mid[0]), float(mid[1])),
        axis=(math.cos(radians), math.sin(radians)),
        normal=(-math.sin(radians), math.cos(radians)),
    )


def _binarize(image: np.ndarray) -> np.ndarray:
    """Привести к чёрному по белому: выравнивание фона, затем порог Оцу.

    На чистом листе достаточно глобального порога. На испорченном (``photo``)
    освещение неравномерно, и глобальный порог либо съедает чёрточки в
    засвеченной части, либо превращает бумагу в чернила в затенённой.
    Поэтому сперва локальное выравнивание делением на размытую копию —
    это дешёвый аналог адаптивного порога без размытой рамки по краям.
    """
    import cv2

    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    background = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    flat = cv2.divide(blurred, background, scale=255)
    _, out = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


# ---------------------------------------------------------------------------
# Два прохода
# ---------------------------------------------------------------------------


def _candidates(
    crop: CropFrame, engine: Engine, max_height_pt: float, px_per_pt: float
) -> list[Candidate]:
    """Грубый проход: все блоки-кандидаты с рамками, без белого списка."""
    binary = _binarize(crop.image)
    found: list[Candidate] = []
    for text, conf, box in engine.read_crop(binary, psm=11, whitelist=False):
        if not text.strip():
            continue
        width = box[2] - box[0]
        height = box[3] - box[1]
        if height < MIN_BLOCK_HEIGHT_PX:
            continue
        aspect = width / max(1, height)
        if not (MIN_BLOCK_ASPECT <= aspect <= MAX_BLOCK_ASPECT):
            continue
        if height / px_per_pt > max_height_pt:
            continue
        centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        distance = crop.distance_to_axis(centre)
        # Полоса, в которой может стоять подпись. В кропе ось отрезка
        # проходит по середине, а подпись смещена вверх или вниз на
        # NORMAL_PAD_PT; за её пределами находятся чужие подписи.
        band = NORMAL_PAD_PT * px_per_pt * 0.85
        found.append(
            Candidate(
                text=text.strip(), conf=conf, box_px=box, centre_px=centre,
                height_px=height, distance_px=distance,
                in_normal_band=distance <= band, angle=crop.angle,
            )
        )
    found.sort(key=lambda c: c.distance_px)
    return found


def _read_block(
    crop: CropFrame, engine: Engine, box: tuple[int, int, int, int]
) -> list[tuple[str, float]]:
    """Точный проход: вырезать один блок и прочитать его как строку цифр."""
    import cv2

    x0, y0, x1, y1 = box
    tight = crop.image[
        max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4
    ]
    if tight.size == 0:
        return []
    enlarged = cv2.resize(
        tight, None, fx=BLOCK_UPSCALE, fy=BLOCK_UPSCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    enlarged = cv2.copyMakeBorder(
        enlarged, BLOCK_BORDER_PX, BLOCK_BORDER_PX,
        BLOCK_BORDER_PX, BLOCK_BORDER_PX, cv2.BORDER_CONSTANT, value=255,
    )
    enlarged = _binarize(enlarged)
    # Рамка блока здесь не нужна: точное прочтение уже привязано к рамке
    # кандидата, а рамка из ``psm 7`` описывает уже увеличенный кроп, и
    # пересчитывать её обратно пришлось бы в другую систему координат.
    return [(t, c) for t, c, _ in engine.read_crop(enlarged, psm=7)]


def read_segment(
    raster: PageRaster,
    engine: Engine,
    a: np.ndarray,
    b: np.ndarray,
    *,
    angles: tuple[float, ...] = ANGLE_OFFSETS_DEG,
    scale: float = UPSCALE,
    erase_width_px: int = 9,
    leader_end: np.ndarray | None = None,
    max_candidates: int = 4,
) -> list[Reading]:
    """Все прочтения подписи для одного размерного отрезка.

    ``a`` и ``b`` — концы отрезка **в пикселях готовой картинки**.
    ``leader_end`` — конец выноски, если подпись вынесена: она лежит вне
    полосы вокруг отрезка, и без отдельной обработки кроп её не увидит.
    """
    angle_line = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

    # Затирание линии: на 500 dpi штрих 0.48 pt — это 3.3 px, и для движка
    # он неотличим от цифры «1». Толщина затирания берётся с запасом
    # относительно штриха, наконечники — с запасом относительно неё.
    base = _erase_line(
        raster.image, a, b, erase_width_px, tip_radius=erase_width_px * 2
    )
    if leader_end is not None:
        base = _erase_line(base, (a + b) / 2.0, leader_end, max(2, erase_width_px - 3))

    readings: list[Reading] = []

    for offset in angles:
        crop = _make_crop(
            base, a, b, angle_line + offset,
            scale=scale, px_per_pt=raster.px_per_pt,
        )
        for cand in _candidates(crop, engine, MAX_BLOCK_HEIGHT_PT, raster.px_per_pt):
            if not cand.plausible:
                continue
            for text, conf in _read_block(crop, engine, cand.box_px):
                parsed = words_mod.parse(text)
                source_box = crop.to_source(cand.box_px)
                readings.append(
                    Reading(
                        text=text, value=parsed.value, conf=conf,
                        angle=crop.angle, distance_px=cand.distance_px,
                        box_px=source_box,
                        kind="side" if leader_end is None else "leader",
                        fixes=parsed.fixes, reason=parsed.reason,
                    )
                )
            if len(readings) >= max_candidates:
                break
        if len(readings) >= max_candidates:
            break

    # Выноска: подпись стоит на её конце, то есть заметно в стороне от
    # отрезка. Кроп строится вокруг самого конца выноски, и ось отрезка
    # внутри него уже не проходит по середине — поэтому расстояние до
    # «оси» здесь не используется, а отбор идёт по форме блока.
    if leader_end is not None:
        direction = leader_end - (a + b) / 2.0
        norm = float(np.hypot(*direction))
        if norm > 1e-6:
            direction = direction / norm
            fake_b = leader_end + direction * 30.0
            crop = _make_crop(
                base, leader_end, fake_b, angle_line + offset,
                scale=scale, px_per_pt=raster.px_per_pt,
            )
            for cand in _candidates(crop, engine, MAX_BLOCK_HEIGHT_PT, raster.px_per_pt):
                for text, conf in _read_block(crop, engine, cand.box_px):
                    parsed = words_mod.parse(text)
                    readings.append(
                        Reading(
                            text=text, value=parsed.value, conf=conf,
                            angle=crop.angle, distance_px=cand.distance_px,
                            box_px=crop.to_source(cand.box_px),
                            kind="leader", fixes=parsed.fixes, reason=parsed.reason,
                        )
                    )
    return readings
