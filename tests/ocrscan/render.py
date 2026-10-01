"""Рендер листа в растр и профили деградации.

Зачем три профиля, если на входе всегда один и тот же PDF. Потому что
единственный способ проверить работу на скане — это **испортить лист
заранее** и посмотреть, что получится. На идеальном векторном рендере любая
разница между OCR-движками объясняется их архитектурой; на испорченном листе
видно ещё и то, что архитектура не спасла.

Три профиля — не три абстрактных уровня шума, а три реальных источника:

``clean``    Лист как есть, векторный рендер 500 dpi. Это то, что отдаёт
             родной экспорт из CAD. Считается контрольным: если и на нём
             есть ошибки, дело в алгоритме, а не в качестве картинки.
``scan200``  Офисный МФУ, копирование с автоопределением. Типичный результат
             «скан 200 dpi»: передискретизация сглаживает тонкие линии,
             оптика слегка размывает, JPEG вносит блоки, и есть небольшой
             перекос из-за неплотно прижатой крышки.
``photo``    Камера телефона над листом на столе. Свет падает неравномерно,
             есть смаз от движения, соль-шум в тенях, а лист лежит с
             поворотом. Капитель цифры тут около 11 px — читаемость под
             вопросом, и это надо измерить, а не предположить.

Порядок искажений выбран по порядку их возникновения в реальности: бумага →
оптика (блюр) → датчик (шум, засветка) → кодирование (JPEG). Перекос листа
применяется последним к готовому изображению: это не в точности то же, что
повернуть лист до сканирования, но отличие сводится к одному пересчёту
пикселей и на устойчивость распознавания не влияет.

Про перекос. В растре нигде не зашиты углы ``±30°`` и ``90°``: наклон
размерной линии измеряется у каждой найденной линии, и подпись читается в
системе координат, повёрнутой вместе с ней. Благодаря этому перекос не
ломает конвейер, а поворачивает вместе с листом. Здесь это проверяется
измерением, а не обещанием: профили ``scan200`` и ``photo`` отличаются от
``clean`` именно на ``skew_deg``.

``PageRaster`` дополнительно умеет переводить пиксели обратно в пункты PDF,
чтобы результат растового разбора можно было положить рядом с эталоном из
вектора и сравнить.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import pymupdf

# Промежуточное разрешение рендера. Все профили сначала рисуются здесь и
# потом приводятся к своему: так передискретизация делается один раз и
# одинаково для всех листов.
BASE_DPI = 500

# Обрезанные пустые поля убираются по той же причине, что и в основном
# конвейере (``isokrus.trim``): чертёж занимает 37-53% площади листа, а
# пустое поле не добавляет ничего, кроме расхода пикселей на разрешение.
TRIM = True


@dataclass(frozen=True)
class Profile:
    """Один вариант искажения листа. Все величины — в единицах профиля."""

    name: str
    dpi: int
    blur: float = 0.0              # σ гауссова размывки, px
    motion_length: int = 0         # длина ядра смаза, px (0 — нет)
    motion_angle: float = 0.0      # направление смаза, градусы
    gauss_noise: float = 0.0       # σ нормального шума, уровни 0-255
    salt_pepper: float = 0.0       # доля импульсного шума, 0-1
    shade: float = 0.0             # разброс освещённости, 0-1
    jpeg_quality: int = 100        # качество JPEG; 100 — фактически без потерь
    skew_deg: float = 0.0          # перекос листа, градусы
    seed: int = 20240917

    @property
    def px_per_pt(self) -> float:
        return self.dpi / 72.0


PROFILES: dict[str, Profile] = {
    # Контрольный лист без искажений. С него начинают, иначе нельзя понять,
    # чья именно ошибка обсуждается — картинки или алгоритма.
    "clean": Profile(name="clean", dpi=500),

    # Офисный МФУ: передискретизация до 200 dpi, оптика, JPEG и перекос от
    # неплотно прижатой крышки. 200 dpi — самый частый режим копирования,
    # и самый низкий, на котором чертёж вообще ещё читают глазами.
    "scan200": Profile(
        name="scan200",
        dpi=200,
        blur=0.6,
        motion_length=3,
        motion_angle=15.0,
        gauss_noise=2.5,
        salt_pepper=0.0004,
        shade=0.12,
        jpeg_quality=75,
        skew_deg=0.3,
    ),

    # Телефон над листом на столе: неравномерный свет, смаз, засветка теней,
    # JPEG с агрессивным сжатием и заметный перекос.
    "photo": Profile(
        name="photo",
        dpi=150,
        blur=0.5,
        motion_length=9,
        motion_angle=8.0,
        gauss_noise=4.0,
        salt_pepper=0.0015,
        shade=0.45,
        jpeg_quality=50,
        skew_deg=1.5,
    ),
}


# ---------------------------------------------------------------------------
# Искажения
# ---------------------------------------------------------------------------


def _resize(image: np.ndarray, dpi: int) -> np.ndarray:
    """Привести картинку к нужному разрешению.

    ``INTER_AREA`` — усреднение по площади, а не выборка: именно так ведёт
    себя оптическая система при передискретизации. Ближайший сосед
    превратил бы тонкую линию в пунктир из отдельных точек и сломал бы
    детектор наконечников нечестным образом.
    """
    scale = dpi / BASE_DPI
    if abs(scale - 1.0) < 1e-9:
        return image
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _motion_blur(image: np.ndarray, length: int, angle: float) -> np.ndarray:
    """Смаз от движения: линейное ядро нужной длины под нужным углом."""
    if length < 2:
        return image
    kernel = np.zeros((length, length), dtype=np.float32)
    centre = (length - 1) / 2.0
    radians = math.radians(angle)
    dx, dy = math.cos(radians), math.sin(radians)
    for step in range(-(length // 2), length // 2 + 1):
        x = int(round(centre + dx * step))
        y = int(round(centre + dy * step))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    total = float(kernel.sum()) or 1.0
    kernel /= total
    return cv2.filter2D(image, -1, kernel)


def _shade(image: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Неравномерное освещение: пологий градиент яркости по всему листу.

    Именно пологий. Реальный блик от лампы в комнате — это широкая дуга, а не
    пятно, и на ней самая большая опасность для адаптивного порога: в
    освещённой части чертёж выцветает до еле заметного, а порог, взятый по
    всему листу, там ничего не находит.
    """
    if amount <= 0:
        return image
    height, width = image.shape[:2]
    # Случайное направление: света справа-сверху и сверху-слева встречаются
    # одинаково часто, и фиксированный выбор делал бы проверку однобокой.
    angle = float(rng.uniform(0, math.pi))
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    field = xs * math.cos(angle) + ys * math.sin(angle)
    field = 1.0 + amount * field * rng.uniform(0.4, 1.0)
    shaded = image.astype(np.float32) * field
    return np.clip(shaded, 0, 255).astype(np.uint8)


def _noise(
    image: np.ndarray, gauss: float, salt_pepper: float, rng: np.random.Generator
) -> np.ndarray:
    """Шум датчика: нормальный в светах и импульсный в тенях."""
    out = image.astype(np.float32)
    if gauss > 0:
        out += rng.normal(0.0, gauss, out.shape)
    if salt_pepper > 0:
        mask = rng.random(out.shape[:2])
        out[mask < salt_pepper / 2] = 0.0
        out[mask > 1.0 - salt_pepper / 2] = 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    """Ходит через реальный кодек JPEG, а не через ``imdecode`` заглушку."""
    if quality >= 100:
        return image
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)


def _skew(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Повернуть лист на ``angle`` градусов вокруг центра.

    Возвращает вместе с картинкой матрицу ``2x3``, которая переводит точку
    готового изображения в точку исходного рендера. Отдавать её наружу
    обязательно: без неё нельзя сравнить результат растового разбора с
    эталоном из вектора, а сравнение здесь — главный инструмент проверки.
    """
    if abs(angle) < 1e-6:
        return image, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    height, width = image.shape[:2]
    centre = (width / 2.0, height / 2.0)
    forward = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        image, forward, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    # ``getRotationMatrix2D`` описывает переход «из готовой картинки в
    # исходную», то есть именно то, что нужно для обратного перевода.
    return rotated, forward


# ---------------------------------------------------------------------------
# Страница-растр
# ---------------------------------------------------------------------------


@dataclass
class PageRaster:
    """Лист как картинка плюс всё, чтобы вернуть координаты в пунктах PDF.

    Внутри модулей геометрии работа ведётся в **пикселях готовой картинки**:
    это единственная система координат, которая есть и у чистого листа, и у
    испорченного. ``to_points`` переводит результат обратно в систему PDF, в
    которой задан эталон.
    """

    image: np.ndarray
    profile: Profile
    page_number: int
    page_rect: tuple[float, float, float, float]
    skew_back: np.ndarray
    trim_offset_px: tuple[int, int] = (0, 0)

    # -- геометрия ---------------------------------------------------------

    @property
    def px_per_pt(self) -> float:
        return self.profile.px_per_pt

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    def _apply(self, matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Аффинное преобразование на массиве точек ``(..., 2)``."""
        arr = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return arr @ matrix[:2, :2].T + matrix[:2, 2]

    def px_from_pt(self, points: np.ndarray) -> np.ndarray:
        """Точки PDF → пиксели готовой картинки.

        Шаги идут строго в том же порядке, в каком картинка собиралась:
        масштаб рендера, обрезка пустых полей, поворот листа. Перестановка
        любых двух шагов не бросает исключение, а сдвигает координаты на
        единицы пунктов — и это выглядит не как ошибка координат, а как
        «OCR не находит подписи рядом с размером».
        """
        arr = np.atleast_2d(np.asarray(points, dtype=np.float64)) * self.px_per_pt
        arr = arr - np.asarray(self.trim_offset_px, dtype=np.float64)
        if self.profile.skew_deg:
            # ``getRotationMatrix2D`` описывает переход «из готовой картинки
            # в исходную», поэтому для движения вперёд нужна обратная.
            forward = np.vstack([self.skew_back, [0.0, 0.0, 1.0]])
            arr = self._apply(np.linalg.inv(forward), arr)
        return arr

    def to_points(self, points: np.ndarray) -> np.ndarray:
        """Пиксели готовой картинки → пункты PDF. ``(..., 2)`` на входе."""
        arr = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if self.profile.skew_deg:
            forward = np.vstack([self.skew_back, [0.0, 0.0, 1.0]])
            arr = self._apply(forward, arr)
        arr = arr + np.asarray(self.trim_offset_px, dtype=np.float64)
        return arr / self.px_per_pt

    def to_points_poly(self, box: tuple[float, float, float, float]) -> tuple[float, ...]:
        """Рамка из пикселей — в пункты PDF, как коробка из четырёх точек."""
        x0, y0, x1, y1 = box
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        mapped = self.to_points(corners)
        return (
            float(mapped[:, 0].min()), float(mapped[:, 1].min()),
            float(mapped[:, 0].max()), float(mapped[:, 1].max()),
        )

    def length_pt(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        """Длина отрезка в пунктах, если точки заданы в пикселях картинки."""
        mapped = self.to_points(np.array([a, b], dtype=np.float64))
        return float(math.hypot(*(mapped[1] - mapped[0])))

    # -- доступ к пикселям -------------------------------------------------

    def crop(self, box: tuple[float, float, float, float]) -> np.ndarray:
        """Кусок картинки по рамке в пикселях, с ограничением по границам."""
        height, width = self.image.shape[:2]
        x0 = max(0, int(math.floor(box[0])))
        y0 = max(0, int(math.floor(box[1])))
        x1 = min(width, int(math.ceil(box[2])))
        y1 = min(height, int(math.ceil(box[3])))
        if x1 <= x0 or y1 <= y0:
            return np.zeros((1, 1), dtype=np.uint8)
        return self.image[y0:y1, x0:x1]

    def save(self, path: str) -> None:
        cv2.imwrite(path, self.image)


# ---------------------------------------------------------------------------
# Рендер
# ---------------------------------------------------------------------------


def _trim_empty_border(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Отрезать пустые поля, оставив небольшой запас белого.

    Тот же приём, что в ``isokrus.trim``, и по той же причине: на этих листах
    пустое поле занимает до половины площади, и оно съедает разрешение,
    которого и так едва хватает на подписи размеров.
    """
    ink = image < 200
    rows = np.flatnonzero(ink.any(axis=1))
    columns = np.flatnonzero(ink.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return image, (0, 0, image.shape[1], image.shape[0])
    pad = 12
    y0 = max(0, int(rows[0]) - pad)
    y1 = min(image.shape[0], int(rows[-1]) + pad + 1)
    x0 = max(0, int(columns[0]) - pad)
    x1 = min(image.shape[1], int(columns[-1]) + pad + 1)
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def render_page(
    page: pymupdf.Page, profile: Profile, *, trim: bool = TRIM
) -> PageRaster:
    """Лист PDF → картинка по профилю."""
    pixmap = page.get_pixmap(dpi=BASE_DPI, colorspace=pymupdf.csGRAY, alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )[:, :, 0].copy()
    rect = (page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1)

    # Сначала передискретизация: она единственная из всех операций, которая
    # меняет разрешение, и обязана идти до размытия и шума — иначе шум
    # окажется крупнее, чем сам сигнал.
    image = _resize(image, profile.dpi)

    # Обрезка — тоже до размытия и шума. Иначе пустое поле получит тот же
    # шум и JPEG-артефакты, что и чертёж, и порог «есть ли содержимое» перестанет
    # отличать поле от линий. Её сдвиг сохраняется: картинка больше не
    # начинается в точке (0, 0) системы PDF.
    offset = (0, 0)
    if trim:
        image, box = _trim_empty_border(image)
        offset = (box[0], box[1])

    rng = np.random.default_rng(profile.seed + page.number)
    if profile.blur > 0:
        image = cv2.GaussianBlur(image, (0, 0), profile.blur)
    if profile.motion_length >= 2:
        image = _motion_blur(image, profile.motion_length, profile.motion_angle)
    if profile.shade > 0:
        image = _shade(image, profile.shade, rng)
    if profile.gauss_noise > 0 or profile.salt_pepper > 0:
        image = _noise(image, profile.gauss_noise, profile.salt_pepper, rng)
    image = _jpeg(image, profile.jpeg_quality)
    image, skew_back = _skew(image, profile.skew_deg)

    return PageRaster(
        image=image,
        profile=profile,
        page_number=page.number + 1,
        page_rect=rect,
        skew_back=skew_back,
        trim_offset_px=offset,
    )


def render_pdf(
    pdf_path: str,
    profile: Profile,
    pages: Sequence[int] | None = None,
    *,
    trim: bool = TRIM,
) -> list[PageRaster]:
    """Листы PDF как картинки. ``pages`` — номера с единицы, ``None`` — все."""
    with pymupdf.open(pdf_path) as document:
        wanted = (
            list(range(document.page_count))
            if not pages
            else [p - 1 for p in pages if 1 <= p <= document.page_count]
        )
        return [render_page(document.load_page(i), profile, trim=trim) for i in wanted]
