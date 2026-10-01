"""Распознавание: общий интерфейс и три реализации.

Модуль намеренно узкий. Каждый движок умеет ровно одно — **найти на листе
слова и вернуть их вместе с рамками в пунктах PDF**. Дальше с ними работает
``words.py``, а потом общая часть конвейера из ``dimscan``. Всё остальное —
детектор, разбор иерархии, подсчёт сумм — не знает, какой движок дал слова.

Почему рамки нужны точные. В ``dimscan`` подпись размеру подбирается по
расстоянию до отрезка: берётся ближайшее число, а числовые подписи имеют
приоритет над остальными, потому что рядом с размером стоят ``DN50`` и
координаты привязки. Эта же логика работает и для OCR-слов, но только если
рамка пришла в той же системе координат, что и отрезок, — в пунктах PDF.
Поэтому каждый движок переводит пиксели в пункты сам, используя
``PageRaster.to_points``.

Про полностраничный режим. Подписи на изометрии подписаны и наклонно, и
вертикально: из 114 подписей на 10 листах 73 повёрнуты на ±30°, 22 стоят
вертикально, и только 19 горизонтальны. Детектор ``dimscan`` это учитывает
по ``angle`` у найденного отрезка, но сам текстовый слой от наклона не
зависит. С OCR иначе: движок ищет текст в своей ориентации, и подпись,
повёрнутая на 30°, на скане может быть или найдена с меньшей уверенностью,
или не найдена вовсе. Отсюда два режима:

``psm`` 11/12  «разреженный текст» — на листе текста мало, он разбросан
             точками, и это единственный режим, в котором движок не
             склеивает соседние подписи в одну строку;
``psm`` 6     «блок текста» — на плотных листах (лист 2, 17 отрезков)
             разреженный режим дробит подписи на части, и это видно по
             метрикам: растёт число прочтений, падает точность.

Числовой белый список включается везде. Он убирает большой класс ошибок
разметки: латиница рядом с цифрами на листе есть всегда («X», «Y», «DN»,
«LT»), и без белого списка движок охотно читает подпись размера как слово.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
import pymupdf

from . import words as words_mod
from .render import PageRaster

# Windows: tesseract ставится в Program Files и не всегда попадает в PATH.
# Искать надо в типовых местах, иначе ошибка выглядит как «модуль не
# установлен», хотя установлен именно бинарник.
_TESSERACT_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
)

# Белый список: цифры, точка, запятая, минус, пробел и два знака неравенства.
# Неравенства оставлены не для подписей размеров — их тут не бывает, — а
# чтобы различение «<2> это номер оси» и «<12> это размер» решалось уже в
# ``words.py``, по содержимому, а не молча терялось на этапе распознавания.
TESS_WHITELIST = "0123456789.,-<>()"


@dataclass
class OcrWord:
    """Одно прочитанное слово в системе координат PDF."""

    text: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 в пунктах PDF
    conf: float
    engine: str
    #: 0 — совпало с ориентиром движка, -1 — движок уверяется в повороте
    angle: float = 0.0

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "box": [round(v, 2) for v in self.box],
            "conf": round(self.conf, 1),
            "engine": self.engine,
            "angle": round(self.angle, 1),
        }


@dataclass
class OcrPageResult:
    """Что один движок прочитал на одном листе."""

    page_number: int
    engine: str
    profile: str
    words: list[OcrWord] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def numbers(self) -> list[OcrWord]:
        """Только те слова, которые разбираются в число размера.

        Важно: проверка идёт через ``words.is_number``, а не через
        ``words.parse``. Разница в том, что ``parse`` чинит буквы-цифры и
        вытаскивает число из мусора, а такая подпись уже не годится как
        опора: она может оказаться обрывком координаты, и тогда к размеру
        приклеится не то.
        """
        return [w for w in self.words if words_mod.is_number(w.text)]

    def summary(self) -> dict:
        return {
            "page": self.page_number,
            "engine": self.engine,
            "profile": self.profile,
            "words": len(self.words),
            "numbers": len(self.numbers),
            "seconds": round(self.seconds, 2),
        }


class Engine(Protocol):
    """Общий контракт движка."""

    name: str

    def available(self) -> tuple[bool, str]:
        """Готов ли движок и, если нет, чем это лечится."""
        ...

    def read_page(self, raster: PageRaster, *, psm: int | None = None) -> OcrPageResult:
        """Прочитать лист целиком."""
        ...


# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------


def find_tesseract() -> str | None:
    """Путь к бинарнику: сначала PATH, потом типовые места установки."""
    env = os.environ.get("TESSERACT_CMD")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in _TESSERACT_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


class TesseractEngine:
    """Tesseract через ``pytesseract``.

    Работает в два захода. Первый — ``osd`` (определение ориентации) — нужен
    не для красоты: на листах с перекосом ``scan200`` и ``photo`` движок без
    него читает всю страницу вертикально и не находит ничего. Второй —
    ``image_to_data``, из которого берутся слова, рамки и уверенность.
    """

    name = "tess"

    def __init__(self, lang: str = "eng", psm: int = 11) -> None:
        self.lang = lang
        self.psm = psm

    def available(self) -> tuple[bool, str]:
        binary = find_tesseract()
        if binary is None:
            return False, (
                "не найден tesseract.exe. Установить: winget install "
                "UB-Mannheim.TesseractOCR, либо задать путь в TESSERACT_CMD"
            )
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            return False, "не установлен pytesseract: pip install pytesseract"
        return True, binary

    def _config(self, psm: int) -> str:
        return (
            f"--psm {psm} --oem 1 -c tessedit_char_whitelist={TESS_WHITELIST} "
            "-c preserve_interword_spaces=1"
        )

    def read_crop(
        self, crop: np.ndarray, *, psm: int = 11, whitelist: bool = True
    ) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        """Прочитать один кроп: список ``(текст, уверенность, рамка)``.

        Рамка в пикселях кропа возвращается не для красоты, а потому что
        без неё невозможна вторая, решающая стадия. В кропе вокруг отрезка
        лежит не только подпись: там же оказываются номера узлов в рамках,
        кусок пунктирной осевой, стрелки и подписи соседних размеров. Пока
        движок не сообщает, **где** он увидел цифры, выбрать нужную строку
        нечем, и остаётся надеяться, что движок сам догадается.

        Про режимы. Для кропа используется ``psm 11`` («разреженный текст»),
        а не ``psm 7`` («одна строка»). Это видно на кропе листа 8: подпись
        «6000» стоит под размерной линией, а линия и пунктир осевой — над
        ней. Режим «одна строка» выбирает самую длинную горизонтальную
        полосу и читает её, то есть штрих линии вместо цифр, и выдаёт пустоту
        с уверенностью около нуля. Разреженный режим берёт все блоки по
        отдельности и читает подпись уверенностью 96 на том же кропе.

        Белый список на кропе обязателен, на странице — нет. В кропе кроме
        подписи ничего нет, и латинские буквы там означают только ошибку.
        """
        import pytesseract
        from PIL import Image

        if pytesseract.pytesseract.tesseract_cmd in (None, "tesseract"):
            pytesseract.pytesseract.tesseract_cmd = find_tesseract()

        config = f"--psm {psm} --oem 1"
        if whitelist:
            config += f" -c tessedit_char_whitelist={TESS_WHITELIST}"

        data = pytesseract.image_to_data(
            Image.fromarray(crop), lang=self.lang, config=config,
            output_type=pytesseract.Output.DICT,
        )
        out: list[tuple[str, float, tuple[int, int, int, int]]] = []
        for index, raw in enumerate(data["text"]):
            if not raw or not raw.strip():
                continue
            try:
                conf = float(data["conf"][index])
            except (TypeError, ValueError):
                continue
            if conf < 0:
                continue
            left, top = int(data["left"][index]), int(data["top"][index])
            box = (
                left, top, left + int(data["width"][index]),
                top + int(data["height"][index]),
            )
            out.append((raw.strip(), conf, box))
        return out

    def read_page(self, raster: PageRaster, *, psm: int | None = None) -> OcrPageResult:
        import pytesseract
        from PIL import Image

        # ``pytesseract`` по умолчанию зовёт «tesseract», полагаясь на PATH.
        # На Windows бинарник из Program Files в PATH часто не попадает, и
        # проверка версии падает раньше, чем начнётся распознавание. Поэтому
        # путь проставляется всегда, когда он найден, а не только когда поле
        # ещё равно ``None``.
        binary = find_tesseract()
        if binary and pytesseract.pytesseract.tesseract_cmd in (None, "tesseract"):
            pytesseract.pytesseract.tesseract_cmd = binary

        image = Image.fromarray(raster.image)
        started = time.perf_counter()
        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=self._config(psm or self.psm),
            output_type=pytesseract.Output.DICT,
        )
        elapsed = time.perf_counter() - started

        boxes: dict[tuple, str] = {}
        meta: dict[tuple, list[float]] = {}
        for index, raw in enumerate(data["text"]):
            if not raw or not raw.strip():
                continue
            try:
                conf = float(data["conf"][index])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue  # движок не смог оценить — зачем нам такое слово
            key = (
                data["block_num"][index], data["par_num"][index],
                data["line_num"][index], data["word_num"][index],
            )
            left, top = int(data["left"][index]), int(data["top"][index])
            width, height = int(data["width"][index]), int(data["height"][index])
            boxes.setdefault(key, "")
            boxes[key] += raw
            entry = meta.setdefault(key, [left, top, left + width, top + height, 0.0, 0.0])
            entry[0] = min(entry[0], left)
            entry[1] = min(entry[1], top)
            entry[2] = max(entry[2], left + width)
            entry[3] = max(entry[3], top + height)
            entry[4] += conf
            entry[5] += 1

        result = OcrPageResult(
            page_number=raster.page_number, engine=self.name,
            profile=raster.profile.name, seconds=elapsed,
        )
        for key, text in boxes.items():
            x0, y0, x1, y1, conf_sum, count = meta[key]
            points = raster.to_points(
                np.array([[x0, y0], [x1, y1]], dtype=np.float64)
            )
            result.words.append(
                OcrWord(
                    text=text,
                    box=(
                        float(points[:, 0].min()), float(points[:, 1].min()),
                        float(points[:, 0].max()), float(points[:, 1].max()),
                    ),
                    conf=conf_sum / max(1.0, count),
                    engine=self.name,
                )
            )
        return result


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------


class PaddleEngine:
    """PaddleOCR: детектор текста плюс распознавание.

    Отличие от Tesseract, которое важно именно здесь. PaddleOCR сначала
    находит текстовые блоки детектором, а потом читает каждый — и умеет
    поворачивать найденный блок перед распознаванием. На этих листах 95
    подписей из 114 повёрнуты, так что это не улучшение качества, а
    необходимое условие: подпись, повёрнутую на 30°, движок без поворота
    либо не находит вовсе, либо читает с уверенностью около нуля.

    Модель загружается один раз на весь прогон, поэтому объект создаётся
    один раз и переиспользуется на всех листах.
    """

    name = "paddle"

    def __init__(self, lang: str = "en", *, det_db_unclip: float = 2.0) -> None:
        self.lang = lang
        self.det_db_unclip = det_db_unclip
        self._ocr = None

    def available(self) -> tuple[bool, str]:
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            return False, "не установлен paddleocr: pip install paddlepaddle paddleocr"
        return True, "paddleocr"

    def _ensure(self):
        if self._ocr is not None:
            return self._ocr
        from paddleocr import PaddleOCR

        # ``use_angle_cls`` включаем осознанно: на изометрии подписи
        # повёрнуты в четыре стороны, и без классификатора движок читает их
        # как мусор, уверенно выдавая буквы вместо цифр.
        self._ocr = PaddleOCR(
            lang=self.lang,
            use_angle_cls=True,
            use_textline_orientation=False,
            det_db_unclip=self.det_db_unclip,
            show_log=False,
        )
        return self._ocr

    def read_page(self, raster: PageRaster, *, psm: int | None = None) -> OcrPageResult:
        ocr = self._ensure()
        started = time.perf_counter()
        # Paddle ждёт BGR, а лист у нас серый: без пересборки каналов движок
        # получит трёхканальную картинку из серого и отдельно хуже прочитает
        # тонкие штрихи.
        bgr = cv_bgr_from_gray(raster.image)
        raw = ocr.ocr(bgr, cls=True)
        elapsed = time.perf_counter() - started

        result = OcrPageResult(
            page_number=raster.page_number, engine=self.name,
            profile=raster.profile.name, seconds=elapsed,
        )
        for entry in raw or []:
            for item in entry or []:
                if not item:
                    continue
                polygon, (text, conf) = item[0], item[1]
                if not text or not str(text).strip():
                    continue
                points = raster.to_points(np.asarray(polygon, dtype=np.float64))
                result.words.append(
                    OcrWord(
                        text=str(text).strip(),
                        box=(
                            float(points[:, 0].min()), float(points[:, 1].min()),
                            float(points[:, 0].max()), float(points[:, 1].max()),
                        ),
                        conf=float(conf) * 100.0,
                        engine=self.name,
                    )
                )
        return result


def cv_bgr_from_gray(image: np.ndarray) -> np.ndarray:
    """Серый лист → трёхканальная картинка, как ждёт OpenCV-совместимый движок."""
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# EasyOCR
# ---------------------------------------------------------------------------


class EasyEngine:
    """Запасной движок: работает на уже установленном ``torch``.

    Слабее Paddle на повёрнутом тексте и на узких шрифтах, но ставится
    догадкой: ``torch`` в проекте уже есть, и если Paddle на целевой машине
    не собирается, не остаётся совсем ничего.
    """

    name = "easy"

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._reader = None

    def available(self) -> tuple[bool, str]:
        try:
            import easyocr  # noqa: F401
        except ImportError:
            return False, "не установлен easyocr: pip install easyocr"
        return True, "easyocr"

    def _ensure(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader([self.lang], gpu=False, verbose=False)
        return self._reader

    def read_page(self, raster: PageRaster, *, psm: int | None = None) -> OcrPageResult:
        reader = self._ensure()
        started = time.perf_counter()
        raw = reader.readtext(raster.image, detail=1, paragraph=False)
        elapsed = time.perf_counter() - started

        result = OcrPageResult(
            page_number=raster.page_number, engine=self.name,
            profile=raster.profile.name, seconds=elapsed,
        )
        for polygon, text, conf in raw or []:
            if not text or not str(text).strip():
                continue
            points = raster.to_points(np.asarray(polygon, dtype=np.float64))
            result.words.append(
                OcrWord(
                    text=str(text).strip(),
                    box=(
                        float(points[:, 0].min()), float(points[:, 1].min()),
                        float(points[:, 0].max()), float(points[:, 1].max()),
                    ),
                    conf=float(conf) * 100.0,
                    engine=self.name,
                )
            )
        return result


# ---------------------------------------------------------------------------
# Реестр
# ---------------------------------------------------------------------------

ENGINES: dict[str, type] = {
    "tess": TesseractEngine,
    "paddle": PaddleEngine,
    "easy": EasyEngine,
}


def build(name: str, **kwargs) -> Engine:
    if name not in ENGINES:
        raise KeyError(f"неизвестный движок {name!r}; доступны: {sorted(ENGINES)}")
    return ENGINES[name](**kwargs)


def report_availability() -> list[tuple[str, bool, str]]:
    """Готовность всех движков — чтобы не выяснять это серединой прогона."""
    out: list[tuple[str, bool, str]] = []
    for name in ENGINES:
        engine = build(name)
        ok, detail = engine.available()
        out.append((name, ok, detail))
    return out
