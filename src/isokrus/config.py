"""Конфигурация приложения.

Все параметры берутся из переменных окружения (поддерживается `.env`).
Ничего не валидируется на импорте, кроме числовых полей — падать с
понятной ошибкой должен сам CLI, а не модуль.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - защита от мусора в .env
        raise ValueError(f"Переменная {name}={raw!r} должна быть целым числом") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover
        raise ValueError(f"Переменная {name}={raw!r} должна быть числом") from exc


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default


def _env_price(name: str, default: float) -> float:
    """Цена за 1M токенов в валюте провайдера (обычно USD)."""
    return _env_float(name, default)


_PRICE_KEYS = ("input", "output", "reasoning", "cache_write", "cache_read")


def _env_pricing_table(name: str) -> dict[str, dict[str, float]]:
    """Разобрать JSON-таблицу тарифов по моделям из переменной окружения."""
    raw = os.getenv(name)
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Переменная {name} должна быть JSON-объектом с тарифами по моделям"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"Переменная {name} должна быть JSON-объектом")
    table: dict[str, dict[str, float]] = {}
    for model, prices in data.items():
        if not isinstance(prices, dict):
            raise ValueError(f"{name}: тариф модели {model!r} должен быть объектом")
        table[str(model)] = {
            key: float(prices.get(key, 0.0) or 0.0) for key in _PRICE_KEYS
        }
    return table


# --- LLM ------------------------------------------------------------------

LLM_API_KEY: str | None = os.getenv("LLM_API_KEY")
LLM_BASE_URL: str = (
    os.getenv("LLM_BASE_URL")
    or os.getenv("OPENROUTER_BASE_URL")
    or "https://api.dslab.tech/v1"
)
LLM_MODEL: str = _env_str("LLM_MODEL", "glm-5.3-flash")
LLM_TIMEOUT: float = _env_float("LLM_TIMEOUT", 600.0)
LLM_MAX_RETRIES: int = _env_int("LLM_MAX_RETRIES", 3)
LLM_RETRY_BACKOFF: float = _env_float("LLM_RETRY_BACKOFF", 2.0)

# По умолчанию модель всегда работает в режиме глубокого размышления.
# Параметр ``temperature`` не задаётся вовсе: провайдер применяет свой
# дефолт, а вариант с передачей нуля был убран — он объявлялся в meta.json
# экспериментов, но до API никогда не доходил.
LLM_REASONING_EFFORT: str = _env_str("LLM_REASONING_EFFORT", "high")

# Модели, у которых режим рассуждения включается не через ``effort``.
#
# Шлюз один на всех, но тело запроса у моделей разное: ``gpt-6-luna`` и
# ``glm-5.3-flash`` понимают ``{"reasoning": {"effort": ...}}``, а ``minimax-m3``
# — ``{"reasoning": {"enabled": true}}``, и ``effort`` там молча игнорируется.
# Признак — ноль токенов рассуждения в ответе при заданном режиме: модель
# отвечает, но не размышляет, и прогон выглядит рабочим, а на деле отключает
# ровно то, ради чего его запускали. Поэтому список задаётся явно, а не
# угадывается по имени модели.
REASONING_ENABLED_MODELS: frozenset[str] = frozenset(
    name.strip().lower()
    for name in _env_str("LLM_REASONING_ENABLED_MODELS", "minimax-m3").split(",")
    if name.strip()
)


def uses_reasoning_switch(model: str | None) -> bool:
    """У этой модели рассуждение включается флагом, а не уровнем."""
    return (model or LLM_MODEL).strip().lower() in REASONING_ENABLED_MODELS

# Тарифы за 1M токенов — общие, переопределяются env.
# Токены рассуждения входят в completion_tokens, но тарифицируются отдельно.
# Кеш тоже: провайдеры OpenAI-совместимого вида включают в prompt_tokens и
# запись в кеш, и чтение из него, поэтому тарифицировать их надо отдельными
# ценами, иначе один и тот же токен оплачивается дважды.
PRICING_INPUT_PER_MTOK: float = _env_price("LLM_PRICING_INPUT", 0.0)
PRICING_OUTPUT_PER_MTOK: float = _env_price("LLM_PRICING_OUTPUT", 0.0)
PRICING_REASONING_PER_MTOK: float = _env_price("LLM_PRICING_REASONING", 0.0)
PRICING_CACHE_WRITE_PER_MTOK: float = _env_price("LLM_PRICING_CACHE_WRITE", 0.0)
PRICING_CACHE_READ_PER_MTOK: float = _env_price("LLM_PRICING_CACHE_READ", 0.0)
PRICING_CURRENCY: str = _env_str("LLM_PRICING_CURRENCY", "USD")

# Тарифы по моделям: LLM_PRICING_TABLE — JSON вида
#   {"gpt-6-luna": {"input": 10, "output": 50, "reasoning": 50,
#                   "cache_write": 12.5, "cache_read": 1.25}, "default": {...}}
# Зачем: модель переключается флагом --model, а тарифы у моделей разные, и
# при общих настройках прогон по дорогой модели молча считался по дешёвым
# ценам. Если таблица задана, а модели в ней нет — стоимость неизвестна, и
# мы говорим об этом, вместо того чтобы подставить чужой тариф.
PRICING_TABLE: dict[str, dict[str, float]] = _env_pricing_table("LLM_PRICING_TABLE")


@dataclass(frozen=True)
class Pricing:
    """Тарифы одной модели за 1M токенов."""

    input: float
    output: float
    reasoning: float
    cache_write: float
    cache_read: float
    source: str  # "модель", "общие" или "неизвестно"

    @property
    def known(self) -> bool:
        return self.source != "неизвестно"


def pricing_for(model: str | None) -> Pricing:
    """Тарифы для конкретной модели.

    Порядок: запись в таблице по имени модели -> ``default`` из той же
    таблицы -> общие настройки. Если задана таблица, но модели в ней нет и
    ``default`` отсутствует, тариф считается неизвестным: подставить общие
    цены значило бы показать в отчёте уверенное число, посчитанное не по
    той модели, и это хуже, чем честное «не знаю».
    """
    name = model or LLM_MODEL
    if PRICING_TABLE:
        prices = PRICING_TABLE.get(name) or PRICING_TABLE.get("default")
        if prices:
            # Отсутствующие ключи читаем как 0, а не падаем: запись в таблице
            # может быть неполной (например, без тарифа на кеш), и это
            # повод посчитать часть стоимости как ноль, а не упасть.
            def price(key: str) -> float:
                return float(prices.get(key, 0.0) or 0.0)

            output = price("output")
            return Pricing(
                input=price("input"),
                output=output,
                reasoning=price("reasoning") or output,
                cache_write=price("cache_write"),
                cache_read=price("cache_read"),
                source="таблица",
            )
        return Pricing(0.0, 0.0, 0.0, 0.0, 0.0, "неизвестно")
    if not pricing_enabled():
        return Pricing(0.0, 0.0, 0.0, 0.0, 0.0, "неизвестно")
    return Pricing(
        input=PRICING_INPUT_PER_MTOK,
        output=PRICING_OUTPUT_PER_MTOK,
        reasoning=PRICING_REASONING_PER_MTOK or PRICING_OUTPUT_PER_MTOK,
        cache_write=PRICING_CACHE_WRITE_PER_MTOK,
        cache_read=PRICING_CACHE_READ_PER_MTOK,
        source="общие",
    )


def pricing_enabled() -> bool:
    """Заданы ли хоть какие-то тарифы. Если нет — стоимость не считаем."""
    if PRICING_TABLE:
        return True
    return bool(
        PRICING_INPUT_PER_MTOK
        or PRICING_OUTPUT_PER_MTOK
        or PRICING_REASONING_PER_MTOK
    )


# --- Обработка ------------------------------------------------------------

MAX_PARALLEL: int = _env_int("MAX_PARALLEL", 10)
RENDER_DPI: int = _env_int("RENDER_DPI", 320)
# Ограничение по большей стороне: слишком большие картинки дороже и медленнее.
MAX_IMAGE_SIDE: int = _env_int("MAX_IMAGE_SIDE", 4096)
IMAGE_MIME: str = _env_str("IMAGE_MIME", "image/png")
SAVE_PAGE_IMAGES: bool = os.getenv("SAVE_PAGE_IMAGES", "1") not in {"0", "false", "False"}
ANNOTATE_DRAWINGS: bool = os.getenv("ANNOTATE_DRAWINGS", "1") not in {"0", "false", "False"}

# --- Обрезка пустых полей листа --------------------------------------------

# Обрезать поля, по которым нет нарисованного (аналог str.strip для картинки).
# Рамка ищется по пикселям, заранее заданных полей нет. Выключается флагом
# --no-trim, если нужно сравнить прогоны на полных листах.
TRIM_MARGINS: bool = os.getenv("TRIM_MARGINS", "1") not in {"0", "false", "False"}
# Белый запас вокруг среза, px: у края не должна вплотную стоять графика.
TRIM_PAD_PX: int = _env_int("TRIM_PAD_PX", 12)
# На сколько уровней пиксель должен быть темнее фона, чтобы считаться
# нарисованным. Фон определяется по левому верхнему углу листа, поэтому
# серый скан обрабатывается так же, как белый.
TRIM_NOISE: int = _env_int("TRIM_NOISE", 8)
# Сколько тёмных пикселей в строке/столбце считать содержимым. 1 = не резать
# ничего, что хоть чем-то похоже на содержимое. Больше 1 нужно для сканов:
# одиночная тёмная пылинка иначе отменяет обрезку всего листа.
TRIM_MIN_INK: int = _env_int("TRIM_MIN_INK", 1)
# Поле тоньше этого числа клеток уменьшенной копии — край, а не поле.
TRIM_MIN_RUN: int = _env_int("TRIM_MIN_RUN", 2)
# Ширина уменьшенной копии, по которой ищется рамка, px. Меньше = точнее и
# медленнее.
TRIM_PROBE_SIDE: int = _env_int("TRIM_PROBE_SIDE", 2048)
# Не обрезать, если результат уже этого размера, px.
TRIM_MIN_SIDE: int = _env_int("TRIM_MIN_SIDE", 96)

# Система координат, в которой модель возвращает bbox.
#   norm1000 — 0..1000 от левого верхнего угла (историческое поведение);
#   px       — пиксели картинки, как в блоке извлечённого текста.
ANNOTATE_COORD_SPACE: str = _env_str("ANNOTATE_COORD_SPACE", "norm1000")

# --- Текстовый слой PDF ---------------------------------------------------

# Доставать подписи из PDF и передавать их модели вместе с координатами.
EXTRACT_PDF_TEXT: bool = os.getenv("EXTRACT_PDF_TEXT", "1") not in {"0", "false", "False"}
# Отсечь совсем мелкий текст (высота шрифта в пикселях рендера).
PDF_TEXT_MIN_HEIGHT_PX: float = _env_float("PDF_TEXT_MIN_HEIGHT_PX", 0.0)
# Лимит подписей на лист в промте (0 = без лимита). Обрезание равномерное,
# чтобы не отрезать половину листа.
PDF_TEXT_MAX_ITEMS: int = _env_int("PDF_TEXT_MAX_ITEMS", 0)


# --- Разбор размеров (dimscan) ---------------------------------------------

# Искать размеры в векторе PDF до рендера и передавать их модели. Данные
# приходят из того же PDF и в тех же координатах, что и текстовый слой, но
# с готовой привязкой: какое число чему принадлежит. Выключается флагом
# --no-dimscan, если нужно сравнить прогоны с API и без него.
USE_DIMSCAN: bool = os.getenv("DIMSCAN", "1") not in {"0", "false", "False"}
# Минимальная длина размерной линии, в пунктах эталонного листа: порог сам
# пересчитывается под масштаб каждого листа, поэтому задаётся один на весь
# комплект. 0 — отключить порог, брать всё, что длиннее наконечника.
DIMSCAN_MIN_LENGTH: float = _env_float("DIMSCAN_MIN_LENGTH", 12.0)
# Передавать ли модели список чисел, отсеянных как «не размер» (позиционные
# знаки в квадратах, координаты привязки, марки труб). Это самый полезный
# кусок блока: по картинке «9» в квадрате от размера «9» не отличить, и без
# этого списка модель стабильно берёт лишнее в расчёт.
DIMSCAN_REJECTED: bool = os.getenv("DIMSCAN_REJECTED", "1") not in {"0", "false", "False"}
# Рисовать ли на листе пометки найденных размеров (рамки вокруг чисел,
# плашки с метками, подсветка размерных линий) и отправлять в модель
# подсвеченный снимок вместо исходного.
#
# Выключено по умолчанию **по замерам**: на десяти листах подсветка дала 5, 5, 7
# и 5 верных ответов против 6, 6, 6 и 7 без неё, а разброс суммарной ошибки
# вырос с 284–1529 мм до 1742–10021 мм. Гипотеза, почему так: без подсветки
# модель видит на чистом листе, какая линия внутри какой, и по этому выкидывает
# вложенные размеры; перерисовка всех линий одним цветом уравнивает их.
# Версия недоказанная — подтвердилась лишь для 6 из 11 исключений, которые
# модель делает. Код и тесты на месте, включить можно флагом --overlay.
DIMSCAN_OVERLAY: bool = os.getenv("DIMSCAN_OVERLAY", "0") in {"1", "true", "True"}

# --- Метки размеров на листе, которые уходят в модель ----------------------
#
# Отдельно от ``DIMSCAN_OVERLAY``, потому что подсветка линий и метки решают
# разные задачи. Подсветка перерисовывает линии и по замеру падает точность (см.
# ``DIMSCAN_OVERLAY`` выше). Метка ``P8`` у подписи ``378`` чертёж не меняет
# ни на пиксель, а задачу снимает: блок разбора спрашивает в ответе про ``P8``,
# и на листе 4 число 159 нарисовано четыре раза — P3, P7, P13 и P17, — так что
# без метки на листе выбрать нужное нечем, остаётся угадывать по координатам
# рамки из блока. Влияние меток на точность не измерено.
DIMSCAN_LABELS: bool = os.getenv("DIMSCAN_LABELS", "0") in {"1", "true", "True"}

# --- Зачистка листа от того, что размером не является -----------------------
#
# Шаг предобработки между разбором размеров и отправкой листа в модель: с
# картинки закрашиваются знаки в квадратах, опорные знаки «О3», выноски знаков
# и числа, которые разбор отверг (данные привязки, номера листов). Модель
# получает лист и список размеров одновременно, и всё, что на картинке похоже на
# размер, тянет её в расчёт — на девятом листе верны все пять размеров, но рядом
# лежат квадраты с «1» и «2», которые тоже видны.
#
# На диск идёт исходный лист из ``data``, в модель — зачищенный (см.
# ``PageImage.model_image``): зачистка нужна модели, а человеку для сверки
# нужна настоящая выдача. Закрашивание не трогает подписи найденных размеров:
# на всех десяти листах под проверкой не осталось ни одного задетого размера.
DIMSCAN_CLEAN: bool = os.getenv("DIMSCAN_CLEAN", "0") in {"1", "true", "True"}
DIMSCAN_CLEAN_FILTERS: tuple[str, ...] = tuple(
    part.strip() for part in _env_str("DIMSCAN_CLEAN_FILTERS",
                                      "signs,leaders,notes,rejected").split(",")
    if part.strip()
)

# --- Вырезы для сопоставления текста с графикой ---------------------------
#
# Вырез отправляется модели, чтобы она увидела элемент вместе с тем, что его
# окружает. Размер задан **долей меньшей стороны листа**, а не в пикселях:
# при смене ``RENDER_DPI`` вырез остаётся тем же куском листа, но в большем
# разрешении, и модель получает картинку без «шакальной» лестницы ступенек.
# Доля также снимает вопрос о размере листов: у комплекта они одинаковые,
# а если попадётся лист другого формата, пропорция сохранится.
#
# Доля 0.108 — это 91 пункт по стороне на листе A3, 728 пикселей при dpi 600.
# Меньше не хватало контекста: в вырезе не было видно, куда уходит выноска, и
# отличить стрелку знака от размера было не по чему. Больше не нужно — в кадр
# начинает попадать соседняя трасса вместе с её размерами, и вопрос «чья это
# стрелка» становится неоднозначным: при доле 0.165 в одном вырезе на листе 4
# оказывалось сразу девять указателей и алфавит уходил в двузначные метки.
PRUNE_CROP_RATIO: float = float(os.getenv("PRUNE_CROP_RATIO", "0.108"))
# Минимум по ширине элемента: вырез не может быть уже самой подписи, иначе
# модель не увидит, что перед ней.
PRUNE_CROP_MIN_RATIO: float = float(os.getenv("PRUNE_CROP_MIN_RATIO", "1.6"))
# Поле вокруг элемента внутри выреза, та же доля стороны листа.
PRUNE_CROP_PAD_RATIO: float = float(os.getenv("PRUNE_CROP_PAD_RATIO", "0.02"))
# Разрешение рендера для вырезов. Задаётся отдельно от ``RENDER_DPI``:
# вырезы маленькие, и им выгодно разрешение выше, чем целому листу, которого
# модель всё равно не может видеть в деталях.
PRUNE_CROP_DPI: int = int(os.getenv("PRUNE_CROP_DPI", "600"))

# --- Сопоставление подписи с линией ----------------------------------------
#
# Модель, которой показывают вырез вокруг подписи. Дешёвая и без глубоких
# размышлений: нужен один короткий ответ на один вырез, а не решение задачи.
# Размышления на ``low`` — это про скорость, а не про аккуратность: отвечать
# нужно по картинке, где всё крупно и однозначно.
PRUNE_JUDGE_MODEL: str = _env_str("PRUNE_JUDGE_MODEL", "glm-5.3-flash")
PRUNE_JUDGE_REASONING: str = _env_str("PRUNE_JUDGE_REASONING", "low")
# Вырезов одного листа одновременно. Ставится ниже, чем у основного прогона:
# запросов на лист десятки, и при десяти одновременных провайдер отвечает 429.
PRUNE_MAX_PARALLEL: int = _env_int("PRUNE_MAX_PARALLEL", 4)
# Порог уверенности, ниже которого решение модели не принимается и элемент
# остаётся на листе. Ложное удаление размера стоит дороже лишнего числа в
# сумме: потерянный размер не восстановить, а лишний номер узла модель
# отбросит сама, получив список настоящих размеров.
PRUNE_JUDGE_MIN_CONFIDENCE: float = float(
    os.getenv("PRUNE_JUDGE_MIN_CONFIDENCE", "0.5"))

# На сколько треугольник наконечника может отстоять от конца линии и всё ещё
# считаться наконечником **этой** линии. Указатель ищется по острию, а острие
# у части стрелок нарисовано поверх конца отрезка и уходит вперёд на длину
# треугольника — около трёх пунктов. Без запаса на листе 4 не находилось
# двадцати восьми выносок из тридцати шести: все они с наконечником, все
# отстоят от конца на 3.1 пункта, и все терялись вместе со знаками, которые
# они держали.
POINTER_ARROW_SLACK: float = float(os.getenv("POINTER_ARROW_SLACK", "4.0"))
# На сколько хвост указателя может стоять от подписи. Подпись может отстоять
# от хвоста на ширину рамки знака и на поле текста — иначе выноска к знаку
# «ОБОГРЕВ» не найдёт свою подпись и будет удалена вместе с ней.
POINTER_LABEL_GAP: float = float(os.getenv("POINTER_LABEL_GAP", "26.0"))

# --- Сопоставление подписи с её указателем ---------------------------------
#
# Вырезов в одном запросе к модели. Три — по разумению: два выреза модель
# сопоставляет уверенно, четыре уже путает метки между картинками, и на пятом
# ответ становится потоком догадок. Проверяется на листе 8 с наибольшим
# числом удаляемых элементов.
PRUNE_BATCH_SIZE: int = _env_int("PRUNE_BATCH_SIZE", 3)
# Разрешение, которым рисуются вырезы для сопоставления. Задаётся отдельно от
# ``RENDER_DPI``: в вырезе важен размер подписи в пикселях, а не размер листа.
PRUNE_LINK_DPI: int = _env_int("PRUNE_LINK_DPI", 600)

# --- Метки на вырезе -------------------------------------------------------
#
# Кегль метки. Пять с половиной пунктов при dpi 600 даёт подложку 87 на 36
# пикселей: это читаемо для модели и мало настолько, чтобы влезать в промежуток
# между линиями. Восемь пунктов дают 126 на 53, и такой подложке на плотных
# листах места не находится вовсе — из 573 вырезов чистыми выходили 223.
LABEL_SIZE_PT: float = float(os.getenv("LABEL_SIZE_PT", "5.5"))
# На сколько пикселей линия считается препятствием для метки. Три — с запасом
# на серый от сглаживания по краю буквы: подложка вплотную к графике выглядит
# как разрыв в линии.
LABEL_BUFFER_PX: int = _env_int("LABEL_BUFFER_PX", 3)
# На каком расстоянии от подписи искать место для метки. Сто пятьдесят
# пикселей при 600 dpi — это 18 пунктов: достаточно, чтобы обойти плотную
# графику, и мало, чтобы метка не улетела от своей подписи. При радиусе в 60
# пикселей из 573 вырезов чистыми выходили 455, при ста пятидесяти — все.
LABEL_SEARCH_PX: int = _env_int("LABEL_SEARCH_PX", 150)


# --- Пути -----------------------------------------------------------------

RESEARCH_DIR: Path = _env_path("RESEARCH_DIR", PROJECT_ROOT / "research")
DEFAULT_EXPERIMENT: str = _env_str("EXPERIMENT", "exp_01")
OUTPUT_DIR: Path = _env_path("OUTPUT_DIR", PROJECT_ROOT / "output")
