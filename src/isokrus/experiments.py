"""Загрузка экспериментов из папки ``research/``.

Структура эксперимента (имя папки = номер/идентификатор эксперимента)::

    research/
      exp_01/
        meta.json        # опционально: model, reasoning_effort, описание
        system.md        # системный промт
        user.md          # опционально: текст пользовательской инструкции
        response.json    # JSON Schema ответа (конвертируется в pydantic-модель)

Также поддерживается ``system.txt`` и ``response.schema.json`` как алиасы.
Запуск конкретной версии: ``--experiment exp_02`` или ``--experiment ./research/exp_02``.

``system.md`` может содержать токен ``{{PDF_TEXT}}`` — на его место
подставляется блок подписей, извлечённых из PDF вместе с координатами
(см. ``textmap.py``), и токен ``{{DIMENSIONS}}`` — блок размеров, найденных
разбором вектора (см. ``dimscan.py``). Если токена нет, блок дописывается
в конец промта. Оба выключаются в ``meta.json`` ключами ``pdf_text`` и
``dimensions``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from . import config
from .dimscan import DIMENSIONS_TOKEN, format_dimensions_block
from .errors import ExperimentError
from .schema import load_schema_file, schema_to_model
from .textmap import PDF_TEXT_TOKEN, format_text_block

if TYPE_CHECKING:  # избегаем цикла: extract -> dimscan/textmap, experiments -> extract
    from .extract import PageImage

SYSTEM_NAMES = ("system.md", "system.txt", "system_prompt.md", "system_prompt.txt")
USER_NAMES = ("user.md", "user.txt", "prompt.md")
RESPONSE_NAMES = (
    "response.json",
    "response.schema.json",
    "schema.json",
    "response_model.json",
)
META_NAMES = ("meta.json", "experiment.json", "config.json")

EXPERIMENT_RE = re.compile(r"^exp[_-]?\d+", re.IGNORECASE)


# Заголовок, с которого начинается раздел про размеры, и абзацы после него.
# Нужны, чтобы убрать раздел целиком, когда разбор выключен, а не оставить
# заголовок с токеном посередине.
_DIMENSIONS_HEADING = re.compile(
    r"\n#+\s*(Разобранные размеры|Размеры, найденные на листе).*",
    re.IGNORECASE,
)


def _has_dimension_scan(page: "PageImage | None") -> bool:
    """Разбор размеров на листе действительно выполнялся.

    Отличать «выключено» от «выполнилось и ничего не на��ёл» обязательно: в
    первом случае блок в промте не нужен, во втором — нужен и должен прямо
    сказать, что это дефект разбора.
    """
    if page is None:
        return False
    return bool(page.meta.get("dimscan") or page.meta.get("dimscan_error"))


def _strip_dimensions_section(text: str) -> str:
    """Убрать раздел про размеры, если разбор выключен."""
    match = _DIMENSIONS_HEADING.search(text)
    if match is None:
        return text
    return text[: match.start()].rstrip() + "\n"


@dataclass
class Experiment:
    """Готовая к использованию связка: промт + схема ответа."""

    name: str
    path: Path
    system: str
    response_model: type[BaseModel]
    user: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    notes: str = ""
    files: list[str] = field(default_factory=list)
    use_pdf_text: bool = True  # подставлять извлечённый текст в system.md
    use_dimensions: bool = True  # подставлять разобранные размеры в system.md

    def render_system(self, page: "PageImage | None" = None) -> str:
        """Системный промт эксперимента, готовый к отправке.

        Два блока данных о листе, каждый со своим токеном:

        * ``{{PDF_TEXT}}`` — все подписи шрифта с координатами;
        * ``{{DIMENSIONS}}`` — размеры, найденные разбором вектора, с их
          рамками и **списком отсеянных чисел** (позиционные знаки,
          координаты привязки, марки труб).

        Второй блок важнее первого для расчёта: по картинке нельзя отличить
        размер «9» от позиционного знака «9» в квадрате, и без разбора модель
        стабильно берёт лишнее в сумму. Но он не диктует решение — что
        обобщённый, а что вложенный, модель по-прежнему разбирает сама,
        потому что рисунок не в масштабе и по геометрии это не определить.

        Токена в промте нет — блок дописывается в конец, как и ``{{PDF_TEXT}}:
        старые эксперименты продолжают работать.
        """
        if page is None:
            return self.system
        text = self.system
        if self.use_pdf_text:
            block = format_text_block(page.text_items, page.width, page.height)
            text = (
                text.replace(PDF_TEXT_TOKEN, block)
                if PDF_TEXT_TOKEN in text
                else f"{text}\n\n{block}"
            )
        if self.use_dimensions and _has_dimension_scan(page):
            block = format_dimensions_block(
                page.dimensions,
                page.rejected_numbers,
                with_rejected=config.DIMSCAN_REJECTED,
            )
            text = (
                text.replace(DIMENSIONS_TOKEN, block)
                if DIMENSIONS_TOKEN in text
                else f"{text}\n\n{block}"
            )
        elif DIMENSIONS_TOKEN in text:
            # Токен в промте есть, а разбор выключен: молчаливый блок «размеров
            # не найдено» хуже отсутствия блока — в нём прямо сказано, что это
            # дефект разбора, и модель начнёт оправдываться за него. Поэтому
            # убираем и сам заголовок, и оставшиеся после него абзацы.
            text = _strip_dimensions_section(text)
        return text


def _first_existing(folder: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def resolve_experiment_dir(name_or_path: str | Path) -> Path:
    """Имя эксперимента (``exp_01``) -> путь к папке в research/."""
    candidate = Path(name_or_path)
    if candidate.is_dir():
        return candidate
    if candidate.parent != Path(".") and candidate.is_file():
        return candidate.parent

    research = config.RESEARCH_DIR
    folder = research / str(name_or_path)
    if folder.is_dir():
        return folder

    available = list_experiments()
    raise ExperimentError(
        f"Эксперимент {name_or_path!r} не найден в {research}. "
        f"Доступные: {', '.join(available) if available else '—'} "
        f"(создайте папку с system.md и response.json)"
    )


def list_experiments(research_dir: Path | None = None) -> list[str]:
    """Список доступных экспериментов, отсортированный по номеру."""
    root = research_dir or config.RESEARCH_DIR
    if not root.is_dir():
        return []
    names = [
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and (SYSTEM_NAMES[0] in {f.name for f in entry.iterdir()} or EXPERIMENT_RE.match(entry.name))
    ]
    return sorted(names, key=lambda n: (int(re.search(r"\d+", n).group()) if re.search(r"\d+", n) else 0, n))


def load_experiment(name_or_path: str | Path) -> Experiment:
    """Прочитать папку эксперимента и собрать промт + pydantic-модель ответа."""
    folder = resolve_experiment_dir(name_or_path)

    system_file = _first_existing(folder, SYSTEM_NAMES)
    if system_file is None:
        raise ExperimentError(
            f"В {folder} нет системного промта (ожидался один из: {', '.join(SYSTEM_NAMES)})"
        )
    system = system_file.read_text(encoding="utf-8").strip()
    if not system:
        raise ExperimentError(f"Системный промт {system_file.name} пуст")

    response_file = _first_existing(folder, RESPONSE_NAMES)
    if response_file is None:
        raise ExperimentError(
            f"В {folder} нет схемы ответа (ожидался один из: {', '.join(RESPONSE_NAMES)})"
        )
    schema = load_schema_file(response_file)
    try:
        model_cls = schema_to_model(schema, model_name=response_file.stem or "Response")
    except ExperimentError:
        raise
    except Exception as exc:  # pydantic/model_rebuild падает по-разному
        raise ExperimentError(f"Схема {response_file.name} не конвертируется: {exc}") from exc

    user_file = _first_existing(folder, USER_NAMES)
    user = user_file.read_text(encoding="utf-8").strip() if user_file else None

    meta: dict = {}
    meta_file = _first_existing(folder, META_NAMES)
    if meta_file is not None:
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"{meta_file.name}: некорректный JSON — {exc}") from exc
        if not isinstance(meta, dict):
            raise ExperimentError(f"{meta_file.name}: ожидался JSON-объект")

    return Experiment(
        name=folder.name,
        path=folder,
        system=system,
        response_model=model_cls,
        user=user,
        model=meta.get("model") or meta.get("llm_model"),
        reasoning_effort=meta.get("reasoning_effort") or meta.get("reasoning"),
        notes=str(meta.get("description") or meta.get("notes") or ""),
        files=sorted(f.name for f in folder.iterdir() if f.is_file()),
        use_pdf_text=bool(meta.get("pdf_text", True)),
        use_dimensions=bool(meta.get("dimensions", meta.get("dimscan", True))),
    )
