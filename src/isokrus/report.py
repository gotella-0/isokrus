"""Формирование итоговой таблицы (CSV / Excel) по обработанным листам.

Формат полей соответствует ТЗ: номер листа, обозначение линии, длина в мм и м,
формула расчёта, статус и замечания. Колонки берутся из ответа LLM по ключам,
перечисленным в :data:`COLUMN_SPEC` — если конкретный эксперимент использует
другие имена полей, спецификацию можно переопределить на уровне модуля.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .indices import resolve_dimension_indices
from .pipeline import PageResult, RunResult

# (заголовок в отчёте, ключи ответа по приоритету)
COLUMN_SPEC: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Лист", ("sheet_number", "sheet", "page_number", "list_number")),
    ("Обозначение линии", ("line_designation", "line", "designation", "line_id", "marking")),
    ("Длина, мм", ("total_length_mm", "length_mm", "total_mm", "length")),
    ("Длина, м", ("total_length_m", "length_m", "total_m")),
    ("Формула", ("formula", "calculation", "computation", "calculation_formula")),
    ("Участков", ("segments_count", "segment_count", "segments_total")),
    ("Статус", ("status", "confidence", "state")),
    ("Замечания", ("notes", "issues", "remarks", "warnings", "comment")),
)

CSV_NAME = "result.csv"
XLSX_NAME = "result.xlsx"
STATS_NAME = "stats.json"


@dataclass
class Row:
    """Строка итоговой таблицы + сырые поля для служебных нужд."""

    values: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)


def _first(node: dict, keys: Sequence[str]) -> Any:
    for key in keys:
        if key in node and node[key] not in (None, "", [], {}):
            return node[key]
    return None


def _flatten_notes(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v).strip() for v in value if str(v).strip())
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def _as_number(value: Any) -> Any:
    """Приводит к числу, если это возможно; суммы в мм считаем программно."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.replace(",", ".").strip()
        try:
            return float(cleaned)
        except ValueError:
            return value
    return value


def _segments_of(response: dict) -> list[dict]:
    """Список участков трассы. Имя ключа зависит от схемы эксперимента,
    поэтому перебираем известные синонимы."""
    for key in ("segments", "dimensions", "parts", "sections", "runs", "branches", "items"):
        value = response.get(key)
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
    return []


_LENGTH_KEYS = ("length_mm", "length", "value_mm", "total_mm", "value")

# Иерархическая схема: ветви несут обобщённую длину и вложенные участки.
# Итог — сумма только корневых (parent) значений: дочерние уже учтены внутри
# них, и складывать всё вместе значит посчитать одни и те же миллиметры дважды.
_PARENT_KEYS = (
    "parent_length", "parent_total", "overall_length", "summary_length",
    "total_length", "total_length_mm", "total_mm",
)
_BRANCH_KEYS = ("branches", "legs", "runs", "routes", "sections")


def _branch_list(response: dict) -> list[dict]:
    """Ветви из ответа — по любому известному имени списка."""
    for key in _BRANCH_KEYS:
        value = response.get(key)
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
    return []


def _parent_length(branch: dict) -> Any:
    for key in _PARENT_KEYS:
        value = branch.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _hierarchy_summary(response: dict) -> tuple[list[float], list[str]] | None:
    """Корневые длины ветвей, если ответ построен как иерархия.

    Возвращает ``None``, если ни в одной ветви нет обобщённой длины — тогда
    считаем по-старому, суммой плоских участков.
    """
    branches = _branch_list(response)
    if not branches:
        return None
    roots = [v for v in (_parent_length(b) for b in branches) if isinstance(v, (int, float))]
    if not roots:
        return None
    return [float(v) for v in roots], [b.get("branch_id", "") for b in branches]


def _hierarchy_notes(response: dict) -> str:
    parts: list[str] = []
    for branch in _branch_list(response):
        name = branch.get("branch_id") or "?"
        axis = branch.get("axis") or "?"
        note = str(branch.get("notes") or "").strip()
        if note:
            parts.append(f"{name} ({axis}): {note}")
    return "; ".join(parts)


def _segment_length(segment: dict) -> Any:
    """Длина участка из первого подходящего поля."""
    for key in _LENGTH_KEYS:
        value = segment.get(key)
        if value is not None:
            return _as_number(value)
    return None


def _segment_lengths(segments: Iterable[dict]) -> list[tuple[Any, dict]]:
    return [(v, s) for s in segments if (v := _segment_length(s)) is not None]


def _sum_segments(segments: Iterable[dict]) -> tuple[float, list[str]]:
    """Суммирует длины участков программно — по ТЗ суммирование не на LLM."""
    total = 0.0
    problems: list[str] = []
    for index, segment in enumerate(segments, start=1):
        value = _segment_length(segment)
        if isinstance(value, (int, float)):
            total += float(value)
        else:
            problems.append(f"участок {index}: длина не определена")
    return total, problems


def build_rows(results: Sequence[PageResult]) -> list[Row]:
    """Построить строки таблицы по страницам (включая ошибочные)."""
    rows: list[Row] = []

    for result in results:
        if not result.ok:
            rows.append(
                Row(
                    values={
                        "Лист": result.page.page_number,
                        "Обозначение линии": "",
                        "Длина, мм": "",
                        "Длина, м": "",
                        "Формула": "",
                        "Участков": "",
                        "Статус": "ошибка",
                        "Замечания": result.error or "нет ответа модели",
                    },
                    raw={"page_number": result.page.page_number, "error": result.error},
                )
            )
            continue

        response = result.response or {}
        # Ответ со ссылками на метки размеров (P7) разворачиваем в плоский
        # список с числами: дальше сумму считает программа, а модель не может
        # назвать число, которого на листе нет.
        resolved = resolve_dimension_indices(response, result.page)
        if resolved is not None:
            response = resolved
        segments = _segments_of(response)
        hierarchy = _hierarchy_summary(response)
        computed_total, problems = _sum_segments(segments)

        values: dict[str, Any] = {}
        for title, keys in COLUMN_SPEC:
            found = _first(response, keys)
            if title in {"Длина, мм"} and found is None and computed_total:
                found = computed_total  # подставляем программную сумму
            values[title] = found

        values["Лист"] = _first(response, ("sheet_number", "sheet", "list_number")) or result.page.page_number
        values["Длина, мм"] = _as_number(values["Длина, мм"])
        length_mm = values["Длина, мм"]

        if hierarchy is not None:
            # Иерархия: складываем только корневые длины ветвей. Дочерние
            # участки внутри них уже учтены, поэтому в сумму не идут.
            roots, _names = hierarchy
            computed_total = sum(roots)
            values["Длина, мм"] = computed_total
            length_mm = computed_total
            values["Формула"] = (
                f"{' + '.join(f'{v:g}' for v in roots)} = {computed_total:g} мм"
            )
            values["Участков"] = len(_branch_list(response))
            remarks = _hierarchy_notes(response)
            if remarks:
                values["Замечания"] = "; ".join(filter(None, [values.get("Замечания"), remarks]))
        else:
            if values.get("Длина, м") in (None, "") and isinstance(length_mm, (int, float)):
                # Метры считаем программно, если модель их не вернула.
                values["Длина, м"] = round(float(length_mm) / 1000.0, 3)
            if values.get("Формула") in (None, "") and segments:
                # Формулу собираем программно — по ТЗ суммирование не на LLM.
                # В формуле стоит вычисленная сумма, а не заявленная моделью:
                # иначе «200 + 320 = 900» выглядело бы как арифметической ошибкой.
                parts = " + ".join(
                    f"{v:g}" for v, _ in _segment_lengths(segments)
                )
                values["Формула"] = f"{parts} = {computed_total:g} мм"
            values["Участков"] = len(segments) or values.get("Участков", "")

        if hierarchy is not None and isinstance(length_mm, (int, float)):
            values["Длина, м"] = round(float(length_mm) / 1000.0, 3)

        values["Замечания"] = _flatten_notes(values.get("Замечания"))
        if problems and hierarchy is None:
            values["Замечания"] = "; ".join(
                filter(None, [values["Замечания"], "; ".join(problems)])
            )
        if resolved is not None:
            # Метки, которых нет на листе, — это уже не вопрос точности, а
            # вопрос доверия к ответу: модель сослалась на размер, которого
            # разбор не нашёл. Молча выкидывать такое нельзя.
            unresolved = resolved.get("unresolved_marks") or []
            dropped = resolved.get("excluded") or []
            bits = []
            if unresolved:
                bits.append(
                    "в ответе есть метки размеров, которых нет на листе: "
                    + ", ".join(unresolved)
                )
            if dropped:
                bits.append(f"исключено размеров: {len(dropped)}")
            notes = resolved.get("notes") or ""
            if notes:
                bits.append(str(notes))
            values["Замечания"] = "; ".join(filter(None, [values["Замечания"], *bits]))

        # Контроль: заявленная моделью сумма против программной.
        if computed_total and isinstance(length_mm, (int, float)):
            delta = abs(float(length_mm) - computed_total)
            if delta > 1.0:
                values["Статус"] = f"расхождение {delta:.0f} мм"
                values["Замечания"] = "; ".join(
                    filter(
                        None,
                        [
                            values["Замечания"],
                            f"модель указала {length_mm} мм, сумма участков {computed_total:.0f} мм",
                        ],
                    )
                )

        rows.append(
            Row(
                # Схема эксперимента может не содержать ни обозначения линии,
                # ни статуса: пустая ячейка честнее, чем None в CSV.
                values={title: values.get(title) if values.get(title) is not None else ""
                        for title, _ in COLUMN_SPEC},
                raw={"page_number": result.page.page_number, **response},
            )
        )

    # Таблица должна идти по порядку листов, а не в порядке завершения
    # параллельных вызовов.
    rows.sort(key=lambda r: _sheet_sort_key(r))
    return rows


def _sheet_sort_key(row: Row) -> tuple[int, float, int]:
    """Ключ сортировки: номер листа числом, иначе как строка, иначе по странице."""
    raw = row.raw.get("sheet_number", row.values.get("Лист", ""))
    number = _as_number(raw)
    if isinstance(number, (int, float)):
        return (0, float(number), row.raw.get("page_number", 0))
    if isinstance(raw, str) and raw.strip():
        return (1, 0.0, row.raw.get("page_number", 0))
    return (2, 0.0, row.raw.get("page_number", 0))


def write_csv(rows: Sequence[Row], path: str | Path) -> Path:
    """CSV в UTF-8 с BOM, чтобы Excel корректно открыл кириллицу."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    headers = [title for title, _ in COLUMN_SPEC]
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.values.get(h, "") for h in headers})
    return target


def write_xlsx(rows: Sequence[Row], path: str | Path) -> Path | None:
    """Excel с двумя листами: результат и статистика. None, если нет openpyxl."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
        from openpyxl.utils import get_column_letter
    except ImportError:
        return None

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    headers = [title for title, _ in COLUMN_SPEC]
    workbook = Workbook()

    sheet = workbook.active
    sheet.title = "Результат"
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in rows:
        sheet.append([row.values.get(h, "") for h in headers])
    for index, title in enumerate(headers, start=1):
        width = max(len(title) + 2, *(len(str(r.values.get(title, ""))) + 2 for r in rows or [Row({})]))
        sheet.column_dimensions[get_column_letter(index)].width = min(width, 70)
    sheet.freeze_panes = "A2"
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    stats = workbook.create_sheet("Статистика")
    stats.append(["Показатель", "Значение"])
    for cell in stats[1]:
        cell.font = Font(bold=True)
    stats.column_dimensions["A"].width = 32
    stats.column_dimensions["B"].width = 26
    workbook.save(target)
    return target


def build_stats(run_result: RunResult) -> dict:
    """Метрики прогона: время, обращения на страницу, стоимость."""
    from . import config

    total_pages = len(run_result.pages)
    usage = run_result.usage.as_dict()
    ok_pages = len(run_result.succeeded)
    trimmed_pages = [r for r in run_result.pages if r.page.crop_box]
    model = run_result.experiment.model or config.LLM_MODEL
    _pricing = config.pricing_for(model)

    stats = {
        "experiment": run_result.experiment.name,
        "pdf": str(run_result.pdf_path),
        "model": model,
        "reasoning_effort": run_result.experiment.reasoning_effort
        or config.LLM_REASONING_EFFORT,
        "pages_total": total_pages,
        "pages_ok": ok_pages,
        "pages_failed": total_pages - ok_pages,
        "duration_seconds": round(run_result.duration, 2),
        "seconds_per_page": round(run_result.duration / total_pages, 2) if total_pages else 0,
        "calls_per_page": round(usage["calls"] / total_pages, 2) if total_pages else 0,
        "retries": usage["retries"],
        # Причины повторов: сетевая ошибка лечится ретраем, невалидный ответ
        # модели — промтом и схемой. По одному счётчику их не различить.
        "retry_reasons": usage.get("retry_reasons", []),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "visible_completion_tokens": usage["visible_completion_tokens"],
        "cost_input": usage["input_cost"],
        "cost_output": usage["output_cost"],
        "cost_reasoning": usage["reasoning_cost"],
        "cost_total": usage["cost"],
        "cost_per_page": round(usage["cost"] / total_pages, 6) if total_pages else 0,
        # Стоимость компонентов, включая кеш: сумма кешных токенов должна
        # сходиться с биллингом провайдера, иначе расчёт нельзя проверить.
        "cost_cache_write": usage.get("cache_write_cost", 0.0),
        "cost_cache_read": usage.get("cache_read_cost", 0.0),
        "currency": config.PRICING_CURRENCY,
        # Тарифы берём для той модели, что реально работала, а не для
        # настроек по умолчанию: модель переключается флагом --model, и при
        # общих ценах отчёт показывал бы стоимость дорогой модели по тарифу
        # дешёвой. Если тариф неизвестен — cost_total ниже помечается.
        "pricing_per_mtok": {
            "input": _pricing.input,
            "output": _pricing.output,
            "reasoning": _pricing.reasoning,
            "cache_write": _pricing.cache_write,
            "cache_read": _pricing.cache_read,
            "source": _pricing.source,
        },
        "cost_known": usage.get("pricing_known", True),
        "cache_write_tokens": usage.get("cache_write_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_tokens", 0),
        "parallel_limit": config.MAX_PARALLEL,
        # Факт прогона, а не значение переменной окружения: с --no-trim лист
        # остаётся полным, и в статистике должно быть видно именно это.
        "trim_margins": bool(trimmed_pages),
        "pages_trimmed": len(trimmed_pages),
        # Разбор размеров: сколько нашли и сколько отбросили как «не размер».
        # Ноль найденных при включённом разборе — это подозрительно и должно
        # быть видно в прогоне, а не всплыть потом в сумме.
        "dimscan_enabled": any(
            (r.page.meta.get("dimscan") for r in run_result.pages)
        ),
        "dimensions_total": sum(len(r.page.dimensions) for r in run_result.pages),
        "dimensions_pointers": sum(
            1 for r in run_result.pages for d in r.page.dimensions if d.kind != "span"
        ),
        "rejected_numbers_total": sum(
            len(r.page.rejected_numbers) for r in run_result.pages
        ),
        "pages_dimscan_failed": [
            r.page.page_number for r in run_result.pages if r.page.meta.get("dimscan_error")
        ],
        "image_area_ratio_mean": round(
            sum(r.page.area_ratio for r in run_result.pages) / total_pages, 4
        ) if total_pages else 1.0,
    }
    for result in run_result.pages:
        if result.error:
            stats.setdefault("errors", []).append(
                {"page": result.page.page_number, "error": result.error}
            )
    return stats
