"""Разрешение ссылок на размеры по их меткам ``P7``.

Зачем. В ``exp_04`` поле ``parent_length`` означало сразу два разных числа:
и «обобщённый размер, нарисованный на листе», и «сумму группы отрезков,
которую посчитали». Разобрать это нечем, и на десяти листах это дало основную
часть ошибок — на листах 1 и 9 модели склеивали 900 + 200 в «1100» и 123 + 244
+ 123 в «490», и наоборот там, где из листа надо выбросить вложенные размеры,
та же склейка считала их дважды.

Здесь ответ со ссылками на метки превращается в плоский список участков с
числами, которые дальше обрабатываются как обычно: сумму считает программа,
разметка обводит настоящие подписи. Модель при этом не может назвать число,
которого на листе нет, — она выбирает из списка.

Преобразование возвращает **новый** ответ: исходный словарь не меняется,
и в ``responses.json`` остаётся то, что вернула модель.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # цикл: extract -> dimscan, indices -> extract
    from .extract import PageImage

# Ключи ответа, по которым модель ссылается на метки размеров.
_INCLUDED_KEYS = ("included", "taken", "kept", "in_total", "counted")
_EXCLUDED_KEYS = ("excluded", "dropped", "nested", "not_counted", "skipped")

# Поля, которые могут содержать метку в записи об исключении.
_INDEX_FIELDS = ("index", "mark", "id", "label", "dimension", "size")


def _reference(value: Any) -> str | None:
    """Метка размера из значения ответа: строка или первый подходящий ключ."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in _INDEX_FIELDS:
            found = value.get(key)
            if isinstance(found, str) and found.strip():
                return found.strip()
            if isinstance(found, (int, float)) and not isinstance(found, bool):
                return f"P{int(found)}"
    return None


def _reason(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("reason", "why", "note", "comment", "notes"):
            found = value.get(key)
            if found:
                return str(found).strip()
        return ""
    return str(value).strip() if value is not None else ""


def _find_list(response: dict, keys: tuple[str, ...]) -> list[Any] | None:
    for key in keys:
        value = response.get(key)
        if isinstance(value, list):
            return value
    return None


def looks_indexed(response: dict | None) -> bool:
    """Ответ построен как выбор размеров по меткам."""
    if not isinstance(response, dict):
        return False
    return _find_list(response, _INCLUDED_KEYS) is not None


def resolve_dimension_indices(
    response: dict | None,
    page: "PageImage | None",
) -> dict | None:
    """Ответ со ссылками на метки -> плоский ответ с числами.

    Возвращает ``None``, если ответа нет, он не построен по меткам или на
    листе нет разбора размеров: тогда вызывающий код работает как раньше, и
    эксперимент без разбора не ломается.
    """
    if not looks_indexed(response) or page is None or not page.dimensions:
        return None
    assert response is not None  # для mypy: следствие looks_indexed

    by_id = {mark.line_id: mark for mark in page.dimensions}

    def segment(mark_id: str) -> dict | None:
        mark = by_id.get(mark_id)
        if mark is None:
            return None
        return {
            "mark": mark.line_id,
            "value": float(mark.value),
            "label": mark.label,
            "kind": mark.kind,
        }

    raw_included = _find_list(response, _INCLUDED_KEYS) or []
    raw_excluded = _find_list(response, _EXCLUDED_KEYS) or []

    segments: list[dict] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for item in raw_included:
        reference = _reference(item)
        if reference is None:
            continue
        if reference not in by_id:
            unknown.append(reference)
            continue
        # Повторная метка означает, что размер посчитан дважды: на листе он
        # один, и второй раз его добавлять нельзя.
        if reference in seen:
            continue
        seen.add(reference)
        found = segment(reference)
        if found is not None:
            segments.append(found)

    excluded: list[dict] = []
    dropped: set[str] = set()
    for item in raw_excluded:
        reference = _reference(item)
        if reference is None:
            continue
        entry = {
            "mark": reference,
            "reason": _reason(item),
            "known": reference in by_id,
        }
        if reference in by_id:
            entry["value"] = float(by_id[reference].value)
            dropped.add(reference)
        else:
            unknown.append(reference)
        excluded.append(entry)

    # Метка, о которой модель не сказала ничего, считается включённой.
    # Ответ читается как список исключений, а не как список участков: молча
    # выбрасывать размер, который модель просто не упомянула, — это потерять
    # на десятом листе 2450 мм. На всех прогонах забытая метка была ровно одна,
    # и ровно на неё пришлось всё расхождение.
    unclassified = [
        mark_id for mark_id in by_id
        if mark_id not in seen and mark_id not in dropped
    ]
    for mark_id in unclassified:
        found = segment(mark_id)
        if found is not None:
            found["unclassified"] = True
            segments.append(found)

    resolved: dict[str, Any] = {
        "segments": segments,
        "excluded": excluded,
        "included_marks": [s["mark"] for s in segments],
        "unclassified_marks": unclassified,
        "unresolved_marks": sorted(set(unknown)),
        "notes": str(response.get("notes") or ""),
    }
    # Остальные поля ответа не теряем: вдруг модель вернула что-то ещё
    # полезное (bbox, формулу), и это не должно молча исчезнуть.
    for key, value in response.items():
        if key not in resolved:
            resolved.setdefault(key, value)
    return resolved
