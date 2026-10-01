"""Проверки учёта вызовов LLM (запуск из корня проекта, без pytest и без API):

    python tests/test_llm_usage.py

Повторы у конвейера бывают двух сортов — упал запрос (сеть, шлюз, таймаут) или
упала проверка ответа (модель вернула не JSON или не по схеме). Чинятся они
противоположно, а по счётчику повторов неразличимы, поэтому причина обязана
попадать в статистику прогона. Именно это здесь и проверяется.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isokrus.llm import Usage, _short_error

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  OK    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")


print("1. причина повтора попадает в статистику")
usage = Usage()
check("повторов нет — список пуст", usage.as_dict()["retry_reasons"] == [])
usage.note_retry("запрос: 502 Bad Gateway")
usage.note_retry("ответ не прошёл проверку: не JSON")
data = usage.as_dict()
check("причин столько же, сколько повторов",
      len(data["retry_reasons"]) == 2, str(data["retry_reasons"]))
check("причины видны в as_dict()", "502" in data["retry_reasons"][0])
check("счётчик повторов не сломан", data["retries"] == 0)

print("2. слияние счётчиков не теряет причины")
other = Usage()
other.note_retry("ответ не прошёл проверку: не по схеме")
usage.merge(other)
check("причины перенеслись", len(usage.as_dict()["retry_reasons"]) == 3)

print("3. тексты причин различают разные поломки")
empty = _short_error(ValueError("x"), "   ")
check("пустой content опознан", "пустой" in empty, empty)

try:
    json.loads("{не json")
except json.JSONDecodeError as exc:
    broken = _short_error(exc, "{не json")
check("не-JSON опознан", "не JSON" in broken, broken)

from pydantic import BaseModel, ValidationError


class Sample(BaseModel):
    value: int


try:
    Sample(value="не число")  # type: ignore[arg-type]
except ValidationError as exc:
    schema_bad = _short_error(exc, '{"value": "не число"}')
check("несоответствие схеме опознано", "не по схеме" in schema_bad, schema_bad)
check("в тексте есть путь до поля", "value" in schema_bad, schema_bad)

# Оборванный ответ и пустой — разные поломки, и лечатся они по-разному.
cut = _short_error(ValueError("x"), '{"branches": [{"axis": "Z"', "length")
check("обрыв на лимите опознан", "оборван" in cut, cut)
check("обрыв отличается от пустого", "пустой" not in cut, cut)
check("finish_reason попал в текст", "length" in cut, cut)

print("4. перегрузка 429 — это повтор, а не фатальная ошибка")
from isokrus.llm import _is_fatal, _is_retryable  # noqa: E402

concurrency = Exception(
    "Error code: 429 - {'error': {'type': 'too_many_concurrent_requests', "
    "'message': 'Concurrent request limit reached for this API key. Retry "
    "after active requests finish.', 'code': 'api_key_concurrency_limit'}}"
)
# Ключевой случай: текст 429 у этого провайдера упоминает «API key», и наивная
# проверка на ключ объявляла перегрузку фатальной — лист падал без повторов.
check("429 распознан как повторяемый", _is_retryable(concurrency))
check("429 НЕ фатален", not _is_fatal(concurrency))

for text, why in (
    ("429 Too Many Requests", "простой лимит запросов"),
    ("Rate limit exceeded for model", "лимит частоты"),
    ("Read timed out", "таймаут чтения"),
    ("503 Service Unavailable", "сервис недоступен"),
    ("overloaded_error", "перегрузка сервера"),
):
    check(f"повторяемо: {why}", _is_retryable(Exception(text)), text)
    check(f"не фатально: {why}", not _is_fatal(Exception(text)), text)

print("5. настоящие фатальные ошибки остались фатальными")
for text in (
    "Invalid API key provided",
    "You do not have access to model gpt-9",
    "insufficient_quota: billing",
    "context_length_exceeded",
):
    check(f"фатально: {text}", _is_fatal(Exception(text)), text)
    check(f"не повторяем: {text}", not _is_retryable(Exception(text)), text)

print()
if FAILURES:
    print(f"ПРОВАЛЕНО: {', '.join(FAILURES)}")
    raise SystemExit(1)
print("Все проверки пройдены.")
