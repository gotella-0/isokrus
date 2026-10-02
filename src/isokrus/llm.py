"""Слой обращения к LLM через OpenAI-совместимый API.

Ключевые требования:
  * ответ всегда в формате JSON, и всегда strict (``json_schema`` +
    ``strict: true``) — никакого prompt-enforced JSON;
  * сетевые/серверные сбои → повтор ровно того же запроса;
  * ответ получен, но JSON невалиден → повтор с возвратом ошибки модели;
  * режим размышления по умолчанию ``high`` (через ``extra_body.reasoning``;
    у части моделей вместо уровня передаётся флаг включения — см.
    ``config.uses_reasoning_switch``);
  * картинка передаётся как base64 в content-части user-сообщения.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Iterable

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from . import config
from .errors import ConfigError, LLMError
from .schema import model_to_strict_schema
from .usage import USAGE, Usage, _estimate_cost

__all__ = ["Usage", "USAGE", "call_structured", "get_client", "build_user_content"]

# Один клиент на процесс: openai-клиент thread-safe, а лимиты на соединение
# переиспользовать выгоднее, чем открывать новый на каждый вызов.
_client: OpenAI | None = None
_client_lock = threading.Lock()

# Сколько символов невалидного ответа возвращаем модели в repair-раунде.
_REPAIR_ECHO_LIMIT = 6000


def get_client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            if not config.LLM_API_KEY:
                raise ConfigError(
                    "Не задан LLM_API_KEY. Создайте .env (см. .env.example) "
                    "или передайте переменную окружения."
                )
            _client = OpenAI(
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_BASE_URL,
                timeout=config.LLM_TIMEOUT,
                max_retries=0,  # ретраи делаем сами, чтобы видеть, что именно сломалось
            )
        return _client


# --- Вспомогательное -------------------------------------------------------


def _sleep_backoff(attempt: int) -> None:
    """Экспоненциальная пауза перед повтором (потолок — 30 с)."""
    delay = min(config.LLM_RETRY_BACKOFF * (2**attempt), 30.0)
    if delay > 0:
        time.sleep(delay)


def _extract_json(text: str) -> dict:
    """Достать JSON из ответа: чистый, в markdown-блоке или с обвязкой."""
    text = text.strip()
    if not text:
        raise json.JSONDecodeError("Пустой ответ", "", 0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1).strip())

    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("В ответе не найден JSON", text, 0)


def _validate(text: str, response_model: type[BaseModel]) -> BaseModel:
    return response_model.model_validate(_extract_json(text))


def _repair_messages(
    base: list[dict], raw_text: str, error: Exception, response_model: type[BaseModel]
) -> list[dict]:
    """Переписка для повтора: ответ модели + текст ошибки валидации.

    Собирается заново от ``base`` на каждой попытке, чтобы диалог не рос.
    """
    schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
    if len(raw_text) > _REPAIR_ECHO_LIMIT:
        raw_text = raw_text[:_REPAIR_ECHO_LIMIT] + "\n... (ответ обрезан)"
    return [
        *base,
        {"role": "assistant", "content": raw_text or "(пустой ответ)"},
        {
            "role": "user",
            "content": (
                f"Твой предыдущий ответ не прошёл валидацию: {error}\n\n"
                "Верни тот же результат, но строго валидным JSON по схеме:\n"
                f"{schema}\n\n"
                "Только JSON-объект, без пояснений и markdown."
            ),
        },
    ]


def build_user_content(text: str, image_data_urls: Iterable[str]) -> list[dict]:
    """User-сообщение: текстовая инструкция + изображения в base64."""
    content: list[dict] = [{"type": "text", "text": text}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def call_structured(
    system: str,
    user: str,
    response_model: type[BaseModel],
    model: str | None = None,
    reasoning_effort: str | None = None,
    image_data_urls: Iterable[str] = (),
    usage: Usage | None = None,
) -> BaseModel:
    """Вызвать LLM и вернуть объект ``response_model``.

    ``model`` переопределяет ``config.LLM_MODEL`` для всех попыток.
    ``reasoning_effort`` уходит провайдеру через параметр ``reasoning``
    (extra_body); по умолчанию берётся ``config.LLM_REASONING_EFFORT`` (``high``).

    Все попытки идут в strict-режиме: ``response_format`` с ``json_schema`` и
    ``strict: true``. Других форматов нет — если провайдер не умеет structured
    outputs, это фатальная ошибка, а не повод слать нестрогий запрос.

    Поведение повторов (``LLM_MAX_RETRIES`` ретраев + одна первая попытка):
      * сетевой/серверный сбой (ответ не получен) — повторяем **абсолютно тот же**
        запрос, что и в прошлый раз: те же messages, тот же response_format;
      * ответ получен, но JSON не распарсился или не прошёл валидацию модели —
        повторяем в strict-режиме, подставив модели её же ответ и текст ошибки
        (repair-раунд).

    Бросает :class:`LLMError`, если все попытки исчерпаны.
    """
    effective_model = model or config.LLM_MODEL
    effort = reasoning_effort if reasoning_effort is not None else config.LLM_REASONING_EFFORT
    stats = usage or USAGE
    client = get_client()

    user_content: Any = (
        build_user_content(user, image_data_urls) if image_data_urls else user
    )
    base_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    messages = base_messages

    # Тело с режимом рассуждения у разных моделей разное, и расхождение не
    # бросается в глаза: модель с чужим параметром отвечает как ни в чём не
    # бывало, просто без рассуждения. Поэтому форма выбирается по модели
    # (см. ``config.uses_reasoning_switch``), а не одна на всех.
    if not effort:
        extra_body: dict | None = None
    elif config.uses_reasoning_switch(effective_model):
        extra_body = {"reasoning": {"enabled": True}}
    else:
        extra_body = {"reasoning": {"effort": effort}}
    # Strict JSON-контракт один и тот же для всех попыток.
    request_format = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": model_to_strict_schema(response_model),
            "strict": True,
        },
    }

    last_error: Exception | None = None
    last_validation_error: Exception | None = None
    attempts = max(1, config.LLM_MAX_RETRIES + 1)

    for attempt in range(attempts):
        # 1) Запрос. Сеть упала — тело запроса не трогаем, повтор будет 1-в-1.
        try:
            response = client.chat.completions.create(
                model=effective_model,
                messages=messages,
                response_format=request_format,
                extra_body=extra_body,
                max_tokens=35000,
            )
        except Exception as exc:
            if _is_fatal(exc) or _structured_unsupported(exc):
                stats.record_failure()
                raise LLMError(_api_error_message(exc)) from exc
            last_error = exc
            stats.note_retry(f"запрос: {_api_error_message(exc)}")
            _sleep_backoff(attempt)
            continue

        raw_text = response.choices[0].message.content or ""
        (
            in_cost,
            out_cost,
            reason_cost,
            write_cost,
            read_cost,
        ) = _estimate_cost(getattr(response, "usage", None), effective_model)
        stats.record(
            getattr(response, "usage", None),
            in_cost,
            out_cost,
            reason_cost,
            write_cost,
            read_cost,
            retry=attempt > 0,
            pricing_known=config.pricing_for(effective_model).known,
        )

        # 2) Ответ есть — валидируем. Невалидный JSON чинится повтором с ошибкой.
        try:
            return _validate(raw_text, response_model)
        except (ValidationError, json.JSONDecodeError) as exc:
            last_error = last_validation_error = exc
            stats.note_retry(
                f"ответ не прошёл проверку: {_short_error(exc, raw_text)}"
            )
            if attempt + 1 < attempts:
                messages = _repair_messages(base_messages, raw_text, exc, response_model)
                _sleep_backoff(attempt)

    stats.record_failure()
    if last_validation_error is not None:
        raise LLMError("LLM не вернула валидный ответ") from last_error
    raise LLMError(_api_error_message(last_error))


def _short_error(
    exc: BaseException,
    raw_text: str,
    finish_reason: str | None = None,
) -> str:
    """Короткая причина, по которой ответ не прошёл, — для журнала повторов.

    Четыре случая требуют разного лечения, и по одному счётчику повторов их не
    различить:

    * ``content`` пустой — модель потратила всё на рассуждение и не вывела
      JSON (лечится лимитом reasoning, а не промтом);
    * ответ **оборван** — JSON начат, но не закончен (``finish_reason=length``):
      это не «модель молчала», а не хватило бюджета на сам ответ, и лечится
      иначе, чем пустой content;
    * ответ не JSON — провайдер или модель протащили текст (лечится промтом);
    * JSON есть, но не по схеме — лечится схемой ответа.

    Оборванный ответ замечен не по тексту, а по ``finish_reason``: по обрывку
    JSON нельзя отличить «кончились токены» от «модель замолчала на середине».
    """
    if finish_reason == "length" and raw_text.strip():
        return (
            f"ответ оборван на лимите ({finish_reason}), "
            f"принято {len(raw_text)} симв. недописанного JSON"
        )
    if not raw_text.strip():
        reason = f"finish_reason={finish_reason}" if finish_reason else ""
        return f"модель вернула пустой content ({reason})"
    if isinstance(exc, json.JSONDecodeError):
        head = raw_text.strip()[:120].replace("\n", " ")
        return f"ответ не JSON (поз. {exc.pos}): {head!r}"
    if isinstance(exc, ValidationError):
        parts = [
            f"{'.'.join(str(v) for v in err['loc']) or 'корень'}: {err['msg']}"
            for err in exc.errors()[:3]
        ]
        return "не по схеме: " + "; ".join(parts)
    return f"{type(exc).__name__}: {exc}"


def _api_error_message(exc: BaseException | None) -> str:
    """Человекочитаемый текст для LLMError по последней ошибке запроса."""
    if exc is None:
        return "LLM API не ответила"
    if _structured_unsupported(exc):
        return "Провайдер не поддерживает strict JSON (json_schema response_format)"
    return f"LLM API недоступна или вернула ошибку запроса: {exc}"


def _structured_unsupported(exc: BaseException) -> bool:
    """Провайдер отверг ``response_format``: strict JSON недоступен."""
    message = str(exc).lower()
    return any(
        token in message for token in ("response_format", "json_schema", "structured")
    )


def _is_retryable(exc: Exception) -> bool:
    """Ошибки, которые обязательно надо повторить: сервер занят, лимит, таймаут.

    Проверяется **до** ``_is_fatal``, и это не порядок ради порядка. Текст
    ответа 429 у этого провайдера выглядит так:
    ``Concurrent request limit reached for this API key`` — слово «api key» в
    нём есть, и наивная проверка на ключ объявляла перегрузку фатальной: лист
    падал без единого повтора, хотя ``LLM_MAX_RETRIES`` настроен. Перегрузка
    проходит сама, стоит подождать, поэтому повтор здесь обязателен.
    """
    message = str(exc).lower()
    retryable_tokens = (
        "429",
        "rate limit",
        "rate_limit",
        "too many",
        "too_many",
        "concurrency limit",
        "concurrent request",
        "timeout",
        "timed out",
        "temporarily",
        "overloaded",
        "service unavailable",
        "502",
        "503",
        "504",
        "connection reset",
        "connection error",
    )
    return any(token in message for token in retryable_tokens)


def _is_fatal(exc: Exception) -> bool:
    """Ошибки, повторять которые бессмысленно: нет ключа, нет денег, нет модели.

    Сначала отсекается перегрузка: текст ответа 429 у этого провайдера
    содержит упоминание ключа (``Concurrent request limit reached for this
    API key``), и без отсечки она выглядела бы как проблема авторизации —
    лист падал бы без единого повтора, хотя ``LLM_MAX_RETRIES`` настроен.
    Перегрузка проходит сама, стоит подождать, поэтому повтор здесь обязателен.
    """
    if _is_retryable(exc):
        return False
    message = str(exc).lower()
    fatal_tokens = (
        "api key",
        "unauthorized",
        "authentication",
        "insufficient",
        "insufficient_quota",
        "credit",
        "not_found_error",
        "model not found",
        "do not have access to model",
        "no access to model",
        "does not exist",
        "context_length",
    )
    return any(token in message for token in fatal_tokens)
