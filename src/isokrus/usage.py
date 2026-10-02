"""Учёт токенов и стоимости по всем вызовам LLM.

Отдельно от слоя вызова (:mod:`.llm`) счётчик живёт по двум причинам. Во-первых,
это самостоятельная единица: им пользуются конвейер и отчёт, и править тарифы
удобнее, не трогая код запросов. Во-вторых, в этом файле собран весь разбор
``usage`` провайдера — места, где без проверок легко посчитать токены дважды
(кеш уже сидит в ``prompt_tokens``, рассуждение — в ``completion_tokens``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from . import config


@dataclass
class Usage:
    """Счётчик токенов и денег по всем вызовам процесса."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    retries: int = 0
    failures: int = 0
    # Причины повторов: без них по прогону видно только «повторов 1», и
    # нельзя отличить сетевую ошибку от невалидного ответа модели — а чинить
    # надо разное.
    retry_reasons: list[str] = field(default_factory=list, repr=False)
    input_cost: float = 0.0
    output_cost: float = 0.0
    reasoning_cost: float = 0.0
    cache_write_cost: float = 0.0
    cache_read_cost: float = 0.0
    # Токены кеша храним отдельно от промта: именно их тарифицируют по своей
    # цене, и без них невозможно сверить счёт с расчётом — сумма кеша должна
    # совпадать с тем, что показывает биллинг провайдера.
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    pricing_known: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        usage: Any,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        reasoning_cost: float = 0.0,
        cache_write_cost: float = 0.0,
        cache_read_cost: float = 0.0,
        retry: bool = False,
        pricing_known: bool = True,
    ) -> None:
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
        cache_write, cache_read = _cache_tokens(usage)
        with self._lock:
            self.calls += 1
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.reasoning_tokens += reasoning
            self.cache_write_tokens += cache_write
            self.cache_read_tokens += cache_read
            self.input_cost += input_cost
            self.output_cost += output_cost
            self.reasoning_cost += reasoning_cost
            self.cache_write_cost += cache_write_cost
            self.cache_read_cost += cache_read_cost
            # Тариф мог быть неизвестен хотя бы для части вызовов — тогда сумма
            # в отчёте занижена, и молчать об этом нельзя.
            self.pricing_known = self.pricing_known and pricing_known
            if retry:
                self.retries += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1

    def note_retry(self, reason: str) -> None:
        """Запомнить, из-за чего понадобился повтор.

        Повторов бывает ровно два вида, и они чинятся противоположно: упал
        запрос (сеть, шлюз, таймаут — чинится ретраем и настройкой) или упала
        валидация ответа (модель выдала не JSON / не по схеме — чинится
        промтом и схемой). Счётчик ``retries`` разницы не показывает.
        """
        with self._lock:
            self.retry_reasons.append(reason)

    @property
    def cost(self) -> float:
        return (self.input_cost + self.output_cost + self.reasoning_cost
                + self.cache_write_cost + self.cache_read_cost)

    def merge(self, other: "Usage") -> None:
        with self._lock, other._lock:
            self.calls += other.calls
            self.prompt_tokens += other.prompt_tokens
            self.completion_tokens += other.completion_tokens
            self.reasoning_tokens += other.reasoning_tokens
            self.retries += other.retries
            self.failures += other.failures
            self.retry_reasons.extend(other.retry_reasons)
            self.input_cost += other.input_cost
            self.output_cost += other.output_cost
            self.reasoning_cost += other.reasoning_cost
            self.cache_write_cost += other.cache_write_cost
            self.cache_read_cost += other.cache_read_cost
            self.cache_write_tokens += other.cache_write_tokens
            self.cache_read_tokens += other.cache_read_tokens
            self.pricing_known = self.pricing_known and other.pricing_known

    @property
    def visible_completion_tokens(self) -> int:
        """completion_tokens без токенов рассуждения."""
        return max(self.completion_tokens - self.reasoning_tokens, 0)

    def as_dict(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "retries": self.retries,
                "retry_reasons": list(self.retry_reasons),
                "failures": self.failures,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "visible_completion_tokens": self.visible_completion_tokens,
                "cache_write_tokens": self.cache_write_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "input_cost": round(self.input_cost, 6),
                "output_cost": round(self.output_cost, 6),
                "reasoning_cost": round(self.reasoning_cost, 6),
                "cache_write_cost": round(self.cache_write_cost, 6),
                "cache_read_cost": round(self.cache_read_cost, 6),
                "cost": round(self.cost, 6),
                # False — тариф для этой модели неизвестен, сумма занижена.
                "pricing_known": self.pricing_known,
            }


# Счётчик по умолчанию: одиночные вызовы вне конвейера пишут сюда.
USAGE = Usage()


def _cache_tokens(usage: Any) -> tuple[int, int]:
    """``(запись в кеш, чтение из кеша)`` из ответа провайдера.

    Имена полей у провайдеров разные, поэтому берём первое, что нашлось:
    ``prompt_tokens_details`` — основное, ``cache_read_tokens`` на верхнем
    уровне — запасное (так отдаёт часть шлюзов).
    """
    details = getattr(usage, "prompt_tokens_details", None)
    write = getattr(details, "cache_write_tokens", None)
    if write is None:
        extra = getattr(usage, "model_extra", None) or {}
        write = extra.get("cache_write_tokens", 0)
    read = getattr(details, "cached_tokens", None)
    if read is None:
        extra = getattr(usage, "model_extra", None) or {}
        read = extra.get("cache_read_tokens", 0)
    return int(write or 0), int(read or 0)


def _estimate_cost(usage: Any, model: str | None) -> tuple[float, ...]:
    """Стоимость вызова: ``(input, output, reasoning, cache_write, cache_read)``.

    Две особенности формата usage, из-за которых нужен разбор, а не одно
    умножение:

    * ``completion_tokens`` уже включают ``reasoning_tokens``, поэтому обычный
      output считаем как ``completion - reasoning`` — иначе reasoning
      посчитался бы дважды: отдельно и внутри output;
    * ``prompt_tokens`` включают и запись в кеш, и чтение из него. Проверено
      на провайдере: 2358 = 2355 кеш + 3 служебных, при повторе
      ``cached_tokens`` растёт, а ``prompt_tokens`` не меняется. Значит обычный
      вход — это ``prompt минус кеш``, иначе кеш оплачивается дважды: по цене
      входа и по цене кеша.
    """
    pricing = config.pricing_for(model)
    if not pricing.known:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) or 0
    # Провайдер может не отдать детали или сообщить больше, чем в completion —
    # в этом случае весь completion считаем reasoning, а не уходим в минус.
    reasoning = min(reasoning, completion)
    visible = completion - reasoning

    cache_write, cache_read = _cache_tokens(usage)
    # Кеш не может занимать больше промта; провайдеры иногда округляют вверх.
    cache_write = min(cache_write, prompt)
    cache_read = min(cache_read, max(prompt - cache_write, 0))
    plain_input = max(prompt - cache_write - cache_read, 0)

    m = 1_000_000
    return (
        plain_input / m * pricing.input,
        visible / m * pricing.output,
        reasoning / m * pricing.reasoning,
        cache_write / m * pricing.cache_write,
        cache_read / m * pricing.cache_read,
    )
