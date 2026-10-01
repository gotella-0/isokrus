"""Тарифы по моделям и учёт кеша (запуск из корня проекта, без pytest и без API):

    python tests/test_pricing.py

Проверяется то, что тихо портит счёт денег:

* тариф берётся для модели, которая реально работала, а не общий;
* если тариф неизвестен — стоимость помечается, а не подставляется чужая;
* кеш не оплачивается дважды: ``prompt_tokens`` у провайдера включают и
  запись в кеш, и чтение из него (проверено запросом к провайдеру:
  2358 = 2355 кеш + 3 служебных), поэтому обычный вход считается как
  ``prompt минус кеш``;
* reasoning не оплачивается дважды внутри completion.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isokrus import config
from isokrus.llm import _cache_tokens, _estimate_cost

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  OK    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")


def usage(prompt=0, completion=0, reasoning=0, write=0, read=0):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        prompt_tokens_details=SimpleNamespace(
            cache_write_tokens=write, cached_tokens=read
        ),
    )


print("1. тариф выбирается по имени модели")
saved = config.PRICING_TABLE
config.PRICING_TABLE = {
    "gpt-6-luna": {"input": 10.0, "output": 50.0, "reasoning": 50.0,
                   "cache_write": 12.5, "cache_read": 1.25},
    "glm-5.3-flash": {"input": 7.5, "output": 25.0, "reasoning": 25.0},
}
luna = config.pricing_for("gpt-6-luna")
flash = config.pricing_for("glm-5.3-flash")
check("у луны её тариф", (luna.input, luna.output, luna.reasoning)
      == (10.0, 50.0, 50.0), str(luna))
check("запись в кеш у луны 12.5", luna.cache_write == 12.5, str(luna))
check("у flash его тариф", (flash.input, flash.output) == (7.5, 25.0), str(flash))
check("источник — таблица", luna.source == "таблица" and flash.source == "таблица")

print("2. неизвестная модель честно помечается, а не получает чужой тариф")
config.PRICING_TABLE = {"gpt-6-luna": {"input": 10.0, "output": 50.0}}
unknown = config.pricing_for("some-other-model")
check("тариф неизвестен", unknown.source == "неизвестно", str(unknown))
check("нули вместо чужих цен", unknown.input == 0.0 and unknown.output == 0.0)
check("known=False", unknown.known is False)

print("3. default в таблице покрывает остальные модели")
config.PRICING_TABLE = {
    "gpt-6-luna": {"input": 10.0, "output": 50.0},
    "default": {"input": 1.0, "output": 2.0},
}
check("default применён", config.pricing_for("что-то ещё").input == 1.0)

print("4. reasoning без отдельного тарифа берёт цену output")
config.PRICING_TABLE = {"m": {"input": 1.0, "output": 2.0, "reasoning": 0.0}}
check("reasoning = output", config.pricing_for("m").reasoning == 2.0)

print("5. кеш не оплачивается дважды (факт пробы: 2358 = 2355 кеш + 3)")
config.PRICING_TABLE = {
    "gpt-6-luna": {"input": 10.0, "output": 50.0, "reasoning": 50.0,
                   "cache_write": 12.5, "cache_read": 1.25}
}
first = usage(prompt=2358, completion=27, reasoning=20, write=2355)
second = usage(prompt=2358, completion=27, reasoning=20, read=2355)
inp, out, rea, wr, rd = _estimate_cost(first, "gpt-6-luna")
check("при записи кеша обычный вход = 3 токена",
      abs(inp - 3 / 1_000_000 * 10.0) < 1e-12, str(inp))
check("запись в кеш посчитана по 12.5",
      abs(wr - 2355 / 1_000_000 * 12.5) < 1e-12, str(wr))
check("чтения в первом вызове нет", rd == 0.0, str(rd))

inp2, _, _, wr2, rd2 = _estimate_cost(second, "gpt-6-luna")
check("при чтении кеша обычный вход тоже 3 токена",
      abs(inp2 - 3 / 1_000_000 * 10.0) < 1e-12, str(inp2))
check("чтение посчитано по 1.25",
      abs(rd2 - 2355 / 1_000_000 * 1.25) < 1e-12, str(rd2))
check("записи во втором вызове нет", wr2 == 0.0, str(wr2))
# Кеш считается один раз и по своей цене: 2355 токенов один раз записались
# по 12.5, потом один раз прочитались по 1.25. Если бы токены кеша тарифицировались
# ещё и как обычный вход, сумма была бы заметно больше.
cache_total = wr + rd2
expected = 2355 / 1_000_000 * 12.5 + 2355 / 1_000_000 * 1.25
check("кеш посчитан один раз по своим ценам",
      abs(cache_total - expected) < 1e-12, f"{cache_total} != {expected}")
check("кеш не переплачен как обычный вход",
      cache_total < 2358 / 1_000_000 * 10.0 + 2355 / 1_000_000 * 12.5,
      f"{cache_total}")

print("6. reasoning не оплачивается дважды внутри completion")
plain = usage(prompt=1_000_000, completion=1000, reasoning=400)
_, out_v, rea_v, _, _ = _estimate_cost(plain, "gpt-6-luna")
check("output = completion - reasoning = 600",
      abs(out_v - 600 / 1_000_000 * 50.0) < 1e-12, str(out_v))
check("reasoning посчитан отдельно",
      abs(rea_v - 400 / 1_000_000 * 50.0) < 1e-12, str(rea_v))

print("7. провайдер не должен уводить счёт в минус")
weird = usage(prompt=100, completion=100, reasoning=500, write=100, read=100)
inp_w, out_w, rea_w, wr_w, rd_w = _estimate_cost(weird, "gpt-6-luna")
check("все компоненты неотрицательны",
      min(inp_w, out_w, rea_w, wr_w, rd_w) >= 0.0,
      f"{inp_w} {out_w} {rea_w} {wr_w} {rd_w}")
check("кеш не больше промта", rd_w <= 100 / 1_000_000 * 1.25 + 1e-12)

print("8. неизвестный тариф даёт нули, а не выдуманную сумму")
zeros = _estimate_cost(usage(prompt=1_000_000, completion=1000), "неизвестная")
check("все нули", zeros == (0.0, 0.0, 0.0, 0.0, 0.0), str(zeros))

print("9. чтение полей usage терпимо к разным провайдерам")
check("cache_write_tokens из model_extra",
      _cache_tokens(SimpleNamespace(
          prompt_tokens_details=None,
          model_extra={"cache_write_tokens": 7},
      )) == (7, 0))
check("cache_read_tokens из model_extra",
      _cache_tokens(SimpleNamespace(
          prompt_tokens_details=None,
          model_extra={"cache_read_tokens": 9},
      )) == (0, 9))
check("нет деталей вообще — нули",
      _cache_tokens(SimpleNamespace(prompt_tokens_details=None)) == (0, 0))

config.PRICING_TABLE = saved

print()
if FAILURES:
    print(f"ПРОВАЛЕНО: {', '.join(FAILURES)}")
    raise SystemExit(1)
print("Все проверки пройдены.")
