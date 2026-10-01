"""Переопределение модели/размышления флагами: доходит ли до места отправки.

Без обращения к API: собираем аргументы как это делает CLI, применяем тот же
помощник и смотрим, что выбрал бы конвейер. Проверяем и то, что подмена
действительно меняет объект, и то, что без флагов ничего не ломается.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isokrus import config
from isokrus.cli import _experiment, build_parser

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  OK    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")


def args_for(extra: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(["run", "x.pdf", *extra])


print("1. без флагов всё как задано в эксперименте")
plain = _experiment(args_for([]))
meta_high = plain.reasoning_effort == "high"
check("reasoning остался из meta.json", meta_high, str(plain.reasoning_effort))
check("модель осталась из конфига", plain.model is None, str(plain.model))

print("2. --model и --reasoning перекрывают")
patched = _experiment(args_for(["--model", "gpt-6-luna", "--reasoning", "medium"]))
check("модель перекрыта", patched.model == "gpt-6-luna", str(patched.model))
check("reasoning перекрыт", patched.reasoning_effort == "medium",
      str(patched.reasoning_effort))

print("3. подмена не портит остальной эксперимент")
check("системный промт тот же", patched.system == plain.system)
# Классы схемы создаются заново при каждой загрузке эксперимента, поэтому
# сравнивать надо содержимое, а не идентичность объекта.
check("схема ответа та же",
      patched.response_model.model_json_schema()
      == plain.response_model.model_json_schema())
check("набор полей ответа тот же",
      set(patched.response_model.model_fields)
      == set(plain.response_model.model_fields),
      str(set(patched.response_model.model_fields)))
check("токены подписей те же", patched.use_pdf_text == plain.use_pdf_text)
check("блок размеров тот же флаг", patched.use_dimensions == plain.use_dimensions)

print("4. частичное перекрытие не затирает соседнее поле")
only_model = _experiment(args_for(["--model", "gpt-6-luna"]))
check("reasoning не сброшен", only_model.reasoning_effort == "high",
      str(only_model.reasoning_effort))
only_reason = _experiment(args_for(["--reasoning", "medium"]))
check("модель не сброшена", only_reason.model is None, str(only_reason.model))

print("5. исходный эксперимент на диске не изменён")
reloaded = _experiment(args_for([]))
check("meta.json не тронут",
      reloaded.model is None and reloaded.reasoning_effort == "high")

print("6. флаги есть только у команд, которые ходят в API")
parser = build_parser()
for command in ("run", "analyze"):
    sub = parser.parse_args([command, "x.pdf", "--model", "m", "--reasoning", "low"])
    check(f"{command}: --model и --reasoning принимаются",
          sub.model == "m" and sub.reasoning == "low")
# У extract/text/dims нет обращения к API: флаги модели там бессмысленны
# и только путают в help. argparse сообщает о неизвестном флаге через
# parser.error(), то есть через SystemExit — значит отказ это и есть успех.
import contextlib  # noqa: E402
import io  # noqa: E402

for command in ("extract", "text", "dims"):
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parser.parse_args([command, "x.pdf", "--model", "m"])
        except SystemExit:
            rejected = True
        else:
            rejected = False
    check(f"{command}: --model не предлагается", rejected,
          "флаг принят командой, которая в API не ходит")

print("7. статистика прогона покажет фактическую модель")
# build_stats читает модель из объекта эксперимента, значит подмена обязана
# доехать туда же — иначе отчёт врёт о том, чем считали.
from isokrus.report import build_stats  # noqa: E402

src = Path(__file__).resolve().parents[1] / "src" / "isokrus" / "report.py"
text = src.read_text(encoding="utf-8")
check("stats берёт модель из эксперимента",
      "run_result.experiment.model or config.LLM_MODEL" in text)
check("stats берёт reasoning из эксперимента",
      "run_result.experiment.reasoning_effort" in text)
check("build_stats доступна", callable(build_stats))
check("конфиг по умолчанию на месте", config.LLM_MODEL == "glm-5.3-flash",
      config.LLM_MODEL)

print()
if FAILURES:
    print(f"ПРОВАЛЕНО: {', '.join(FAILURES)}")
    raise SystemExit(1)
print("Все проверки пройдены.")
