"""Правило «неупомянутый размер считается включённым».

На прогоне с зачисткой модель на десятом листе вернула 14 меток из 15: P7
(2450) не попала ни в список участков, ни в список исключений. Резолвер молча
её выбросил, и весь лист разошёлся на 2450 мм — при том, что остальные
четырнадцать меток были выбраны верно.

    python tests/test_unclassified_marks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isokrus import dimscan
from isokrus.extract import PageImage
from isokrus.indices import resolve_dimension_indices

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  OK    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")


def sheet(values: dict[str, float]) -> PageImage:
    return PageImage(
        page_number=1, data=b"", width=100, height=100,
        dimensions=tuple(
            dimscan.DimensionMark(
                line_id=mark, value=float(value), label=str(int(value)),
                kind="span", bbox=(0.0, 0.0, 0.0, 0.0),
                line_bbox=(0.0, 0.0, 0.0, 0.0),
            )
            for mark, value in values.items()
        ),
    )


print("1. неупомянутый размер попадает в сумму")
page = sheet({"P1": 154.0, "P2": 220.0, "P3": 2450.0})
resolved = resolve_dimension_indices(
    {"included": ["P1", "P2"], "excluded": [], "notes": ""}, page
)
assert resolved is not None
check("P3 добавлен в участки",
      [s["mark"] for s in resolved["segments"]] == ["P1", "P2", "P3"],
      str([s["mark"] for s in resolved["segments"]]))
check("сумма считает забытый размер",
      sum(s["value"] for s in resolved["segments"]) == 2824.0,
      str(sum(s["value"] for s in resolved["segments"])))
check("забытая метка помечена",
      resolved.get("unclassified_marks") == ["P3"],
      str(resolved.get("unclassified_marks")))
check("забытый участок помечен в самом участке",
      [s.get("unclassified") for s in resolved["segments"]] == [None, None, True],
      str([s.get("unclassified") for s in resolved["segments"]]))

print("\n2. явно исключённый размер в сумму не попадает")
resolved = resolve_dimension_indices(
    {"included": ["P1", "P2"],
     "excluded": [{"index": "P3", "reason": "вложен"}], "notes": ""},
    page,
)
assert resolved is not None
check("исключение уважено",
      [s["mark"] for s in resolved["segments"]] == ["P1", "P2"],
      str([s["mark"] for s in resolved["segments"]]))
check("в неупомянутые не попал",
      resolved.get("unclassified_marks") == [],
      str(resolved.get("unclassified_marks")))

print("\n3. исключение записью с причиной и без")
resolved = resolve_dimension_indices(
    {"included": ["P1"],
     "excluded": [{"mark": "P2", "reason": "вложен в P3"}], "notes": ""},
    page,
)
assert resolved is not None
check("исключение по полю mark уважено",
      resolved.get("unclassified_marks") == ["P3"],
      str(resolved.get("unclassified_marks")))

print("\n4. повтор метки и неизвестная метка не ломают разбор")
resolved = resolve_dimension_indices(
    {"included": ["P1", "P1", "P99"], "excluded": [], "notes": ""},
    sheet({"P1": 100.0, "P2": 200.0}),
)
assert resolved is not None
check("повтор посчитан один раз",
      sum(s["value"] for s in resolved["segments"]) == 300.0,
      str(resolved["segments"]))
check("неизвестная метка попала в unresolved",
      resolved.get("unresolved_marks") == ["P99"],
      str(resolved.get("unresolved_marks")))
check("неизвестная метка не стала участком",
      [s["mark"] for s in resolved["segments"]] == ["P1", "P2"],
      str([s["mark"] for s in resolved["segments"]]))

print("\n5. ответ без меток остаётся нетронутым")
check("непостроенный ответ -> None",
      resolve_dimension_indices({"length": 100.0}, page) is None)
check("нет ответа -> None",
      resolve_dimension_indices(None, page) is None)
check("нет разбора -> None",
      resolve_dimension_indices({"included": ["P1"]},
                                PageImage(page_number=1, data=b"",
                                          width=1, height=1)) is None)

print()
if FAILURES:
    print(f"ПРОВАЛЕНО {len(FAILURES)}: {', '.join(FAILURES)}")
    raise SystemExit(1)
print("Все проверки пройдены.")