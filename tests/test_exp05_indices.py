"""Проверка exp_05 и разрешения меток — без обращения к API.

Запуск из корня проекта:

    python tests/test_exp05_indices.py

Проверяется ровно то, из-за чего схема и писалась: модель отвечает метками, а
числа и сумму считает программа. Поэтому главное здесь — не «посчиталось ли
правильно», а подставились ли в сумму именно нарисованные значения, и не
посчитался ли один размер дважды.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isokrus import dimscan
from isokrus.experiments import load_experiment
from isokrus.extract import extract_pages
from isokrus.indices import looks_indexed, resolve_dimension_indices
from isokrus.report import build_rows

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  OK    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")


PDF = "02_Изометрии_10_листов.pdf"

print("1. блок размеров содержит метки")
page = extract_pages(PDF, pages=["9"]).pages[0]
check("размеров найдено", len(page.dimensions) == 5, str(len(page.dimensions)))
marks = [m.line_id for m in page.dimensions]
check("метки уникальны", len(set(marks)) == len(marks), str(marks))
block = dimscan.format_dimensions_block(page.dimensions)
check("метка в строке блока", f'{marks[0]} "' in block, block.splitlines()[-1][:60])
check("в шапке объяснено про метки", "метка размера" in block)

print("1a. метки идут подряд по порядку показа (важно для атрибуции)")
sheet8 = extract_pages(PDF, pages=["8"]).pages[0]
ids = [m.line_id for m in sheet8.dimensions]
check("метки без пропусков и по порядку",
      ids == [f"P{i}" for i in range(1, len(ids) + 1)], str(ids))
positions = [round(m.bbox[1], 1) for m in sheet8.dimensions]
check("порядок меток совпадает с порядком чтения",
      positions == sorted(positions), str(positions))

print("2. на листе 9 два размера по 123 — метки их различают")
twos = [m for m in page.dimensions if m.value == 123.0]
check("оба 123 найдены", len(twos) == 2, str(len(twos)))
check("метки у них разные", twos[0].line_id != twos[1].line_id)

print("3. ответ по меткам превращается в числа")
response = {"included": [m.line_id for m in page.dimensions],
            "excluded": [], "notes": ""}
check("ответ распознан как меточный", looks_indexed(response))
resolved = resolve_dimension_indices(response, page)
check("резолв не пустой", resolved is not None)
assert resolved is not None
values = sorted(s["value"] for s in resolved["segments"])
check("числа совпали с размерами", values == sorted(m.value for m in page.dimensions),
      str(values))
check("сумма равна сумме всех размеров", abs(sum(values) - 860) < 0.5, str(sum(values)))

print("4. повторная метка не считается дважды")
twice = {"included": [m.line_id for m in page.dimensions] * 2, "excluded": []}
doubled = resolve_dimension_indices(twice, page)
assert doubled is not None
check("длин не вырос", len(doubled["segments"]) == 5, str(len(doubled["segments"])))

print("5. неизвестная метка видна, а не молча потеряна")
unknown = {"included": ["P1", "P99"], "excluded": [{"index": "P404", "reason": "x"}]}
report = resolve_dimension_indices(unknown, page)
assert report is not None
check("неизвестные метки перечислены",
      report["unresolved_marks"] == ["P404", "P99"], str(report["unresolved_marks"]))
check("в сумму не попала неизвестная метка",
      all(s["mark"] != "P99" for s in report["segments"]),
      str(report["segments"]))
# Прочие метки листа модель не упоминала — они считаются включёнными
# (см. tests/test_unclassified_marks.py), поэтому в участках их не одна.
check("в сумму попали известные метки листа",
      {s["mark"] for s in report["segments"]} == {"P1", "P2", "P3", "P4", "P5"},
      str([s["mark"] for s in report["segments"]]))

print("6. ответ не по меткам не трогаем")
flat = {"segments": [{"value": 100}, {"value": 200}], "notes": "обычный эксперимент"}
check("не меточный", not looks_indexed(flat))
check("резолв вернул None", resolve_dimension_indices(flat, page) is None)

print("7. лист без разбора не ломает разметку")
scanned = extract_pages(PDF, pages=["9"], with_dimensions=False).pages[0]
check("разбора нет", not scanned.dimensions)
check("резолв вернул None",
      resolve_dimension_indices({"included": ["P1"]}, scanned) is None)

print("8. эксперимент собирается и в отчёт попадают числа")
experiment = load_experiment("exp_05")
system = experiment.render_system(page)
check("токен подставлен", "{{DIMENSIONS}}" not in system)
check("метки в промте", f'{marks[0]} "' in system)
check("текстовый слой выключен", experiment.use_pdf_text is False)
check("размеры включены", experiment.use_dimensions is True)


class _Result:
    def __init__(self, page, response):
        self.page = page
        self.response = response
        self.ok = True
        self.error = None


rows = build_rows([_Result(page, response)])
row = rows[0].values
check("сумма в отчёте равна 860", abs(float(row["Длина, мм"]) - 860) < 0.5,
      str(row["Длина, мм"]))
check("метров посчитано", abs(float(row["Длина, м"]) - 0.86) < 0.001, str(row["Длина, м"]))
check("участков 5", row["Участков"] == 5, str(row["Участков"]))
check("в формуле все пять слагаемых", row["Формула"].count("+") == 4,
      str(row["Формула"]))
check("неизвестных меток в замечаниях нет", "которых нет на листе" not in str(row["Замечания"]),
      str(row["Замечания"]))

print("9. неизвестная метка попадает в замечания отчёта")
bad_rows = build_rows([_Result(page, {"included": ["P1", "P99"], "excluded": [],
                                     "notes": ""})])
remarks = str(bad_rows[0].values["Замечания"])
check("метка отмечена в замечаниях", "P99" in remarks, remarks)

print()
if FAILURES:
    print(f"ПРОВАЛЕНО: {', '.join(FAILURES)}")
    raise SystemExit(1)
print("Все проверки пройдены.")
