"""Сверка ручных замеров из ``reference/`` с эталоном из кода.

Единственный источник истины — ``tests/dimscan/reference.py``: там лежит
поразмерный эталон ``PER_SHEET``, из которого суммы выводятся программно.
Файлы ``reference/list_XX.txt`` — человекочитаемая форма того же эталона:
значения построчно, последняя строка — «Итог - NNNN».

Проверяется две вещи: (1) набор значений в txt совпадает с ``PER_SHEET``,
(2) «Итог» в txt равен сумме из ``EXPECTED``. Расхождение — ошибка: значит,
кто-то поправил одну из форм и не поправил вторую.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tests" / "dimscan"))

from reference import EXPECTED, PER_SHEET  # noqa: E402

# Строка итога: ``итого - 5563``. Опознаётся по числу в конце строки, чтобы
# не зависеть от языка первого слова. Значение может прийти и суммой одной
# строкой (``210+13+178``) — это не итог, а значения.
TOTAL_RE = re.compile(r"^(?:\S+\s+)*-\s*(\d+)\s*$")


def parse_label(path: Path) -> tuple[list[int], int | None]:
    """Значения из txt и итог из последней строки (``None``, если нет).

    Формы строк, встречающиеся в метках:

    * по одному значению на строку: ``320``;
    * сумма одной строкой: ``210+13+178+...`` — делится по ``+``;
    * итог: ``итого - 5563`` — опознаётся по числу в конце строки.
    """
    total: int | None = None
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = TOTAL_RE.match(line)
        if match:
            total = int(match.group(1))
        elif re.fullmatch(r"\d+(?:\+\d+)*", line):
            values.extend(int(part) for part in line.split("+"))
        else:
            raise ValueError(f"{path.name}: нераспознанная строка {line!r}")
    return values, total


def main() -> int:
    failures: list[str] = []
    labels = sorted(ROOT.glob("reference/list_*.txt"))
    if not labels:
        print("reference/list_*.txt не найдены")
        return 2

    for path in labels:
        page = int(path.stem.removeprefix("list_"))
        values, total = parse_label(path)
        truth = list(PER_SHEET.get(page, ()))
        if not truth:
            failures.append(f"{path.name}: для листа {page} нет эталона")
            continue
        if sorted(values) != sorted(truth):
            extra = sorted(set(values) - set(truth))
            missing = sorted(set(truth) - set(values))
            failures.append(
                f"{path.name}: набор значений расходится с эталоном "
                f"(лишние {extra}, недостающие {missing})"
            )
        expected = EXPECTED[page]
        if total is None:
            failures.append(f"{path.name}: нет строки «Итог»")
        elif abs(total - expected) > 0.5:
            failures.append(
                f"{path.name}: Итог {total} != сумма эталона {expected:g}"
            )

    if failures:
        print("Расхождения эталона:")
        for line in failures:
            print(f"  FAIL {line}")
        return 1
    print(f"OK: {len(labels)} меток совпадают с эталоном "
          f"(PER_SHEET и EXPECTED из tests/dimscan/reference.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
