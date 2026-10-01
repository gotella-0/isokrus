"""Проверка интеграции разбора размеров в конвейер.

Запуск из корня проекта, без pytest и без обращения к API::

    python tests/test_isokrus_dimscan.py

Проверяется то, что может сломаться молча и при этом стоит денег на каждом
листе: разбор размеров должен попасть **в ту же систему координат**, что и
текстовый слой, дойти до системного промта и не ломать конвейер, когда на
листе геометрии нет вовсе.

1. порядок «PDF -> разбор -> рендер -> обрезка» и одна матрица на обе части;
2. рамки размеров внутри картинки и совпадают с рамкой того же числа в
   текстовом слое — иначе модель получает две разные карты листа;
3. блок размеров доходит до ``system.md`` по токену ``{{DIMENSIONS}}``, а
   список отсеянных чисел — тоже;
4. ``--no-dimscan`` действительно отключает разбор, а не просто прячет блок;
5. лист без размеров (скан) не роняет прогон и честно сообщает, что размеров
   не найдено.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isokrus.experiments import load_experiment
from isokrus.extract import extract_pages

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  OK    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")


pdfs = sorted(glob.glob("02_*.pdf"))
if not pdfs:
    print("  пропуск: 02_*.pdf не найден")
    raise SystemExit(1)
PDF = pdfs[0]

print("1. разбор идёт по вектору, координаты — от обрезанной картинки")
result = extract_pages(PDF, pages=["2", "3"])
pages = {p.page_number: p for p in result.pages}
for number, page in sorted(pages.items()):
    info = page.meta.get("dimscan") or {}
    check(
        f"лист {number}: разбор выполнен до рендера, {len(page.dimensions)} размеров",
        bool(page.dimensions) and info.get("dimensions") == len(page.dimensions),
        f"{info}",
    )
    check(
        f"лист {number}: масштаб определён по кеглю ({info.get('scale_source')})",
        info.get("scale") is not None and 0.9 < float(info["scale"]) < 1.1,
        f"{info.get('scale')} ({info.get('scale_source')})",
    )
    inside = [
        d.label for d in page.dimensions
        if not (0 <= d.bbox[0] and d.bbox[2] <= page.width
                and 0 <= d.bbox[1] and d.bbox[3] <= page.height)
    ]
    check(
        f"лист {number}: все рамки внутри картинки {page.width}x{page.height}",
        not inside,
        f"вылезли: {inside}",
    )

print("2. рамка размера совпадает с рамкой того же числа в текстовом слое")
for number, page in sorted(pages.items()):
    by_text: dict[str, list] = {}
    for item in page.text_items:
        by_text.setdefault(item.text, []).append(item)
    mismatched: list[str] = []
    matched = 0
    for mark in page.dimensions:
        candidates = by_text.get(mark.label) or []
        if not candidates:
            continue
        # Текстовый слой режет строку на слова, а размер может состоять из
        # нескольких («102 00» в одном спа́не режется как одно слово), поэтому
        # ищем не точное равенство рамок, а пересечение.
        hit = any(
            not (item.x1 <= mark.bbox[0] or item.x0 >= mark.bbox[2]
                 or item.y1 <= mark.bbox[1] or item.y0 >= mark.bbox[3])
            for item in candidates
        )
        if hit:
            matched += 1
        else:
            mismatched.append(mark.label)
    check(
        f"лист {number}: {matched} из {len(page.dimensions)} размеров "
        f"совпали с текстовым слоем",
        not mismatched,
        f"не совпали: {mismatched}",
    )

print("3. блок доходит до системного промта")
experiment = load_experiment("exp_04")
page = pages[3]
system = experiment.render_system(page)
check("токен {{PDF_TEXT}} подставлен", "{{PDF_TEXT}}" not in system)
check("токен {{DIMENSIONS}} подставлен", "{{DIMENSIONS}}" not in system)
check("блок размеров в промте", "# Размеры, найденные на листе" in system)
check("список отсеянных в промте", "НЕ являются" in system)
for mark in page.dimensions[:3]:
    check(
        f"  значение {mark.label} есть в промте",
        f'"{mark.label}"' in system,
    )
without = experiment.render_system(None)
check("без страницы промт не меняется", without == experiment.system)

print("4. отключение разбора --no-dimscan")
off = extract_pages(PDF, pages=["3"], with_dimensions=False)
check(
    "размеров нет, отсеянных нет",
    not off.pages[0].dimensions and not off.pages[0].rejected_numbers,
)
check(
    "в meta видно, что разбора не было",
    off.pages[0].meta.get("dimscan") is None,
)
check(
    "текстовый слой при этом на месте",
    bool(off.pages[0].text_items),
)
# Главное: выключенный разбор не должен оставлять в промте пустой блок с
# текстом «размеров не найдено, это дефект разбора» — модель начала бы
# оправдываться за чужую ошибку вместо того, чтобы просто считать чертёж.
system_off = experiment.render_system(off.pages[0])
check("раздел про размеры убран из промта", "Разобранные размеры" not in system_off)
check("блока размеров в промте нет", "# Размеры, найденные" not in system_off)
check(
    "нет упоминания дефекта разбора",
    "дефект разбора" not in system_off,
)
check(
    "токены подставлены, промт короче",
    "{{PDF_TEXT}}" not in system_off
    and len(system_off) < len(system),
)

print("5. лист без векторной геометрии не роняет прогон")
scans = sorted(glob.glob("03_*.pdf"))
if scans:
    scanned = extract_pages(scans[0])
    sp = scanned.pages[0]
    check("скан извлечён", bool(sp.data))
    check("разбор не уронил лист", sp.meta.get("dimscan_error") is None)
    check("размеров на скане нет", not sp.dimensions)
    block = __import__("isokrus.dimscan", fromlist=["x"]).format_dimensions_block(
        sp.dimensions, sp.rejected_numbers
    )
    check("блок честно говорит, что размеров нет", "не найдено" in block)
else:
    print("  пропуск: 03_*.pdf не найден")

print()
if FAILURES:
    print(f"ПРОВАЛЕНО: {', '.join(FAILURES)}")
    raise SystemExit(1)
print("Все проверки пройдены.")
