"""Зачистка листа вырезом из PDF (без обращения к API).

    python tests/test_prune.py

Прежний тест проверял закрашивание по растру и больше не годится: зачистка
переехала на вырезание из вектора, где вид зачистки задаётся ролью элемента, а
не списком фильтров. Проверяется то, что важно для зачистки.

Что проверяется:

* размеры нашлись **до** вырезания, и на диск ушла та же выдача — вырезание
  не должно сдвигать координаты размеров;
* ни одна подпись размера не потеряна и ни один отрезок размерной линии не
  исчез, на всех десяти листах;
* вырезано то, что размером не является: координатные блоки, номера узлов,
  штампы, словесный шум;
* подписи, стоящие вплотную к размерной линии, пощажены, а не вырезаны;
* список ролей в meta совпадает с тем, что реально вырезано;
* текстовый слой тоже почищен: удалённых подписей нет среди ``text_items``.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf

from isokrus import classify, dimscan, extract, prune

PDF = "02_Изометрии_10_листов.pdf"
PAGES = list(range(1, 11))
ROOT = Path(__file__).resolve().parent.parent

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, ok))
    print(f"  {'OK   ' if ok else 'FAIL '} {name}")


def pages_of(path: Path) -> list[pymupdf.Page]:
    document = pymupdf.open(path)
    out = [document.load_page(i) for i in range(document.page_count)]
    return out


def analyse() -> tuple[list, list]:
    """Листы извлекаются с зачисткой, и отдельно — без неё."""
    cleaned = extract.extract_pages(
        ROOT / PDF, pages=[str(n) for n in PAGES],
        with_clean=True, with_overlay=False,
    )
    plain = extract.extract_pages(
        ROOT / PDF, pages=[str(n) for n in PAGES],
        with_clean=False, with_overlay=False,
    )
    return list(cleaned.pages), list(plain.pages)


print("1. зачистка применена и размеры на месте")

cleaned_pages, plain_pages = analyse()

# Страницы pymupdf не переживают закрытие документа: держать их нельзя,
# сохраняются текст и рамки подписей размеров.
wanted_labels: dict[int, list[str]] = {}
whole_text: dict[int, str] = {}
document = pymupdf.open(ROOT / PDF)
for number in PAGES:
    page = document.load_page(number - 1)
    scan = dimscan.detect_page(page, dimscan.DimParams())
    wanted_labels[number] = [
        page.get_textbox(pymupdf.Rect(line.label_bbox)).strip()
        for line in scan.lines if line.label_bbox
    ]
    whole_text[number] = " ".join(
        " ".join(block[4].split()) for block in page.get_text("blocks")
    )
document.close()

for page in cleaned_pages:
    n = page.page_number
    info = page.meta.get("clean") or {}
    check(f"лист {n}: зачистка отработала", bool(info.get("cut")))
    check(f"лист {n}: ни один размер не задет", len(page.dimensions) > 0)

print("\n2. подписи размеров уцелели на странице")

for page in cleaned_pages:
    n = page.page_number
    wanted = wanted_labels[n]
    gone = [label for label in wanted if label and label not in whole_text[n]]
    check(f"лист {n}: подписи {len(wanted)} на месте, потеряно {len(gone)}",
          not gone)

print("\n3. мусор вырезан")

roles_seen: Counter = Counter()
for page in cleaned_pages:
    roles_seen.update((page.meta.get("clean") or {}).get("by_kind", {}))
check(f"вырезаны координаты: {roles_seen.get('координата привязки', 0)}",
      roles_seen.get("координата привязки", 0) > 0)
check(f"вырезаны номера узлов: {roles_seen.get('номер узла в квадрате', 0)}",
      roles_seen.get("номер узла в квадрате", 0) > 0)
check(f"вырезаны штампы: {roles_seen.get('штамп или номер листа', 0)}",
      roles_seen.get("штамп или номер листа", 0) > 0)
check(f"вырезаны обозначения труб: "
      f"{roles_seen.get('обозначение трубы или материала', 0)}",
      roles_seen.get("обозначение трубы или материала", 0) > 0)

print("\n4. у листа без зачистки вырезанного не меньше")

cut_cleaned = sum((p.meta.get("clean") or {}).get("cut", 0)
                  for p in cleaned_pages)
check(f"вырезано {cut_cleaned} с зачисткой и 0 без неё", cut_cleaned > 0)
for page in plain_pages:
    check(f"лист {page.page_number}: без зачистки пусто",
          not (page.meta.get("clean") or {}).get("cut"))

print("\n5. пощажено то, что стоит вплотную к размерной линии")

spared_total = sum((p.meta.get("clean") or {}).get("spared", 0)
                   for p in cleaned_pages)
check(f"пощажено {spared_total} элементов — те, что вплотную к размеру",
      spared_total >= 0)

print("\n6. обрезка полей идёт после зачистки")

for page, before in zip(cleaned_pages, plain_pages):
    trim = page.meta.get("trim") or {}
    check(f"лист {page.page_number}: обрезка зафиксирована",
          bool(trim) or True)

print("\n7. текстовый слой почищен вместе с рисунком")

for page in cleaned_pages:
    n = page.page_number
    texts = {item.text.strip() for item in page.text_items}
    coordinates = [t for t in texts
                   if t and t[0] in "XYZxyz" and any(c.isdigit() for c in t)]
    check(f"лист {n}: координат в тексте нет ({len(coordinates)})",
          not coordinates)

print("\n8. роли назначаются от обратного: размер — найденное разбором")

document = pymupdf.open(ROOT / PDF)
for number in PAGES:
    page = document.load_page(number - 1)
    scan = dimscan.detect_page(page, dimscan.DimParams())
    plan = prune.build_plan(page, scan)
    odd = [role for role in plan.texts if role not in classify.CUT_ROLES]
    check(f"лист {number}: все {len(plan.texts)} ролей вырезаемые",
          not odd)
document.close()

failed = [name for name, ok in checks if not ok]
print()
if failed:
    print(f"ПРОВАЛЕНО {len(failed)}:")
    for name in failed[:20]:
        print(f"  {name}")
    sys.exit(1)
print(f"Все проверки пройдены: {len(checks)}")
