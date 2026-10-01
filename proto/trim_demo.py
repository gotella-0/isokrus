"""Демонстрация обрезки пустых полей: что именно она делает с листом.

Алгоритм живёт в ``src/isokrus/trim.py`` — здесь только наглядный отчёт.
Скрипт ничего не меняет в конвейере и не ходит в API.

    python proto/trim_demo.py                       # 02_*.pdf, листы 1-3
    python proto/trim_demo.py --pages 1-10 --dpi 320
    python proto/trim_demo.py --pdf 03_Пример.pdf

Рядом с отчётом сохраняются три картинки на лист:

``NN_full_box.png``  — весь лист с красной рамкой найденного содержимого;
``NN_trim.png``      — результат обрезки (то, что ушло бы в модель);
``NN_compare.png``   — оба в одной высоте: видно, сколько деталей добавляется.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from isokrus import config
from isokrus.cli import _split_pages
from isokrus.extract import extract_pages

RED = (0.9, 0.1, 0.1)
COMPARE_HEIGHT = 1400  # px — высота картинок в сравнительном файле


def with_box(pixmap: pymupdf.Pixmap, box: tuple[int, int, int, int],
             label: str = "") -> bytes:
    """Весь лист с красной рамкой найденного содержимого."""
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=pixmap.width, height=pixmap.height)
        page.insert_image(page.rect, stream=pixmap.tobytes("png"))
        page.draw_rect(pymupdf.Rect(*box), color=RED, width=6)
        if label:
            page.insert_text((30, 60), label, fontsize=48, color=RED)
        return page.get_pixmap(alpha=False).tobytes("png")
    finally:
        doc.close()


def side_by_side(full: pymupdf.Pixmap, trimmed: pymupdf.Pixmap) -> bytes:
    """Обе картинки в одной высоте — видно выигрыш в деталях."""
    width_full = round(full.width * COMPARE_HEIGHT / full.height)
    width_trim = round(trimmed.width * COMPARE_HEIGHT / trimmed.height)
    gap = 40
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=width_full + width_trim + gap * 3,
                            height=COMPARE_HEIGHT + gap * 2)
        left = pymupdf.Rect(gap, gap, gap + width_full, gap + COMPARE_HEIGHT)
        right = pymupdf.Rect(gap * 2 + width_full, gap,
                             gap * 2 + width_full + width_trim, gap + COMPARE_HEIGHT)
        page.insert_image(left, stream=full.tobytes("png"))
        page.insert_image(right, stream=trimmed.tobytes("png"))
        page.draw_rect(left, color=RED, width=4)
        page.draw_rect(right, color=(0.0, 0.5, 0.2), width=4)
        return page.get_pixmap(alpha=False).tobytes("png")
    finally:
        doc.close()


def main() -> int:
    default_pdf = glob.glob("02_*.pdf")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", default=default_pdf[0] if default_pdf else None)
    parser.add_argument("--pages", default="1-3", help="номера листов: 1-10 или 1,3,7")
    parser.add_argument("--dpi", type=int, default=config.RENDER_DPI)
    parser.add_argument("--out", default="proto/out")
    args = parser.parse_args()
    if not args.pdf or not Path(args.pdf).exists():
        print("PDF не найден: укажи --pdf", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = None if args.pages.lower() == "all" else _split_pages([args.pages])

    print(f"{args.pdf}, dpi={args.dpi}, поля={config.TRIM_MARGINS}, "
          f"pad={config.TRIM_PAD_PX}, шум={config.TRIM_NOISE}")
    header = (f"{'лист':>4} {'было':>11} {'стало':>11} {'площадь':>8} "
              f"{'деталь':>7} {'PNG':>14} {'поля L/T/R/B':>22} {'подписи':>10}")
    print(header)
    print("-" * len(header))

    for trim in (False, True):
        started = time.perf_counter()
        result = extract_pages(args.pdf, dpi=args.dpi, pages=pages, trim_margins=trim)
        elapsed = time.perf_counter() - started
        full_pixmaps = {
            page.page_number: pymupdf.Pixmap(
                extract_pages(args.pdf, dpi=args.dpi, pages=[str(page.page_number)],
                              trim_margins=False).pages[0].data
            )
            for page in result.pages
        } if trim else {}

        for page in result.pages:
            full = full_pixmaps.get(page.page_number)
            box = page.crop_box
            if box is None:
                print(f"{page.page_number:>4} {f'{page.width}x{page.height}':>11} "
                      f"{'без обрезки':>11} {'100%':>8}")
                continue
            trimmed = pymupdf.Pixmap(page.data)
            stem = out_dir / f"{page.page_number:02d}"
            (out_dir / f"{stem.name}_full_box.png").write_bytes(
                with_box(full, box, f"list {page.page_number}: crop"))
            (out_dir / f"{stem.name}_trim.png").write_bytes(page.data)
            (out_dir / f"{stem.name}_compare.png").write_bytes(
                side_by_side(full, trimmed))
            margins = (box[0], box[1], full.width - box[2], full.height - box[3])
            full_kb = len(full.tobytes("png")) // 1024
            print(f"{page.page_number:>4} {f'{full.width}x{full.height}':>11} "
                  f"{f'{trimmed.width}x{trimmed.height}':>11} "
                  f"{page.area_ratio:>7.0%} "
                  f"{f'x{full.width / trimmed.width:.2f}':>7} "
                  f"{f'{full_kb}->{page.size_bytes // 1024} KB':>14} "
                  f"{'/'.join(str(m) for m in margins):>22} "
                  f"{f'{len(page.text_items)} шт.':>10}")
        print(f"    обрезка={trim}: {elapsed:.2f} с на "
              f"{len(result.pages)} лист(ов)\n")

    print(f"Файлы: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
