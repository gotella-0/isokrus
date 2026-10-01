"""
Извлечение текста из PDF без LLM — чистый PyMuPDF.

Режимы:

* разборочный — текст страниц (сырой / по координатам / по блокам), поиск
  вхождений, произвольные единицы координат;
* ``--prompt`` — ровно тот блок подписей с координатами, который ``isokrus``
  подставляет в системный промт вместо догадок модели о положении размеров.

Арифметика координат берётся из ``isokrus.textmap`` — единственное
определение на проект, иначе скрипт и конвейер считали бы по-разному.

Примеры:
    python extract_text.py 02_Изометрии_10_листов.pdf
    python extract_text.py *.pdf --mode blocks --out-dir text
    python extract_text.py doc.pdf --pages 1-3,7 --json
    python extract_text.py doc.pdf --coords            # текст + координаты
    python extract_text.py doc.pdf --find "DN50"        # координаты вхождений
    python extract_text.py doc.pdf --coords --units px --dpi 320
    python extract_text.py doc.pdf --coords --units mm --origin bottom-left
    python extract_text.py 02_Изометрии_10_листов.pdf --pages 8 --prompt
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pymupdf

# Скрипт запускается из корня репозитория и должен работать без установки
# пакета, поэтому src добавляем в путь вручную — до импорта isokrus.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from isokrus import config
from isokrus.textmap import (
    extract_text_items,
    format_text_block,
    to_units,
    unit_scale,
)


# ---------------------------------------------------------------- режимы

def _words(page, min_size=0.0):
    """Слова страницы с их боксами: (x0, y0, x1, y1, текст)."""
    out = []
    for w in page.get_text("words"):
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        if min_size and (y1 - y0) < min_size:
            continue
        out.append((x0, y0, x1, y1, text))
    return out


def _cluster_lines(words, y_tol=3.0):
    """Группирует слова в строки по близости Y, затем сортирует по X."""
    rows = []
    for w in sorted(words, key=lambda w: (round(w[1], 1), w[0])):
        for row in rows:
            if abs(row[0] - w[1]) <= y_tol:
                row[1].append(w)
                row[0] = (row[0] + w[1]) / 2
                break
        else:
            rows.append([w[1], [w]])

    lines = []
    for _, row in rows:
        row.sort(key=lambda w: w[0])
        lines.append(" ".join(w[4] for w in row))
    return lines


def extract_page(page, mode="sorted", min_size=0.0, y_tol=3.0):
    if mode == "raw":
        return page.get_text()

    if mode == "sorted":
        return "\n".join(_cluster_lines(_words(page, min_size), y_tol))

    if mode == "blocks":
        parts = []
        for b in sorted(page.get_text("blocks"), key=lambda b: (round(b[1], 1), b[0])):
            txt = b[4].strip()
            if txt:
                parts.append(txt)
        return "\n".join(parts)

    raise ValueError(f"Неизвестный режим: {mode}")


# ---------------------------------------------------------------- система координат

def page_geometry(page, units="pt", dpi=72, origin="top-left"):
    """
    Отдаёт (scale, mapper, page_size) для пересчёта координат из пунктов PDF.

    units  — pt (как в PDF) | px (пиксели рендера при dpi) | in | mm
    origin — top-left (как в PDF, Y вниз) | bottom-left (CAD-привычка, Y вверх)

    Сам множитель и пересчёт живут в isokrus.textmap: конвейер и этот скрипт
    обязаны считать одинаково.
    """
    scale = unit_scale(units, dpi)
    w, h = page.rect.width, page.rect.height

    def mapper(box):
        return to_units(box, scale, h, origin)

    return scale, mapper, [round(w * scale, 2), round(h * scale, 2)]


def extract_items(page, min_size=0.0, mapper=None):
    """
    Текст страницы с координатами.

    Возвращает список строк:
        {"text", "bbox": [x0,y0,x1,y1], "center": [x,y],
         "size", "font", "dir": [dx,dy], "angle"}

    Система координат задаётся через mapper (см. page_geometry);
    по умолчанию — пункты PDF, origin в левом верхнем углу, Y вниз.
    """
    lines = []
    data = page.get_text("rawdict")

    def span_text(s):
        return "".join(c["c"] for c in s.get("chars", []))

    for block in data["blocks"]:
        if block.get("type") != 0:      # 1 = изображение
            continue
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if span_text(s).strip()]
            if not spans:
                continue
            size = max(s["size"] for s in spans)
            if min_size and size < min_size:
                continue

            spans.sort(key=lambda s: s["bbox"][0])
            x0 = min(s["bbox"][0] for s in spans)
            y0 = min(s["bbox"][1] for s in spans)
            x1 = max(s["bbox"][2] for s in spans)
            y1 = max(s["bbox"][3] for s in spans)
            dx, dy = line.get("dir", (1.0, 0.0))
            bbox = (mapper((x0, y0, x1, y1)) if mapper
                    else [round(v, 2) for v in (x0, y0, x1, y1)])

            lines.append({
                "text": " ".join(span_text(s) for s in spans).strip(),
                "bbox": bbox,
                "center": [round((bbox[0] + bbox[2]) / 2, 2),
                           round((bbox[1] + bbox[3]) / 2, 2)],
                "size": round(size, 2),
                "font": spans[0]["font"],
                "dir": [round(dx, 3), round(dy, 3)],
                "angle": round(math.degrees(math.atan2(-dy, dx)), 1),
            })

    lines.sort(key=lambda ln: (round(ln["bbox"][1], 1), ln["bbox"][0]))
    return lines


def find_items(doc, needle, whole_line=False, case_sensitive=False,
               units="pt", dpi=72, origin="top-left"):
    """
    Ищет needle и возвращает страницу + координаты каждого вхождения.

    whole_line=False -> точные вхождения через page.search_for() (bbox только совпадения)
    whole_line=True  -> совпадение по целой извлечённой строке (bbox всей строки)
    """
    hits = []
    pages = [
        p for p in range(len(doc))
        if (needle if case_sensitive else needle.lower())
        in (doc[p].get_text() if case_sensitive else doc[p].get_text().lower())
    ]
    for p in pages:
        page = doc[p]
        _, mapper, _ = page_geometry(page, units, dpi, origin)
        if not whole_line and not case_sensitive:
            for rect in page.search_for(needle):
                bbox = mapper(rect)
                hits.append({
                    "page": p + 1,
                    "text": needle,
                    "bbox": bbox,
                    "center": [round((bbox[0] + bbox[2]) / 2, 2),
                               round((bbox[1] + bbox[3]) / 2, 2)],
                })
        else:
            for ln in extract_items(page, mapper=mapper):
                hay = ln["text"] if case_sensitive else ln["text"].lower()
                if (needle if case_sensitive else needle.lower()) in hay:
                    hits.append({"page": p + 1, **ln})
    return hits


# ---------------------------------------------------------------- диапазон страниц

def parse_pages(spec, total):
    """'1-3,7' -> [0,1,2,6] (нумерация с 1, как в PDF)."""
    if not spec:
        return list(range(total))
    pages = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, _, b = part.partition("-")
            pages.extend(range(int(a) - 1, int(b)))
        else:
            pages.append(int(part) - 1)
    bad = [p + 1 for p in pages if not 0 <= p < total]
    if bad:
        sys.exit(f"Страницы вне диапазона 1..{total}: {bad}")
    return pages


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Извлечь текст из PDF (PyMuPDF, без LLM).")
    ap.add_argument("pdfs", nargs="+", type=Path, help="файл(ы) PDF")
    ap.add_argument(
        "-m", "--mode",
        choices=["raw", "sorted", "blocks"],
        default="sorted",
        help="raw — как в PDF; sorted — строки по координатам; blocks — по блокам (по умолчанию sorted)",
    )
    ap.add_argument("--pages", help="номера страниц, например 1-3,7")
    ap.add_argument("--out", type=Path, help="вывод в файл (иначе stdout)")
    ap.add_argument("--out-dir", type=Path, help="папка: <имя>.txt и <имя>/page_NN.txt")
    ap.add_argument("--min-size", type=float, default=0.0,
                    help="отбросить строки ниже указанной высоты шрифта (отсечь мусор)")
    ap.add_argument("--y-tol", type=float, default=3.0,
                    help="допуск склейки строк по Y, в пунктах (по умолчанию 3)")
    ap.add_argument("--json", action="store_true", help="вывод JSON: файл -> страницы -> текст")
    ap.add_argument("--stats", action="store_true", help="только статистика по страницам")

    ap.add_argument("--coords", action="store_true",
                    help="текст вместе с координатами (bbox в пикселях PDF)")
    ap.add_argument("--find", metavar="ТЕКСТ",
                    help="найти вхождения и вывести их координаты")
    ap.add_argument("--coords-only", action="store_true",
                    help="с --find: искать по целым строкам, а не только по тексту")
    ap.add_argument("--case-sensitive", action="store_true", help="учёт регистра при поиске")
    ap.add_argument("--page-dims", action="store_true",
                    help="добавить размеры страницы в выбранных единицах")
    ap.add_argument("--units", choices=["pt", "px", "in", "mm"], default="pt",
                    help="единицы координат: pt — как в PDF, px — пиксели рендера, in, mm "
                         "(по умолчанию pt)")
    ap.add_argument("--dpi", type=float, default=72.0,
                    help="dpi для --units px (по умолчанию 72 = 1:1 с пунктами)")
    ap.add_argument("--origin", choices=["top-left", "bottom-left"], default="top-left",
                    help="где начало координат: top-left как в PDF (Y вниз), "
                         "bottom-left как в CAD (Y вверх)")
    ap.add_argument("--prompt", action="store_true",
                    help="вывести блок подписей с координатами в пикселях рендера "
                         "— ровно то, что уходит в системный промт")
    ap.add_argument("--render-dpi", type=float, default=float(config.RENDER_DPI),
                    help="dpi рендера для --prompt (по умолчанию RENDER_DPI = "
                         f"{config.RENDER_DPI}, та же, что у картинок isokrus)")
    args = ap.parse_args()

    if args.prompt and args.render_dpi <= 0:
        sys.exit("--render-dpi должен быть больше нуля")

    if args.out and args.out_dir:
        sys.exit("Укажи что-то одно: --out или --out-dir")
    if args.coords_only and not args.find:
        sys.exit("--coords-only работает только вместе с --find")
    if args.dpi <= 0:
        sys.exit("--dpi должен быть больше нуля")
    if args.units != "px" and args.dpi != 72:
        print(f"Внимание: --dpi {args.dpi:g} игнорируется, "
              f"так как --units {args.units} (dpi влияет только на px)",
              file=sys.stderr)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for pdf_path in args.pdfs:
        if not pdf_path.is_file():
            sys.exit(f"Файл не найден: {pdf_path}")

        with pymupdf.open(pdf_path) as doc:
            # --- блок для системного промта (пиксели рендера при --dpi)
            if args.prompt:
                pages = parse_pages(args.pages, len(doc))
                zoom = args.render_dpi / 72.0
                matrix = pymupdf.Matrix(zoom, zoom)
                chunks = []
                for n in pages:
                    page = doc[n]
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    items = extract_text_items(page, matrix, args.min_size)
                    chunks.append(
                        f"===== {pdf_path.name} — лист {n + 1} "
                        f"({pixmap.width}x{pixmap.height} px, {args.render_dpi:g} dpi) =====\n"
                        + format_text_block(items, pixmap.width, pixmap.height)
                    )
                    pixmap = None
                sys.stdout.write("\n\n".join(chunks) + "\n")
                continue

            # --- поиск вхождений с координатами
            if args.find is not None:
                hits = find_items(doc, args.find,
                                  whole_line=args.coords_only,
                                  case_sensitive=args.case_sensitive,
                                  units=args.units, dpi=args.dpi,
                                  origin=args.origin)
                data = json.dumps(
                    {pdf_path.name: hits}, ensure_ascii=False, indent=2
                )
                if args.out:
                    args.out.write_text(data, encoding="utf-8")
                    print(f"-> {args.out}  ({len(hits)} совпадений)")
                else:
                    sys.stdout.write(data + "\n")
                continue

            # --- весь текст страниц с координатами
            if args.coords:
                pages = parse_pages(args.pages, len(doc))
                payload = {"_units": args.units, "_origin": args.origin,
                           "_dpi": args.dpi if args.units == "px" else None}
                for n in pages:
                    page = doc[n]
                    _, mapper, page_size = page_geometry(page, args.units,
                                                         args.dpi, args.origin)
                    entry = {"items": extract_items(page, args.min_size, mapper)}
                    if args.page_dims:
                        entry["width"], entry["height"] = page_size
                    payload[f"page_{n + 1}"] = entry
                data = json.dumps(
                    {pdf_path.name: payload}, ensure_ascii=False, indent=2
                )
                if args.out:
                    args.out.write_text(data, encoding="utf-8")
                    print(f"-> {args.out}")
                else:
                    sys.stdout.write(data + "\n")
                continue

            pages = parse_pages(args.pages, len(doc))
            texts = [
                extract_page(doc[i], args.mode, args.min_size, args.y_tol)
                for i in pages
            ]

        if args.stats:
            for n, t in zip(pages, texts, strict=True):
                report.append(f"{pdf_path.name}  стр.{n + 1:>3}  символов: {len(t):>6}  строк: {t.count(chr(10)) + 1:>4}")
            continue

        if args.json:
            payload = {
                str(pdf_path): {
                    f"page_{i + 1}": t for i, t in zip(pages, texts, strict=True)
                }
            }
            data = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            joined = "\n".join(
                f"===== {pdf_path.name} — стр. {i + 1} =====\n{t}" for i, t in zip(pages, texts, strict=True)
            )
            data = joined

        if args.out:
            args.out.write_text(data, encoding="utf-8")
            print(f"-> {args.out}  ({len(data)} символов)")
        elif args.out_dir:
            txt_path = args.out_dir / f"{pdf_path.stem}.txt"
            txt_path.write_text(data, encoding="utf-8")
            print(f"-> {txt_path}")
            for n, t in zip(pages, texts, strict=True):
                page_path = args.out_dir / f"page_{n + 1:02d}.txt"
                page_path.write_text(t, encoding="utf-8")
        else:
            sys.stdout.write(data + "\n")

    if args.stats:
        print("\n".join(report))


if __name__ == "__main__":
    main()
