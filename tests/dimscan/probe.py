"""Разведка листа: текстовый слой, заливки, штрихи и вырезка в PNG.

Не часть пайплайна dimscan, а рабочий инструмент: когда подпись приклеилась
не к тому размеру, первое, на что надо посмотреть, — сам текстовый слой, сами
заливки и глазами кусок листа. Поэтому здесь нет красивых имён: только дамп
и один файл на диск.

    python tests/dimscan/probe.py 2
    python tests/dimscan/probe.py 2 --fills
    python tests/dimscan/probe.py 3 --png page3.png
    python tests/dimscan/probe.py 3 --png num.png --crop 130,240,240,285
    python tests/dimscan/probe.py 3 --png num.png --crop 130,240,240,285 --dpi 600

Про плотность. По умолчанию (``--dpi 0``) она **не задана**, а выбирается по
задаче, и это важно: для вырезки в одно число плотность не нужна, нужна
читаемость, поэтому зум подбирается так, чтобы вырезка была около
``AUTO_TARGET_PX`` пикселей в ширину. Умножать плотность вручную — значит
угадывать; на листе A1 дефолтные 72 dpi дают нечитаемую кашу из штрихов.

Отдельно: плотность здесь влияет **только на картинку для глаз** и никак не
влияет на разбор. Детектор работает с вектором PDF в пунктах, поэтому
``--dpi`` на результат ``detect`` не влияет никак — см. ``detect.sheet_scale``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PDF = "02_Изометрии_10_листов.pdf"

# Во сколько пикселей должна быть вырезка в ширину при автоматическом зуме.
AUTO_TARGET_PX = 1200
# Плотность всего листа, когда вырезки нет: 150 dpi хватает, чтобы разглядеть
# лист целиком, и лист при этом не превращается в файл на сотни мегабайт.
FULL_PAGE_DPI = 150.0
# Потолок автозума и размера картинки, чтобы команда не подвесила машину
# вырезкой в пару пунктов на всю страницу.
MAX_DPI = 2400.0
MAX_PIXELS = 40_000_000


def _clip_of(raw: str | None) -> pymupdf.Rect | None:
    if not raw:
        return None
    parts = [float(v) for v in raw.replace(" ", "").split(",") if v]
    if len(parts) != 4:
        raise SystemExit(f"--crop ждёт четыре числа через запятую, получено {len(parts)}")
    return pymupdf.Rect(*parts)


def _zoom_for(clip: pymupdf.Rect, whole: bool, dpi: float) -> float:
    """Матрица рендера: из плотности или автоматически, но всегда в узде.

    ``zoom`` — во сколько раз картинка крупнее, чем при 72 dpi, то есть
    ровно то, что имеет смысл показывать человеку. Плотность в dpi и зум
    путать не надо: у вырезки в одно число dpi получается запредельная
    (сотни), и именно поэтому она тут и не задаётся по умолчанию.
    """
    if dpi and dpi > 0:
        zoom = dpi / 72.0
    elif whole:
        zoom = FULL_PAGE_DPI / 72.0
    else:
        zoom = min(MAX_DPI, AUTO_TARGET_PX * 72.0 / max(clip.width, 1.0)) / 72.0
    # Подрезаем по числу пикселей, а не по «на глаз»: квадрат листа при
    # автозуме в 1200 px — это уже 12 мегапикселей, дальше — уже не картинка.
    pixels = clip.width * clip.height * zoom * zoom
    if pixels > MAX_PIXELS:
        zoom *= (MAX_PIXELS / pixels) ** 0.5
    return zoom


def main() -> int:
    ap = argparse.ArgumentParser(description="Разведка одного листа")
    ap.add_argument("page", type=int, nargs="?", default=1)
    ap.add_argument("--pdf", default=PDF)
    ap.add_argument("--fills", action="store_true", help="заливки (рамки, стрелки)")
    ap.add_argument("--images", action="store_true", help="растровые блоки")
    ap.add_argument("--strokes", action="store_true", help="штрихи заданной толщины")
    ap.add_argument("--png", metavar="FILE", help="сохранить лист или вырезку в PNG")
    ap.add_argument("--crop", metavar="X0,Y0,X1,Y1",
                    help="вырезка в пунктах PDF, как в координатах дампа")
    ap.add_argument("--dpi", type=float, default=0.0,
                    help="плотность; 0 (по умолчанию) — подобрать под вырезку")
    args = ap.parse_args()

    clip = _clip_of(args.crop)
    with pymupdf.open(args.pdf) as doc:
        page = doc[args.page - 1]
        print(f"лист {args.page}, размер {page.rect}, путей {len(page.get_drawings())}")

        if args.images:
            for img in page.get_images(full=True):
                print("  image", img[:4])
            print("  текстовых блоков:", len(page.get_text("dict")["blocks"]))

        print("\n-- текст --")
        for x0, y0, x1, y1, word, block, line, word_no in page.get_text("words"):
            if clip is not None and not clip.intersects(pymupdf.Rect(x0, y0, x1, y1)):
                continue
            print(f"  [{x0:7.2f} {y0:7.2f} {x1:7.2f} {y1:7.2f}] {word!r}")

        if args.fills:
            print("\n-- заливки --")
            for i, path in enumerate(page.get_drawings()):
                if not path.get("fill"):
                    continue
                if clip is not None and not clip.intersects(path["rect"]):
                    continue
                r = path["rect"]
                print(f"  #{i:3d} type={path['type']} fill={path['fill']} "
                      f"w={path.get('width')} rect=({r.x0:.2f} {r.y0:.2f} "
                      f"{r.x1:.2f} {r.y1:.2f}) items={[k[0] for k in path['items']]}")

        if args.strokes:
            print("\n-- штрихи --")
            for i, path in enumerate(page.get_drawings()):
                if path.get("fill"):
                    continue
                if clip is not None and not clip.intersects(path["rect"]):
                    continue
                w = float(path.get("width") or 0)
                r = path["rect"]
                print(f"  #{i:3d} w={w:.2f} items={[k[0] for k in path['items']]} "
                      f"rect=({r.x0:.2f} {r.y0:.2f} {r.x1:.2f} {r.y1:.2f})")

        if args.png:
            box = clip if clip is not None else page.rect
            zoom = _zoom_for(box, whole=clip is None, dpi=args.dpi)
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(zoom, zoom), clip=box, alpha=False
            )
            target = Path(args.png)
            target.parent.mkdir(parents=True, exist_ok=True)
            pixmap.save(target)
            print(f"\n  -> {target}: {pixmap.width}x{pixmap.height} px, "
                  f"зум x{zoom:.1f} ({72.0 * zoom:.0f} dpi"
                  + ("" if args.dpi > 0 else ", подобрано") + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
