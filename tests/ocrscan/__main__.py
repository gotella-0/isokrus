"""Проверка: координаты туда-обратно сходятся.

Отдельно от всех остальных модулей, потому что ошибка в переводе координат
не бросает исключение — она молча сдвигает все размеры, и после этого любая
метрика говорит о чём угодно, кроме реального качества распознавания.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pymupdf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.ocrscan import render  # noqa: E402

PDF = "02_Изометрии_10_листов.pdf"


def main() -> int:
    ok = True
    with pymupdf.open(PDF) as document:
        page = document.load_page(7)
        for name, profile in render.PROFILES.items():
            raster = render.render_page(page, profile)
            scale = raster.px_per_pt
            print(f"{name:9s} dpi={profile.dpi:4d}  {raster.width}x{raster.height}"
                  f"  px/pt={scale:.3f}  сдвиг обрезки={raster.trim_offset_px}")

            # Точки по углам и в центре: углы страдают от перекоса сильнее
            # всего, поэтому проверять только центр нельзя.
            probes = np.array([
                [0.0, 0.0],
                [1190.55, 0.0],
                [0.0, 841.89],
                [1190.55, 841.89],
                [595.0, 420.0],
                [300.0, 250.0],
            ], dtype=np.float64)

            back = raster.to_points(raster.px_from_pt(probes))
            err = float(np.abs(back - probes).max())
            flag = "ок" if err < 1e-6 else "РАСХОЖДЕНИЕ"
            if err >= 1e-6:
                ok = False
            print(f"          туда-обратно: макс. ошибка {err:.2e} pt  {flag}")

            # Длина отрезка: в PDF это геометрическая мера, и она обязана
            # совпасть с эталоном из векторного detect.py независимо от
            # профиля. Иначе сравнение сумм бессмысленно.
            a_pt, b_pt = (486.0, 300.0), (900.0, 500.0)
            a_px = raster.px_from_pt(np.array([a_pt], dtype=np.float64))[0]
            b_px = raster.px_from_pt(np.array([b_pt], dtype=np.float64))[0]
            measured = raster.length_pt(tuple(a_px), tuple(b_px))
            exact = float(np.hypot(*(np.array(b_pt) - np.array(a_pt))))
            delta = abs(measured - exact)
            if delta > 1e-6:
                ok = False
            print(f"          длина 450.00 pt = {measured:.4f} pt "
                  f"(расхождение {delta:.2e})  {'ок' if delta < 1e-6 else 'РАСХОЖДЕНИЕ'}")
            print()

    print("ИТОГО:", "все проверки прошли" if ok else "есть расхождения")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
