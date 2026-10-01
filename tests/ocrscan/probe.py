"""Пробный прогон движка на одном листе: сколько он вообще нашёл.

Отдельный модуль и отдельная команда — потому что первая же проверка после
установки движка должна отвечать на простой вопрос «работает ли он и видит ли
цифры», а не падать в середине сетки экспериментов. Плюс сразу видно
структуру ответа: рамки в пунктах, уверенность, доля чисел среди всех слов.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.ocrscan import engines, render, words  # noqa: E402

DEFAULT_PDF = "02_Изометрии_10_листов.pdf"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проба движка OCR на одном листе")
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    parser.add_argument("--engine", default="tess", choices=sorted(engines.ENGINES))
    parser.add_argument("--profile", default="clean", choices=sorted(render.PROFILES))
    parser.add_argument("--page", type=int, default=8)
    parser.add_argument("--psm", type=int, default=None)
    parser.add_argument("--psm-list", default="11,12,6")
    parser.add_argument("--limit", type=int, default=25, help="сколько слов показать")
    parser.add_argument("--out", default=None, help="выгрузить все слова в JSON")
    args = parser.parse_args(argv)

    import pymupdf

    print("готовность движков:")
    for name, ok, detail in engines.report_availability():
        print(f"  {name:8s} {'готов' if ok else 'НЕ ГОТОВ'}  {detail}")
    print()

    engine = engines.build(args.engine)
    ok, detail = engine.available()
    if not ok:
        print(f"движок {args.engine} недоступен: {detail}")
        return 2

    profile = render.PROFILES[args.profile]
    with pymupdf.open(args.pdf) as document:
        page = document.load_page(args.page - 1)
        raster = render.render_page(page, profile)
    print(f"лист {args.page}, профиль {args.profile}, картинка {raster.width}x{raster.height}")
    print()

    for psm in [int(v) for v in args.psm_list.split(",") if v.strip()]:
        result = engine.read_page(raster, psm=psm)
        numbers = result.numbers
        print(f"  psm={psm:2d}  слов {len(result.words):4d}  "
              f"из них чисел {len(numbers):3d}  {result.seconds:6.1f} с")
        for item in numbers[: args.limit]:
            parsed = words.parse(item.text)
            box = item.box
            print(f"      {item.text!r:>10s} conf={item.conf:5.1f} "
                  f"value={parsed.value!s:>8s} "
                  f"box=({box[0]:7.1f},{box[1]:7.1f})-({box[2]:7.1f},{box[3]:7.1f}) "
                  f"h={item.height:4.1f}pt")
        print()

    if args.out:
        result = engine.read_page(raster, psm=args.psm or 11)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "pdf": args.pdf, "page": args.page, "profile": args.profile,
                    "engine": args.engine,
                    "summary": result.summary(),
                    "words": [w.as_dict() for w in result.words],
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
