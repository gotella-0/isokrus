"""Удаление лишнего с листа: подписи и рамки знаков вырезаются из PDF.

Почему именно вырез, а не закрашивание по растру. Лист векторный, и
``apply_redactions`` убирает элемент из содержимого страницы: в рендере
не остаётся ни серого ореола от сглаживания, ни обрезка буквы, а подписи
и рамки исчезают и из текстового слоя — модель не видит их даже в списке
текста. Раньше зачистка закрашивала по растру (голосованием по картам
«удаляемое/оставляемое»), и от неё остались бы все три проблемы. Растровый
путь был удалён: он сложнее вдвое и работал хуже.

Что удаляется. План строится по ролям: вырезаются элементы, которые разбор
и регулярка признали не-размерами (координаты привязки, номера узлов,
опорные знаки, штамп), а вместе с подписью — рамка знака, в которую она
помещается. Выноски и стрелки сейчас не трогаются: обрезать их до того,
как известно, чьи они, нельзя, а закрашивать все подряд — значит стереть
половину размерных линий.

Главное правило модуля: **размер не трогать**. Проверка там же, в
:mod:`.redact` — прямоугольник, задевающий подпись размера или его линию,
не применяется вовсе, а попадает в отчёт.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pymupdf

from . import classify, redact

# На сколько прямоугольник выреза шире подписи: рамка знака нарисована на
# пару пунктов шире текста внутри неё.
FRAME_PAD = 2.4


@dataclass(frozen=True)
class Plan:
    """Что и как убираем с листа."""
    page_number: int
    # Подписи: вырезаются из содержимого страницы.
    text: tuple[tuple[float, float, float, float], ...] = ()
    texts: tuple[str, ...] = ()
    # Графика: только закрашивается, из PDF не вырезается.
    lines: tuple[tuple[float, float, float, float], ...] = ()
    # Размерные линии и их выноски: закрашивать нельзя.
    keep: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()

    @property
    def total(self) -> int:
        return len(self.text) + len(self.lines)

    def counts(self) -> dict[str, int]:
        return {"подписи": len(self.text), "графика": len(self.lines)}


def _keep_geometry(scan) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Отрезки, которые вырезание обязано сохранить.

    Это все размерные линии из разбора: на листе 8 размерная линия
    ``24100`` стоит вплотную к рамке знака, и без явного списка guard-режим
    резал бы наконечники.
    """
    return tuple(
        ((line.a[0], line.a[1]), (line.b[0], line.b[1]))
        for line in scan.lines
    )


def build_plan(pdf_page: pymupdf.Page, scan, only_digits: bool = False) -> Plan:
    """План удаления по ролям, назначенным регуляркой и разбором.

    Прямоугольник подписи расширяется до рамки, в которую она попала: рамка
    опорного знака нарисована шире текста, и по одному тексту она в
    прямоугольник не помещается — остаётся огрызок в виде скобки.
    """
    elements = classify.assign_roles(
        classify.elements_of(pdf_page, only_with_digits=only_digits),
        [tuple(line.label_bbox) for line in scan.lines if line.label_bbox],
    )
    frames = [pymupdf.Rect(c.rect) for c in scan.contours]

    texts: list[tuple[float, float, float, float]] = []
    names: list[str] = []
    for element in elements:
        if element.role not in classify.CUT_ROLES:
            continue
        rect = pymupdf.Rect(element.rect)
        for frame in frames:
            grown = pymupdf.Rect(frame.x0 - 1, frame.y0 - 1,
                                 frame.x1 + 1, frame.y1 + 1)
            if grown.contains(rect):
                rect |= frame
                break
        rect = pymupdf.Rect(rect.x0 - FRAME_PAD, rect.y0 - FRAME_PAD,
                            rect.x1 + FRAME_PAD, rect.y1 + FRAME_PAD)
        texts.append(tuple(rect))
        names.append(element.role)

    return Plan(
        page_number=pdf_page.number + 1,
        text=tuple(texts), texts=tuple(names),
        lines=tuple(removable_frames(pdf_page, scan, texts)),
        keep=_keep_geometry(scan),
    )


def removable_frames(
    pdf_page: pymupdf.Page, scan, texts: Sequence[tuple[float, ...]],
) -> list[tuple[float, float, float, float]]:
    """Рамки знаков, в которые попала удаляемая подпись.

    Только рамки. Выноски и стрелки — отдельный вопрос, и они пока не
    трогаются: обрезать их до того, как известно, чьи они, нельзя, а
    закрашивать все подряд — значит стереть половину размерных линий.
    """
    boxes = [pymupdf.Rect(t) for t in texts]
    out: list[tuple[float, float, float, float]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for contour in scan.contours:
        frame = pymupdf.Rect(contour.rect)
        key = tuple(int(v) for v in frame)
        if key in seen:
            continue
        grown = pymupdf.Rect(frame.x0 - 1, frame.y0 - 1,
                             frame.x1 + 1, frame.y1 + 1)
        if any(grown.contains(box) for box in boxes):
            seen.add(key)
            out.append(tuple(frame))

    return out


def apply_plan(
    pdf_page: pymupdf.Page,
    plan: Plan,
    guard: "redact._Guard | None" = None,
    pad: float = FRAME_PAD,
) -> "redact.RedactReport":
    """Вырезать подписи и рамки знаков из содержимого страницы.

    Режим графики — ``REMOVE_IF_COVERED``. Он сносит только то, что целиком
    лежит в прямоугольнике, поэтому размерная линия, пересекающая прямоугольник
    насквозь, остаётся: проверено на всех десяти листах при расширении от 0 до
    3.6 пункта — исчезло ноль отрезков из 115 размерных линий. Режим «задетая
    графика» уносил бы и их, вместе с наконечниками.

    Все прямоугольники применяются **одним** вызовом. Это требование PyMuPDF, а
    не аккуратность: после ``apply_redactions`` страница переписывается, и
    следующий вызов по той же странице графику уже не трогает.
    """
    boxes: list[redact.RedactBox] = [
        redact.RedactBox(rect=rect, mark=f"T{index}",
                         reason=plan.texts[index] if index < len(plan.texts)
                         else "")
        for index, rect in enumerate(plan.text)
    ]
    boxes += [
        redact.RedactBox(rect=rect, mark=f"F{index}", reason="рамка знака")
        for index, rect in enumerate(plan.lines)
    ]
    return redact.redact(pdf_page, boxes, guard, pad=pad)