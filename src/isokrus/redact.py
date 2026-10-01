"""Вырезание лишнего из PDF, а не закрашивание картинки.

Почему именно так. Лист — векторный, и стирать элемент правильно на нём:
``add_redact_annot`` + ``apply_redactions`` удаляет из содержимого страницы и
подпись, и графику, попавшую в прямоугольник. В рендере не остаётся ничего:
ни белого прямоугольника, ни остатков подписи. Заливка белым по растру
оставляет и то, и другое — и выглядит как правка чертежа, а не как
предобработка.

Два прохода, как и задумано:

* первый — по элементам, которые модель признала не размерами: подпись
  вырезается вместе с рамкой и выноской, которых она касается;
* второй — по выноскам, которые после первого прохода остались без
  привязки: знак убрали, стрелка осталась.

Главное правило модуля: **размер не трогать**. Прямоугольник, который
задевает подпись размера или его размерную линию, не применяется вовсе, а
попадает в отчёт. Редakция с ``REMOVE_IF_TOUCHED`` сносит графику, которую
 Rectangle пересекает, а размерная линия может проходить в паре пунктов от
подписи, и молча снести её — значит испортить лист.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pymupdf

# Области слегка расширяются: подпись в PDF и нарисованная рамка знака
# расходятся на доли пункта, и по узкой рамке остаётся обрезок буквы.
TOUCH = 1.2
# На столько расширяется прямоугольник вырезания, чтобы в него целиком
# поместилась рамка знака: она нарисована на пару пунктов шире подписи.
FRAME_PAD = 2.4


def _grown(rect: pymupdf.Rect, pad: float) -> pymupdf.Rect:
    """Рамка с запасом с четырёх сторон."""
    return pymupdf.Rect(rect.x0 - pad, rect.y0 - pad,
                        rect.x1 + pad, rect.y1 + pad)


@dataclass(frozen=True)
class RedactBox:
    """Прямоугольник вырезания в пунктах PDF."""

    rect: tuple[float, float, float, float]
    mark: str
    reason: str

    def as_rect(self) -> pymupdf.Rect:
        return pymupdf.Rect(self.rect)


@dataclass(frozen=True)
class RedactReport:
    """Что вырезано, что пощажено и почему."""

    applied: tuple[RedactBox, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()   # (метка, причина)

    @property
    def total(self) -> int:
        return len(self.applied)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for box in self.applied:
            out[box.reason] = out.get(box.reason, 0) + 1
        return out


@dataclass
class _Guard:
    """Размеры листа, которые нельзя задевать ничем."""

    labels: list[pymupdf.Rect] = field(default_factory=list)
    # Размерные линии хранятся отрезками, а не их прямоугольниками: у
    # наклонной линии прямоугольник накрывает треть листа, и сравнение с ним
    # блокировало бы всё, что стоит рядом с размером, — почти весь лист.
    lines: list[tuple[pymupdf.Point, pymupdf.Point]] = field(default_factory=list)

    def blocked(self, rect: pymupdf.Rect, pad: float = TOUCH) -> str | None:
        """Что именно пострадало бы, если бы применили прямоугольник."""
        grown = _grown(rect, pad)
        for label in self.labels:
            if grown.intersects(label):
                return "подпись размера"
        for a, b in self.lines:
            if _segment_hits(a, b, grown):
                return "размерная линия"
        return None


def _segment_hits(a: pymupdf.Point, b: pymupdf.Point, rect: pymupdf.Rect,
                  step: float = 2.0) -> bool:
    """Пересекает ли отрезок прямоугольник.

    Отрезок перебирается точками с шагом ``step`` вместо точного расстояния
    до прямой: шаг в два пункта заведомо мельче толщины линии и дат��
    правильный ответ там, где прямоугольник касается её края.
    """
    length = abs(a.x - b.x) + abs(a.y - b.y)
    steps = max(1, int(length / step))
    for index in range(steps + 1):
        t = index / steps
        point = pymupdf.Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
        if rect.contains(point):
            return True
    return False


def guard_of(dimensions: Iterable) -> _Guard:
    """Собрать запрещённые области из результата разбора.

    Берутся именно пункты PDF (``DimLine``), а не пиксели готовой картинки:
    редактируется исходная страница, и рамки должны быть в её координатах.

    Концы отрезка берутся из полей ``a`` и ``b``. Имена ``start``/``end``
    здесь не подставились бы: getattr вернул бы ``None``, список запрещённых
    линий вышел бы пустым, и второй проход снёс бы размерные линии вместе с
    наконечниками — молча, потому что «пощажено» показывало бы ноль.
    """
    labels: list[pymupdf.Rect] = []
    lines: list[tuple[pymupdf.Point, pymupdf.Point]] = []
    for line in dimensions:
        box = getattr(line, "label_bbox", None)
        if box is not None:
            labels.append(pymupdf.Rect(box))
        a, b = getattr(line, "a", None), getattr(line, "b", None)
        if a is not None and b is not None:
            lines.append((pymupdf.Point(a), pymupdf.Point(b)))
        # У выноски-указателя подпись стоит не на самой линии, и без её рамки
        # подпись проходит мимо запрета.
        lead = getattr(line, "leader_end", None)
        if lead is not None and a is not None:
            lines.append((pymupdf.Point(a), pymupdf.Point(lead)))
    return _Guard(labels=labels, lines=lines)


def boxes_from_roles(
    elements: Sequence,
    keep: bool,
    frame_of: Sequence[pymupdf.Rect] = (),
) -> list[RedactBox]:
    """Прямоугольники вырезания по решению о роли элементов.

    ``keep=True`` — прямоугольники тех, кого **оставляем** (размеры);
    ``keep=False`` — тех, кого **убираем**. Разделение прямое, а не «всё, что
    не размер»: решение принимает не этот модуль, и по умолчанию перечислять
    роли опаснее, чем спросить по флагу.

    Прямоугольник растёт до рамки, в которую подпись попала: рамка опорного
    знака нарисована шире текста, и по одному тексту в прямоугольник не
    помещается — остаётся огрызок в виде скобки.
    """
    frames = [pymupdf.Rect(r) for r in frame_of]
    out: list[RedactBox] = []
    for element in elements:
        wanted = getattr(element, "keep", False)
        if wanted is not keep:
            continue
        rect = pymupdf.Rect(element.rect)
        for frame in frames:
            grown = _grown(frame, 1.0)
            if grown.contains(rect):
                rect |= frame
                break
        out.append(
            RedactBox(
                rect=(rect.x0, rect.y0, rect.x1, rect.y1),
                mark=getattr(element, "mark", ""),
                reason=getattr(element, "reason", "") or getattr(element, "role", ""),
            )
        )
    return out


def orphan_removals(
    pdf_page: pymupdf.Page,
    limits,
    removed: Sequence[RedactBox],
    keep_segments: Sequence[tuple[pymupdf.Point, pymupdf.Point]] = (),
) -> list[RedactBox]:
    """Выноски, оставшиеся без знака, — второй проход.

    После первого прохода на листе остаются стрелки, у которых подпись
    вырезана: на картинке они неотличимы от стрелок размера, и модель
    посчитает их ещё одним отрезком. Ищутся тонкие отрезки, у которых один
    из концов упирается в вырезанную область, а сами они ни с одной размерной
    линией не совпадают.

    ``keep_segments`` — то, что нужно уберечь помимо размерных линий: линии,
    за которые модель зацепила подпись. Их тоже нельзя сносить, иначе у размера
    пропадёт наконечник.
    """
    from .clean import _segments  # локальный импорт: цикл clean -> redact

    segments = _segments(pdf_page.get_drawings(), limits)
    guard = _Guard(lines=list(keep_segments))
    out: list[RedactBox] = []
    for index, (a, b) in enumerate(segments):
        pa = pymupdf.Point(a)
        pb = pymupdf.Point(b)
        if abs(pa.x - pb.x) + abs(pa.y - pb.y) < 4.0:
            continue
        if not any(_grown(pymupdf.Rect(box.rect), 3.0).contains(pa)
                   or _grown(pymupdf.Rect(box.rect), 3.0).contains(pb)
                   for box in removed):
            continue
        if guard.blocked(pymupdf.Rect(pa, pb)) is not None:
            continue
        out.append(
            RedactBox(
                rect=(min(pa.x, pb.x), min(pa.y, pb.y),
                      max(pa.x, pb.x), max(pa.y, pb.y)),
                mark=f"L{index}", reason="выноска без знака",
            )
        )
    return out


def orphaned_leaders(
    pdf_page: pymupdf.Page,
    limits,
    removed: Sequence[RedactBox],
) -> list[RedactBox]:
    """Прежнее имя прежней функции: выноски без знака."""
    from .clean import _segments  # локальный импорт: цикл clean -> redact

    segments = _segments(pdf_page.get_drawings(), limits)
    guard = _Guard(labels=[pymupdf.Rect(*b.rect) for b in removed])
    out: list[RedactBox] = []
    for index, (a, b) in enumerate(segments):
        pa = pymupdf.Point(a)
        pb = pymupdf.Point(b)
        if abs(pa.x - pb.x) + abs(pa.y - pb.y) < 4.0:
            continue
        touched = [
            box for box in removed
            if _grown(pymupdf.Rect(box.rect), 3.0).contains(pa)
            or _grown(pymupdf.Rect(box.rect), 3.0).contains(pb)
        ]
        if not touched:
            continue
        if guard.blocked(pymupdf.Rect(pa, pb)) is not None:
            continue
        out.append(
            RedactBox(
                rect=(min(pa.x, pb.x), min(pa.y, pb.y),
                      max(pa.x, pb.x), max(pa.y, pb.y)),
                mark=f"L{index}", reason="выноска без знака",
            )
        )
    return out


def redact(
    pdf_page: pymupdf.Page,
    boxes: Sequence[RedactBox],
    guard: _Guard | None = None,
    graphics: int = pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
    pad: float = FRAME_PAD,
) -> RedactReport:
    """Вырезать элементы со страницы.

    Режим — ``PDF_REDACT_LINE_ART_REMOVE_IF_COVERED``, а не ``..._IF_TOUCHED``.
    Разница видна на опорных знаках: выноска «О3» упирается в размерную линию,
    и режим «задетая графика» унёс бы её вместе с наконечниками. «Перекрытая
    графика» сносит только то, что целиком лежит в прямоугольнике, поэтому
    размерная линия рядом остаётся.

    Прямоугольник расширяется на ``FRAME_PAD``: рамка знака нарисована чуть
    шире подписи, и по рамке подписи она не попала бы внутрь — осталась бы
    огрызком рамки без подписи.
    """
    guard = guard or _Guard()
    applied: list[RedactBox] = []
    skipped: list[tuple[str, str]] = []
    wanted: list[RedactBox] = []

    for box in boxes:
        area = _grown(box.as_rect(), pad)
        reason = guard.blocked(area)
        if reason is not None:
            skipped.append((box.mark, reason))
            continue
        wanted.append(box)

    if not wanted:
        return RedactReport()

    for box in wanted:
        pdf_page.add_redact_annot(_grown(box.as_rect(), pad))
    pdf_page.apply_redactions(graphics=graphics,
                              text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    applied.extend(wanted)
    return RedactReport(tuple(applied), tuple(skipped))