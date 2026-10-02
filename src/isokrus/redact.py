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

# На столько расширяется прямоугольник вырезания, чтобы в него целиком
# поместилась рамка знака: она нарисована на пару пунктов шире подписи.
FRAME_PAD = 2.4
# Запас, с которым проверяется, не задевает ли прямоугольник размер. Раньше он
# стоял 1.2 пункта поверх ``FRAME_PAD``, и на десяти листах из-за него
# пощажены были 33 элемента — при том, что ни один размер при вырезании не
# пострадал: режим ``REMOVE_IF_COVERED`` сносит только то, что целиком лежит в
# прямоугольнике, а линия, проходящая насквозь, уцелевает. Запас был нужен
# для режима «задетая графика», который больше не используется.
TOUCH = 0.0


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


def _label_free(guard: _Guard, rect: pymupdf.Rect, pad: float = 0.6) -> bool:
    """Не заденет ли прямоугольник подпись размера.

    Запас остаётся ненулевым в отличие от проверки линий: подпись маленькая, и
    без запаса вырезание её съедает. На листе 6 подпись `2400` начиналась в
    0.9 пункта от прямоугольника `О5`, и при фоллбеке без запаса пропала
    вместе с размером.
    """
    grown = _grown(rect, pad)
    return not any(grown.intersects(label) for label in guard.labels)


def _line_crosses(guard: _Guard, rect: pymupdf.Rect, pad: float = 0.6) -> bool:
    """Пересекает ли прямоугольник размерную линию.

    Режим ``REMOVE_IF_COVERED`` такую линию не сносит, поэтому блокировать её
    незачем: на десяти листах потеря отрезков размерных линий равна нулю при
    любом расширении.
    """
    grown = _grown(rect, pad)
    return any(_segment_hits(a, b, grown) for a, b in guard.lines)


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
    # Прямоугольник вместе со своим отступом. Отступ у каждого свой: общий
    # ``pad`` либо ноль, см. фоллбек ниже. Хранить парами обязательно — при
    # двух отдельных списках ``zip`` сопоставлял прямоугольники не со своими
    # отступами, и на листе 6 подпись размера `2400` теряла первую цифру.
    wanted: list[tuple[RedactBox, float]] = []

    for box in boxes:
        area = _grown(box.as_rect(), pad)
        reason = guard.blocked(area)
        if reason is None:
            wanted.append((box, pad))
            continue
        # Расширение нужно, чтобы накрыть рамку знака: она нарисована шире
        # текста. Но расширение задевает и размерные линии рядом, и вырезание
        # блокируется целиком: на листе 1 блок `2 / DN50X50 / X 48300` стоял
        # в 0.8 пункта от линии размера `320` и оставался со всей разметкой.
        #
        # Отсюда фоллбек: тот же прямоугольник без расширения. Текст вырезается
        # точно, а огрызок рамки знака — несравнимо лучше невырезанного блока:
        # рамку модель за размер не примет, а число внутри посчитает.
        narrow_area = box.as_rect()
        if _label_free(guard, narrow_area) and not _line_crosses(guard,
                                                                  narrow_area):
            wanted.append((box, 0.0))
            continue
        skipped.append((box.mark, reason))

    if not wanted:
        return RedactReport()

    for box, shrink in wanted:
        pdf_page.add_redact_annot(_grown(box.as_rect(), shrink))
    pdf_page.apply_redactions(graphics=graphics,
                              text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    applied.extend(box for box, _ in wanted)
    return RedactReport(tuple(applied), tuple(skipped))