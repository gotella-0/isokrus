"""Сопоставление подписи с линией: смотрим на вырез, а не гадаем по тексту.

Зачем. Разбор умеет находить размерные линии, но не умеет связывать их с
подписями. А связка и есть то, из-за чего лист не читается: число ``71``
рядом с чужой выноской для модели выглядит так же, как длина трубы, и
единственный способ различить — посмотреть на картинку.

Что показываем модели. Один вырез вокруг подписи. Внутри него сама подпись,
наконечники и кусок линии, к которой она стоит. Размер выреза задан долей
листа, а не числом пикселей, поэтому при смене dpi картинка остаётся тем же
куском чертежа в большем разрешении — и модель видит буквы, а не лестницу
ступенек.

Почему ответ с меткой, а не «да/нет». Модель отвечает на вопрос «что это за
подпись и к какой линии она относится» и возвращает метку того элемента, о
 котором говорит. Разбор ответа превращает метку в прямоугольник: если
модель назвала элемент, тот элемент — размер и его линия неприкосновенна. Если
она ответила на вопрос про другой элемент или ушла в ``непонятно``, элемент
считается мусором. Так решается и вторая задача — убрать выноску, которая
осталась без знака: она удаляется вместе с элементом, чьим она признана.

Модуль ничего не вырезает: он только говорит, что и куда. Вырезает
:mod:`.redact`, и там же проверка, что размер не задет.
"""

from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

from pydantic import BaseModel, Field

from . import classify, config, llm

# Что модель может сказать о линии рядом с подписью.
LINE_DIMENSION = "размерная"
LINE_LEADER = "выноска"
LINE_NONE = "нет"

# Модель отвечает на вопрос про один элемент и обязана назвать его метку.
# Метка нужна и для проверки: ответ без неё бесполезен, из него нечего взять.
SYSTEM = """\
Ты читаешь изометрический чертёж трубопровода. Тебе показывают **один вырез** \
листа вокруг текстовой подписи. Ответь на два вопроса про эту подпись: что это \
и к какой линии она относится.

## Что уже известно программно

Содержимое подписи разобрано: текст известен. Все размерные линии с \
наконечниками на листе уже найдены. Неизвестно одно — к какой линии относится \
эта подпись и является ли она вообще размером.

Ты не считаешь длину трассы и не выбираешь размеры. Ты отвечаешь про одну \
подпись.

## Как отличить

* **размер** — подпись стоит у линии, у которой **оба конца в наконечниках \
(стрелки внутрь отрезка). Число при этом — длина в миллиметрах.
* **выноска знака** — подпись лежит в прямоугольной рамке, и от неё идёт линия \
с наконечником **на конце**, упирающаяся в трубу. Так помечают опоры, задвижки, \
штуцары. Это не размер.
* **координата привязки** — подпись внутри блока вида «СМ. … / X … / Y … / \
Z+ …», числа шестизначные, рядом номер чертежа.
* **номер узла** — число или знак внутри квадратной рамки.
* **опорный знак** — «О3», «О4», «4/3» в рамке.
* **штамп** — «Лист … из …», «SW_…».
* **обозначение трубы** — «DN50», «Н.О.», «ШТУРВАЛ UP».

## Правила ответа

1. Смотри на картинку, а не на текст подписи. Подпись «6000» может быть \
размером, а «78500» — координатой, и по одному тексту это не различить.
2. Наконечник — **жирный треугольник** на конце линии. У размерной линии их два, \
по краям. У выноски знака один, и он смотрит на трубу.
3. Если рядом видна размерная линия с двумя наконечниками — это размер, даже \
если подпись стоит не на самой линии, а на выноске от неё.
4. Не выдумывай: если по вырезу нельзя понять, выбери `непонятно`.
5. Отвечай строго одной строкой JSON, без пояснений вне JSON.
"""


class Verdict(BaseModel):
    """Ответ модели про один элемент."""

    mark: str = Field(
        description="Метка элемента из вопроса, копией. Не придумывай свою."
    )
    role: str = Field(
        description="Что за подпись: размер, координата привязки, номер узла, "
                    "опорный знак, штамп, обозначение трубы, прочее, непонятно"
    )
    line: str = Field(
        description=f"Что за линия рядом: {LINE_DIMENSION} — размерная с "
                    f"наконечниками по краям, {LINE_LEADER} — выноска знака, "
                    f"{LINE_NONE} — линии рядом не видно"
    )
    confidence: float = Field(
        description="Насколько уверен, от 0 до 1"
    )


@dataclass(frozen=True)
class Judgement:
    """Решение по одному элементу листа."""

    mark: str
    role: str
    line: str
    confidence: float
    source: str = "llm"           # "llm" или "regex" (если модель не спросила)
    reason: str = ""

    @property
    def sure(self) -> bool:
        """Насколько решение можно принимать к исполнению."""
        return self.confidence >= config.PRUNE_JUDGE_MIN_CONFIDENCE

    @property
    def keep(self) -> bool:
        """Оставить подпись на листе.

        Размер остаётся, если модель назвала его **и** уверилась в ответе.
        Неуверенный ответ — это не «оставь», а «не знаю»: вырезать на
        основании догадки значит рискнуть потерять размер, а оставить лишний
        номер узла не стоит ничего — в модель уходит и картинка, и список
        размеров, и лишний номер в списке не числится.
        """
        return self.role == classify.ROLE_DIMENSION and self.sure

    @property
    def anchor(self) -> bool:
        """Стоит ли подпись у размерной линии — по ней решаем, чья это линия.

        Отдельное свойство, а не часть ``keep``: подпись может быть размером по
        смыслу, но стоять у выноски, и тогда линия, к которой она привязана,
        определяется иначе. Сводить эти два вопроса в одно свойство — значит
        потерять один из них.
        """
        return self.line == LINE_DIMENSION and self.sure


@dataclass
class Judgements:
    """Решения по всем элементам листа."""

    page_number: int
    items: dict[str, Judgement] = field(default_factory=dict)
    usage: llm.Usage | None = None
    # (метка, причина) для элементов, о которых модель не ответила.
    failed: tuple[tuple[str, str], ...] = ()

    def of(self, element: classify.Element) -> Judgement:
        return self.items.get(element.mark) or _fallback(element)

    def kept(self) -> list[Judgement]:
        return [judge for judge in self.items.values() if judge.keep]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for judge in self.items.values():
            key = f"{judge.role} / {judge.line}"
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _fallback(element: classify.Element) -> Judgement:
    """Что решить, если модель про элемент не ответила.

    Ответ без роли безопаснее удаления: элемент с буквами уходит, голое число
    остаётся и попадёт в разбор как подпись размера. Лучше лишний кандидат в
    сумме, чем молча вырезанный размер.
    """
    return Judgement(
        mark=element.mark, role=element.role, line=LINE_NONE, confidence=0.0,
        source="regex", reason=element.reason or "модель не ответила",
    )


def _question(element: classify.Element) -> str:
    return (
        f"Метка элемента: {element.mark}\n"
        f"Подпись: {element.text}\n"
        f"Верни JSON с полями mark (ровно {element.mark}), role, line, confidence."
    )


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def judge_element(
    element: classify.Element,
    usage: llm.Usage | None = None,
    model: str | None = None,
) -> Judgement:
    """Спросить модель про один элемент."""
    if not element.image:
        return _fallback(element)
    verdict = llm.call_structured(
        system=SYSTEM,
        user=_question(element),
        response_model=Verdict,
        model=model or config.PRUNE_JUDGE_MODEL,
        reasoning_effort=config.PRUNE_JUDGE_REASONING,
        image_data_urls=[_data_url(element.image)],
        usage=usage,
    )
    # Модель обязана назвать именно тот элемент, о котором её спрашивали. Если
    # назвала другой, ответ не про этот элемент, и брать его нельзя: подмена
    # метки — верный признак того, что модель запуталась в вырезах.
    if verdict.mark.strip() != element.mark:
        return Judgement(
            mark=element.mark, role=element.role, line=LINE_NONE,
            confidence=0.0, source="llm",
            reason=f"модель ответила про {verdict.mark}, а спросили про "
                   f"{element.mark}",
        )
    return Judgement(
        mark=verdict.mark.strip(),
        role=_normalise(verdict.role),
        line=_normalise_line(verdict.line),
        confidence=max(0.0, min(1.0, float(verdict.confidence))),
        source="llm", reason=f"модель: {verdict.role.strip()}",
    )


_ROLE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (classify.ROLE_DIMENSION, ("размер", "длина", "габарит")),
    (classify.ROLE_COORDINATE, ("координат", "привязк", "ось", "осг")),
    (classify.ROLE_NODE, ("узел", "номер узла", "позицион")),
    (classify.ROLE_SUPPORT, ("опор", "кран", "задвижк", "штуцар")),
    (classify.ROLE_TITLE, ("штамп", "лист", "чертёж", "чертеж", "комплект")),
    (classify.ROLE_SPEC, ("труб", "материал", "dn", "обозначение")),
)


def _normalise(role: str) -> str:
    """Привести ответ модели к одной из ролей модуля.

    Модель отвечает свободным текстом, и ровно те формулировки, что есть в
    правилах, она не обязан повторять. Неопознанное остаётся как есть и
    трактуется как «не размер».
    """
    text = role.strip().lower()
    if "непонят" in text:
        return classify.ROLE_OTHER
    for role_name, words in _ROLE_KEYWORDS:
        if any(word in text for word in words):
            return role_name
    return classify.ROLE_OTHER


def _normalise_line(line: str) -> str:
    text = line.strip().lower()
    if "вынос" in text or "знак" in text or "указател" in text:
        return LINE_LEADER
    if "размер" in text or "наконечник" in text:
        return LINE_DIMENSION
    return LINE_NONE


def judge_page(
    pdf_page,
    elements: Sequence[classify.Element],
    model: str | None = None,
    max_parallel: int | None = None,
) -> Judgements:
    """Спросить модель про все элементы листа.

    Элементы с однозначной ролью по регулярке — координаты, номера узлов,
    штампы — модели не показываются: по собственному тексту они уже опознаны,
    а картинка может только сбить. Показывается то, что регуляркой не решается,
    то есть **голые числа**: для них разбор нужен, а он ошибается в обе стороны.
    """
    asked = [e for e in elements if e.role == classify.ROLE_OTHER]
    workers = max(1, min(
        max_parallel or config.PRUNE_MAX_PARALLEL, len(asked) or 1))
    usage = llm.Usage()

    items: dict[str, Judgement] = {}
    failures: list[tuple[str, str]] = []
    if asked:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(judge_element, e, usage, model): e
                    for e in asked}
            for job, element in jobs.items():
                try:
                    items[element.mark] = job.result()
                except Exception as exc:      # noqa: BLE001 — решение об одном
                    # Один неудачный вырез не должен ронять весь лист: элемент
                    # остаётся, а причина пишется в отчёт. Удалять по
                    # отсутствию ответа нельзя — это молчаливая потеря размера.
                    failures.append((element.mark, llm._api_error_message(exc)))
                    items[element.mark] = _fallback(element)

    for element in elements:
        if element.mark not in items:
            verdict = classify.role_of_text(element.text)
            role, reason = verdict if verdict else (element.role, element.reason)
            items[element.mark] = Judgement(
                mark=element.mark, role=role,
                line=LINE_DIMENSION if role == classify.ROLE_DIMENSION else LINE_NONE,
                confidence=1.0, source="regex", reason=reason,
            )

    result = Judgements(page_number=pdf_page.number + 1, items=items, usage=usage)
    result.failed = tuple(failures)
    return result


def as_json(judgements: Judgements) -> str:
    return json.dumps(
        {mark: {"role": j.role, "line": j.line, "confidence": j.confidence,
                "source": j.source}
         for mark, j in judgements.items.items()},
        ensure_ascii=False, indent=2,
    )
