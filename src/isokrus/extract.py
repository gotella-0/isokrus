"""Извлечение изображений страниц из PDF.

Шаги на лист идут именно в таком порядке::

    открыть страницу -> dimscan (вектор) -> рендер -> trim -> подписи

Сначала **разбор размеров**, потом картинка. Детектор работает с вектором
PDF в пунктах, поэтому обрезка полей на нём не сказывается никак — но
держать его до рендера всё же правильно: это единственный шаг, где страница
ещё открыта, и на тяжёлом листе он отрабатывает раньше, чем появится хоть
один пиксель. Обрезать перед разбором было бы бессмысленно дороже, а не
дешевле.

Изображения рендерятся в память и никуда не сохраняются на этом этапе:
объекты ``bytes`` живут до конца конвейера, а на диск попадают уже
после обработки (см. ``isokrus.pipeline``), чтобы в output всегда лежали
именно те картинки, которые ушли в LLM.

Шаги рендера и обрезки менять местами нельзя: сначала текст, потом обрезка
означала бы ручной сдвиг всех рамок, который легко забыть. Рамки размеров
пересчитываются в пиксели **обрезанной** картинки той же матрицей, поэтому
в промт, в разметку и на картинку попадают одни и те же координаты.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import pymupdf

from . import config
from .dimscan import DimensionMark, DimParams, detect_page, marks_in_pixels
from .errors import ExtractionError
from .overlay import render_overlay
from .prune import apply_plan, build_plan
from .redact import guard_of
from .textmap import TextItem, box_to_pixels, extract_text_items, inside_box, shifted_matrix
from .trim import trim

MimeType = str


@dataclass(frozen=True)
class PageImage:
    """Одно изображение страницы, целиком в памяти.

    ``text_items`` и ``dimensions`` лежат в той же системе координат, что и
    ``data``: обе части получены из одного рендера одной матрицей. Поэтому
    подпись, найденная в текстовом слое, и размер, найденный в векторе,
    показываются ровно там же, где нарисованы на листе.
    """

    page_number: int  # 1-based, как на чертеже
    data: bytes  # исходные байты изображения (PNG)
    width: int
    height: int
    mime_type: MimeType = config.IMAGE_MIME
    label: str = ""  # "list_01.png"
    text_items: tuple[TextItem, ...] = ()  # подписи PDF в пикселях этой картинки
    # Размеры, найденные в векторе PDF, — тоже в пикселях этой картинки.
    # В отличие от text_items здесь уже решено, **что является размером**:
    # позиционные знаки в квадратах, координаты привязки и марки труб сюда
    # не попадают, и модель не тратит на них внимание.
    dimensions: tuple[DimensionMark, ...] = ()
    # Отсеянные числа — значениями, чтобы модель знала, что нарисовано, но
    # размером не является. По картинке это неразличимо, а именно здесь
    # чаще всего и берут лишнее в расчёт.
    rejected_numbers: tuple[str, ...] = ()
    # Лист с подсвеченными размерами: у каждой размерной линии нарисована
    # метка P7 и наконечники (см. ``overlay.py``). Уходит в модель вместо
    # исходного PNG. На диск по-прежнему кладётся исходник из ``data``:
    # подсветка нужна модели, а человеку для сверки нужна настоящая выдача.
    # Лист с закрашенными знаками, координатами привязки и прочим шумом
    # (см. ``clean.py``). Уходит в модель вместо ``data``; на диск по-прежнему
    # кладётся исходник: зачистка — улучшение для машины, а не правка чертежа.
    cleaned: bytes = b""
    overlay: bytes = b""
    meta: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def name(self) -> str:
        return self.label or f"page_{self.page_number:02d}.png"

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def crop_box(self) -> tuple[int, int, int, int] | None:
        """Рамка обрезки ``(x0, y0, x1, y1)`` или ``None``, если не обрезали."""
        info = self.meta.get("trim")
        return tuple(info["box"]) if info else None  # type: ignore[return-value]

    @property
    def area_ratio(self) -> float:
        """Доля площади листа, ушедшей в модель: 1.0 — лист не обрезали."""
        info = self.meta.get("trim")
        return float(info["area_ratio"]) if info else 1.0

    def base64_image(self) -> str:
        """base64-представление для OpenAI-совместимого API."""
        import base64

        return base64.b64encode(self.model_image()).decode("utf-8")

    def model_image(self) -> bytes:
        """Байты изображения, которые уходят в модель.

        Сначала подсвеченный лист, потом зачищенный, иначе исходный. Подмена
        происходит здесь, а не в конвейере, по одной причине: отправить
        обработанный лист и сохранить исходник — две разные вещи, и если бы
        выбор делал конвейер, легко было бы сохранить не то, что ушло в
        модель, а это ломает сверку результата с картинкой.
        """
        return self.overlay or self.cleaned or self.data

    def data_url(self) -> str:
        """Готовая data:-ссылка для ``image_url``."""
        return f"data:{self.mime_type};base64,{self.base64_image()}"

    def matrix(self) -> pymupdf.Matrix:
        """Матрица «пункты PDF -> пиксели этой картинки».

        Уже сдвинутая на обрезку, поэтому рамка из вектора попадает туда же,
        куда подпись из текстового слоя. Считается из одних и тех же значений
        каждый раз: иначе рамки из разных источников разъедутся, и это будет
        выглядеть как ошибка фильтра, а не как ошибка пересчёта.
        """
        zoom = float(self.meta.get("dpi") or 300.0) / 72.0
        info = self.meta.get("trim")
        dx, dy = (info["box"][0], info["box"][1]) if info else (0, 0)
        return shifted_matrix(pymupdf.Matrix(zoom, zoom), dx, dy)

    def pixel_box(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        """Рамка из пунктов PDF в пиксели этой картинки."""
        return box_to_pixels(box, self.matrix())

    def pt_to_px(self, points: float) -> float:
        """Величина в пунктах PDF -> пиксели картинки."""
        return float(points) * float(self.meta.get("dpi") or 300.0) / 72.0

    def save(self, directory: Path, stem: str | None = None) -> Path:
        """Сохранение исходных байтов на диск. Возвращает путь к файлу."""
        directory.mkdir(parents=True, exist_ok=True)
        suffix = self.mime_type.split("/")[-1] or "png"
        target = directory / f"{stem or self.name.rsplit('.', 1)[0]}.{suffix}"
        target.write_bytes(self.data)
        return target


@dataclass
class ExtractionResult:
    """Результат извлечения: страницы в памяти плюс метаданные PDF."""

    pages: list[PageImage]
    source_pdf: Path
    page_count: int

    def __iter__(self) -> Iterator[PageImage]:
        return iter(self.pages)

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> PageImage:
        return self.pages[index]


def _scale_for(page: pymupdf.Page, dpi: int, max_side: int) -> pymupdf.Matrix:
    """Матрица рендера с учётом лимита по большей стороне.

    Размер в пикселях = размер в пунктах * zoom, поэтому сравнивать надо
    именно пункты, а не дюймы.
    """
    zoom = dpi / 72.0
    longest_pt = max(page.rect.width, page.rect.height)
    if longest_pt * zoom > max_side:
        zoom = max_side / longest_pt
    return pymupdf.Matrix(zoom, zoom)


def _parse_selection(page_count: int, pages: Sequence[str] | None) -> list[int]:
    """Разбор ``--pages``: ``"1-10"``, ``"1,3,7"``, ``"all"``, ``None``."""
    if not pages:
        return list(range(1, page_count + 1))

    wanted: set[int] = set()
    for chunk in pages:
        chunk = chunk.strip()
        if not chunk or chunk.lower() == "all":
            continue
        if "-" in chunk:
            start_s, _, end_s = chunk.partition("-")
            try:
                start, end = int(start_s), int(end_s)
            except ValueError as exc:
                raise ExtractionError(f"Некорректный диапазон страниц: {chunk!r}") from exc
            if start > end:
                start, end = end, start
            wanted.update(range(start, end + 1))
        else:
            try:
                wanted.add(int(chunk))
            except ValueError as exc:
                raise ExtractionError(f"Некорректный номер страницы: {chunk!r}") from exc

    unknown = sorted(n for n in wanted if not 1 <= n <= page_count)
    if unknown:
        raise ExtractionError(
            f"Страницы {unknown} отсутствуют в PDF (в документе {page_count})"
        )
    return sorted(wanted)


def extract_pages(
    pdf_path: str | Path,
    dpi: int | None = None,
    pages: Sequence[str] | None = None,
    max_side: int | None = None,
    stem: str | None = None,
    with_text: bool | None = None,
    trim_margins: bool | None = None,
    with_dimensions: bool | None = None,
    dim_min_length: float | None = None,
    with_overlay: bool | None = None,
    with_clean: bool | None = None,
    clean_filters: Sequence[str] | None = None,
) -> ExtractionResult:
    """Открыть PDF и отрендерить выбранные страницы в памяти.

    :param pdf_path: путь к PDF, подаётся через CLI.
    :param dpi: плотность рендера (по умолчанию ``config.RENDER_DPI``).
    :param pages: выборка страниц вида ``["1-10"]`` или ``["1", "3"]``.
    :param max_side: ограничение по большей стороне в пикселях.
    :param stem: префикс имён сохраняемых файлов (по умолчанию — имя PDF).
    :param with_text: извлекать ли текстовый слой (по умолчанию
        ``config.EXTRACT_PDF_TEXT``). Подписи кладутся в ``PageImage.text_items``
        в пикселях той же картинки, что ушла в модель.
    :param trim_margins: обрезать ли пустые поля (по умолчанию
        ``config.TRIM_MARGINS``). Подписи и размеры при этом извлекаются
        уже в координатах обрезанной картинки.
    :param with_dimensions: искать ли размеры в векторе (по умолчанию
        ``config.USE_DIMSCAN``). Выключается флагом ``--no-dimscan``, когда
        нужно сравнить прогон с этим блоком и без него.
    :param dim_min_length: минимальная длина размерной линии, в пунктах
        эталонного листа (порог сам пересчитывается под масштаб листа).
    :param with_clean: закрашивать ли на картинке то, что размером не является
        (по умолчанию ``config.DIMSCAN_CLEAN``). Работает только вместе с
        ``with_dimensions``: зачистка опирается на то, что разбор уже отверг
        эти числа, и без разбора их нечем обосновать.
    :param clean_filters: какие зачистки применять (по умолчанию
        ``config.DIMSCAN_CLEAN_FILTERS``).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise ExtractionError(f"PDF не найден: {path}")
    if not path.is_file():
        raise ExtractionError(f"Ожидался файл, а не каталог: {path}")

    dpi = dpi or config.RENDER_DPI
    max_side = max_side or config.MAX_IMAGE_SIDE
    want_text = config.EXTRACT_PDF_TEXT if with_text is None else with_text
    want_trim = config.TRIM_MARGINS if trim_margins is None else trim_margins
    want_dims = config.USE_DIMSCAN if with_dimensions is None else with_dimensions
    want_overlay = config.DIMSCAN_OVERLAY if with_overlay is None else with_overlay
    # Зачистка без разбора размеров бессмысленна: она закрашивает то, что
    # разбор отверг, а отвергать нечем. Молча выключаем, а не падаем.
    want_clean = (config.DIMSCAN_CLEAN if with_clean is None else with_clean) \
        and want_dims
    # ``clean_filters`` больше не используется: зачистка переехала с закрашивания
    # по растру на вырезание из PDF, где вид зачистки задаётся ролью элемента,
    # а не списком фильтров. Параметр оставлен, чтобы не ломать вызовы.
    _ = clean_filters

    try:
        document = pymupdf.open(path)
    except Exception as exc:  # pymupdf бросает разные исключения
        raise ExtractionError(f"Не удалось открыть PDF {path.name}: {exc}") from exc

    with document:
        if document.needs_pass:
            raise ExtractionError(f"PDF {path.name} защищён паролем")
        total = document.page_count
        if total == 0:
            raise ExtractionError(f"PDF {path.name} не содержит страниц")

        selected = _parse_selection(total, pages)
        failed: list[str] = []
        prefix = stem or path.stem
        result: list[PageImage] = []
        # Порог длины задаётся один на комплект, в пунктах эталонного листа:
        # сам детектор пересчитывает его под масштаб каждой страницы. Ноль
        # трактуем как «порог не нужен» и подставляем длину наконечника —
        # иначе отрезок короче отрезка не отсекается и в сумму идёт штрих.
        raw_min = config.DIMSCAN_MIN_LENGTH if dim_min_length is None else dim_min_length
        dim_params = DimParams(min_length=raw_min if raw_min > 0 else 0.5)

        for number in selected:
            page = document.load_page(number - 1)

            # Шаг 1: разбор размеров по вектору, до рендера. Отдельная
            # ошибка здесь не роняет лист: лист с картинкой и текстом
            # полезнее листа, который не отрисовался из-за детектора.
            scan = None
            rejected: tuple[str, ...] = ()
            dim_error: str | None = None
            if want_dims:
                try:
                    scan = detect_page(page, dim_params)
                    rejected = tuple(scan.rejected_numbers)
                except Exception as exc:
                    dim_error = f"{type(exc).__name__}: {exc}"
                    scan = None

            # Шаг 1б: зачистка вырезом из PDF, до рендера. Лист векторный, и
            # удалять элементы правильно на нём: в рендере не остаётся ни
            # серого ореола от сглаживания, ни обрезки буквы, а подписи и
            # рамки знаков исчезают из текстового слоя тоже — модель не видит
            # их даже в списке текста.
            #
            # Раньше зачистка шла после рендера и закрашивала по растру, а
            # обрезка полей — до неё. Теперь порядок обратный, и обрезка
            # оказывается после зачистки: поля, освободившиеся после удаления
            # подписей и штампов, срезаются тоже. Размеры считаются по
            # разбору, сделанному до вырезания: координаты у них в пунктах
            # PDF, и вырезание их не сдвигает.
            redaction_info: dict = {}
            redaction_error: str | None = None
            if scan is not None and want_clean:
                try:
                    report = apply_plan(
                        page, build_plan(page, scan), guard_of(scan.lines)
                    )
                    redaction_info = {
                        "cut": report.total,
                        "spared": len(report.skipped),
                        "by_kind": report.counts(),
                    }
                except Exception as exc:  # зачистка — улучшение, не условие
                    redaction_error = f"{type(exc).__name__}: {exc}"

            matrix = _scale_for(page, dpi, max_side)
            try:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            except Exception as exc:
                raise ExtractionError(
                    f"Не удалось отрендерить страницу {number}: {exc}"
                ) from exc

            # Шаг 2: обрезка пустых полей. Рамка ищется по нарисованным
            # пикселям, поэтому на диск и в модель уходит обрезанный лист.
            trimmed = trim(pixmap) if want_trim else None
            if trimmed is not None and not trimmed.skipped:
                pixmap = trimmed.pixmap
            trim_info = trimmed.as_dict() if trimmed is not None else None
            dx, dy = (trimmed.box[0], trimmed.box[1]) if trim_info else (0, 0)

            # tobytes отдаёт PNG прямо в память — файлов на диске до конца
            # обработки не создаётся.
            data = pixmap.tobytes("png")
            width, height = pixmap.width, pixmap.height

            # Матрица в координатах обрезанной картинки. Её используют и
            # подписи, и размеры: одна матрица на обе части листа, поэтому
            # разойтись им негде.
            to_pixels = shifted_matrix(matrix, dx, dy)

            # Шаг 3: размеры в пиксели этой же картинки. Пересчитываем
            # один раз и после среза, потому что матрица сдвинута на него.
            dims: tuple[DimensionMark, ...] = ()
            dim_info: dict = {}
            if scan is not None:
                dims = marks_in_pixels(
                    scan, lambda b: box_to_pixels(b, to_pixels)
                )
                dim_info = {
                    "dimensions": len(dims),
                    "pointers": sum(1 for d in dims if d.kind != "span"),
                    "rejected_numbers": len(rejected),
                    "scale": scan.stats.get("scale"),
                    "scale_source": scan.stats.get("scale_source"),
                    "width_pt": scan.width_used,
                }

            # Шаг 4: подсветка размеров на той же картинке. Модели больше не
            # нужно читать числа на листе глазами — она выбирает метки, и это
            # убирает целый класс ошибок (24100, прочитанное как 22100).
            overlay_bytes = b""
            overlay_error: str | None = None

            # Шаг 5: текст той же матрицей, что и растр, иначе подписи
            # разъедутся с картинкой, которую видит модель. Если лист
            # обрезан, матрица сдвинута на срез — подписи сразу приходят в
            # координатах обрезанной картинки, двигать их вручную не нужно.
            items: tuple[TextItem, ...] = ()
            text_error: str | None = None
            dropped = 0
            if want_text:
                try:
                    found = extract_text_items(
                        page,
                        to_pixels,
                        min_height_px=config.PDF_TEXT_MIN_HEIGHT_PX,
                        max_items=config.PDF_TEXT_MAX_ITEMS or None,
                    )
                    if trim_info:
                        # Рамка среза задана в координатах полного листа, а
                        # подписи уже сдвинуты в систему обрезанной картинки,
                        # поэтому сверяем их с размерами самой картинки.
                        kept, dropped = inside_box(found, (0, 0, width, height))
                        found = kept
                    items = tuple(found)
                except Exception as exc:
                    # Сбой текстового слоя не должен ронять расчёт, но молчать
                    # о нём тоже нельзя: без подписей и промт, и разметка
                    # молча деградируют. Причина попадает в meta.
                    text_error = f"{type(exc).__name__}: {exc}"

            # Шаг 6: зачистка листа от того, что размером не является. Идёт
            # Зачистка уже применена к странице, поэтому отдельной картинки
            # нет: ``data`` и есть зачищенный лист. Поле ``cleaned`` остаётся
            # пустым, и ``model_image`` отдаёт ``data``.

            # Шаг 7: подсветка размеров поверх зачистки. Иначе подсветка
            # вернула бы на лист ровно то, что зачистка только что убрала, а
            # наконечники и метки P7 лежат поверх того, что осталось.
            if dims and want_overlay:
                try:
                    overlay_bytes = render_overlay(
                        data, width, height, dims
                    )
                except Exception as exc:  # подсветка — улучшение, не условие
                    overlay_error = f"{type(exc).__name__}: {exc}"

            result.append(
                PageImage(
                    page_number=number,
                    data=data,
                    width=width,
                    height=height,
                    label=f"{prefix}_{number:02d}.png",
                    text_items=items,
                    dimensions=dims,
                    rejected_numbers=rejected,
                    cleaned=b"",
                    overlay=overlay_bytes,
                    meta={
                        "dpi": round(72.0 * matrix[0], 1),
                        "page_size_pt": (round(page.rect.width, 1), round(page.rect.height, 1)),
                        "text_items": len(items),
                        "text_items_dropped": dropped,
                        "text_error": text_error,
                        "dimscan": dim_info or None,
                        "dimscan_error": dim_error,
                        "clean": redaction_info or None,
                        "clean_error": redaction_error,
                        "overlay": bool(overlay_bytes),
                        "overlay_error": overlay_error,
                        "trim": trim_info,
                    },
                )
            )
            pixmap = None  # освободить память PyMuPDF
            if text_error:
                failed.append(f"лист {number} — {text_error}")
            if dim_error:
                failed.append(f"лист {number} — разбор размеров: {dim_error}")
            if dropped:
                # Подписи вне среза — их видел текстовый слой, но не увидела
                # картинка. Молча терять их нельзя: это деградация промта.
                failed.append(
                    f"лист {number} — {dropped} подписей вне обрезки"
                )

        if failed:
            # Не прерываем прогон, но и не делаем вид, что подписей нет:
            # иначе деградацию промта и разметки заметят только по итогу.
            print(
                f"Внимание: замечания при извлечении ({len(failed)} шт.): "
                + "; ".join(failed[:5]),
                file=sys.stderr,
            )

    return ExtractionResult(pages=result, source_pdf=path, page_count=total)
