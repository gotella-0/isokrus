"""Конвейер: извлечение страниц -> параллельный анализ -> артефакты.

Порядок извлечения (см. ``extract.py``): PDF -> разбор размеров по вектору
-> рендер -> обрезка полей -> подписи. Разбор размеров идёт первым по
данным, а не по скорости: он работает с вектором и от обрезки не зависит,
но показывает модере, **что на листе является размером**, ещё до того как
появится картинка.

Параллелизм ограничен ``config.MAX_PARALLEL`` (по умолчанию 10 одновременных
вызовов LLM). Оригинальные байты изображений остаются в памяти до конца
обработки и сохраняются в ``output/<эксперимент>/pages/`` оттуда же.

Картинки, которые видит модель, — уже обрезанные по пустым полям (см.
``trim.py``): и в base64, и на диск, и в разметке это одна и та же версия.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from . import config
from .errors import IsokrusError
from .experiments import Experiment
from .extract import ExtractionResult, PageImage, extract_pages
from .llm import Usage, call_structured

ProgressHook = Callable[[int, int, PageImage], None]


@dataclass
class PageResult:
    """Итог по одной странице."""

    page: PageImage
    response: dict | None = None
    error: str | None = None
    elapsed: float = 0.0
    usage: Usage = field(default_factory=Usage)
    system_prompt: str | None = None
    user_prompt: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.response is not None


@dataclass
class RunResult:
    """Итог по всему прогону."""

    experiment: Experiment
    pages: list[PageResult]
    output_dir: Path
    started_at: datetime
    finished_at: datetime
    usage: Usage
    pdf_path: Path

    @property
    def duration(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def succeeded(self) -> list[PageResult]:
        return [p for p in self.pages if p.ok]

    @property
    def failed(self) -> list[PageResult]:
        return [p for p in self.pages if not p.ok]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def default_output_dir(experiment: Experiment, when: datetime | None = None) -> Path:
    """Каталог прогона по умолчанию: ``output/<эксперимент>_<дата>_<время>``.

    Метка времени добавляется к имени эксперимента, чтобы каждый прогон
    попадал в свой каталог и результаты не перезаписывали друг друга.
    """
    stamp = (when or _now()).strftime("%Y%m%d_%H%M%S")
    return config.OUTPUT_DIR / f"{experiment.name}_{stamp}"


def _default_user_instruction(experiment: Experiment, page: PageImage) -> str:
    """Инструкция пользователя, если в папке эксперимента нет ``user.md``."""
    base = experiment.user or (
        "Проанализируй этот лист изометрического чертежа трубопровода "
        "и верни результат строго по схеме."
    )
    return (
        f"{base}\n\nЛист № {page.page_number}. "
        f"Размер изображения: {page.width}x{page.height} px."
    )


def analyze_page(
    page: PageImage,
    experiment: Experiment,
    usage: Usage | None = None,
) -> PageResult:
    """Обработать одну страницу: собрать контент, вызвать LLM, разобрать ответ."""
    stats = usage or Usage()
    instruction = _default_user_instruction(experiment, page)
    # Промт с подставленным текстовым слоем листа: координаты подписей оттуда
    # же используются разметкой, поэтому модель и разметка не могут разойтись.
    system = experiment.render_system(page)
    started = time.perf_counter()

    try:
        # Оригинальные байты -> base64 ровно один раз, изображение остаётся в памяти.
        result = call_structured(
            system=system,
            user=instruction,
            response_model=experiment.response_model,
            model=experiment.model,
            reasoning_effort=experiment.reasoning_effort,
            image_data_urls=[page.data_url()],
            usage=stats,
        )
    except IsokrusError as exc:
        return PageResult(
            page=page,
            error=f"{type(exc).__name__}: {exc}",
            elapsed=time.perf_counter() - started,
            usage=stats,
            system_prompt=system,
            user_prompt=instruction,
        )
    except Exception as exc:  # неожиданная ошибка не должна ронять весь прогон
        return PageResult(
            page=page,
            error=f"{type(exc).__name__}: {exc}",
            elapsed=time.perf_counter() - started,
            usage=stats,
            system_prompt=system,
            user_prompt=instruction,
        )

    return PageResult(
        page=page,
        response=result.model_dump(mode="json"),
        elapsed=time.perf_counter() - started,
        usage=stats,
        system_prompt=system,
        user_prompt=instruction,
    )


def analyze_pages(
    pages: Sequence[PageImage],
    experiment: Experiment,
    max_parallel: int | None = None,
    progress: ProgressHook | None = None,
) -> tuple[list[PageResult], Usage]:
    """Обработать страницы параллельно, сохранив исходный порядок в выдаче."""
    workers = max(1, min(max_parallel or config.MAX_PARALLEL, len(pages) or 1))
    total_usage = Usage()
    results: list[PageResult] = []
    lock = threading.Lock()
    done = 0

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="isokrus") as pool:
        futures = {
            pool.submit(analyze_page, page, experiment, total_usage): page for page in pages
        }
        for future in as_completed(futures):
            page = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # analyze_page уже ловит всё, но подстрахуемся
                result = PageResult(page=page, error=f"{type(exc).__name__}: {exc}")
            with lock:
                results.append(result)
                done += 1
            if progress:
                progress(done, len(pages), page)

    results.sort(key=lambda r: r.page.page_number)
    return results, total_usage


def run(
    pdf_path: str | Path,
    experiment: Experiment,
    output_dir: str | Path | None = None,
    pages: Sequence[str] | None = None,
    dpi: int | None = None,
    max_parallel: int | None = None,
    save_page_images: bool | None = None,
    progress: ProgressHook | None = None,
    trim_margins: bool | None = None,
    with_dimensions: bool | None = None,
    with_overlay: bool | None = None,
    with_labels: bool | None = None,
    with_clean: bool | None = None,
) -> RunResult:
    """Полный цикл: извлечь страницы в память, обработать параллельно, сохранить.

    Порядок извлечения задан в :mod:`.extract` и не переставляется здесь:
    PDF -> разбор размеров по вектору -> вырезание шума -> рендер -> обрезка ->
    подписи -> метки или подсветка размеров.

    ``with_overlay`` и ``with_labels`` раньше сюда не доходили: флаги читались в
    ``cli``, но в :func:`extract_pages` не передавались, и подсветка молча не
    включалась ни в одном прогоне. Теперь оба доходят.
    """
    started = _now()

    extraction: ExtractionResult = extract_pages(
        pdf_path,
        dpi=dpi,
        pages=pages,
        trim_margins=trim_margins,
        with_dimensions=with_dimensions,
        with_overlay=with_overlay,
        with_labels=with_labels,
        with_clean=with_clean,
    )
    if not extraction.pages:
        raise IsokrusError("Не удалось извлечь ни одной страницы")

    out_root = Path(output_dir) if output_dir else default_output_dir(experiment, started)
    out_root.mkdir(parents=True, exist_ok=True)

    results, usage = analyze_pages(
        extraction.pages, experiment, max_parallel=max_parallel, progress=progress
    )

    # Изображения сохраняются после обработки, из тех же объектов в памяти.
    should_save = config.SAVE_PAGE_IMAGES if save_page_images is None else save_page_images
    if should_save:
        for result in results:
            stem = result.page.label.rsplit(".", 1)[0]
            result.page.save(out_root / "pages", stem=stem)
            # Ровно то, что ушло в модель: с подсветкой, с зачисткой и в
            # обрезанных полях. Кладётся всегда, независимо от ``--no-images``:
            # флаг экономит место на исходниках, а этот снимок — единственный
            # способ проверить ответ модели по картинке.
            result.page.save_model_image(out_root / "to_llm_imgs", stem=stem)

    finished = _now()
    return RunResult(
        experiment=experiment,
        pages=results,
        output_dir=out_root,
        started_at=started,
        finished_at=finished,
        usage=usage,
        pdf_path=extraction.source_pdf,
    )


def write_json_report(run_result: RunResult, path: Path | None = None) -> Path:
    """Полный JSON-отчёт: промт и сырые ответы LLM по каждому листу + статистика."""
    target = path or run_result.output_dir / "responses.json"
    payload = {
        "experiment": {
            "name": run_result.experiment.name,
            "path": str(run_result.experiment.path),
            "model": run_result.experiment.model or config.LLM_MODEL,
            "reasoning_effort": run_result.experiment.reasoning_effort
            or config.LLM_REASONING_EFFORT,
            "files": run_result.experiment.files,
        },
        "source_pdf": str(run_result.pdf_path),
        "started_at": run_result.started_at.isoformat(),
        "finished_at": run_result.finished_at.isoformat(),
        "duration_seconds": round(run_result.duration, 2),
        "pages_total": len(run_result.pages),
        "pages_ok": len(run_result.succeeded),
        "pages_failed": len(run_result.failed),
        "usage": run_result.usage.as_dict(),
        "pages": [
            {
                "page_number": r.page.page_number,
                "image": r.page.name,
                "image_size": [r.page.width, r.page.height],
                "image_bytes": r.page.size_bytes,
                "image_area_ratio": round(r.page.area_ratio, 4),
                "crop_box": list(r.page.crop_box) if r.page.crop_box else None,
                "text_items": len(r.page.text_items),
                "dimensions": len(r.page.dimensions),
                "dimension_pointers": sum(
                    1 for d in r.page.dimensions if d.kind != "span"
                ),
                "rejected_numbers": len(r.page.rejected_numbers),
                "dimscan": r.page.meta.get("dimscan"),
                "dimscan_error": r.page.meta.get("dimscan_error"),
                "elapsed_seconds": round(r.elapsed, 2),
                "error": r.error,
                "system_prompt": r.system_prompt,
                "user_prompt": r.user_prompt,
                "response": r.response,
            }
            for r in run_result.pages
        ],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target
