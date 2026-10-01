"""Иерархия ошибок приложения."""

from __future__ import annotations


class IsokrusError(Exception):
    """Базовая ошибка приложения."""


class ConfigError(IsokrusError):
    """Некорректная или неполная конфигурация."""


class ExtractionError(IsokrusError):
    """Не удалось открыть PDF или отрендерить страницу."""


class LLMError(IsokrusError):
    """LLM не вернула валидный ответ (или API недоступна)."""


class ExperimentError(IsokrusError):
    """Проблема с папкой эксперимента в research/."""


class SchemaError(ExperimentError):
    """JSON-схема эксперимента не соответствует контракту пайплайна."""
