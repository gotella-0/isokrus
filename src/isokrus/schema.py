"""Конвертация JSON Schema в Pydantic-модель.

Эксперименты в ``research/`` описывают ответ простым файлом ``response.json``.
Чтобы не заставлять писать Python-код под каждый эксперимент, схема
конвертируется в ``pydantic.BaseModel`` на лету. Тот же файл уходит в
``response_format`` как ``json_schema`` — то есть контракт ровно один.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, create_model

from .errors import SchemaError

_SCALARS: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


class _NameRegistry:
    """Реестр уникальных имён моделей в рамках одной схемы."""

    def __init__(self) -> None:
        self._used: set[str] = set()

    def unique(self, base: str) -> str:
        name = re.sub(r"[^0-9a-zA-Z_]", "_", base)
        if not name or not name[0].isalpha():
            name = f"Model_{name}"
        if name[0].islower():
            name = name[0].upper() + name[1:]
        candidate, counter = name, 1
        while candidate in self._used:
            counter += 1
            candidate = f"{name}{counter}"
        self._used.add(candidate)
        return candidate


def _title_of(schema: dict, fallback: str) -> str:
    title = schema.get("title")
    return str(title) if title else fallback


def _description_of(schema: dict) -> str:
    return str(schema.get("description", "")) or ""


def _literal_from_enum(values: list[Any]) -> Any:
    """Enum -> typing.Literal, если все значения примитивы."""
    if values and all(isinstance(v, (str, int, bool)) and v is not None for v in values):
        if len(set(values)) != len(values):
            raise SchemaError("Enum содержит дубликаты значений")
        return Literal[tuple(values)]  # type: ignore[valid-type]
    return Any


class _SchemaCompiler:
    """Рекурсивная компиляция JSON Schema в pydantic-типы."""

    def __init__(self, root: dict) -> None:
        self.root = root
        self.registry = _NameRegistry()
        # $defs/definitions, объявленные в корневой схеме.
        self.definitions: dict[str, dict] = {
            **root.get("$defs", {}),
            **root.get("definitions", {}),
        }
        self._models: dict[str, type[BaseModel]] = {}

    def compile(self, schema: dict, name: str) -> Any:
        if not isinstance(schema, dict):
            raise SchemaError(f"Узел схемы должен быть объектом, получено {type(schema)!r}")

        # --- $ref -------------------------------------------------------
        if "$ref" in schema:
            return self._compile_ref(schema["$ref"], name)

        # --- anyOf / oneOf ----------------------------------------------
        for key in ("anyOf", "oneOf"):
            if key in schema:
                return self._compile_union(schema[key], schema, name)

        # --- enum --------------------------------------------------------
        if "enum" in schema:
            return _literal_from_enum(schema["enum"])

        # --- константа ---------------------------------------------------
        if "const" in schema:
            return Literal[schema["const"]]  # type: ignore[valid-type]

        raw_type = schema.get("type")

        # type может быть списком: ["string", "null"] -> Optional[str]
        if isinstance(raw_type, list):
            # description и title описывают конструкцию целиком, а не каждый
            # её вариант, поэтому внутрь вариантов они не копируются.
            body = {k: v for k, v in schema.items() if k not in {"type", "description", "title"}}
            variants = [self.compile({"type": t, **body}, name) for t in raw_type]
            return Union[tuple(variants)]  # type: ignore[valid-type]

        if raw_type == "object" or (raw_type is None and "properties" in schema):
            return self._compile_object(schema, name)

        if raw_type == "array":
            items = schema.get("items")
            if not isinstance(items, dict):
                raise SchemaError("Массив должен объявлять items")
            inner = self.compile(items, f"{name}_item")
            return list[inner]  # type: ignore[valid-type]

        if raw_type in _SCALARS:
            scalar = _SCALARS[raw_type]
            # Минимальные ограничения int на целые поля (номера листов и т.п.).
            if raw_type == "integer" and ("minimum" in schema or "maximum" in schema):
                return int
            return scalar

        if raw_type is None and not schema:
            return Any

        raise SchemaError(f"Неподдерживаемый тип в схеме: {raw_type!r} (у блока {name!r})")

    def _compile_ref(self, ref: str, name: str) -> Any:
        if not ref.startswith("#/"):
            raise SchemaError(f"Внешние $ref не поддерживаются: {ref!r}")
        parts = ref.lstrip("#/").split("/")
        if parts[0] not in ("$defs", "definitions"):
            raise SchemaError(f"Ссылка должна указывать на $defs: {ref!r}")
        def_name = parts[-1]
        if def_name in self._models:
            return self._models[def_name]
        if def_name not in self.definitions:
            raise SchemaError(f"Ссылка $ref на несуществующее определение: {ref!r}")
        return self._compile_definition(def_name)

    def _compile_definition(self, def_name: str) -> type[BaseModel]:
        if def_name in self._models:
            return self._models[def_name]

        # Рекурсивные ссылки: сначала резервируем имя, потом наполняем поля.
        schema = self.definitions[def_name]
        is_object = schema.get("type") == "object" or "properties" in schema
        if not is_object:
            return self.compile(schema, def_name)  # type: ignore[return-value]

        placeholder = create_model(
            self.registry.unique(_title_of(schema, def_name)), __base__=BaseModel
        )
        self._models[def_name] = placeholder
        self._build_model(placeholder.__name__, schema, existing=placeholder, replace=True)
        return placeholder

    def _compile_union(self, variants: list[dict], parent: dict, name: str) -> Any:
        if not variants:
            raise SchemaError(f"Пустой anyOf/oneOf в блоке {name!r}")

        compiled = [self.compile(v, name) for v in variants]
        # anyOf: [X, null] -> Optional[X]
        if len(compiled) == 2 and type(None) in compiled:
            inner = next(c for c in compiled if c is not type(None))
            return Union[inner, None]  # type: ignore[valid-type]
        if len(set(map(id, compiled))) == 1:
            return compiled[0]
        return Union[tuple(compiled)]  # type: ignore[valid-type]

    def _compile_object(self, schema: dict, name: str) -> type[BaseModel]:
        model_name = self.registry.unique(_title_of(schema, name))
        model = create_model(model_name, __base__=BaseModel)
        return self._build_model(model_name, schema, existing=model)

    def _build_model(
        self, model_name: str, schema: dict, existing: type[BaseModel] | None = None, replace: bool = False
    ) -> type[BaseModel]:
        properties: dict[str, Any] = schema.get("properties", {})
        required: set[str] = set(schema.get("required", []))
        additional = schema.get("additionalProperties", False)

        fields: dict[str, tuple[Any, Any]] = {}
        for prop_name, prop_schema in properties.items():
            if not isinstance(prop_schema, dict):
                raise SchemaError(f"Поле {prop_name!r}: схема должна быть объектом")
            ann = self.compile(prop_schema, f"{model_name}_{prop_name}")
            description = _description_of(prop_schema)
            if prop_name in required:
                # Обязательное поле: значение по умолчанию только у nullable-типов.
                default = None if _is_optional(ann) else ...
                fields[prop_name] = (
                    ann,
                    Field(default, description=description or None),
                )
            else:
                fields[prop_name] = (
                    Union[ann, None],  # type: ignore[valid-type]
                    Field(None, description=description or None),
                )

        config = ConfigDict(
            extra="allow" if additional is True else "forbid",
            populate_by_name=True,
        )
        model = create_model(
            model_name,
            __base__=BaseModel,
            __config__=config,
            **fields,
        )

        if existing is not None and replace:
            # Наполняем зарезервированную модель: рекурсивные $ref уже смотрят
            # именно на неё, поэтому переносим аннотации и поля, а затем
            # пересобираем core-схему.
            existing.__annotations__ = dict(model.__annotations__)
            existing.__pydantic_fields__ = dict(model.__pydantic_fields__)
            existing.__pydantic_decorators__ = model.__pydantic_decorators__
            existing.model_config.update(config)
            existing.model_rebuild(force=True)
            return existing
        return model


def _is_optional(annotation: Any) -> bool:
    if annotation is Any or annotation is type(None):
        return True
    if get_origin(annotation) is Union:
        return type(None) in get_args(annotation)
    # list[X] / dict — не nullable по умолчанию.
    return False


def schema_to_model(schema: dict, model_name: str = "Response") -> type[BaseModel]:
    """Собрать pydantic-модель верхнего уровня из JSON Schema."""
    if not isinstance(schema, dict):
        raise SchemaError("Схема должна быть JSON-объектом")
    schema = _inline_defs(schema)
    compiler = _SchemaCompiler(schema)
    if schema.get("type") == "object" or "properties" in schema:
        model = compiler._compile_object(schema, model_name)
        model.model_rebuild(force=True)
        return model
    # Не-объектная корневая схема: оборачиваем в объект с единственным полем.
    inner = compiler.compile(schema, model_name)
    return create_model(
        compiler.registry.unique(model_name),
        __config__=ConfigDict(extra="forbid"),
        value=(inner, Field(..., description="Значение ответа")),
    )


def _inline_defs(schema: dict) -> dict:
    """Копия схемы, где $defs гарантированно присутствует (для strict-режима)."""
    result = dict(schema)
    defs = {**schema.get("definitions", {}), **schema.get("$defs", {})}
    if defs:
        result["$defs"] = defs
        result.pop("definitions", None)
    return result


def load_schema_file(path: str | Path) -> dict:
    """Прочитать и минимально проверить JSON-схему эксперимента.

    Принимаются оба формата: голая схема ответа (``type``/``properties`` в
    корне) и обёртка ``response_format`` от OpenAI
    (``{"name": ..., "strict": true, "schema": {...}}``) — второй просто
    распаковывается, ``strict`` и ``name`` и так задаёт сам конвейер.
    """
    schema_path = Path(path)
    if not schema_path.exists():
        raise SchemaError(f"Файл схемы не найден: {schema_path}")
    try:
        data = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{schema_path.name}: некорректный JSON — {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaError(f"{schema_path.name}: ожидался JSON-объект")

    if "type" not in data and "properties" not in data:
        inner = data.get("schema")
        if isinstance(inner, dict):
            return inner
    return data


def model_to_strict_schema(model: type[BaseModel]) -> dict:
    """JSON Schema модели в виде, пригодном для strict structured outputs.

    OpenAI требует, чтобы в каждом объекте был указан весь набор полей в
    ``required`` и ``additionalProperties: false``. Опциональные поля мы
    оставляем nullable, поэтому объявление их обязательными не ломает ответ.
    """
    schema = model.model_json_schema()

    def harden(node: Any) -> Any:
        if isinstance(node, dict):
            node = {k: harden(v) for k, v in node.items()}
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                if isinstance(node.get("properties"), dict):
                    node["required"] = list(node["properties"].keys())
            return node
        if isinstance(node, list):
            return [harden(item) for item in node]
        return node

    return harden(schema)
