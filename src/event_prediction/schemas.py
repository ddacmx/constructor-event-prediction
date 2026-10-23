"""Small dependency-free validator for the JSON Schema subset used by this project."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .domain import ArtifactError

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "configs" / "schemas"
SCHEMA_FILES = {
    "dataset-record.v1": "dataset-record.v1.schema.json",
    "split-manifest.v1": "split-manifest.v1.schema.json",
    "model-manifest.v2": "model-manifest.v2.schema.json",
    "seed-result.v2": "seed-result.v2.schema.json",
    "aggregate-result.v2": "aggregate-result.v2.schema.json",
    "experiment-manifest.v2": "experiment-manifest.v2.schema.json",
}


def load_schema(name: str) -> dict[str, Any]:
    try:
        value = json.loads((SCHEMA_DIR / SCHEMA_FILES[name]).read_text(encoding="utf-8"))
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot load JSON Schema {name}: {exc}") from exc
    if not isinstance(value, dict) or value.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ArtifactError(f"invalid JSON Schema document: {name}")
    return value


def validate_schema(name: str, value: Any) -> None:
    schema = load_schema(name)
    _validate(schema, value, "$", schema)


def _validate(schema: dict[str, Any], value: Any, path: str, root: dict[str, Any]) -> None:
    if "$ref" in schema:
        reference = schema["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise ArtifactError(f"schema {path}: unsupported reference {reference!r}")
        target: Any = root
        try:
            for part in reference[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
        except (KeyError, TypeError) as exc:
            raise ArtifactError(f"schema {path}: unresolved reference {reference!r}") from exc
        if not isinstance(target, dict):
            raise ArtifactError(f"schema {path}: reference does not identify a schema")
        _validate(target, value, path, root)
        return
    expected = schema.get("type")
    matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": type(value) is bool,
        "null": value is None,
    }
    if isinstance(expected, list):
        if not any(item in matches and matches[item] for item in expected):
            raise ArtifactError(f"schema {path}: expected one of {expected}")
    elif expected in matches and not matches[expected]:
        raise ArtifactError(f"schema {path}: expected {expected}")
    if expected == "number" and not math.isfinite(float(value)):
        raise ArtifactError(f"schema {path}: number must be finite")
    if "const" in schema and value != schema["const"]:
        raise ArtifactError(f"schema {path}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ArtifactError(f"schema {path}: value is not in enum")
    if isinstance(value, dict):
        required = set(schema.get("required", []))
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        if len(value) < schema.get("minProperties", 0):
            raise ArtifactError(f"schema {path}: too few properties")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            raise ArtifactError(f"schema {path}: too many properties")
        if additional is False and set(value) - set(properties):
            raise ArtifactError(f"schema {path}: unexpected fields {sorted(set(value) - set(properties))}")
        # Report local scalar violations before cascading missing fields in nested
        # objects; this keeps diagnostics field-specific for partially malformed artifacts.
        ordered_items = sorted(value.items(), key=lambda pair: isinstance(pair[1], (dict, list)))
        for key, item in ordered_items:
            child = properties.get(key, additional if isinstance(additional, dict) else None)
            if isinstance(child, dict):
                _validate(child, item, f"{path}.{key}", root)
        missing = sorted(required - set(value))
        if missing:
            raise ArtifactError(f"schema {path}: missing required fields {missing}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ArtifactError(f"schema {path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ArtifactError(f"schema {path}: too many items")
        if schema.get("uniqueItems") and len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
            raise ArtifactError(f"schema {path}: items must be unique")
        if isinstance(schema.get("items"), dict):
            for index, item in enumerate(value):
                _validate(schema["items"], item, f"{path}[{index}]", root)
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ArtifactError(f"schema {path}: string is too short")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ArtifactError(f"schema {path}: string does not match pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value < schema.get("minimum", value):
        raise ArtifactError(f"schema {path}: value is below minimum")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > schema.get("maximum", value):
        raise ArtifactError(f"schema {path}: value is above maximum")
