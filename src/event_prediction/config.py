"""Versioned, fully effective configuration loading and deterministic hashing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .domain import ConfigurationError


# The effective simulator definition is serialized by ``configs/default.json`` and
# participates in GeneratorConfig.digest(); these constants only provide the same
# defaults for library callers that do not pass a file.
DEFAULT_EVENT_CATALOG = {
    "api_interface_modified": "An API interface was modified.",
    "dependency_upgraded": "A software dependency was upgraded.",
    "compatibility_checked": "Compatibility with dependent components was checked.",
    "ci_failed": "Continuous integration reported a failure.",
    "bug_fixed": "A defect discovered during delivery was fixed.",
    "integration_test_passed": "Integration tests completed successfully.",
    "review_approved": "A reviewer approved the change.",
    "deployed": "The change was deployed.",
    "rolled_back": "A deployment was reverted.",
}
DEFAULT_TRANSITIONS = {
    "api_interface_modified": {"compatibility_checked": .36, "ci_failed": .18, "dependency_upgraded": .16, "integration_test_passed": .15, "api_interface_modified": .15},
    "dependency_upgraded": {"compatibility_checked": .38, "ci_failed": .20, "integration_test_passed": .16, "dependency_upgraded": .14, "api_interface_modified": .12},
    "compatibility_checked": {"integration_test_passed": .34, "ci_failed": .24, "bug_fixed": .14, "review_approved": .16, "compatibility_checked": .12},
    "ci_failed": {"bug_fixed": .44, "compatibility_checked": .18, "dependency_upgraded": .15, "ci_failed": .11, "api_interface_modified": .12},
    "bug_fixed": {"compatibility_checked": .29, "integration_test_passed": .27, "ci_failed": .19, "review_approved": .10, "bug_fixed": .15},
    "integration_test_passed": {"review_approved": .39, "ci_failed": .16, "compatibility_checked": .15, "bug_fixed": .10, "deployed": .20},
    "review_approved": {"deployed": .49, "integration_test_passed": .19, "ci_failed": .11, "compatibility_checked": .10, "api_interface_modified": .11},
    "deployed": {"__end__": .40, "rolled_back": .24, "ci_failed": .12, "bug_fixed": .10, "api_interface_modified": .14},
    "rolled_back": {"__end__": .24, "bug_fixed": .31, "api_interface_modified": .20, "dependency_upgraded": .15, "compatibility_checked": .10},
}
DEFAULT_HISTORY_MODIFIERS = {
    "fallback_weights": {"bug_fixed": .08, "review_approved": .08, "compatibility_checked": .08, "deployed": .05, "rolled_back": .08},
    "ci_failed_bug_fixed_multiplier": 2.2,
    "ci_failed_review_approved_multiplier": .45,
    "unresolved_compatibility_checked_multiplier": 1.8,
    "unresolved_deployed_multiplier": .35,
    "successful_test_review_approved_multiplier": 2.0,
    "review_ready_deployed_multiplier": 2.2,
    "deployed_rollback_with_failures_multiplier": 1.8,
    "deployed_rollback_without_failures_multiplier": .7,
}


def _copy_nested(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


@dataclass(frozen=True)
class GeneratorConfig:
    schema_version: str = "generator-config.v2"
    min_length: int = 5
    max_length: int = 14
    history_window: int = 4
    recency_cap: int = 10
    laplace_alpha: float = 1.0
    regularization_candidates: tuple[float, ...] = (0.1, 1.0, 10.0)
    bootstrap_replicates: int = 200
    final_seeds: tuple[int, ...] = (20261021, 20261022, 20261023, 20261024, 20261025)
    event_catalog: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EVENT_CATALOG))
    initial_weights: dict[str, float] = field(default_factory=lambda: {"api_interface_modified": .58, "dependency_upgraded": .42})
    transition_weights: dict[str, dict[str, float]] = field(default_factory=lambda: _copy_nested(DEFAULT_TRANSITIONS))
    history_modifiers: dict[str, Any] = field(default_factory=lambda: _copy_nested(DEFAULT_HISTORY_MODIFIERS))
    termination_policy: dict[str, Any] = field(default_factory=lambda: {"end_token": "__end__", "rollback_is_terminal": True})

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != "generator-config.v2":
            errors.append("schema_version must be 'generator-config.v2'")
        integer_fields = {
            "min_length": self.min_length, "max_length": self.max_length,
            "history_window": self.history_window, "recency_cap": self.recency_cap,
            "bootstrap_replicates": self.bootstrap_replicates,
        }
        for name, value in integer_fields.items():
            if type(value) is not int:
                errors.append(f"{name} must be an integer (bool is not accepted)")
        if type(self.min_length) is int and self.min_length < 2:
            errors.append("min_length must be at least 2")
        # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-validate-config:p1:inst-sdg-check-termination
        if type(self.max_length) is int and type(self.min_length) is int and self.max_length <= self.min_length:
            errors.append("max_length must be greater than min_length")
        # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-validate-config:p1:inst-sdg-check-termination
        if type(self.history_window) is int and self.history_window < 1:
            errors.append("history_window must be positive")
        if type(self.recency_cap) is int and self.recency_cap < 1:
            errors.append("recency_cap must be positive")
        if isinstance(self.laplace_alpha, bool) or not isinstance(self.laplace_alpha, (int, float)) or not math.isfinite(self.laplace_alpha) or self.laplace_alpha <= 0:
            errors.append("laplace_alpha must be a finite number greater than zero")
        if not isinstance(self.regularization_candidates, tuple) or not self.regularization_candidates:
            errors.append("regularization_candidates must be a non-empty array")
        elif any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in self.regularization_candidates):
            errors.append("regularization_candidates elements must be finite numbers greater than zero (bool is not accepted)")
        if type(self.bootstrap_replicates) is int and self.bootstrap_replicates < 20:
            errors.append("bootstrap_replicates must be at least 20")
        if not isinstance(self.final_seeds, tuple) or len(self.final_seeds) != 5 or any(type(value) is not int for value in self.final_seeds) or len(set(self.final_seeds)) != 5:
            errors.append("final_seeds must contain exactly five distinct integers")
        if self.event_catalog != DEFAULT_EVENT_CATALOG:
            errors.append("event_catalog must match the approved CURRENT catalog")
        if not isinstance(self.initial_weights, dict) or set(self.initial_weights) != {"api_interface_modified", "dependency_upgraded"}:
            errors.append("initial_weights must define the two approved opening events")
        _validate_weight_map("initial_weights", self.initial_weights, errors)
        if not isinstance(self.transition_weights, dict) or set(self.transition_weights) != set(DEFAULT_EVENT_CATALOG):
            errors.append("transition_weights must define every catalog event")
        for predecessor, values in self.transition_weights.items() if isinstance(self.transition_weights, dict) else ():
            if not isinstance(values, dict):
                errors.append(f"transition_weights.{predecessor} must be an object")
                continue
            unknown = set(values) - set(DEFAULT_EVENT_CATALOG) - {"__end__"}
            if unknown:
                errors.append(f"transition_weights.{predecessor} has unknown events: {sorted(unknown)}")
            _validate_weight_map(f"transition_weights.{predecessor}", values, errors)
        if not isinstance(self.history_modifiers, dict) or set(self.history_modifiers) != set(DEFAULT_HISTORY_MODIFIERS):
            errors.append("history_modifiers fields must match the approved policy")
        else:
            fallbacks = self.history_modifiers.get("fallback_weights")
            expected_fallbacks = {
                "bug_fixed", "review_approved", "compatibility_checked", "deployed", "rolled_back"
            }
            if not isinstance(fallbacks, dict):
                errors.append("history_modifiers.fallback_weights must be an object")
            else:
                missing = sorted(expected_fallbacks - set(fallbacks))
                extra = sorted(set(fallbacks) - expected_fallbacks)
                if missing:
                    errors.append(f"history_modifiers.fallback_weights missing keys: {missing}")
                if extra:
                    errors.append(f"history_modifiers.fallback_weights has unexpected keys: {extra}")
                for key in sorted(expected_fallbacks & set(fallbacks)):
                    value = fallbacks[key]
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or value <= 0):
                        errors.append(
                            f"history_modifiers.fallback_weights.{key} must be a finite number greater than zero"
                        )
            for name, value in self.history_modifiers.items():
                if name != "fallback_weights" and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                    errors.append(f"history_modifiers.{name} must be a finite number greater than zero")
        if self.termination_policy != {"end_token": "__end__", "rollback_is_terminal": True}:
            errors.append("termination_policy must preserve the approved end token and terminal rollback")
        if errors:
            raise ConfigurationError("invalid configuration: " + "; ".join(errors))

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["regularization_candidates"] = list(self.regularization_candidates)
        value["final_seeds"] = list(self.final_seeds)
        return value

    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_weight_map(name: str, values: Any, errors: list[str]) -> None:
    if not isinstance(values, dict) or not values:
        errors.append(f"{name} must be a non-empty object")
        return
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values.values()
    ):
        errors.append(f"{name} weights must be finite non-negative numbers")
    if not any(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        for value in values.values()
    ):
        errors.append(f"{name} must have a positive weight")


# @cpt-algo:cpt-event-prediction-algo-synthetic-data-generation-validate-config:p1
def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> GeneratorConfig:
    raw: dict[str, Any] = {}
    if path is not None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"cannot read configuration {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigurationError("configuration root must be a JSON object")
    if overrides:
        raw.update({key: value for key, value in overrides.items() if value is not None})
    allowed = set(GeneratorConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown configuration fields: {', '.join(unknown)}")
    for name in ("regularization_candidates", "final_seeds"):
        if name in raw and isinstance(raw[name], list):
            raw[name] = tuple(raw[name])
    try:
        config = GeneratorConfig(**raw)
    except TypeError as exc:
        raise ConfigurationError(f"invalid configuration values: {exc}") from exc
    config.validate()
    # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-validate-config:p1:inst-sdg-config-result
    result = config
    # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-validate-config:p1:inst-sdg-config-result
    return result
