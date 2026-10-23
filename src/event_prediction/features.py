"""Leakage-safe observable prefix feature extraction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .catalog import EVENT_TYPES, validate_event_code
from .domain import ConfigurationError, LeakageError

ALLOWED_PREFIX_KEYS = frozenset({"prefix"})
RECENCY_EVENTS = (
    "ci_failed",
    "bug_fixed",
    "integration_test_passed",
    "review_approved",
    "deployed",
)


def _coerce_prefix(value: Sequence[str] | Mapping[str, Any]) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        extra = set(value) - ALLOWED_PREFIX_KEYS
        if extra:
            raise LeakageError(f"feature input contains forbidden fields: {', '.join(sorted(extra))}")
        if "prefix" not in value:
            raise LeakageError("feature input must contain prefix")
        value = value["prefix"]
    if isinstance(value, (str, bytes)):
        raise ConfigurationError("prefix must be a sequence of event codes, not text")
    prefix = tuple(value)
    if not prefix:
        raise ConfigurationError("prefix must be non-empty")
    for event in prefix:
        validate_event_code(event)
    return prefix


# @cpt-algo:cpt-event-prediction-algo-ml-prediction-extract-features:p1
# @cpt-dod:cpt-event-prediction-dod-ml-prediction-feature-pipeline:p1
def extract_features(value: Sequence[str] | Mapping[str, Any], recency_cap: int = 10) -> dict[str, float | str]:
    # @cpt-begin:cpt-event-prediction-algo-ml-prediction-extract-features:p1:inst-ml-feature-allowlist
    prefix = _coerce_prefix(value)
    # @cpt-end:cpt-event-prediction-algo-ml-prediction-extract-features:p1:inst-ml-feature-allowlist
    if recency_cap < 1:
        raise ConfigurationError("recency cap must be positive")
    features: dict[str, float | str] = {}
    # @cpt-begin:cpt-event-prediction-algo-ml-prediction-extract-features:p1:inst-ml-feature-lags
    for lag in range(1, 4):
        features[f"lag_{lag}"] = prefix[-lag] if len(prefix) >= lag else "<START>"
    # @cpt-end:cpt-event-prediction-algo-ml-prediction-extract-features:p1:inst-ml-feature-lags
    counts = Counter(prefix)
    length = len(prefix)
    # @cpt-begin:cpt-event-prediction-algo-ml-prediction-extract-features:p1:inst-ml-feature-counts
    for event in EVENT_TYPES:
        features[f"count::{event}"] = float(counts[event])
        features[f"proportion::{event}"] = counts[event] / length
    # @cpt-end:cpt-event-prediction-algo-ml-prediction-extract-features:p1:inst-ml-feature-counts
    # @cpt-begin:cpt-event-prediction-algo-ml-prediction-extract-features:p1:inst-ml-feature-transitions
    for previous, current in zip(prefix, prefix[1:]):
        key = f"transition::{previous}->{current}"
        features[key] = float(features.get(key, 0.0)) + 1.0
    # @cpt-end:cpt-event-prediction-algo-ml-prediction-extract-features:p1:inst-ml-feature-transitions
    # @cpt-begin:cpt-event-prediction-algo-ml-prediction-extract-features:p1:inst-ml-feature-context
    features["prefix_length"] = float(length)
    features["ci_failure_count"] = float(counts["ci_failed"])
    features["fix_after_failure"] = float(
        any(a == "ci_failed" and b == "bug_fixed" for a, b in zip(prefix, prefix[1:]))
    )
    features["has_successful_test"] = float("integration_test_passed" in counts)
    features["has_review_approval"] = float("review_approved" in counts)
    # @cpt-end:cpt-event-prediction-algo-ml-prediction-extract-features:p1:inst-ml-feature-context
    # @cpt-begin:cpt-event-prediction-algo-ml-prediction-extract-features:p1:inst-ml-feature-recency
    for event in RECENCY_EVENTS:
        if event in counts:
            last_index = length - 1 - prefix[::-1].index(event)
            features[f"recency::{event}"] = float(min(recency_cap, length - 1 - last_index))
            features[f"seen::{event}"] = 1.0
        else:
            features[f"recency::{event}"] = float(recency_cap)
            features[f"seen::{event}"] = 0.0
    # @cpt-end:cpt-event-prediction-algo-ml-prediction-extract-features:p1:inst-ml-feature-recency
    # @cpt-begin:cpt-event-prediction-algo-ml-prediction-extract-features:p1:inst-ml-feature-result
    result = features
    # @cpt-end:cpt-event-prediction-algo-ml-prediction-extract-features:p1:inst-ml-feature-result
    return result

