"""Metrics, cluster-aware uncertainty, paired comparisons, and error analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .catalog import EVENT_TYPES
from .domain import ConfigurationError, PredictionBatch, PrefixExample


def stable_predictions(probabilities: np.ndarray) -> list[str]:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(EVENT_TYPES):
        raise ConfigurationError("probability matrix shape does not match event vocabulary")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ConfigurationError("probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ConfigurationError("probability rows must sum to one")
    return [EVENT_TYPES[int(index)] for index in np.argmax(probabilities, axis=1)]


# @cpt-algo:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1
def metric_record(targets: Sequence[str], probabilities: np.ndarray) -> dict[str, Any]:
    # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1:inst-eval-classify
    predictions = stable_predictions(probabilities)
    # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1:inst-eval-classify
    # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1:inst-eval-macro-f1
    precision, recall, class_f1, support = precision_recall_fscore_support(
        targets, predictions, labels=EVENT_TYPES, zero_division=0
    )
    macro_f1 = float(f1_score(targets, predictions, labels=EVENT_TYPES, average="macro", zero_division=0))
    # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1:inst-eval-macro-f1
    # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1:inst-eval-accuracy
    accuracy = float(accuracy_score(targets, predictions))
    # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1:inst-eval-accuracy
    # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1:inst-eval-confusion
    matrix = confusion_matrix(targets, predictions, labels=EVENT_TYPES).astype(int).tolist()
    # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1:inst-eval-confusion
    # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1:inst-eval-zero-support
    per_class = {
        event: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "support": int(support[index]),
        }
        for index, event in enumerate(EVENT_TYPES)
    }
    # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1:inst-eval-zero-support
    # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1:inst-eval-metric-result
    result = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "class_order": list(EVENT_TYPES),
        "confusion_matrix": matrix,
        "per_class": per_class,
    }
    # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1:inst-eval-metric-result
    return result


def _indices_by_sequence(examples: Sequence[PrefixExample]) -> dict[str, np.ndarray]:
    values: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        values[example.sequence_id].append(index)
    return {key: np.asarray(indices, dtype=int) for key, indices in values.items()}


def _validated_probability_arrays(
    examples: Sequence[PrefixExample], batches: Mapping[str, PredictionBatch]
) -> dict[str, np.ndarray]:
    expected_keys = tuple(example.key for example in examples)
    if len(set(expected_keys)) != len(expected_keys):
        raise ConfigurationError("evaluation example keys must be unique")
    expected_shape = (len(examples), len(EVENT_TYPES))
    arrays: dict[str, np.ndarray] = {}
    for name, batch in batches.items():
        if not isinstance(batch, PredictionBatch):
            raise ConfigurationError(f"{name} predictions must be a keyed PredictionBatch")
        if batch.example_keys != expected_keys:
            raise ConfigurationError(f"{name} prediction keys do not exactly match ordered evaluation examples")
        if len(set(batch.example_keys)) != len(batch.example_keys):
            raise ConfigurationError(f"{name} prediction keys must be unique")
        if batch.vocabulary != EVENT_TYPES:
            raise ConfigurationError(f"{name} prediction vocabulary does not match the fixed event vocabulary")
        values = np.asarray(batch.probabilities)
        if values.shape != expected_shape:
            raise ConfigurationError(f"{name} predictions have shape {values.shape}; expected {expected_shape}")
        stable_predictions(values)
        arrays[name] = values
    return arrays


# @cpt-algo:cpt-event-prediction-algo-benchmarking-evaluation-clustered-bootstrap:p1
def clustered_bootstrap(
    examples: Sequence[PrefixExample],
    model_probabilities: Mapping[str, PredictionBatch],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if replicates < 20:
        raise ConfigurationError("bootstrap requires at least 20 replicates")
    # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-clustered-bootstrap:p1:inst-eval-clusters
    clusters = _indices_by_sequence(examples)
    identifiers = tuple(sorted(clusters))
    if len(identifiers) < 2:
        raise ConfigurationError("bootstrap requires at least two sequence clusters")
    # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-clustered-bootstrap:p1:inst-eval-clusters
    targets = np.asarray([example.target for example in examples])
    arrays = _validated_probability_arrays(examples, model_probabilities)
    predictions = {name: np.asarray(stable_predictions(values)) for name, values in arrays.items()}
    if "logistic_regression" not in predictions:
        raise ConfigurationError("paired bootstrap requires logistic_regression predictions")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {f"{name}.accuracy": [] for name in predictions}
    samples.update({f"{name}.macro_f1": [] for name in predictions})
    for baseline in ("global_frequency", "markov"):
        if baseline in predictions:
            samples[f"logistic_minus_{baseline}.accuracy"] = []
            samples[f"logistic_minus_{baseline}.macro_f1"] = []
    for _ in range(replicates):
        # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-clustered-bootstrap:p1:inst-eval-resample
        chosen = rng.choice(identifiers, size=len(identifiers), replace=True)
        indices = np.concatenate([clusters[str(identifier)] for identifier in chosen])
        # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-clustered-bootstrap:p1:inst-eval-resample
        y_true = targets[indices]
        # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-clustered-bootstrap:p1:inst-eval-bootstrap-stat
        accuracy_values: dict[str, float] = {}
        macro_f1_values: dict[str, float] = {}
        for name, predicted in predictions.items():
            current = predicted[indices]
            accuracy_values[name] = float(accuracy_score(y_true, current))
            samples[f"{name}.accuracy"].append(accuracy_values[name])
            macro_f1_values[name] = float(f1_score(y_true, current, labels=EVENT_TYPES, average="macro", zero_division=0))
            samples[f"{name}.macro_f1"].append(macro_f1_values[name])
        for baseline in ("global_frequency", "markov"):
            key = f"logistic_minus_{baseline}.accuracy"
            if key in samples:
                samples[key].append(accuracy_values["logistic_regression"] - accuracy_values[baseline])
                samples[f"logistic_minus_{baseline}.macro_f1"].append(macro_f1_values["logistic_regression"] - macro_f1_values[baseline])
        # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-clustered-bootstrap:p1:inst-eval-bootstrap-stat
    # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-clustered-bootstrap:p1:inst-eval-percentiles
    intervals = {
        name: {
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
        }
        for name, values in samples.items()
    }
    # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-clustered-bootstrap:p1:inst-eval-percentiles
    # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-clustered-bootstrap:p1:inst-eval-interval-result
    result = {
        "cluster_count": len(identifiers),
        "replicates": replicates,
        "seed": seed,
        "intervals_95": intervals,
    }
    # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-clustered-bootstrap:p1:inst-eval-interval-result
    return result


def _category(example: PrefixExample, target: str, predicted: str, support: Counter[str]) -> list[str]:
    categories: list[str] = []
    if support[target] < max(5, sum(support.values()) * 0.03):
        categories.append("rare_target")
    if "ci_failed" in example.prefix[-3:] and target in {"bug_fixed", "compatibility_checked"}:
        categories.append("post_failure_recovery")
    if target in {"integration_test_passed", "review_approved"} or predicted in {"integration_test_passed", "review_approved"}:
        categories.append("readiness_ambiguity")
    if {target, predicted} == {"deployed", "rolled_back"}:
        categories.append("deployment_rollback_ambiguity")
    return categories or ["other"]


# @cpt-algo:cpt-event-prediction-algo-benchmarking-evaluation-error-analysis:p1
# @cpt-dod:cpt-event-prediction-dod-benchmarking-evaluation-errors:p1
def analyze_errors(
    examples: Sequence[PrefixExample], probabilities: np.ndarray, max_examples_per_category: int = 5
) -> dict[str, Any]:
    predictions = stable_predictions(probabilities)
    targets = [example.target for example in examples]
    support = Counter(targets)
    # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-error-analysis:p1:inst-eval-group-errors
    category_order = (
        "rare_target", "post_failure_recovery", "readiness_ambiguity",
        "deployment_rollback_ambiguity", "other",
    )
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in category_order}
    pair_counts: Counter[str] = Counter({
        f"{target}->{predicted}": 0
        for target in EVENT_TYPES
        for predicted in EVENT_TYPES
        if target != predicted
    })
    band_counts: Counter[str] = Counter({"short": 0, "medium": 0, "long": 0})
    for index, (example, target, predicted) in enumerate(zip(examples, targets, predictions, strict=True)):
        if target == predicted:
            continue
        predicted_index = EVENT_TYPES.index(predicted)
        true_index = EVENT_TYPES.index(target)
        margin = float(probabilities[index, predicted_index] - probabilities[index, true_index])
        record = {
            "confidence_margin": margin,
            "example_key": example.key,
            "predicted": predicted,
            "prefix": list(example.prefix),
            "prefix_length_band": "short" if len(example.prefix) <= 3 else "medium" if len(example.prefix) <= 7 else "long",
            "target": target,
        }
        pair_counts[f"{target}->{predicted}"] += 1
        band_counts[record["prefix_length_band"]] += 1
        # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-error-analysis:p1:inst-eval-group-errors
        # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-error-analysis:p1:inst-eval-tag-errors
        for category in _category(example, target, predicted, support):
            buckets[category].append(record)
        # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-error-analysis:p1:inst-eval-tag-errors
    # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-error-analysis:p1:inst-eval-rank-errors
    ranked = {
        category: {
            "count": len(records),
            "examples": sorted(records, key=lambda item: (-item["confidence_margin"], item["example_key"]))[
                :max_examples_per_category
            ],
        }
        for category, records in buckets.items()
    }
    # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-error-analysis:p1:inst-eval-rank-errors
    # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-error-analysis:p1:inst-eval-sample-errors
    bounded = ranked
    # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-error-analysis:p1:inst-eval-sample-errors
    # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-error-analysis:p1:inst-eval-error-result
    error_count = sum(pair_counts.values())
    result = {
        "semantic_categories": {"count": error_count, "examples": [], "groups": bounded},
        "true_predicted_pairs": {
            "count": error_count,
            "examples": [],
            "groups": {key: pair_counts[key] for key in sorted(pair_counts)},
        },
        "prefix_length_bands": {
            "count": error_count,
            "examples": [],
            "groups": {key: band_counts.get(key, 0) for key in ("short", "medium", "long")},
        },
    }
    # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-error-analysis:p1:inst-eval-error-result
    return result


def evaluate_models(
    examples: Sequence[PrefixExample], model_probabilities: Mapping[str, PredictionBatch], bootstrap_seed: int, replicates: int
) -> dict[str, Any]:
    targets = [example.target for example in examples]
    # @cpt-begin:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1:inst-eval-align
    arrays = _validated_probability_arrays(examples, model_probabilities)
    # @cpt-end:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1:inst-eval-align
    metrics = {name: metric_record(targets, values) for name, values in arrays.items()}
    return {
        "example_count": len(examples),
        "metrics": metrics,
        "bootstrap": clustered_bootstrap(examples, model_probabilities, bootstrap_seed, replicates),
        "error_analysis": analyze_errors(examples, model_probabilities["logistic_regression"].probabilities),
    }

