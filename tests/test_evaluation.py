from __future__ import annotations

import numpy as np
import pytest

from event_prediction.catalog import EVENT_TYPES
from event_prediction.domain import ConfigurationError, PredictionBatch, PrefixExample
from event_prediction.evaluation import analyze_errors, clustered_bootstrap, evaluate_models, metric_record, stable_predictions


def _examples() -> list[PrefixExample]:
    return [
        PrefixExample(f"seq-{index // 3}", (EVENT_TYPES[index % 9],) * ((index % 3) + 1), EVENT_TYPES[(index + 1) % 9])
        for index in range(27)
    ]


def _probabilities(examples: list[PrefixExample], shift: int = 0) -> np.ndarray:
    matrix = np.full((len(examples), len(EVENT_TYPES)), 0.01)
    for row, example in enumerate(examples):
        target = EVENT_TYPES.index(example.target)
        selected = (target + shift) % len(EVENT_TYPES)
        matrix[row, selected] = 0.92
        matrix[row] /= matrix[row].sum()
    return matrix


def _batch(examples: list[PrefixExample], shift: int = 0) -> PredictionBatch:
    return PredictionBatch(tuple(item.key for item in examples), _probabilities(examples, shift))


# Scenario: cpt-event-prediction-algo-benchmarking-evaluation-metrics
def test_metrics_use_fixed_vocabulary() -> None:
    examples = _examples()
    values = metric_record([item.target for item in examples], _probabilities(examples))
    assert values["accuracy"] == 1.0
    assert values["macro_f1"] == 1.0
    assert np.asarray(values["confusion_matrix"]).shape == (9, 9)
    assert sum(item["support"] for item in values["per_class"].values()) == len(examples)


# Scenario: cpt-event-prediction-algo-benchmarking-evaluation-clustered-bootstrap
def test_clustered_bootstrap_is_deterministic_and_paired() -> None:
    examples = _examples()
    probabilities = {
        "global_frequency": _batch(examples, 2),
        "markov": _batch(examples, 1),
        "logistic_regression": _batch(examples),
    }
    first = clustered_bootstrap(examples, probabilities, 44, 20)
    second = clustered_bootstrap(examples, probabilities, 44, 20)
    assert first == second
    assert first["cluster_count"] == 9
    interval = first["intervals_95"]["logistic_minus_markov.accuracy"]
    assert interval["lower"] == pytest.approx(1.0)
    assert interval["upper"] == pytest.approx(1.0)


def test_error_analysis_and_aligned_evaluation() -> None:
    examples = _examples()
    wrong = _probabilities(examples, 1)
    errors = analyze_errors(examples, wrong, max_examples_per_category=2)
    assert errors
    assert all(len(value["examples"]) <= 2 for value in errors.values())
    result = evaluate_models(
        examples,
        {
            "global_frequency": PredictionBatch(tuple(item.key for item in examples), wrong),
            "markov": PredictionBatch(tuple(item.key for item in examples), wrong),
            "logistic_regression": _batch(examples),
        },
        9,
        20,
    )
    assert result["example_count"] == 27
    assert result["metrics"]["logistic_regression"]["accuracy"] == 1.0
    with pytest.raises(ConfigurationError, match="shape"):
        stable_predictions(np.ones((3, 3)))

