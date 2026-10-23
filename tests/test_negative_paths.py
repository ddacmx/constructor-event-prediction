from __future__ import annotations

import json

import numpy as np
import pytest

from event_prediction.artifacts import read_dataset, write_dataset
from event_prediction.baselines import GlobalFrequencyPredictor, MarkovPredictor
from event_prediction.catalog import EVENT_TYPES, validate_event_code
from event_prediction.config import GeneratorConfig, load_config
from event_prediction.domain import ArtifactError, ConfigurationError, EventSequence, LeakageError, ObservedEvent, PredictionBatch, PrefixExample
from event_prediction.evaluation import clustered_bootstrap, evaluate_models, stable_predictions
from event_prediction.features import extract_features
from event_prediction.model import load_model, train_logistic
from event_prediction.simulator import generate_sequences


def test_domain_validation_errors() -> None:
    with pytest.raises(ValueError, match="unknown"):
        validate_event_code("unknown")
    with pytest.raises(ValueError, match="description"):
        ObservedEvent(EVENT_TYPES[0], "wrong", 0)
    with pytest.raises(ValueError, match="position"):
        ObservedEvent(EVENT_TYPES[0], "An API interface was modified.", -1)
    event = ObservedEvent(EVENT_TYPES[0], "An API interface was modified.", 0)
    with pytest.raises(ValueError, match="at least two"):
        EventSequence("x", (event,), 1)
    with pytest.raises(ValueError, match="non-empty"):
        PrefixExample("x", (), EVENT_TYPES[0])


def test_all_config_validation_errors(tmp_path) -> None:
    invalid_values = [
        {"schema_version": "bad"},
        {"min_length": 1},
        {"history_window": 0},
        {"recency_cap": 0},
        {"laplace_alpha": 0},
        {"regularization_candidates": []},
        {"regularization_candidates": [-1]},
        {"bootstrap_replicates": 1},
    ]
    for values in invalid_values:
        base = GeneratorConfig().__dict__ | values
        with pytest.raises(ConfigurationError):
            GeneratorConfig(**base).validate()
    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="root"):
        load_config(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="cannot read"):
        load_config(path)


def test_artifact_and_baseline_negative_paths(tmp_path) -> None:
    with pytest.raises(ArtifactError, match="empty"):
        write_dataset([], tmp_path / "empty.jsonl", GeneratorConfig(), 1)
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ArtifactError, match="empty"):
        read_dataset(path)
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="line 1"):
        read_dataset(path)
    with pytest.raises(ConfigurationError, match="requires"):
        GlobalFrequencyPredictor.fit([])
    with pytest.raises(ConfigurationError, match="requires"):
        MarkovPredictor.fit([], 1.0)
    model = MarkovPredictor.fit([PrefixExample("a", (EVENT_TYPES[0],), EVENT_TYPES[1])])
    with pytest.raises(ConfigurationError, match="non-empty"):
        model.predict_proba([()])


def test_feature_input_negative_paths() -> None:
    with pytest.raises(LeakageError, match="must contain"):
        extract_features({})
    with pytest.raises(ConfigurationError, match="not text"):
        extract_features("api_interface_modified")
    with pytest.raises(ConfigurationError, match="recency"):
        extract_features((EVENT_TYPES[0],), recency_cap=0)


def test_evaluation_negative_paths() -> None:
    examples = [PrefixExample("one", (EVENT_TYPES[0],), EVENT_TYPES[1])]
    bad = np.full((1, 9), 0.1)
    with pytest.raises(ConfigurationError, match="sum"):
        stable_predictions(bad)
    good = np.full((1, 9), 1 / 9)
    with pytest.raises(ConfigurationError, match="two sequence"):
        clustered_bootstrap(examples, {"logistic_regression": PredictionBatch((examples[0].key,), good)}, 1, 20)
    examples2 = examples + [PrefixExample("two", (EVENT_TYPES[1],), EVENT_TYPES[2])]
    good2 = np.full((2, 9), 1 / 9)
    with pytest.raises(ConfigurationError, match="logistic_regression"):
        clustered_bootstrap(examples2, {"markov": PredictionBatch(tuple(item.key for item in examples2), good2)}, 1, 20)
    with pytest.raises(ConfigurationError, match="shape"):
        evaluate_models(examples2, {"logistic_regression": PredictionBatch(tuple(item.key for item in examples2), good)}, 1, 20)


def test_model_negative_paths(complete_examples, tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="must not be empty"):
        train_logistic([], complete_examples("v"), (1.0,), 1)
    with pytest.raises(ConfigurationError, match="finite and positive"):
        train_logistic(complete_examples("t"), complete_examples("v"), (0.0,), 1)
    path = tmp_path / "wrong.model.json"
    path.write_bytes(b"not-json-or-executable")
    with pytest.raises(ArtifactError, match="safe model"):
        load_model(path)

