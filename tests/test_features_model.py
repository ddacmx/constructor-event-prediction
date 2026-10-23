from __future__ import annotations

import numpy as np
import pytest

from event_prediction.catalog import EVENT_TYPES
from event_prediction.domain import ArtifactError, ConfigurationError, LeakageError
from event_prediction.features import extract_features
from event_prediction.model import load_model, predict_prefix, save_model, train_logistic


# Scenario: cpt-event-prediction-algo-ml-prediction-extract-features
def test_features_cover_history_and_reject_leakage() -> None:
    prefix = ("api_interface_modified", "ci_failed", "bug_fixed", "integration_test_passed")
    values = extract_features(prefix, recency_cap=7)
    assert values["lag_1"] == "integration_test_passed"
    assert values["lag_3"] == "ci_failed"
    assert values["count::ci_failed"] == 1
    assert values["proportion::ci_failed"] == pytest.approx(0.25)
    assert values["transition::ci_failed->bug_fixed"] == 1
    assert values["fix_after_failure"] == 1
    assert values["recency::integration_test_passed"] == 0
    forbidden = ("target", "split", "seed", "sequence_id", "future", "simulator_state", "provenance")
    for key in forbidden:
        with pytest.raises(LeakageError, match="forbidden"):
            extract_features({"prefix": prefix, key: "leak"})
    with pytest.raises(ConfigurationError, match="non-empty"):
        extract_features([])


def test_later_events_cannot_change_prefix_features() -> None:
    prefix = ("api_interface_modified", "compatibility_checked")
    first_future = prefix + ("ci_failed",)
    second_future = prefix + ("review_approved",)
    assert extract_features(first_future[:2]) == extract_features(second_future[:2])


# Scenario: cpt-event-prediction-flow-ml-prediction-train-select
def test_logistic_training_prediction_selection_and_round_trip(complete_examples, tmp_path) -> None:
    training = complete_examples("train")
    validation = complete_examples("validation")
    model = train_logistic(training, validation, (1.0, 0.1), seed=19, provenance={
        "config_hash": "1" * 64,
        "split_seed": 29,
        "training_examples_hash": "2" * 64,
        "final_profile_hash": "3" * 64,
    })
    assert model.selected_c in {0.1, 1.0}
    assert [row["c"] for row in model.selection_report] == [0.1, 1.0]
    prefixes = [item.prefix for item in validation[:5]]
    probabilities = model.predict_proba(prefixes)
    assert probabilities.shape == (5, 9)
    assert np.all(probabilities >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    prediction = predict_prefix(model, prefixes[0])
    assert prediction["predicted_event"] in EVENT_TYPES
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
    path = tmp_path / "model.model.json"
    metadata = save_model(model, path)
    loaded = load_model(path)
    assert np.allclose(loaded.predict_proba(prefixes), probabilities)
    assert metadata["schema_version"] == "event-predictor.v2"
    with pytest.raises(ArtifactError, match="overwrite"):
        save_model(model, path)


def test_logistic_requires_all_training_classes(complete_examples) -> None:
    values = [item for item in complete_examples("train") if item.target != EVENT_TYPES[-1]]
    with pytest.raises(ConfigurationError, match="lacks event classes"):
        train_logistic(values, complete_examples("validation"), (1.0,), seed=1)


def test_load_model_rejects_untrusted_artifact(tmp_path) -> None:
    path = tmp_path / "bad.model.json"
    path.write_bytes(b"not a model")
    with pytest.raises(ArtifactError, match="cannot load safe"):
        load_model(path)

