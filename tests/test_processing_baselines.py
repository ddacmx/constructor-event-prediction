from __future__ import annotations

import numpy as np
import pytest

from event_prediction.baselines import GlobalFrequencyPredictor, MarkovPredictor
from event_prediction.catalog import EVENT_TYPES
from event_prediction.config import GeneratorConfig
from event_prediction.domain import ConfigurationError, PrefixExample
from event_prediction.processing import build_prefix_examples, split_and_build_examples, split_sequences
from event_prediction.simulator import generate_sequences


# Scenario: cpt-event-prediction-algo-baseline-prediction-build-examples
def test_split_is_deterministic_disjoint_and_precedes_prefixes() -> None:
    sequences = generate_sequences(20, 2, GeneratorConfig(min_length=4, max_length=8))
    first_manifest, first = split_and_build_examples(sequences, 90)
    second_manifest, second = split_and_build_examples(sequences, 90)
    assert first_manifest == second_manifest
    assert first == second
    train, validation, test = map(set, (first_manifest.train_ids, first_manifest.validation_ids, first_manifest.test_ids))
    assert not train & validation and not train & test and not validation & test
    assert train | validation | test == {item.sequence_id for item in sequences}
    assert (len(train), len(validation), len(test)) == (14, 3, 3)
    for split_name, examples in first.items():
        allowed = set(getattr(first_manifest, f"{split_name}_ids"))
        assert all(example.sequence_id in allowed for example in examples)


def test_prefix_semantics() -> None:
    sequence = generate_sequences(3, 3, GeneratorConfig(min_length=4, max_length=7))[0]
    examples = build_prefix_examples([sequence])
    events = sequence.event_types()
    assert len(examples) == len(events) - 1
    for index, example in enumerate(examples, start=1):
        assert example.prefix == events[:index]
        assert example.target == events[index]


def test_split_rejects_duplicates_and_too_few() -> None:
    values = generate_sequences(7, 8, GeneratorConfig(min_length=4, max_length=7))
    with pytest.raises(ConfigurationError, match="unique"):
        split_sequences(values + [values[0]], 1)
    with pytest.raises(ConfigurationError, match="at least 7"):
        split_sequences(values[:6], 1)


# Scenario: cpt-event-prediction-algo-baseline-prediction-global-frequency
def test_global_frequency_matches_training_counts() -> None:
    examples = [
        PrefixExample("a", (EVENT_TYPES[0],), EVENT_TYPES[1]),
        PrefixExample("b", (EVENT_TYPES[0],), EVENT_TYPES[1]),
        PrefixExample("c", (EVENT_TYPES[1],), EVENT_TYPES[2]),
    ]
    model = GlobalFrequencyPredictor.fit(examples)
    probabilities = model.predict_proba([(EVENT_TYPES[0],), (EVENT_TYPES[2],)])
    assert probabilities.shape == (2, 9)
    assert probabilities[0, 1] == pytest.approx(2 / 3)
    assert probabilities[0].sum() == pytest.approx(1.0)


# Scenario: cpt-event-prediction-algo-baseline-prediction-markov
def test_markov_is_smoothed_normalized_and_has_fallback() -> None:
    examples = [
        PrefixExample("a", (EVENT_TYPES[0],), EVENT_TYPES[1]),
        PrefixExample("b", (EVENT_TYPES[0],), EVENT_TYPES[1]),
        PrefixExample("c", (EVENT_TYPES[1],), EVENT_TYPES[2]),
    ]
    model = MarkovPredictor.fit(examples, alpha=1.0)
    probabilities = model.predict_proba([(EVENT_TYPES[0],), ("unknown-predecessor",)])
    assert probabilities.shape == (2, 9)
    assert np.all(probabilities > 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0, 1] > probabilities[0, 2]
    with pytest.raises(ConfigurationError, match="greater than zero"):
        MarkovPredictor.fit(examples, alpha=0)

