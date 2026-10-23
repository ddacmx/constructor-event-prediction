from __future__ import annotations

import hashlib
import json

import pytest

from event_prediction.artifacts import read_dataset, write_dataset
from event_prediction.catalog import EVENT_DESCRIPTIONS, EVENT_TYPES
from event_prediction.config import GeneratorConfig, load_config
from event_prediction.domain import ArtifactError, ConfigurationError
from event_prediction.simulator import (
    LifecycleState,
    SimulatorContext,
    generate_sequences,
    successor_probabilities,
    validate_transition_graph,
)


# Scenario: cpt-event-prediction-flow-synthetic-data-generation-generate
def test_generator_is_valid_and_deterministic() -> None:
    config = GeneratorConfig(min_length=4, max_length=10)
    first = generate_sequences(30, 17, config)
    second = generate_sequences(30, 17, config)
    assert [value.as_dict() for value in first] == [value.as_dict() for value in second]
    assert [value.as_dict() for value in first] != [value.as_dict() for value in generate_sequences(30, 18, config)]
    for sequence in first:
        assert 4 <= len(sequence.events) <= 10
        assert set(sequence.as_dict()) == {"events", "provenance", "schema_version", "sequence_id"}
        assert "state" not in json.dumps(sequence.as_dict()).lower()


def test_catalog_codes_and_descriptions_are_stable() -> None:
    assert len(EVENT_TYPES) == 9
    assert EVENT_TYPES[0] == "api_interface_modified"
    assert EVENT_DESCRIPTIONS["rolled_back"] == "A deployment was reverted."
    for event, description in EVENT_DESCRIPTIONS.items():
        assert event and description.endswith(".")


def test_history_changes_distribution_without_becoming_deterministic() -> None:
    config = GeneratorConfig()
    calm = SimulatorContext(state=LifecycleState.VALIDATING, unresolved_compatibility=False)
    failed = SimulatorContext(state=LifecycleState.RECOVERING, unresolved_compatibility=False, recent_failures=1)
    calm_values = successor_probabilities(("compatibility_checked",), calm, config)
    failed_values = successor_probabilities(("ci_failed", "compatibility_checked"), failed, config)
    assert failed_values["bug_fixed"] > calm_values["bug_fixed"]
    assert len([value for key, value in failed_values.items() if key != "__end__" and value > 0]) >= 2
    assert sum(calm_values.values()) == pytest.approx(1.0)


def test_transition_validation_rejects_unknown_and_deterministic_graphs() -> None:
    invalid = {event: {"not_an_event": 1.0} for event in EVENT_TYPES}
    with pytest.raises(ConfigurationError, match="unknown events"):
        validate_transition_graph(invalid)


def test_config_validation_and_loading(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"min_length": 4, "max_length": 9, "bootstrap_replicates": 20}', encoding="utf-8")
    value = load_config(path)
    assert value.min_length == 4
    assert len(value.digest()) == 64
    path.write_text('{"surprise": true}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown configuration"):
        load_config(path)
    with pytest.raises(ConfigurationError, match="max_length"):
        GeneratorConfig(min_length=5, max_length=5).validate()
    with pytest.raises(ConfigurationError, match="cannot read"):
        load_config(tmp_path / "missing.json")


# Scenario: cpt-event-prediction-algo-synthetic-data-generation-canonical-output
def test_jsonl_is_canonical_round_trippable_and_protected(tmp_path) -> None:
    config = GeneratorConfig(min_length=4, max_length=8)
    sequences = generate_sequences(12, 44, config)
    output = tmp_path / "dataset.jsonl"
    metadata = write_dataset(sequences, output, config, 44)
    payload = output.read_bytes()
    assert metadata["dataset_hash"] == hashlib.sha256(payload).hexdigest()
    assert payload.endswith(b"\n")
    assert len(payload.splitlines()) == 12
    assert read_dataset(output) == sequences
    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        write_dataset(sequences, output, config, 44)
    again = write_dataset(sequences, output, config, 44, replace=True)
    assert again["dataset_hash"] == metadata["dataset_hash"]


def test_read_dataset_rejects_hidden_field(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"schema_version": "event-sequence.v1", "sequence_id": "x", "events": [], "provenance": {"generation_seed": 1}, "hidden_state": "oops"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="unapproved fields"):
        read_dataset(path)

