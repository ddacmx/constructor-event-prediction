"""Typed observable domain records and validation errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .catalog import EVENT_DESCRIPTIONS, validate_event_code

DATASET_SCHEMA = "event-sequence.v1"
MANIFEST_SCHEMA = "dataset-manifest.v1"
RESULT_SCHEMA = "benchmark-result.v2"
MODEL_SCHEMA = "event-predictor.v2"


class EventPredictionError(Exception):
    """Base class for actionable user-facing failures."""


class ConfigurationError(EventPredictionError):
    """Raised when a configuration violates declared invariants."""


class ArtifactError(EventPredictionError):
    """Raised when a persisted artifact is invalid or unsafe to replace."""


class LeakageError(EventPredictionError):
    """Raised when non-observable data crosses the feature boundary."""


@dataclass(frozen=True)
class ObservedEvent:
    event_type: str
    description: str
    position: int

    def __post_init__(self) -> None:
        validate_event_code(self.event_type)
        if self.description != EVENT_DESCRIPTIONS[self.event_type]:
            raise ValueError("event description does not match the stable catalog")
        if self.position < 0:
            raise ValueError("event position must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "event_type": self.event_type,
            "position": self.position,
        }


@dataclass(frozen=True)
class EventSequence:
    sequence_id: str
    events: tuple[ObservedEvent, ...]
    generation_seed: int
    schema_version: str = DATASET_SCHEMA

    def __post_init__(self) -> None:
        if not self.sequence_id:
            raise ValueError("sequence_id must not be empty")
        if len(self.events) < 2:
            raise ValueError("a sequence must contain at least two events")
        if tuple(event.position for event in self.events) != tuple(range(len(self.events))):
            raise ValueError("event positions must be contiguous and zero-based")

    def event_types(self) -> tuple[str, ...]:
        return tuple(event.event_type for event in self.events)

    def as_dict(self) -> dict[str, Any]:
        return {
            "events": [event.as_dict() for event in self.events],
            "provenance": {"generation_seed": self.generation_seed},
            "schema_version": self.schema_version,
            "sequence_id": self.sequence_id,
        }


@dataclass(frozen=True)
class PrefixExample:
    sequence_id: str
    prefix: tuple[str, ...]
    target: str

    def __post_init__(self) -> None:
        if not self.prefix:
            raise ValueError("prefix must be non-empty")
        for code in self.prefix:
            validate_event_code(code)
        validate_event_code(self.target)

    @property
    def key(self) -> str:
        return f"{self.sequence_id}:{len(self.prefix)}"


@dataclass(frozen=True)
class SplitManifest:
    seed: int
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "split-manifest.v1",
            "seed": self.seed,
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "test_ids": list(self.test_ids),
        }

    def validate(self, expected_ids: set[str] | None = None) -> None:
        groups = (set(self.train_ids), set(self.validation_ids), set(self.test_ids))
        if any(len(values) != len(original) for values, original in zip(groups, (self.train_ids, self.validation_ids, self.test_ids), strict=True)):
            raise ConfigurationError("split manifest contains duplicate sequence identifiers")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ConfigurationError("split manifest assignments overlap")
        if expected_ids is not None and set().union(*groups) != expected_ids:
            raise ConfigurationError("split manifest union does not match the dataset")


@dataclass(frozen=True)
class PredictionBatch:
    """Predictions bound to the exact ordered examples they describe."""

    example_keys: tuple[str, ...]
    probabilities: Any
    vocabulary: tuple[str, ...] = tuple(EVENT_DESCRIPTIONS)

