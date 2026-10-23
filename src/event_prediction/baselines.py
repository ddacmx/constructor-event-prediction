"""Transparent next-event probability baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .catalog import EVENT_TYPES
from .domain import ConfigurationError, PredictionBatch, PrefixExample


class Predictor(Protocol):
    def predict_proba(self, prefixes: Sequence[tuple[str, ...]]) -> np.ndarray: ...
    def predict_examples(self, examples: Sequence[PrefixExample]) -> PredictionBatch: ...


@dataclass(frozen=True)
class GlobalFrequencyPredictor:
    probabilities: np.ndarray
    vocabulary: tuple[str, ...] = EVENT_TYPES

    # @cpt-algo:cpt-event-prediction-algo-baseline-prediction-global-frequency:p1
    @classmethod
    def fit(cls, examples: Sequence[PrefixExample]) -> "GlobalFrequencyPredictor":
        if not examples:
            raise ConfigurationError("global-frequency predictor requires training examples")
        # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-global-frequency:p1:inst-base-global-count
        counts = Counter(example.target for example in examples)
        # @cpt-end:cpt-event-prediction-algo-baseline-prediction-global-frequency:p1:inst-base-global-count
        # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-global-frequency:p1:inst-base-global-normalize
        values = np.asarray([counts[event] for event in EVENT_TYPES], dtype=float)
        total = values.sum()
        if total <= 0:
            raise ConfigurationError("training targets are empty")
        probabilities = values / total
        # @cpt-end:cpt-event-prediction-algo-baseline-prediction-global-frequency:p1:inst-base-global-normalize
        # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-global-frequency:p1:inst-base-global-tie
        # Stable EVENT_TYPES column order supplies deterministic argmax tie-breaking.
        probabilities = probabilities.copy()
        # @cpt-end:cpt-event-prediction-algo-baseline-prediction-global-frequency:p1:inst-base-global-tie
        # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-global-frequency:p1:inst-base-global-result
        result = cls(probabilities)
        # @cpt-end:cpt-event-prediction-algo-baseline-prediction-global-frequency:p1:inst-base-global-result
        return result

    def predict_proba(self, prefixes: Sequence[tuple[str, ...]]) -> np.ndarray:
        return np.tile(self.probabilities, (len(prefixes), 1))

    def predict_examples(self, examples: Sequence[PrefixExample]) -> PredictionBatch:
        return PredictionBatch(tuple(item.key for item in examples), self.predict_proba([item.prefix for item in examples]))


@dataclass(frozen=True)
class MarkovPredictor:
    rows: dict[str, np.ndarray]
    fallback: np.ndarray
    alpha: float
    vocabulary: tuple[str, ...] = EVENT_TYPES

    # @cpt-algo:cpt-event-prediction-algo-baseline-prediction-markov:p1
    @classmethod
    def fit(cls, examples: Sequence[PrefixExample], alpha: float = 1.0) -> "MarkovPredictor":
        if alpha <= 0 or not np.isfinite(alpha):
            raise ConfigurationError("Markov alpha must be finite and greater than zero")
        if not examples:
            raise ConfigurationError("Markov predictor requires training examples")
        target_counts = Counter(example.target for example in examples)
        conditional: dict[str, Counter[str]] = defaultdict(Counter)
        # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-markov:p1:inst-base-markov-count
        for example in examples:
            conditional[example.prefix[-1]][example.target] += 1
        # @cpt-end:cpt-event-prediction-algo-baseline-prediction-markov:p1:inst-base-markov-count
        fallback_counts = np.asarray([target_counts[event] + alpha for event in EVENT_TYPES], dtype=float)
        fallback = fallback_counts / fallback_counts.sum()
        rows: dict[str, np.ndarray] = {}
        for predecessor in EVENT_TYPES:
            # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-markov:p1:inst-base-markov-smooth
            values = np.asarray(
                [conditional[predecessor][successor] + alpha for successor in EVENT_TYPES], dtype=float
            )
            # @cpt-end:cpt-event-prediction-algo-baseline-prediction-markov:p1:inst-base-markov-smooth
            # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-markov:p1:inst-base-markov-normalize
            rows[predecessor] = values / values.sum()
            # @cpt-end:cpt-event-prediction-algo-baseline-prediction-markov:p1:inst-base-markov-normalize
        # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-markov:p1:inst-base-markov-result
        result = cls(rows=rows, fallback=fallback, alpha=alpha)
        # @cpt-end:cpt-event-prediction-algo-baseline-prediction-markov:p1:inst-base-markov-result
        return result

    def predict_proba(self, prefixes: Sequence[tuple[str, ...]]) -> np.ndarray:
        if any(not prefix for prefix in prefixes):
            raise ConfigurationError("Markov prediction requires non-empty prefixes")
        # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-markov:p1:inst-base-markov-fallback
        result = np.vstack([self.rows.get(prefix[-1], self.fallback) for prefix in prefixes])
        # @cpt-end:cpt-event-prediction-algo-baseline-prediction-markov:p1:inst-base-markov-fallback
        return result

    def predict_examples(self, examples: Sequence[PrefixExample]) -> PredictionBatch:
        return PredictionBatch(tuple(item.key for item in examples), self.predict_proba([item.prefix for item in examples]))


# @cpt-flow:cpt-event-prediction-flow-baseline-prediction-fit-apply:p1
# @cpt-dod:cpt-event-prediction-dod-baseline-prediction-models:p1
def fit_baselines(examples: Sequence[PrefixExample], alpha: float) -> dict[str, Predictor]:
    return {
        "global_frequency": GlobalFrequencyPredictor.fit(examples),
        "markov": MarkovPredictor.fit(examples, alpha),
    }

