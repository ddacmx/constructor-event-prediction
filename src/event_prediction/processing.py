"""Sequence-level partitioning and prefix derivation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .domain import ConfigurationError, EventSequence, PrefixExample, SplitManifest


# @cpt-algo:cpt-event-prediction-algo-baseline-prediction-build-examples:p1
# @cpt-dod:cpt-event-prediction-dod-baseline-prediction-sequence-split:p1
def split_sequences(sequences: Iterable[EventSequence], seed: int) -> tuple[SplitManifest, dict[str, list[EventSequence]]]:
    items = list(sequences)
    identifiers = [sequence.sequence_id for sequence in items]
    if len(set(identifiers)) != len(identifiers):
        raise ConfigurationError("sequence identifiers must be unique")
    if len(items) < 7:
        raise ConfigurationError("at least 7 sequences are required for a 70/15/15 split")
    # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-build-examples:p1:inst-base-permute-ids
    order = np.random.default_rng(seed).permutation(len(items))
    # @cpt-end:cpt-event-prediction-algo-baseline-prediction-build-examples:p1:inst-base-permute-ids
    # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-build-examples:p1:inst-base-allocate
    n_train = int(len(items) * 0.70)
    n_validation = int(len(items) * 0.15)
    groups = {
        "train": [items[index] for index in order[:n_train]],
        "validation": [items[index] for index in order[n_train : n_train + n_validation]],
        "test": [items[index] for index in order[n_train + n_validation :]],
    }
    # @cpt-end:cpt-event-prediction-algo-baseline-prediction-build-examples:p1:inst-base-allocate
    # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-build-examples:p1:inst-base-assert-disjoint
    id_sets = {name: {sequence.sequence_id for sequence in values} for name, values in groups.items()}
    if any(id_sets[a] & id_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("split sets overlap")
    if set().union(*id_sets.values()) != set(identifiers):
        raise AssertionError("split union does not equal input identifiers")
    # @cpt-end:cpt-event-prediction-algo-baseline-prediction-build-examples:p1:inst-base-assert-disjoint
    manifest = SplitManifest(
        seed=seed,
        train_ids=tuple(sequence.sequence_id for sequence in groups["train"]),
        validation_ids=tuple(sequence.sequence_id for sequence in groups["validation"]),
        test_ids=tuple(sequence.sequence_id for sequence in groups["test"]),
    )
    # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-build-examples:p1:inst-base-example-result
    result = (manifest, groups)
    # @cpt-end:cpt-event-prediction-algo-baseline-prediction-build-examples:p1:inst-base-example-result
    return result


def build_prefix_examples(sequences: Iterable[EventSequence]) -> list[PrefixExample]:
    examples: list[PrefixExample] = []
    for sequence in sequences:
        events = sequence.event_types()
        # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-build-examples:p1:inst-base-create-prefix
        for target_index in range(1, len(events)):
            # @cpt-begin:cpt-event-prediction-algo-baseline-prediction-build-examples:p1:inst-base-observable-example
            examples.append(PrefixExample(sequence.sequence_id, events[:target_index], events[target_index]))
            # @cpt-end:cpt-event-prediction-algo-baseline-prediction-build-examples:p1:inst-base-observable-example
        # @cpt-end:cpt-event-prediction-algo-baseline-prediction-build-examples:p1:inst-base-create-prefix
    return examples


def split_and_build_examples(
    sequences: Iterable[EventSequence], seed: int
) -> tuple[SplitManifest, dict[str, list[PrefixExample]]]:
    manifest, groups = split_sequences(sequences, seed)
    return manifest, {name: build_prefix_examples(values) for name, values in groups.items()}
