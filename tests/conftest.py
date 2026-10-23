from __future__ import annotations

from collections.abc import Callable

import pytest

from event_prediction.catalog import EVENT_TYPES
from event_prediction.domain import PrefixExample


@pytest.fixture
def complete_examples() -> Callable[[str], list[PrefixExample]]:
    def factory(prefix_id: str) -> list[PrefixExample]:
        examples: list[PrefixExample] = []
        for repeat in range(3):
            for index, target in enumerate(EVENT_TYPES):
                prefix = tuple(EVENT_TYPES[(index + offset + repeat) % len(EVENT_TYPES)] for offset in range(1, 5))
                examples.append(PrefixExample(f"{prefix_id}-{repeat}-{index}", prefix, target))
        return examples

    return factory

