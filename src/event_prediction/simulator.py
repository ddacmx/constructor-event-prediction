"""History-aware probabilistic delivery-event simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

from .catalog import EVENT_DESCRIPTIONS, EVENT_TYPES
from .config import DEFAULT_TRANSITIONS, GeneratorConfig
from .domain import ConfigurationError, EventSequence, ObservedEvent

END = "__end__"

BASE_TRANSITIONS: dict[str, dict[str, float]] = DEFAULT_TRANSITIONS


class LifecycleState(str, Enum):
    CHANGE_OPEN = "CHANGE_OPEN"
    VALIDATING = "VALIDATING"
    RECOVERING = "RECOVERING"
    APPROVED = "APPROVED"
    DEPLOYED = "DEPLOYED"
    TERMINAL = "TERMINAL"


@dataclass
class SimulatorContext:
    """Private state: deliberately has no serialization method."""

    state: LifecycleState = LifecycleState.CHANGE_OPEN
    unresolved_compatibility: bool = True
    recent_failures: int = 0
    successful_test: bool = False
    review_ready: bool = False


def validate_transition_graph(transitions: Mapping[str, Mapping[str, float]] = BASE_TRANSITIONS) -> None:
    errors: list[str] = []
    # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-validate-config:p1:inst-sdg-check-catalog
    if set(transitions) != set(EVENT_TYPES):
        errors.append("transition table must define every event exactly once")
    # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-validate-config:p1:inst-sdg-check-catalog
    for predecessor, successors in transitions.items():
        unknown = set(successors) - set(EVENT_TYPES) - {END}
        if unknown:
            errors.append(f"{predecessor} references unknown events: {sorted(unknown)}")
        # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-validate-config:p1:inst-sdg-check-weights
        if any(not math.isfinite(weight) or weight < 0 for weight in successors.values()):
            errors.append(f"{predecessor} has a non-finite or negative weight")
        # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-validate-config:p1:inst-sdg-check-weights
        # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-validate-config:p1:inst-sdg-check-stochasticity
        positive_events = [name for name, value in successors.items() if name != END and value > 0]
        if len(positive_events) < 2:
            errors.append(f"{predecessor} must have at least two positive event successors")
        # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-validate-config:p1:inst-sdg-check-stochasticity
    if errors:
        raise ConfigurationError("invalid transition graph: " + "; ".join(errors))


def _normalized(weights: Mapping[str, float]) -> tuple[tuple[str, ...], np.ndarray]:
    # Sampling order is part of the simulator definition.  JSON object insertion
    # order must never influence which random variate maps to which event.  Every
    # distribution therefore follows the fixed observable vocabulary, with the
    # private termination token placed last when present.
    unknown = set(weights) - set(EVENT_TYPES) - {END}
    if unknown:
        raise ConfigurationError(f"transition distribution has unknown events: {sorted(unknown)}")
    candidate_order = (*EVENT_TYPES, END)
    positive = [(event, weights[event]) for event in candidate_order if event in weights and weights[event] > 0]
    if not positive:
        raise ConfigurationError("transition distribution has no positive weights")
    names = tuple(item[0] for item in positive)
    values = np.asarray([item[1] for item in positive], dtype=float)
    values /= values.sum()
    return names, values


def successor_probabilities(
    history: tuple[str, ...], context: SimulatorContext, config: GeneratorConfig
) -> dict[str, float]:
    """Return bounded, history-sensitive successor probabilities."""

    # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-base-weights
    weights = dict(config.transition_weights[history[-1]])
    # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-base-weights
    recent = history[-config.history_window :]
    modifiers = config.history_modifiers
    fallbacks = modifiers["fallback_weights"]
    if "ci_failed" in recent:
        weights["bug_fixed"] = weights.get("bug_fixed", fallbacks["bug_fixed"]) * modifiers["ci_failed_bug_fixed_multiplier"]
        weights["review_approved"] = weights.get("review_approved", fallbacks["review_approved"]) * modifiers["ci_failed_review_approved_multiplier"]
    if context.unresolved_compatibility:
        weights["compatibility_checked"] = weights.get("compatibility_checked", fallbacks["compatibility_checked"]) * modifiers["unresolved_compatibility_checked_multiplier"]
        weights["deployed"] = weights.get("deployed", fallbacks["deployed"]) * modifiers["unresolved_deployed_multiplier"]
    if context.successful_test:
        weights["review_approved"] = weights.get("review_approved", fallbacks["review_approved"]) * modifiers["successful_test_review_approved_multiplier"]
    if context.review_ready:
        weights["deployed"] = weights.get("deployed", fallbacks["deployed"]) * modifiers["review_ready_deployed_multiplier"]
    if context.state is LifecycleState.DEPLOYED:
        rollback_factor = modifiers["deployed_rollback_with_failures_multiplier"] if context.recent_failures else modifiers["deployed_rollback_without_failures_multiplier"]
        weights["rolled_back"] = weights.get("rolled_back", fallbacks["rolled_back"]) * rollback_factor
    names, probabilities = _normalized(weights)
    return dict(zip(names, probabilities, strict=True))


# @cpt-state:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1
def _update_context(context: SimulatorContext, event: str) -> None:
    # @cpt-begin:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-validating
    if event in {"api_interface_modified", "dependency_upgraded"}:
        context.state = LifecycleState.CHANGE_OPEN
        context.unresolved_compatibility = True
        context.successful_test = False
        context.review_ready = False
    elif event == "compatibility_checked":
        context.state = LifecycleState.VALIDATING
        context.unresolved_compatibility = False
    # @cpt-end:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-validating
    # @cpt-begin:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-early-failure
    # @cpt-begin:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-validation-failure
    elif event == "ci_failed":
        context.state = LifecycleState.RECOVERING
        context.recent_failures = min(3, context.recent_failures + 1)
        context.review_ready = False
    # @cpt-end:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-validation-failure
    # @cpt-end:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-early-failure
    # @cpt-begin:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-fixed
    elif event == "bug_fixed":
        context.state = LifecycleState.VALIDATING
        context.recent_failures = max(0, context.recent_failures - 1)
    # @cpt-end:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-fixed
    elif event == "integration_test_passed":
        context.state = LifecycleState.VALIDATING
        context.successful_test = True
        context.review_ready = not context.unresolved_compatibility
    # @cpt-begin:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-approved
    elif event == "review_approved":
        context.state = LifecycleState.APPROVED
        context.review_ready = True
    # @cpt-end:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-approved
    # @cpt-begin:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-deployed
    elif event == "deployed":
        context.state = LifecycleState.DEPLOYED
    # @cpt-end:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-deployed
    # @cpt-begin:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-rollback-end
    elif event == "rolled_back":
        context.state = LifecycleState.TERMINAL
    # @cpt-end:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-rollback-end


# @cpt-algo:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1
def sample_sequence(index: int, seed: int, rng: np.random.Generator, config: GeneratorConfig) -> EventSequence:
    # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-open-sequence
    context = SimulatorContext()
    opening_names, opening_probabilities = _normalized(config.initial_weights)
    opening = str(rng.choice(opening_names, p=opening_probabilities))
    history: list[str] = [opening]
    _update_context(context, opening)
    # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-open-sequence
    # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-loop
    while len(history) < config.max_length:
        # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-apply-modifiers
        probabilities = successor_probabilities(tuple(history), context, config)
        # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-apply-modifiers
        if len(history) < config.min_length:
            probabilities.pop(END, None)
            probabilities.pop("rolled_back", None)
            # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-normalize
            names, probs = _normalized(probabilities)
            # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-normalize
        else:
            names = tuple(probabilities)
            probs = np.asarray(tuple(probabilities.values()), dtype=float)
        # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-sample-next
        next_event = str(rng.choice(names, p=probs))
        # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-sample-next
        if next_event == END:
            context.state = LifecycleState.TERMINAL
            break
        history.append(next_event)
        # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-update-context
        _update_context(context, next_event)
        # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-update-context
        # @cpt-begin:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-rollback-end
        if context.state is LifecycleState.TERMINAL:
            break
        # @cpt-end:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-rollback-end
    # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-loop
    # @cpt-begin:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-success-end
    if context.state is LifecycleState.DEPLOYED:
        context.state = LifecycleState.TERMINAL
    # @cpt-end:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1:inst-sdg-state-success-end
    # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-describe-events
    events = tuple(
        ObservedEvent(event_type=code, description=EVENT_DESCRIPTIONS[code], position=position)
        for position, code in enumerate(history)
    )
    # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-describe-events
    # @cpt-begin:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-project
    # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-sequence-result
    result = EventSequence(f"seq-{index:06d}", events, generation_seed=seed)
    # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-sample-sequence:p1:inst-sdg-sequence-result
    # @cpt-end:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-project
    return result


# @cpt-flow:cpt-event-prediction-flow-synthetic-data-generation-generate:p1
# @cpt-dod:cpt-event-prediction-dod-synthetic-data-generation-generator:p1
def generate_sequences(count: int, seed: int, config: GeneratorConfig) -> list[EventSequence]:
    # @cpt-begin:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-validate
    # @cpt-begin:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-invalid
    if count < 3:
        raise ConfigurationError("sequence count must be at least 3")
    # @cpt-end:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-invalid
    config.validate()
    validate_transition_graph(config.transition_weights)
    # @cpt-end:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-validate
    # @cpt-begin:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-seed
    rng = np.random.default_rng(seed)
    # @cpt-end:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-seed
    # @cpt-begin:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-generate-each
    result = [sample_sequence(index, seed, rng, config) for index in range(count)]
    # @cpt-end:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-generate-each
    return result
