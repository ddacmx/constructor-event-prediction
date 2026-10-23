---
status: ACCEPTED
date: 2026-10-20
description: Decision to use a history-aware probabilistic event simulator with an observable-only output boundary.
---

# Use a History-Aware Probabilistic Event Simulator

Decision to generate auditable, stochastic delivery histories with private lifecycle context and observable-only records.


<!-- toc -->

- [Context and Problem Statement](#context-and-problem-statement)
- [Decision Drivers](#decision-drivers)
- [Considered Options](#considered-options)
- [Decision Outcome](#decision-outcome)
  - [Consequences](#consequences)
  - [Confirmation](#confirmation)
- [Pros and Cons of the Options](#pros-and-cons-of-the-options)
  - [History-Aware Probabilistic State Machine](#history-aware-probabilistic-state-machine)
  - [Fixed First-Order Markov Chain](#fixed-first-order-markov-chain)
  - [Language-Model-Generated Event Narratives](#language-model-generated-event-narratives)
- [More Information](#more-information)
- [Traceability](#traceability)

<!-- /toc -->

**ID**: `cpt-event-prediction-adr-probabilistic-event-simulator`
## Context and Problem Statement

The assignment needs a meaningful synthetic dataset, but no production histories are available. The generator must create learnable software-delivery patterns without becoming a deterministic script or exposing private simulator context to the predictor.

## Decision Drivers

* Explicit and reviewable event semantics and transition probabilities
* Non-deterministic outcomes with longer-history effects
* Exact seed-based reproduction
* Strong separation between simulator internals and observable model inputs
* Feasible implementation and testing within the submission schedule

## Considered Options

* History-aware probabilistic state machine with internal lifecycle context
* Fixed first-order Markov chain
* Language-model-generated event narratives

## Decision Outcome

Chosen option: "History-aware probabilistic state machine with internal lifecycle context", because it provides explicit controllable transitions, stochastic branches, and higher-order effects while remaining deterministic under a seed and independently testable.

### Consequences

* Good, because transition rules and modifiers can be inspected and unit tested.
* Good, because the benchmark can contain signal that a prefix-based model may learn beyond the first-order baseline.
* Good, because hidden simulator context can be kept in memory and omitted from JSONL records.
* Bad, because plausible probabilities require calibration and diagnostics.
* Bad, because synthetic findings cannot be generalized to real engineering organizations.

### Confirmation

Design review must confirm the observation boundary. Automated tests must verify fixed-seed output, probability normalization, multiple successor support, history-sensitive transition changes, valid termination, and absence of hidden-state keys in emitted records.

## Pros and Cons of the Options

### History-Aware Probabilistic State Machine

The simulator combines base transition weights with bounded modifiers derived from recent events and internal lifecycle context, then normalizes and samples the next observable event.

* Good, because causal-looking lifecycle patterns remain explainable.
* Good, because random seeds fully control sampling.
* Good, because the target difficulty can be adjusted without changing model code.
* Bad, because poorly chosen modifiers may create rare or unreachable events.

### Fixed First-Order Markov Chain

The next event depends only on the immediately preceding event.

* Good, because implementation and analytical inspection are simple.
* Bad, because the Markov baseline would closely match the data-generating process and leave little reason for prefix features.
* Bad, because it cannot represent escalation after repeated failures or review readiness after a longer successful path.

### Language-Model-Generated Event Narratives

A generative language model proposes sequences or event text.

* Good, because surface narratives may appear varied.
* Bad, because generation requires an external service or large local model.
* Bad, because exact transition probabilities, reproducibility, and leakage boundaries become harder to audit.
* Bad, because the added scope does not improve the assignment's core statistical demonstration.

## More Information

Observable event types include interface modification, dependency upgrade, compatibility check, CI failure, bug fix, integration test, review approval, deployment, and rollback. Exact transition tables and modifier values belong in the feature specification and versioned configuration, not this decision record.

## Traceability

- **PRD**: [PRD.md](../PRD.md)
- **DESIGN**: [DESIGN.md](../DESIGN.md)

This decision directly addresses the following requirements or design elements:

* `cpt-event-prediction-fr-generate-sequences` - establishes the separate simulator approach.
* `cpt-event-prediction-fr-history-dependent-transitions` - provides longer-history behavior.
* `cpt-event-prediction-fr-prevent-leakage` - creates a strict hidden-state boundary.
* `cpt-event-prediction-nfr-reproducibility` - makes every sampled decision seed-controlled.
* `cpt-event-prediction-design-local-pipeline` - constrains the generator component in the technical design.
