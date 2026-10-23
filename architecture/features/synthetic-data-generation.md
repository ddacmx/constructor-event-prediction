---
description: Feature specification for a deterministic history-aware probabilistic event simulator and observable JSONL dataset.
---

# Feature: Synthetic Data Generation

Specification for producing deterministic observable JSONL histories from a configurable history-aware probabilistic simulator.


<!-- toc -->

- [1. Feature Context](#1-feature-context)
  - [1.1 Overview](#11-overview)
  - [1.2 Purpose](#12-purpose)
  - [1.3 Actors](#13-actors)
  - [1.4 References](#14-references)
  - [1.5 Observable Event Catalog](#15-observable-event-catalog)
- [2. Actor Flows (CDSL)](#2-actor-flows-cdsl)
  - [Generate a Seeded Dataset](#generate-a-seeded-dataset)
- [3. Processes / Business Logic (CDSL)](#3-processes--business-logic-cdsl)
  - [Validate Generator Configuration](#validate-generator-configuration)
  - [Sample One Sequence](#sample-one-sequence)
  - [Canonicalize Dataset Output](#canonicalize-dataset-output)
- [4. States (CDSL)](#4-states-cdsl)
  - [Delivery Lifecycle Context State Machine](#delivery-lifecycle-context-state-machine)
- [5. Definitions of Done](#5-definitions-of-done)
  - [Validated Probabilistic Generator](#validated-probabilistic-generator)
  - [Leakage-Safe Portable Dataset](#leakage-safe-portable-dataset)
- [6. Acceptance Criteria](#6-acceptance-criteria)

<!-- /toc -->

- [ ] `p1` - **ID**: `cpt-event-prediction-featstatus-synthetic-data-generation`
## 1. Feature Context

- [ ] `p1` - `cpt-event-prediction-feature-synthetic-data-generation`

### 1.1 Overview

This feature defines the observable software-delivery event vocabulary and generates deterministic, variable-length JSONL sequences from a history-aware probabilistic simulator.

### 1.2 Purpose

The generator supplies meaningful training and evaluation data while keeping simulator-only lifecycle context outside the model observation boundary.

**Requirements**: `cpt-event-prediction-fr-event-taxonomy`, `cpt-event-prediction-fr-generate-sequences`, `cpt-event-prediction-fr-history-dependent-transitions`, `cpt-event-prediction-fr-jsonl-dataset`, `cpt-event-prediction-fr-prevent-leakage`, `cpt-event-prediction-nfr-reproducibility`

**Principles**: `cpt-event-prediction-principle-observable-boundary`, `cpt-event-prediction-principle-immutable-artifacts`

### 1.3 Actors

| Actor | Role in Feature |
|-------|-----------------|
| `cpt-event-prediction-actor-ml-engineer` | Chooses configuration, count, and seed and inspects diagnostics. |
| `cpt-event-prediction-actor-automation-runner` | Repeats generation and checks deterministic artifacts. |

### 1.4 References

- **PRD**: [PRD.md](../PRD.md)
- **Design**: [DESIGN.md](../DESIGN.md)
- **Dependencies**: None

### 1.5 Observable Event Catalog

| Event type | Short description | Typical role |
|------------|-------------------|--------------|
| `api_interface_modified` | An API interface was modified. | Begins compatibility-sensitive work. |
| `dependency_upgraded` | A software dependency was upgraded. | Begins dependency-sensitive work. |
| `compatibility_checked` | Compatibility with dependent components was checked. | Reduces integration risk. |
| `ci_failed` | Continuous integration reported a failure. | Creates recovery pressure. |
| `bug_fixed` | A defect discovered during delivery was fixed. | Recovers from failure. |
| `integration_test_passed` | Integration tests completed successfully. | Establishes release readiness. |
| `review_approved` | A reviewer approved the change. | Enables deployment when validation is sufficient. |
| `deployed` | The change was deployed. | Completes a successful delivery path. |
| `rolled_back` | A deployment was reverted. | Recovers from a risky or failed release. |

The catalog is versioned. Descriptions are stable observations, not generated text features.

## 2. Actor Flows (CDSL)

**Use cases**: `cpt-event-prediction-usecase-generate-dataset`

### Generate a Seeded Dataset

- [x] `p1` - **ID**: `cpt-event-prediction-flow-synthetic-data-generation-generate`

**Actor**: `cpt-event-prediction-actor-ml-engineer`

**Success Scenarios**:
- A request for 10,000 sequences produces a complete JSONL file and manifest.
- Repeating the request with identical inputs produces the same canonical hashes.

**Error Scenarios**:
- Invalid probabilities, unknown event types, impossible termination limits, or an existing protected output path stop the run before finalization.

**Steps**:
1. [x] - `p1` - Provide generation configuration, sequence count, seed, and output path - `inst-sdg-request`
2. [x] - `p1` - Validate the event catalog, transition graph, modifiers, length bounds, and output policy - `inst-sdg-validate`
3. [x] - `p1` - **IF** validation fails, return all configuration errors without a finalized dataset - `inst-sdg-invalid`
4. [x] - `p1` - Initialize one explicit random generator from the requested seed - `inst-sdg-seed`
5. [x] - `p1` - **FOR EACH** requested sequence, sample an observable lifecycle until a valid terminal condition - `inst-sdg-generate-each`
6. [x] - `p1` - Project each sampled lifecycle to observable event records and discard private simulator context - `inst-sdg-project`
7. [x] - `p1` - Write canonical JSONL and a manifest with schema, configuration hash, seed, count, and dataset hash - `inst-sdg-write`
8. [x] - `p1` - **RETURN** artifact paths, counts, diagnostics, and hashes - `inst-sdg-return`

## 3. Processes / Business Logic (CDSL)

### Validate Generator Configuration

- [x] `p1` - **ID**: `cpt-event-prediction-algo-synthetic-data-generation-validate-config`

**Input**: Event catalog, base transition weights, history modifiers, initial distribution, termination policy, and length bounds.

**Output**: Validated immutable configuration or a complete validation-error list.

**Steps**:
1. [x] - `p1` - Verify event codes are unique, descriptions are non-empty, and every referenced event exists - `inst-sdg-check-catalog`
2. [x] - `p1` - Verify all base weights and modifier factors are finite and non-negative - `inst-sdg-check-weights`
3. [x] - `p1` - Verify every reachable nonterminal situation has at least two positive-probability successors - `inst-sdg-check-stochasticity`
4. [x] - `p1` - Verify maximum length is greater than minimum length and every path can terminate by the maximum - `inst-sdg-check-termination`
5. [x] - `p1` - **RETURN** validated configuration or errors - `inst-sdg-config-result`

### Sample One Sequence

- [x] `p1` - **ID**: `cpt-event-prediction-algo-synthetic-data-generation-sample-sequence`

**Input**: Validated configuration and explicit random generator.

**Output**: One EventSequence containing only observable events.

**Canonical simulator law**: The versioned numeric weights, multipliers, length bounds, history window, and termination values are defined once in `configs/default.json`; this specification references those values and does not duplicate them. One `numpy.random.default_rng(seed)` instance is created per dataset and is consumed in sequence-index order. Mapping insertion order supplies the candidate order passed to `rng.choice`, so equal weights do not introduce a second tie-breaking rule.

For each successor draw, copy the base weights for the most recent observable event and take `recent = history[-history_window:]`. Apply modifiers in this exact order, cumulatively when they target the same successor: (1) when `ci_failed` occurs in `recent`, modify `bug_fixed` and `review_approved`; (2) when private context says compatibility is unresolved, modify `compatibility_checked` and `deployed`; (3) when a successful integration test has been observed, modify `review_approved`; (4) when private context is review-ready, modify `deployed`; and (5) only in private state `DEPLOYED`, modify `rolled_back`, selecting the with-failures factor when `recent_failures > 0` and the without-failures factor otherwise. Each update is `current base weight if present, otherwise the named fallback weight`, multiplied by the named factor. The fallback object contains exactly the five target keys used by those rules.

Discard non-positive successor weights and normalize the remaining finite weights by their sum. Before `min_length`, remove both `__end__` and `rolled_back` and renormalize. At or after `min_length`, use the already normalized distribution unchanged. `__end__` stops without becoming an observation; `rolled_back` is appended, moves context to `TERMINAL`, and ends immediately. Reaching `max_length` stops without an additional draw. The private context update happens after each observable append, and terminal rollback cannot be followed by another event.

**Steps**:
1. [x] - `p1` - Initialize private lifecycle context and sample a valid opening event - `inst-sdg-open-sequence`
2. [x] - `p1` - **WHILE** the sequence has not met a terminal condition or maximum length - `inst-sdg-loop`
   1. [x] - `p1` - Read base successor weights for the current observable event - `inst-sdg-base-weights`
   2. [x] - `p1` - Apply bounded modifiers for recent failures, unresolved compatibility, successful testing, review readiness, and post-deployment risk - `inst-sdg-apply-modifiers`
   3. [x] - `p1` - Normalize positive weights into a probability distribution - `inst-sdg-normalize`
   4. [x] - `p1` - Sample and append the next observable event - `inst-sdg-sample-next`
   5. [x] - `p1` - Update private lifecycle context from the sampled observation - `inst-sdg-update-context`
3. [x] - `p1` - Project event codes to stable descriptions and positions - `inst-sdg-describe-events`
4. [x] - `p1` - **RETURN** the observable EventSequence without private context - `inst-sdg-sequence-result`

### Canonicalize Dataset Output

- [x] `p1` - **ID**: `cpt-event-prediction-algo-synthetic-data-generation-canonical-output`

**Input**: Ordered EventSequence values and generation provenance.

**Output**: Final JSONL dataset, manifest, and SHA-256 hashes.

**Steps**:
1. [x] - `p1` - Serialize one sequence per line with stable key order and UTF-8 newline policy - `inst-sdg-serialize`
2. [x] - `p1` - **IF** a record contains an unapproved field, reject the record as a leakage-boundary violation - `inst-sdg-reject-hidden-field`
3. [x] - `p1` - Write to temporary files and compute counts and hashes - `inst-sdg-write-temp`
4. [x] - `p1` - Atomically finalize dataset and manifest after every record validates - `inst-sdg-finalize`
5. [x] - `p1` - **RETURN** artifact metadata - `inst-sdg-output-result`

## 4. States (CDSL)

### Delivery Lifecycle Context State Machine

- [x] `p1` - **ID**: `cpt-event-prediction-state-synthetic-data-generation-lifecycle`

**States**: CHANGE_OPEN, VALIDATING, RECOVERING, APPROVED, DEPLOYED, TERMINAL

**Initial State**: CHANGE_OPEN

**Transitions**:
1. [x] - `p1` - **FROM** CHANGE_OPEN **TO** VALIDATING **WHEN** a compatibility check or integration test is observed - `inst-sdg-state-validating`
2. [x] - `p1` - **FROM** CHANGE_OPEN **TO** RECOVERING **WHEN** CI failure is observed - `inst-sdg-state-early-failure`
3. [x] - `p1` - **FROM** VALIDATING **TO** RECOVERING **WHEN** CI failure is observed - `inst-sdg-state-validation-failure`
4. [x] - `p1` - **FROM** RECOVERING **TO** VALIDATING **WHEN** a bug fix is observed - `inst-sdg-state-fixed`
5. [x] - `p1` - **FROM** VALIDATING **TO** APPROVED **WHEN** review approval follows sufficient successful validation - `inst-sdg-state-approved`
6. [x] - `p1` - **FROM** APPROVED **TO** DEPLOYED **WHEN** deployment is observed - `inst-sdg-state-deployed`
7. [x] - `p1` - **FROM** DEPLOYED **TO** TERMINAL **WHEN** the episode ends successfully - `inst-sdg-state-success-end`
8. [x] - `p1` - **FROM** DEPLOYED **TO** TERMINAL **WHEN** rollback is observed - `inst-sdg-state-rollback-end`

These states affect sampling probabilities but are never emitted in the dataset.

## 5. Definitions of Done

### Validated Probabilistic Generator

- [x] `p1` - **ID**: `cpt-event-prediction-dod-synthetic-data-generation-generator`

The system **MUST** generate valid non-deterministic histories whose transition distributions change under documented history conditions and whose same-seed output is byte reproducible.

**Implements**:
- `cpt-event-prediction-flow-synthetic-data-generation-generate`
- `cpt-event-prediction-algo-synthetic-data-generation-validate-config`
- `cpt-event-prediction-algo-synthetic-data-generation-sample-sequence`

**Constraints**: `cpt-event-prediction-constraint-cpu-local`

**Touches**:
- CLI: `generate`
- Files: dataset JSONL, generation manifest
- Entities: EventType, EventSequence, SimulatorContext

### Leakage-Safe Portable Dataset

- [x] `p1` - **ID**: `cpt-event-prediction-dod-synthetic-data-generation-dataset`

The system **MUST** finalize only records that match the observable allowlist and include sufficient non-feature provenance to reproduce the dataset.

**Implements**:
- `cpt-event-prediction-algo-synthetic-data-generation-canonical-output`

**Constraints**: `cpt-event-prediction-constraint-schedule`

**Touches**:
- Files: versioned JSONL and JSON schemas
- Entities: ObservedEvent, EventSequence

## 6. Acceptance Criteria

- [ ] Identical seed and configuration produce identical JSONL and manifest hashes.
- [ ] Different seeds produce at least two distinct dataset hashes in a fixed smoke-test set.
- [ ] Every generated sequence respects configured length, vocabulary, reachability, and termination invariants.
- [ ] At least one documented history modifier changes a successor probability while leaving multiple successors possible.
- [ ] A generated 10,000-sequence dataset contains no simulator-context field and passes schema validation.
- [ ] Frequency and transition diagnostics identify unreachable events, zero-support successors, and severe class imbalance.
