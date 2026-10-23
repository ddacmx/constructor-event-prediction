---
description: Feature specification for sequence-level splitting and transparent frequency and Markov predictors.
---

# Feature: Baseline Prediction

Specification for leakage-safe sequence splitting and two transparent next-event probability baselines.


<!-- toc -->

- [1. Feature Context](#1-feature-context)
  - [1.1 Overview](#11-overview)
  - [1.2 Purpose](#12-purpose)
  - [1.3 Actors](#13-actors)
  - [1.4 References](#14-references)
- [2. Actor Flows (CDSL)](#2-actor-flows-cdsl)
  - [Fit and Apply Baselines](#fit-and-apply-baselines)
- [3. Processes / Business Logic (CDSL)](#3-processes--business-logic-cdsl)
  - [Partition Sequences and Build Prefixes](#partition-sequences-and-build-prefixes)
  - [Fit Global-Frequency Predictor](#fit-global-frequency-predictor)
  - [Fit Smoothed Markov Predictor](#fit-smoothed-markov-predictor)
- [4. States (CDSL)](#4-states-cdsl)
  - [Stateless Fitted Baselines](#stateless-fitted-baselines)
- [5. Definitions of Done](#5-definitions-of-done)
  - [Leakage-Safe Sequence Split](#leakage-safe-sequence-split)
  - [Two Reference Predictors](#two-reference-predictors)
- [6. Acceptance Criteria](#6-acceptance-criteria)

<!-- /toc -->

- [ ] `p1` - **ID**: `cpt-event-prediction-featstatus-baseline-prediction`
## 1. Feature Context

- [ ] `p1` - `cpt-event-prediction-feature-baseline-prediction`

### 1.1 Overview

This feature partitions complete sequences, derives supervised prefix examples within each partition, and implements global-frequency and Laplace-smoothed first-order Markov predictors.

### 1.2 Purpose

The baselines quantify dataset difficulty and immediate-transition signal while establishing a shared prediction interface for evaluation.

**Requirements**: `cpt-event-prediction-fr-baseline-models`, `cpt-event-prediction-fr-sequence-split`, `cpt-event-prediction-fr-prevent-leakage`, `cpt-event-prediction-nfr-reproducibility`, `cpt-event-prediction-nfr-testability`

**Principles**: `cpt-event-prediction-principle-baselines-first`, `cpt-event-prediction-principle-sequence-unit`

### 1.3 Actors

| Actor | Role in Feature |
|-------|-----------------|
| `cpt-event-prediction-actor-ml-engineer` | Fits and inspects reference predictors. |
| `cpt-event-prediction-actor-automation-runner` | Reproduces splits and baseline predictions. |

### 1.4 References

- **PRD**: [PRD.md](../PRD.md)
- **Design**: [DESIGN.md](../DESIGN.md)
- **Dependencies**: `cpt-event-prediction-feature-synthetic-data-generation`

## 2. Actor Flows (CDSL)

**Use cases**: `cpt-event-prediction-usecase-run-benchmark`

### Fit and Apply Baselines

- [x] `p1` - **ID**: `cpt-event-prediction-flow-baseline-prediction-fit-apply`

**Actor**: `cpt-event-prediction-actor-automation-runner`

**Success Scenarios**:
- Both baselines fit on training examples and return aligned probability vectors for validation or test prefixes.

**Error Scenarios**:
- Duplicate sequence membership, unknown events, empty training data, or a vocabulary mismatch stops prediction.

**Steps**:
1. [x] - `p1` - Load validated observable sequences and the experiment split seed - `inst-base-load`
2. [x] - `p1` - Assign complete sequence identifiers to train, validation, and test partitions - `inst-base-split`
3. [x] - `p1` - Derive prefix-target examples independently inside each partition - `inst-base-prefixes`
4. [x] - `p1` - Fit global target counts and predecessor-successor counts from training examples only - `inst-base-fit`
5. [x] - `p1` - Predict aligned class probabilities for the requested held-out partition - `inst-base-predict`
6. [x] - `p1` - **RETURN** split manifest, baseline metadata, and predictions - `inst-base-return`

## 3. Processes / Business Logic (CDSL)

### Partition Sequences and Build Prefixes

- [x] `p1` - **ID**: `cpt-event-prediction-algo-baseline-prediction-build-examples`

**Input**: Validated EventSequence collection, split ratios, and split seed.

**Output**: Disjoint SplitManifest and prefix-target examples for each partition.

**Steps**:
1. [x] - `p1` - Deterministically permute unique sequence identifiers using the split seed - `inst-base-permute-ids`
2. [x] - `p1` - Allocate identifiers by stable 70/15/15 boundary calculation - `inst-base-allocate`
3. [x] - `p1` - Assert pairwise disjoint sets whose union equals all input identifiers - `inst-base-assert-disjoint`
4. [x] - `p1` - **FOR EACH** sequence within one partition, create one example for every non-empty prefix with a following event - `inst-base-create-prefix`
5. [x] - `p1` - Store only sequence identifier, prefix events, prefix length, and target event in an example - `inst-base-observable-example`
6. [x] - `p1` - **RETURN** manifest and partitioned examples - `inst-base-example-result`

### Fit Global-Frequency Predictor

- [x] `p1` - **ID**: `cpt-event-prediction-algo-baseline-prediction-global-frequency`

**Input**: Training prefix-target examples and complete event vocabulary.

**Output**: One normalized target distribution used for every prefix.

**Steps**:
1. [x] - `p1` - Count target events in training examples only - `inst-base-global-count`
2. [x] - `p1` - Normalize counts over the complete vocabulary - `inst-base-global-normalize`
3. [x] - `p1` - Break equal-probability top classes by stable event-catalog order - `inst-base-global-tie`
4. [x] - `p1` - **RETURN** predictor metadata and probability distribution - `inst-base-global-result`

### Fit Smoothed Markov Predictor

- [x] `p1` - **ID**: `cpt-event-prediction-algo-baseline-prediction-markov`

**Input**: Training prefix-target examples, complete vocabulary, and Laplace smoothing value alpha greater than zero.

**Output**: Conditional next-event probability table keyed by the last observed event.

**Steps**:
1. [x] - `p1` - Count each training target by its immediately preceding observable event - `inst-base-markov-count`
2. [x] - `p1` - Add alpha to every possible successor count for every vocabulary event - `inst-base-markov-smooth`
3. [x] - `p1` - Normalize each predecessor row and verify its probabilities sum to one - `inst-base-markov-normalize`
4. [x] - `p1` - **IF** prediction receives an unsupported predecessor, use the smoothed global distribution - `inst-base-markov-fallback`
5. [x] - `p1` - **RETURN** conditional predictor and metadata - `inst-base-markov-result`

## 4. States (CDSL)

### Stateless Fitted Baselines

No entity lifecycle state machine is required. Baseline models are immutable after fitting and have only validated or rejected artifact status.

## 5. Definitions of Done

### Leakage-Safe Sequence Split

- [x] `p1` - **ID**: `cpt-event-prediction-dod-baseline-prediction-sequence-split`

The system **MUST** generate reproducible disjoint 70/15/15 sequence partitions before creating prefix examples.

**Implements**:
- `cpt-event-prediction-algo-baseline-prediction-build-examples`

**Constraints**: `cpt-event-prediction-constraint-evaluation-protocol`

**Touches**:
- Files: split manifest
- Entities: EventSequence, PrefixExample, SplitManifest

### Two Reference Predictors

- [x] `p1` - **ID**: `cpt-event-prediction-dod-baseline-prediction-models`

The system **MUST** fit and apply both baseline probability predictors from training data only through the shared predictor protocol.

**Implements**:
- `cpt-event-prediction-flow-baseline-prediction-fit-apply`
- `cpt-event-prediction-algo-baseline-prediction-global-frequency`
- `cpt-event-prediction-algo-baseline-prediction-markov`

**Constraints**: `cpt-event-prediction-constraint-cpu-local`

**Touches**:
- Entities: PrefixExample, Prediction

## 6. Acceptance Criteria

- [ ] No sequence identifier appears in more than one split, and split ratios match documented rounding rules.
- [ ] Prefix examples contain only earlier events and exactly one subsequent target.
- [ ] Global-frequency probabilities match training target frequencies and sum to one.
- [ ] Every Markov row assigns non-zero probability to every class and sums to one.
- [ ] Repeated fitting on identical training examples produces identical metadata and predictions.
- [ ] Validation and test targets cannot affect baseline counts.
