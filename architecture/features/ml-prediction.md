---
description: Feature specification for observable historical features and multinomial logistic-regression prediction.
---

# Feature: Machine Learning Prediction

Specification for observable prefix features, validation-only model selection, and reproducible multinomial logistic-regression inference.


<!-- toc -->

- [1. Feature Context](#1-feature-context)
  - [1.1 Overview](#11-overview)
  - [1.2 Purpose](#12-purpose)
  - [1.3 Actors](#13-actors)
  - [1.4 References](#14-references)
- [2. Actor Flows (CDSL)](#2-actor-flows-cdsl)
  - [Train and Select the Learned Predictor](#train-and-select-the-learned-predictor)
  - [Predict from an Observable Prefix](#predict-from-an-observable-prefix)
- [3. Processes / Business Logic (CDSL)](#3-processes--business-logic-cdsl)
  - [Extract Historical Prefix Features](#extract-historical-prefix-features)
  - [Select Logistic Regression Configuration](#select-logistic-regression-configuration)
- [4. States (CDSL)](#4-states-cdsl)
  - [Model Artifact State Machine](#model-artifact-state-machine)
- [5. Definitions of Done](#5-definitions-of-done)
  - [Leakage-Safe Feature Pipeline](#leakage-safe-feature-pipeline)
  - [Validated Multiclass Predictor](#validated-multiclass-predictor)
- [6. Acceptance Criteria](#6-acceptance-criteria)

<!-- /toc -->

- [ ] `p1` - **ID**: `cpt-event-prediction-featstatus-ml-prediction`
## 1. Feature Context

- [ ] `p1` - `cpt-event-prediction-feature-ml-prediction`

### 1.1 Overview

This feature derives sparse historical features from observable prefixes and trains a validation-selected multinomial logistic-regression predictor.

### 1.2 Purpose

The learned model tests whether longer observable history adds predictive value beyond global frequencies and first-order transitions.

**Requirements**: `cpt-event-prediction-fr-logistic-regression`, `cpt-event-prediction-fr-sequence-split`, `cpt-event-prediction-fr-prevent-leakage`, `cpt-event-prediction-nfr-reproducibility`, `cpt-event-prediction-nfr-testability`, `cpt-event-prediction-nfr-local-runtime`

**Principles**: `cpt-event-prediction-principle-observable-boundary`, `cpt-event-prediction-principle-baselines-first`

### 1.3 Actors

| Actor | Role in Feature |
|-------|-----------------|
| `cpt-event-prediction-actor-ml-engineer` | Reviews features, selects the validation objective, and trains the model. |
| `cpt-event-prediction-actor-technical-reviewer` | Supplies an observable prefix and inspects predicted probabilities. |

### 1.4 References

- **PRD**: [PRD.md](../PRD.md)
- **Design**: [DESIGN.md](../DESIGN.md)
- **Dependencies**: `cpt-event-prediction-feature-synthetic-data-generation`

## 2. Actor Flows (CDSL)

**Use cases**: `cpt-event-prediction-usecase-run-benchmark`, `cpt-event-prediction-usecase-predict-prefix`

### Train and Select the Learned Predictor

- [x] `p1` - **ID**: `cpt-event-prediction-flow-ml-prediction-train-select`

**Actor**: `cpt-event-prediction-actor-ml-engineer`

**Success Scenarios**:
- Candidate regularization values are fitted on training data, evaluated on validation macro-F1, and resolved by a deterministic tie-break.

**Error Scenarios**:
- A feature contains a forbidden field, a class is absent from training, or optimization fails to converge within policy.

**Steps**:
1. [x] - `p1` - Load training and validation PrefixExample collections and the fixed event vocabulary - `inst-ml-load-examples`
2. [x] - `p1` - Fit feature vocabulary and numeric transformations using training prefixes only - `inst-ml-fit-features`
3. [x] - `p1` - Transform training and validation prefixes without target or hidden-state fields - `inst-ml-transform`
4. [x] - `p1` - **FOR EACH** declared regularization candidate, fit a deterministic multinomial logistic-regression model - `inst-ml-fit-candidate`
5. [x] - `p1` - Evaluate validation macro-F1 and convergence metadata for each candidate - `inst-ml-validate-candidate`
6. [x] - `p1` - Select the best candidate using score then the declared simpler-model tie-break - `inst-ml-select`
7. [x] - `p1` - Persist the selected pipeline with vocabulary, classes, configuration hashes, and library versions - `inst-ml-persist`
8. [x] - `p1` - **RETURN** the trusted local model artifact and selection report - `inst-ml-train-result`

### Predict from an Observable Prefix

- [x] `p1` - **ID**: `cpt-event-prediction-flow-ml-prediction-predict-prefix`

**Actor**: `cpt-event-prediction-actor-technical-reviewer`

**Success Scenarios**:
- A valid non-empty event prefix returns one top event and a normalized probability for every event class.

**Error Scenarios**:
- Empty prefixes, unknown events, schema mismatches, or untrusted model paths produce a clear error and no prediction.

**Steps**:
1. [x] - `p1` - Load a locally produced model artifact and verify its metadata and schema versions - `inst-ml-load-model`
2. [x] - `p1` - Validate the supplied prefix against the model event catalog - `inst-ml-validate-prefix`
3. [x] - `p1` - Transform the prefix with the fitted feature pipeline - `inst-ml-transform-prefix`
4. [x] - `p1` - Compute multiclass probabilities and choose the stable highest-probability class - `inst-ml-probabilities`
5. [x] - `p1` - **RETURN** the predicted event and ordered probability table - `inst-ml-predict-result`

## 3. Processes / Business Logic (CDSL)

### Extract Historical Prefix Features

- [x] `p1` - **ID**: `cpt-event-prediction-algo-ml-prediction-extract-features`

**Input**: One non-empty ordered prefix containing event types only.

**Output**: Sparse categorical and numeric feature mapping.

**Steps**:
1. [x] - `p1` - Reject any input key outside the observable-prefix allowlist - `inst-ml-feature-allowlist`
2. [x] - `p1` - Encode the last one, two, and three event identities with boundary markers for short prefixes - `inst-ml-feature-lags`
3. [x] - `p1` - Count each observed event and convert counts to prefix proportions - `inst-ml-feature-counts`
4. [x] - `p1` - Count observed adjacent transitions within the prefix - `inst-ml-feature-transitions`
5. [x] - `p1` - Record prefix length, CI-failure count, fix-after-failure indicator, and whether successful test and review events occurred - `inst-ml-feature-context`
6. [x] - `p1` - Record capped recency since failure, fix, integration test, approval, and deployment when observed - `inst-ml-feature-recency`
7. [x] - `p1` - **RETURN** feature mapping without target, split, seed, sequence identity, or simulator context - `inst-ml-feature-result`

### Select Logistic Regression Configuration

- [x] `p1` - **ID**: `cpt-event-prediction-algo-ml-prediction-select-model`

**Input**: Training matrix and labels, validation matrix and labels, declared regularization grid, and model seed.

**Output**: Selected fitted pipeline and complete selection report.

**Steps**:
1. [x] - `p1` - Verify all event classes have training support and matrices contain finite values - `inst-ml-check-training`
2. [x] - `p1` - **FOR EACH** regularization candidate, fit with fixed solver, tolerance, maximum iterations, and seed - `inst-ml-grid-fit`
3. [x] - `p1` - Exclude a candidate that violates the convergence policy and record the reason - `inst-ml-convergence-policy`
4. [x] - `p1` - Rank valid candidates by validation macro-F1, then by stronger regularization, then stable numeric order - `inst-ml-rank`
5. [x] - `p1` - Freeze the selected pipeline before any test prediction - `inst-ml-freeze`
6. [x] - `p1` - **RETURN** selected pipeline and candidate table - `inst-ml-selection-result`

## 4. States (CDSL)

### Model Artifact State Machine

- [x] `p2` - **ID**: `cpt-event-prediction-state-ml-prediction-model-artifact`

**States**: UNFITTED, CANDIDATE, SELECTED, VERIFIED

**Initial State**: UNFITTED

**Transitions**:
1. [x] - `p1` - **FROM** UNFITTED **TO** CANDIDATE **WHEN** a configuration fits on training data - `inst-ml-state-candidate`
2. [x] - `p1` - **FROM** CANDIDATE **TO** SELECTED **WHEN** validation ranking chooses the configuration - `inst-ml-state-selected`
3. [x] - `p1` - **FROM** SELECTED **TO** VERIFIED **WHEN** metadata, serialization round-trip, and prediction checks pass - `inst-ml-state-verified`

## 5. Definitions of Done

### Leakage-Safe Feature Pipeline

- [x] `p1` - **ID**: `cpt-event-prediction-dod-ml-prediction-feature-pipeline`

The system **MUST** derive the declared sparse features exclusively from events before the target and fit transformation state on training data only.

**Implements**:
- `cpt-event-prediction-algo-ml-prediction-extract-features`

**Constraints**: `cpt-event-prediction-constraint-evaluation-protocol`

**Touches**:
- Entities: PrefixExample

### Validated Multiclass Predictor

- [x] `p1` - **ID**: `cpt-event-prediction-dod-ml-prediction-logistic-model`

The system **MUST** select and persist a deterministic multinomial logistic-regression pipeline using validation data without consulting test labels.

**Implements**:
- `cpt-event-prediction-flow-ml-prediction-train-select`
- `cpt-event-prediction-flow-ml-prediction-predict-prefix`
- `cpt-event-prediction-algo-ml-prediction-select-model`

**Constraints**: `cpt-event-prediction-constraint-cpu-local`

**Touches**:
- CLI: `predict`
- Files: trusted model artifact and model metadata
- Entities: Prediction

## 6. Acceptance Criteria

- [ ] A forbidden future, target, split, seed, identifier, or simulator-context field is rejected by the feature boundary.
- [ ] Feature values for a prefix are unchanged when later sequence events are changed.
- [ ] Candidate selection uses validation macro-F1 only and has a deterministic tie-break.
- [ ] Identical inputs, versions, and seeds produce equivalent coefficients and probabilities within documented tolerance.
- [ ] Every prediction returns one probability per event class, values in the closed interval zero to one, and a sum of one within tolerance.
- [ ] Model metadata records event catalog, feature vocabulary, selected configuration, seeds, dependency versions, and input hashes.
