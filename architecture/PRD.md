---
description: Requirements for reproducible synthetic software-delivery next-event prediction and evaluation.
---

# PRD - Software Delivery Event Prediction

Requirements for a reproducible local system that simulates software-delivery histories, predicts the next observable event, and evaluates results without leakage.

<!-- toc -->

- [1. Overview](#1-overview)
  - [1.1 Purpose](#11-purpose)
  - [1.2 Background / Problem Statement](#12-background--problem-statement)
  - [1.3 Goals (Business Outcomes)](#13-goals-business-outcomes)
  - [1.4 Glossary](#14-glossary)
- [2. Actors](#2-actors)
  - [2.1 Human Actors](#21-human-actors)
  - [2.2 System Actors](#22-system-actors)
- [3. Operational Concept & Environment](#3-operational-concept--environment)
  - [3.1 Module-Specific Environment Constraints](#31-module-specific-environment-constraints)
- [4. Scope](#4-scope)
  - [4.1 In Scope](#41-in-scope)
  - [4.2 Out of Scope](#42-out-of-scope)
- [5. Functional Requirements](#5-functional-requirements)
  - [5.1 Event Domain and Data Generation](#51-event-domain-and-data-generation)
  - [5.2 Prediction](#52-prediction)
  - [5.3 Evaluation and Demonstration](#53-evaluation-and-demonstration)
- [6. Non-Functional Requirements](#6-non-functional-requirements)
  - [6.1 NFR Inclusions](#61-nfr-inclusions)
  - [6.2 NFR Exclusions](#62-nfr-exclusions)
- [7. Public Library Interfaces](#7-public-library-interfaces)
  - [7.1 Public API Surface](#71-public-api-surface)
  - [7.2 External Integration Contracts](#72-external-integration-contracts)
- [8. Use Cases](#8-use-cases)
  - [8.1 Core Use Cases](#81-core-use-cases)
- [9. Acceptance Criteria](#9-acceptance-criteria)
- [10. Dependencies](#10-dependencies)
- [11. Assumptions](#11-assumptions)
- [12. Risks](#12-risks)

<!-- /toc -->

## 1. Overview

### 1.1 Purpose

Software Delivery Event Prediction is a reproducible learning system that creates synthetic sequences of software architecture and delivery events and predicts the next observable event from a sequence prefix. It exists to demonstrate that meaningful temporal structure can be simulated, learned, and evaluated without relying on proprietary production data.

The system must support transparent baselines, a trained multiclass predictor, statistical comparison, and a command-line demonstration that a reviewer can reproduce locally.

### 1.2 Background / Problem Statement

The employer assignment requires a defined event sequence, a separate meaningful synthetic-data generator, a next-event prediction program built with Constructor Fabric, and a benchmark with statistical evaluation. No production event log is supplied, so the project must create data whose transition patterns are neither random noise nor a deterministic chain.

The main product risk is a demonstration that appears predictive only because future or hidden simulator information leaked into model inputs. A second risk is an evaluation that treats correlated prefixes as independent splits or selects a model using the test set. The product therefore requires observable-only records, sequence-level isolation, declared seeds, and baseline comparisons.

### 1.3 Goals (Business Outcomes)

- A reviewer can generate 10,000 synthetic sequences and reproduce the same dataset from the same configuration and seed.
- A reviewer can compare the learned predictor with two transparent baselines across five independent experimental seeds.
- A reviewer can run one documented command-line demonstration and inspect machine-readable metrics and human-readable error analysis.
- Every implementation feature remains traceable to requirements and design decisions through Constructor Fabric CPT identifiers.

### 1.4 Glossary

| Term | Definition |
|------|------------|
| Event | An observable software architecture or delivery lifecycle occurrence represented by a stable event type and short description. |
| Sequence | An ordered list of observable events from one synthetic delivery episode. |
| Prefix | All observed events before the event being predicted. |
| Hidden simulator state | Internal generation context used to vary transition probabilities; it is never exposed to model features. |
| Experimental seed | A seed controlling a complete generate, split, train, and evaluate repetition. |
| Baseline | A transparent reference predictor used to establish whether the learned model adds value. |

## 2. Actors

### 2.1 Human Actors

#### ML Engineer

**ID**: `cpt-event-prediction-actor-ml-engineer`

**Role**: Configures generation and experiments, runs benchmarks, and investigates errors.
**Needs**: Deterministic commands, explicit configurations, testable outputs, and diagnostic reports.

#### Technical Reviewer

**ID**: `cpt-event-prediction-actor-technical-reviewer`

**Role**: Evaluates the hiring assignment and portfolio project on an independent local machine.
**Needs**: A concise setup path, evidence that predictions are real, and traceability from requirements to implementation.

### 2.2 System Actors

#### Local Automation Runner

**ID**: `cpt-event-prediction-actor-automation-runner`

**Role**: Executes deterministic generation, tests, training, evaluation, and demonstration commands in local or CI environments.

## 3. Operational Concept & Environment

### 3.1 Module-Specific Environment Constraints

- The complete workflow MUST run on a CPU-only Windows or POSIX development machine.
- The workflow MUST not require cloud services, a database, a browser interface, or proprietary data.
- The benchmark target is 10,000 sequences; scale beyond that target is not a Phase 2 requirement.

## 4. Scope

### 4.1 In Scope

- A documented taxonomy of observable software architecture and delivery events.
- A configurable, history-dependent probabilistic sequence generator.
- JSONL dataset production with deterministic seeds and explicit configuration provenance.
- Global-frequency and Laplace-smoothed first-order Markov baselines.
- A multinomial logistic-regression predictor using observable prefix history.
- Sequence-level 70/15/15 train, validation, and test splits.
- Five-seed benchmarking with accuracy, macro-F1, confusion matrices, uncertainty intervals, paired comparisons, and error analysis.
- Automated tests, machine-readable result files, and a working command-line demonstration.
- Constructor Fabric documentation, identifiers, validation, and traceability.

### 4.2 Out of Scope

- Production telemetry ingestion or claims about production performance.
- Deep learning, GPU support, online learning, or neural sequence models.
- A web frontend, hosted API, database service, cloud deployment, or distributed training.
- Prediction of timestamps, durations, event text, or multiple future steps.
- Causal inference or interpretation of the synthetic process as real organizational behavior.

## 5. Functional Requirements

### 5.1 Event Domain and Data Generation

#### Define an Observable Event Taxonomy

- [ ] `p1` - **ID**: `cpt-event-prediction-fr-event-taxonomy`

The system **MUST** define stable event types and short descriptions covering architecture changes, dependency work, compatibility checks, CI outcomes, fixes, integration tests, review approval, deployment, and rollback.

**Rationale**: A clear vocabulary makes generated sequences understandable and predictions auditable.

**Actors**: `cpt-event-prediction-actor-ml-engineer`, `cpt-event-prediction-actor-technical-reviewer`

#### Generate Meaningful Event Sequences

- [ ] `p1` - **ID**: `cpt-event-prediction-fr-generate-sequences`

The system **MUST** generate a configurable number of variable-length event sequences from explicit probabilistic transition rules.

**Rationale**: The assignment requires a separate generator that produces meaningful synthetic sequences.

**Actors**: `cpt-event-prediction-actor-ml-engineer`, `cpt-event-prediction-actor-automation-runner`

#### Apply History-Dependent Behavior

- [ ] `p1` - **ID**: `cpt-event-prediction-fr-history-dependent-transitions`

The generator **MUST** vary transition probabilities using prior observable events and internal lifecycle context so that the event process is non-deterministic and not reducible to one fixed chain.

**Rationale**: Meaningful next-event prediction requires learnable but imperfect temporal structure.

**Actors**: `cpt-event-prediction-actor-ml-engineer`

#### Produce Portable Dataset Records

- [ ] `p1` - **ID**: `cpt-event-prediction-fr-jsonl-dataset`

The system **MUST** write one self-contained sequence per JSONL record with schema version, sequence identifier, ordered observable events, and non-feature provenance metadata.

**Rationale**: Portable records make inspection, streaming, and independent reproduction straightforward.

**Actors**: `cpt-event-prediction-actor-ml-engineer`, `cpt-event-prediction-actor-technical-reviewer`

#### Prevent Information Leakage

- [ ] `p1` - **ID**: `cpt-event-prediction-fr-prevent-leakage`

The system **MUST** exclude future events, split membership, hidden simulator state, and target-derived values from prediction features.

**Rationale**: Leakage would invalidate the benchmark.

**Actors**: `cpt-event-prediction-actor-technical-reviewer`, `cpt-event-prediction-actor-automation-runner`

### 5.2 Prediction

#### Provide Transparent Baselines

- [ ] `p1` - **ID**: `cpt-event-prediction-fr-baseline-models`

The system **MUST** provide a global-frequency predictor and a Laplace-smoothed first-order Markov predictor.

**Rationale**: Baselines establish whether historical features add predictive value.

**Actors**: `cpt-event-prediction-actor-ml-engineer`, `cpt-event-prediction-actor-technical-reviewer`

#### Train a Multiclass Historical Predictor

- [ ] `p1` - **ID**: `cpt-event-prediction-fr-logistic-regression`

The system **MUST** train a multinomial logistic-regression model from observable historical events and prefix-derived features and return one next-event class with class probabilities.

**Rationale**: A simple interpretable learned model is appropriate for a small reproducible assignment while supporting richer history than a first-order baseline.

**Actors**: `cpt-event-prediction-actor-ml-engineer`, `cpt-event-prediction-actor-technical-reviewer`

### 5.3 Evaluation and Demonstration

#### Isolate Complete Sequences Across Splits

- [ ] `p1` - **ID**: `cpt-event-prediction-fr-sequence-split`

The system **MUST** assign complete sequences, not individual prefixes, to 70% training, 15% validation, and 15% test partitions.

**Rationale**: Prefixes from the same sequence are correlated and must not cross evaluation boundaries.

**Actors**: `cpt-event-prediction-actor-automation-runner`

#### Evaluate Across Independent Seeds

- [ ] `p1` - **ID**: `cpt-event-prediction-fr-multi-seed-evaluation`

The benchmark **MUST** run five declared experimental seeds, use validation results for model selection, keep test labels unavailable to selection, and report accuracy and macro-F1 for every model.

**Rationale**: Independent repetitions expose sensitivity to generated data and split randomness.

**Actors**: `cpt-event-prediction-actor-ml-engineer`, `cpt-event-prediction-actor-technical-reviewer`

#### Report Statistical and Diagnostic Evidence

- [ ] `p1` - **ID**: `cpt-event-prediction-fr-statistical-reporting`

The system **MUST** report per-seed results, mean and standard deviation, confidence intervals where appropriate, paired comparisons against both baselines, aggregate confusion matrices, and categorized error analysis.

**Rationale**: A point estimate alone does not demonstrate robust predictive value.

**Actors**: `cpt-event-prediction-actor-technical-reviewer`

#### Provide a Reproducible Demonstration

- [ ] `p1` - **ID**: `cpt-event-prediction-fr-cli-demo`

The system **MUST** provide documented command-line execution for dataset generation, benchmarking, single-prefix prediction, and automated verification.

**Rationale**: The repository must be self-contained and independently runnable.

**Actors**: `cpt-event-prediction-actor-technical-reviewer`, `cpt-event-prediction-actor-automation-runner`

#### Preserve Fabric Traceability

- [ ] `p1` - **ID**: `cpt-event-prediction-fr-fabric-traceability`

The repository **MUST** preserve Constructor Fabric links among requirements, decisions, design components, feature specifications, and later code markers, and include a short account of actual Fabric use.

**Rationale**: The assignment explicitly requires Constructor Fabric usage.

**Actors**: `cpt-event-prediction-actor-technical-reviewer`

## 6. Non-Functional Requirements

### 6.1 NFR Inclusions

#### Deterministic Reproduction

- [ ] `p1` - **ID**: `cpt-event-prediction-nfr-reproducibility`

Identical versioned configuration, dependency lock, and seed **MUST** reproduce byte-equivalent dataset records and equivalent split assignments and metrics within documented numeric tolerances.

**Threshold**: Dataset JSONL hashes and split manifests match exactly; reported floating-point metrics match to at least six decimal places on the supported environment.

**Rationale**: Reviewers need evidence they can reproduce rather than trust.

#### Statistical Rigor

- [ ] `p1` - **ID**: `cpt-event-prediction-nfr-statistical-rigor`

All uncertainty and paired comparisons **MUST** respect the sequence as the independent sampling unit.

**Threshold**: Five complete experimental repetitions; 95% sequence-clustered intervals for reported test differences where the sample supports estimation.

**Rationale**: Prefix-level resampling would overstate precision.

#### Local Runtime

- [ ] `p2` - **ID**: `cpt-event-prediction-nfr-local-runtime`

The standard 10,000-sequence benchmark **MUST** complete on a CPU-only development machine without an external service.

**Threshold**: Target completion within 15 minutes and peak memory below 4 GB on the documented reference machine.

**Rationale**: The project must be practical for hiring review and portfolio use.

#### Automated Verifiability

- [ ] `p1` - **ID**: `cpt-event-prediction-nfr-testability`

Critical generation, leakage prevention, splitting, model, metric, serialization, and CLI behavior **MUST** be covered by automated tests.

**Threshold**: All documented test commands pass; critical modules target at least 90% statement coverage.

**Rationale**: Automated evidence reduces ambiguity in a time-limited review.

#### Review Usability

- [ ] `p2` - **ID**: `cpt-event-prediction-nfr-review-usability`

A technical reviewer **MUST** be able to reach a representative prediction demonstration after following the repository instructions without undocumented manual steps.

**Threshold**: One setup section and one demonstration command; machine-readable results are retained separately from console summaries.

**Rationale**: The deliverable is evaluated as a self-contained repository.

### 6.2 NFR Exclusions

- High availability: Not applicable because the product is an offline local experiment, not a service.
- Authentication and authorization: Not applicable because the product has no network or multi-user interface.
- Personal-data privacy: Not applicable because the generator emits synthetic lifecycle events and processes no personal data.
- Accessibility and internationalization: Not applicable because the only user interface is a technical command line with English machine-readable outputs.
- Disaster recovery and business continuity: Not applicable because all results are reproducible from versioned configuration and seeds.

## 7. Public Library Interfaces

### 7.1 Public API Surface

#### Command-Line Interface

- [ ] `p1` - **ID**: `cpt-event-prediction-interface-cli`

**Type**: CLI

**Stability**: stable for the submitted repository

**Description**: Provides generation, benchmark, prediction, and verification entry points with explicit inputs, outputs, and seeds.

**Breaking Change Policy**: Any breaking command or output change requires a documented schema or command version change.

#### Dataset and Benchmark Files

- [ ] `p1` - **ID**: `cpt-event-prediction-interface-artifacts`

**Type**: Versioned JSONL and JSON data formats

**Stability**: stable for the submitted repository

**Description**: Defines portable sequence records, split manifests, model metadata, and benchmark result records.

**Breaking Change Policy**: Readers reject unsupported schema versions with a clear error.

### 7.2 External Integration Contracts

#### Local File Contract

- [ ] `p2` - **ID**: `cpt-event-prediction-contract-local-files`

**Direction**: provided by library

**Protocol/Format**: UTF-8 JSONL and JSON

**Compatibility**: Schema-versioned; the submitted major version remains readable throughout the project.

## 8. Use Cases

### 8.1 Core Use Cases

#### Generate a Reproducible Dataset

- [ ] `p1` - **ID**: `cpt-event-prediction-usecase-generate-dataset`

**Actor**: `cpt-event-prediction-actor-ml-engineer`

**Preconditions**:
- A valid generation configuration and seed are available.

**Main Flow**:
1. The engineer requests 10,000 sequences.
2. The system validates configuration, generates observable event histories, and writes JSONL plus provenance.
3. A repeated run with the same inputs produces the same dataset hash.

**Postconditions**:
- A versioned, inspectable dataset is ready for splitting.

**Alternative Flows**:
- **Invalid transition configuration**: The system rejects generation before writing a partial final dataset.

#### Benchmark Next-Event Predictors

- [ ] `p1` - **ID**: `cpt-event-prediction-usecase-run-benchmark`

**Actor**: `cpt-event-prediction-actor-automation-runner`

**Preconditions**:
- Generation and experiment configurations are valid.

**Main Flow**:
1. The runner performs five seeded generate, split, train, select, and evaluate repetitions.
2. The system evaluates the frequency, Markov, and logistic-regression predictors on identical test examples within each seed.
3. The system writes metrics, intervals, paired comparisons, confusion matrices, and errors.

**Postconditions**:
- Machine-readable and human-readable benchmark evidence is available without altering test-based model selection.

**Alternative Flows**:
- **A seed fails**: The benchmark exits non-zero and records the failed seed without reporting a complete aggregate.

#### Predict the Next Event

- [ ] `p1` - **ID**: `cpt-event-prediction-usecase-predict-prefix`

**Actor**: `cpt-event-prediction-actor-technical-reviewer`

**Preconditions**:
- A validated trained model and event prefix are available.

**Main Flow**:
1. The reviewer submits an observable event prefix.
2. The system validates event types, derives prefix features, and returns the most likely next event and probabilities.
3. The reviewer compares the result with the documented event vocabulary.

**Postconditions**:
- The prediction can be inspected without access to simulator internals.

**Alternative Flows**:
- **Unknown event**: The system returns a clear validation error and no prediction.

## 9. Acceptance Criteria

- [ ] A fresh environment can generate exactly 10,000 valid JSONL sequence records from a documented seed.
- [ ] The generator exhibits at least one history-dependent probability change and produces more than one valid successor for every nonterminal lifecycle situation covered by the configured process.
- [ ] Leakage tests demonstrate that only prefix-observable fields reach feature extraction.
- [ ] All three predictors run on identical sequence-level splits for five seeds and produce accuracy, macro-F1, and confusion matrices.
- [ ] The benchmark reports means, standard deviations, sequence-clustered uncertainty, paired baseline comparisons, and error categories without using test results for model selection.
- [ ] A documented CLI command returns a next-event prediction from a supplied prefix.
- [ ] Constructor Studio validation and CPT traceability checks pass for the registered artifacts before implementation begins.

## 10. Dependencies

| Dependency | Description | Criticality |
|------------|-------------|-------------|
| Constructor Studio and SDLC kit | Artifact workflow, validation, and traceability | p1 |
| Python runtime | Local generation, training, evaluation, and CLI execution | p1 |
| Classical machine-learning library | Multiclass logistic regression and standard metrics | p1 |

## 11. Assumptions

- Synthetic event behavior is intended to be plausible and learnable, not a calibrated model of any real organization.
- The reviewer has a CPU-only machine and can install project-scoped Python dependencies.
- Ten thousand sequences are sufficient for the assignment demonstration; the implementation will expose the count as configuration.
- Event descriptions are fixed human-readable labels rather than generated natural language.

## 12. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Generator is too easy or deterministic | Inflated results that do not demonstrate modeling skill | Require stochastic alternatives, history modifiers, entropy diagnostics, and baseline comparison. |
| Generator is too noisy | All models approach chance and the demo is uninformative | Calibrate transitions on validation-only pilot runs and freeze the final configuration before test evaluation. |
| Hidden state leaks into features | Invalid benchmark | Maintain separate simulator and observation records and assert an explicit feature allowlist. |
| Prefixes leak across splits | Optimistic test metrics | Split sequence identifiers first, then derive prefixes inside each partition. |
| Small or imbalanced classes destabilize macro-F1 | Misleading aggregate | Apply smoothing, report class support and confusion matrices, and analyze rare-class errors. |
| Schedule exceeds the October 24 target | Incomplete hiring submission | Keep a local CLI architecture and classical models; exclude UI, cloud, databases, and deep learning. |
