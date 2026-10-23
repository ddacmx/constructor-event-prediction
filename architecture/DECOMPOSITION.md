---
description: Feature boundaries and dependency order for implementing the event-prediction system.
---

# Decomposition: Software Delivery Event Prediction

Feature boundaries and dependency order for implementing the approved event-generation, prediction, evaluation, and demonstration design.

<!-- toc -->

- [1. Overview](#1-overview)
- [2. Entries](#2-entries)
  - [2.1 Synthetic Data Generation - HIGH](#21-synthetic-data-generation---high)
  - [2.2 Baseline Prediction - HIGH](#22-baseline-prediction---high)
  - [2.3 Machine Learning Prediction - HIGH](#23-machine-learning-prediction---high)
  - [2.4 Benchmarking and Evaluation - HIGH](#24-benchmarking-and-evaluation---high)
  - [2.5 Reproducible CLI Demonstration - MEDIUM](#25-reproducible-cli-demonstration---medium)
- [3. Feature Dependencies](#3-feature-dependencies)

<!-- /toc -->

## 1. Overview

The design is decomposed into five cohesive features that follow the data lifecycle: generate observable sequences, establish baselines, train the learned predictor, evaluate all models, and expose the workflow through a reproducible CLI. Data generation is the foundation. Baselines and the learned predictor can then be developed independently. Benchmarking depends on both prediction paths, and the CLI demonstration integrates the completed workflow.

Every PRD functional and non-functional requirement is assigned to at least one feature. Shared design principles and constraints are listed where they materially govern a feature; component ownership remains exclusive except that the CLI demonstration integrates existing component interfaces without taking ownership of their logic.

## 2. Entries

**Overall implementation status:**

- [ ] `p1` - **ID**: `cpt-event-prediction-status-overall`

### 2.1 [Synthetic Data Generation](features/synthetic-data-generation.md) - HIGH

- [ ] `p1` - **ID**: `cpt-event-prediction-feature-synthetic-data-generation`

- **Purpose**: Define the event domain and produce deterministic JSONL sequences from a history-aware probabilistic simulator while preserving a strict observation boundary.
- **Depends On**: None
- **Scope**:
  - Event catalog, transition configuration, internal simulator lifecycle, seeded sampling, validation, and dataset serialization
  - Dataset provenance and generation diagnostics
- **Out of scope**:
  - Model fitting, metric computation, or benchmark claims
  - Production telemetry ingestion
- **Requirements Covered**:
  - [ ] `p1` - `cpt-event-prediction-fr-event-taxonomy`
  - [ ] `p1` - `cpt-event-prediction-fr-generate-sequences`
  - [ ] `p1` - `cpt-event-prediction-fr-history-dependent-transitions`
  - [ ] `p1` - `cpt-event-prediction-fr-jsonl-dataset`
  - [ ] `p1` - `cpt-event-prediction-fr-prevent-leakage`
  - [ ] `p1` - `cpt-event-prediction-nfr-reproducibility`
  - [ ] `p2` - `cpt-event-prediction-nfr-local-runtime`
- **Design Principles Covered**:
  - [ ] `p1` - `cpt-event-prediction-principle-observable-boundary`
  - [ ] `p2` - `cpt-event-prediction-principle-immutable-artifacts`
- **Design Constraints Covered**:
  - [ ] `p1` - `cpt-event-prediction-constraint-cpu-local`
  - [ ] `p1` - `cpt-event-prediction-constraint-schedule`
- **Domain Model Entities**:
  - EventType, ObservedEvent, EventSequence, SimulatorContext
- **Design Components**:
  - [ ] `p1` - `cpt-event-prediction-component-simulator`
  - [ ] `p1` - `cpt-event-prediction-component-artifact-io`
- **API**:
  - `generate` command contract
  - Versioned dataset JSONL and manifest formats
- **Sequences**:
  - `cpt-event-prediction-seq-generate-dataset`
- **Data**:
  - Observable JSONL sequence records and generation manifest

### 2.2 [Baseline Prediction](features/baseline-prediction.md) - HIGH

- [ ] `p1` - **ID**: `cpt-event-prediction-feature-baseline-prediction`

- **Purpose**: Implement transparent global-frequency and smoothed first-order transition references on leakage-safe training examples.
- **Depends On**: `cpt-event-prediction-feature-synthetic-data-generation`
- **Scope**:
  - Sequence partitioning boundary, prefix-target derivation, shared predictor interface, and baseline probability estimates
- **Out of scope**:
  - Learned historical features, aggregate statistical claims, and CLI packaging
- **Requirements Covered**:
  - [ ] `p1` - `cpt-event-prediction-fr-baseline-models`
  - [ ] `p1` - `cpt-event-prediction-fr-sequence-split`
  - [ ] `p1` - `cpt-event-prediction-fr-prevent-leakage`
  - [ ] `p1` - `cpt-event-prediction-nfr-reproducibility`
  - [ ] `p1` - `cpt-event-prediction-nfr-testability`
- **Design Principles Covered**:
  - [ ] `p1` - `cpt-event-prediction-principle-baselines-first`
  - [ ] `p1` - `cpt-event-prediction-principle-sequence-unit`
- **Design Constraints Covered**:
  - [ ] `p1` - `cpt-event-prediction-constraint-cpu-local`
  - [ ] `p1` - `cpt-event-prediction-constraint-evaluation-protocol`
- **Domain Model Entities**:
  - EventSequence, PrefixExample, SplitManifest, Prediction
- **Design Components**:
  - [ ] `p1` - `cpt-event-prediction-component-splitter`
  - [ ] `p1` - `cpt-event-prediction-component-baselines`
- **API**:
  - Shared fit and probability-predict protocol
- **Sequences**:
  - `cpt-event-prediction-seq-run-benchmark`
- **Data**:
  - Split manifest and aligned baseline predictions

### 2.3 [Machine Learning Prediction](features/ml-prediction.md) - HIGH

- [ ] `p1` - **ID**: `cpt-event-prediction-feature-ml-prediction`

- **Purpose**: Transform observable prefixes into sparse historical features and train a validation-selected multinomial logistic-regression predictor.
- **Depends On**: `cpt-event-prediction-feature-synthetic-data-generation`
- **Scope**:
  - Historical feature extraction, training-only fitting, validation selection, probability prediction, and model metadata
- **Out of scope**:
  - Neural models, test-based selection, or simulator state access
- **Requirements Covered**:
  - [ ] `p1` - `cpt-event-prediction-fr-logistic-regression`
  - [ ] `p1` - `cpt-event-prediction-fr-sequence-split`
  - [ ] `p1` - `cpt-event-prediction-fr-prevent-leakage`
  - [ ] `p1` - `cpt-event-prediction-nfr-reproducibility`
  - [ ] `p1` - `cpt-event-prediction-nfr-testability`
  - [ ] `p2` - `cpt-event-prediction-nfr-local-runtime`
- **Design Principles Covered**:
  - [ ] `p1` - `cpt-event-prediction-principle-observable-boundary`
  - [ ] `p1` - `cpt-event-prediction-principle-baselines-first`
- **Design Constraints Covered**:
  - [ ] `p1` - `cpt-event-prediction-constraint-cpu-local`
  - [ ] `p1` - `cpt-event-prediction-constraint-evaluation-protocol`
- **Domain Model Entities**:
  - PrefixExample, Prediction
- **Design Components**:
  - [ ] `p1` - `cpt-event-prediction-component-feature-pipeline`
  - [ ] `p1` - `cpt-event-prediction-component-logistic-model`
- **API**:
  - Shared fit and probability-predict protocol
  - Versioned trusted model artifact and metadata
- **Sequences**:
  - `cpt-event-prediction-seq-predict-prefix`
- **Data**:
  - Feature vocabulary and model metadata

### 2.4 [Benchmarking and Evaluation](features/benchmarking-evaluation.md) - HIGH

- [ ] `p1` - **ID**: `cpt-event-prediction-feature-benchmarking-evaluation`

- **Purpose**: Run five isolated experiments, compare all predictors on aligned test examples, and produce statistically responsible result and error reports.
- **Depends On**: `cpt-event-prediction-feature-baseline-prediction`, `cpt-event-prediction-feature-ml-prediction`
- **Scope**:
  - Metrics, confusion matrices, clustered intervals, paired comparisons, aggregation, error categories, and result schemas
- **Out of scope**:
  - Model implementation changes based on test outcomes or claims about real-world performance
- **Requirements Covered**:
  - [ ] `p1` - `cpt-event-prediction-fr-multi-seed-evaluation`
  - [ ] `p1` - `cpt-event-prediction-fr-statistical-reporting`
  - [ ] `p1` - `cpt-event-prediction-fr-sequence-split`
  - [ ] `p1` - `cpt-event-prediction-nfr-statistical-rigor`
  - [ ] `p1` - `cpt-event-prediction-nfr-reproducibility`
  - [ ] `p1` - `cpt-event-prediction-nfr-testability`
- **Design Principles Covered**:
  - [ ] `p1` - `cpt-event-prediction-principle-baselines-first`
  - [ ] `p1` - `cpt-event-prediction-principle-sequence-unit`
  - [ ] `p2` - `cpt-event-prediction-principle-immutable-artifacts`
- **Design Constraints Covered**:
  - [ ] `p1` - `cpt-event-prediction-constraint-evaluation-protocol`
  - [ ] `p1` - `cpt-event-prediction-constraint-schedule`
- **Domain Model Entities**:
  - SplitManifest, Prediction, BenchmarkRun
- **Design Components**:
  - [ ] `p1` - `cpt-event-prediction-component-evaluator`
- **API**:
  - Versioned per-seed and aggregate benchmark result formats
- **Sequences**:
  - `cpt-event-prediction-seq-run-benchmark`
- **Data**:
  - Metrics, confusion matrices, intervals, paired comparisons, and error records

### 2.5 [Reproducible CLI Demonstration](features/reproducible-cli-demo.md) - MEDIUM

- [ ] `p2` - **ID**: `cpt-event-prediction-feature-reproducible-cli-demo`

- **Purpose**: Integrate the completed pipeline behind stable commands and provide the reviewer-facing reproducibility and Fabric-use path.
- **Depends On**: `cpt-event-prediction-feature-synthetic-data-generation`, `cpt-event-prediction-feature-baseline-prediction`, `cpt-event-prediction-feature-ml-prediction`, `cpt-event-prediction-feature-benchmarking-evaluation`
- **Scope**:
  - Command validation, orchestration, exit behavior, documentation, smoke test, and prediction demonstration
- **Out of scope**:
  - Web UI, hosted API, package registry publication, or GitHub publication automation
- **Requirements Covered**:
  - [ ] `p1` - `cpt-event-prediction-fr-cli-demo`
  - [ ] `p1` - `cpt-event-prediction-fr-fabric-traceability`
  - [ ] `p1` - `cpt-event-prediction-nfr-review-usability`
  - [ ] `p1` - `cpt-event-prediction-nfr-testability`
  - [ ] `p2` - `cpt-event-prediction-nfr-local-runtime`
- **Design Principles Covered**:
  - [ ] `p2` - `cpt-event-prediction-principle-immutable-artifacts`
- **Design Constraints Covered**:
  - [ ] `p1` - `cpt-event-prediction-constraint-cpu-local`
  - [ ] `p1` - `cpt-event-prediction-constraint-schedule`
- **Domain Model Entities**:
  - EventSequence, PrefixExample, BenchmarkRun
- **Design Components**:
  - [ ] `p1` - `cpt-event-prediction-component-cli`
- **API**:
  - `generate`, `benchmark`, `predict`, and `verify` command contracts
- **Sequences**:
  - `cpt-event-prediction-seq-generate-dataset`
  - `cpt-event-prediction-seq-run-benchmark`
  - `cpt-event-prediction-seq-predict-prefix`
- **Data**:
  - Console summaries and links to machine-readable artifacts

## 3. Feature Dependencies

```text
cpt-event-prediction-feature-synthetic-data-generation
    |-- cpt-event-prediction-feature-baseline-prediction --|
    |-- cpt-event-prediction-feature-ml-prediction --------|-- cpt-event-prediction-feature-benchmarking-evaluation
    |--------------------------------------------------------------------------------------------|
                                                                                                 `-- cpt-event-prediction-feature-reproducible-cli-demo
```

**Dependency Rationale**:

- `cpt-event-prediction-feature-baseline-prediction` and `cpt-event-prediction-feature-ml-prediction` require `cpt-event-prediction-feature-synthetic-data-generation` because both consume the event catalog and observable sequence contract.
- `cpt-event-prediction-feature-baseline-prediction` and `cpt-event-prediction-feature-ml-prediction` are independent after the shared split and prefix contracts are fixed.
- `cpt-event-prediction-feature-benchmarking-evaluation` requires both model features so paired comparisons use aligned predictions.
- `cpt-event-prediction-feature-reproducible-cli-demo` integrates all completed workflows and therefore comes last.
