---
description: Feature specification for five-seed metrics, clustered uncertainty, paired comparisons, and error analysis.
---

# Feature: Benchmarking and Evaluation

Specification for five-seed model comparison, cluster-aware uncertainty, paired tests, confusion matrices, and error analysis.


<!-- toc -->

- [1. Feature Context](#1-feature-context)
  - [1.1 Overview](#11-overview)
  - [1.2 Purpose](#12-purpose)
  - [1.3 Actors](#13-actors)
  - [1.4 References](#14-references)
- [2. Actor Flows (CDSL)](#2-actor-flows-cdsl)
  - [Run the Final Benchmark](#run-the-final-benchmark)
- [3. Processes / Business Logic (CDSL)](#3-processes--business-logic-cdsl)
  - [Compute Per-Seed Evaluation](#compute-per-seed-evaluation)
  - [Estimate Sequence-Clustered Uncertainty](#estimate-sequence-clustered-uncertainty)
  - [Analyze Prediction Errors](#analyze-prediction-errors)
- [4. States (CDSL)](#4-states-cdsl)
  - [Benchmark Run State Machine](#benchmark-run-state-machine)
- [5. Definitions of Done](#5-definitions-of-done)
  - [Complete Statistical Benchmark](#complete-statistical-benchmark)
  - [Reproducible Error Analysis](#reproducible-error-analysis)
- [6. Acceptance Criteria](#6-acceptance-criteria)

<!-- /toc -->

- [ ] `p1` - **ID**: `cpt-event-prediction-featstatus-benchmarking-evaluation`
## 1. Feature Context

- [ ] `p1` - `cpt-event-prediction-feature-benchmarking-evaluation`

### 1.1 Overview

This feature runs five end-to-end experiments and produces aligned model metrics, sequence-clustered uncertainty, paired accuracy and macro-F1 baseline comparisons, fixed-order per-seed and elementwise-summed aggregate confusion matrices and supports, and error analysis.

**Completion evidence**: The published pointer [`artifacts/final/benchmark.json`](../../artifacts/final/benchmark.json) records `COMPLETE`; its hash-bound run includes the machine-readable aggregate and generated [`evaluation-report.md`](../../artifacts/final/runs/7b9514e839241507f358/evaluation-report.md), and strict artifact verification independently accepted all 37 artifacts.

### 1.2 Purpose

The benchmark demonstrates what the models predict, how correctness is measured, and how stable any advantage is under independent generation and split seeds.

**Requirements**: `cpt-event-prediction-fr-multi-seed-evaluation`, `cpt-event-prediction-fr-statistical-reporting`, `cpt-event-prediction-fr-sequence-split`, `cpt-event-prediction-nfr-statistical-rigor`, `cpt-event-prediction-nfr-reproducibility`, `cpt-event-prediction-nfr-testability`

**Principles**: `cpt-event-prediction-principle-baselines-first`, `cpt-event-prediction-principle-sequence-unit`, `cpt-event-prediction-principle-immutable-artifacts`

### 1.3 Actors

| Actor | Role in Feature |
|-------|-----------------|
| `cpt-event-prediction-actor-ml-engineer` | Configures experiments and investigates errors. |
| `cpt-event-prediction-actor-technical-reviewer` | Interprets metrics, comparisons, uncertainty, and limitations. |
| `cpt-event-prediction-actor-automation-runner` | Executes all seeds and fails incomplete runs. |

### 1.4 References

- **PRD**: [PRD.md](../PRD.md)
- **Design**: [DESIGN.md](../DESIGN.md)
- **Dependencies**: `cpt-event-prediction-feature-baseline-prediction`, `cpt-event-prediction-feature-ml-prediction`

## 2. Actor Flows (CDSL)

**Use cases**: `cpt-event-prediction-usecase-run-benchmark`

### Run the Final Benchmark

- [ ] `p1` - **ID**: `cpt-event-prediction-flow-benchmarking-evaluation-run`

**Actor**: `cpt-event-prediction-actor-automation-runner`

**Success Scenarios**:
- Five declared seeds complete with aligned predictions for all three models and produce per-seed plus aggregate reports.

**Error Scenarios**:
- A seed failure, split overlap, missing prediction, incompatible vocabulary, or incomplete metric prevents a final aggregate from being marked complete.

**Steps**:
1. [ ] - `p1` - Load and validate the frozen benchmark configuration and five distinct seed bundles - `inst-eval-load-config`
2. [ ] - `p1` - **FOR EACH** seed bundle, generate data and create a new sequence-level split - `inst-eval-seed-data`
3. [ ] - `p1` - Fit baselines and logistic candidates from training data and select the learned model on validation only - `inst-eval-fit-models`
4. [ ] - `p1` - Freeze selected models and predict the common test examples once - `inst-eval-test-once`
5. [ ] - `p1` - Compute per-model metrics, supports, confusion matrices, and sequence-clustered intervals - `inst-eval-metrics`
6. [ ] - `p1` - Compute paired differences between logistic regression and each baseline on aligned sequence clusters - `inst-eval-paired`
7. [ ] - `p1` - Categorize representative errors without changing models or configuration - `inst-eval-errors`
8. [ ] - `p1` - Aggregate per-seed scalar values with arithmetic mean and sample standard deviation, and elementwise-sum each model's fixed-order 9x9 confusion matrices and class supports - `inst-eval-aggregate`
9. [ ] - `p1` - Write complete machine-readable results and a concise Markdown report with limitations - `inst-eval-write`
10. [ ] - `p1` - **RETURN** complete status only when all five seed records validate - `inst-eval-return`

## 3. Processes / Business Logic (CDSL)

### Compute Per-Seed Evaluation

- [x] `p1` - **ID**: `cpt-event-prediction-algo-benchmarking-evaluation-metrics`

**Input**: Aligned test targets, sequence identifiers, class vocabulary, and probability predictions from all models.

**Output**: Accuracy, macro-F1, class support, confusion matrix, and classified prediction records per model.

**Steps**:
1. [x] - `p1` - Verify every model prediction aligns to the same ordered test example keys - `inst-eval-align`
2. [x] - `p1` - Convert probabilities to class predictions using stable catalog-order tie-breaking - `inst-eval-classify`
3. [x] - `p1` - Compute accuracy over all test prefix examples - `inst-eval-accuracy`
4. [x] - `p1` - Compute per-class precision, recall, and F1 and macro-average across the fixed vocabulary - `inst-eval-macro-f1`
5. [x] - `p1` - Compute confusion counts in fixed event-catalog row and column order - `inst-eval-confusion`
6. [x] - `p1` - Record zero-support and undefined-metric handling explicitly - `inst-eval-zero-support`
7. [x] - `p1` - **RETURN** validated per-model metric record - `inst-eval-metric-result`

### Estimate Sequence-Clustered Uncertainty

- [x] `p1` - **ID**: `cpt-event-prediction-algo-benchmarking-evaluation-clustered-bootstrap`

**Input**: Per-example predictions grouped by test sequence, bootstrap seed, and resample count.

**Output**: Percentile 95% intervals for model metrics and paired logistic-minus-baseline accuracy and macro-F1 differences.

**Steps**:
1. [x] - `p1` - Treat each unique test sequence identifier as one resampling cluster - `inst-eval-clusters`
2. [x] - `p1` - **FOR EACH** bootstrap replicate, sample sequence identifiers with replacement and include all prefixes of each sampled sequence - `inst-eval-resample`
3. [x] - `p1` - Recompute each model metric and aligned logistic-minus-baseline accuracy difference - `inst-eval-bootstrap-stat`
4. [x] - `p1` - Select declared percentile bounds and record bootstrap seed and replicate count - `inst-eval-percentiles`
5. [x] - `p1` - **RETURN** intervals without treating prefixes as independent - `inst-eval-interval-result`

### Analyze Prediction Errors

- [x] `p1` - **ID**: `cpt-event-prediction-algo-benchmarking-evaluation-error-analysis`

**Input**: Test prefixes, targets, probabilities, sequence identifiers, and event catalog.

**Output**: Counts and representative examples for declared error categories.

**Steps**:
1. [x] - `p1` - Group errors by true and predicted event pair and prefix-length band - `inst-eval-group-errors`
2. [x] - `p1` - Tag errors involving rare targets, post-failure recovery, readiness ambiguity, or deployment-versus-rollback ambiguity - `inst-eval-tag-errors`
3. [x] - `p1` - Rank examples deterministically by confidence margin then stable example key - `inst-eval-rank-errors`
4. [x] - `p1` - Select a bounded representative sample per category - `inst-eval-sample-errors`
5. [x] - `p1` - **RETURN** diagnostic records without feeding them back into final model selection - `inst-eval-error-result`

## 4. States (CDSL)

### Benchmark Run State Machine

- [ ] `p1` - **ID**: `cpt-event-prediction-state-benchmarking-evaluation-run`

**States**: CONFIGURED, RUNNING, INCOMPLETE, COMPLETE

**Initial State**: CONFIGURED

**Transitions**:
1. [ ] - `p1` - **FROM** CONFIGURED **TO** RUNNING **WHEN** configuration and seed validation pass - `inst-eval-state-running`
2. [ ] - `p1` - **FROM** RUNNING **TO** INCOMPLETE **WHEN** any seed or artifact validation fails - `inst-eval-state-incomplete`
3. [ ] - `p1` - **FROM** RUNNING **TO** COMPLETE **WHEN** all five seeds and aggregate records validate - `inst-eval-state-complete`

## 5. Definitions of Done

### Complete Statistical Benchmark

- [x] `p1` - **ID**: `cpt-event-prediction-dod-benchmarking-evaluation-statistics`

The system **MUST** report the specified per-seed and aggregate evidence for all three predictors while preserving validation and test boundaries. Aggregate confusion evidence is the elementwise sum of the validated per-seed 9x9 matrices in fixed event-catalog order; aggregate support is the corresponding per-class sum and is not substituted for seed-level uncertainty.

**Implements**:
- `cpt-event-prediction-flow-benchmarking-evaluation-run`
- `cpt-event-prediction-algo-benchmarking-evaluation-metrics`
- `cpt-event-prediction-algo-benchmarking-evaluation-clustered-bootstrap`

**Constraints**: `cpt-event-prediction-constraint-evaluation-protocol`

**Touches**:
- CLI: `benchmark`
- Files: per-seed and aggregate benchmark JSON, Markdown report
- Entities: Prediction, BenchmarkRun

### Reproducible Error Analysis

- [x] `p1` - **ID**: `cpt-event-prediction-dod-benchmarking-evaluation-errors`

The system **MUST** produce deterministic categorized error records and representative observable prefixes without changing the selected models.

**Implements**:
- `cpt-event-prediction-algo-benchmarking-evaluation-error-analysis`

**Constraints**: `cpt-event-prediction-constraint-schedule`

**Touches**:
- Files: error-analysis JSON and Markdown section
- Entities: PrefixExample, Prediction

## 6. Acceptance Criteria

- [x] Exactly five complete independent seed records are required before aggregate status is COMPLETE.
- [x] All three models are evaluated on identical test example keys within each seed.
- [x] Accuracy, macro-F1, fixed-order 9x9 confusion matrix, and class support are present for every model and seed.
- [x] Aggregate mean and sample standard deviation are computed from the five per-seed metrics, not pooled prefixes.
- [x] Each aggregate confusion matrix is the elementwise fixed-order sum of its five seed matrices, and each aggregate class support is the matching sum of seed supports.
- [x] Confidence intervals and paired accuracy and macro-F1 differences resample complete test sequences.
- [x] Test metrics never influence hyperparameters, features, transition calibration, or model choice.
- [x] Error analysis covers rare classes, recovery paths, readiness ambiguity, and deployment-versus-rollback confusion with bounded examples.
- [x] Reports label all conclusions as synthetic-domain evidence and contain no fabricated results.
