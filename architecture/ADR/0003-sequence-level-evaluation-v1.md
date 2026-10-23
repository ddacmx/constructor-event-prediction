---
status: ACCEPTED
date: 2026-10-20
description: Decision to use five-seed sequence-level repeated holdout evaluation with clustered uncertainty.
---

# Use Sequence-Level Repeated Holdout Evaluation

Decision to isolate complete sequences, select on validation only, and report repeated paired evidence with cluster-aware uncertainty.


<!-- toc -->

- [Context and Problem Statement](#context-and-problem-statement)
- [Decision Drivers](#decision-drivers)
- [Considered Options](#considered-options)
- [Decision Outcome](#decision-outcome)
  - [Consequences](#consequences)
  - [Confirmation](#confirmation)
- [Pros and Cons of the Options](#pros-and-cons-of-the-options)
  - [Five-Seed Sequence-Level Repeated Holdout](#five-seed-sequence-level-repeated-holdout)
  - [Random Prefix-Level Split](#random-prefix-level-split)
  - [Sequence-Level K-Fold Cross-Validation](#sequence-level-k-fold-cross-validation)
- [More Information](#more-information)
- [Traceability](#traceability)

<!-- /toc -->

**ID**: `cpt-event-prediction-adr-sequence-level-evaluation`
## Context and Problem Statement

Every sequence yields multiple prefix-target examples, so example-level random splitting would place strongly related observations in different partitions. The evaluation must prevent this leakage, reserve the test set for final reporting, and quantify seed sensitivity and paired model differences.

## Decision Drivers

* Isolation of all prefixes from the same sequence
* A validation-only model-selection boundary
* Direct paired comparison on identical held-out sequences
* Honest uncertainty under within-sequence correlation
* Reproducible runtime suitable for five complete repetitions

## Considered Options

* Five-seed sequence-level 70/15/15 repeated holdout with clustered uncertainty
* Random prefix-level 70/15/15 split
* Sequence-level k-fold cross-validation with a separate final test set

## Decision Outcome

Chosen option: "Five-seed sequence-level 70/15/15 repeated holdout with clustered uncertainty", because it prevents related prefixes from crossing partitions, supports validation-only selection, gives five independent end-to-end results, and stays within the local runtime budget.

### Consequences

* Good, because all models are compared on identical examples within a seed.
* Good, because resampling complete sequences respects the natural dependence structure.
* Good, because per-seed results reveal generation and split sensitivity.
* Bad, because five repetitions provide only a small sample for across-seed inference.
* Bad, because repeated holdout uses training data less efficiently than nested cross-validation.

### Confirmation

Split manifests must show disjoint sequence identifiers at 70/15/15 proportions. Tests must prove prefix derivation happens after partitioning. Hyperparameters must be chosen only from validation metrics. Test output must include per-seed accuracy and macro-F1, class support, confusion matrices, mean and sample standard deviation, sequence-clustered 95% bootstrap intervals, and paired accuracy differences against both baselines.

## Pros and Cons of the Options

### Five-Seed Sequence-Level Repeated Holdout

* Good, because it balances rigor, clarity, and runtime.
* Good, because paired bootstrap resampling can retain all prefixes belonging to a sampled sequence.
* Bad, because estimates may vary more than a large k-fold design.

### Random Prefix-Level Split

* Good, because it is simple and yields many examples in every partition.
* Bad, because near-duplicate prefixes from one sequence can leak across train and test.
* Bad, because standard confidence intervals would falsely treat correlated prefixes as independent.

### Sequence-Level K-Fold Cross-Validation

* Good, because it uses data efficiently.
* Bad, because nested selection plus an untouched test set expands runtime and reporting complexity.
* Bad, because it is unnecessary for the fixed 10,000-sequence assignment target.

## More Information

Confidence intervals describe uncertainty for the synthetic test sequences under each configured experiment, not uncertainty about real-world organizations. Across-seed mean and standard deviation are descriptive because five seeds are too few for strong distributional claims.

## Traceability

- **PRD**: [PRD.md](../PRD.md)
- **DESIGN**: [DESIGN.md](../DESIGN.md)

This decision directly addresses the following requirements or design elements:

* `cpt-event-prediction-fr-sequence-split` - establishes the partition unit and ratios.
* `cpt-event-prediction-fr-multi-seed-evaluation` - defines five independent repetitions.
* `cpt-event-prediction-fr-statistical-reporting` - defines paired and clustered statistical evidence.
* `cpt-event-prediction-nfr-statistical-rigor` - preserves sequence-level dependence.
* `cpt-event-prediction-design-local-pipeline` - constrains benchmark orchestration.
