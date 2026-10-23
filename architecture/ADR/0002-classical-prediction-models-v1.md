---
status: ACCEPTED
date: 2026-10-20
description: Decision to use transparent baselines and multinomial logistic regression for next-event prediction.
---

# Use Transparent Baselines and Multinomial Logistic Regression

Decision to use two transparent references and one sparse linear multiclass model for a clear, reproducible performance ladder.


<!-- toc -->

- [Context and Problem Statement](#context-and-problem-statement)
- [Decision Drivers](#decision-drivers)
- [Considered Options](#considered-options)
- [Decision Outcome](#decision-outcome)
  - [Consequences](#consequences)
  - [Confirmation](#confirmation)
- [Pros and Cons of the Options](#pros-and-cons-of-the-options)
  - [Transparent Baselines plus Multinomial Logistic Regression](#transparent-baselines-plus-multinomial-logistic-regression)
  - [Gradient-Boosted Decision Trees](#gradient-boosted-decision-trees)
  - [Neural Sequence Model](#neural-sequence-model)
- [More Information](#more-information)
- [Traceability](#traceability)

<!-- /toc -->

**ID**: `cpt-event-prediction-adr-classical-prediction-models`
## Context and Problem Statement

The predictor must demonstrate genuine next-event learning while remaining understandable, reproducible, and practical for a five-day hiring assignment. The chosen model family must also be meaningfully stronger than transparent references without introducing GPU or neural-network infrastructure.

## Decision Drivers

* Direct comparability with simple frequency and transition heuristics
* Support for multiclass probabilities and sparse historical features
* CPU-only, deterministic training with mature tooling
* Inspectable coefficients and straightforward failure analysis
* Low implementation and operational complexity

## Considered Options

* Global-frequency and smoothed Markov baselines plus multinomial logistic regression
* Gradient-boosted decision trees as the primary learned model
* Recurrent or transformer neural sequence model

## Decision Outcome

Chosen option: "Global-frequency and smoothed Markov baselines plus multinomial logistic regression", because it creates a clear performance ladder and lets the learned model use prefix history while keeping training, interpretation, and reproduction simple.

### Consequences

* Good, because both baselines expose dataset difficulty and transition signal.
* Good, because sparse categorical and numeric prefix features fit naturally in a linear multiclass pipeline.
* Good, because validation can select regularization without large search cost.
* Bad, because linear decision boundaries may miss complex interactions.
* Bad, because event probabilities can be miscalibrated even when accuracy is strong.

### Confirmation

The implementation must train all models from training data only, apply Laplace smoothing to unseen Markov successors, use a fixed deterministic solver configuration, and verify that the logistic model consumes only prefix-derived features. Benchmark output must compare all models on identical held-out examples.

## Pros and Cons of the Options

### Transparent Baselines plus Multinomial Logistic Regression

* Good, because each increase in model complexity has an interpretable purpose.
* Good, because the model supports class probabilities and sparse feature vectors.
* Good, because CPU runtime is small enough for repeated experiments.
* Bad, because feature engineering is required to represent recency and longer history.

### Gradient-Boosted Decision Trees

* Good, because nonlinear interactions can be learned with little manual scaling.
* Bad, because sparse event-history encoding is less natural and probability comparison is less transparent.
* Bad, because it adds tuning breadth that is unnecessary for the assignment.

### Neural Sequence Model

* Good, because recurrent or attention layers can model order directly.
* Bad, because added dependencies, tuning variance, and training cost reduce reproducibility.
* Bad, because 10,000 synthetic sequences do not justify the infrastructure or portfolio complexity.

## More Information

The learned feature set will include recent event identities, prefix event counts and proportions, transition counts, prefix length, failure and recovery indicators, and recency features. It will never include future events or internal simulator state.

## Traceability

- **PRD**: [PRD.md](../PRD.md)
- **DESIGN**: [DESIGN.md](../DESIGN.md)

This decision directly addresses the following requirements or design elements:

* `cpt-event-prediction-fr-baseline-models` - defines the two reference predictors.
* `cpt-event-prediction-fr-logistic-regression` - selects the learned model family.
* `cpt-event-prediction-fr-prevent-leakage` - limits features to the observable prefix.
* `cpt-event-prediction-nfr-local-runtime` - avoids GPU and deep-learning requirements.
* `cpt-event-prediction-design-local-pipeline` - defines the modeling tier of the local pipeline.
