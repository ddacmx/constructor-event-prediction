---
description: Feature specification for stable local commands, provenance, automated checks, and reviewer demonstration.
---

# Feature: Reproducible CLI Demonstration

Specification for stable reviewer commands, provenance, automated verification, and a genuine Constructor Fabric usage record.


<!-- toc -->

- [1. Feature Context](#1-feature-context)
  - [1.1 Overview](#11-overview)
  - [1.2 Purpose](#12-purpose)
  - [1.3 Actors](#13-actors)
  - [1.4 References](#14-references)
- [2. Actor Flows (CDSL)](#2-actor-flows-cdsl)
  - [Run the Reviewer Demonstration](#run-the-reviewer-demonstration)
  - [Run Automated Verification](#run-automated-verification)
- [3. Processes / Business Logic (CDSL)](#3-processes--business-logic-cdsl)
  - [Dispatch a CLI Command](#dispatch-a-cli-command)
  - [Record Reproduction Provenance](#record-reproduction-provenance)
- [4. States (CDSL)](#4-states-cdsl)
  - [Stateless Command Execution](#stateless-command-execution)
- [5. Definitions of Done](#5-definitions-of-done)
  - [Stable Reviewer CLI](#stable-reviewer-cli)
  - [Automated Verification and Fabric Evidence](#automated-verification-and-fabric-evidence)
- [6. Acceptance Criteria](#6-acceptance-criteria)

<!-- /toc -->

- [ ] `p2` - **ID**: `cpt-event-prediction-featstatus-reproducible-cli-demo`
## 1. Feature Context

- [ ] `p2` - `cpt-event-prediction-feature-reproducible-cli-demo`

### 1.1 Overview

This feature integrates generation, benchmarking, prediction, and verification behind stable local commands and reviewer-focused documentation.

### 1.2 Purpose

The demonstration makes the assignment independently runnable and shows where machine-readable evidence and Constructor Fabric traceability can be inspected.

**Requirements**: `cpt-event-prediction-fr-cli-demo`, `cpt-event-prediction-fr-fabric-traceability`, `cpt-event-prediction-nfr-review-usability`, `cpt-event-prediction-nfr-testability`, `cpt-event-prediction-nfr-local-runtime`

**Principles**: `cpt-event-prediction-principle-immutable-artifacts`

### 1.3 Actors

| Actor | Role in Feature |
|-------|-----------------|
| `cpt-event-prediction-actor-technical-reviewer` | Follows setup, runs the demo, and inspects predictions and reports. |
| `cpt-event-prediction-actor-automation-runner` | Runs tests and artifact checks non-interactively. |
| `cpt-event-prediction-actor-ml-engineer` | Diagnoses invalid inputs and reproduces benchmark artifacts. |

### 1.4 References

- **PRD**: [PRD.md](../PRD.md)
- **Design**: [DESIGN.md](../DESIGN.md)
- **Dependencies**: `cpt-event-prediction-feature-synthetic-data-generation`, `cpt-event-prediction-feature-baseline-prediction`, `cpt-event-prediction-feature-ml-prediction`, `cpt-event-prediction-feature-benchmarking-evaluation`

## 2. Actor Flows (CDSL)

**Use cases**: `cpt-event-prediction-usecase-generate-dataset`, `cpt-event-prediction-usecase-run-benchmark`, `cpt-event-prediction-usecase-predict-prefix`

### Run the Reviewer Demonstration

- [ ] `p1` - **ID**: `cpt-event-prediction-flow-reproducible-cli-demo-reviewer`

**Actor**: `cpt-event-prediction-actor-technical-reviewer`

**Success Scenarios**:
- The reviewer installs locked dependencies, runs a small smoke benchmark, predicts from a supplied prefix, and locates full benchmark instructions and Fabric evidence.

**Error Scenarios**:
- Unsupported Python, missing files, invalid event prefixes, or failed verification produce actionable errors and non-zero exit status.

**Steps**:
1. [ ] - `p1` - Follow the documented environment setup and verify supported versions - `inst-cli-setup`
2. [ ] - `p1` - Run the documented verification procedure and stop if any project, dependency, or Fabric check fails - `inst-cli-verify`
3. [ ] - `p1` - Run the smoke demonstration with its declared configuration and seed - `inst-cli-smoke`
4. [ ] - `p1` - Inspect model comparison output and the linked machine-readable result - `inst-cli-inspect-results`
5. [ ] - `p1` - Submit an example observable prefix to the selected model - `inst-cli-submit-prefix`
6. [ ] - `p1` - Inspect the predicted event and ordered probabilities - `inst-cli-inspect-prediction`
7. [ ] - `p1` - Locate the full benchmark command, limitations, and Constructor Fabric usage record - `inst-cli-locate-evidence`
8. [ ] - `p1` - **RETURN** successful demonstration status - `inst-cli-demo-result`

### Run Automated Verification

- [ ] `p1` - **ID**: `cpt-event-prediction-flow-reproducible-cli-demo-verify`

**Actor**: `cpt-event-prediction-actor-automation-runner`

**Success Scenarios**:
- One documented non-interactive verification procedure runs project CLI checks, tests and coverage policy, dependency checks, and Fabric artifact validation as separate commands.

**Error Scenarios**:
- Any failed check returns non-zero and identifies the failing command or artifact.

**Steps**:
1. [ ] - `p1` - Validate runtime and locked dependency versions - `inst-cli-verify-runtime`
2. [ ] - `p1` - Run project CLI verification for deterministic dataset and schema smoke checks - `inst-cli-verify-data`
3. [ ] - `p1` - Run unit, integration, and CLI test suites with branch coverage at or above 90%, then run the locked-environment dependency check - `inst-cli-verify-tests`
4. [ ] - `p1` - Separately run Constructor Studio artifact, ID-resolution, traceability, and specification-coverage commands - `inst-cli-verify-fabric`
5. [ ] - `p1` - **IF** any check fails, return non-zero with its output path - `inst-cli-verify-fail`
6. [ ] - `p1` - **RETURN** a concise pass summary and evidence paths - `inst-cli-verify-result`

## 3. Processes / Business Logic (CDSL)

### Dispatch a CLI Command

- [x] `p1` - **ID**: `cpt-event-prediction-algo-reproducible-cli-demo-dispatch`

**Input**: Command name, validated arguments, configuration paths, and output policy.

**Output**: Domain workflow result and stable process exit code.

**Steps**:
1. [x] - `p1` - Parse the command and reject unknown options or conflicting inputs - `inst-cli-parse`
2. [x] - `p1` - Resolve paths relative to the invoking project without changing the working directory implicitly - `inst-cli-resolve-paths`
3. [x] - `p1` - **IF** the command would overwrite a protected artifact, require an explicit replacement flag - `inst-cli-protect-output`
4. [x] - `p1` - Dispatch to generation, benchmark, prediction, or verification application service - `inst-cli-dispatch-service`
5. [x] - `p1` - Format a concise console summary and retain detailed JSON evidence - `inst-cli-format`
6. [x] - `p1` - Map validation, I/O, model, and incomplete-benchmark failures to documented exit codes - `inst-cli-exit-code`
7. [x] - `p1` - **RETURN** exit code and artifact links - `inst-cli-dispatch-result`

### Record Reproduction Provenance

- [ ] `p1` - **ID**: `cpt-event-prediction-algo-reproducible-cli-demo-provenance`

**Input**: Command arguments, resolved configurations, dependency versions, seeds, and produced artifacts.

**Output**: Stable provenance block embedded in the relevant manifest or result file.

**Steps**:
1. [ ] - `p1` - Canonicalize effective configuration and hash its content - `inst-cli-hash-config`
2. [ ] - `p1` - Record command version, schema versions, Python and dependency versions, and all random seeds - `inst-cli-record-versions`
3. [ ] - `p1` - Record input and output hashes without environment-specific absolute paths - `inst-cli-record-hashes`
4. [ ] - `p1` - **RETURN** provenance suitable for independent comparison - `inst-cli-provenance-result`

## 4. States (CDSL)

### Stateless Command Execution

No persistent UI or service state machine is required. Each command is an isolated process that either returns success with finalized artifacts or returns failure without marking incomplete output as final.

## 5. Definitions of Done

### Stable Reviewer CLI

- [x] `p1` - **ID**: `cpt-event-prediction-dod-reproducible-cli-demo-commands`

The system **MUST** expose documented `generate`, `benchmark`, `predict`, and `verify` commands with consistent errors, exit codes, seeds, and output policies.

**Implements**:
- `cpt-event-prediction-flow-reproducible-cli-demo-reviewer`
- `cpt-event-prediction-algo-reproducible-cli-demo-dispatch`
- `cpt-event-prediction-algo-reproducible-cli-demo-provenance`

**Constraints**: `cpt-event-prediction-constraint-cpu-local`, `cpt-event-prediction-constraint-schedule`

**Touches**:
- CLI: `generate`, `benchmark`, `predict`, `verify`
- Files: README, configuration examples, provenance blocks
- Entities: EventSequence, BenchmarkRun

### Automated Verification and Fabric Evidence

- [ ] `p1` - **ID**: `cpt-event-prediction-dod-reproducible-cli-demo-verification`

The repository **MUST** provide one documented multi-command verification path covering automated project checks and the actual Constructor Studio skills, pins, validation, traceability, and specification-coverage commands used.

**Implements**:
- `cpt-event-prediction-flow-reproducible-cli-demo-verify`

**Constraints**: `cpt-event-prediction-constraint-schedule`

**Touches**:
- CLI: `verify`
- Files: README, Constructor Fabric usage note, machine-readable test outputs

## 6. Acceptance Criteria

- [ ] A clean supported environment reaches a smoke prediction through documented commands without hidden setup steps.
- [ ] Every command supports explicit configuration, seed where relevant, output destination, and actionable validation errors.
- [ ] Protected artifacts are not silently overwritten.
- [ ] Machine-readable results remain separate from concise console summaries and contain reproduction provenance.
- [ ] The documented verification procedure treats a non-zero result from any project, dependency, or Fabric command as a failed verification.
- [ ] Documentation distinguishes the quick smoke demonstration from the five-seed final benchmark.
- [ ] The Fabric usage note names actual installed versions, pinned revisions, skills, commands, artifacts, and known limitations without claiming unexecuted work.
