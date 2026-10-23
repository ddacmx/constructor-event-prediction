# Employer requirement-to-evidence checklist

The employer-provided assignment and its working transcription remain local and are excluded from the public release. This checklist is a concise, paraphrased mapping from the assignment's requirements to repository evidence.

| Employer requirement | Project evidence | Release status |
|---|---|---|
| Define time-ordered discrete events with short descriptions | `configs/default.json` event catalog; `src/event_prediction/catalog.py`; generated JSONL records | Satisfied |
| Build a separate meaningful synthetic sequence generator | `src/event_prediction/simulator.py`; `event-prediction generate`; generator tests | Satisfied |
| Predict the next event | Global-frequency, first-order Markov, and multinomial logistic-regression implementations under `src/event_prediction/` | Satisfied |
| Use Constructor Fabric | Pinned Studio configuration, 11 registered artifacts, CPT markers, and the execution record in `docs/fabric-usage.md` | Satisfied |
| Show correct/incorrect evaluation | Held-out accuracy, macro-F1, per-class results, confusion matrices, paired comparisons, and error analysis | Satisfied |
| Include a PRD and/or product design | `architecture/PRD.md`, ADRs, `architecture/DESIGN.md`, DECOMPOSITION, and FEATURE specs | Satisfied |
| Include the generator and prediction model | Reproducible Python package under `src/event_prediction/` | Satisfied |
| Provide a working demo, benchmark, and statistical evaluation | README commands; `artifacts/final/benchmark.json`; hash-bound machine-readable results and Markdown report | Satisfied |
| Provide clear run instructions in a self-contained GitHub repository | Root README, exact dependency lock, configuration, source, tests, specifications, and final evidence in the proposed release set | Satisfied after publication |
| Include a short Markdown explanation of Fabric use | `docs/fabric-usage.md` | Satisfied |

No mandatory technical deliverable is missing from the proposed repository. Publishing the repository and sending its link are intentionally deferred to Phase 5B and the later submission step.
