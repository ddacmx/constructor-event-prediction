"""End-to-end benchmark orchestration."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sklearn
import shutil
import uuid

from . import __version__
from .artifacts import canonical_json, read_json, resolve_portable_reference, sha256_file, write_bytes, write_json
from .baselines import GlobalFrequencyPredictor, MarkovPredictor, fit_baselines
from .catalog import EVENT_TYPES
from .config import GeneratorConfig
from .domain import ArtifactError, ConfigurationError, PredictionBatch, PrefixExample, RESULT_SCHEMA, SplitManifest
from .evaluation import evaluate_models
from .model import _deterministic_npz, load_model, save_model, train_logistic
from .processing import split_and_build_examples
from .simulator import generate_sequences
from .schemas import validate_schema

FINAL_PROFILE_VERSION = "final-benchmark-profile.v1"
FINAL_SEQUENCE_COUNT = 10_000
FINAL_SEEDS = (20261021, 20261022, 20261023, 20261024, 20261025)
FINAL_CONFIG_HASH = "83b78d453501c33415d8e834ea3f0d46d66aa44278ae2fb1e1f471f1f12e2b51"
PREDICTION_EVIDENCE_SCHEMA = "keyed-predictions.v1"
BASELINE_STATE_SCHEMA = "baseline-state.v1"
BASELINE_TRAINING_PROTOCOL = "observable-prefix-baselines.v1"


def _frozen_final_profile() -> dict[str, Any]:
    return {
        "schema_version": FINAL_PROFILE_VERSION,
        "count": FINAL_SEQUENCE_COUNT,
        "seeds": list(FINAL_SEEDS),
        "config_hash": FINAL_CONFIG_HASH,
    }


def _profile_hash(profile: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(profile).encode("utf-8")).hexdigest()


def _examples_hash(examples: Sequence[Any]) -> str:
    payload = [
        {"prefix": list(example.prefix), "sequence_id": example.sequence_id, "target": example.target}
        for example in examples
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_prediction_evidence(
    path: Path,
    examples: Sequence[PrefixExample],
    batches: dict[str, PredictionBatch],
    *,
    replace: bool,
) -> tuple[str, str]:
    expected_models = {"global_frequency", "markov", "logistic_regression"}
    if set(batches) != expected_models:
        raise ArtifactError("prediction evidence must contain exactly the three approved models")
    expected_keys = tuple(example.key for example in examples)
    arrays: dict[str, np.ndarray] = {
        "example_keys": np.asarray(expected_keys, dtype=np.str_),
        "sequence_ids": np.asarray([example.sequence_id for example in examples], dtype=np.str_),
        "target_indices": np.asarray([EVENT_TYPES.index(example.target) for example in examples], dtype=np.int64),
    }
    for name in sorted(expected_models):
        batch = batches[name]
        if batch.example_keys != expected_keys or batch.vocabulary != EVENT_TYPES:
            raise ArtifactError(f"{name} prediction evidence is not keyed to the ordered test examples")
        values = np.asarray(batch.probabilities, dtype=np.float64)
        # evaluate_models already applies the full probability contract; keep
        # the persisted representation exact and deterministic.
        arrays[f"probabilities_{name}"] = values
    payload = _deterministic_npz(arrays)
    digest = hashlib.sha256(payload).hexdigest()
    evidence_path = path.with_name(f"{path.stem}.{digest[:16]}.npz")
    write_bytes(evidence_path, payload, replace=replace)
    return evidence_path.name, digest


def _load_prediction_evidence(
    path: Path,
    expected_sha256: str,
    examples: Sequence[PrefixExample],
) -> dict[str, PredictionBatch]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"cannot read prediction evidence {path}: {exc}") from exc
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ArtifactError("prediction evidence hash does not match the seed result")
    required = {
        "example_keys", "sequence_ids", "target_indices",
        "probabilities_global_frequency", "probabilities_markov", "probabilities_logistic_regression",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise ArtifactError("prediction evidence arrays are incomplete or unexpected")
            arrays = {name: np.asarray(archive[name]) for name in required}
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactError(f"cannot safely read prediction evidence: {exc}") from exc
    count = len(examples)
    keys = tuple(str(value) for value in arrays["example_keys"].tolist())
    sequence_ids = [str(value) for value in arrays["sequence_ids"].tolist()]
    targets = arrays["target_indices"]
    if arrays["example_keys"].ndim != 1 or arrays["example_keys"].dtype.kind != "U":
        raise ArtifactError("prediction evidence keys must be a Unicode vector")
    if arrays["sequence_ids"].ndim != 1 or arrays["sequence_ids"].dtype.kind != "U":
        raise ArtifactError("prediction evidence sequence IDs must be a Unicode vector")
    if targets.dtype != np.dtype("int64") or targets.shape != (count,):
        raise ArtifactError("prediction evidence targets have an invalid representation")
    expected_keys = tuple(example.key for example in examples)
    if keys != expected_keys or sequence_ids != [example.sequence_id for example in examples]:
        raise ArtifactError("prediction evidence keys/order do not match trusted held-out examples")
    expected_targets = np.asarray([EVENT_TYPES.index(example.target) for example in examples], dtype=np.int64)
    if not np.array_equal(targets, expected_targets):
        raise ArtifactError("prediction evidence targets contradict trusted held-out targets")
    batches: dict[str, PredictionBatch] = {}
    for name in ("global_frequency", "markov", "logistic_regression"):
        values = arrays[f"probabilities_{name}"]
        if values.dtype != np.dtype("float64") or values.shape != (count, len(EVENT_TYPES)):
            raise ArtifactError(f"{name} prediction evidence has invalid dimensions or dtype")
        batches[name] = PredictionBatch(keys, values)
    return batches


def _validated_probability_vector(value: np.ndarray, name: str) -> np.ndarray:
    if value.dtype != np.dtype("float64") or value.shape != (len(EVENT_TYPES),):
        raise ArtifactError(f"baseline {name} has invalid dtype or dimensions")
    if (not np.all(np.isfinite(value)) or np.any(value < 0) or np.any(value > 1)
            or not np.isclose(value.sum(), 1.0, rtol=0, atol=1e-12)):
        raise ArtifactError(f"baseline {name} is not a finite normalized probability vector")
    return value


def _baseline_state_parts(metadata: dict[str, Any], arrays_path: Path) -> dict[str, Any]:
    expected_fields = {
        "schema_version", "arrays", "arrays_sha256", "event_vocabulary", "alpha", "provenance"
    }
    if set(metadata) != expected_fields or metadata.get("schema_version") != BASELINE_STATE_SCHEMA:
        raise ArtifactError("baseline-state manifest fields or schema are invalid")
    if metadata.get("event_vocabulary") != list(EVENT_TYPES):
        raise ArtifactError("baseline-state vocabulary order is invalid")
    alpha = metadata.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not np.isfinite(alpha) or alpha <= 0:
        raise ArtifactError("baseline-state alpha must be finite and positive")
    provenance = metadata.get("provenance")
    required_provenance = {
        "config_hash", "split_seed", "training_examples_hash", "final_profile_hash", "training_protocol"
    }
    if not isinstance(provenance, dict) or set(provenance) != required_provenance:
        raise ArtifactError("baseline-state provenance is incomplete or unexpected")
    if provenance.get("training_protocol") != BASELINE_TRAINING_PROTOCOL:
        raise ArtifactError("baseline-state training protocol is unsupported")
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            required_arrays = {"global_probabilities", "markov_rows", "markov_fallback"}
            if set(archive.files) != required_arrays:
                raise ArtifactError("baseline-state arrays are incomplete or unexpected")
            arrays = {name: np.asarray(archive[name]) for name in required_arrays}
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactError(f"cannot safely read baseline-state arrays: {exc}") from exc
    global_probabilities = _validated_probability_vector(arrays["global_probabilities"], "global probabilities")
    fallback = _validated_probability_vector(arrays["markov_fallback"], "Markov fallback")
    rows = arrays["markov_rows"]
    if rows.dtype != np.dtype("float64") or rows.shape != (len(EVENT_TYPES), len(EVENT_TYPES)):
        raise ArtifactError("baseline Markov rows have invalid dtype or dimensions")
    for index, event in enumerate(EVENT_TYPES):
        _validated_probability_vector(rows[index], f"Markov row {event}")
    return {
        "global_frequency": GlobalFrequencyPredictor(global_probabilities.copy()),
        "markov": MarkovPredictor(
            rows={event: rows[index].copy() for index, event in enumerate(EVENT_TYPES)},
            fallback=fallback.copy(),
            alpha=float(alpha),
        ),
    }


def _save_baseline_state(
    baselines: dict[str, Any],
    path: Path,
    *,
    alpha: float,
    provenance: dict[str, Any],
    replace: bool,
) -> dict[str, Any]:
    if set(baselines) != {"global_frequency", "markov"}:
        raise ArtifactError("baseline state must contain exactly global-frequency and Markov predictors")
    global_predictor = baselines["global_frequency"]
    markov_predictor = baselines["markov"]
    if not isinstance(global_predictor, GlobalFrequencyPredictor) or not isinstance(markov_predictor, MarkovPredictor):
        raise ArtifactError("baseline predictors have unexpected types")
    arrays = {
        "global_probabilities": np.asarray(global_predictor.probabilities, dtype=np.float64),
        "markov_rows": np.asarray([markov_predictor.rows[event] for event in EVENT_TYPES], dtype=np.float64),
        "markov_fallback": np.asarray(markov_predictor.fallback, dtype=np.float64),
    }
    payload = _deterministic_npz(arrays)
    digest = hashlib.sha256(payload).hexdigest()
    arrays_name = f"{path.stem}.{digest[:16]}.npz"
    arrays_path = path.parent / arrays_name
    metadata = {
        "schema_version": BASELINE_STATE_SCHEMA,
        "arrays": arrays_name,
        "arrays_sha256": digest,
        "event_vocabulary": list(EVENT_TYPES),
        "alpha": float(alpha),
        "provenance": {**provenance, "training_protocol": BASELINE_TRAINING_PROTOCOL},
    }
    write_bytes(arrays_path, payload, replace=replace)
    _baseline_state_parts(metadata, arrays_path)
    write_json(path, metadata, replace=replace)
    return {"manifest": path.name, **metadata}


def _load_baseline_state(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = read_json(path)
    if not isinstance(metadata, dict):
        raise ArtifactError("baseline-state manifest must be an object")
    arrays_path = resolve_portable_reference(path.parent, metadata.get("arrays"), field="baseline-state arrays reference")
    if arrays_path.parent != path.parent.resolve():
        raise ArtifactError("baseline-state arrays reference must be a sibling filename")
    if sha256_file(arrays_path) != metadata.get("arrays_sha256"):
        raise ArtifactError("baseline-state arrays hash does not match its manifest")
    return metadata, _baseline_state_parts(metadata, arrays_path)


def _provenance(config: GeneratorConfig, seeds: Sequence[int]) -> dict[str, Any]:
    config_hash = config.digest()
    versions = {
        "command_version": __version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
    }
    provenance = {
        **versions,
        "config_hash": config_hash,
        "seeds": list(seeds),
        "final_profile_hash": _profile_hash(_frozen_final_profile()),
    }
    result = provenance
    return result


def run_seed(count: int, seed: int, config: GeneratorConfig, output_dir: Path, replace: bool) -> dict[str, Any]:
    # @cpt-begin:cpt-event-prediction-flow-baseline-prediction-fit-apply:p1:inst-base-load
    sequences = generate_sequences(count, seed, config)
    # @cpt-end:cpt-event-prediction-flow-baseline-prediction-fit-apply:p1:inst-base-load
    # @cpt-begin:cpt-event-prediction-flow-baseline-prediction-fit-apply:p1:inst-base-split
    manifest, examples = split_and_build_examples(sequences, seed + 10_000)
    # @cpt-end:cpt-event-prediction-flow-baseline-prediction-fit-apply:p1:inst-base-split
    # @cpt-begin:cpt-event-prediction-flow-baseline-prediction-fit-apply:p1:inst-base-prefixes
    training = examples["train"]
    validation = examples["validation"]
    test = examples["test"]
    # @cpt-end:cpt-event-prediction-flow-baseline-prediction-fit-apply:p1:inst-base-prefixes
    # @cpt-begin:cpt-event-prediction-flow-baseline-prediction-fit-apply:p1:inst-base-fit
    baselines = fit_baselines(training, config.laplace_alpha)
    # @cpt-end:cpt-event-prediction-flow-baseline-prediction-fit-apply:p1:inst-base-fit
    training_provenance = {
        "config_hash": config.digest(),
        "split_seed": seed + 10_000,
        "training_examples_hash": _examples_hash(training),
        "final_profile_hash": _profile_hash(_frozen_final_profile()),
    }
    logistic = train_logistic(
        training,
        validation,
        config.regularization_candidates,
        seed=seed + 20_000,
        recency_cap=config.recency_cap,
        provenance=training_provenance,
    )
    # @cpt-begin:cpt-event-prediction-flow-baseline-prediction-fit-apply:p1:inst-base-predict
    probabilities = {
        name: predictor.predict_examples(test) for name, predictor in baselines.items()
    }
    # @cpt-end:cpt-event-prediction-flow-baseline-prediction-fit-apply:p1:inst-base-predict
    probabilities["logistic_regression"] = logistic.predict_examples(test)
    evaluation = evaluate_models(test, probabilities, seed + 30_000, config.bootstrap_replicates)
    split_value = manifest.as_dict()
    manifest.validate({sequence.sequence_id for sequence in sequences})
    validate_schema("split-manifest.v1", split_value)
    split_path = output_dir / f"split-seed-{seed}.json"
    write_json(split_path, split_value, replace=replace)
    split_hash = sha256_file(split_path)
    model_path = output_dir / f"model-seed-{seed}.model.json"
    # @cpt-begin:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-persist
    model_metadata = save_model(logistic, model_path, replace=replace)
    # @cpt-end:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-persist
    test_keys = [example.key for example in test]
    test_keys_hash = hashlib.sha256(canonical_json(test_keys).encode("utf-8")).hexdigest()
    evidence_name, evidence_hash = _write_prediction_evidence(
        output_dir / f"predictions-seed-{seed}.npz", test, probabilities, replace=replace
    )
    baseline_metadata = _save_baseline_state(
        baselines,
        output_dir / f"baseline-state-seed-{seed}.json",
        alpha=config.laplace_alpha,
        provenance=training_provenance,
        replace=replace,
    )
    sequence_lengths = {sequence.sequence_id: len(sequence.events) for sequence in sequences}
    result = {
        "dataset_sequence_count": count,
        "baseline_state": baseline_metadata,
        "evaluation": evaluation,
        "model": model_metadata,
        "prediction_evidence": evidence_name,
        "prediction_evidence_schema": PREDICTION_EVIDENCE_SCHEMA,
        "prediction_evidence_sha256": evidence_hash,
        "schema_version": "benchmark-seed-result.v2",
        "seed": seed,
        "split_manifest": split_path.name,
        "split_manifest_sha256": split_hash,
        "test_example_count": len(test),
        "test_sequence_lengths": [sequence_lengths[sequence_id] for sequence_id in manifest.test_ids],
        "test_example_keys_sha256": test_keys_hash,
        "test_example_keys_definition": "sha256(canonical JSON array of PrefixExample.key in evaluation order)",
        "split": {
            "test_sequences": len(manifest.test_ids),
            "train_sequences": len(manifest.train_ids),
            "validation_sequences": len(manifest.validation_ids),
        },
    }
    validate_schema("seed-result.v2", result)
    write_json(output_dir / f"seed-{seed}.json", result, replace=replace)
    # @cpt-begin:cpt-event-prediction-flow-baseline-prediction-fit-apply:p1:inst-base-return
    completed = result
    # @cpt-end:cpt-event-prediction-flow-baseline-prediction-fit-apply:p1:inst-base-return
    return completed


# @cpt-dod:cpt-event-prediction-dod-benchmarking-evaluation-statistics:p1
def _aggregate(seed_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_name in ("global_frequency", "markov", "logistic_regression"):
        models[model_name] = {}
        for metric_name in ("accuracy", "macro_f1"):
            values = [result["evaluation"]["metrics"][model_name][metric_name] for result in seed_results]
            models[model_name][metric_name] = {
                "mean": statistics.fmean(values),
                "sample_stddev": statistics.stdev(values) if len(values) > 1 else None,
                "values": values,
            }
        matrices = [np.asarray(result["evaluation"]["metrics"][model_name]["confusion_matrix"], dtype=int) for result in seed_results]
        models[model_name]["summed_confusion_matrix"] = np.sum(matrices, axis=0).astype(int).tolist()
        models[model_name]["summed_class_support"] = {
            event: sum(result["evaluation"]["metrics"][model_name]["per_class"][event]["support"] for result in seed_results)
            for event in EVENT_TYPES
        }
    paired: dict[str, Any] = {}
    for baseline in ("markov", "global_frequency"):
        for metric in ("accuracy", "macro_f1"):
            values = [
                result["evaluation"]["metrics"]["logistic_regression"][metric]
                - result["evaluation"]["metrics"][baseline][metric]
                for result in seed_results
            ]
            paired[f"logistic_minus_{baseline}.{metric}"] = {
                "mean": statistics.fmean(values),
                "sample_stddev": statistics.stdev(values) if len(values) > 1 else None,
                "values": values,
                "sequence_clustered_intervals_by_seed": [
                    result["evaluation"]["bootstrap"]["intervals_95"][f"logistic_minus_{baseline}.{metric}"]
                    for result in seed_results
                ],
            }
    result = {
        "confusion_matrix_order": list(EVENT_TYPES),
        "schema_version": "benchmark-aggregate.v2",
        "seed_count": len(seed_results),
        "models": models,
        "paired_comparisons": paired,
        "sequence_clustered_intervals_by_seed": {
            str(item["seed"]): item["evaluation"]["bootstrap"]["intervals_95"] for item in seed_results
        },
        "error_analysis_by_seed": {str(item["seed"]): item["evaluation"]["error_analysis"] for item in seed_results},
        "interpretation": "Means and sample standard deviations summarize per-seed scalar metrics; summed confusion matrices and supports are pooled counts, not seed-level uncertainty estimates.",
    }
    validate_schema("aggregate-result.v2", result)
    return result


def _report(aggregate: dict[str, Any], status: str, count: int, seeds: Sequence[int]) -> str:
    lines = ["# Evaluation report", "", "Synthetic-domain evaluation evidence; results do not establish real-world performance.", "", f"Status: **{status}**", f"Sequences per seed: {count}", f"Seeds: {', '.join(map(str, seeds))}", "", "| Model | Accuracy mean | Accuracy sample SD | Macro-F1 mean | Macro-F1 sample SD |", "|---|---:|---:|---:|---:|"]
    for name in ("global_frequency", "markov", "logistic_regression"):
        values = aggregate["models"][name]
        fmt = lambda value: "n/a" if value is None else f"{value:.6f}"
        lines.append(f"| {name} | {fmt(values['accuracy']['mean'])} | {fmt(values['accuracy']['sample_stddev'])} | {fmt(values['macro_f1']['mean'])} | {fmt(values['macro_f1']['sample_stddev'])} |")
    lines += ["", "## Per-seed scalar metrics and paired values", ""]
    for index, seed in enumerate(seeds):
        lines += [f"### Seed {seed}", "", "| Quantity | Value |", "|---|---:|"]
        for model_name in ("global_frequency", "markov", "logistic_regression"):
            for metric_name in ("accuracy", "macro_f1"):
                value = aggregate["models"][model_name][metric_name]["values"][index]
                lines.append(f"| {model_name}.{metric_name} | {fmt(value)} |")
        for comparison, values in sorted(aggregate["paired_comparisons"].items()):
            lines.append(f"| {comparison} | {fmt(values['values'][index])} |")
    lines += ["", "## Paired logistic-regression differences", "", "| Comparison | Mean | Sample SD |", "|---|---:|---:|"]
    for name, values in sorted(aggregate["paired_comparisons"].items()):
        lines.append(f"| {name} | {fmt(values['mean'])} | {fmt(values['sample_stddev'])} |")
    lines += ["", "## Aggregate confusion matrices and class supports", ""]
    for name in ("global_frequency", "markov", "logistic_regression"):
        values = aggregate["models"][name]
        lines += [f"### {name}", "", "Order: " + ", ".join(aggregate["confusion_matrix_order"]), "", "```text"]
        lines.extend(" ".join(map(str, row)) for row in values["summed_confusion_matrix"])
        lines += ["```", "", "Supports: " + ", ".join(f"{event}={values['summed_class_support'][event]}" for event in EVENT_TYPES), ""]
    lines += [
        "## Sequence-clustered 95% confidence intervals by seed",
        "",
        "The aggregate JSON preserves every per-seed clustered interval below; it does not fabricate a pooled cross-seed interval.",
        "",
    ]
    for seed, intervals in sorted(aggregate["sequence_clustered_intervals_by_seed"].items(), key=lambda item: int(item[0])):
        lines.append(f"### Seed {seed}")
        lines.append("")
        for name, interval in sorted(intervals.items()):
            lines.append(f"- {name}: [{interval['lower']:.6f}, {interval['upper']:.6f}]")
        lines.append("")
    lines += ["## Deterministic logistic-regression error analysis", ""]
    for seed, analysis in sorted(aggregate["error_analysis_by_seed"].items(), key=lambda item: int(item[0])):
        lines.append(f"### Seed {seed}")
        for section in ("semantic_categories", "true_predicted_pairs", "prefix_length_bands"):
            lines.append(f"- {section}: {canonical_json(analysis[section]['groups'])}")
    lines += ["", "Confidence intervals are sequence-clustered within each seed. Paired comparisons use identical ordered test examples. Seed-level scalar summaries are intentionally separate from summed confusion-count evidence.", ""]
    return "\n".join(lines)


def _completion_status(count: int, seeds: Sequence[int], config: GeneratorConfig) -> str:
    profile = _frozen_final_profile()
    return "COMPLETE" if (
        count == profile["count"]
        and list(seeds) == profile["seeds"]
        and config.digest() == profile["config_hash"]
    ) else "PRELIMINARY"


def _validate_staged_run(stage: Path, seed_results: Sequence[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, str]:
    validate_schema("aggregate-result.v2", aggregate)
    for result in seed_results:
        validate_schema("seed-result.v2", result)
        _validate_evaluation_structure(result)
        split_path = resolve_portable_reference(stage, result["split_manifest"], field="split manifest reference")
        if sha256_file(split_path) != result["split_manifest_sha256"]:
            raise ArtifactError("split manifest hash mismatch")
        validate_schema("split-manifest.v1", read_json(split_path))
        model_path = resolve_portable_reference(stage, result["model"]["manifest"], field="model manifest reference")
        if not model_path.exists():
            raise ArtifactError("model manifest is missing")
        load_model_for_validation(model_path)
    required = [path for path in stage.iterdir() if path.is_file()]
    return {path.name: sha256_file(path) for path in sorted(required)}


def load_model_for_validation(path: Path) -> None:
    from .model import load_model
    load_model(path)


def _expected_split_ids(count: int, split_seed: int) -> tuple[list[str], list[str], list[str]]:
    identifiers = [f"seq-{index:06d}" for index in range(count)]
    order = np.random.default_rng(split_seed).permutation(count)
    train_count = int(count * .70)
    validation_count = int(count * .15)
    return tuple(
        [identifiers[index] for index in indices]
        for indices in (
            order[:train_count],
            order[train_count:train_count + validation_count],
            order[train_count + validation_count:],
        )
    )  # type: ignore[return-value]


def _require_finite_numbers(value: Any, path: str = "$") -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, float) and not np.isfinite(value):
        raise ArtifactError(f"non-finite numeric evidence at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_numbers(child, f"{path}[{index}]")


def _validate_evaluation_structure(seed_result: dict[str, Any]) -> None:
    evaluation = seed_result["evaluation"]
    if set(evaluation) != {"example_count", "metrics", "bootstrap", "error_analysis"}:
        raise ArtifactError("seed evaluation fields do not match the v2 contract")
    if set(evaluation["metrics"]) != {"global_frequency", "markov", "logistic_regression"}:
        raise ArtifactError("seed evaluation must contain exactly the three approved models")
    for model_name, metric in evaluation["metrics"].items():
        if set(metric) != {"accuracy", "macro_f1", "class_order", "confusion_matrix", "per_class"}:
            raise ArtifactError(f"{model_name} metric fields do not match the v2 contract")
        matrix = metric["confusion_matrix"]
        if len(matrix) != len(EVENT_TYPES) or any(len(row) != len(EVENT_TYPES) for row in matrix):
            raise ArtifactError(f"{model_name} confusion matrix has invalid dimensions")
        if any(type(cell) is not int or cell < 0 for row in matrix for cell in row):
            raise ArtifactError(f"{model_name} confusion matrix contains invalid counts")
        if metric["class_order"] != list(EVENT_TYPES) or set(metric["per_class"]) != set(EVENT_TYPES):
            raise ArtifactError(f"{model_name} per-class evidence does not match the fixed vocabulary order")
        if not 0 <= metric["accuracy"] <= 1 or not 0 <= metric["macro_f1"] <= 1:
            raise ArtifactError(f"{model_name} scalar metric is outside [0, 1]")
        for event in EVENT_TYPES:
            record = metric["per_class"][event]
            if set(record) != {"precision", "recall", "f1", "support"}:
                raise ArtifactError(f"{model_name} per-class fields do not match the v2 contract")
            if type(record["support"]) is not int or record["support"] < 0:
                raise ArtifactError(f"{model_name} class support is invalid")
            if any(
                isinstance(record[name], bool)
                or not isinstance(record[name], (int, float))
                or not 0 <= record[name] <= 1
                for name in ("precision", "recall", "f1")
            ):
                raise ArtifactError(f"{model_name} per-class metric is invalid")
        supports = [metric["per_class"][event]["support"] for event in EVENT_TYPES]
        if supports != [sum(matrix[index]) for index in range(len(EVENT_TYPES))]:
            raise ArtifactError(f"{model_name} class supports contradict its confusion matrix")
        if sum(supports) != evaluation["example_count"]:
            raise ArtifactError(f"{model_name} support does not equal the test-example count")
        total = sum(sum(row) for row in matrix)
        accuracy = sum(matrix[index][index] for index in range(len(EVENT_TYPES))) / total
        f1_values: list[float] = []
        for index, event in enumerate(EVENT_TYPES):
            true_positive = matrix[index][index]
            predicted = sum(row[index] for row in matrix)
            actual = sum(matrix[index])
            precision = true_positive / predicted if predicted else 0.0
            recall = true_positive / actual if actual else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            persisted = metric["per_class"][event]
            if not all(np.isclose(persisted[name], value, rtol=0, atol=1e-12) for name, value in (("precision", precision), ("recall", recall), ("f1", f1))):
                raise ArtifactError(f"{model_name} per-class metrics contradict its confusion matrix")
            f1_values.append(f1)
        if not np.isclose(metric["accuracy"], accuracy, rtol=0, atol=1e-12):
            raise ArtifactError(f"{model_name} accuracy contradicts its confusion matrix")
        if not np.isclose(metric["macro_f1"], statistics.fmean(f1_values), rtol=0, atol=1e-12):
            raise ArtifactError(f"{model_name} macro-F1 contradicts its confusion matrix")
    bootstrap = evaluation["bootstrap"]
    if set(bootstrap) != {"cluster_count", "replicates", "seed", "intervals_95"}:
        raise ArtifactError("bootstrap fields do not match the v2 contract")
    if any(type(bootstrap[name]) is not int for name in ("cluster_count", "replicates", "seed")):
        raise ArtifactError("bootstrap integer evidence is invalid")
    if bootstrap["cluster_count"] != seed_result["split"]["test_sequences"] or bootstrap["replicates"] < 20:
        raise ArtifactError("bootstrap cluster count or replicate count is invalid")
    expected_intervals = {
        *(f"{model}.{metric}" for model in ("global_frequency", "markov", "logistic_regression") for metric in ("accuracy", "macro_f1")),
        *(f"logistic_minus_{baseline}.{metric}" for baseline in ("global_frequency", "markov") for metric in ("accuracy", "macro_f1")),
    }
    if set(bootstrap["intervals_95"]) != expected_intervals:
        raise ArtifactError("bootstrap intervals do not cover the approved models and paired comparisons")
    for name, interval in bootstrap["intervals_95"].items():
        if set(interval) != {"lower", "upper"} or interval["lower"] > interval["upper"]:
            raise ArtifactError(f"bootstrap interval is invalid: {name}")
        lower_bound = -1 if name.startswith("logistic_minus_") else 0
        if interval["lower"] < lower_bound or interval["upper"] > 1:
            raise ArtifactError(f"bootstrap interval is outside its metric bounds: {name}")
    _validate_error_analysis(evaluation["error_analysis"], evaluation["example_count"])
    _require_finite_numbers(seed_result)


def _validate_error_analysis(analysis: Any, example_count: int) -> None:
    if not isinstance(analysis, dict) or set(analysis) != {"semantic_categories", "true_predicted_pairs", "prefix_length_bands"}:
        raise ArtifactError("error-analysis fields do not match the v2 contract")
    for section in analysis.values():
        if not isinstance(section, dict) or set(section) != {"count", "examples", "groups"}:
            raise ArtifactError("error-analysis section fields do not match the v2 contract")
        if type(section["count"]) is not int or not 0 <= section["count"] <= example_count or section["examples"] != []:
            raise ArtifactError("error-analysis section count or examples is invalid")
    error_count = analysis["true_predicted_pairs"]["count"]
    if any(section["count"] != error_count for section in analysis.values()):
        raise ArtifactError("error-analysis section counts disagree")
    pair_keys = {f"{true}->{predicted}" for true in EVENT_TYPES for predicted in EVENT_TYPES if true != predicted}
    pairs = analysis["true_predicted_pairs"]["groups"]
    if set(pairs) != pair_keys or any(type(value) is not int or value < 0 for value in pairs.values()) or sum(pairs.values()) != error_count:
        raise ArtifactError("true/predicted error counts are incomplete or inconsistent")
    bands = analysis["prefix_length_bands"]["groups"]
    if set(bands) != {"short", "medium", "long"} or any(type(value) is not int or value < 0 for value in bands.values()) or sum(bands.values()) != error_count:
        raise ArtifactError("prefix-length error counts are incomplete or inconsistent")
    categories = analysis["semantic_categories"]["groups"]
    expected_categories = {"rare_target", "post_failure_recovery", "readiness_ambiguity", "deployment_rollback_ambiguity", "other"}
    if set(categories) != expected_categories:
        raise ArtifactError("semantic error categories are incomplete")
    for category, group in categories.items():
        if not isinstance(group, dict) or set(group) != {"count", "examples"}:
            raise ArtifactError(f"semantic error category is malformed: {category}")
        if (type(group["count"]) is not int or group["count"] < len(group["examples"])
                or group["count"] > error_count or len(group["examples"]) > 5):
            raise ArtifactError(f"semantic error category count/examples are invalid: {category}")
        for record in group["examples"]:
            if set(record) != {"confidence_margin", "example_key", "predicted", "prefix", "prefix_length_band", "target"}:
                raise ArtifactError(f"semantic error example is malformed: {category}")
            if record["target"] not in EVENT_TYPES or record["predicted"] not in EVENT_TYPES or record["target"] == record["predicted"]:
                raise ArtifactError(f"semantic error example labels are invalid: {category}")
            if record["prefix_length_band"] not in {"short", "medium", "long"}:
                raise ArtifactError(f"semantic error example band is invalid: {category}")


# @cpt-state:cpt-event-prediction-state-benchmarking-evaluation-run:p1
def validate_benchmark_manifest(
    output_dir: Path, config: GeneratorConfig, *, _manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate a published pointer and every hash-bound required artifact."""
    manifest = read_json(output_dir / "benchmark.json") if _manifest is None else _manifest
    validate_schema("experiment-manifest.v2", manifest)
    if manifest["status"] == "INCOMPLETE":
        return manifest
    profile = _frozen_final_profile()
    if manifest.get("final_profile") != profile or manifest.get("final_profile_hash") != _profile_hash(profile):
        raise ArtifactError("benchmark is not bound to the frozen final protocol profile")
    if manifest["provenance"].get("final_profile_hash") != manifest["final_profile_hash"]:
        raise ArtifactError("benchmark provenance is not bound to the final profile hash")
    if manifest["provenance"].get("config_hash") != config.digest():
        raise ArtifactError("benchmark provenance config hash does not match the effective configuration")
    if manifest["provenance"].get("seeds") != manifest["seeds"]:
        raise ArtifactError("benchmark provenance seeds do not match the ordered experiment seeds")
    if manifest["event_vocabulary"] != list(EVENT_TYPES):
        raise ArtifactError("benchmark event vocabulary does not match the fixed catalog order")
    run_root = resolve_portable_reference(output_dir, manifest["run"], field="run reference")
    if not run_root.is_dir():
        raise ArtifactError("benchmark run reference is not a published run directory")
    for reference, digest in manifest["artifacts"].items():
        path = resolve_portable_reference(output_dir, reference)
        try:
            path.relative_to(run_root)
        except ValueError as exc:
            raise ArtifactError(f"published artifact is outside the declared run root: {reference}") from exc
        if sha256_file(path) != digest:
            raise ArtifactError(f"published artifact hash mismatch: {reference}")
    required = {manifest["aggregate_result"], manifest["report"], *manifest["seed_results"]}
    if not required <= set(manifest["artifacts"]):
        raise ArtifactError("benchmark manifest omits a required artifact hash")
    for reference in required:
        path = resolve_portable_reference(output_dir, reference)
        try:
            path.relative_to(run_root)
        except ValueError as exc:
            raise ArtifactError(f"required artifact is outside the declared run root: {reference}") from exc
    aggregate = read_json(resolve_portable_reference(output_dir, manifest["aggregate_result"]))
    validate_schema("aggregate-result.v2", aggregate)
    if canonical_json(manifest["aggregate"]) != canonical_json(aggregate):
        raise ArtifactError("inline aggregate contradicts the hash-bound aggregate artifact")
    if aggregate["seed_count"] != len(manifest["seeds"]) or len(manifest["seed_results"]) != len(manifest["seeds"]):
        raise ArtifactError("benchmark does not contain exactly one aggregate and seed result per declared seed")
    validated_seeds: list[int] = []
    validated_results: list[dict[str, Any]] = []
    for reference in manifest["seed_results"]:
        seed_result = read_json(resolve_portable_reference(output_dir, reference, field="seed result reference"))
        validate_schema("seed-result.v2", seed_result)
        _validate_evaluation_structure(seed_result)
        validated_seeds.append(seed_result["seed"])
        validated_results.append(seed_result)
        split_path = resolve_portable_reference(run_root, seed_result["split_manifest"], field="split manifest reference")
        split_ref = split_path.relative_to(output_dir.resolve()).as_posix()
        if split_ref not in manifest["artifacts"]:
            raise ArtifactError("split manifest is absent from the transitive artifact hash closure")
        if sha256_file(split_path) != seed_result["split_manifest_sha256"]:
            raise ArtifactError("published split manifest hash mismatch")
        split_value = read_json(split_path)
        validate_schema("split-manifest.v1", split_value)
        split = SplitManifest(
            seed=split_value["seed"], train_ids=tuple(split_value["train_ids"]),
            validation_ids=tuple(split_value["validation_ids"]), test_ids=tuple(split_value["test_ids"]),
        )
        try:
            split.validate()
        except (ConfigurationError, TypeError, ValueError) as exc:
            raise ArtifactError(f"invalid persisted split manifest: {exc}") from exc
        if split.seed != seed_result["seed"] + 10_000:
            raise ArtifactError("split seed is not experimental seed + 10000")
        expected = _expected_split_ids(manifest["count"], split.seed)
        if (list(split.train_ids), list(split.validation_ids), list(split.test_ids)) != expected:
            raise ArtifactError("split assignments do not match the deterministic 70/15/15 allocation")
        counts = seed_result["split"]
        if (len(split.train_ids), len(split.validation_ids), len(split.test_ids)) != (counts["train_sequences"], counts["validation_sequences"], counts["test_sequences"]):
            raise ArtifactError("split manifest counts do not match seed result")
        if len(split.train_ids) + len(split.validation_ids) + len(split.test_ids) != manifest["count"]:
            raise ArtifactError("split manifest does not cover the declared sequence count")
        expected_counts = (int(manifest["count"] * .70), int(manifest["count"] * .15))
        expected_counts = (*expected_counts, manifest["count"] - sum(expected_counts))
        if (counts["train_sequences"], counts["validation_sequences"], counts["test_sequences"]) != expected_counts:
            raise ArtifactError("split counts violate the exact 70/15/15 allocation rule")
        lengths = seed_result["test_sequence_lengths"]
        if len(lengths) != len(split.test_ids) or any(type(value) is not int or value < 2 for value in lengths):
            raise ArtifactError("test sequence-length evidence is invalid")
        generated = generate_sequences(manifest["count"], seed_result["seed"], config)
        generated_lengths = {sequence.sequence_id: len(sequence.events) for sequence in generated}
        if lengths != [generated_lengths[identifier] for identifier in split.test_ids]:
            raise ArtifactError("test sequence-length evidence does not match deterministic generation")
        keys = [f"{identifier}:{position}" for identifier, length in zip(split.test_ids, lengths, strict=True) for position in range(1, length)]
        if len(keys) != seed_result["test_example_count"] or len(keys) != seed_result["evaluation"]["example_count"]:
            raise ArtifactError("test example count contradicts ordered split evidence")
        if hashlib.sha256(canonical_json(keys).encode("utf-8")).hexdigest() != seed_result["test_example_keys_sha256"]:
            raise ArtifactError("ordered test-example key hash does not match split evidence")
        model_manifest = resolve_portable_reference(run_root, seed_result["model"]["manifest"], field="model manifest reference")
        model_ref = model_manifest.relative_to(output_dir.resolve()).as_posix()
        arrays_ref = resolve_portable_reference(run_root, seed_result["model"]["arrays"], field="model arrays reference").relative_to(output_dir.resolve()).as_posix()
        if model_ref not in manifest["artifacts"] or arrays_ref not in manifest["artifacts"]:
            raise ArtifactError("model evidence is absent from the transitive artifact hash closure")
        persisted_model = read_json(model_manifest)
        if canonical_json({key: value for key, value in seed_result["model"].items() if key != "manifest"}) != canonical_json(persisted_model):
            raise ArtifactError("seed-result model metadata contradicts its model manifest")
        if seed_result["model"]["provenance"].get("config_hash") != config.digest() or seed_result["model"]["provenance"].get("final_profile_hash") != manifest["final_profile_hash"]:
            raise ArtifactError("model provenance is not bound to the benchmark profile/configuration")
        if seed_result["model"]["provenance"].get("split_seed") != split.seed or seed_result["model"].get("model_seed") != seed_result["seed"] + 20_000:
            raise ArtifactError("model seed provenance does not match the benchmark seed protocol")
        regenerated_split, regenerated_examples = split_and_build_examples(generated, split.seed)
        if regenerated_split.as_dict() != split_value:
            raise ArtifactError("split manifest does not match deterministic regenerated evidence")
        if seed_result["model"]["provenance"].get("training_examples_hash") != _examples_hash(regenerated_examples["train"]):
            raise ArtifactError("model training-example provenance does not match the deterministic training split")
        if seed_result["evaluation"]["bootstrap"]["seed"] != seed_result["seed"] + 30_000:
            raise ArtifactError("bootstrap seed provenance does not match the benchmark seed protocol")
        if seed_result["evaluation"]["bootstrap"]["replicates"] != config.bootstrap_replicates:
            raise ArtifactError("bootstrap replicate evidence does not match the effective configuration")
        evidence_fields = {
            "prediction_evidence", "prediction_evidence_schema", "prediction_evidence_sha256"
        }
        present_evidence_fields = evidence_fields & set(seed_result)
        if present_evidence_fields and present_evidence_fields != evidence_fields:
            raise ArtifactError("seed result contains incomplete prediction-evidence metadata")
        baseline_predictors: dict[str, Any] | None = None
        baseline_value = seed_result.get("baseline_state")
        if baseline_value is None:
            if manifest["status"] == "COMPLETE" or present_evidence_fields:
                raise ArtifactError("strict prediction evidence requires persisted baseline state")
        else:
            if not isinstance(baseline_value, dict) or "manifest" not in baseline_value:
                raise ArtifactError("baseline-state seed metadata is malformed")
            baseline_manifest = resolve_portable_reference(
                run_root, baseline_value["manifest"], field="baseline-state manifest reference"
            )
            baseline_manifest_ref = baseline_manifest.relative_to(output_dir.resolve()).as_posix()
            baseline_arrays = resolve_portable_reference(
                run_root, baseline_value.get("arrays"), field="baseline-state arrays reference"
            )
            baseline_arrays_ref = baseline_arrays.relative_to(output_dir.resolve()).as_posix()
            if baseline_manifest_ref not in manifest["artifacts"] or baseline_arrays_ref not in manifest["artifacts"]:
                raise ArtifactError("baseline state is absent from the transitive artifact hash closure")
            persisted_baseline, baseline_predictors = _load_baseline_state(baseline_manifest)
            if canonical_json({key: value for key, value in baseline_value.items() if key != "manifest"}) != canonical_json(persisted_baseline):
                raise ArtifactError("seed-result baseline metadata contradicts its manifest")
            if not np.isclose(float(persisted_baseline["alpha"]), config.laplace_alpha, rtol=0, atol=0):
                raise ArtifactError("baseline-state alpha contradicts the effective configuration")
            expected_baseline_provenance = {
                "config_hash": config.digest(),
                "split_seed": split.seed,
                "training_examples_hash": _examples_hash(regenerated_examples["train"]),
                "final_profile_hash": manifest["final_profile_hash"],
                "training_protocol": BASELINE_TRAINING_PROTOCOL,
            }
            if persisted_baseline["provenance"] != expected_baseline_provenance:
                raise ArtifactError("baseline-state provenance contradicts the trusted training protocol")
        loaded_model = load_model(model_manifest)
        if not present_evidence_fields:
            if manifest["status"] == "COMPLETE":
                raise ArtifactError("COMPLETE benchmark requires strict keyed prediction evidence")
        else:
            if seed_result["prediction_evidence_schema"] != PREDICTION_EVIDENCE_SCHEMA:
                raise ArtifactError("unsupported prediction-evidence schema")
            evidence_path = resolve_portable_reference(
                run_root, seed_result["prediction_evidence"], field="prediction evidence reference"
            )
            evidence_ref = evidence_path.relative_to(output_dir.resolve()).as_posix()
            if evidence_ref not in manifest["artifacts"]:
                raise ArtifactError("prediction evidence is absent from the transitive artifact hash closure")
            batches = _load_prediction_evidence(
                evidence_path, seed_result["prediction_evidence_sha256"], regenerated_examples["test"]
            )
            if baseline_predictors is None:
                raise ArtifactError("prediction evidence cannot be verified without persisted baseline state")
            for baseline_name in ("global_frequency", "markov"):
                baseline_recomputed = baseline_predictors[baseline_name].predict_examples(regenerated_examples["test"])
                if (baseline_recomputed.example_keys != batches[baseline_name].example_keys
                        or not np.allclose(
                            baseline_recomputed.probabilities,
                            batches[baseline_name].probabilities,
                            rtol=0,
                            atol=1e-12,
                        )):
                    raise ArtifactError(
                        f"persisted {baseline_name} predictions contradict safely loaded baseline state"
                    )
            logistic_recomputed = loaded_model.predict_examples(regenerated_examples["test"])
            if (logistic_recomputed.example_keys != batches["logistic_regression"].example_keys
                    or not np.allclose(
                        logistic_recomputed.probabilities,
                        batches["logistic_regression"].probabilities,
                        rtol=0,
                        atol=1e-12,
                    )):
                raise ArtifactError("persisted logistic predictions contradict the safely loaded saved model")
            recomputed_evaluation = evaluate_models(
                regenerated_examples["test"], batches, seed_result["seed"] + 30_000,
                config.bootstrap_replicates,
            )
            if canonical_json(recomputed_evaluation) != canonical_json(seed_result["evaluation"]):
                raise ArtifactError("published evaluation contradicts keyed prediction evidence and held-out targets")
        if seed_result["dataset_sequence_count"] != manifest["count"]:
            raise ArtifactError("seed result sequence count does not match experiment manifest")
    if validated_seeds != manifest["seeds"]:
        raise ArtifactError("seed results do not exactly match the ordered declared seeds")
    recomputed = _aggregate(validated_results)
    if canonical_json(recomputed) != canonical_json(aggregate):
        raise ArtifactError("aggregate evidence is stale or contradicts validated seed results")
    expected_status = _completion_status(manifest["count"], manifest["seeds"], config)
    if manifest["status"] != expected_status:
        raise ArtifactError("benchmark completion status does not satisfy the declared completion predicate")
    if manifest["status"] == "COMPLETE" and aggregate["seed_count"] != 5:
        raise ArtifactError("COMPLETE benchmark aggregate must contain exactly five seeds")
    report_path = resolve_portable_reference(output_dir, manifest["report"], field="report reference")
    try:
        report = report_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ArtifactError(f"cannot read benchmark report: {exc}") from exc
    if report != _report(aggregate, manifest["status"], manifest["count"], manifest["seeds"]):
        raise ArtifactError("Markdown report is stale or contradicts the validated JSON evidence")
    return manifest


# @cpt-flow:cpt-event-prediction-flow-benchmarking-evaluation-run:p1
def run_benchmark(
    count: int,
    seeds: Sequence[int],
    config: GeneratorConfig,
    output_dir: Path,
    replace: bool = False,
) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("benchmark seeds must be non-empty and distinct")
    output_dir.mkdir(parents=True, exist_ok=True)
    pointer = output_dir / "benchmark.json"
    if pointer.exists() and not replace:
        raise ValueError(f"refusing to overwrite existing benchmark pointer: {pointer}; pass --replace")
    identity = {"config_hash": config.digest(), "count": count, "seeds": list(seeds)}
    run_id = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()[:20]
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    candidate = run_id
    suffix = 1
    while (runs_dir / candidate).exists():
        if not replace:
            raise ValueError(f"benchmark run already exists: {candidate}; pass --replace")
        candidate = f"{run_id}-{suffix}"
        suffix += 1
    run_id = candidate
    stage = output_dir / f".staging-{run_id}-{uuid.uuid4().hex}"
    stage.mkdir()
    old_pointer = pointer.read_bytes() if pointer.exists() else None
    pointer_published = False
    run_dir: Path | None = None
    try:
        seed_results = [run_seed(count, seed, config, stage, replace=False) for seed in seeds]
        aggregate = _aggregate(seed_results)
        write_json(stage / "aggregate.json", aggregate)
        status = _completion_status(count, seeds, config)
        write_bytes(stage / "evaluation-report.md", _report(aggregate, status, count, seeds).encode("utf-8"))
        hashes = _validate_staged_run(stage, seed_results, aggregate)
        run_dir = runs_dir / run_id
        stage.replace(run_dir)
        artifact_hashes = {f"runs/{run_id}/{name}": digest for name, digest in hashes.items()}
        result = {
            "aggregate": aggregate,
            "aggregate_result": f"runs/{run_id}/aggregate.json",
            "artifacts": artifact_hashes,
            "count": count,
            "event_vocabulary": list(EVENT_TYPES),
            "final_profile": _frozen_final_profile(),
            "final_profile_hash": _profile_hash(_frozen_final_profile()),
            "provenance": _provenance(config, seeds),
            "report": f"runs/{run_id}/evaluation-report.md",
            "run": f"runs/{run_id}",
            "schema_version": RESULT_SCHEMA,
            "seed_results": [f"runs/{run_id}/seed-{seed}.json" for seed in seeds],
            "seeds": list(seeds),
            "status": status,
        }
        validate_schema("experiment-manifest.v2", result)
        for relative, digest in result["artifacts"].items():
            artifact_path = resolve_portable_reference(output_dir, relative)
            if sha256_file(artifact_path) != digest:
                raise ValueError(f"published artifact hash mismatch: {relative}")
        # Validate the entire candidate closure and all semantic relationships
        # before publishing the canonical pointer.
        validate_benchmark_manifest(output_dir, config, _manifest=result)
        write_json(pointer, result, replace=True)
        pointer_published = True
        validate_benchmark_manifest(output_dir, config)
        return result
    except BaseException as exc:
        recovery_notes: list[str] = []
        # The canonical pointer is authoritative.  Recover it before attempting
        # diagnostics or cleanup, neither of which may mask the primary error.
        if pointer_published:
            try:
                if old_pointer is None:
                    pointer.unlink(missing_ok=True)
                else:
                    write_bytes(pointer, old_pointer, replace=True)
            except Exception as recovery_error:
                recovery_notes.append(f"restore benchmark pointer: {recovery_error}")
        for disposable in (stage, run_dir):
            if disposable is None:
                continue
            try:
                if disposable.exists():
                    shutil.rmtree(disposable)
            except Exception as cleanup_error:
                recovery_notes.append(f"cleanup {disposable}: {cleanup_error}")
        try:
            failure_dir = output_dir / "failures"
            failure_dir.mkdir(exist_ok=True)
            failure = {
                "artifacts": {}, "count": count, "error": f"{type(exc).__name__}: {exc}",
                "provenance": _provenance(config, seeds), "run": f"failures/{run_id}.incomplete.json",
                "schema_version": RESULT_SCHEMA, "seeds": list(seeds), "status": "INCOMPLETE",
            }
            write_json(failure_dir / f"{run_id}.incomplete.json", failure, replace=True)
        except Exception as diagnostic_error:
            recovery_notes.append(f"write failure diagnostic: {diagnostic_error}")
        if recovery_notes and hasattr(exc, "add_note"):
            exc.add_note("benchmark transaction recovery: " + "; ".join(recovery_notes))
        raise
