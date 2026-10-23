from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

import event_prediction.benchmark as benchmark_module
from event_prediction.artifacts import resolve_portable_reference, sha256_file, write_dataset, write_json
from event_prediction.baselines import GlobalFrequencyPredictor, MarkovPredictor
from event_prediction.benchmark import (
    FINAL_CONFIG_HASH,
    FINAL_SEEDS,
    _completion_status,
    _validate_error_analysis,
    _validate_evaluation_structure,
    run_benchmark,
    validate_benchmark_manifest,
)
from event_prediction.catalog import EVENT_TYPES
from event_prediction.cli import main
from event_prediction.config import GeneratorConfig, load_config
from event_prediction.domain import ArtifactError, ConfigurationError, PredictionBatch, PrefixExample
from event_prediction.evaluation import analyze_errors, evaluate_models
from event_prediction.model import _deterministic_npz, load_model, save_model, train_logistic
from event_prediction.simulator import LifecycleState, SimulatorContext, _update_context, generate_sequences
from event_prediction.schemas import validate_schema


def _uniform_batch(examples: list[PrefixExample]) -> PredictionBatch:
    values = np.full((len(examples), len(EVENT_TYPES)), 1 / len(EVENT_TYPES))
    return PredictionBatch(tuple(item.key for item in examples), values)


@pytest.fixture(scope="module")
def valid_seed_result(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    output = tmp_path_factory.mktemp("valid-seed-result")
    config = GeneratorConfig(min_length=4, max_length=8, bootstrap_replicates=20, regularization_candidates=(0.1,))
    result = run_benchmark(70, (151,), config, output)
    return json.loads((output / result["seed_results"][0]).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def valid_benchmark_tree(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, GeneratorConfig]:
    output = tmp_path_factory.mktemp("valid-benchmark-tree")
    config = GeneratorConfig(min_length=4, max_length=8, bootstrap_replicates=20, regularization_candidates=(0.1,))
    run_benchmark(70, (161,), config, output)
    return output, config


# @cpt-state:cpt-event-prediction-state-synthetic-data-generation-lifecycle:p1
# Scenario: cpt-event-prediction-state-synthetic-data-generation-lifecycle
def test_rollback_is_unconditionally_terminal_and_last() -> None:
    context = SimulatorContext(state=LifecycleState.DEPLOYED)
    _update_context(context, "rolled_back")
    assert context.state is LifecycleState.TERMINAL
    sequences = generate_sequences(300, 712, GeneratorConfig(min_length=4, max_length=12))
    rolled_back = [item.event_types() for item in sequences if "rolled_back" in item.event_types()]
    assert rolled_back
    assert all(events[-1] == "rolled_back" for events in rolled_back)


# @cpt-algo:cpt-event-prediction-algo-benchmarking-evaluation-metrics:p1
# Scenario: cpt-event-prediction-algo-benchmarking-evaluation-metrics
def test_prediction_permutation_is_rejected() -> None:
    examples = [
        PrefixExample("a", (EVENT_TYPES[0],), EVENT_TYPES[1]),
        PrefixExample("b", (EVENT_TYPES[1],), EVENT_TYPES[2]),
    ]
    batch = _uniform_batch(examples)
    permuted = PredictionBatch(tuple(reversed(batch.example_keys)), batch.probabilities)
    with pytest.raises(ConfigurationError, match="ordered"):
        evaluate_models(examples, {"logistic_regression": permuted}, 1, 20)


# @cpt-flow:cpt-event-prediction-flow-ml-prediction-train-select:p1
# Scenario: cpt-event-prediction-flow-ml-prediction-train-select
def test_transformer_fits_training_only_and_selection_has_no_test_input() -> None:
    training = [PrefixExample(f"t-{repeat}-{index}", (EVENT_TYPES[index],), target) for repeat in range(2) for index, target in enumerate(EVENT_TYPES)]
    validation = [PrefixExample(f"v-{index}", (EVENT_TYPES[0], EVENT_TYPES[0]), target) for index, target in enumerate(EVENT_TYPES)]
    model = train_logistic(training, validation, (0.1, 1.0), seed=33)
    assert "transition::api_interface_modified->api_interface_modified" not in model.vectorizer.feature_names_
    assert model.selected_c in {0.1, 1.0}
    assert all("test" not in key for row in model.selection_report for key in row)
    held_out_a = [PrefixExample("held", (EVENT_TYPES[0],), EVENT_TYPES[1])]
    held_out_b = [PrefixExample("held", (EVENT_TYPES[0],), EVENT_TYPES[2])]
    assert np.array_equal(model.predict_examples(held_out_a).probabilities, model.predict_examples(held_out_b).probabilities)


def test_config_rejects_bool_and_non_numeric_candidates(tmp_path: Path) -> None:
    for value, field in ((True, "min_length"), (False, "bootstrap_replicates")):
        path = tmp_path / f"{field}.json"
        path.write_text(json.dumps({field: value}), encoding="utf-8")
        with pytest.raises(ConfigurationError, match=field):
            load_config(path)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps({"regularization_candidates": ["1.0"]}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="regularization_candidates"):
        load_config(path)


def test_cli_reports_field_specific_config_error_without_traceback(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "bad-config.json"
    config_path.write_text('{"min_length":true}', encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        main(["generate", "--config", str(config_path), "--count", "12", "--seed", "1", "--output", str(tmp_path / "data.jsonl")])
    assert caught.value.code == 2
    stderr = capsys.readouterr().err
    assert "min_length must be an integer" in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize(
    ("schema_name", "value", "message"),
    [
        ("dataset-record.v1", [], "expected object"),
        ("dataset-record.v1", {}, "missing required"),
        ("dataset-record.v1", {"events": [{"description": "x", "event_type": "x", "position": 0}, {"description": "y", "event_type": "y", "position": 1}], "provenance": {"generation_seed": 1}, "schema_version": "bad", "sequence_id": "x"}, "constant"),
        ("dataset-record.v1", {"events": [], "provenance": {"generation_seed": 1}, "schema_version": "event-sequence.v1", "sequence_id": "x"}, "too few"),
        ("split-manifest.v1", {"schema_version": "split-manifest.v1", "seed": 1, "train_ids": ["x", "x"], "validation_ids": [], "test_ids": []}, "unique"),
        ("model-manifest.v2", {"schema_version": "event-predictor.v2", "arrays": "x", "arrays_sha256": "bad", "classes": [], "feature_names": [], "selected_c": 1.0, "provenance": {}}, "pattern"),
        ("aggregate-result.v2", {"schema_version": "benchmark-aggregate.v2", "models": {}, "paired_comparisons": {}, "seed_count": 0}, "minimum"),
        ("experiment-manifest.v2", {"schema_version": "benchmark-result.v2", "status": "BAD", "count": 1, "seeds": [1], "run": "x", "artifacts": {}, "provenance": {}}, "enum"),
    ],
)
def test_json_schema_validation_rejects_invalid_documents(schema_name: str, value: object, message: str) -> None:
    with pytest.raises(ArtifactError, match=message):
        validate_schema(schema_name, value)


@pytest.mark.parametrize(
    "changes",
    [
        {"final_seeds": (1, 1, 2, 3, 4)},
        {"event_catalog": {}},
        {"initial_weights": {"api_interface_modified": True, "dependency_upgraded": 1.0}},
        {"transition_weights": {}},
        {"history_modifiers": {}},
        {"termination_policy": {"end_token": "__end__", "rollback_is_terminal": False}},
    ],
)
def test_effective_simulator_config_rejects_invalid_sections(changes: dict[str, object]) -> None:
    values = GeneratorConfig().__dict__ | changes
    with pytest.raises(ConfigurationError):
        GeneratorConfig(**values).validate()


# @cpt-dod:cpt-event-prediction-dod-benchmarking-evaluation-errors:p1
# Scenario: cpt-event-prediction-dod-benchmarking-evaluation-errors
def test_tiny_benchmark_persists_portable_auditable_evidence(tmp_path: Path) -> None:
    config = GeneratorConfig(min_length=4, max_length=8, bootstrap_replicates=20, regularization_candidates=(0.1,))
    result = run_benchmark(90, (51,), config, tmp_path)
    assert result["status"] == "PRELIMINARY"
    assert all(not Path(reference).is_absolute() for reference in result["artifacts"])
    assert all(not Path(reference).is_absolute() for reference in result["seed_results"])
    for reference, digest in result["artifacts"].items():
        assert hashlib.sha256((tmp_path / reference).read_bytes()).hexdigest() == digest
    seed = json.loads((tmp_path / result["seed_results"][0]).read_text(encoding="utf-8"))
    split = json.loads((tmp_path / result["run"] / seed["split_manifest"]).read_text(encoding="utf-8"))
    groups = [set(split[name]) for name in ("train_ids", "validation_ids", "test_ids")]
    assert [len(group) for group in groups] == [62, 13, 15]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert len(set().union(*groups)) == 90
    model_path = tmp_path / result["run"] / seed["model"]["manifest"]
    loaded = load_model(model_path)
    assert loaded.selected_c == 0.1
    validate_benchmark_manifest(tmp_path, config)
    report_path = tmp_path / result["report"]
    report_payload = report_path.read_bytes()
    report_path.unlink()
    with pytest.raises(ArtifactError, match="cannot hash"):
        validate_benchmark_manifest(tmp_path, config)
    report_path.write_bytes(report_payload)
    arrays_path = model_path.parent / seed["model"]["arrays"]
    arrays_path.write_bytes(arrays_path.read_bytes() + b"corrupt")
    with pytest.raises(ArtifactError, match="hash"):
        validate_benchmark_manifest(tmp_path, config)


def test_completion_predicate_requires_declared_seeds_and_full_count() -> None:
    config = GeneratorConfig()
    assert config.digest() == FINAL_CONFIG_HASH
    assert load_config(Path("configs/default.json")).digest() == FINAL_CONFIG_HASH
    assert config.final_seeds == FINAL_SEEDS
    assert _completion_status(10_000, config.final_seeds, config) == "COMPLETE"
    assert _completion_status(9999, config.final_seeds, config) == "PRELIMINARY"
    assert _completion_status(10_000, (1, 2, 3, 4, 5), config) == "PRELIMINARY"
    assert _completion_status(10_000, tuple(reversed(config.final_seeds)), config) == "PRELIMINARY"
    assert _completion_status(10_000, config.final_seeds, GeneratorConfig(history_window=3)) == "PRELIMINARY"


@pytest.mark.parametrize("key", sorted(GeneratorConfig().history_modifiers["fallback_weights"]))
@pytest.mark.parametrize("bad_value", [True, False, 0, -0.1, float("nan"), float("inf")])
def test_fallback_weight_mutation_matrix_is_field_specific(key: str, bad_value: object) -> None:
    config = GeneratorConfig()
    modifiers = json.loads(json.dumps(config.history_modifiers))
    modifiers["fallback_weights"][key] = bad_value
    with pytest.raises(ConfigurationError, match=rf"fallback_weights\.{key}"):
        GeneratorConfig(**(config.__dict__ | {"history_modifiers": modifiers})).validate()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_fallback_weight_key_set_is_exact(mutation: str) -> None:
    config = GeneratorConfig()
    modifiers = json.loads(json.dumps(config.history_modifiers))
    if mutation == "missing":
        modifiers["fallback_weights"].pop("rolled_back")
        message = "missing keys"
    else:
        modifiers["fallback_weights"]["surprise"] = 1.0
        message = "unexpected keys"
    with pytest.raises(ConfigurationError, match=message):
        GeneratorConfig(**(config.__dict__ | {"history_modifiers": modifiers})).validate()


@pytest.mark.parametrize(
    "reference",
    ["/absolute.json", "C:/absolute.json", "C:drive-relative.json", "../escape.json", "a/../escape.json", "a\\escape.json", "a//unnormalized.json", "./local.json"],
)
def test_portable_reference_rejects_noncanonical_paths(tmp_path: Path, reference: str) -> None:
    with pytest.raises(ArtifactError, match="portable relative|drive-relative|outside"):
        resolve_portable_reference(tmp_path, reference)


def test_portable_reference_rejects_symlink_escape(tmp_path: Path, monkeypatch) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        # Windows commonly denies symlink creation without Developer Mode. Model
        # the same resolved-path escape so containment behavior remains tested.
        real_resolve = Path.resolve
        escaped_candidate = link / "evidence.json"

        def resolve_with_escape(path: Path, *args, **kwargs) -> Path:
            if path == escaped_candidate:
                return outside.resolve() / "evidence.json"
            return real_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", resolve_with_escape)
    with pytest.raises(ArtifactError, match="outside"):
        resolve_portable_reference(tmp_path, "link/evidence.json")


@pytest.mark.parametrize("with_existing_pair", [False, True])
def test_dataset_second_publish_failure_preserves_complete_pair(
    tmp_path: Path, monkeypatch, with_existing_pair: bool
) -> None:
    import event_prediction.artifacts as artifacts_module

    output = tmp_path / "dataset.jsonl"
    manifest = output.with_suffix(".jsonl.manifest.json")
    if with_existing_pair:
        output.write_bytes(b"old dataset\n")
        manifest.write_bytes(b"old manifest\n")
    old_dataset = output.read_bytes() if output.exists() else None
    old_manifest = manifest.read_bytes() if manifest.exists() else None
    real_replace = artifacts_module.os.replace

    def fail_second_publish(source, destination):
        if Path(destination).resolve() == manifest.resolve() and str(source).endswith(".candidate"):
            raise OSError("injected second publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(artifacts_module.os, "replace", fail_second_publish)
    sequences = generate_sequences(12, 71, GeneratorConfig(min_length=4, max_length=8))
    with pytest.raises(OSError, match="second publish"):
        write_dataset(sequences, output, GeneratorConfig(min_length=4, max_length=8), 71, replace=with_existing_pair)
    assert (output.read_bytes() if output.exists() else None) == old_dataset
    assert (manifest.read_bytes() if manifest.exists() else None) == old_manifest
    assert not list(tmp_path.glob("*.candidate"))
    assert not list(tmp_path.glob("*.rollback"))


def test_error_analysis_has_complete_deterministic_group_domains() -> None:
    examples = [
        PrefixExample("one", (EVENT_TYPES[1],), EVENT_TYPES[1]),
        PrefixExample("two", (EVENT_TYPES[2],) * 4, EVENT_TYPES[2]),
    ]
    probabilities = np.zeros((2, len(EVENT_TYPES)))
    probabilities[:, 0] = 1.0
    analysis = analyze_errors(examples, probabilities)
    assert len(analysis["true_predicted_pairs"]["groups"]) == len(EVENT_TYPES) * (len(EVENT_TYPES) - 1)
    assert set(analysis["prefix_length_bands"]["groups"]) == {"short", "medium", "long"}
    assert set(analysis["semantic_categories"]["groups"]) == {
        "rare_target", "post_failure_recovery", "readiness_ambiguity", "deployment_rollback_ambiguity", "other"
    }


def test_failed_seed_publishes_incomplete_without_touching_prior_run(tmp_path: Path, monkeypatch) -> None:
    prior = tmp_path / "runs" / "prior" / "evidence.txt"
    prior.parent.mkdir(parents=True)
    prior.write_text("preserve", encoding="utf-8")
    (tmp_path / "benchmark.json").write_text('{"status":"COMPLETE"}\n', encoding="utf-8")

    def fail_seed(*args, **kwargs):
        raise RuntimeError("injected partial failure")

    monkeypatch.setattr(benchmark_module, "run_seed", fail_seed)
    with pytest.raises(RuntimeError, match="injected"):
        run_benchmark(20, (7,), GeneratorConfig(bootstrap_replicates=20), tmp_path, replace=True)
    pointer = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert pointer["status"] == "COMPLETE"
    failures = list((tmp_path / "failures").glob("*.incomplete.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text(encoding="utf-8"))["status"] == "INCOMPLETE"
    assert prior.read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob(".staging-*"))


def test_postwrite_validation_failure_restores_pointer_and_removes_candidate_run(tmp_path: Path, monkeypatch) -> None:
    previous = b'{"status":"previous-valid-pointer"}\n'
    (tmp_path / "benchmark.json").write_bytes(previous)
    real_validate = benchmark_module.validate_benchmark_manifest

    def fail_after_pointer(output_dir, config, **kwargs):
        if kwargs.get("_manifest") is None:
            raise ArtifactError("injected postwrite validation failure")
        return real_validate(output_dir, config, **kwargs)

    monkeypatch.setattr(benchmark_module, "validate_benchmark_manifest", fail_after_pointer)
    config = GeneratorConfig(min_length=4, max_length=8, bootstrap_replicates=20, regularization_candidates=(0.1,))
    with pytest.raises(ArtifactError, match="postwrite"):
        run_benchmark(70, (81,), config, tmp_path, replace=True)
    assert (tmp_path / "benchmark.json").read_bytes() == previous
    assert not list((tmp_path / "runs").iterdir())
    assert len(list((tmp_path / "failures").glob("*.incomplete.json"))) == 1


def test_missing_nested_hash_closure_entry_is_rejected(tmp_path: Path) -> None:
    config = GeneratorConfig(min_length=4, max_length=8, bootstrap_replicates=20, regularization_candidates=(0.1,))
    result = run_benchmark(70, (91,), config, tmp_path)
    seed_result = json.loads((tmp_path / result["seed_results"][0]).read_text(encoding="utf-8"))
    split_ref = f"{result['run']}/{seed_result['split_manifest']}"
    pointer = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    pointer["artifacts"].pop(split_ref)
    write_json(tmp_path / "benchmark.json", pointer, replace=True)
    with pytest.raises(ArtifactError, match="hash closure"):
        validate_benchmark_manifest(tmp_path, config)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "coef_shape", "coef_dtype", "coef_nan", "intercept_inf", "classes_order", "scale_shape"],
)
def test_model_npz_contract_rejects_malformed_arrays(complete_examples, tmp_path: Path, mutation: str) -> None:
    model = train_logistic(
        complete_examples("train"), complete_examples("validation"), (0.1,), 101,
        provenance={"config_hash": "1" * 64, "split_seed": 102, "training_examples_hash": "2" * 64, "final_profile_hash": "3" * 64},
    )
    manifest_path = tmp_path / "model.json"
    metadata = save_model(model, manifest_path)
    arrays_path = tmp_path / metadata["arrays"]
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if mutation == "missing":
        arrays.pop("scale")
    elif mutation == "extra":
        arrays["surprise"] = np.asarray([1], dtype=np.int64)
    elif mutation == "coef_shape":
        arrays["coef"] = arrays["coef"][:, :-1]
    elif mutation == "coef_dtype":
        arrays["coef"] = arrays["coef"].astype(np.float32)
    elif mutation == "coef_nan":
        arrays["coef"][0, 0] = np.nan
    elif mutation == "intercept_inf":
        arrays["intercept"][0] = np.inf
    elif mutation == "classes_order":
        arrays["classes"] = arrays["classes"][::-1]
    elif mutation == "scale_shape":
        arrays["scale"] = arrays["scale"][:-1]
    payload = _deterministic_npz(arrays)
    arrays_path.write_bytes(payload)
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    persisted["arrays_sha256"] = hashlib.sha256(payload).hexdigest()
    write_json(manifest_path, persisted, replace=True)
    with pytest.raises(ArtifactError):
        load_model(manifest_path)


@pytest.mark.parametrize("mutation", ["missing", "extra", "class_order", "feature_count", "recency", "nonfinite"])
def test_model_manifest_contract_rejects_inconsistent_metadata(complete_examples, tmp_path: Path, mutation: str) -> None:
    model = train_logistic(
        complete_examples("train"), complete_examples("validation"), (0.1,), 111,
        provenance={"config_hash": "1" * 64, "split_seed": 112, "training_examples_hash": "2" * 64, "final_profile_hash": "3" * 64},
    )
    manifest_path = tmp_path / "model.json"
    save_model(model, manifest_path)
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        metadata.pop("model_seed")
    elif mutation == "extra":
        metadata["surprise"] = True
    elif mutation == "class_order":
        metadata["classes"] = list(reversed(metadata["classes"]))
    elif mutation == "feature_count":
        metadata["feature_count"] += 1
    elif mutation == "recency":
        metadata["recency_cap"] = 0
    elif mutation == "nonfinite":
        metadata["selected_c"] = float("nan")
    write_json(manifest_path, metadata, replace=True)
    with pytest.raises(ArtifactError):
        load_model(manifest_path)


@pytest.mark.parametrize("mutation", ["split_seed", "test_order", "example_count", "key_hash"])
def test_persisted_split_and_test_key_mutations_are_rejected(tmp_path: Path, mutation: str) -> None:
    config = GeneratorConfig(min_length=4, max_length=8, bootstrap_replicates=20, regularization_candidates=(0.1,))
    result = run_benchmark(70, (121,), config, tmp_path)
    pointer_path = tmp_path / "benchmark.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    seed_ref = pointer["seed_results"][0]
    seed_path = tmp_path / seed_ref
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    split_ref = f"{pointer['run']}/{seed['split_manifest']}"
    split_path = tmp_path / split_ref
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if mutation == "split_seed":
        split["seed"] += 1
    elif mutation == "test_order":
        split["test_ids"] = list(reversed(split["test_ids"]))
    elif mutation == "example_count":
        seed["test_example_count"] += 1
    elif mutation == "key_hash":
        seed["test_example_keys_sha256"] = "0" * 64
    if mutation in {"split_seed", "test_order"}:
        write_json(split_path, split, replace=True)
        seed["split_manifest_sha256"] = sha256_file(split_path)
        pointer["artifacts"][split_ref] = seed["split_manifest_sha256"]
    write_json(seed_path, seed, replace=True)
    pointer["artifacts"][seed_ref] = sha256_file(seed_path)
    write_json(pointer_path, pointer, replace=True)
    with pytest.raises(ArtifactError):
        validate_benchmark_manifest(tmp_path, config)


@pytest.mark.parametrize("mutation", ["seed_accuracy", "empty_seed_metrics", "aggregate_mean"])
def test_stale_or_mutated_seed_and_aggregate_statistics_are_rejected(tmp_path: Path, mutation: str) -> None:
    config = GeneratorConfig(min_length=4, max_length=8, bootstrap_replicates=20, regularization_candidates=(0.1,))
    result = run_benchmark(70, (131,), config, tmp_path)
    pointer_path = tmp_path / "benchmark.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if mutation in {"seed_accuracy", "empty_seed_metrics"}:
        seed_ref = pointer["seed_results"][0]
        seed_path = tmp_path / seed_ref
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
        if mutation == "seed_accuracy":
            seed["evaluation"]["metrics"]["markov"]["accuracy"] = 0.0
        else:
            seed["evaluation"]["metrics"] = {}
        write_json(seed_path, seed, replace=True)
        pointer["artifacts"][seed_ref] = sha256_file(seed_path)
    else:
        aggregate_ref = pointer["aggregate_result"]
        aggregate_path = tmp_path / aggregate_ref
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        aggregate["models"]["markov"]["accuracy"]["mean"] = 0.0
        pointer["aggregate"] = aggregate
        write_json(aggregate_path, aggregate, replace=True)
        pointer["artifacts"][aggregate_ref] = sha256_file(aggregate_path)
    write_json(pointer_path, pointer, replace=True)
    with pytest.raises(ArtifactError):
        validate_benchmark_manifest(tmp_path, config)


def test_report_contains_json_bound_metrics_intervals_matrices_and_errors(tmp_path: Path) -> None:
    config = GeneratorConfig(min_length=4, max_length=8, bootstrap_replicates=20, regularization_candidates=(0.1,))
    result = run_benchmark(70, (141,), config, tmp_path)
    report = (tmp_path / result["report"]).read_text(encoding="utf-8")
    assert "Synthetic-domain evaluation evidence" in report
    assert "Aggregate confusion matrices and class supports" in report
    assert "Per-seed scalar metrics and paired values" in report
    assert "Sequence-clustered 95% confidence intervals by seed" in report
    assert "logistic_minus_markov.accuracy" in report
    assert "logistic_minus_markov.macro_f1" in report
    assert "true_predicted_pairs" in report
    assert f"{result['aggregate']['models']['markov']['accuracy']['mean']:.6f}" in report
    assert f"{result['aggregate']['paired_comparisons']['logistic_minus_markov.macro_f1']['values'][0]:.6f}" in report


@pytest.mark.parametrize(
    "mutation",
    [
        "evaluation_fields", "models", "metric_fields", "matrix_shape", "matrix_count", "class_order",
        "scalar", "class_fields", "support_type", "class_score", "support_rows", "support_total",
        "class_recompute", "accuracy_recompute", "f1_recompute", "bootstrap_fields", "bootstrap_type",
        "cluster_count", "interval_keys", "interval_order", "interval_bounds", "nonfinite",
    ],
)
def test_seed_evaluation_semantic_mutations_are_rejected(valid_seed_result: dict[str, object], mutation: str) -> None:
    seed = copy.deepcopy(valid_seed_result)
    evaluation = seed["evaluation"]
    metric = evaluation["metrics"]["markov"]
    event = EVENT_TYPES[0]
    if mutation == "evaluation_fields":
        evaluation["surprise"] = True
    elif mutation == "models":
        evaluation["metrics"].pop("markov")
    elif mutation == "metric_fields":
        metric["surprise"] = True
    elif mutation == "matrix_shape":
        metric["confusion_matrix"][0].pop()
    elif mutation == "matrix_count":
        metric["confusion_matrix"][0][0] = -1
    elif mutation == "class_order":
        metric["class_order"] = list(reversed(metric["class_order"]))
    elif mutation == "scalar":
        metric["accuracy"] = 2.0
    elif mutation == "class_fields":
        metric["per_class"][event]["surprise"] = 1
    elif mutation == "support_type":
        metric["per_class"][event]["support"] = True
    elif mutation == "class_score":
        metric["per_class"][event]["precision"] = 2.0
    elif mutation == "support_rows":
        metric["per_class"][event]["support"] += 1
    elif mutation == "support_total":
        for name in EVENT_TYPES:
            metric["per_class"][name]["support"] = sum(metric["confusion_matrix"][EVENT_TYPES.index(name)])
        evaluation["example_count"] += 1
    elif mutation == "class_recompute":
        metric["per_class"][event]["precision"] = 0.123456789
    elif mutation == "accuracy_recompute":
        metric["accuracy"] = 0.123456789
    elif mutation == "f1_recompute":
        metric["macro_f1"] = 0.123456789
    elif mutation == "bootstrap_fields":
        evaluation["bootstrap"]["surprise"] = 1
    elif mutation == "bootstrap_type":
        evaluation["bootstrap"]["seed"] = True
    elif mutation == "cluster_count":
        evaluation["bootstrap"]["cluster_count"] += 1
    elif mutation == "interval_keys":
        evaluation["bootstrap"]["intervals_95"].pop("markov.accuracy")
    elif mutation == "interval_order":
        evaluation["bootstrap"]["intervals_95"]["markov.accuracy"] = {"lower": 0.9, "upper": 0.1}
    elif mutation == "interval_bounds":
        evaluation["bootstrap"]["intervals_95"]["markov.accuracy"] = {"lower": -0.1, "upper": 0.5}
    elif mutation == "nonfinite":
        seed["test_sequence_lengths"].append(float("nan"))
    with pytest.raises(ArtifactError):
        _validate_evaluation_structure(seed)


def _valid_error_analysis() -> dict[str, object]:
    examples = [
        PrefixExample("one", (EVENT_TYPES[1],), EVENT_TYPES[1]),
        PrefixExample("two", (EVENT_TYPES[2],) * 4, EVENT_TYPES[2]),
    ]
    probabilities = np.zeros((2, len(EVENT_TYPES)))
    probabilities[:, 0] = 1.0
    return analyze_errors(examples, probabilities)


@pytest.mark.parametrize(
    "mutation",
    [
        "root_type", "root_fields", "section_type", "section_fields", "count_type", "count_range",
        "top_examples", "count_disagreement", "pair_keys", "pair_negative", "pair_sum", "band_keys",
        "band_negative", "band_sum", "categories", "group_type", "group_fields", "group_count",
        "group_examples", "record_fields", "record_label", "record_equal", "record_band",
    ],
)
def test_error_analysis_semantic_mutations_are_rejected(mutation: str) -> None:
    analysis = copy.deepcopy(_valid_error_analysis())
    if mutation == "root_type":
        analysis = []
    elif mutation == "root_fields":
        analysis["surprise"] = {}
    elif mutation == "section_type":
        analysis["true_predicted_pairs"] = []
    elif mutation == "section_fields":
        analysis["true_predicted_pairs"]["surprise"] = 1
    elif mutation == "count_type":
        analysis["true_predicted_pairs"]["count"] = True
    elif mutation == "count_range":
        analysis["true_predicted_pairs"]["count"] = 3
    elif mutation == "top_examples":
        analysis["true_predicted_pairs"]["examples"] = [{}]
    elif mutation == "count_disagreement":
        analysis["semantic_categories"]["count"] = 1
    elif mutation == "pair_keys":
        analysis["true_predicted_pairs"]["groups"].pop(next(iter(analysis["true_predicted_pairs"]["groups"])))
    elif mutation == "pair_negative":
        analysis["true_predicted_pairs"]["groups"][next(iter(analysis["true_predicted_pairs"]["groups"]))] = -1
    elif mutation == "pair_sum":
        analysis["true_predicted_pairs"]["groups"][f"{EVENT_TYPES[1]}->{EVENT_TYPES[0]}"] += 1
    elif mutation == "band_keys":
        analysis["prefix_length_bands"]["groups"].pop("long")
    elif mutation == "band_negative":
        analysis["prefix_length_bands"]["groups"]["long"] = -1
    elif mutation == "band_sum":
        analysis["prefix_length_bands"]["groups"]["long"] += 1
    elif mutation == "categories":
        analysis["semantic_categories"]["groups"].pop("other")
    elif mutation == "group_type":
        analysis["semantic_categories"]["groups"]["other"] = []
    elif mutation == "group_fields":
        analysis["semantic_categories"]["groups"]["other"]["surprise"] = 1
    elif mutation == "group_count":
        analysis["semantic_categories"]["groups"]["rare_target"]["count"] = 0
    elif mutation == "group_examples":
        group = analysis["semantic_categories"]["groups"]["rare_target"]
        group["examples"] = group["examples"] * 3
        group["count"] = len(group["examples"])
    else:
        record = analysis["semantic_categories"]["groups"]["rare_target"]["examples"][0]
        if mutation == "record_fields":
            record["surprise"] = 1
        elif mutation == "record_label":
            record["target"] = "not-an-event"
        elif mutation == "record_equal":
            record["target"] = record["predicted"]
        elif mutation == "record_band":
            record["prefix_length_band"] = "huge"
    with pytest.raises(ArtifactError):
        _validate_error_analysis(analysis, 2)


@pytest.mark.parametrize(
    "mutation",
    [
        "profile", "profile_provenance", "config_provenance", "seed_provenance", "vocabulary",
        "run_missing", "artifact_outside", "artifact_hash", "required_hash", "inline_aggregate",
        "seed_count", "split_hash", "split_overlap", "split_counts", "length_shape", "length_value",
        "model_closure", "model_inline", "model_config", "model_seed", "training_hash", "bootstrap_seed",
        "bootstrap_replicates", "dataset_count", "declared_seed", "status", "report",
    ],
)
def test_published_manifest_semantic_mutations_are_rejected(
    valid_benchmark_tree: tuple[Path, GeneratorConfig], tmp_path: Path, mutation: str
) -> None:
    source, config = valid_benchmark_tree
    output = tmp_path / "copy"
    shutil.copytree(source, output)
    pointer_path = output / "benchmark.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    seed_ref = pointer["seed_results"][0]
    seed_path = output / seed_ref
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    split_ref = f"{pointer['run']}/{seed['split_manifest']}"
    split_path = output / split_ref
    model_ref = f"{pointer['run']}/{seed['model']['manifest']}"
    model_path = output / model_ref

    def publish_seed() -> None:
        write_json(seed_path, seed, replace=True)
        pointer["artifacts"][seed_ref] = sha256_file(seed_path)

    def publish_model(model: dict[str, object]) -> None:
        write_json(model_path, model, replace=True)
        pointer["artifacts"][model_ref] = sha256_file(model_path)

    if mutation == "profile":
        pointer["final_profile"]["count"] += 1
    elif mutation == "profile_provenance":
        pointer["provenance"]["final_profile_hash"] = "0" * 64
    elif mutation == "config_provenance":
        pointer["provenance"]["config_hash"] = "0" * 64
    elif mutation == "seed_provenance":
        pointer["provenance"]["seeds"] = [999]
    elif mutation == "vocabulary":
        pointer["event_vocabulary"] = list(reversed(pointer["event_vocabulary"]))
    elif mutation == "run_missing":
        pointer["run"] = "runs/missing"
    elif mutation == "artifact_outside":
        outside = output / "outside.json"
        outside.write_text("outside", encoding="utf-8")
        pointer["artifacts"]["outside.json"] = sha256_file(outside)
    elif mutation == "artifact_hash":
        pointer["artifacts"][seed_ref] = "0" * 64
    elif mutation == "required_hash":
        pointer["artifacts"].pop(pointer["report"])
    elif mutation == "inline_aggregate":
        pointer["aggregate"]["interpretation"] += " changed"
    elif mutation == "seed_count":
        pointer["seeds"].append(999)
        pointer["provenance"]["seeds"].append(999)
    elif mutation == "split_hash":
        seed["split_manifest_sha256"] = "0" * 64
        publish_seed()
    elif mutation == "split_overlap":
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split["validation_ids"][0] = split["train_ids"][0]
        write_json(split_path, split, replace=True)
        seed["split_manifest_sha256"] = sha256_file(split_path)
        pointer["artifacts"][split_ref] = seed["split_manifest_sha256"]
        publish_seed()
    elif mutation == "split_counts":
        seed["split"]["train_sequences"] += 1
        publish_seed()
    elif mutation == "length_shape":
        seed["test_sequence_lengths"].append(2)
        publish_seed()
    elif mutation == "length_value":
        seed["test_sequence_lengths"][0] += 1
        publish_seed()
    elif mutation == "model_closure":
        pointer["artifacts"].pop(f"{pointer['run']}/{seed['model']['arrays']}")
    elif mutation in {"model_inline", "model_config", "model_seed", "training_hash"}:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        if mutation == "model_inline":
            model["selected_c"] = float(model["selected_c"]) + 1.0
        elif mutation == "model_config":
            model["provenance"]["config_hash"] = "0" * 64
            seed["model"]["provenance"]["config_hash"] = "0" * 64
        elif mutation == "model_seed":
            model["model_seed"] += 1
            seed["model"]["model_seed"] += 1
        else:
            model["provenance"]["training_examples_hash"] = "0" * 64
            seed["model"]["provenance"]["training_examples_hash"] = "0" * 64
        publish_model(model)
        if mutation != "model_inline":
            publish_seed()
    elif mutation == "bootstrap_seed":
        seed["evaluation"]["bootstrap"]["seed"] += 1
        publish_seed()
    elif mutation == "bootstrap_replicates":
        seed["evaluation"]["bootstrap"]["replicates"] += 1
        publish_seed()
    elif mutation == "dataset_count":
        seed["dataset_sequence_count"] += 1
        publish_seed()
    elif mutation == "declared_seed":
        pointer["seeds"] = [999]
        pointer["provenance"]["seeds"] = [999]
    elif mutation == "status":
        pointer["status"] = "COMPLETE"
    elif mutation == "report":
        report_path = output / pointer["report"]
        report_path.write_text(report_path.read_text(encoding="utf-8") + "stale\n", encoding="utf-8")
        pointer["artifacts"][pointer["report"]] = sha256_file(report_path)
    write_json(pointer_path, pointer, replace=True)
    with pytest.raises(ArtifactError):
        validate_benchmark_manifest(output, config)


def test_reordered_effective_configuration_is_behaviorally_canonical(tmp_path: Path) -> None:
    def reverse_objects(value):
        if isinstance(value, dict):
            return {key: reverse_objects(child) for key, child in reversed(list(value.items()))}
        if isinstance(value, list):
            return [reverse_objects(child) for child in value]
        return value

    original_path = tmp_path / "original.json"
    reordered_path = tmp_path / "reordered.json"
    raw = GeneratorConfig().as_dict()
    original_path.write_text(json.dumps(raw), encoding="utf-8")
    reordered_path.write_text(json.dumps(reverse_objects(raw)), encoding="utf-8")
    original = load_config(original_path)
    reordered = load_config(reordered_path)
    assert original.digest() == reordered.digest() == FINAL_CONFIG_HASH
    assert [item.as_dict() for item in generate_sequences(100, 8675309, original)] == [
        item.as_dict() for item in generate_sequences(100, 8675309, reordered)
    ]


def test_frozen_profile_schema_binds_literal_hash_and_ordered_seeds() -> None:
    profile = benchmark_module._frozen_final_profile()
    base = {
        "schema_version": "benchmark-result.v2",
        "status": "PRELIMINARY",
        "count": 1,
        "seeds": [1],
        "run": "runs/x",
        "artifacts": {"runs/x/a": "0" * 64},
        "provenance": {
            "command_version": "x", "numpy_version": "x", "python_version": "x",
            "scikit_learn_version": "x", "config_hash": "0" * 64, "seeds": [1],
            "final_profile_hash": benchmark_module._profile_hash(profile),
        },
        "event_vocabulary": list(EVENT_TYPES),
        "final_profile": profile,
        "final_profile_hash": benchmark_module._profile_hash(profile),
        "aggregate": {"x": 1},
        "aggregate_result": "runs/x/a",
        "report": "runs/x/a",
        "seed_results": ["runs/x/a"],
    }
    validate_schema("experiment-manifest.v2", base)
    reversed_seeds = copy.deepcopy(base)
    reversed_seeds["final_profile"]["seeds"] = list(reversed(FINAL_SEEDS))
    with pytest.raises(ArtifactError):
        validate_schema("experiment-manifest.v2", reversed_seeds)
    changed_hash = copy.deepcopy(base)
    changed_hash["final_profile"]["config_hash"] = "0" * 64
    with pytest.raises(ArtifactError):
        validate_schema("experiment-manifest.v2", changed_hash)


def test_dataset_restore_precedes_failing_candidate_cleanup(tmp_path: Path, monkeypatch) -> None:
    import event_prediction.artifacts as artifacts_module

    output = tmp_path / "dataset.jsonl"
    manifest = output.with_suffix(".jsonl.manifest.json")
    output.write_bytes(b"old dataset\n")
    manifest.write_bytes(b"old manifest\n")
    real_replace = artifacts_module.os.replace
    real_unlink = Path.unlink

    def fail_second_publish(source, destination):
        if Path(destination).resolve() == manifest.resolve() and str(source).endswith(".candidate"):
            raise OSError("primary publish failure")
        return real_replace(source, destination)

    def fail_candidate_cleanup(path: Path, *args, **kwargs):
        if path.name.endswith(".candidate"):
            raise OSError("secondary candidate cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(artifacts_module.os, "replace", fail_second_publish)
    monkeypatch.setattr(Path, "unlink", fail_candidate_cleanup)
    config = GeneratorConfig(min_length=4, max_length=8)
    with pytest.raises(OSError, match="primary publish failure"):
        write_dataset(generate_sequences(12, 171, config), output, config, 171, replace=True)
    assert output.read_bytes() == b"old dataset\n"
    assert manifest.read_bytes() == b"old manifest\n"


@pytest.mark.parametrize("diagnostic_failure", ["mkdir", "write"])
def test_pointer_recovery_precedes_best_effort_diagnostics(
    tmp_path: Path, monkeypatch, diagnostic_failure: str
) -> None:
    previous = b'{"status":"previous-valid-pointer"}\n'
    pointer = tmp_path / "benchmark.json"
    pointer.write_bytes(previous)
    real_validate = benchmark_module.validate_benchmark_manifest
    real_mkdir = Path.mkdir
    real_write_json = benchmark_module.write_json

    def fail_after_pointer(output_dir, config, **kwargs):
        if kwargs.get("_manifest") is None:
            raise ArtifactError("primary post-pointer failure")
        return real_validate(output_dir, config, **kwargs)

    def maybe_fail_mkdir(path: Path, *args, **kwargs):
        if diagnostic_failure == "mkdir" and path.name == "failures":
            raise OSError("diagnostic mkdir failure")
        return real_mkdir(path, *args, **kwargs)

    def maybe_fail_write(path: Path, value, replace=False):
        if diagnostic_failure == "write" and path.parent.name == "failures":
            raise OSError("diagnostic write failure")
        return real_write_json(path, value, replace=replace)

    monkeypatch.setattr(benchmark_module, "validate_benchmark_manifest", fail_after_pointer)
    monkeypatch.setattr(Path, "mkdir", maybe_fail_mkdir)
    monkeypatch.setattr(benchmark_module, "write_json", maybe_fail_write)
    config = GeneratorConfig(min_length=4, max_length=8, bootstrap_replicates=20, regularization_candidates=(0.1,))
    with pytest.raises(ArtifactError, match="primary post-pointer failure"):
        run_benchmark(70, (181,), config, tmp_path, replace=True)
    assert pointer.read_bytes() == previous
    assert not list((tmp_path / "runs").iterdir())
    assert not list(tmp_path.glob(".staging-*"))


def test_self_consistent_metrics_tamper_conflicting_with_predictions_is_rejected(tmp_path: Path) -> None:
    config = GeneratorConfig(min_length=4, max_length=8, bootstrap_replicates=20, regularization_candidates=(0.1,))
    result = run_benchmark(70, (191,), config, tmp_path)
    pointer_path = tmp_path / "benchmark.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    seed_ref = pointer["seed_results"][0]
    seed_path = tmp_path / seed_ref
    seed_result = json.loads(seed_path.read_text(encoding="utf-8"))
    generated = generate_sequences(70, 191, config)
    _, examples = benchmark_module.split_and_build_examples(generated, 10_191)
    uniform = _uniform_batch(examples["test"])
    seed_result["evaluation"] = evaluate_models(
        examples["test"],
        {name: uniform for name in ("global_frequency", "markov", "logistic_regression")},
        30_191,
        20,
    )
    write_json(seed_path, seed_result, replace=True)
    pointer["artifacts"][seed_ref] = sha256_file(seed_path)
    write_json(pointer_path, pointer, replace=True)
    with pytest.raises(ArtifactError, match="keyed prediction evidence"):
        validate_benchmark_manifest(tmp_path, config)


def test_hash_updated_prediction_target_tamper_is_rejected(tmp_path: Path) -> None:
    config = GeneratorConfig(min_length=4, max_length=8, bootstrap_replicates=20, regularization_candidates=(0.1,))
    result = run_benchmark(70, (201,), config, tmp_path)
    pointer_path = tmp_path / "benchmark.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    seed_ref = pointer["seed_results"][0]
    seed_path = tmp_path / seed_ref
    seed_result = json.loads(seed_path.read_text(encoding="utf-8"))
    evidence_ref = f"{pointer['run']}/{seed_result['prediction_evidence']}"
    evidence_path = tmp_path / evidence_ref
    with np.load(evidence_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["target_indices"] = arrays["target_indices"].copy()
    arrays["target_indices"][0] = (int(arrays["target_indices"][0]) + 1) % len(EVENT_TYPES)
    payload = _deterministic_npz(arrays)
    evidence_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    seed_result["prediction_evidence_sha256"] = digest
    write_json(seed_path, seed_result, replace=True)
    pointer["artifacts"][evidence_ref] = digest
    pointer["artifacts"][seed_ref] = sha256_file(seed_path)
    write_json(pointer_path, pointer, replace=True)
    with pytest.raises(ArtifactError, match="trusted held-out targets"):
        validate_benchmark_manifest(tmp_path, config)


def test_coordinated_baseline_statistics_tamper_is_rejected_by_saved_state(tmp_path: Path) -> None:
    config = GeneratorConfig(min_length=4, max_length=8, bootstrap_replicates=20, regularization_candidates=(0.1,))
    result = run_benchmark(70, (206,), config, tmp_path)
    pointer_path = tmp_path / "benchmark.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    seed_ref = pointer["seed_results"][0]
    seed_path = tmp_path / seed_ref
    seed_result = json.loads(seed_path.read_text(encoding="utf-8"))
    evidence_ref = f"{pointer['run']}/{seed_result['prediction_evidence']}"
    evidence_path = tmp_path / evidence_ref
    with np.load(evidence_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    uniform = np.full_like(arrays["probabilities_global_frequency"], 1 / len(EVENT_TYPES))
    arrays["probabilities_global_frequency"] = uniform
    arrays["probabilities_markov"] = uniform.copy()
    evidence_payload = _deterministic_npz(arrays)
    evidence_path.write_bytes(evidence_payload)
    evidence_hash = hashlib.sha256(evidence_payload).hexdigest()
    seed_result["prediction_evidence_sha256"] = evidence_hash

    generated = generate_sequences(70, 206, config)
    _, examples = benchmark_module.split_and_build_examples(generated, 10_206)
    keys = tuple(example.key for example in examples["test"])
    batches = {
        "global_frequency": PredictionBatch(keys, arrays["probabilities_global_frequency"]),
        "markov": PredictionBatch(keys, arrays["probabilities_markov"]),
        "logistic_regression": PredictionBatch(keys, arrays["probabilities_logistic_regression"]),
    }
    seed_result["evaluation"] = evaluate_models(examples["test"], batches, 30_206, 20)
    write_json(seed_path, seed_result, replace=True)
    aggregate = benchmark_module._aggregate([seed_result])
    aggregate_path = tmp_path / pointer["aggregate_result"]
    write_json(aggregate_path, aggregate, replace=True)
    report_path = tmp_path / pointer["report"]
    report_path.write_text(
        benchmark_module._report(aggregate, pointer["status"], pointer["count"], pointer["seeds"]),
        encoding="utf-8",
    )
    pointer["aggregate"] = aggregate
    pointer["artifacts"][evidence_ref] = evidence_hash
    pointer["artifacts"][seed_ref] = sha256_file(seed_path)
    pointer["artifacts"][pointer["aggregate_result"]] = sha256_file(aggregate_path)
    pointer["artifacts"][pointer["report"]] = sha256_file(report_path)
    write_json(pointer_path, pointer, replace=True)
    with pytest.raises(ArtifactError, match="global_frequency predictions contradict"):
        validate_benchmark_manifest(tmp_path, config)


@pytest.mark.parametrize("mutation", ["missing", "arrays", "alpha", "vocabulary", "provenance"])
def test_strict_baseline_state_mutations_are_rejected(
    valid_benchmark_tree: tuple[Path, GeneratorConfig], tmp_path: Path, mutation: str
) -> None:
    source, config = valid_benchmark_tree
    output = tmp_path / "copy"
    shutil.copytree(source, output)
    pointer_path = output / "benchmark.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    seed_ref = pointer["seed_results"][0]
    seed_path = output / seed_ref
    seed_result = json.loads(seed_path.read_text(encoding="utf-8"))
    baseline = seed_result["baseline_state"]
    manifest_ref = f"{pointer['run']}/{baseline['manifest']}"
    arrays_ref = f"{pointer['run']}/{baseline['arrays']}"
    manifest_path = output / manifest_ref
    arrays_path = output / arrays_ref

    if mutation == "missing":
        seed_result.pop("baseline_state")
    else:
        persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "arrays":
            with np.load(arrays_path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            arrays["global_probabilities"] = np.roll(arrays["global_probabilities"], 1)
            payload = _deterministic_npz(arrays)
            arrays_path.write_bytes(payload)
            persisted["arrays_sha256"] = hashlib.sha256(payload).hexdigest()
            pointer["artifacts"][arrays_ref] = persisted["arrays_sha256"]
        elif mutation == "alpha":
            persisted["alpha"] = float(persisted["alpha"]) + 0.5
        elif mutation == "vocabulary":
            persisted["event_vocabulary"] = list(reversed(persisted["event_vocabulary"]))
        else:
            persisted["provenance"]["training_examples_hash"] = "0" * 64
        write_json(manifest_path, persisted, replace=True)
        seed_result["baseline_state"] = {"manifest": baseline["manifest"], **persisted}
        pointer["artifacts"][manifest_ref] = sha256_file(manifest_path)
    write_json(seed_path, seed_result, replace=True)
    pointer["artifacts"][seed_ref] = sha256_file(seed_path)
    write_json(pointer_path, pointer, replace=True)
    with pytest.raises(ArtifactError):
        validate_benchmark_manifest(output, config)


def test_extreme_finite_model_parameters_are_rejected_at_safe_probe(complete_examples, tmp_path: Path) -> None:
    model = train_logistic(
        complete_examples("train"), complete_examples("validation"), (0.1,), 211,
        provenance={"config_hash": "1" * 64, "split_seed": 212, "training_examples_hash": "2" * 64, "final_profile_hash": "3" * 64},
    )
    manifest_path = tmp_path / "model.json"
    metadata = save_model(model, manifest_path)
    arrays_path = tmp_path / metadata["arrays"]
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["coef"] = np.full_like(arrays["coef"], np.finfo(np.float64).max)
    arrays["intercept"] = np.full_like(arrays["intercept"], np.finfo(np.float64).max)
    payload = _deterministic_npz(arrays)
    arrays_path.write_bytes(payload)
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    persisted["arrays_sha256"] = hashlib.sha256(payload).hexdigest()
    write_json(manifest_path, persisted, replace=True)
    with pytest.raises(ArtifactError, match="non-finite decision scores"):
        load_model(manifest_path)


def _baseline_artifact(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    probabilities = np.full(len(EVENT_TYPES), 1 / len(EVENT_TYPES), dtype=np.float64)
    predictors = {
        "global_frequency": GlobalFrequencyPredictor(probabilities.copy()),
        "markov": MarkovPredictor(
            rows={event: probabilities.copy() for event in EVENT_TYPES},
            fallback=probabilities.copy(),
            alpha=1.0,
        ),
    }
    provenance = {
        "config_hash": "1" * 64,
        "split_seed": 17,
        "training_examples_hash": "2" * 64,
        "final_profile_hash": "3" * 64,
    }
    manifest_path = tmp_path / "baseline.json"
    inline = benchmark_module._save_baseline_state(
        predictors, manifest_path, alpha=1.0, provenance=provenance, replace=False
    )
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest_path, inline, metadata


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_field", "extra_field", "schema", "vocabulary", "alpha_bool", "alpha_zero",
        "alpha_nonfinite", "provenance_missing", "protocol",
    ],
)
def test_baseline_state_manifest_contract_rejects_invalid_metadata(tmp_path: Path, mutation: str) -> None:
    manifest_path, _, metadata = _baseline_artifact(tmp_path)
    arrays_path = tmp_path / metadata["arrays"]
    if mutation == "missing_field":
        metadata.pop("alpha")
    elif mutation == "extra_field":
        metadata["surprise"] = True
    elif mutation == "schema":
        metadata["schema_version"] = "baseline-state.future"
    elif mutation == "vocabulary":
        metadata["event_vocabulary"] = list(reversed(metadata["event_vocabulary"]))
    elif mutation == "alpha_bool":
        metadata["alpha"] = True
    elif mutation == "alpha_zero":
        metadata["alpha"] = 0
    elif mutation == "alpha_nonfinite":
        metadata["alpha"] = float("inf")
    elif mutation == "provenance_missing":
        metadata["provenance"].pop("training_examples_hash")
    else:
        metadata["provenance"]["training_protocol"] = "unsupported"
    with pytest.raises(ArtifactError):
        benchmark_module._baseline_state_parts(metadata, arrays_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing", "extra", "global_dtype", "global_shape", "global_nan", "global_bounds",
        "global_sum", "fallback_shape", "rows_dtype", "rows_shape", "row_nan", "row_bounds", "row_sum",
    ],
)
def test_baseline_state_npz_contract_rejects_invalid_arrays(tmp_path: Path, mutation: str) -> None:
    manifest_path, _, metadata = _baseline_artifact(tmp_path)
    arrays_path = tmp_path / metadata["arrays"]
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if mutation == "missing":
        arrays.pop("markov_fallback")
    elif mutation == "extra":
        arrays["surprise"] = np.asarray([1], dtype=np.int64)
    elif mutation == "global_dtype":
        arrays["global_probabilities"] = arrays["global_probabilities"].astype(np.float32)
    elif mutation == "global_shape":
        arrays["global_probabilities"] = arrays["global_probabilities"][:-1]
    elif mutation == "global_nan":
        arrays["global_probabilities"] = arrays["global_probabilities"].copy()
        arrays["global_probabilities"][0] = np.nan
    elif mutation == "global_bounds":
        arrays["global_probabilities"] = arrays["global_probabilities"].copy()
        arrays["global_probabilities"][0] = -0.1
    elif mutation == "global_sum":
        arrays["global_probabilities"] = arrays["global_probabilities"] * 0.5
    elif mutation == "fallback_shape":
        arrays["markov_fallback"] = arrays["markov_fallback"][:-1]
    elif mutation == "rows_dtype":
        arrays["markov_rows"] = arrays["markov_rows"].astype(np.float32)
    elif mutation == "rows_shape":
        arrays["markov_rows"] = arrays["markov_rows"][:-1]
    elif mutation == "row_nan":
        arrays["markov_rows"] = arrays["markov_rows"].copy()
        arrays["markov_rows"][0, 0] = np.nan
    elif mutation == "row_bounds":
        arrays["markov_rows"] = arrays["markov_rows"].copy()
        arrays["markov_rows"][0, 0] = 1.1
    else:
        arrays["markov_rows"] = arrays["markov_rows"].copy()
        arrays["markov_rows"][0] *= 0.5
    bad_path = tmp_path / f"bad-{mutation}.npz"
    bad_path.write_bytes(_deterministic_npz(arrays))
    with pytest.raises(ArtifactError):
        benchmark_module._baseline_state_parts(metadata, bad_path)


@pytest.mark.parametrize("mutation", ["non_object", "escape", "nested", "hash"])
def test_baseline_state_loader_rejects_unsafe_manifest_and_references(tmp_path: Path, mutation: str) -> None:
    manifest_path, _, metadata = _baseline_artifact(tmp_path)
    if mutation == "non_object":
        write_json(manifest_path, [], replace=True)
    elif mutation == "escape":
        metadata["arrays"] = "../outside.npz"
        write_json(manifest_path, metadata, replace=True)
    elif mutation == "nested":
        metadata["arrays"] = "nested/arrays.npz"
        write_json(manifest_path, metadata, replace=True)
    else:
        metadata["arrays_sha256"] = "0" * 64
        write_json(manifest_path, metadata, replace=True)
    with pytest.raises(ArtifactError):
        benchmark_module._load_baseline_state(manifest_path)


@pytest.mark.parametrize("mutation", ["set", "global_type", "markov_type"])
def test_baseline_state_writer_rejects_unexpected_predictors(tmp_path: Path, mutation: str) -> None:
    probabilities = np.full(len(EVENT_TYPES), 1 / len(EVENT_TYPES), dtype=np.float64)
    predictors = {
        "global_frequency": GlobalFrequencyPredictor(probabilities.copy()),
        "markov": MarkovPredictor(
            rows={event: probabilities.copy() for event in EVENT_TYPES},
            fallback=probabilities.copy(), alpha=1.0,
        ),
    }
    if mutation == "set":
        predictors.pop("markov")
    elif mutation == "global_type":
        predictors["global_frequency"] = object()
    else:
        predictors["markov"] = object()
    with pytest.raises(ArtifactError):
        benchmark_module._save_baseline_state(
            predictors,
            tmp_path / "baseline.json",
            alpha=1.0,
            provenance={
                "config_hash": "1" * 64, "split_seed": 1,
                "training_examples_hash": "2" * 64, "final_profile_hash": "3" * 64,
            },
            replace=False,
        )


@pytest.mark.parametrize("mutation", ["closure", "inline"])
def test_published_baseline_state_integration_rejects_missing_closure_or_inline_mismatch(
    valid_benchmark_tree: tuple[Path, GeneratorConfig], tmp_path: Path, mutation: str
) -> None:
    source, config = valid_benchmark_tree
    output = tmp_path / "copy"
    shutil.copytree(source, output)
    pointer_path = output / "benchmark.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    seed_ref = pointer["seed_results"][0]
    seed_path = output / seed_ref
    seed_result = json.loads(seed_path.read_text(encoding="utf-8"))
    baseline = seed_result["baseline_state"]
    if mutation == "closure":
        pointer["artifacts"].pop(f"{pointer['run']}/{baseline['manifest']}")
    else:
        seed_result["baseline_state"]["alpha"] = float(baseline["alpha"]) + 0.25
        write_json(seed_path, seed_result, replace=True)
        pointer["artifacts"][seed_ref] = sha256_file(seed_path)
    write_json(pointer_path, pointer, replace=True)
    with pytest.raises(ArtifactError):
        validate_benchmark_manifest(output, config)
