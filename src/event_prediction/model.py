"""Validation-selected sparse multinomial logistic regression."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler

from .catalog import EVENT_TYPES
from .domain import ArtifactError, ConfigurationError, MODEL_SCHEMA, PredictionBatch, PrefixExample
from .features import extract_features
from .artifacts import canonical_json, resolve_portable_reference, write_bytes, write_json
from .schemas import validate_schema

@dataclass
class LogisticPredictor:
    vectorizer: DictVectorizer
    scaler: MaxAbsScaler
    estimator: LogisticRegression
    selected_c: float
    selection_report: list[dict[str, Any]]
    model_seed: int
    recency_cap: int
    provenance: dict[str, Any]

    def predict_proba(self, prefixes: Sequence[tuple[str, ...]]) -> np.ndarray:
        matrix = self.vectorizer.transform([extract_features(prefix, self.recency_cap) for prefix in prefixes])
        scaled = self.scaler.transform(matrix)
        if scaled.data.size and not np.all(np.isfinite(scaled.data)):
            raise ArtifactError("model produced non-finite scaled features")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            decisions = np.asarray(self.estimator.decision_function(scaled), dtype=float)
            if not np.all(np.isfinite(decisions)):
                raise ArtifactError("model produced non-finite decision scores")
            partial = np.asarray(self.estimator.predict_proba(scaled), dtype=float)
        if (partial.shape != (len(prefixes), len(self.estimator.classes_))
                or not np.all(np.isfinite(partial))
                or np.any(partial < 0) or np.any(partial > 1)):
            raise ArtifactError("model produced invalid class probabilities")
        aligned = np.zeros((len(prefixes), len(EVENT_TYPES)), dtype=float)
        positions = {event: index for index, event in enumerate(EVENT_TYPES)}
        for source, event in enumerate(self.estimator.classes_):
            aligned[:, positions[str(event)]] = partial[:, source]
        if not np.all(np.isfinite(aligned)) or np.any(aligned < 0) or np.any(aligned > 1):
            raise ArtifactError("model produced invalid aligned probabilities")
        row_sums = aligned.sum(axis=1, keepdims=True)
        if not np.all(np.isfinite(row_sums)) or np.any(row_sums <= 0):
            raise ArtifactError("model produced an invalid probability row")
        normalized = aligned / row_sums
        if (not np.all(np.isfinite(normalized)) or np.any(normalized < 0) or np.any(normalized > 1)
                or not np.allclose(normalized.sum(axis=1), 1.0, rtol=0, atol=1e-12)):
            raise ArtifactError("model probabilities are not finite normalized rows")
        return normalized

    def predict_examples(self, examples: Sequence[PrefixExample]) -> PredictionBatch:
        return PredictionBatch(tuple(item.key for item in examples), self.predict_proba([item.prefix for item in examples]))

    def metadata(self) -> dict[str, Any]:
        return {
            "classes": [str(value) for value in self.estimator.classes_],
            "event_catalog": list(EVENT_TYPES),
            "feature_count": len(self.vectorizer.feature_names_),
            "feature_names": list(self.vectorizer.feature_names_),
            "model_seed": self.model_seed,
            "python_version": platform.python_version(),
            "schema_version": MODEL_SCHEMA,
            "scikit_learn_version": sklearn.__version__,
            "selected_c": self.selected_c,
            "selection_report": self.selection_report,
            "provenance": self.provenance,
            "recency_cap": self.recency_cap,
        }


def _fit_estimator(matrix: Any, labels: Sequence[str], c_value: float, seed: int) -> tuple[LogisticRegression, bool]:
    estimator = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=2_000,
        tol=1e-4,
        random_state=seed,
        n_jobs=1,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(matrix, labels)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return estimator, converged


# @cpt-algo:cpt-event-prediction-algo-ml-prediction-select-model:p1
# @cpt-flow:cpt-event-prediction-flow-ml-prediction-train-select:p1
# @cpt-dod:cpt-event-prediction-dod-ml-prediction-logistic-model:p1
def train_logistic(
    training: Sequence[PrefixExample],
    validation: Sequence[PrefixExample],
    candidates: Sequence[float],
    seed: int,
    recency_cap: int = 10,
    provenance: dict[str, Any] | None = None,
) -> LogisticPredictor:
    # @cpt-begin:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-load-examples
    if not training or not validation:
        raise ConfigurationError("training and validation examples must not be empty")
    # @cpt-end:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-load-examples
    # @cpt-begin:cpt-event-prediction-algo-ml-prediction-select-model:p1:inst-ml-check-training
    support = {example.target for example in training}
    missing = sorted(set(EVENT_TYPES) - support)
    if missing:
        raise ConfigurationError(f"training data lacks event classes: {', '.join(missing)}")
    # @cpt-end:cpt-event-prediction-algo-ml-prediction-select-model:p1:inst-ml-check-training
    # @cpt-begin:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-fit-features
    vectorizer = DictVectorizer(sparse=True, sort=True)
    train_matrix = vectorizer.fit_transform([extract_features(item.prefix, recency_cap) for item in training])
    # @cpt-end:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-fit-features
    # @cpt-begin:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-transform
    validation_matrix = vectorizer.transform([extract_features(item.prefix, recency_cap) for item in validation])
    scaler = MaxAbsScaler()
    train_matrix = scaler.fit_transform(train_matrix)
    validation_matrix = scaler.transform(validation_matrix)
    # @cpt-end:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-transform
    train_labels = [item.target for item in training]
    validation_labels = [item.target for item in validation]
    report: list[dict[str, Any]] = []
    fitted: dict[float, LogisticRegression] = {}
    # @cpt-begin:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-fit-candidate
    # @cpt-begin:cpt-event-prediction-algo-ml-prediction-select-model:p1:inst-ml-grid-fit
    for c_value in sorted(set(float(value) for value in candidates)):
        if c_value <= 0 or not np.isfinite(c_value):
            raise ConfigurationError("regularization candidates must be finite and positive")
        estimator, converged = _fit_estimator(train_matrix, train_labels, c_value, seed)
        # @cpt-end:cpt-event-prediction-algo-ml-prediction-select-model:p1:inst-ml-grid-fit
        # @cpt-begin:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-validate-candidate
        predicted = estimator.predict(validation_matrix)
        score = float(f1_score(validation_labels, predicted, labels=EVENT_TYPES, average="macro", zero_division=0))
        report.append({"c": c_value, "converged": converged, "validation_macro_f1": score})
        # @cpt-end:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-validate-candidate
        # @cpt-begin:cpt-event-prediction-algo-ml-prediction-select-model:p1:inst-ml-convergence-policy
        if converged:
            fitted[c_value] = estimator
        # @cpt-end:cpt-event-prediction-algo-ml-prediction-select-model:p1:inst-ml-convergence-policy
        # Re-open for each declared candidate after the first iteration.
        # @cpt-begin:cpt-event-prediction-algo-ml-prediction-select-model:p1:inst-ml-grid-fit
    if not fitted:
        raise ConfigurationError("no logistic-regression candidate satisfied convergence policy")
    # @cpt-end:cpt-event-prediction-algo-ml-prediction-select-model:p1:inst-ml-grid-fit
    # @cpt-end:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-fit-candidate
    valid_rows = [row for row in report if row["converged"]]
    # Higher score first; lower C means stronger regularization and wins exact ties.
    # @cpt-begin:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-select
    # @cpt-begin:cpt-event-prediction-algo-ml-prediction-select-model:p1:inst-ml-rank
    winner = sorted(valid_rows, key=lambda row: (-row["validation_macro_f1"], row["c"]))[0]
    # @cpt-end:cpt-event-prediction-algo-ml-prediction-select-model:p1:inst-ml-rank
    # @cpt-end:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-select
    # @cpt-begin:cpt-event-prediction-algo-ml-prediction-select-model:p1:inst-ml-freeze
    selected = LogisticPredictor(
        vectorizer,
        scaler,
        fitted[winner["c"]],
        winner["c"],
        report,
        seed,
        recency_cap,
        dict(provenance or {}),
    )
    # @cpt-end:cpt-event-prediction-algo-ml-prediction-select-model:p1:inst-ml-freeze
    selected.selection_report = list(selected.selection_report)
    # @cpt-begin:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-train-result
    # @cpt-begin:cpt-event-prediction-algo-ml-prediction-select-model:p1:inst-ml-selection-result
    result = selected
    # @cpt-end:cpt-event-prediction-algo-ml-prediction-select-model:p1:inst-ml-selection-result
    # @cpt-end:cpt-event-prediction-flow-ml-prediction-train-select:p1:inst-ml-train-result
    return result


# @cpt-flow:cpt-event-prediction-flow-ml-prediction-predict-prefix:p1
def predict_prefix(model: LogisticPredictor, prefix: tuple[str, ...]) -> dict[str, Any]:
    # @cpt-begin:cpt-event-prediction-flow-ml-prediction-predict-prefix:p1:inst-ml-validate-prefix
    extract_features(prefix, model.recency_cap)
    # @cpt-end:cpt-event-prediction-flow-ml-prediction-predict-prefix:p1:inst-ml-validate-prefix
    # @cpt-begin:cpt-event-prediction-flow-ml-prediction-predict-prefix:p1:inst-ml-transform-prefix
    probabilities = model.predict_proba([prefix])[0]
    # @cpt-end:cpt-event-prediction-flow-ml-prediction-predict-prefix:p1:inst-ml-transform-prefix
    # @cpt-begin:cpt-event-prediction-flow-ml-prediction-predict-prefix:p1:inst-ml-probabilities
    best_index = int(np.argmax(probabilities))
    prediction = {
        "predicted_event": EVENT_TYPES[best_index],
        "probabilities": {event: float(probabilities[index]) for index, event in enumerate(EVENT_TYPES)},
    }
    # @cpt-end:cpt-event-prediction-flow-ml-prediction-predict-prefix:p1:inst-ml-probabilities
    # @cpt-begin:cpt-event-prediction-flow-ml-prediction-predict-prefix:p1:inst-ml-predict-result
    result = prediction
    # @cpt-end:cpt-event-prediction-flow-ml-prediction-predict-prefix:p1:inst-ml-predict-result
    return result


# @cpt-state:cpt-event-prediction-state-ml-prediction-model-artifact:p2
def save_model(model: LogisticPredictor, path: Path, replace: bool = False) -> dict[str, Any]:
    if path.exists() and not replace:
        raise ArtifactError(f"refusing to overwrite existing model: {path}; pass --replace")
    learned_classes = [str(value) for value in model.estimator.classes_]
    if set(learned_classes) != set(EVENT_TYPES) or len(learned_classes) != len(EVENT_TYPES):
        raise ArtifactError("cannot persist a model without the exact event class vocabulary")
    class_positions = [learned_classes.index(event) for event in EVENT_TYPES]
    arrays = {
        "classes": np.asarray(EVENT_TYPES, dtype=np.str_),
        "coef": np.asarray(model.estimator.coef_[class_positions], dtype=np.float64),
        "intercept": np.asarray(model.estimator.intercept_[class_positions], dtype=np.float64),
        "n_iter": np.asarray(model.estimator.n_iter_, dtype=np.int64),
        "scale": np.asarray(model.scaler.scale_, dtype=np.float64),
        "max_abs": np.asarray(model.scaler.max_abs_, dtype=np.float64),
    }
    payload = _deterministic_npz(arrays)
    digest = hashlib.sha256(payload).hexdigest()
    arrays_name = f"{path.stem}.{digest[:16]}.npz"
    arrays_path = path.parent / arrays_name
    if arrays_path.exists() and hashlib.sha256(arrays_path.read_bytes()).hexdigest() != digest:
        raise ArtifactError(f"model array artifact collision: {arrays_path}")
    metadata = {**model.metadata(), "classes": list(EVENT_TYPES), "arrays": arrays_name, "arrays_sha256": digest}
    validate_schema("model-manifest.v2", metadata)
    # @cpt-begin:cpt-event-prediction-state-ml-prediction-model-artifact:p2:inst-ml-state-candidate
    if not arrays_path.exists():
        write_bytes(arrays_path, payload, replace=False)
    # Verify safe reconstruction before publishing the JSON commit record.
    _load_model_parts(metadata, arrays_path)
    # @cpt-end:cpt-event-prediction-state-ml-prediction-model-artifact:p2:inst-ml-state-candidate
    # @cpt-begin:cpt-event-prediction-state-ml-prediction-model-artifact:p2:inst-ml-state-selected
    write_json(path, metadata, replace=replace)
    # @cpt-end:cpt-event-prediction-state-ml-prediction-model-artifact:p2:inst-ml-state-selected
    load_model(path)
    return {"manifest": path.name, **metadata}


def load_model(path: Path) -> LogisticPredictor:
    # @cpt-begin:cpt-event-prediction-flow-ml-prediction-predict-prefix:p1:inst-ml-load-model
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        validate_schema("model-manifest.v2", metadata)
    except (OSError, UnicodeError, json.JSONDecodeError, ArtifactError) as exc:
        raise ArtifactError(f"cannot load safe model manifest {path}: {exc}") from exc
    arrays_ref = metadata["arrays"]
    arrays_path = resolve_portable_reference(path.parent, arrays_ref, field="model arrays reference")
    if arrays_path.parent != path.parent.resolve():
        raise ArtifactError("model arrays reference must be a sibling filename")
    try:
        payload = arrays_path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"cannot load model arrays {arrays_path}: {exc}") from exc
    if hashlib.sha256(payload).hexdigest() != metadata["arrays_sha256"]:
        raise ArtifactError("model arrays hash does not match manifest")
    # @cpt-end:cpt-event-prediction-flow-ml-prediction-predict-prefix:p1:inst-ml-load-model
    try:
        model = _load_model_parts(metadata, arrays_path)
    except ArtifactError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, UnicodeError) as exc:
        raise ArtifactError(f"cannot safely reconstruct model: {exc}") from exc
    # @cpt-begin:cpt-event-prediction-state-ml-prediction-model-artifact:p2:inst-ml-state-verified
    try:
        probe = model.predict_proba([(EVENT_TYPES[0],)])
        if probe.shape != (1, len(EVENT_TYPES)) or not np.all(np.isfinite(probe)):
            raise ArtifactError("model safe prediction probe returned invalid probabilities")
    except ArtifactError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ArtifactError(f"model failed its safe prediction check: {exc}") from exc
    # @cpt-end:cpt-event-prediction-state-ml-prediction-model-artifact:p2:inst-ml-state-verified
    return model


def _deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return output.getvalue()


def _load_model_parts(metadata: dict[str, Any], arrays_path: Path) -> LogisticPredictor:
    try:
        with np.load(arrays_path, allow_pickle=False) as values:
            required = {"classes", "coef", "intercept", "n_iter", "scale", "max_abs"}
            if set(values.files) != required:
                raise ArtifactError("model arrays contain unexpected or missing entries")
            arrays = {name: np.asarray(values[name]) for name in required}
    except (OSError, ValueError, TypeError, zipfile.BadZipFile) as exc:
        raise ArtifactError(f"cannot safely read model arrays: {exc}") from exc
    feature_names = metadata["feature_names"]
    feature_count = metadata["feature_count"]
    if any(not isinstance(name, str) or not name for name in feature_names):
        raise ArtifactError("model feature names must be non-empty strings")
    if (len(feature_names) != len(set(feature_names)) or len(feature_names) != feature_count
            or len(feature_names) != len(arrays["scale"])):
        raise ArtifactError("model feature metadata is inconsistent")
    stored_classes = arrays["classes"].astype(str).tolist()
    if (arrays["classes"].ndim != 1 or stored_classes != list(EVENT_TYPES)
            or metadata["classes"] != list(EVENT_TYPES)
            or len(stored_classes) != len(set(stored_classes))):
        raise ArtifactError("model class vocabulary is inconsistent")
    if metadata["event_catalog"] != list(EVENT_TYPES):
        raise ArtifactError("model event catalog is inconsistent")
    expected_dtypes = {
        "coef": np.dtype("float64"), "intercept": np.dtype("float64"),
        "n_iter": np.dtype("int64"), "scale": np.dtype("float64"), "max_abs": np.dtype("float64"),
    }
    for name, dtype in expected_dtypes.items():
        if arrays[name].dtype != dtype:
            raise ArtifactError(f"model array {name} has unexpected dtype")
    if arrays["classes"].dtype.kind != "U":
        raise ArtifactError("model classes array must use a Unicode dtype")
    class_count = len(EVENT_TYPES)
    if arrays["coef"].shape != (class_count, feature_count):
        raise ArtifactError("model coefficient dimensions are inconsistent")
    if arrays["intercept"].shape != (class_count,):
        raise ArtifactError("model intercept dimensions are inconsistent")
    if arrays["scale"].shape != (feature_count,) or arrays["max_abs"].shape != (feature_count,):
        raise ArtifactError("model scaler dimensions are inconsistent")
    if arrays["n_iter"].shape != (1,):
        raise ArtifactError("model iteration dimensions are inconsistent")
    for name in ("coef", "intercept", "scale", "max_abs"):
        if not np.all(np.isfinite(arrays[name])):
            raise ArtifactError(f"model array {name} contains non-finite values")
    if np.any(arrays["scale"] <= 0) or np.any(arrays["max_abs"] < 0) or np.any(arrays["n_iter"] <= 0):
        raise ArtifactError("model scaler or iteration arrays contain invalid values")
    report = metadata["selection_report"]
    if not np.isfinite(metadata["selected_c"]) or metadata["selected_c"] <= 0:
        raise ArtifactError("selected_c must be finite and positive")
    for row in report:
        if (set(row) != {"c", "converged", "validation_macro_f1"}
                or type(row["converged"]) is not bool
                or isinstance(row["c"], bool) or not isinstance(row["c"], (int, float))
                or not np.isfinite(row["c"]) or row["c"] <= 0
                or isinstance(row["validation_macro_f1"], bool)
                or not isinstance(row["validation_macro_f1"], (int, float))
                or not np.isfinite(row["validation_macro_f1"])
                or not 0 <= row["validation_macro_f1"] <= 1):
            raise ArtifactError("model selection report contains an invalid row")
    if len({float(row["c"]) for row in report}) != len(report):
        raise ArtifactError("model selection report contains duplicate candidates")
    converged = [row for row in report if row["converged"]]
    if not converged:
        raise ArtifactError("model selection report has no converged candidate")
    winner = sorted(converged, key=lambda row: (-row["validation_macro_f1"], row["c"]))[0]
    if float(winner["c"]) != float(metadata["selected_c"]):
        raise ArtifactError("selected_c contradicts the validation-only selection report")
    vectorizer = DictVectorizer(sparse=True, sort=True)
    vectorizer.feature_names_ = list(feature_names)
    vectorizer.vocabulary_ = {name: index for index, name in enumerate(feature_names)}
    scaler = MaxAbsScaler()
    scaler.scale_ = arrays["scale"]
    scaler.max_abs_ = arrays["max_abs"]
    scaler.n_features_in_ = len(feature_names)
    scaler.n_samples_seen_ = 1
    estimator = LogisticRegression(C=float(metadata["selected_c"]), solver="lbfgs", max_iter=2_000, tol=1e-4, random_state=int(metadata["model_seed"]), n_jobs=1)
    estimator.classes_ = arrays["classes"].astype(str)
    estimator.coef_ = arrays["coef"]
    estimator.intercept_ = arrays["intercept"]
    estimator.n_iter_ = arrays["n_iter"]
    estimator.n_features_in_ = len(feature_names)
    return LogisticPredictor(vectorizer, scaler, estimator, float(metadata["selected_c"]), list(metadata["selection_report"]), int(metadata["model_seed"]), int(metadata["recency_cap"]), dict(metadata["provenance"]))
