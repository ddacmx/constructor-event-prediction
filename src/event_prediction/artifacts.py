"""Canonical, atomic file artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable

from .catalog import EVENT_TYPES
from .config import GeneratorConfig
from .domain import ArtifactError, DATASET_SCHEMA, EventSequence, MANIFEST_SCHEMA, ObservedEvent
from .schemas import validate_schema

SEQUENCE_KEYS = {"events", "provenance", "schema_version", "sequence_id"}
EVENT_KEYS = {"description", "event_type", "position"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_write(path: Path, content: bytes, replace: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # @cpt-begin:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1:inst-cli-protect-output
    if path.exists() and not replace:
        raise ArtifactError(f"refusing to overwrite existing artifact: {path}; pass --replace")
    # @cpt-end:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1:inst-cli-protect-output
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def write_bytes(path: Path, content: bytes, replace: bool = False) -> None:
    """Atomically publish already-validated bytes."""
    _atomic_write(path, content, replace)


def _validate_observable_record(record: dict[str, Any]) -> None:
    if set(record) != SEQUENCE_KEYS:
        raise ArtifactError(f"sequence contains unapproved fields: {sorted(set(record) - SEQUENCE_KEYS)}")
    validate_schema("dataset-record.v1", record)
    if record["schema_version"] != DATASET_SCHEMA:
        raise ArtifactError("unsupported dataset schema")
    if set(record["provenance"]) != {"generation_seed"}:
        raise ArtifactError("provenance contains unapproved fields")
    for event in record["events"]:
        if set(event) != EVENT_KEYS:
            raise ArtifactError("event contains unapproved fields")


# @cpt-algo:cpt-event-prediction-algo-synthetic-data-generation-canonical-output:p1
# @cpt-dod:cpt-event-prediction-dod-synthetic-data-generation-dataset:p1
def write_dataset(
    sequences: Iterable[EventSequence], output: Path, config: GeneratorConfig, seed: int, replace: bool = False
) -> dict[str, Any]:
    records = [sequence.as_dict() for sequence in sequences]
    if not records:
        raise ArtifactError("cannot write an empty dataset")
    # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-canonical-output:p1:inst-sdg-reject-hidden-field
    for record in records:
        _validate_observable_record(record)
    # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-canonical-output:p1:inst-sdg-reject-hidden-field
    # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-canonical-output:p1:inst-sdg-serialize
    payload = "".join(canonical_json(record) + "\n" for record in records).encode("utf-8")
    # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-canonical-output:p1:inst-sdg-serialize
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "config_hash": config.digest(),
        "count": len(records),
        "dataset_hash": digest,
        "dataset_schema": DATASET_SCHEMA,
        "event_vocabulary": list(EVENT_TYPES),
        "schema_version": MANIFEST_SCHEMA,
        "seed": seed,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if (output.exists() or manifest_path.exists()) and not replace:
        raise ArtifactError(f"refusing to overwrite dataset or manifest at {output}; pass --replace")
    # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-canonical-output:p1:inst-sdg-write-temp
    output = output.resolve()
    manifest_path = manifest_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    transaction_id = uuid.uuid4().hex
    new_output = output.parent / f".{output.name}.{transaction_id}.candidate"
    new_manifest = output.parent / f".{manifest_path.name}.{transaction_id}.candidate"
    old_output = output.parent / f".{output.name}.{transaction_id}.rollback"
    old_manifest = output.parent / f".{manifest_path.name}.{transaction_id}.rollback"
    backed_up_output = False
    backed_up_manifest = False
    published_output = False
    published_manifest = False
    try:
        # Materialize and fsync the complete new pair before touching either
        # member of an existing pair.  This makes preparation failures harmless.
        _atomic_write(new_output, payload, replace=False)
        _atomic_write(new_manifest, (canonical_json(manifest) + "\n").encode("utf-8"), replace=False)
        if output.exists():
            os.replace(output, old_output)
            backed_up_output = True
        if manifest_path.exists():
            os.replace(manifest_path, old_manifest)
            backed_up_manifest = True
        # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-canonical-output:p1:inst-sdg-write-temp
        # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-canonical-output:p1:inst-sdg-finalize
        os.replace(new_output, output)
        published_output = True
        os.replace(new_manifest, manifest_path)
        published_manifest = True
    except BaseException as primary:
        # Restore the last visible canonical pair before touching disposable
        # candidates.  os.replace intentionally overwrites a partially
        # published member on Windows, avoiding a cleanup dependency in the
        # restoration path.
        rollback_errors: list[str] = []
        for destination, backup, backed_up, published in (
            (output, old_output, backed_up_output, published_output),
            (manifest_path, old_manifest, backed_up_manifest, published_manifest),
        ):
            try:
                if backed_up:
                    os.replace(backup, destination)
                elif published:
                    destination.unlink(missing_ok=True)
            except Exception as exc:
                rollback_errors.append(f"restore {destination.name}: {exc}")
        # Cleanup is explicitly best-effort and cannot prevent or mask pair
        # restoration or the primary publication exception.
        for disposable in (new_output, new_manifest, old_output, old_manifest):
            try:
                disposable.unlink(missing_ok=True)
            except Exception as exc:
                rollback_errors.append(f"cleanup {disposable.name}: {exc}")
        if rollback_errors and hasattr(primary, "add_note"):
            primary.add_note("dataset transaction recovery: " + "; ".join(rollback_errors))
        raise
    else:
        for backup in (old_output, old_manifest):
            try:
                backup.unlink(missing_ok=True)
            except Exception:
                # The committed canonical pair is valid; stale private backup
                # cleanup is non-authoritative and may be retried manually.
                pass
    # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-canonical-output:p1:inst-sdg-finalize
    # @cpt-begin:cpt-event-prediction-algo-synthetic-data-generation-canonical-output:p1:inst-sdg-output-result
    result = {"dataset": str(output), "manifest": str(manifest_path), **manifest}
    # @cpt-end:cpt-event-prediction-algo-synthetic-data-generation-canonical-output:p1:inst-sdg-output-result
    return result


def read_dataset(path: Path) -> list[EventSequence]:
    sequences: list[EventSequence] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArtifactError(f"cannot read dataset {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
            _validate_observable_record(record)
            events = tuple(ObservedEvent(**event) for event in record["events"])
            sequences.append(
                EventSequence(
                    sequence_id=record["sequence_id"],
                    events=events,
                    generation_seed=int(record["provenance"]["generation_seed"]),
                    schema_version=record["schema_version"],
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, ArtifactError) as exc:
            raise ArtifactError(f"invalid dataset record at line {line_number}: {exc}") from exc
    if not sequences:
        raise ArtifactError("dataset is empty")
    return sequences


def write_json(path: Path, value: Any, replace: bool = False) -> None:
    _atomic_write(path, (canonical_json(value) + "\n").encode("utf-8"), replace)


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ArtifactError(f"cannot hash artifact {path}: {exc}") from exc


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSON artifact {path}: {exc}") from exc


def resolve_portable_reference(root: Path, reference: str, *, field: str = "artifact reference") -> Path:
    """Resolve a canonical POSIX relative reference and prove containment before access."""
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise ArtifactError(f"{field} is not a portable relative path: {reference!r}")
    pure = PurePosixPath(reference)
    if pure.is_absolute() or pure.as_posix() != reference or any(part in {"", ".", ".."} for part in pure.parts):
        raise ArtifactError(f"{field} is not a portable relative path: {reference!r}")
    if ":" in pure.parts[0]:
        raise ArtifactError(f"{field} is drive-relative or contains a drive: {reference!r}")
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*pure.parts).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactError(f"{field} resolves outside its artifact root: {reference!r}") from exc
    return candidate
