"""Stable command-line interface for the local workflow."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .artifacts import write_dataset
from .benchmark import run_benchmark
from .catalog import EVENT_TYPES
from .config import load_config
from .domain import EventPredictionError
from .model import load_model, predict_prefix
from .processing import split_and_build_examples
from .simulator import generate_sequences
from .schemas import SCHEMA_FILES, load_schema, validate_schema


def _path(value: str) -> Path:
    # @cpt-begin:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1:inst-cli-resolve-paths
    result = Path(value).expanduser().resolve()
    # @cpt-end:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1:inst-cli-resolve-paths
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="event-prediction", description="Synthetic next-event prediction")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate a canonical JSONL dataset")
    generate.add_argument("--config", type=_path)
    generate.add_argument("--count", type=int, default=10_000)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--output", type=_path, required=True)
    generate.add_argument("--replace", action="store_true")

    benchmark = commands.add_parser("benchmark", help="run one or more preliminary benchmark seeds")
    benchmark.add_argument("--config", type=_path)
    benchmark.add_argument("--count", type=int, default=1_000)
    benchmark.add_argument("--seeds", type=int, nargs="+", required=True)
    benchmark.add_argument("--output-dir", type=_path, required=True)
    benchmark.add_argument("--replace", action="store_true")

    predict = commands.add_parser("predict", help="predict the next event from an observable prefix")
    predict.add_argument("--model", type=_path, required=True)
    predict.add_argument("--prefix", nargs="+", required=True, metavar="EVENT")

    verify = commands.add_parser("verify", help="run deterministic local data and model boundary checks")
    verify.add_argument("--config", type=_path)
    verify.add_argument("--seed", type=int, default=20261021)
    return parser


def _generate(args: argparse.Namespace) -> dict[str, object]:
    # @cpt-begin:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-request
    config = load_config(args.config)
    # @cpt-end:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-request
    sequences = generate_sequences(args.count, args.seed, config)
    # @cpt-begin:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-write
    artifacts = write_dataset(sequences, args.output, config, args.seed, replace=args.replace)
    # @cpt-end:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-write
    # @cpt-begin:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-return
    result = artifacts
    # @cpt-end:cpt-event-prediction-flow-synthetic-data-generation-generate:p1:inst-sdg-return
    return result


def _benchmark(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config)
    result = run_benchmark(args.count, args.seeds, config, args.output_dir, replace=args.replace)
    return {
        "result": str(args.output_dir / "benchmark.json"),
        "status": result["status"],
        "metrics": result["aggregate"],
    }


def _predict(args: argparse.Namespace) -> dict[str, object]:
    model = load_model(args.model)
    return predict_prefix(model, tuple(args.prefix))


def _verify(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config, {"bootstrap_replicates": 20})
    first = generate_sequences(40, args.seed, config)
    second = generate_sequences(40, args.seed, config)
    if [item.as_dict() for item in first] != [item.as_dict() for item in second]:
        raise RuntimeError("same-seed generation was not deterministic")
    if any(set(item.as_dict()) != {"events", "provenance", "schema_version", "sequence_id"} for item in first):
        raise RuntimeError("observable dataset boundary failed")
    manifest, examples = split_and_build_examples(first, args.seed + 1)
    validate_schema("dataset-record.v1", first[0].as_dict())
    validate_schema("split-manifest.v1", manifest.as_dict())
    for schema_name in SCHEMA_FILES:
        load_schema(schema_name)
    all_ids = set(manifest.train_ids) | set(manifest.validation_ids) | set(manifest.test_ids)
    if len(all_ids) != 40:
        raise RuntimeError("sequence split verification failed")
    if not all(examples.values()):
        raise RuntimeError("prefix construction verification failed")
    return {
        "checks": ["configuration", "json_schemas", "deterministic_generation", "observable_boundary", "sequence_split", "prefixes"],
        "event_count": len(EVENT_TYPES),
        "numpy_version": np.__version__,
        "status": "PASS",
    }


# @cpt-algo:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1
# @cpt-dod:cpt-event-prediction-dod-reproducible-cli-demo-commands:p1
def main(argv: Sequence[str] | None = None) -> int:
    # @cpt-begin:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1:inst-cli-parse
    parser = _parser()
    args = parser.parse_args(argv)
    # @cpt-end:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1:inst-cli-parse
    handlers = {"generate": _generate, "benchmark": _benchmark, "predict": _predict, "verify": _verify}
    try:
        # @cpt-begin:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1:inst-cli-dispatch-service
        result = handlers[args.command](args)
        # @cpt-end:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1:inst-cli-dispatch-service
    except (EventPredictionError, ValueError, OSError, RuntimeError) as exc:
        # @cpt-begin:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1:inst-cli-exit-code
        parser.exit(2, f"error: {exc}\n")
        # @cpt-end:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1:inst-cli-exit-code
    # @cpt-begin:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1:inst-cli-format
    print(json.dumps(result, indent=2, sort_keys=True))
    # @cpt-end:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1:inst-cli-format
    # @cpt-begin:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1:inst-cli-dispatch-result
    exit_code = 0
    # @cpt-end:cpt-event-prediction-algo-reproducible-cli-demo-dispatch:p1:inst-cli-dispatch-result
    return exit_code




if __name__ == "__main__":
    raise SystemExit(main())

