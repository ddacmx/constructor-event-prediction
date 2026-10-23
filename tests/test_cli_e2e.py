from __future__ import annotations

import json
from pathlib import Path

import pytest

from event_prediction.cli import main


# Scenario: cpt-event-prediction-flow-reproducible-cli-demo-verify
def test_verify_and_generate_cli(tmp_path, capsys) -> None:
    assert main(["verify", "--seed", "12"]) == 0
    verify = json.loads(capsys.readouterr().out)
    assert verify["status"] == "PASS"
    output = tmp_path / "dataset.jsonl"
    assert main(["generate", "--count", "12", "--seed", "12", "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["count"] == 12
    assert output.exists()
    assert Path(result["manifest"]).exists()


# Scenario: cpt-event-prediction-flow-reproducible-cli-demo-reviewer
def test_benchmark_then_predict_cli(tmp_path, capsys) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"min_length": 4, "max_length": 8, "bootstrap_replicates": 20, "regularization_candidates": [0.1]}),
        encoding="utf-8",
    )
    output = tmp_path / "benchmark"
    assert main(["benchmark", "--config", str(config), "--count", "120", "--seeds", "23", "--output-dir", str(output)]) == 0
    benchmark = json.loads(capsys.readouterr().out)
    assert benchmark["status"] == "PRELIMINARY"
    manifest = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    model = output / manifest["run"] / "model-seed-23.model.json"
    assert model.exists()
    assert main([
        "predict", "--model", str(model), "--prefix",
        "api_interface_modified", "compatibility_checked", "integration_test_passed",
    ]) == 0
    prediction = json.loads(capsys.readouterr().out)
    assert prediction["predicted_event"] in prediction["probabilities"]
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)


def test_cli_errors_are_actionable(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["predict", "--model", str(tmp_path / "missing.joblib"), "--prefix", "invalid"])
    assert caught.value.code == 2
    assert "error:" in capsys.readouterr().err

