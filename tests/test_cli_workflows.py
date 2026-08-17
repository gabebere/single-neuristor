from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.config import ConfigError, apply_overrides, load_toml
from neuristor.experimental_waveforms import load_converted_sweep
from neuristor.lab_estimates import (
    estimate_environmental_conductance,
    heating_branch_resistance_ohm,
)
from neuristor.model import YuanhangResistParams
from neuristor.runs import RunRegistry
from neuristor.workflows import run_simulation, run_sweep


ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _tiny_current_config(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.toml"
    path.write_text(
        """\
schema_version = 1
name = "Tiny current smoke test"
kind = "simulation"
model = "current"
seed = 0

[time]
dt_ns = 1.0
duration_us = 0.03

[input]
amplitude_uA = 400.0

[electrical]
C_pF = 0.0

[thermal]
C_th_pJ_per_K = 49.62776831
S_e_mW_per_K = 0.20558726
T0_K = 325.0

[resistance]
preset = "yuanhang"
start_branch = "insulator"
"""
    )
    return path


def test_config_override_is_typed_and_typo_safe(tmp_path: Path) -> None:
    config = load_toml(_tiny_current_config(tmp_path))
    updated = apply_overrides(config, ["input.amplitude_uA=625", "electrical.C_pF=10.5"])
    assert updated["input"]["amplitude_uA"] == 625
    assert updated["electrical"]["C_pF"] == 10.5
    try:
        apply_overrides(config, ["electrical.capacitance=2"])
    except ConfigError as exc:
        assert "Unknown configuration path" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("A misspelled override path should fail")


def test_current_workflow_writes_complete_bundle(tmp_path: Path) -> None:
    config = load_toml(_tiny_current_config(tmp_path))
    bundle = run_simulation(config, output_root=tmp_path / "runs", command="test")
    assert (bundle.root / "run.json").is_file()
    assert (bundle.root / "resolved_config.json").is_file()
    assert (bundle.root / "metrics.json").is_file()
    assert (bundle.root / "traces.csv").is_file()
    assert (bundle.root / "figures" / "overview.png").is_file()
    assert (bundle.root / "report.md").is_file()
    manifest = json.loads((bundle.root / "run.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["summary"]["metallic_voltage_floor_V"] > 0.0
    traces = pd.read_csv(bundle.root / "traces.csv")
    assert set(
        [
            "time_us",
            "current_uA",
            "metallic_voltage_floor_V",
            "voltage_V",
            "temperature_K",
            "resistance_ohm",
        ]
    ) <= set(traces)
    expected_floor = traces["current_uA"] * 1e-6 * float(manifest["summary"]["metallic_resistance_ohm"])
    assert (traces["metallic_voltage_floor_V"] - expected_floor).abs().max() < 1e-8


def test_runtime_failure_is_visible_in_bundle(tmp_path: Path) -> None:
    config = copy.deepcopy(load_toml(_tiny_current_config(tmp_path)))
    config["resistance"]["preset"] = "missing-preset.json"
    try:
        run_simulation(config, output_root=tmp_path / "runs", command="test failure")
    except ConfigError:
        pass
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Missing resistance preset should fail")
    manifests = list((tmp_path / "runs").glob("*/run.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "failed"
    assert "Resistance preset does not exist" in manifest["error"]


def test_cli_runs_current_recipe(tmp_path: Path) -> None:
    config = _tiny_current_config(tmp_path)
    result = runner.invoke(
        app, ["simulate", "current", "--config", str(config), "--output-root", str(tmp_path / "runs")]
    )
    assert result.exit_code == 0, result.output
    assert "Completed:" in result.output
    manifests = list((tmp_path / "runs").glob("*/run.json"))
    assert len(manifests) == 1


def test_registry_discovers_new_and_legacy_public_records(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\nversion='0'\n")
    local = tmp_path / "runs" / "new-id"
    local.mkdir(parents=True)
    (local / "run.json").write_text(
        json.dumps({"id": "new-id", "name": "new", "model": "current", "kind": "simulation", "status": "completed"})
    )
    legacy = tmp_path / "public_jobs" / "old-id"
    legacy.mkdir(parents=True)
    (legacy / "job.json").write_text(
        json.dumps({"id": "old-id", "name": "old", "source_model": "voltage", "type": "legacy", "status": "completed"})
    )
    records = RunRegistry(tmp_path).discover()
    assert {(record.id, record.legacy) for record in records} == {("new-id", False), ("old-id", True)}


def test_registry_prefers_published_copy_of_same_run(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\nversion='0'\n")
    manifest = {"id": "same-id", "name": "same", "model": "current", "kind": "simulation", "status": "completed"}
    for root, storage in ((tmp_path / "runs", "local"), (tmp_path / "public_jobs", "public")):
        directory = root / "same-id"
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps({**manifest, "storage": storage}))
    records = RunRegistry(tmp_path).discover()
    assert len(records) == 1
    assert records[0].storage == "public"


def test_checked_in_experiment_recipes_validate() -> None:
    for path in (ROOT / "experiments").rglob("*.toml"):
        load_toml(path)


def test_small_sweep_uses_same_simulation_workflow(tmp_path: Path) -> None:
    base = _tiny_current_config(tmp_path)
    sweep_path = tmp_path / "sweep.toml"
    sweep_path.write_text(
        f"""\
schema_version = 1
name = "Tiny capacitance sweep"
kind = "sweep"
model = "current"

[sweep]
base_config = "{base.name}"
max_points = 4

[[sweep.axes]]
path = "electrical.C_pF"
values = [0.0, 1.0]
"""
    )
    bundle = run_sweep(load_toml(sweep_path), output_root=tmp_path / "runs", command="test sweep")
    summary = pd.read_csv(bundle.root / "sweep.csv")
    assert summary["electrical.C_pF"].tolist() == [0.0, 1.0]
    assert (bundle.root / "figures" / "sweep.png").is_file()


def test_environmental_conductance_uses_settled_pre_onset_trace() -> None:
    params = YuanhangResistParams()
    time_ns = np.arange(-200.0, 251.0)
    target_temperature_K = 330.0
    target_resistance_ohm = float(heating_branch_resistance_ohm(target_temperature_K, params))
    stable_current_uA = np.where(time_ns < 0.0, 0.0, 100.0)
    stable_voltage_mV = np.where(
        time_ns < 0.0,
        0.0,
        100.0 * target_resistance_ohm / 1000.0,
    )
    traces = pd.concat(
        [
            pd.DataFrame(
                {
                    "source_file": "100mv0_converted.csv",
                    "time_ns": time_ns,
                    "input_current_uA": stable_current_uA,
                    "output_voltage_mV": stable_voltage_mV,
                }
            ),
            pd.DataFrame(
                {
                    "source_file": "200mv0_converted.csv",
                    "time_ns": time_ns,
                    "input_current_uA": np.where(time_ns < 0.0, 0.0, 120.0),
                    "output_voltage_mV": np.where(time_ns < 0.0, 0.0, 200.0),
                }
            ),
        ],
        ignore_index=True,
    )
    summary = pd.DataFrame(
        {
            "source_file": ["100mv0_converted.csv", "200mv0_converted.csv"],
            "current_step_uA": [100.0, 120.0],
            "oscillation_detected": [False, True],
        }
    )
    estimate = estimate_environmental_conductance(
        traces,
        summary,
        resistance=params,
        ambient_temperature_K=300.0,
        ambient_interval_K=(300.0, 300.0),
        bootstrap_samples=50,
    )
    row = estimate.result.iloc[0]
    expected_power_uW = 100.0 * (100.0 * target_resistance_ohm / 1000.0) * 1e-3
    assert row["selected_trace"] == "100mv0_converted.csv"
    assert float(row["inferred_temperature_K"]) == pytest.approx(target_temperature_K)
    assert float(row["power_uW"]) == pytest.approx(expected_power_uW)
    assert float(row["S_e_uW_per_K"]) == pytest.approx(expected_power_uW / 30.0)
    assert abs(float(row["resistance_drift_fraction"])) < 1e-12


def test_environmental_conductance_cli_writes_bundle(tmp_path: Path) -> None:
    params = YuanhangResistParams()
    time_ns = np.arange(-200.0, 301.0)
    resistance_ohm = float(heating_branch_resistance_ohm(330.0, params))
    stable_voltage_mV = 100.0 * resistance_ohm / 1000.0
    stable = pd.DataFrame(
        {
            "time": time_ns,
            "current": np.where(time_ns < 0.0, 0.0, 100.0),
            "voltage": np.where(time_ns < 0.0, 0.0, stable_voltage_mV),
        }
    )
    oscillating_voltage = 200.0 + 30.0 * np.sin(2.0 * np.pi * time_ns / 20.0)
    oscillating = pd.DataFrame(
        {
            "time": time_ns,
            "current": np.where(time_ns < 0.0, 0.0, 120.0),
            "voltage": np.where(time_ns < 0.0, 0.0, oscillating_voltage),
        }
    )
    stable.to_csv(tmp_path / "100mv0_converted.csv", header=False, index=False)
    oscillating.to_csv(tmp_path / "200mv0_converted.csv", header=False, index=False)
    preset = tmp_path / "resistance.json"
    preset.write_text(json.dumps({"resist_params": dataclasses.asdict(params)}))
    output_root = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "analyze",
            "conductance",
            "--data",
            str(tmp_path),
            "--resistance-preset",
            str(preset),
            "--ambient-K",
            "300",
            "--ambient-interval-K",
            "300,300",
            "--bootstrap-samples",
            "20",
            "--output-root",
            str(output_root),
        ],
    )
    assert result.exit_code == 0, result.output
    bundles = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(bundles) == 1
    assert (bundles[0] / "conductance_estimate.csv").is_file()
    assert (bundles[0] / "figures" / "environmental_conductance.png").is_file()


def test_numerical_waveform_loader_preserves_units_and_computes_power(tmp_path: Path) -> None:
    time_ns = list(range(-200, 251))
    current_uA = [0.0 if time < 0 else 100.0 for time in time_ns]
    voltage_mV = [0.0 if time < 0 else 200.0 for time in time_ns]
    pd.DataFrame({"t": time_ns, "i": current_uA, "v": voltage_mV}).to_csv(
        tmp_path / "100mv0_converted.csv", header=False, index=False
    )
    traces, summary = load_converted_sweep(tmp_path)
    assert traces["output_power_uW"].max() == 20.0
    assert summary.iloc[0]["current_plateau_uA"] == 100.0
    assert summary.iloc[0]["voltage_plateau_mean_mV"] == 200.0
    assert bool(summary.iloc[0]["oscillation_detected"]) is False
