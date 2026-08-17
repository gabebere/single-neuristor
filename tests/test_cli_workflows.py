from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.config import ConfigError, apply_overrides, load_toml
from neuristor.lab_estimates import estimate_lab_parameters
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


def test_lab_estimates_keep_thermal_capacitance_scenario_dependent() -> None:
    summary = pd.DataFrame(
        {
            "frame_index": range(5),
            "current_inferred_uA": [50.0, 100.0, 150.0, 200.0, 250.0],
            "v_plateau_mean_mV": [100.0, 150.0, 200.0, 250.0, 190.0],
            "v_plateau_vpp_mV": [2.0, 3.0, 5.0, 8.0, 30.0],
            "v_slope_0_30_mV_per_ns": [2.5, 5.0, 7.5, 10.0, 12.5],
        }
    )
    estimates = estimate_lab_parameters(
        summary,
        transition_temperature_K=337.0,
        ambient_temperatures_K=[325.0],
        thermal_times_ns=[10.0, 20.0],
    )
    assert estimates.electrical_capacitance["C_slope_pF"].median() == 20.0
    conductance = float(estimates.thermal_conductance.iloc[0]["S_e_mW_per_K"])
    assert estimates.thermal_capacitance["C_th_pJ_per_K"].tolist() == [conductance * 10.0, conductance * 20.0]
