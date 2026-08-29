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
    estimate_thermal_capacitance,
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


def _synthetic_thermal_waveforms(
    params: YuanhangResistParams,
    *,
    ambient_K: float,
    conductance_mW_per_K: float,
    capacitance_pJ_per_K: float,
    electrical_capacitance_pF: float = 0.0,
) -> pd.DataFrame:
    """Generate self-consistent heating edges with an optional electrical current."""

    time_ns = np.arange(-200.0, 251.0)
    frames: list[pd.DataFrame] = []
    for drive_mV, plateau_current_uA in ((100, 80.0), (150, 100.0), (200, 120.0)):
        resistive_current_uA = np.zeros_like(time_ns)
        ramp = (time_ns > -15.0) & (time_ns < 15.0)
        resistive_current_uA[ramp] = plateau_current_uA * (time_ns[ramp] + 15.0) / 30.0
        resistive_current_uA[time_ns >= 15.0] = plateau_current_uA
        temperature_K = np.full_like(time_ns, ambient_K)
        voltage_mV = np.zeros_like(time_ns)
        for index in range(len(time_ns)):
            resistance_ohm = float(
                heating_branch_resistance_ohm(temperature_K[index], params)
            )
            voltage_mV[index] = resistive_current_uA[index] * resistance_ohm / 1000.0
            if index + 1 < len(time_ns):
                power_mW = resistive_current_uA[index] * voltage_mV[index] * 1e-6
                temperature_K[index + 1] = temperature_K[index] + (
                    power_mW
                    - conductance_mW_per_K * (temperature_K[index] - ambient_K)
                ) / capacitance_pJ_per_K
        measured_current_uA = (
            resistive_current_uA
            + float(electrical_capacitance_pF) * np.gradient(voltage_mV, time_ns)
        )
        frames.append(
            pd.DataFrame(
                {
                    "source_file": f"{drive_mV}mv0_converted.csv",
                    "nominal_drive_mV": float(drive_mV),
                    "time_ns": time_ns,
                    "input_current_uA": measured_current_uA,
                    "output_voltage_mV": voltage_mV,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_thermal_capacitance_fit_recovers_synthetic_time_constant() -> None:
    params = YuanhangResistParams(
        R0=0.8,
        Ea_over_k=2537.0,
        Rm0=18.2,
        Rm_factor=1.0,
        w=6.88,
        Tc_K=333.49,
        beta=0.299,
        T_min_K=285.0,
        T_max_K=375.0,
    )
    ambient_K = 314.4
    conductance = 0.003675
    expected_capacitance = 0.05
    traces = _synthetic_thermal_waveforms(
        params,
        ambient_K=ambient_K,
        conductance_mW_per_K=conductance,
        capacitance_pJ_per_K=expected_capacitance,
    )
    estimate = estimate_thermal_capacitance(
        traces,
        resistance=params,
        S_e_mW_per_K=conductance,
        ambient_temperature_K=ambient_K,
        near_transition_check_mV=None,
        bootstrap_samples=20,
        fit_window_jitter_ns=0,
    )
    row = estimate.result.iloc[0]
    assert float(row["C_th_pJ_per_K"]) == pytest.approx(expected_capacitance, rel=0.03)
    assert float(row["tau_th_ns"]) == pytest.approx(
        expected_capacitance / conductance,
        rel=0.03,
    )
    assert len(estimate.trace_fits) == 3


def test_thermal_capacitance_fit_subtracts_adopted_electrical_current() -> None:
    params = YuanhangResistParams(
        R0=0.8,
        Ea_over_k=2537.0,
        Rm0=18.2,
        Rm_factor=1.0,
        w=6.88,
        Tc_K=333.49,
        beta=0.299,
        T_min_K=285.0,
        T_max_K=375.0,
    )
    electrical_capacitance_pF = 0.39
    expected_capacitance_pJ_per_K = 0.05
    traces = _synthetic_thermal_waveforms(
        params,
        ambient_K=314.4,
        conductance_mW_per_K=0.003675,
        capacitance_pJ_per_K=expected_capacitance_pJ_per_K,
        electrical_capacitance_pF=electrical_capacitance_pF,
    )
    estimate = estimate_thermal_capacitance(
        traces,
        resistance=params,
        S_e_mW_per_K=0.003675,
        ambient_temperature_K=314.4,
        electrical_capacitance_pF=electrical_capacitance_pF,
        near_transition_check_mV=None,
        bootstrap_samples=20,
        fit_window_jitter_ns=0,
    )
    row = estimate.result.iloc[0]
    assert float(row["electrical_capacitance_pF"]) == pytest.approx(0.39)
    assert float(row["C_th_pJ_per_K"]) == pytest.approx(expected_capacitance_pJ_per_K, rel=0.04)


def test_thermal_capacitance_cli_writes_bundle(tmp_path: Path) -> None:
    params = YuanhangResistParams(
        R0=0.8,
        Ea_over_k=2537.0,
        Rm0=18.2,
        Rm_factor=1.0,
        w=6.88,
        Tc_K=333.49,
        beta=0.299,
        T_min_K=285.0,
        T_max_K=375.0,
    )
    traces = _synthetic_thermal_waveforms(
        params,
        ambient_K=314.4,
        conductance_mW_per_K=0.003675,
        capacitance_pJ_per_K=0.05,
    )
    for source_file, frame in traces.groupby("source_file"):
        frame[["time_ns", "input_current_uA", "output_voltage_mV"]].to_csv(
            tmp_path / str(source_file),
            header=False,
            index=False,
        )
    preset = tmp_path / "resistance.json"
    preset.write_text(json.dumps({"resist_params": dataclasses.asdict(params)}))
    output_root = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "analyze",
            "thermal-capacitance",
            "--data",
            str(tmp_path),
            "--resistance-preset",
            str(preset),
            "--conductance-mW-per-K",
            "0.003675",
            "--near-transition-check-mV",
            "200",
            "--bootstrap-samples",
            "20",
            "--fit-window-jitter-ns",
            "0",
            "--output-root",
            str(output_root),
        ],
    )
    assert result.exit_code == 0, result.output
    bundles = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(bundles) == 1
    assert (bundles[0] / "thermal_capacitance_estimate.csv").is_file()
    assert (bundles[0] / "thermal_trajectories.csv").is_file()
    assert (bundles[0] / "figures" / "thermal_capacitance.png").is_file()


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


def test_lab_analysis_cli_archives_first_oscillating_trace_plot(tmp_path: Path) -> None:
    time_ns = np.arange(-200.0, 301.0)
    stable = pd.DataFrame(
        {
            "time": time_ns,
            "current": np.where(time_ns < 0.0, 0.0, 100.0),
            "voltage": np.where(time_ns < 0.0, 0.0, 200.0),
        }
    )
    oscillating_voltage = 200.0 + 35.0 * np.sin(2.0 * np.pi * time_ns / 20.0)
    oscillating = pd.DataFrame(
        {
            "time": time_ns,
            "current": np.where(time_ns < 0.0, 0.0, 120.0),
            "voltage": np.where(time_ns < 0.0, 0.0, oscillating_voltage),
        }
    )
    stable.to_csv(tmp_path / "100mv0_converted.csv", header=False, index=False)
    oscillating.to_csv(tmp_path / "200mv0_converted.csv", header=False, index=False)
    output_root = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "analyze",
            "lab",
            "--data",
            str(tmp_path),
            "--output-root",
            str(output_root),
        ],
    )
    assert result.exit_code == 0, result.output
    bundles = list(output_root.iterdir())
    assert len(bundles) == 1
    assert (bundles[0] / "figures" / "pre_onset_trace.png").is_file()
    assert (bundles[0] / "figures" / "oscillation_onset_trace.png").is_file()
    assert (bundles[0] / "figures" / "oscillation_onset_bracket.png").is_file()
    metrics = json.loads((bundles[0] / "metrics.json").read_text())
    assert metrics["last_nonoscillating_source_file"] == "100mv0_converted.csv"
    assert metrics["first_oscillating_source_file"] == "200mv0_converted.csv"
    assert metrics["last_nonoscillating_current_step_uA"] == pytest.approx(100.0)
    assert metrics["first_oscillating_current_step_uA"] == pytest.approx(120.0)


def test_model_validation_cli_writes_blind_comparison_bundle(tmp_path: Path) -> None:
    time_ns = np.arange(-200.0, 301.0)
    stable_voltage = np.where(time_ns < 0.0, 0.0, 200.0)
    oscillating_voltage = np.where(
        time_ns < 0.0,
        0.0,
        200.0 + 35.0 * np.sin(2.0 * np.pi * time_ns / 20.0),
    )
    for drive, current, voltage in (
        (100, 100.0, stable_voltage),
        (200, 120.0, oscillating_voltage),
    ):
        pd.DataFrame(
            {
                "time": time_ns,
                "current": np.where(time_ns < 0.0, 0.0, current),
                "voltage": voltage,
            }
        ).to_csv(tmp_path / f"{drive}mv0_converted.csv", header=False, index=False)

    recipe = tmp_path / "validation.toml"
    recipe.write_text(
        f'''schema_version = 1
name = "Tiny model validation"
kind = "simulation"
model = "current"

[time]
dt_ns = 1.0
pre_us = 0.2
duration_us = 0.3
[input]
amplitude_uA = 120.0
on_us = 0.0
off_us = 0.3
[initial]
temperature_K = 325.0
[electrical]
C_pF = 0.0
[thermal]
C_th_pJ_per_K = 49.62776831
S_e_mW_per_K = 0.20558726
T0_K = 325.0
[resistance]
preset = "yuanhang"
start_branch = "insulator"
[lab_validation]
data_directory = "{tmp_path.as_posix()}"
convergence_dt_ns = 0.5
convergence_drives_mV = [100.0]
[sensitivity]
electrical_capacitances_pF = [0.0]
thermal_capacitances_pJ_per_K = [49.62776831]
pulse_duration_ns = 300.0
pre_duration_ns = 200.0
post_duration_ns = 50.0
[output]
root = "runs"
'''
    )
    output_root = tmp_path / "validation-runs"
    result = runner.invoke(
        app,
        ["analyze", "model-validation", "--config", str(recipe), "--output-root", str(output_root)],
    )
    assert result.exit_code == 0, result.output
    bundle = next(output_root.iterdir())
    assert (bundle / "comparison.csv").is_file()
    assert (bundle / "capacitance_sensitivity.csv").is_file()
    assert (bundle / "figures" / "model_vs_experiment.png").is_file()
    assert (bundle / "figures" / "representative_traces.png").is_file()
    assert (bundle / "figures" / "capacitance_sensitivity.png").is_file()
    metrics = json.loads((bundle / "metrics.json").read_text())
    assert metrics["measured_oscillating_waveforms"] == 1
    assert metrics["waveforms"] == 2
