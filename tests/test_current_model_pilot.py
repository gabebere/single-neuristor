"""Independent electrical/thermal controls for optional material-model hypotheses."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from neuristor.config import load_toml
from neuristor.current_drive_sim import (
    CurrentDriveParams,
    ExperimentalCurrentModel,
    simulate_current_waveforms,
    simulate_experimental_current_waveforms,
    two_region_electrical_channels,
    coupled_thermal_propagator,
    advance_coupled_temperatures,
)
from neuristor.model import YuanhangResistParams
from neuristor.workflows import _current_params_from_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", ["baseline", "phase_lag"])
@pytest.mark.parametrize("capacitance", [0.0, 5e-12])
def test_original_and_zero_phase_time_are_bitwise_original(kind, capacitance):
    base, *_ = _current_params_from_config(
        load_toml(ROOT / "experiments/current/specimen_filament_surroundings_20261027.toml")
    )
    base = replace(base, C_F=capacitance, t_pre_s=1e-9, t_end_s=15e-9, dt_s=0.05e-9)
    time = np.array([-1.0, 0.0, 15.0]) * 1e-9
    current = np.array([[0.0, 0.0], [267.0, 606.0], [267.0, 606.0]])
    original = simulate_current_waveforms(current, base, waveform_time_s=time, audit_proximity=True)
    experimental = simulate_experimental_current_waveforms(
        current, base, ExperimentalCurrentModel(kind=kind, phase_tau_s=0.0), waveform_time_s=time, audit_proximity=True
    )
    for left, right in zip(original, experimental):
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.parametrize("kind", ["warm_parallel", "series_bottleneck"])
def test_uniform_temperature_preserves_measured_law_and_topology_balance(kind):
    resistance = np.geomspace(18.0, 1e6, 50)
    voltage = np.linspace(0.0, 1.0, 50)
    for weight in [0.1, 0.5, 0.9]:
        uniform = two_region_electrical_channels(voltage, resistance, resistance, weight, kind)
        np.testing.assert_allclose(uniform["R"], resistance, rtol=1e-7)
        nonuniform = two_region_electrical_channels(voltage, resistance, 2 * resistance, weight, kind)
        np.testing.assert_allclose(
            nonuniform["P_primary"] + nonuniform["P_secondary"], voltage**2 / nonuniform["R"], rtol=2e-7
        )
        if kind == "warm_parallel":
            np.testing.assert_allclose(
                nonuniform["I_primary"] + nonuniform["I_secondary"], nonuniform["I_device"], rtol=2e-7
            )
        else:
            np.testing.assert_allclose(nonuniform["V_primary"] + nonuniform["V_secondary"], voltage, rtol=2e-7)
            np.testing.assert_array_equal(nonuniform["I_primary"], nonuniform["I_secondary"])


def test_internal_thermal_exchange_conserves_energy_without_bath_or_joule_heat():
    c1, c2 = 2e-14, 7e-14
    propagator = coupled_thermal_propagator(
        C_primary_J_per_K=c1,
        C_secondary_J_per_K=c2,
        S_primary_W_per_K=0.0,
        S_secondary_W_per_K=0.0,
        coupling_W_per_K=3e-6,
        dt_s=20e-9,
    )
    a, b = advance_coupled_temperatures(350.0, 300.0, 0.0, 0.0, T0_K=300.0, propagator=propagator)
    assert c1 * (a - 300) + c2 * (b - 300) == pytest.approx(c1 * 50.0, rel=1e-12)
    assert 300 < b < a < 350


@pytest.mark.parametrize("kind", ["warm_parallel", "series_bottleneck"])
def test_two_smaller_steps_converge_to_independent_rc_and_thermal_ode(kind):
    current, resistance, capacitance = 2e-3, 20.0, 10e-12
    fraction, c1, c2, s1, s2, k = 0.3, 1e-15, 3e-15, 0.2e-6, 0.4e-6, 1e-6

    def rhs(t, temperatures):
        voltage = current * resistance * (1 - np.exp(-t / (capacitance * resistance)))
        power = voltage**2 / resistance
        a, b = temperatures
        return [
            (fraction * power - s1 * (a - 300) - k * (a - b)) / c1,
            ((1 - fraction) * power - s2 * (b - 300) + k * (a - b)) / c2,
        ]

    reference = solve_ivp(rhs, [0.0, 5e-9], [300.0, 300.0], rtol=1e-11, atol=1e-10, dense_output=True)
    errors = []
    for dt in [0.05, 0.025, 0.0125]:
        params = CurrentDriveParams(
            dt_s=dt * 1e-9,
            t_end_s=5e-9,
            C_F=capacitance,
            C_th_J_per_K=c1,
            S_e_W_per_K=s1,
            T0_K=300.0,
            T_init_K=300.0,
            resist_params=YuanhangResistParams(R0=0.0, Rm0=resistance, Rm_factor=1.0),
        )
        model = ExperimentalCurrentModel(
            kind=kind, primary_fraction=fraction, C_secondary_J_per_K=c2, S_secondary_W_per_K=s2, coupling_W_per_K=k
        )
        out = simulate_experimental_current_waveforms([2000.0, 2000.0], params, model, waveform_time_s=[0.0, 5e-9])[0]
        expected = reference.sol(float(out["t"][-1]))
        errors.append(np.max(np.abs(np.array([out["T"][-1], out["T_secondary"][-1]]) - expected)))
        exact_voltage = current * resistance * (1 - np.exp(-out["t"].astype(float) / (capacitance * resistance)))
        # Late RC updates encounter the inherited float32 quantization floor.
        # Require exact original-solver identity throughout, and independently
        # check the analytically resolved charging part at the same tolerance.
        original = simulate_current_waveforms([2000.0, 2000.0], params, waveform_time_s=[0.0, 5e-9])[0]
        np.testing.assert_array_equal(out["V_vo2"], original["V_vo2"])
        charging = out["t"] <= 2 * capacitance * resistance
        np.testing.assert_allclose(out["V_vo2"][charging], exact_voltage[charging], rtol=3e-7, atol=1e-8)
    assert errors[1] < 0.6 * errors[0] and errors[2] < 0.6 * errors[1]
    assert errors[2] < 0.12


@pytest.mark.parametrize("kind", ["warm_parallel", "series_bottleneck", "phase_lag"])
def test_algebraic_voltage_limit_and_all_heat_channels(kind):
    params = CurrentDriveParams(dt_s=0.05e-9, t_end_s=3e-9, C_F=0.0, T0_K=300.0, T_init_K=300.0)
    model = ExperimentalCurrentModel(kind=kind, phase_tau_s=1e-9)
    out = simulate_experimental_current_waveforms([400.0, 400.0], params, model, waveform_time_s=[0.0, 3e-9])[0]
    np.testing.assert_allclose(out["V_vo2"], out["I_in"] * out["R"], rtol=1e-7)
    np.testing.assert_allclose(out["P_primary"] + out["P_secondary"], out["P"], rtol=3e-7)


def test_phase_state_follows_independent_exponential_step_target(monkeypatch):
    import neuristor.current_drive_sim as physics

    class ForcedTarget:
        def __init__(self, params, *, size, **kwargs):
            self.count = 0
            self.size = size

        def initialize(self, temperature):
            pass

        def evaluate(self, temperature):
            g = np.full(self.size, 0.0 if self.count < 5 else 1.0, dtype=np.float32)
            self.count += 1
            return np.full(self.size, 20.0, dtype=np.float32), g

    monkeypatch.setattr(physics, "HysteresisArray", ForcedTarget)
    p = CurrentDriveParams(
        dt_s=0.1e-9,
        t_end_s=3e-9,
        T0_K=300.0,
        T_init_K=300.0,
        resist_params=YuanhangResistParams(R0=0.0, Rm0=20.0, Rm_factor=1.0),
    )
    out = simulate_experimental_current_waveforms(
        [400.0, 400.0], p, ExperimentalCurrentModel(kind="phase_lag", phase_tau_s=0.5e-9), waveform_time_s=[0.0, 3e-9]
    )[0]
    expected = 1 - np.exp(-np.maximum(np.arange(len(out["t"])) - 5, 0) * 0.1 / 0.5)
    np.testing.assert_allclose(out["g_dyn"], expected, rtol=2e-7, atol=1e-7)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(primary_fraction=0.0),
        dict(phase_tau_s=-1.0),
        dict(coupling_W_per_K=-1.0),
        dict(C_secondary_J_per_K=0.0),
        dict(kind="unknown"),
    ],
)
def test_experimental_parameters_reject_invalid_physics(kwargs):
    with pytest.raises(ValueError):
        ExperimentalCurrentModel(**kwargs)


def test_model_pilot_cli_writes_grid_ablations_and_full_checks(tmp_path):
    import json
    import pandas as pd
    from typer.testing import CliRunner
    from neuristor.cli import app

    recipe = ROOT / "experiments/current/specimen_model_pilot_20261028.toml"
    config = load_toml(recipe)
    anchor = config["model_pilot"]["anchors"][0]
    args = ["analyze", "model-pilot", "--config", str(recipe), "--output-root", str(tmp_path)]
    for override in [
        "model_pilot.workers=1",
        "model_pilot.weights=[0.5]",
        "model_pilot.secondary_capacity_ratios=[10.0]",
        "model_pilot.coupling_ratios=[1.0]",
        "model_pilot.phase_tau_ns=[0.6]",
        "model_pilot.screen_dt_ns=0.5",
        "model_pilot.verification_dt_ns=[0.5,0.25,0.125,0.0625]",
        "model_pilot.constant_duration_ns=80.0",
        'model_pilot.anchors=[{name="smoke",values=' + json.dumps(anchor["values"]) + "}]",
    ]:
        args += ["--set", override]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    root = next(tmp_path.glob("*/run.json")).parent
    manifest = json.loads((root / "run.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    assert manifest["status"] == "completed" and metrics["verified_cases"] == 7
    assert metrics["declared_cases"] == 4 and metrics["total_screened_cases"] > 4
    assert metrics["long_control_cases"] == 4 and not metrics["calibration_adopted"]
    comparison = pd.read_csv(root / "comparison.csv")
    assert sum(comparison.role == "matched_baseline") == 3
    for folder in (root / "verification").iterdir():
        table = pd.read_csv(folder / "verification.csv")
        assert len(table) == 22 * 4 and int(table.target.sum()) == 11 * 4
    for artifact in manifest["artifacts"]:
        assert (root / artifact["path"]).is_file(), artifact
    assert (root / "source/neuristor/current_drive_sim.py").is_file()
    assert (root / "figures/pilot_waveforms.png").is_file()


@pytest.mark.parametrize("secondary", [400.0, float("nan")])
def test_secondary_temperature_is_a_physical_gate(secondary):
    from types import SimpleNamespace
    from neuristor.fitting.steady_inference import evaluate_settled
    from neuristor.fitting.joint_inference import natural_vector

    config = load_toml(ROOT / "experiments/current/specimen_model_pilot_20261028.toml")
    base, *_ = _current_params_from_config(config)
    t = np.arange(251.0)
    v = 200 + 10 * np.sin(2 * np.pi * 0.05 * t)
    data = SimpleNamespace(
        time_ns=t, voltage_mV=v[:, None], current_uA=np.full((251, 1), 400.0), nominal_drives_mV=np.array([1.0])
    )

    def fake(*args, **kwargs):
        return [
            dict(
                t=t * 1e-9,
                V_vo2=v * 1e-3,
                T=np.full(251, 314.4),
                T_secondary=np.full(251, secondary),
                minimum_proximity_slope=np.array(1.0),
            )
        ]

    if np.isnan(secondary):
        with pytest.raises(ValueError, match="secondary temperature"):
            evaluate_settled(data, natural_vector(base), base, [0], config["steady"], 0.025, simulator=fake)
    else:
        rows, _, outside = evaluate_settled(
            data, natural_vector(base), base, [0], config["steady"], 0.025, simulator=fake
        )
        assert outside and not rows[0]["temperature_valid"]
        assert rows[0]["temperature_max_K"] == 400.0 and rows[0]["primary_temperature_max_K"] == 314.4


def test_realized_metadata_contains_only_active_physical_parameters():
    from neuristor.fitting.model_pilot import realized_units

    base = CurrentDriveParams(C_th_J_per_K=1e-14, S_e_W_per_K=2e-6)
    baseline = realized_units(base, dict(kind="baseline"))
    assert "C_secondary_J_per_K" not in baseline and "phase_tau_s" not in baseline
    phase = realized_units(base, dict(kind="phase_lag", phase_tau_ns=2.0))
    assert phase["phase_tau_s"] == 2e-9 and "primary_fraction" not in phase
    spatial = realized_units(
        base, dict(kind="series_bottleneck", weight=0.25, capacity_ratio=10.0, cooling_ratio=1.0, coupling_ratio=1.0)
    )
    assert spatial["C_primary_J_per_K"] == 2.5e-15
    assert spatial["C_secondary_J_per_K"] == pytest.approx(7.5e-14)
    assert spatial["coupling_W_per_K"] == 2e-6 and "phase_tau_s" not in spatial
