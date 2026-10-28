"""Electrical/thermal consistency and the explicit original-model limit."""

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.config import load_toml
from neuristor.current_drive_sim import (
    CurrentDriveParams,
    filament_effective_resistance,
    simulate_current_step,
    simulate_current_waveforms,
)
from neuristor.model import YuanhangResistParams
from neuristor.workflows import _current_params_from_config

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "experiments/current/specimen_filament_surroundings_20261027.toml"


def test_summary_preserves_partial_loss_after_dataframe_round_trip():
    import pandas as pd
    from neuristor.orchestration.filament_study import summarize

    rows = [dict(temperature_valid=True, temperature_max_K=360., frequency_valid=True,
                 proximity_admissible=True, frequency_relative_error=.1,
                 robust_amplitude_relative_error=.2, predicted_mean_mV=30., measured_mean_mV=20., loss=.05),
            dict(temperature_valid=True, temperature_max_K=350., frequency_valid=False,
                 proximity_admissible=True, frequency_relative_error=None,
                 robust_amplitude_relative_error=-1., predicted_mean_mV=20., measured_mean_mV=20., loss=None)]
    frame = pd.DataFrame(rows)
    assert np.isnan(frame.loss.iloc[1])
    summary = summarize(frame.to_dict('records'))
    assert summary['dynamic_loss'] == pytest.approx(.05)
    assert summary['score'] == pytest.approx(1001000.05)
    assert summary['objective_loss'] is None and summary['invalid_count'] == 1
    frame.loc[:, 'frequency_valid'] = False
    frame.loc[:, 'loss'] = np.nan
    summary = summarize(frame.to_dict('records'))
    assert summary['dynamic_loss'] is None and np.isfinite(summary['score'])


@pytest.mark.parametrize("fraction", [0, -0.1, 1.01, float("nan"), float("inf")])
def test_fraction_rejects_nonphysical_values(fraction):
    with pytest.raises(ValueError, match="fraction"):
        CurrentDriveParams(filament_fraction=fraction)


def test_uniform_temperature_preserves_the_whole_measured_law():
    resistance = np.geomspace(18.0, 1e6, 100)
    for fraction in [1.0, 0.7, 0.1, 0.01]:
        np.testing.assert_allclose(
            filament_effective_resistance(resistance, resistance, fraction), resistance, rtol=2e-7
        )
    assert filament_effective_resistance(20.0, 1e9, 0.1) == pytest.approx(200.0, rel=1e-6)
    with pytest.raises(ValueError, match="positive"):
        filament_effective_resistance(0.0, 100.0, 0.1)


@pytest.mark.parametrize("capacitance", [0.0, 0.39e-12])
def test_original_model_limit_and_batched_filament_current_balance(capacitance):
    base, *_ = _current_params_from_config(load_toml(RECIPE))
    base = replace(base, t_pre_s=0.0, t_end_s=25e-9, dt_s=0.05e-9, C_F=capacitance)
    default = simulate_current_step(400.0, base)
    explicit = simulate_current_step(400.0, replace(base, filament_fraction=1.0))
    for key in default:
        np.testing.assert_array_equal(default[key], explicit[key])
    local = replace(base, filament_fraction=0.1)
    scalar = simulate_current_step(400.0, local)
    time = np.array([0.0, 25e-9])
    batch = simulate_current_waveforms(np.full((2, 1), 400.0), local, waveform_time_s=time)[0]
    for key in scalar:
        np.testing.assert_array_equal(scalar[key], batch[key])
    np.testing.assert_allclose(
        scalar["I_filament"] + scalar["I_surrounding"], scalar["V_vo2"] / scalar["R"], rtol=3e-7, atol=1e-12
    )
    np.testing.assert_allclose(scalar["P_filament"] + scalar["P_surrounding"], scalar["P"], rtol=4e-7, atol=1e-12)
    np.testing.assert_array_equal(scalar["T_surrounding"], np.full_like(scalar["T"], base.T0_K))
    assert np.all(scalar["P_filament"] >= 0) and np.all(scalar["P_surrounding"] >= 0)
    if capacitance == 0:
        np.testing.assert_allclose(scalar["V_vo2"], scalar["I_in"] * scalar["R"], rtol=1e-7)


def test_partitioned_heating_converges_to_independent_analytic_rc_solution():
    # Constant-R control isolates electrical charging and local heat allocation
    # from hysteretic trajectory discontinuities. R0=0 makes R exactly Rm.
    current, resistance, fraction, capacitance, capacity, duration = 2e-3, 20.0, 0.2, 10e-12, 1e-15, 5e-9
    tau = resistance * capacitance
    integral = duration - 2 * tau * (1 - np.exp(-duration / tau)) + 0.5 * tau * (1 - np.exp(-2 * duration / tau))
    expected = 300.0 + fraction * current**2 * resistance * integral / capacity
    errors = []
    for dt_ns in [0.05, 0.025, 0.0125]:
        params = CurrentDriveParams(
            dt_s=dt_ns * 1e-9,
            t_end_s=duration,
            T0_K=300.0,
            T_init_K=300.0,
            C_F=capacitance,
            C_th_J_per_K=capacity,
            S_e_W_per_K=0.0,
            resist_params=YuanhangResistParams(R0=0.0, Rm0=resistance, Rm_factor=1.0),
            filament_fraction=fraction,
        )
        out = simulate_current_step(current * 1e6, params)
        # The authoritative float32 grid can end one step before a requested
        # duration. Compare the analytic solution at the actual last endpoint.
        end = float(out["t"][-1])
        integral = end - 2 * tau * (1 - np.exp(-end / tau)) + 0.5 * tau * (1 - np.exp(-2 * end / tau))
        expected = 300.0 + fraction * current**2 * resistance * integral / capacity
        errors.append(abs(float(out["T"][-1]) - expected))
        np.testing.assert_allclose(out["R"], resistance, rtol=2e-7)
    assert errors[1] < 0.6 * errors[0]
    assert errors[2] < 0.6 * errors[1]
    assert errors[2] < 0.12


def test_extended_search_caches_the_fraction_and_keeps_static_parameters_fixed(monkeypatch):
    from types import SimpleNamespace
    from neuristor.fitting import steady_inference as inference
    from neuristor.fitting.joint_inference import NAMES, natural_vector

    config = load_toml(RECIPE)
    base, *_ = _current_params_from_config(config)
    settings = dict(
        config["steady"],
        restarts=1,
        fixed_resistance_restarts=1,
        fixed_parameters=list(NAMES),
        population=5,
        maximum_generations=2,
        local_evaluations=1,
        maximum_evaluations=16,
        fit_filament_fraction=True,
        filament_fraction_bounds=[0.01, 1.0],
        search_seconds=30.0,
        initial_seed_count=1,
    )
    monkeypatch.setattr(inference, "static_rmse", lambda *args: 0.01)
    monkeypatch.setattr(inference, "settled_features", lambda *args, **kwargs: {})

    def evaluate(data, values, params, indices, settings, dt, measured):
        loss = (params.filament_fraction - 0.2) ** 2
        return [dict(frequency_valid=True, loss=loss, proximity_admissible=True)], None, False

    monkeypatch.setattr(inference, "evaluate_settled", evaluate)
    data = SimpleNamespace(time_ns=np.array([0.0, 1.0]), voltage_mV=np.zeros((2, 1)), train_indices=np.array([0]))
    records = []
    result = inference.search_settled(
        data, None, base, settings, [natural_vector(base)], None, lambda record, state: records.append(record)
    )
    assert result["unique_candidates"] >= 5
    assert len({r["filament_fraction"] for r in records}) >= 5
    assert all(r["values"] == records[0]["values"] for r in records)
    assert all(r["static_fixed"] for r in records)
    assert result["best"]["score"] < 0.02


def test_filament_cli_writes_a_complete_diagnostic_bundle(tmp_path):
    args = ["analyze", "filament-study", "--config", str(RECIPE), "--output-root", str(tmp_path)]
    for override in [
        "steady.restarts=1",
        "steady.fixed_resistance_restarts=1",
        "steady.population=5",
        "steady.maximum_generations=0",
        "steady.maximum_evaluations=5",
        "steady.local_evaluations=1",
        "steady.search_seconds=30.0",
        "steady.search_dt_ns=0.5",
        "steady.verification_dt_ns=[0.5,0.25,0.125]",
        "filament_study.fractions=[1.0,0.1]",
    ]:
        args.extend(["--set", override])
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    root = next(tmp_path.glob("*/run.json")).parent
    manifest = json.loads((root / "run.json").read_text())
    assert manifest["status"] == "completed"
    metrics = json.loads((root / "metrics.json").read_text())
    assert metrics["original_measurements"] == 22 and metrics["target_records"] == 11
    assert metrics["fit_counts"] == {"baseline": 5, "filament": 5}
    assert not metrics["calibration_adopted"]
    for artifact in manifest["artifacts"]:
        assert (root / artifact["path"]).is_file()
    assert (root / "convergence.csv").is_file() and (root / "START_HERE.html").is_file()
    assert len(list((root / "native").glob("*/*.npz"))) == 56
    from neuristor.orchestration.filament_study import _validated_search_reuse
    from neuristor.config import ConfigError

    resolved = json.loads((root / "resolved_config.json").read_text())
    reused, provenance = _validated_search_reuse(root, resolved, [ROOT / "src/neuristor/current_drive_sim.py"])
    assert reused == root and provenance["source_run_id"] == manifest["id"]
    resolved["steady"]["frequency_weight"] = 0.0
    with pytest.raises(ConfigError, match="recipe changed"):
        _validated_search_reuse(root, resolved, [])
