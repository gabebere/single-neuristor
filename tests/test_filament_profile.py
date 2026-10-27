"""Search provenance, optional objective, resumable budgets and a real small CLI run."""

from pathlib import Path
from types import SimpleNamespace
import copy
import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.config import load_toml
from neuristor.fitting.filament_profile import profile_bounds, profile_seeds, acceptance
from neuristor.fitting.joint_inference import natural_vector
from neuristor.fitting import steady_inference as inference
from neuristor.orchestration.filament_profile import (
    _exclusive,
    _history,
    _verify_frozen,
    _verify_completed,
    prepare_filament_profile,
    plan_filament_profile,
)
from neuristor.workflows import _current_params_from_config

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "experiments/current/specimen_filament_overnight_20261027.toml"


def test_thermal_families_and_coupled_seeds_preserve_circuit_capacitance():
    cfg = load_toml(RECIPE)
    base, *_ = _current_params_from_config(cfg)
    original = copy.deepcopy(cfg["steady"]["bounds"])
    bounds = profile_bounds(cfg["steady"], "volume_scaled", 0.1)
    assert bounds["S_e_mW_per_K"] == pytest.approx(np.array(original["S_e_mW_per_K"]) * 0.1)
    assert bounds["tau_th_ns"] == original["tau_th_ns"]
    assert cfg["steady"]["bounds"] == original
    vector = natural_vector(base)
    vector[1] = 0.006
    seeds = profile_seeds(base, [dict(values=vector.tolist(), filament_fraction=1.0)], bounds, 0.1)
    coupled = next(s for s in seeds if s["coupled_thermal_start"])
    assert coupled["values"][1] == pytest.approx(0.0006)
    assert coupled["values"][2] == vector[2] and coupled["values"][0] == vector[0]
    assert seeds[0]["clamped"]
    unchanged = profile_seeds(base, [dict(values=vector.tolist())], original, 1.0)
    assert len(unchanged) == 1
    np.testing.assert_allclose(unchanged[0]["values"], vector)


def test_mean_term_is_opt_in_and_does_not_repair_missing_cycles():
    t = np.arange(0.0, 301.0)
    m = inference.settled_features(t, 50 + 10 * np.sin(2 * np.pi * 0.05 * t), method="extrema")
    p = inference.settled_features(t, 90 + 10 * np.sin(2 * np.pi * 0.05 * t), method="extrema")
    settings = load_toml(RECIPE)["steady"]
    assert inference.settled_loss(m, p, settings)["loss"] == pytest.approx(0.0, abs=1e-12)
    assert inference.settled_loss(m, p, dict(settings, mean_voltage_weight=1.0, mean_voltage_scale_mV=20.0))[
        "loss"
    ] == pytest.approx(4.0)
    flat = inference.settled_features(t, np.full_like(t, 90.0), method="extrema")
    assert inference.settled_loss(m, flat, dict(settings, mean_voltage_weight=1.0))["loss"] is None


def test_cached_evaluations_preserve_cap_without_resimulation(monkeypatch):
    cfg = load_toml(RECIPE)
    base, *_ = _current_params_from_config(cfg)
    settings = dict(
        cfg["steady"],
        restarts=1,
        fixed_resistance_restarts=1,
        population=5,
        maximum_generations=0,
        local_evaluations=1,
        maximum_evaluations=5,
        search_seconds=30.0,
    )
    monkeypatch.setattr(inference, "static_rmse", lambda *a: 0.01)
    monkeypatch.setattr(inference, "settled_features", lambda *a, **kw: {})
    calls = []

    def evaluate(*args):
        calls.append(1)
        return [dict(frequency_valid=True, loss=0.2, proximity_admissible=True)], None, False

    monkeypatch.setattr(inference, "evaluate_settled", evaluate)
    data = SimpleNamespace(time_ns=np.arange(2.0), voltage_mV=np.zeros((2, 1)), train_indices=[0])
    records = []
    inference.search_settled(
        data,
        None,
        base,
        settings,
        [natural_vector(base)],
        None,
        lambda r, s: records.append(copy.deepcopy(r)),
        initial_records=[],
    )
    assert len(calls) == len(records) == 5
    result = inference.search_settled(
        data,
        None,
        base,
        settings,
        [natural_vector(base)],
        None,
        lambda *a: pytest.fail("Evaluation cap must persist"),
        initial_records=records,
    )
    assert result["unique_candidates"] == 5 and len(calls) == 5
    assert result["best"]["score"] == 0.2
    with pytest.raises(ValueError, match="Duplicate"):
        inference.search_settled(
            data,
            None,
            base,
            settings,
            [natural_vector(base)],
            None,
            lambda *a: None,
            initial_records=records + records[:1],
        )
    wrong = copy.deepcopy(records)
    wrong[0]["filament_fraction"] = 0.1
    with pytest.raises(ValueError, match="different filament"):
        inference.search_settled(
            data, None, base, settings, [natural_vector(base)], None, lambda *a: None, initial_records=wrong
        )
    wrong = copy.deepcopy(records)
    wrong[0]["_cache_key"] = [0.0]
    with pytest.raises(ValueError, match="parameter key"):
        inference.search_settled(
            data, None, base, settings, [natural_vector(base)], None, lambda *a: None, initial_records=wrong
        )


def test_acceptance_rejects_missing_frequency_and_quiet_false_positive():
    settings = load_toml(RECIPE)["profile_search"]["success"]
    frame = pd.DataFrame(
        dict(
            target=[True, False],
            frequency_relative_error=[0.0, None],
            robust_amplitude_relative_error=[0.0, None],
            predicted_mean_mV=[20.0, 20.0],
            measured_mean_mV=[20.0, 20.0],
            temperature_valid=[True, True],
            proximity_admissible=[True, True],
            frequency_valid=[True, False],
            measured_sustained=[True, False],
            measured_robust_vpp_mV=[10.0, 1.0],
            sustained=[True, False],
            predicted_robust_vpp_mV=[10.0, 1.0],
        )
    )
    convergence = pd.DataFrame(
        dict(target=[True, True], features_converged=[True, True], waveform_converged=[True, True])
    )
    assert acceptance(frame, convergence, settings)["success"]
    frame.loc[0, "frequency_relative_error"] = None
    result = acceptance(frame, convergence, settings)
    assert not result["success"] and result["frequency_MAPE_defined"] is None
    frame.loc[0, "frequency_relative_error"] = 0.0
    frame.loc[1, ["sustained", "predicted_robust_vpp_mV"]] = [True, 50.0]
    assert not acceptance(frame, convergence, settings)["gates"]["quiet_controls"]


def test_history_recovers_only_incomplete_tail_and_exclusive_lock(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_bytes(b'{"score":1}\n{"score":')
    assert _history(path) == [{"score": 1}]
    assert path.read_bytes() == b'{"score":1}\n'
    path.write_bytes(b'bad\n{"score":1}\n')
    with pytest.raises(ValueError, match="Malformed"):
        _history(path)
    with _exclusive(tmp_path):
        with pytest.raises(OSError):
            with _exclusive(tmp_path):
                pytest.fail("Second supervisor must not acquire lock")


def test_prepare_freezes_sources_and_inputs_and_detects_tampering(tmp_path):
    cfg = load_toml(RECIPE)
    plan = plan_filament_profile(cfg)
    assert sum(r["group"] == "training" for r in plan["split"]) == 8
    assert sum(r["group"] == "optimizer_excluded" for r in plan["split"]) == 3
    bundle = prepare_filament_profile(cfg, output_root=tmp_path)
    assert json.loads((bundle.root / "status.json").read_text())["state"] == "prepared"
    assert not (bundle.root / "stages").exists()
    _verify_frozen(bundle.root)
    numerical = bundle.root / "inputs/seed_vectors.csv"
    numerical.write_text(numerical.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen source/input"):
        _verify_frozen(bundle.root)
    stage = bundle.root / "stages/example"
    stage.mkdir(parents=True)
    (stage / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="Completed stage changed"):
        _verify_completed(bundle.root, dict(completed={"example": {"sha256": {"result.json": "invalid"}}}))


def test_profile_cli_real_small_search_and_all_final_checks(tmp_path):
    args = ["analyze", "filament-profile", "--config", str(RECIPE), "--output-root", str(tmp_path)]
    for value in [
        "profile_search.fractions=[1.0,0.1]",
        "profile_search.maximum_evaluations=5",
        "profile_search.population=5",
        "profile_search.restarts=1",
        "profile_search.maximum_generations=0",
        "profile_search.local_evaluations=1",
        "profile_search.baseline_seconds=30.0",
        "profile_search.forward_seconds=30.0",
        "profile_search.reverse_seconds=30.0",
        "profile_search.refinement_seconds=30.0",
        "profile_search.wall_hours=0.2",
        "profile_search.profile_hours=0.15",
        "profile_search.search_dt_ns=0.5",
        "profile_search.refinement_dt_ns=0.25",
        "profile_search.verification_dt_ns=[0.5,0.25,0.125]",
        "profile_search.final_dt_ns=[0.25,0.125,0.0625]",
        "profile_search.constant_current_duration_ns=80.0",
    ]:
        args += ["--set", value]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    root = next(tmp_path.glob("*/run.json")).parent
    manifest = json.loads((root / "run.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    assert manifest["status"] == "completed"
    assert json.loads((root / "status.json").read_text())["state"] == "completed"
    assert metrics["coverage_complete"] and metrics["final_candidates_verified"] == 6
    assert metrics["constant_control_cases"] == 6 and not metrics["calibration_adopted"]
    assert metrics["search_stages"] >= 11
    for artifact in manifest["artifacts"]:
        assert (root / artifact["path"]).is_file(), artifact
    final = next((root / "stages").glob("final_refine_*/verification.csv"))
    frame = pd.read_csv(final)
    assert len(frame) == 66 and frame.used_in_this_fit.sum() == 33
    assert len(list((final.parent / "native").glob("*.npz"))) == 22
    assert (root / "figures/profile.png").is_file() and (root / "figures/final_waveforms.png").is_file()
    assert "not blind validation" in (root / "report.md").read_text()
    _verify_completed(root, json.loads((root / "state.json").read_text()))
    again = CliRunner().invoke(app, ["analyze", "filament-profile", "--resume", str(root)])
    assert again.exit_code == 1 and "unfinished" in again.output
