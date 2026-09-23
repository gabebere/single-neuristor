import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.config import load_toml
from neuristor.joint_inference import (NAMES, natural_vector, joint_model, encode, decode,
                                      static_rmse, voltage_features, feature_loss)
from neuristor.resistance_custom_analysis import load_experimental_rt
from neuristor.workflows import _current_params_from_config

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "experiments/current/specimen_joint_inference.toml"


def test_joint_coordinates_preserve_physics_and_static_fit():
    config = load_toml(RECIPE)
    base, *_ = _current_params_from_config(config)
    values = natural_vector(base)
    bounds = [config["joint"]["bounds"][name] for name in NAMES]
    np.testing.assert_allclose(decode(encode(values, bounds), bounds), values, rtol=1e-12)
    recovered = joint_model(values, base, .025)
    np.testing.assert_allclose(natural_vector(recovered), values, rtol=1e-12)
    assert np.isclose(recovered.resist_params.R0, base.resist_params.R0)
    rt = load_experimental_rt(ROOT / "data/experimental/100425_chip1_gap3.tsv")
    assert 0.036 < static_rmse(values, rt) < 0.038
    wrong = values.copy()
    wrong[4] += 5
    assert static_rmse(wrong, rt) > 3*static_rmse(values, rt)


def test_joint_loss_penalizes_damped_cycles_and_handles_flat_trace():
    time = np.arange(-200., 251.)
    steady = 180 + 15*np.sin(2*np.pi*.05*time)
    damped = 180 + 15*np.sin(2*np.pi*.05*time)*np.exp(-np.maximum(time-50, 0)/40)
    m = voltage_features(time, steady)
    good = feature_loss(m, m)
    bad = feature_loss(m, voltage_features(time, damped))
    flat = feature_loss(m, voltage_features(time, np.full_like(time, 180)))
    assert sum(good.values()) < 1e-12
    assert bad["amplitude"] > 0.5
    assert flat["amplitude"] > bad["amplitude"]
    assert np.isfinite(list(flat.values())).all()


def test_joint_cli_budgeted_bundle_smoke(tmp_path):
    result = CliRunner().invoke(app, ["analyze", "fit-joint", "--config", str(RECIPE),
        "--set", "joint.population=5", "--set", "joint.generations=0",
        "--set", "joint.local_evaluations=1", "--set", "joint.search_dt_ns=0.5",
        "--set", "joint.verification_dt_ns=[0.5]", "--output-root", str(tmp_path)])
    assert result.exit_code == 0, result.output + repr(result.exception)
    run = next(tmp_path.glob("*/run.json"))
    manifest = json.loads(run.read_text())
    assert manifest["status"] == "completed"
    metrics = json.loads((run.parent/"metrics.json").read_text())
    assert metrics["objective_calls"] <= 12
    assert metrics["simulations"] <= metrics["unique_candidates"]
    assert set(metrics["training_drives_mV"]).isdisjoint(metrics["validation_drives_mV"])
    assert len(metrics["fits"]["static_preserving"]["values"]) == 11
    assert (run.parent/"verification.csv").is_file()
