from pathlib import Path

import numpy as np
import pandas as pd

from neuristor.lab_estimates import _thermal_temperature_from_power_K, reconstruct_hysteresis
from neuristor.model import YuanhangResistParams
from neuristor.config import load_toml
from neuristor.workflows import run_hysteresis_reconstruction


def test_piecewise_linear_thermal_reconstruction_is_exact_for_a_ramp():
    # P(t)=a*t has a closed-form solution; finer replay must preserve it.
    for step in [1.0, .5, .25]:
        t = np.arange(0, 100 + step / 2, step)
        a, conductance, capacitance, ambient = .0001, .004, .05, 314.4
        tau = capacitance / conductance
        expected = ambient + a / conductance * (t - tau * (1 - np.exp(-t / tau)))
        actual = _thermal_temperature_from_power_K(t, a * t, C_th_pJ_per_K=capacitance,
                                                   S_e_mW_per_K=conductance, ambient_temperature_K=ambient)
        assert np.allclose(actual, expected, rtol=0, atol=1e-11)


def test_reconstruction_preserves_known_resistance_and_power_units():
    time = np.arange(-250, 301, dtype=float)
    current = np.where(time >= 0, 50.0, 0.0)
    traces = pd.DataFrame({"source_file": "synthetic", "nominal_drive_mV": 100,
                           "time_ns": time, "input_current_uA": current,
                           "output_voltage_mV": 2 * current})
    result, paths = reconstruct_hysteresis(
        traces, YuanhangResistParams(),
        cases=[{"name": "known", "C_pF": 0.0, "C_th_pJ_per_K": .05, "S_e_mW_per_K": .004}],
        gammas=[.956269682], replay_steps_ns=[.5, .25], ambient_temperature_K=314.4,
    )
    assert np.allclose(result.measured_resistance_mean_ohm, 2000.0)
    assert np.allclose(result.temperature_mean_K, 314.4 + .005 / .004, atol=1e-5)
    assert np.all(result.late_invalid_resistance_fraction == 0)
    assert np.allclose(paths[paths.time_ns >= 150].power_mW, .005)


def test_reconstruction_bundle_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_toml(root / "experiments/current/specimen_hysteresis_reconstruction.toml")
    config["reconstruction"]["cases"] = [{"name": "central"}]
    config["reconstruction"]["gamma_values"] = [.956269682]
    config["reconstruction"]["replay_steps_ns"] = [.5]
    bundle = run_hysteresis_reconstruction(config, output_root=tmp_path)
    assert bundle.manifest["status"] == "completed"
    assert all((bundle.root / item["path"]).exists() for item in bundle.manifest["artifacts"])
    summary = pd.read_csv(bundle.root / "reconstruction.csv")
    assert summary.source_file.nunique() == 22
