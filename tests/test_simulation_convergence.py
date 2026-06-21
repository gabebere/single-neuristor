from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import neuristor.model as model
from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step, simulate_current_steps
from neuristor.current_domain_search import analyze_current_trace
from neuristor.model import (
    YuanhangCircuitParams,
    YuanhangResistParams,
    detect_spike_times,
    series_first,
    simulate_yuanhang,
)


class SimulationConvergenceTests(unittest.TestCase):
    def setUp(self) -> None:
        model._TORCH_HYSTERESIS_AVAILABLE = False

    def test_quasistatic_current_control_settles_across_timesteps(self) -> None:
        payload = json.loads((ROOT / "presets" / "resistance_100425_chip1_gap3.json").read_text())
        resist = YuanhangResistParams(**payload["resist_params"])
        for dt_ns in (0.05, 0.025):
            params = CurrentDriveParams(
                dt_s=dt_ns * 1e-9,
                t_end_s=2_000e-9,
                T0_K=325.0,
                T_init_K=325.0,
                C_F=80e-12,
                C_th_J_per_K=3e-12,
                S_e_W_per_K=0.052838e-3,
                phase_mode="quasistatic",
                resist_params=resist,
                start_branch="insulator",
            )
            out = simulate_current_step(1100.0, params=params, seed=0)
            late = out["t"] >= 0.75 * out["t"][-1]
            self.assertLess(float(np.ptp(out["V_vo2"][late]) * 1e3), 1.0)

    def test_vectorized_current_sweep_matches_serial_traces(self) -> None:
        payload = json.loads((ROOT / "presets" / "resistance_100425_chip1_gap3.json").read_text())
        params = CurrentDriveParams(
            dt_s=0.1e-9,
            t_end_s=500e-9,
            pulse_on_s=150e-9,
            T0_K=311.21937437938016,
            T_init_K=311.21937437938016,
            C_F=25.930953e-12,
            C_th_J_per_K=5e-12,
            S_e_W_per_K=0.1e-3,
            resist_params=YuanhangResistParams(**payload["resist_params"]),
        )
        currents = [1200.0, 1640.0, 2400.0]
        previous_backend = model._TORCH_HYSTERESIS_AVAILABLE
        model._TORCH_HYSTERESIS_AVAILABLE = model.torch is not None
        try:
            serial = [simulate_current_step(current, params=params, seed=0) for current in currents]
            batched = simulate_current_steps(currents, params=params, seed=0)
        finally:
            model._TORCH_HYSTERESIS_AVAILABLE = previous_backend
        for serial_trace, batched_trace in zip(serial, batched):
            for key in ("I_in", "V_vo2", "T", "R", "g_eq", "P"):
                np.testing.assert_array_equal(serial_trace[key], batched_trace[key])

    def test_voltage_oscillator_frequency_converges(self) -> None:
        resist = YuanhangResistParams()
        circuit = YuanhangCircuitParams(
            R_series_kohm=12.0,
            C_par_pF=145.34619293,
            Cth_mW_ns_per_K=49.62776831,
            Sth_mW_per_K=0.20558726,
            T_base_K=325.0,
        )
        frequencies = []
        for dt_ns in (10.0, 5.0):
            out = simulate_yuanhang(
                Vin=14.5,
                t_end=60e-6,
                dt=dt_ns * 1e-9,
                resist_params=resist,
                circuit_params=circuit,
                start_branch="insulator",
            )
            t = np.asarray(out["time_s"], dtype=float)
            current = np.asarray(series_first(out["I_vo2"]), dtype=float)
            late = t >= 20e-6
            spikes = detect_spike_times(t[late].tolist(), current[late].tolist(), threshold_A=1e-3)
            self.assertGreaterEqual(len(spikes), 10)
            frequencies.append(1.0 / float(np.mean(np.diff(spikes))))
        relative_difference = abs(frequencies[0] - frequencies[1]) / frequencies[1]
        self.assertLess(relative_difference, 0.02)

    def test_quasistatic_relaxation_oscillator_converges(self) -> None:
        payload = json.loads((ROOT / "presets" / "resistance_100425_chip1_gap3.json").read_text())
        resist = YuanhangResistParams(**payload["resist_params"])
        frequencies = []
        for dt_ns in (0.01582, 0.00791):
            params = CurrentDriveParams(
                dt_s=dt_ns * 1e-9,
                t_end_s=1_500e-9,
                T0_K=311.21937437938016,
                T_init_K=311.21937437938016,
                C_F=25.930953e-12,
                C_th_J_per_K=5e-12,
                S_e_W_per_K=0.1e-3,
                phase_mode="quasistatic",
                resist_params=resist,
                start_branch="insulator",
            )
            out = simulate_current_step(1640.226226, params=params, seed=0)
            late = out["t"] >= 0.5 * out["t"][-1]
            late_out = {
                key: value[late]
                for key, value in out.items()
                if isinstance(value, np.ndarray) and value.ndim == 1 and value.shape[0] == late.shape[0]
            }
            metrics = analyze_current_trace(
                late_out,
                params=params,
                min_vpp_mV=20.0,
                max_vpp_mV=5_000.0,
                min_cycles=4,
                pulse_on_ns=float(late_out["t"][0] * 1e9),
            )
            self.assertEqual(metrics["oscillatory"], 1.0)
            frequencies.append(metrics["dominant_freq_MHz"])
        self.assertLess(abs(frequencies[0] - frequencies[1]) / frequencies[1], 0.02)


if __name__ == "__main__":
    unittest.main()
