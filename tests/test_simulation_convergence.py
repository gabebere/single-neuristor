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
from neuristor.current_drive_sim import (
    CurrentDriveParams,
    current_drive_operating_estimates,
    sanitize_current_drive_params,
    simulate_current_step,
    simulate_current_steps,
    simulate_current_waveform,
    simulate_current_waveforms,
)
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

    def test_yuanhang_current_control_settles_across_timesteps(self) -> None:
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
            np.testing.assert_array_equal(batched_trace["g_eq"], batched_trace["g_dyn"])

    def test_vectorized_measured_waveforms_match_serial_traces(self) -> None:
        times = np.arange(-20.0, 81.0, 1.0) * 1e-9
        currents = np.column_stack(
            [
                np.where(times < 0.0, 0.0, 120.0),
                np.where(times < 0.0, 0.0, 240.0),
            ]
        )
        params = CurrentDriveParams(
            dt_s=0.5e-9,
            t_pre_s=20e-9,
            t_end_s=80e-9,
            C_F=0.39e-12,
            C_th_J_per_K=0.047873236e-12,
            S_e_W_per_K=0.0036751265e-3,
            T0_K=314.4,
            T_init_K=314.4,
        )
        serial = [
            simulate_current_waveform(currents[:, index], params, waveform_time_s=times)
            for index in range(currents.shape[1])
        ]
        batched = simulate_current_waveforms(currents, params, waveform_time_s=times)
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

    def test_yuanhang_current_relaxation_oscillator_converges(self) -> None:
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

    def test_legacy_current_mode_fields_are_ignored_before_instantiation(self) -> None:
        payload = {
            "dt_s": 0.1e-9,
            "t_end_s": 10e-9,
            "thermal_mode": "double",
            "C_sub_J_per_K": 1e-12,
            "G_hot_sub_W_per_K": 1e-3,
            "T_sub_init_K": 325.0,
            "phase_mode": "dynamic",
            "tau_g_s": 10e-9,
            "domain_count": 8,
            "domain_temperature_span_K": 5.0,
            "domain_coupling_W_per_K": 1e-4,
            "hysteresis_reversal_mode": "turning_point",
            "resist_params": YuanhangResistParams(),
        }
        cleaned = sanitize_current_drive_params(payload)
        params = CurrentDriveParams(**cleaned)
        out = simulate_current_step(100.0, params=params, seed=0)
        self.assertTrue(np.all(np.isfinite(out["V_vo2"])))
        np.testing.assert_array_equal(out["g_eq"], out["g_dyn"])

    def test_zero_capacitance_is_the_algebraic_voltage_limit(self) -> None:
        params = CurrentDriveParams(
            C_F=0.0,
            dt_s=0.1e-9,
            t_end_s=20e-9,
            T0_K=325.0,
            T_init_K=325.0,
        )
        out = simulate_current_step(400.0, params=params, seed=0)
        np.testing.assert_allclose(out["V_vo2"], out["I_in"] * out["R"], rtol=2e-6, atol=1e-7)
        self.assertTrue(np.all(out["V_vo2"] >= 0.0))

    def test_small_capacitance_update_remains_bounded_when_dt_exceeds_rc(self) -> None:
        params = CurrentDriveParams(
            C_F=0.01e-12,
            dt_s=1.0e-9,
            t_end_s=100e-9,
            T0_K=325.0,
            T_init_K=325.0,
        )
        out = simulate_current_step(400.0, params=params, seed=0)
        target = out["I_in"] * out["R"]
        self.assertTrue(np.all(np.isfinite(out["V_vo2"])))
        self.assertTrue(np.all(out["V_vo2"] >= 0.0))
        self.assertLessEqual(float(np.max(out["V_vo2"])), 1.01 * float(np.max(target)))

    def test_operating_estimate_exposes_voltage_floor_and_thermal_only_gap(self) -> None:
        params = CurrentDriveParams()
        report = current_drive_operating_estimates(params, I_uA=400.0)
        self.assertAlmostEqual(report["V_metal_floor_V"], 400e-6 * params.resist_params.Rm, places=7)
        self.assertGreater(
            report["thermal_only_lower_current_uA"],
            report["thermal_only_upper_current_uA"],
        )
        self.assertEqual(report["thermal_only_window_exists"], 0.0)


if __name__ == "__main__":
    unittest.main()
