from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuristor.oscillation_audit import audit_voltage, compare_features


@pytest.mark.parametrize("frequency", [32.0, 50.0, 125.0, 160.0])
def test_late_frequency_has_no_legacy_125_MHz_ceiling(frequency):
    time = np.arange(0.0, 251.0)
    voltage = 180 + 20 * np.sin(2 * np.pi * frequency * time / 1000)
    metrics, _ = audit_voltage(time, voltage)
    assert metrics["sustained"]
    assert abs(metrics["late_frequency_MHz"] - frequency) < .1
    assert abs(metrics["late_periodic_vpp_mV"] - 40) < .1


def test_damped_ringdown_is_not_sustained_even_with_many_regular_peaks():
    time = np.arange(0.0, 251.0)
    voltage = 180 + 30 * np.exp(-(time - 50) / 40) * np.sin(2 * np.pi * time / 16)
    metrics, windows = audit_voltage(time, voltage)
    assert metrics["legacy_detected"]
    assert not metrics["sustained"]
    assert windows.periodic_vpp_mV.iloc[-1] < 2
    assert metrics["amplitude_retention"] < .1


def test_baseline_slope_and_noise_do_not_count_as_periodic_motion():
    time = np.arange(0.0, 251.0)
    voltage = 180 + .1 * time + np.random.default_rng(19).normal(0, 1, len(time))
    metrics, _ = audit_voltage(time, voltage)
    assert not metrics["sustained"]
    assert metrics["late_periodic_vpp_mV"] < 3


def test_constant_trace_and_incomplete_record():
    time = np.arange(0.0, 251.0)
    metrics, _ = audit_voltage(time, np.ones_like(time))
    assert not metrics["sustained"]
    assert np.isnan(metrics["late_frequency_MHz"])
    with pytest.raises(ValueError, match="complete"):
        audit_voltage(time[:-1], np.ones_like(time[:-1]))


def test_feature_score_prefers_matched_waveform_to_wrong_amplitude_or_decay():
    time = np.arange(0.0, 251.0)
    carrier = np.sin(2 * np.pi * time / 20)
    measured, _ = audit_voltage(time, 180 + 20 * carrier)
    oversized, _ = audit_voltage(time, 180 + 100 * carrier)
    damped, _ = audit_voltage(time, 180 + 20 * np.exp(-(time - 50) / 40) * carrier)
    assert compare_features(measured, measured)["feature_score"] < 1e-12
    assert compare_features(measured, oversized)["feature_score"] > 1
    assert compare_features(measured, damped)["feature_score"] > 1


def test_archived_upper_boundary_is_a_detector_false_success():
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(root / "public_jobs/20260907_125315_physics-anchored-specimen-waveform-inference_717797/fitted_traces.csv")
    frame = frame[(frame.fit_mode == "relaxed") & (frame.nominal_drive_mV == 800)]
    measured, _ = audit_voltage(frame.time_ns.to_numpy(), frame.measured_voltage_mV.to_numpy())
    predicted, _ = audit_voltage(frame.time_ns.to_numpy(), frame.predicted_voltage_mV.to_numpy())
    assert measured["sustained"]
    assert predicted["legacy_detected"]
    assert not predicted["sustained"]
    assert predicted["amplitude_retention"] < .02
