"""Window-by-window waveform evidence and controlled parameter maps.

This diagnostic intentionally does not change the historical fit objective or the
physics. A peak-count label can accept damped ringing; here late periodic power,
amplitude retention, and voltage levels are reported independently. Thresholds are
operational checks, not proof of an asymptotic limit cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import detrend, find_peaks, periodogram, savgol_filter

from .current_drive_sim import CurrentDriveParams, simulate_current_waveforms
from .experimental_waveforms import oscillation_metrics
from .parameter_inference import InferenceDataset


@dataclass(frozen=True)
class AuditSettings:
    """Explicit, reusable windows and detection thresholds in laboratory units."""

    window_edges_ns: tuple[float, ...] = (50.0, 100.0, 150.0, 200.0, 250.0)
    frequency_band_MHz: tuple[float, float] = (10.0, 200.0)
    minimum_periodic_vpp_mV: float = 6.0
    minimum_retention: float = 0.5
    minimum_coherence: float = 0.4


def _harmonic_fit(time_ns: np.ndarray, voltage_mV: np.ndarray, frequency_MHz: float) -> tuple[float, float]:
    """Fit a fundamental plus linear baseline, returning Vpp and variance fraction.

    Baseline slope is fitted rather than mistaken for oscillation amplitude. The
    explained fraction refers only to the detrended signal, and is descriptive for
    nonsinusoidal relaxation cycles, whose higher harmonics are not included.
    """

    t = np.asarray(time_ns, dtype=float)
    y = np.asarray(voltage_mV, dtype=float)
    x = (t - np.mean(t)) / max(float(np.ptp(t)), 1.0)
    phase = 2 * np.pi * frequency_MHz * t * 1e-3
    design = np.column_stack((np.sin(phase), np.cos(phase), np.ones_like(t), x))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    variance = float(np.sum(detrend(y) ** 2))
    coherence = max(0.0, 1.0 - float(np.sum(residual ** 2)) / variance) if variance > 1e-16 else 0.0
    return float(2 * np.hypot(*coefficients[:2])), coherence


def dominant_frequency(time_ns: np.ndarray, voltage_mV: np.ndarray, band_MHz: tuple[float, float]) -> float:
    """Find the late dominant spectral frequency without the old 8 ns peak spacing.

    A Hann spectrum selects the strongest component in the documented band. A
    sinusoidal regression refines it locally; zero padding only interpolates the
    spectrum and does not improve the finite record's physical resolution. This
    reports the dominant component, which can differ from a nonsinusoidal cycle's
    fundamental. An independent peak-period estimate is therefore also retained.
    """

    t, y = np.asarray(time_ns, float), np.asarray(voltage_mV, float)
    dt = float(np.median(np.diff(t)))
    low, high = band_MHz
    high = min(high, 0.45 * 1000.0 / dt)
    if low >= high:
        raise ValueError("Frequency band must lie below the sampled Nyquist frequency")
    if np.ptp(y) < 1e-8:
        return float("nan")
    frequencies, power = periodogram(detrend(y), fs=1000.0 / dt, window="hann", nfft=8192)
    selected = (frequencies >= low) & (frequencies <= high)
    peak = float(frequencies[selected][np.argmax(power[selected])])
    half_bin = 500.0 / (t[-1] - t[0] + dt)
    result = minimize_scalar(
        lambda f: -_harmonic_fit(t, y, float(f))[1],
        bounds=(max(low, peak - half_bin), min(high, peak + half_bin)), method="bounded",
    )
    return float(result.x)


def audit_voltage(
    time_ns: np.ndarray, voltage_mV: np.ndarray, settings: AuditSettings = AuditSettings(),
) -> tuple[dict[str, float | bool], pd.DataFrame]:
    """Measure four windows, late periodicity and amplitude decay on equal samples."""

    time, voltage = np.asarray(time_ns, float), np.asarray(voltage_mV, float)
    edges = np.asarray(settings.window_edges_ns, float)
    if (time.ndim != 1 or voltage.shape != time.shape or len(time) < 16
            or not np.all(np.isfinite(time)) or not np.all(np.isfinite(voltage))
            or np.any(np.diff(time) <= 0) or len(edges) != 5 or np.any(np.diff(edges) <= 0)):
        raise ValueError("Audit requires finite equal-length ordered samples and five increasing window edges")
    if time[0] > edges[0] or time[-1] < edges[-1]:
        raise ValueError("Trace does not cover the complete audit window")
    dt = np.diff(time)
    if not np.allclose(dt, np.median(dt), rtol=1e-5, atol=1e-6):
        raise ValueError("Spectral audit requires uniformly sampled data")
    plateau = (time >= edges[0]) & (time <= edges[-1])
    late = (time >= edges[2]) & (time <= edges[-1])
    frequency = dominant_frequency(time[late], voltage[late], settings.frequency_band_MHz)
    rows = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (time >= lo) & (time < hi)
        if mask.sum() < 8:
            raise ValueError("Each audit window needs at least eight samples")
        v = voltage[mask]
        periodic_vpp, coherence = _harmonic_fit(time[mask], v, frequency) if np.isfinite(frequency) else (0.0, 0.0)
        rows.append({
            "window": index, "start_ns": lo, "stop_ns": hi,
            "mean_mV": float(np.mean(v)), "vpp_mV": float(np.ptp(v)),
            "robust_vpp_mV": float(np.quantile(v, .95) - np.quantile(v, .05)),
            "periodic_vpp_mV": periodic_vpp, "coherence": coherence,
        })
    windows = pd.DataFrame(rows)
    periodic_vpp, coherence = _harmonic_fit(time[late], voltage[late], frequency) if np.isfinite(frequency) else (0.0, 0.0)
    # Retain raw Vpp beside the robust envelope so noise and isolated extremes are visible.
    retention = float((windows.robust_vpp_mV.iloc[-1] + 1e-9) / (windows.robust_vpp_mV.iloc[0] + 1e-9))
    minimum_vpp = float(windows.periodic_vpp_mV.min())
    smooth = savgol_filter(voltage[late], 5, 2)
    peaks, _ = find_peaks(smooth, prominence=max(2.0, .15 * float(np.ptp(smooth))), distance=2)
    periods = np.diff(time[late][peaks])
    peak_frequency = float(1000.0 / np.median(periods)) if len(periods) >= 2 else float("nan")
    period_cv = float(np.std(periods) / np.mean(periods)) if len(periods) >= 2 else float("nan")
    legacy = oscillation_metrics(time[plateau], voltage[plateau])
    summary = {
        "legacy_detected": bool(legacy["oscillation_detected"]),
        "legacy_frequency_MHz": float(legacy["oscillation_frequency_MHz"]),
        "late_frequency_MHz": frequency, "late_peak_frequency_MHz": peak_frequency,
        "late_peak_period_cv": period_cv,
        "late_mean_mV": float(np.mean(voltage[late])),
        "late_vpp_mV": float(np.ptp(voltage[late])),
        "late_robust_vpp_mV": float(np.quantile(voltage[late], .95) - np.quantile(voltage[late], .05)),
        "late_periodic_vpp_mV": periodic_vpp, "late_coherence": coherence,
        "minimum_window_periodic_vpp_mV": minimum_vpp,
        "amplitude_retention": retention,
        "sustained": bool(minimum_vpp >= settings.minimum_periodic_vpp_mV
                          and retention >= settings.minimum_retention
                          and coherence >= settings.minimum_coherence),
    }
    return summary, windows


def compare_features(measured: dict, predicted: dict) -> dict[str, float]:
    """Continuous errors with comparable, explicit engineering tolerance scales.

    The four terms use a 20% amplitude ratio, 10% frequency error, 20 mV mean
    voltage error and factor-two envelope-retention error. No classification reward
    is included. These scales rank exploratory candidates; they are not likelihoods
    or experimental confidence intervals. Each trace and each term get equal weight.
    """

    amplitude = float(np.log((predicted["late_robust_vpp_mV"] + 3) / (measured["late_robust_vpp_mV"] + 3)))
    frequency_valid = predicted["late_periodic_vpp_mV"] >= 3.0 and predicted["late_coherence"] >= .2
    f_error = ((predicted["late_frequency_MHz"] / measured["late_frequency_MHz"] - 1)
               if frequency_valid and np.isfinite(predicted["late_frequency_MHz"]) else 1.0)
    mean_error = float(predicted["late_mean_mV"] - measured["late_mean_mV"])
    retention_error = float(np.log(max(predicted["amplitude_retention"], 1e-4)
                                   / max(measured["amplitude_retention"], 1e-4)))
    terms = [(amplitude / np.log(1.2)) ** 2, (f_error / .1) ** 2,
             (mean_error / 20.0) ** 2, (retention_error / np.log(2.0)) ** 2]
    return {
        "amplitude_ratio": float(predicted["late_robust_vpp_mV"] / max(measured["late_robust_vpp_mV"], 1e-9)),
        "frequency_relative_error": float(f_error), "frequency_comparable": bool(frequency_valid),
        "mean_error_mV": mean_error, "retention_log_error": retention_error,
        "feature_score": float(np.mean(terms)),
    }


def sample_predictions(dataset: InferenceDataset, params: CurrentDriveParams, indices: np.ndarray) -> np.ndarray:
    """Call the authoritative solver, preserving the full recorded pre-pulse input.

    Only the unused post-analysis tail is omitted for speed. Predictions are sampled
    at the original oscilloscope times; smaller integration steps do not imply more
    experimental bandwidth.
    """

    local = replace(params, t_pre_s=max(0, -dataset.time_ns[0] * 1e-9),
                    t_end_s=max(0, dataset.time_ns[-1] * 1e-9))
    outputs = simulate_current_waveforms(dataset.current_uA[:, indices], local,
                                        waveform_time_s=dataset.time_ns * 1e-9)
    return np.column_stack([np.interp(dataset.time_ns * 1e-9, out["t"].astype(float),
                                     out["V_vo2"].astype(float)) * 1e3 for out in outputs])


def evaluate_candidate(
    dataset: InferenceDataset, params: CurrentDriveParams, indices: np.ndarray,
    settings: AuditSettings, measured: Sequence[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return separate feature errors, per-window evidence, and sampled voltages."""

    predictions = sample_predictions(dataset, params, indices)
    rows, windows = [], []
    for column, index in enumerate(indices):
        summary, window = audit_voltage(dataset.time_ns, predictions[:, column], settings)
        meta = {"source_setting_mV": dataset.nominal_drives_mV[index],
                "current_uA": dataset.measured_summary.iloc[index].current_step_uA}
        rows.append({**meta, **summary, **compare_features(measured[column], summary)})
        windows.append(window.assign(**meta))
    return pd.DataFrame(rows), pd.concat(windows, ignore_index=True), predictions


def parameter_map(
    dataset: InferenceDataset, base: CurrentDriveParams, indices: np.ndarray, *,
    capacitances_pF: Sequence[float], thermal_times_ns: Sequence[float], gammas: Sequence[float],
    settings: AuditSettings = AuditSettings(), progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map C and Cth/Se at a few gamma values with every other parameter fixed."""

    measured = [audit_voltage(dataset.time_ns, dataset.voltage_mV[:, i], settings)[0] for i in indices]
    rows, windows = [], []
    total = len(capacitances_pF) * len(thermal_times_ns) * len(gammas)
    candidate_id = 0
    for gamma in gammas:
        for tau in thermal_times_ns:
            for capacitance in capacitances_pF:
                params = replace(base, C_F=capacitance * 1e-12,
                                 C_th_J_per_K=base.S_e_W_per_K * tau * 1e-9,
                                 resist_params=replace(base.resist_params, gamma=gamma))
                meta = {"candidate_id": candidate_id, "C_pF": capacitance, "tau_th_ns": tau,
                        "C_th_pJ_per_K": params.C_th_J_per_K * 1e12, "gamma": gamma}
                summary, window, _ = evaluate_candidate(dataset, params, indices, settings, measured)
                rows.append(summary.assign(**meta))
                windows.append(window.assign(**meta))
                candidate_id += 1
                if progress:
                    progress(candidate_id, total)
    return pd.concat(rows, ignore_index=True), pd.concat(windows, ignore_index=True)
