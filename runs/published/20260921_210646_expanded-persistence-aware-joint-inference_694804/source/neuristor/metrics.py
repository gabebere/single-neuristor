"""Shared numerical summaries used by CLI workflows and the dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np


def oscillation_metrics(
    time_s: np.ndarray,
    signal: np.ndarray,
    *,
    start_s: float | None = None,
    stop_s: float | None = None,
    minimum_amplitude: float = 1e-6,
) -> dict[str, float | bool]:
    """Measure sustained cycles using robust mid-quantile threshold crossings."""

    t = np.asarray(time_s, dtype=float).reshape(-1)
    values = np.asarray(signal, dtype=float).reshape(-1)
    mask = np.isfinite(t) & np.isfinite(values)
    if start_s is not None:
        mask &= t >= float(start_s)
    if stop_s is not None:
        mask &= t <= float(stop_s)
    t = t[mask]
    values = values[mask]
    if values.size < 8:
        return _empty_metrics()
    low, high = np.quantile(values, [0.1, 0.9])
    amplitude = float(high - low)
    if amplitude < minimum_amplitude:
        result = _empty_metrics()
        result.update({"minimum": float(np.min(values)), "maximum": float(np.max(values)), "peak_to_peak": amplitude})
        return result
    threshold = 0.5 * (low + high)
    crossings = np.flatnonzero((values[:-1] < threshold) & (values[1:] >= threshold)) + 1
    periods = np.diff(t[crossings]) if crossings.size >= 2 else np.asarray([], dtype=float)
    mean_period = float(np.mean(periods)) if periods.size else float("nan")
    period_cv = float(np.std(periods) / mean_period) if periods.size and mean_period > 0.0 else float("nan")
    frequency_MHz = 1e-6 / mean_period if np.isfinite(mean_period) and mean_period > 0.0 else 0.0
    return {
        "oscillatory": bool(periods.size >= 2 and np.isfinite(period_cv) and period_cv <= 0.2),
        "frequency_MHz": float(frequency_MHz),
        "period_us": float(mean_period * 1e6) if np.isfinite(mean_period) else float("nan"),
        "period_cv": period_cv,
        "cycles": float(periods.size),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "peak_to_peak": float(np.ptp(values)),
    }


def current_run_metrics(
    output: dict[str, np.ndarray],
    *,
    analysis_start_s: float | None = None,
    analysis_stop_s: float | None = None,
) -> dict[str, Any]:
    """Canonical metrics for one ideal-current simulation."""

    metrics = oscillation_metrics(
        output["t"],
        output["V_vo2"],
        start_s=analysis_start_s,
        stop_s=analysis_stop_s,
        minimum_amplitude=0.01,
    )
    mask = _window_mask(output["t"], analysis_start_s, analysis_stop_s)
    metrics.update(
        {
            "voltage_min_V": float(np.min(output["V_vo2"][mask])),
            "voltage_max_V": float(np.max(output["V_vo2"][mask])),
            "temperature_min_K": float(np.min(output["T"][mask])),
            "temperature_max_K": float(np.max(output["T"][mask])),
            "resistance_min_ohm": float(np.min(output["R"][mask])),
            "resistance_max_ohm": float(np.max(output["R"][mask])),
            "power_mean_uW": float(np.mean(output["P"][mask]) * 1e6),
            "power_max_uW": float(np.max(output["P"][mask]) * 1e6),
        }
    )
    return metrics


def voltage_run_metrics(
    time_s: np.ndarray,
    voltage_V: np.ndarray,
    current_A: np.ndarray,
    temperature_K: np.ndarray,
    resistance_ohm: np.ndarray,
    *,
    analysis_start_s: float | None = None,
    analysis_stop_s: float | None = None,
) -> dict[str, Any]:
    """Canonical metrics for one voltage-source simulation."""

    metrics = oscillation_metrics(
        time_s,
        current_A,
        start_s=analysis_start_s,
        stop_s=analysis_stop_s,
        minimum_amplitude=1e-6,
    )
    mask = _window_mask(time_s, analysis_start_s, analysis_stop_s)
    metrics.update(
        {
            "voltage_min_V": float(np.min(voltage_V[mask])),
            "voltage_max_V": float(np.max(voltage_V[mask])),
            "current_min_mA": float(np.min(current_A[mask]) * 1e3),
            "current_max_mA": float(np.max(current_A[mask]) * 1e3),
            "temperature_min_K": float(np.min(temperature_K[mask])),
            "temperature_max_K": float(np.max(temperature_K[mask])),
            "resistance_min_ohm": float(np.min(resistance_ohm[mask])),
            "resistance_max_ohm": float(np.max(resistance_ohm[mask])),
        }
    )
    return metrics


def _window_mask(time_s: np.ndarray, start_s: float | None, stop_s: float | None) -> np.ndarray:
    time = np.asarray(time_s, dtype=float)
    mask = np.isfinite(time)
    if start_s is not None:
        mask &= time >= float(start_s)
    if stop_s is not None:
        mask &= time <= float(stop_s)
    if not np.any(mask):
        raise ValueError("Analysis window contains no samples")
    return mask


def _empty_metrics() -> dict[str, float | bool]:
    return {
        "oscillatory": False,
        "frequency_MHz": 0.0,
        "period_us": float("nan"),
        "period_cv": float("nan"),
        "cycles": 0.0,
        "minimum": float("nan"),
        "maximum": float("nan"),
        "peak_to_peak": 0.0,
    }
