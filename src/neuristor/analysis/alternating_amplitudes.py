"""Primary-cycle measurements for the fixed alternating-reset experiment.

These descriptive measurements preserve the published individual-peak detector.
They do not establish period doubling or change the current-source model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


def cycle_table(time_ns, voltage_mV, window, *, prominence_fraction=0.10, smoothing_ns=0.5):
    """Use individual peak-minus-preceding-trough amplitudes, without neighbor averaging.

    Smoothing locates primary peaks only. A local raw maximum refines each peak;
    raw minima between primary peaks supply the preceding trough. Prominence
    excludes small shoulders on the reset, not just quantization/noise. Edge
    fragments are omitted. These are descriptive signal features, not a new
    physical oscillation or statistical significance criterion.
    """
    t, v = np.asarray(time_ns, float), np.asarray(voltage_mV, float)
    if (
        t.shape != v.shape
        or t.ndim != 1
        or not np.isfinite(t).all()
        or not np.isfinite(v).all()
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("Matching finite, increasing waveform arrays required")
    mask = (t >= window[0]) & (t <= window[1])
    t, v = t[mask], v[mask]
    if len(t) < 9:
        return pd.DataFrame()
    dt = np.median(np.diff(t))
    width = max(5, int(round(smoothing_ns / dt)) | 1)
    width = min(width, len(v) if len(v) % 2 else len(v) - 1)
    smooth = savgol_filter(v, width, 2) if smoothing_ns > 0 else v
    peaks, _ = find_peaks(
        smooth, prominence=max(2.0, prominence_fraction * np.ptp(smooth)), distance=max(1, round(2.0 / dt))
    )
    radius = max(0, round(0.5 / dt))
    peaks = np.unique(
        [max(0, p - radius) + np.argmax(v[max(0, p - radius) : min(len(v), p + radius + 1)]) for p in peaks]
    )
    rows = []
    for n, (left, right) in enumerate(zip(peaks[:-1], peaks[1:])):
        bottom = left + np.argmin(v[left : right + 1])
        if bottom in (left, right) or v[right] <= v[bottom]:
            continue
        rows.append(
            dict(
                cycle_index=len(rows),
                peak_time_ns=t[right],
                preceding_peak_time_ns=t[left],
                trough_time_ns=t[bottom],
                peak_mV=v[right],
                trough_mV=v[bottom],
                vpp_mV=v[right] - v[bottom],
                period_ns=t[right] - t[left],
            )
        )
    return pd.DataFrame(rows)


def summarize_cycles(cycles):
    """Separate two-cycle alternation from broad amplitude variation or drift."""
    if len(cycles) < 6:
        return dict(cycle_count=len(cycles), strong_alternation=False)
    a = cycles.vpp_mV.to_numpy()
    medians = np.array([np.median(a[::2]), np.median(a[1::2])])
    sep = float(np.ptp(medians) / np.mean(medians))
    mad = max(float(np.median(np.abs(a[k::2] - medians[k])) / medians[k]) for k in [0, 1])
    lag1 = float(np.corrcoef(a[:-1], a[1:])[0, 1]) if np.std(a) > 1e-6 else 0.0
    lag2 = float(np.corrcoef(a[:-2], a[2:])[0, 1]) if np.std(a) > 1e-6 else 0.0
    strong = bool(len(a) >= 8 and sep >= 0.10 and mad <= 0.05 and lag1 <= -0.8 and lag2 >= 0.8)
    times = cycles.peak_time_ns.to_numpy()
    return dict(
        cycle_count=len(a),
        phase0_median_vpp_mV=medians[0],
        phase1_median_vpp_mV=medians[1],
        low_phase_median_vpp_mV=float(min(medians)),
        high_phase_median_vpp_mV=float(max(medians)),
        phase_separation_fraction=sep,
        maximum_phase_relative_MAD=mad,
        amplitude_lag1=lag1,
        amplitude_lag2=lag2,
        strong_alternation=strong,
        median_adjacent_peak_period_ns=float(np.median(cycles.period_ns)),
        median_two_peak_period_ns=float(np.median(times[2:] - times[:-2])),
        adjacent_peak_rate_MHz=float(1000 / np.median(cycles.period_ns)),
        two_peak_repeat_rate_MHz=float(1000 / np.median(times[2:] - times[:-2])),
    )
