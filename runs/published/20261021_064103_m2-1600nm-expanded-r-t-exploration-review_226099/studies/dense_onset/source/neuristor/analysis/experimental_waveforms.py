"""Load and summarize the professor-supplied TIA current-sweep waveforms.

The active laboratory workflow operates on numerical oscilloscope exports.  It
never recovers values from plot pixels.  The converted CSV files contain three
headerless columns: relative time, input current, and output voltage.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


_CONVERTED_NAME = re.compile(r"^(?P<drive_mv>\d+)mv0_converted\.csv$", re.IGNORECASE)
BASELINE_WINDOW_NS = (-200.0, -50.0)
PLATEAU_WINDOW_NS = (50.0, 250.0)
SLOPE_WINDOW_NS = (0.0, 30.0)


def _window(time_ns: np.ndarray, bounds_ns: tuple[float, float]) -> np.ndarray:
    return (time_ns >= bounds_ns[0]) & (time_ns <= bounds_ns[1])


def load_converted_trace(path: str | Path) -> pd.DataFrame:
    """Read one untouched ``*_converted.csv`` oscilloscope export.

    Units are encoded by the source workbook and paper: ns, uA, and mV.  The
    CSVs have no header, so assigning names here is part of the documented
    import boundary rather than an inference from a plotted image.
    """

    source = Path(path).expanduser().resolve()
    match = _CONVERTED_NAME.match(source.name)
    if match is None:
        raise ValueError(f"Unsupported converted-waveform filename: {source.name}")
    frame = pd.read_csv(
        source,
        header=None,
        names=["time_ns", "input_current_uA", "output_voltage_mV"],
        dtype=float,
    )
    if frame.empty or frame.shape[1] != 3:
        raise ValueError(f"Waveform is empty or malformed: {source}")
    values = frame[["time_ns", "input_current_uA", "output_voltage_mV"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Waveform contains non-finite values: {source}")
    if np.any(np.diff(frame["time_ns"].to_numpy(dtype=float)) <= 0.0):
        raise ValueError(f"Waveform time must increase strictly: {source}")
    frame.insert(0, "source_file", source.name)
    frame.insert(1, "nominal_drive_mV", float(match.group("drive_mv")))
    frame["output_power_uW"] = (
        frame["input_current_uA"] * frame["output_voltage_mV"] * 1e-3
    )
    return frame


def oscillation_metrics(time_ns: np.ndarray, voltage_mV: np.ndarray) -> dict[str, float | bool]:
    """Return a conservative periodic-peak estimate for the pulse plateau.

    This detector reproduces the visibly coherent Figure 7 operating window;
    it is a descriptive measurement, not a substitute for fitting the physical
    model.  The prominence floor prevents high-current measurement noise from
    being mislabeled as VO2 oscillation.
    """

    if time_ns.size < 9:
        return {
            "oscillation_detected": False,
            "oscillation_frequency_MHz": float("nan"),
            "oscillation_peak_count": 0.0,
            "oscillation_period_cv": float("nan"),
            "oscillation_peak_prominence_mV": float("nan"),
        }
    dt_ns = float(np.median(np.diff(time_ns)))
    smooth = savgol_filter(voltage_mV, 7, 2)
    peaks, properties = find_peaks(
        smooth,
        prominence=3.0,
        distance=max(int(round(8.0 / dt_ns)), 1),
    )
    intervals_ns = np.diff(time_ns[peaks])
    period_cv = (
        float(np.std(intervals_ns) / np.mean(intervals_ns))
        if intervals_ns.size >= 2 and float(np.mean(intervals_ns)) > 0.0
        else float("nan")
    )
    prominence_mV = (
        float(np.median(properties["prominences"])) if peaks.size else float("nan")
    )
    detected = bool(
        peaks.size >= 5
        and np.isfinite(period_cv)
        and period_cv <= 0.15
        and np.isfinite(prominence_mV)
        and prominence_mV >= 6.0
    )
    frequency_MHz = (
        1000.0 / float(np.median(intervals_ns)) if detected else float("nan")
    )
    return {
        "oscillation_detected": detected,
        "oscillation_frequency_MHz": frequency_MHz,
        "oscillation_peak_count": float(peaks.size),
        "oscillation_period_cv": period_cv,
        "oscillation_peak_prominence_mV": prominence_mV,
    }


def summarize_converted_trace(frame: pd.DataFrame) -> dict[str, float | str | bool]:
    """Summarize one raw trace using fixed, documented analysis windows."""

    time_ns = frame["time_ns"].to_numpy(dtype=float)
    current_uA = frame["input_current_uA"].to_numpy(dtype=float)
    voltage_mV = frame["output_voltage_mV"].to_numpy(dtype=float)
    power_uW = frame["output_power_uW"].to_numpy(dtype=float)
    baseline = _window(time_ns, BASELINE_WINDOW_NS)
    plateau = _window(time_ns, PLATEAU_WINDOW_NS)
    slope_window = _window(time_ns, SLOPE_WINDOW_NS)
    if np.sum(baseline) < 3 or np.sum(plateau) < 3 or np.sum(slope_window) < 3:
        raise ValueError("Waveform does not span the documented baseline, edge, and plateau windows")

    current_baseline_uA = float(np.median(current_uA[baseline]))
    current_plateau_uA = float(np.median(current_uA[plateau]))
    voltage_baseline_mV = float(np.median(voltage_mV[baseline]))
    voltage_plateau_mV = float(np.mean(voltage_mV[plateau]))
    voltage_slope = float(np.polyfit(time_ns[slope_window], voltage_mV[slope_window], 1)[0])
    oscillation = oscillation_metrics(time_ns[plateau], voltage_mV[plateau])
    return {
        "source_file": str(frame["source_file"].iloc[0]),
        "nominal_drive_mV": float(frame["nominal_drive_mV"].iloc[0]),
        "samples": float(len(frame)),
        "time_step_ns": float(np.median(np.diff(time_ns))),
        "current_baseline_uA": current_baseline_uA,
        "current_plateau_uA": current_plateau_uA,
        "current_step_uA": current_plateau_uA - current_baseline_uA,
        "voltage_baseline_mV": voltage_baseline_mV,
        "voltage_plateau_mean_mV": voltage_plateau_mV,
        "voltage_step_mV": voltage_plateau_mV - voltage_baseline_mV,
        "voltage_plateau_vpp_mV": float(np.ptp(voltage_mV[plateau])),
        "voltage_slope_0_30_mV_per_ns": voltage_slope,
        "maximum_output_power_uW": float(np.max(power_uW[plateau])),
        **oscillation,
    }


def load_converted_sweep(data_directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load every converted CSV in a directory and return traces plus summary."""

    directory = Path(data_directory).expanduser().resolve()
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob("*.csv"):
        match = _CONVERTED_NAME.match(path.name)
        if match is not None:
            candidates.append((int(match.group("drive_mv")), path))
    paths = [path for _, path in sorted(candidates)]
    if not paths:
        raise FileNotFoundError(f"No *_converted.csv waveforms found in {directory}")
    traces = [load_converted_trace(path) for path in paths]
    summary = pd.DataFrame([summarize_converted_trace(frame) for frame in traces])
    return pd.concat(traces, ignore_index=True), summary.sort_values("current_plateau_uA").reset_index(drop=True)


# Backward-compatible private alias for notebooks or historical imports.  New
# analysis code should use the public name so the laboratory and model traces
# are classified by exactly the same documented detector.
_oscillation_metrics = oscillation_metrics
