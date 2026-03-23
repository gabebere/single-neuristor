from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class PlotBounds:
    x_left_axis: int
    x_right_axis: int
    y_top_axis: int
    y_bottom_axis: int

    @property
    def x_data_start(self) -> int:
        return self.x_left_axis + 1

    @property
    def x_data_stop(self) -> int:
        return self.x_right_axis - 1

    @property
    def y_data_start(self) -> int:
        return self.y_top_axis + 1

    @property
    def y_data_stop(self) -> int:
        return self.y_bottom_axis - 1


@dataclass(frozen=True)
class DigitizedTrace:
    frame_index: int
    image_path: Path
    time_ns: np.ndarray
    v_out_mV: np.ndarray
    i_in_visible_uA: np.ndarray
    green_y_pixels: np.ndarray
    purple_y_pixels: np.ndarray


def _as_int_rgb(image_path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.int16)


def detect_plot_bounds(image_path: str | Path) -> PlotBounds:
    img = _as_int_rgb(image_path)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    green_axis = (g > 200) & (g - np.maximum(r, b) > 120)
    purple_axis = (r > 100) & (b > 100) & (np.minimum(r, b) - g > 80)
    dark = img.mean(axis=2) < 200

    x_left = int(np.argmax(green_axis.sum(axis=0)))
    x_right = int(np.argmax(purple_axis.sum(axis=0)))

    row_counts = dark.sum(axis=1)
    mid = row_counts.size // 2
    y_top = int(np.argmax(row_counts[:mid]))
    y_bottom = int(mid + np.argmax(row_counts[mid:]))
    return PlotBounds(
        x_left_axis=x_left,
        x_right_axis=x_right,
        y_top_axis=y_top,
        y_bottom_axis=y_bottom,
    )


def _trace_from_score(
    crop_rgb: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    y = np.full(mask.shape[1], np.nan, dtype=float)
    for x_idx in range(mask.shape[1]):
        yy = np.where(mask[:, x_idx])[0]
        if yy.size == 0:
            continue
        weights = score[yy, x_idx].astype(float)
        y[x_idx] = float(np.average(yy.astype(float), weights=weights))
    finite = np.where(np.isfinite(y))[0]
    if finite.size < 2:
        raise RuntimeError("Could not extract enough trace pixels from image.")
    y = np.interp(np.arange(y.size), finite, y[finite])
    return y


def digitize_frame(
    image_path: str | Path,
    *,
    frame_index: int,
    bounds: PlotBounds,
    time_min_ns: float = -200.0,
    time_max_ns: float = 600.0,
    y_min_uA: float = -400.0,
    y_max_uA: float = 800.0,
    y_min_mV: float = -400.0,
    y_max_mV: float = 800.0,
) -> DigitizedTrace:
    img = _as_int_rgb(image_path)
    crop = img[
        bounds.y_data_start : bounds.y_data_stop + 1,
        bounds.x_data_start : bounds.x_data_stop + 1,
        :,
    ]
    r = crop[:, :, 0]
    g = crop[:, :, 1]
    b = crop[:, :, 2]

    green_score = g - np.maximum(r, b)
    green_mask = (g > 80) & (green_score > 25)
    purple_score = np.minimum(r, b) - g
    purple_mask = (r > 70) & (b > 70) & (purple_score > 25)

    green_y = _trace_from_score(crop, green_score, green_mask)
    purple_y = _trace_from_score(crop, purple_score, purple_mask)

    x = np.arange(crop.shape[1], dtype=float)
    time_ns = time_min_ns + (x / max(crop.shape[1] - 1, 1)) * (time_max_ns - time_min_ns)

    def y_to_units(y_pixels: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
        y_abs = bounds.y_data_start + y_pixels
        frac = (y_abs - bounds.y_top_axis) / max(bounds.y_bottom_axis - bounds.y_top_axis, 1)
        return y_max - frac * (y_max - y_min)

    return DigitizedTrace(
        frame_index=frame_index,
        image_path=Path(image_path),
        time_ns=time_ns,
        v_out_mV=y_to_units(purple_y, y_min_mV, y_max_mV),
        i_in_visible_uA=y_to_units(green_y, y_min_uA, y_max_uA),
        green_y_pixels=green_y,
        purple_y_pixels=purple_y,
    )


def smooth_trace(values: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size < window:
        return arr.copy()
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def reconstruct_current_waveform(
    trace: DigitizedTrace,
    inferred_current_uA: float,
    *,
    baseline_window_ns: tuple[float, float] = (-180.0, -20.0),
    plateau_window_ns: tuple[float, float] = (80.0, 250.0),
) -> np.ndarray:
    """
    Reconstruct an input-current waveform from the digitized green trace.

    The images clip the green plateau at the top of the plot for high-current frames.
    To preserve edge timing while restoring the intended amplitude, subtract the
    visible baseline, normalize by the visible plateau level, then rescale to the
    inferred frame current.
    """

    t = np.asarray(trace.time_ns, dtype=float)
    i_visible = np.asarray(trace.i_in_visible_uA, dtype=float)
    baseline_mask = (t >= baseline_window_ns[0]) & (t <= baseline_window_ns[1])
    plateau_mask = (t >= plateau_window_ns[0]) & (t <= plateau_window_ns[1])

    baseline_uA = float(np.median(i_visible[baseline_mask])) if np.any(baseline_mask) else 0.0
    centered = i_visible - baseline_uA
    plateau_visible_uA = float(np.median(centered[plateau_mask])) if np.any(plateau_mask) else float("nan")
    plateau_visible_uA = max(plateau_visible_uA, 1e-6)

    normalized = smooth_trace(centered / plateau_visible_uA, window=5)
    normalized = np.clip(normalized, 0.0, 1.05)
    return inferred_current_uA * normalized


def count_turns(values: np.ndarray) -> int:
    arr = smooth_trace(values, window=5)
    d = np.diff(arr)
    if d.size < 2:
        return 0
    return int(np.sum((d[:-1] * d[1:]) < 0.0))


def summarize_trace(trace: DigitizedTrace) -> dict[str, float]:
    t = trace.time_ns
    v = trace.v_out_mV
    i = trace.i_in_visible_uA
    plateau_current = (t >= 80.0) & (t <= 250.0)
    plateau_voltage = (t >= 100.0) & (t <= 250.0)
    onset = (t >= 0.0) & (t <= 30.0)
    peak_window = (t >= 0.0) & (t <= 80.0)
    turnoff_window = (t >= 300.0) & (t <= 450.0)

    plateau_y = trace.green_y_pixels[plateau_current]
    clipped_frac = float(np.mean(plateau_y < 5.0)) if plateau_y.size else 0.0
    onset_slope = float(np.polyfit(t[onset], v[onset], 1)[0]) if np.sum(onset) >= 3 else float("nan")

    return {
        "frame_index": float(trace.frame_index),
        "plateau_current_visible_uA": float(np.mean(i[plateau_current])) if np.any(plateau_current) else float("nan"),
        "plateau_current_clipped_fraction": clipped_frac,
        "v_plateau_mean_mV": float(np.mean(v[plateau_voltage])) if np.any(plateau_voltage) else float("nan"),
        "v_plateau_vpp_mV": float(np.ptp(v[plateau_voltage])) if np.any(plateau_voltage) else float("nan"),
        "v_peak_0_80_mV": float(np.max(v[peak_window])) if np.any(peak_window) else float("nan"),
        "v_turnoff_min_300_450_mV": float(np.min(v[turnoff_window])) if np.any(turnoff_window) else float("nan"),
        "v_slope_0_30_mV_per_ns": onset_slope,
        "v_turn_count_80_250": float(count_turns(v[plateau_current])) if np.any(plateau_current) else 0.0,
    }


def infer_current_sweep(summary_df: pd.DataFrame, *, max_fit_frame: int = 18) -> tuple[float, float]:
    fit_df = summary_df[
        (summary_df["frame_index"] <= float(max_fit_frame))
        & (summary_df["plateau_current_clipped_fraction"] < 0.05)
        & np.isfinite(summary_df["plateau_current_visible_uA"])
    ]
    if len(fit_df) < 3:
        raise RuntimeError("Not enough unclipped green traces to infer current sweep.")
    slope, intercept = np.polyfit(
        fit_df["frame_index"].to_numpy(dtype=float),
        fit_df["plateau_current_visible_uA"].to_numpy(dtype=float),
        1,
    )
    return float(intercept), float(slope)


def digitize_directory(image_dir: str | Path) -> tuple[PlotBounds, list[DigitizedTrace], pd.DataFrame]:
    paths = sorted(Path(image_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {image_dir}")
    bounds = detect_plot_bounds(paths[0])
    traces = [digitize_frame(path, frame_index=idx, bounds=bounds) for idx, path in enumerate(paths)]
    summary_df = pd.DataFrame([summarize_trace(trace) for trace in traces]).sort_values("frame_index").reset_index(drop=True)
    intercept, slope = infer_current_sweep(summary_df)
    summary_df["current_inferred_uA"] = intercept + slope * summary_df["frame_index"].to_numpy(dtype=float)
    return bounds, traces, summary_df


def traces_to_dataframe(traces: Iterable[DigitizedTrace], summary_df: pd.DataFrame) -> pd.DataFrame:
    current_map = summary_df.set_index("frame_index")["current_inferred_uA"].to_dict()
    rows: list[dict[str, float | str]] = []
    for trace in traces:
        current = float(current_map[float(trace.frame_index)])
        i_reconstructed = reconstruct_current_waveform(trace, current)
        for t_ns, v_mV, i_uA, i_recon_uA in zip(trace.time_ns, trace.v_out_mV, trace.i_in_visible_uA, i_reconstructed):
            rows.append(
                {
                    "frame_index": trace.frame_index,
                    "image_path": str(trace.image_path),
                    "current_inferred_uA": current,
                    "time_ns": float(t_ns),
                    "v_out_mV": float(v_mV),
                    "i_in_visible_uA": float(i_uA),
                    "i_in_reconstructed_uA": float(i_recon_uA),
                }
            )
    return pd.DataFrame(rows)
