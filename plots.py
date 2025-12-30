

"""plots.py

Post-processing / analysis utilities for the single-neuristor VO₂ simulator.

Goal: Implement Amir's requested analyses without modifying the simulator core.
We import the simulator and helpers from `model.py` and only do analysis here.

Implements:
1) Baselines (cycle minima) of T and V(on device) vs time.
2) V_max (peak device voltage) vs Vin.
3) Power P(t)=V*I and power peaks (min/max) vs Vin.
4) Capacitance sweep: overlay power for several C values.
5) 3D sweep: frequency as a function of (C, R_load).
6) Resistance in the insulating state vs time.

NOTE: The “heater with a second heat equation” requires editing the simulator ODEs
in `model.py`; we can add that once the formulation/parameters are agreed.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import replace
from typing import Callable, Dict, Iterable, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Import simulator + helpers from model.py
from model import (
    SimOut,
    YuanhangCircuitParams,
    YuanhangResistParams,
    detect_spike_times,
    is_oscillatory,
    series_first,
    simulate_yuanhang,
)


# -----------------------------
# Utility helpers
# -----------------------------

def _window_mask(time_s: List[float], t_start_us: float, t_end_us: float) -> np.ndarray:
    t = np.asarray(time_s, dtype=float)
    t_us = t * 1e6
    return (t_us >= t_start_us) & (t_us <= t_end_us)


def _local_extrema_indices(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_idx, max_idx) for 1D array using a simple discrete test."""
    if y.size < 3:
        return np.array([], dtype=int), np.array([], dtype=int)
    y0 = y[:-2]
    y1 = y[1:-1]
    y2 = y[2:]
    max_mask = (y0 < y1) & (y1 >= y2)
    min_mask = (y0 > y1) & (y1 <= y2)
    max_idx = np.where(max_mask)[0] + 1
    min_idx = np.where(min_mask)[0] + 1
    return min_idx, max_idx


# -----------------------------
# New helper: cycle extrema stats and capacitance sweep extrema plot
# -----------------------------

def _cycle_extrema_stats(y: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (mean_max, std_max, mean_min, std_min) of local extrema across cycles.

    Uses the same discrete local-extrema detector as the baseline plots.
    If too few extrema are detected, fall back to global extrema.
    """
    min_idx, max_idx = _local_extrema_indices(y)

    # Require at least ~2 extrema to compute a meaningful distribution
    if max_idx.size < 2 or min_idx.size < 2:
        return float(np.max(y)), 0.0, float(np.min(y)), 0.0

    ymax = y[max_idx]
    ymin = y[min_idx]
    return float(np.mean(ymax)), float(np.std(ymax)), float(np.mean(ymin)), float(np.std(ymin))


# -----------------------------
# Core time-trace plot (single run)
# -----------------------------

def plot_time_traces(
    data: SimOut,
    R_series_kohm: float,
    title: str = "Neuristor time traces",
) -> None:
    """Plot V_vo2 & V_load, T_vo2, I_vo2, and P_vo2 vs time.

    V_vo2 is the device voltage; V_load = I_load * R_series.
    Assumes single-device output (uses first device if lattice).
    """
    t_us = np.asarray(data["time_s"], dtype=float) * 1e6
    V_vo2 = np.asarray(series_first(data["V_node"]), dtype=float)
    I_load = np.asarray(series_first(data["I_load"]), dtype=float)
    I_vo2 = np.asarray(series_first(data["I_vo2"]), dtype=float)
    T_vo2 = np.asarray(series_first(data["T_K"]), dtype=float)
    V_load = I_load * (float(R_series_kohm) * 1e3)
    P_vo2 = V_vo2 * I_vo2

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(t_us, V_vo2, label="V_vo2 (device)")
    axes[0].plot(t_us, V_load, label="V_load", alpha=0.8)
    axes[0].set_ylabel("Voltage (V)")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    axes[1].plot(t_us, T_vo2, color="tab:red")
    axes[1].set_ylabel("T_vo2 (K)")
    axes[1].grid(True)

    axes[2].plot(t_us, I_vo2, color="tab:green")
    axes[2].set_ylabel("I_vo2 (A)")
    axes[2].grid(True)

    axes[3].plot(t_us, P_vo2, color="tab:purple")
    axes[3].set_ylabel("P_vo2 (W)")
    axes[3].set_xlabel("time (µs)")
    axes[3].grid(True)

    fig.suptitle(title)
    fig.tight_layout()


def compute_sweep_metrics(
    data: SimOut,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
    threshold_A: float = 1e-3,
) -> Dict[str, float]:
    """Compute sweep metrics in a steady-state window for a single SimOut."""
    t = np.asarray(data["time_s"], dtype=float)
    V = np.asarray(series_first(data["V_node"]), dtype=float)
    I = np.asarray(series_first(data["I_vo2"]), dtype=float)
    T = np.asarray(series_first(data["T_K"]), dtype=float)
    P = V * I

    m = _window_mask(data["time_s"], t_start_us, t_end_us)
    if np.any(m):
        t = t[m]
        V = V[m]
        I = I[m]
        T = T[m]
        P = P[m]

    spike_times = detect_spike_times(t.tolist(), I.tolist(), threshold_A=threshold_A)
    if len(spike_times) >= 2:
        isi_us = np.diff(np.asarray(spike_times)) * 1e6
        isi_mean_us = float(np.mean(isi_us)) if isi_us.size else float("nan")
        freq_mhz = 1.0 / isi_mean_us if isi_mean_us > 0 else float("nan")
    else:
        isi_mean_us = float("nan")
        freq_mhz = float("nan")

    return {
        "Vmax": float(np.max(V)),
        "Pmax": float(np.max(P)),
        "Pmin": float(np.min(P)),
        "Tmax": float(np.max(T)),
        "Tmin": float(np.min(T)),
        "ISI_mean_us": isi_mean_us,
        "freq_MHz": float(freq_mhz),
        "oscillatory": bool(is_oscillatory(data, t_start_us=t_start_us, t_end_us=t_end_us, threshold_A=threshold_A)),
    }


def _emit_progress(message: str, progress_cb: Callable[[str], None] | None) -> None:
    if progress_cb is not None:
        progress_cb(message)
    else:
        print(message)


def _param_value(
    param_name: str,
    Vin: float,
    resist_params: YuanhangResistParams,
    circuit_params: YuanhangCircuitParams,
) -> float:
    if param_name == "Vin":
        return float(Vin)
    if hasattr(resist_params, param_name):
        return float(getattr(resist_params, param_name))
    if hasattr(circuit_params, param_name):
        return float(getattr(circuit_params, param_name))
    raise ValueError(f"Unknown parameter: {param_name}")


def sweep_free_variable(
    param_name: str,
    values: List[float],
    Vin: float,
    t_end: float,
    dt: float,
    resist_params: YuanhangResistParams,
    circuit_params: YuanhangCircuitParams,
    start_branch: str = "insulator",
    lattice_shape: Tuple[int, int] = (1, 1),
    noise_seed: int | None = None,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
    threshold_A: float = 1e-3,
    progress_cb: Callable[[str], None] | None = None,
    log_prefix: str = "sweep",
    log_every: int = 1,
) -> Dict[str, List[float]]:
    """Sweep a single parameter and return metrics arrays vs the free variable."""
    resist_fields = {f.name for f in dataclasses.fields(YuanhangResistParams)}
    circuit_fields = {f.name for f in dataclasses.fields(YuanhangCircuitParams)}
    if log_every < 1:
        raise ValueError("log_every must be >= 1")

    results = {
        "values": [],
        "Vmax": [],
        "Pmax": [],
        "Pmin": [],
        "Tmax": [],
        "Tmin": [],
        "ISI_mean_us": [],
        "freq_MHz": [],
        "oscillatory": [],
    }

    total = len(values)
    _emit_progress(f"[{log_prefix}] Running {total} points for {param_name}", progress_cb)
    for idx, val in enumerate(values):
        sim = _simulate_with_param(
            param_name,
            float(val),
            Vin,
            t_end,
            dt,
            resist_params,
            circuit_params,
            start_branch,
            lattice_shape,
            None if noise_seed is None else noise_seed + idx,
            resist_fields,
            circuit_fields,
        )
        metrics = compute_sweep_metrics(sim, t_start_us=t_start_us, t_end_us=t_end_us, threshold_A=threshold_A)
        results["values"].append(val)
        for key in ("Vmax", "Pmax", "Pmin", "Tmax", "Tmin", "ISI_mean_us", "freq_MHz"):
            results[key].append(float(metrics[key]))
        results["oscillatory"].append(bool(metrics["oscillatory"]))
        if (idx + 1) % log_every == 0 or (idx + 1) == total:
            _emit_progress(
                f"[{log_prefix}] {idx+1}/{total}: {param_name}={val} → osc={metrics['oscillatory']}, f={metrics['freq_MHz']:.3g} MHz",
                progress_cb,
            )
    return results


def _simulate_with_param(
    param_name: str,
    value: float,
    Vin: float,
    t_end: float,
    dt: float,
    resist_params: YuanhangResistParams,
    circuit_params: YuanhangCircuitParams,
    start_branch: str,
    lattice_shape: Tuple[int, int],
    noise_seed: int | None,
    resist_fields: set[str],
    circuit_fields: set[str],
) -> SimOut:
    run_vin = Vin
    resist = replace(resist_params)
    circuit = replace(circuit_params)
    if param_name == "Vin":
        run_vin = value
    elif param_name in resist_fields:
        setattr(resist, param_name, float(value))
    elif param_name in circuit_fields:
        setattr(circuit, param_name, float(value))
    else:
        raise ValueError(f"Unknown sweep parameter: {param_name}")
    return simulate_yuanhang(
        Vin=run_vin,
        t_end=t_end,
        dt=dt,
        resist_params=resist,
        circuit_params=circuit,
        start_branch=start_branch,
        lattice_shape=lattice_shape,
        noise_seed=noise_seed,
    )


def _simulate_with_params(
    param_x: str,
    value_x: float,
    param_y: str,
    value_y: float,
    Vin: float,
    t_end: float,
    dt: float,
    resist_params: YuanhangResistParams,
    circuit_params: YuanhangCircuitParams,
    start_branch: str,
    lattice_shape: Tuple[int, int],
    noise_seed: int | None,
    resist_fields: set[str],
    circuit_fields: set[str],
) -> SimOut:
    if param_x == param_y:
        raise ValueError("param_x and param_y must be different")
    run_vin = Vin
    resist = replace(resist_params)
    circuit = replace(circuit_params)
    for param_name, value in ((param_x, value_x), (param_y, value_y)):
        if param_name == "Vin":
            run_vin = float(value)
        elif param_name in resist_fields:
            setattr(resist, param_name, float(value))
        elif param_name in circuit_fields:
            setattr(circuit, param_name, float(value))
        else:
            raise ValueError(f"Unknown sweep parameter: {param_name}")
    return simulate_yuanhang(
        Vin=run_vin,
        t_end=t_end,
        dt=dt,
        resist_params=resist,
        circuit_params=circuit,
        start_branch=start_branch,
        lattice_shape=lattice_shape,
        noise_seed=noise_seed,
    )


def sweep_free_variable_coarse_fine(
    param_name: str,
    start: float,
    stop: float,
    coarse_step: float,
    fine_step: float,
    Vin: float,
    t_end: float,
    dt: float,
    resist_params: YuanhangResistParams,
    circuit_params: YuanhangCircuitParams,
    start_branch: str = "insulator",
    lattice_shape: Tuple[int, int] = (1, 1),
    noise_seed: int | None = None,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
    threshold_A: float = 1e-3,
    progress_cb: Callable[[str], None] | None = None,
) -> Dict[str, object]:
    """Coarse scan to find oscillatory band, then fine sweep to compute metrics."""
    if coarse_step <= 0 or fine_step <= 0:
        raise ValueError("coarse_step and fine_step must be > 0")

    coarse_vals = np.arange(start, stop + 0.5 * coarse_step, coarse_step, dtype=float).tolist()
    coarse_osc = []
    resist_fields = {f.name for f in dataclasses.fields(YuanhangResistParams)}
    circuit_fields = {f.name for f in dataclasses.fields(YuanhangCircuitParams)}

    total = len(coarse_vals)
    _emit_progress(f"[coarse] Running {total} points for {param_name}", progress_cb)
    for idx, val in enumerate(coarse_vals):
        sim = _simulate_with_param(
            param_name,
            float(val),
            Vin,
            t_end,
            dt,
            resist_params,
            circuit_params,
            start_branch,
            lattice_shape,
            None if noise_seed is None else noise_seed + idx,
            resist_fields,
            circuit_fields,
        )
        osc = is_oscillatory(sim, t_start_us=t_start_us, t_end_us=t_end_us, threshold_A=threshold_A)
        coarse_osc.append(bool(osc))
        _emit_progress(f"[coarse] {idx+1}/{total}: {param_name}={val} → osc={osc}", progress_cb)

    if not any(coarse_osc):
        _emit_progress("[coarse] No oscillatory region found.", progress_cb)
        return {
            "coarse_values": coarse_vals,
            "coarse_oscillatory": coarse_osc,
            "band_min": None,
            "band_max": None,
            "fine_results": None,
        }

    osc_vals = [v for v, osc in zip(coarse_vals, coarse_osc) if osc]
    band_min = float(min(osc_vals))
    band_max = float(max(osc_vals))
    _emit_progress(f"[coarse] Detected band: {band_min} – {band_max}", progress_cb)
    fine_vals = np.arange(band_min, band_max + 0.5 * fine_step, fine_step, dtype=float).tolist()

    fine_results = sweep_free_variable(
        param_name,
        fine_vals,
        Vin,
        t_end,
        dt,
        resist_params,
        circuit_params,
        start_branch=start_branch,
        lattice_shape=lattice_shape,
        noise_seed=noise_seed,
        t_start_us=t_start_us,
        t_end_us=t_end_us,
        threshold_A=threshold_A,
        progress_cb=progress_cb,
        log_prefix="fine",
    )

    return {
        "coarse_values": coarse_vals,
        "coarse_oscillatory": coarse_osc,
        "band_min": band_min,
        "band_max": band_max,
        "fine_results": fine_results,
    }


def frequency_from_spikes(
    data: SimOut,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
    threshold_A: float = 1e-3,
    min_spikes: int = 4,
) -> float:
    """Compute frequency from spikes; return NaN if not oscillatory."""
    time_s = data["time_s"]
    I_vo2 = series_first(data["I_vo2"])
    if not time_s or not I_vo2:
        return float("nan")
    t_arr = np.asarray(time_s, dtype=float)
    I_arr = np.asarray(I_vo2, dtype=float)
    t_us = t_arr * 1e6
    mask = (t_us >= t_start_us) & (t_us <= t_end_us)
    if not np.any(mask):
        return float("nan")
    spike_times = detect_spike_times(
        t_arr[mask].tolist(),
        I_arr[mask].tolist(),
        threshold_A=threshold_A,
    )
    if len(spike_times) < min_spikes:
        return float("nan")
    isi_us = np.diff(np.asarray(spike_times)) * 1e6
    if isi_us.size == 0:
        return float("nan")
    mean_isi = float(np.mean(isi_us))
    return 1.0 / mean_isi if mean_isi > 0.0 else float("nan")


def _resolve_bounds_1d(
    param_name: str,
    start: float | None,
    stop: float | None,
    step: float,
    Vin: float,
    t_end: float,
    dt: float,
    resist_params: YuanhangResistParams,
    circuit_params: YuanhangCircuitParams,
    start_branch: str,
    lattice_shape: Tuple[int, int],
    noise_seed: int | None,
    t_start_us: float,
    t_end_us: float,
    threshold_A: float,
    max_coarse_steps: int,
    progress_cb: Callable[[str], None] | None,
    label: str,
) -> Tuple[float, float, List[float], List[bool]]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if start is None:
        start = _param_value(param_name, Vin, resist_params, circuit_params)
    if stop is not None:
        values = np.arange(start, stop + 0.5 * step, step, dtype=float).tolist()
        return float(start), float(stop), values, []

    values = [float(start + i * step) for i in range(max_coarse_steps)]
    resist_fields = {f.name for f in dataclasses.fields(YuanhangResistParams)}
    circuit_fields = {f.name for f in dataclasses.fields(YuanhangCircuitParams)}
    osc_flags: List[bool] = []
    _emit_progress(f"[coarse-{label}] scanning {max_coarse_steps} steps for {param_name}", progress_cb)
    for idx, val in enumerate(values):
        sim = _simulate_with_param(
            param_name,
            float(val),
            Vin,
            t_end,
            dt,
            resist_params,
            circuit_params,
            start_branch,
            lattice_shape,
            None if noise_seed is None else noise_seed + idx,
            resist_fields,
            circuit_fields,
        )
        osc = is_oscillatory(sim, t_start_us=t_start_us, t_end_us=t_end_us, threshold_A=threshold_A)
        osc_flags.append(bool(osc))
        _emit_progress(f"[coarse-{label}] {idx+1}/{max_coarse_steps}: {param_name}={val} → osc={osc}", progress_cb)
    osc_vals = [v for v, osc in zip(values, osc_flags) if osc]
    if not osc_vals:
        _emit_progress(f"[coarse-{label}] no oscillations found; using full coarse range", progress_cb)
        return float(values[0]), float(values[-1]), values, osc_flags
    band_min = float(min(osc_vals))
    band_max = float(max(osc_vals))
    if band_max == values[-1]:
        _emit_progress(f"[coarse-{label}] oscillatory through limit; using full coarse range", progress_cb)
        return float(values[0]), float(values[-1]), values, osc_flags
    _emit_progress(f"[coarse-{label}] band detected {band_min} – {band_max}", progress_cb)
    return band_min, band_max, values, osc_flags


def sweep_frequency_2d(
    param_x: str,
    param_y: str,
    x_start: float | None,
    x_stop: float | None,
    x_step: float,
    y_start: float | None,
    y_stop: float | None,
    y_step: float,
    Vin: float,
    t_end: float,
    dt: float,
    resist_params: YuanhangResistParams,
    circuit_params: YuanhangCircuitParams,
    start_branch: str = "insulator",
    lattice_shape: Tuple[int, int] = (1, 1),
    noise_seed: int | None = None,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
    threshold_A: float = 1e-3,
    min_spikes: int = 4,
    max_coarse_steps: int = 100,
    progress_cb: Callable[[str], None] | None = None,
    row_early_stop: bool = True,
    col_early_stop: bool = True,
) -> Dict[str, object]:
    """Sweep two parameters and return frequency grid (NaN for non-oscillatory points).

    If row_early_stop is True, for each fixed X, Y scanning stops once oscillations
    cease after at least one oscillatory point (assumes monotonicity in Y per X).
    If col_early_stop is True, for each fixed Y, X scanning stops once oscillations
    cease after at least one oscillatory point (assumes monotonicity in X per Y).
    """
    if x_step <= 0 or y_step <= 0:
        raise ValueError("x_step and y_step must be > 0")
    if param_x == param_y:
        raise ValueError("param_x and param_y must be different")

    x_min, x_max, x_coarse, x_osc = _resolve_bounds_1d(
        param_x,
        x_start,
        x_stop,
        x_step,
        Vin,
        t_end,
        dt,
        resist_params,
        circuit_params,
        start_branch,
        lattice_shape,
        noise_seed,
        t_start_us,
        t_end_us,
        threshold_A,
        max_coarse_steps,
        progress_cb,
        "x",
    )
    y_min, y_max, y_coarse, y_osc = _resolve_bounds_1d(
        param_y,
        y_start,
        y_stop,
        y_step,
        Vin,
        t_end,
        dt,
        resist_params,
        circuit_params,
        start_branch,
        lattice_shape,
        noise_seed,
        t_start_us,
        t_end_us,
        threshold_A,
        max_coarse_steps,
        progress_cb,
        "y",
    )

    x_values = np.arange(x_min, x_max + 0.5 * x_step, x_step, dtype=float).tolist()
    y_values = np.arange(y_min, y_max + 0.5 * y_step, y_step, dtype=float).tolist()
    freq_grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)

    resist_fields = {f.name for f in dataclasses.fields(YuanhangResistParams)}
    circuit_fields = {f.name for f in dataclasses.fields(YuanhangCircuitParams)}
    seen_osc_y = [False] * len(y_values)
    stop_y = [False] * len(y_values)
    total = len(x_values) * len(y_values)
    _emit_progress(f"[2d] sweeping {len(x_values)}×{len(y_values)} = {total} points", progress_cb)
    count = 0
    for xi, x_val in enumerate(x_values):
        seen_osc = False
        for yi, y_val in enumerate(y_values):
            if col_early_stop and stop_y[yi]:
                continue
            sim = _simulate_with_params(
                param_x,
                float(x_val),
                param_y,
                float(y_val),
                Vin,
                t_end,
                dt,
                resist_params,
                circuit_params,
                start_branch,
                lattice_shape,
                None if noise_seed is None else noise_seed + count,
                resist_fields,
                circuit_fields,
            )
            freq = frequency_from_spikes(
                sim,
                t_start_us=t_start_us,
                t_end_us=t_end_us,
                threshold_A=threshold_A,
                min_spikes=min_spikes,
            )
            freq_grid[yi, xi] = float(freq)
            osc = bool(np.isfinite(freq))
            if osc:
                seen_osc = True
                seen_osc_y[yi] = True
            elif row_early_stop and seen_osc:
                _emit_progress(
                    f"[2d] early-stop row at {param_x}={x_val}: {param_y}={y_val} non-oscillatory",
                    progress_cb,
                )
                break
            elif col_early_stop and seen_osc_y[yi]:
                stop_y[yi] = True
                _emit_progress(
                    f"[2d] early-stop column at {param_y}={y_val}: {param_x}={x_val} non-oscillatory",
                    progress_cb,
                )
            count += 1
            _emit_progress(
                f"[2d] {count}/{total}: {param_x}={x_val}, {param_y}={y_val} → {'osc' if osc else 'non-osc'}",
                progress_cb,
            )

    return {
        "x_values": x_values,
        "y_values": y_values,
        "freq_MHz": freq_grid,
        "x_coarse_values": x_coarse,
        "x_coarse_oscillatory": x_osc,
        "y_coarse_values": y_coarse,
        "y_coarse_oscillatory": y_osc,
    }


def plot_frequency_2d(
    sweep_results: Dict[str, object],
    x_label: str,
    y_label: str,
    title_prefix: str | None = None,
    log_scale: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap_name: str = "viridis",
    nan_color: str = "lightgray",
) -> None:
    """Plot 3D scatter and 2D heatmap of frequency vs two parameters."""
    x_vals = np.asarray(sweep_results["x_values"], dtype=float)
    y_vals = np.asarray(sweep_results["y_values"], dtype=float)
    freq = np.asarray(sweep_results["freq_MHz"], dtype=float)
    X, Y = np.meshgrid(x_vals, y_vals)
    mask = np.isfinite(freq)

    finite = freq[mask]
    if finite.size == 0:
        finite_min = 0.0
        finite_max = 1.0
    else:
        finite_min = float(np.min(finite))
        finite_max = float(np.max(finite))
    if vmin is None:
        vmin = finite_min
    if vmax is None:
        vmax = finite_max
    if log_scale and (vmin <= 0.0 or not np.isfinite(vmin)):
        log_scale = False

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=nan_color)
    if log_scale:
        from matplotlib.colors import LogNorm

        norm = LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
    else:
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=vmin, vmax=vmax)

    fig3d = plt.figure(figsize=(7.5, 5.5))
    ax3d = fig3d.add_subplot(111, projection="3d")
    sc = ax3d.scatter(X[mask], Y[mask], freq[mask], c=freq[mask], cmap=cmap, norm=norm, s=12)
    ax3d.set_xlabel(x_label)
    ax3d.set_ylabel(y_label)
    ax3d.set_zlabel("Frequency (MHz)")
    ax3d.set_title(f"{title_prefix} — Frequency (3D)" if title_prefix else "Frequency (3D)")
    fig3d.colorbar(sc, ax=ax3d, pad=0.1, shrink=0.7, label="Frequency (MHz)")

    def _edges_from_centers(values: np.ndarray) -> np.ndarray:
        if values.size == 1:
            return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)
        edges = np.zeros(values.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
        edges[0] = values[0] - 0.5 * (values[1] - values[0])
        edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
        return edges

    fig2d, ax2d = plt.subplots(figsize=(6.5, 4.5))
    masked = np.ma.array(freq, mask=~np.isfinite(freq))
    x_edges = _edges_from_centers(x_vals)
    y_edges = _edges_from_centers(y_vals)
    img = ax2d.pcolormesh(
        x_edges,
        y_edges,
        masked,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax2d.set_xlabel(x_label)
    ax2d.set_ylabel(y_label)
    ax2d.set_title(f"{title_prefix} — Frequency (heatmap)" if title_prefix else "Frequency (heatmap)")
    fig2d.colorbar(img, ax=ax2d, label="Frequency (MHz)")


def plot_sweep_metrics(
    sweep_results: Dict[str, List[float]],
    free_label: str = "Free variable",
    title_prefix: str | None = None,
) -> None:
    """Plot key metrics vs the swept variable, each on its own figure."""
    values = np.asarray(sweep_results["values"], dtype=float)

    def _title(metric: str) -> str:
        return f"{title_prefix} — {metric}" if title_prefix else metric

    fig_v, ax_v = plt.subplots(figsize=(6.5, 4.0))
    ax_v.plot(values, sweep_results["Vmax"], "o-")
    ax_v.set_ylabel("Vmax (V)")
    ax_v.set_xlabel(free_label)
    ax_v.set_title(_title("Vmax vs free variable"))
    ax_v.grid(True)

    fig_p, ax_p = plt.subplots(figsize=(6.5, 4.0))
    ax_p.plot(values, sweep_results["Pmax"], "o-", label="Pmax")
    ax_p.plot(values, sweep_results["Pmin"], "o-", label="Pmin")
    ax_p.set_ylabel("Power (W)")
    ax_p.set_xlabel(free_label)
    ax_p.set_title(_title("Pmax/Pmin vs free variable"))
    ax_p.legend(loc="best")
    ax_p.grid(True)

    fig_t, ax_t = plt.subplots(figsize=(6.5, 4.0))
    ax_t.plot(values, sweep_results["Tmax"], "o-", label="Tmax")
    ax_t.plot(values, sweep_results["Tmin"], "o-", label="Tmin")
    ax_t.set_ylabel("Temperature (K)")
    ax_t.set_xlabel(free_label)
    ax_t.set_title(_title("Tmax/Tmin vs free variable"))
    ax_t.legend(loc="best")
    ax_t.grid(True)

    fig_f, ax_f = plt.subplots(figsize=(6.5, 4.0))
    ax_f.plot(values, sweep_results["freq_MHz"], "o-")
    ax_f.set_ylabel("Frequency (MHz)")
    ax_f.set_xlabel(free_label)
    ax_f.set_title(_title("Oscillation frequency vs free variable"))
    ax_f.grid(True)

    fig_isi, ax_isi = plt.subplots(figsize=(6.5, 4.0))
    ax_isi.plot(values, sweep_results["ISI_mean_us"], "o-")
    ax_isi.set_ylabel("Mean ISI (us)")
    ax_isi.set_xlabel(free_label)
    ax_isi.set_title(_title("Mean ISI vs free variable"))
    ax_isi.grid(True)


def plot_capacitance_sweep_power_extrema(
    vin: float,
    C_start_pF: float,
    C_stop_pF: float,
    C_step_pF: float,
    resist_params: YuanhangResistParams,
    circuit_params: YuanhangCircuitParams,
    t_end: float,
    dt: float,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
    C_values_pF: Optional[List[float]] = None,
) -> None:
    """Sweep capacitance and plot cycle-averaged P_max and P_min vs C.

    For each capacitance value:
      1) simulate at fixed Vin
      2) compute P(t)=V*I in the steady-state window
      3) detect local maxima/minima of P(t)
      4) take mean/std across cycles
    """
    if C_step_pF <= 0:
        raise ValueError("C_step_pF must be > 0")

    if C_values_pF is None:
        C_vals = np.arange(C_start_pF, C_stop_pF + 0.5 * C_step_pF, C_step_pF, dtype=float)
    else:
        C_vals = np.asarray(C_values_pF, dtype=float)

    mean_pmax: List[float] = []
    std_pmax: List[float] = []
    mean_pmin: List[float] = []
    std_pmin: List[float] = []

    total = len(C_vals)
    for idx, C in enumerate(C_vals, start=1):
        print(f"[C_sweep] Simulating C = {C:.1f} pF ({idx}/{total})")
        cp = replace(circuit_params, C_par_pF=float(C))
        d = simulate_yuanhang(
            Vin=vin,
            t_end=t_end,
            dt=dt,
            resist_params=resist_params,
            circuit_params=cp,
        )

        P = power_trace_W(d)
        m = _window_mask(d["time_s"], t_start_us, t_end_us)
        Pw = P[m] if np.any(m) else P

        mu_max, sig_max, mu_min, sig_min = _cycle_extrema_stats(Pw)
        mean_pmax.append(mu_max)
        std_pmax.append(sig_max)
        mean_pmin.append(mu_min)
        std_pmin.append(sig_min)

    print("[C_sweep] Completed capacitance sweep.")

    plt.figure(figsize=(9, 4.5))
    plt.errorbar(C_vals, mean_pmax, yerr=std_pmax, fmt="o-", capsize=3, label="⟨P_max⟩ per cycle")
    plt.errorbar(C_vals, mean_pmin, yerr=std_pmin, fmt="o-", capsize=3, label="⟨P_min⟩ per cycle")
    plt.xlabel("C_par (pF)")
    plt.ylabel("Power (W)")
    plt.title(f"Cycle-averaged power extrema vs capacitance (Vin={vin:.2f} V)")
    plt.grid(True)
    plt.legend(loc="best")


def _compute_isi_us(
    data: SimOut,
    threshold_A: float = 1e-3,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
) -> np.ndarray:
    t = np.asarray(data["time_s"], dtype=float)
    I = np.asarray(series_first(data["I_vo2"]), dtype=float)
    m = _window_mask(data["time_s"], t_start_us, t_end_us)
    if not np.any(m):
        return np.array([])
    spike_times = detect_spike_times(t[m].tolist(), I[m].tolist(), threshold_A=threshold_A)
    if len(spike_times) < 2:
        return np.array([])
    return np.diff(np.asarray(spike_times)) * 1e6


# -----------------------------
# Oscillation/auto-Cdomain helpers
# -----------------------------

def _has_oscillations(
    data: SimOut,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
    threshold_A: float = 1e-3,
    min_spikes: int = 4,
) -> bool:
    """True if at least `min_spikes` current spikes exist in the window.

    Mirrors model.py logic (spike detection uses |I| via detect_spike_times).
    min_spikes=4 corresponds to requiring ~3 consecutive cycles.
    """
    return is_oscillatory(
        data,
        t_start_us=t_start_us,
        t_end_us=t_end_us,
        threshold_A=threshold_A,
        min_spikes=min_spikes,
    )


def _auto_find_oscillatory_C_band(
    vin: float,
    C_start_pF: float,
    C_stop_pF: float,
    C_step_coarse_pF: float,
    resist_params: YuanhangResistParams,
    circuit_params: YuanhangCircuitParams,
    t_end: float,
    dt: float,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
    threshold_A: float = 1e-3,
    min_spikes: int = 4,
) -> Tuple[Optional[float], Optional[float]]:
    """Coarsely scan capacitance and return (C_min_osc, C_max_osc)."""
    if C_step_coarse_pF <= 0:
        raise ValueError("C_step_coarse_pF must be > 0")

    C_vals = np.arange(
        C_start_pF,
        C_stop_pF + 0.5 * C_step_coarse_pF,
        C_step_coarse_pF,
        dtype=float,
    )

    print(
        f"[auto_Cdomain] Coarse sweep: C from {C_start_pF:.1f} to {C_stop_pF:.1f} pF "
        f"in steps of {C_step_coarse_pF:.1f} pF ({len(C_vals)} points)"
    )

    C_min_osc: Optional[float] = None
    C_max_osc: Optional[float] = None

    for idx, C in enumerate(C_vals, start=1):
        print(f"[auto_Cdomain]   Testing C={C:.1f} pF ({idx}/{len(C_vals)})")
        cp = replace(circuit_params, C_par_pF=float(C))
        d = simulate_yuanhang(
            Vin=vin,
            t_end=t_end,
            dt=dt,
            resist_params=resist_params,
            circuit_params=cp,
        )

        if _has_oscillations(
            d,
            t_start_us=t_start_us,
            t_end_us=t_end_us,
            threshold_A=threshold_A,
            min_spikes=min_spikes,
        ):
            print(f"[auto_Cdomain]     C={C:.1f} pF -> oscillatory")
            if C_min_osc is None:
                C_min_osc = float(C)
            C_max_osc = float(C)
        elif C_min_osc is not None:
            print(f"[auto_Cdomain]     C={C:.1f} pF -> non-oscillatory (leaving band; stopping)")
            break
        else:
            print(f"[auto_Cdomain]     C={C:.1f} pF -> non-oscillatory")

    return C_min_osc, C_max_osc


def mean_frequency_mhz(
    data: SimOut,
    threshold_A: float = 1e-3,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
) -> float:
    isi = _compute_isi_us(data, threshold_A=threshold_A, t_start_us=t_start_us, t_end_us=t_end_us)
    if isi.size == 0:
        return float("nan")
    mean_isi = float(np.mean(isi))
    if mean_isi <= 0:
        return float("nan")
    return 1.0 / mean_isi  # MHz because ISI is in µs


def power_trace_W(data: SimOut) -> np.ndarray:
    """Instantaneous device power P(t)=V*I."""
    V = np.asarray(series_first(data["V_node"]), dtype=float)
    I = np.asarray(series_first(data["I_vo2"]), dtype=float)
    return V * I


# -----------------------------
# Amir requested plots
# -----------------------------

def plot_baselines_T_and_V_vs_time(
    data: SimOut,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
) -> None:
    """Plot V(t) and T(t) in the window and mark local minima (baselines) and maxima."""
    t = np.asarray(data["time_s"], dtype=float)
    V = np.asarray(series_first(data["V_node"]), dtype=float)
    T = np.asarray(series_first(data["T_K"]), dtype=float)

    m = _window_mask(data["time_s"], t_start_us, t_end_us)
    if not np.any(m):
        raise ValueError("Requested time window is empty.")

    tw = t[m] * 1e6
    Vw = V[m]
    Tw = T[m]

    V_min_idx, V_max_idx = _local_extrema_indices(Vw)
    T_min_idx, T_max_idx = _local_extrema_indices(Tw)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(tw, Vw, linewidth=1)
    if V_min_idx.size:
        ax1.plot(tw[V_min_idx], Vw[V_min_idx], "o", markersize=4, label="V minima (baseline)")
    if V_max_idx.size:
        ax1.plot(tw[V_max_idx], Vw[V_max_idx], "o", markersize=3, label="V maxima")
    ax1.set_ylabel("V_device (V)")
    ax1.grid(True)
    ax1.legend(loc="best")

    ax2.plot(tw, Tw, linewidth=1)
    if T_min_idx.size:
        ax2.plot(tw[T_min_idx], Tw[T_min_idx], "o", markersize=4, label="T minima (baseline)")
    if T_max_idx.size:
        ax2.plot(tw[T_max_idx], Tw[T_max_idx], "o", markersize=3, label="T maxima")
    ax2.set_ylabel("T (K)")
    ax2.set_xlabel("time (µs)")
    ax2.grid(True)
    ax2.legend(loc="best")

    fig.suptitle("Baselines (minima) of V and T vs time")


def plot_Vmax_vs_Vin(
    vin_list: List[float],
    results: Dict[float, SimOut],
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
) -> None:
    vmax_list: List[float] = []
    for v in vin_list:
        d = results[v]
        V = np.asarray(series_first(d["V_node"]), dtype=float)
        m = _window_mask(d["time_s"], t_start_us, t_end_us)
        Vw = V[m] if np.any(m) else V
        vmax_list.append(float(np.max(Vw)))

    plt.figure(figsize=(9, 4.5))
    plt.plot(vin_list, vmax_list, "o-")
    plt.xlabel("Vin (V)")
    plt.ylabel("V_max on device (V)")
    plt.title("V_max (peak device voltage) vs Vin")
    plt.grid(True)


def plot_power_time_and_peaks_vs_Vin(
    vin_list: List[float],
    results: Dict[float, SimOut],
    example_vin: float,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
) -> None:
    # Example P(t)
    if example_vin not in results:
        raise ValueError(f"example_vin={example_vin} must be included in --vin_list")
    d0 = results[example_vin]
    t0 = np.asarray(d0["time_s"], dtype=float)
    m0 = _window_mask(d0["time_s"], t_start_us, t_end_us)
    P0 = power_trace_W(d0)

    plt.figure(figsize=(10, 4.5))
    plt.plot(t0[m0] * 1e6, P0[m0], linewidth=1)
    plt.xlabel("time (µs)")
    plt.ylabel("P_device (W)")
    plt.title(f"Power vs time (Vin={example_vin:.2f} V, {t_start_us:.0f}–{t_end_us:.0f} µs)")
    plt.grid(True)

    # Peaks vs Vin
    pmin: List[float] = []
    pmax: List[float] = []
    for v in vin_list:
        d = results[v]
        P = power_trace_W(d)
        m = _window_mask(d["time_s"], t_start_us, t_end_us)
        Pw = P[m] if np.any(m) else P
        pmin.append(float(np.min(Pw)))
        pmax.append(float(np.max(Pw)))

    plt.figure(figsize=(9, 4.5))
    plt.plot(vin_list, pmax, "o-", label="P_max")
    plt.plot(vin_list, pmin, "o-", label="P_min")
    plt.xlabel("Vin (V)")
    plt.ylabel("Power extrema (W)")
    plt.title("Power peaks (min/max) vs Vin")
    plt.grid(True)
    plt.legend(loc="best")


def plot_capacitance_effect_on_power(
    vin: float,
    C_values_pF: List[float],
    resist_params: YuanhangResistParams,
    circuit_params: YuanhangCircuitParams,
    t_end: float,
    dt: float,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
) -> None:
    plt.figure(figsize=(10, 4.5))
    for C in C_values_pF:
        cp = replace(circuit_params, C_par_pF=float(C))
        d = simulate_yuanhang(
            Vin=vin,
            t_end=t_end,
            dt=dt,
            resist_params=resist_params,
            circuit_params=cp,
        )
        t = np.asarray(d["time_s"], dtype=float)
        m = _window_mask(d["time_s"], t_start_us, t_end_us)
        P = power_trace_W(d)
        plt.plot(t[m] * 1e6, P[m], linewidth=1, label=f"C={C:.1f} pF")

    plt.xlabel("time (µs)")
    plt.ylabel("P_device (W)")
    plt.title(f"Effect of capacitance on power (Vin={vin:.2f} V)")
    plt.grid(True)
    plt.legend(loc="best")


def plot_frequency_3d_vs_C_and_Rload(
    vin: float,
    C_values_pF: List[float],
    Rload_values_kohm: List[float],
    resist_params: YuanhangResistParams,
    circuit_params: YuanhangCircuitParams,
    t_end: float,
    dt: float,
    threshold_A: float = 1e-3,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    C_grid: List[float] = []
    R_grid: List[float] = []
    f_grid: List[float] = []

    for Rk in Rload_values_kohm:
        for C in C_values_pF:
            cp = replace(circuit_params, R_series_kohm=float(Rk), C_par_pF=float(C))
            d = simulate_yuanhang(
                Vin=vin,
                t_end=t_end,
                dt=dt,
                resist_params=resist_params,
                circuit_params=cp,
            )
            f = mean_frequency_mhz(d, threshold_A=threshold_A)
            C_grid.append(float(C))
            R_grid.append(float(Rk))
            f_grid.append(float(f))

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(np.asarray(C_grid), np.asarray(R_grid), np.asarray(f_grid))
    ax.set_xlabel("C_par (pF)")
    ax.set_ylabel("R_load (kΩ)")
    ax.set_zlabel("Frequency (MHz)")
    ax.set_title(f"Frequency vs (C, R_load) at Vin={vin:.2f} V")


def plot_R_insulating_vs_time(
    data: SimOut,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
    g_threshold: float = 0.9,
) -> None:
    """Plot R(t) and highlight insulating-state parts using g(t) threshold."""
    t = np.asarray(data["time_s"], dtype=float)
    R = np.asarray(series_first(data["R_vo2"]), dtype=float)
    g = np.asarray(series_first(data["g"]), dtype=float)

    m = _window_mask(data["time_s"], t_start_us, t_end_us)
    tw = t[m] * 1e6
    Rw = R[m]
    gw = g[m]

    insulating = gw >= g_threshold

    plt.figure(figsize=(10, 4.5))
    plt.plot(tw, Rw, linewidth=1, label="R_vo2(t)")
    if np.any(insulating):
        plt.plot(tw[insulating], Rw[insulating], ".", markersize=2, label=f"Insulating (g≥{g_threshold})")
    plt.xlabel("time (µs)")
    plt.ylabel("R_vo2 (Ω)")
    plt.title("Resistance in insulating state vs time")
    plt.grid(True)
    plt.legend(loc="best")


# -----------------------------
# CLI
# -----------------------------

def _parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analysis utilities for neuristor simulations")

    parser.add_argument("--vin", type=float, default=14.5, help="Single Vin for time-domain plots")
    parser.add_argument("--vin_list", type=str, default="", help="Comma-separated Vin list for sweep plots")

    parser.add_argument("--t_end_us", type=float, default=300.0, help="Simulation duration (µs)")
    parser.add_argument("--dt_ns", type=float, default=10.0, help="Timestep (ns)")

    parser.add_argument("--Rload_kohm", type=float, default=None, help="Override load resistor (kΩ)")
    parser.add_argument("--Cpar_pF", type=float, default=None, help="Override capacitance (pF)")

    parser.add_argument("--C_sweep_pF", type=str, default="", help="Comma-separated capacitances (pF) for C sweep")
    parser.add_argument("--R_sweep_kohm", type=str, default="", help="Comma-separated R_load values (kΩ) for 3D sweep")

    parser.add_argument("--paper", action="store_true", help="Use the paper preset parameters")

    parser.add_argument("--do_baselines", action="store_true", help="Baselines of T and V vs time")
    parser.add_argument("--do_vmax", action="store_true", help="V_max vs Vin (requires --vin_list)")
    parser.add_argument("--do_power", action="store_true", help="Power vs time + power peaks vs Vin (requires --vin_list and example Vin in list)")
    parser.add_argument("--do_Cpower", action="store_true", help="Overlay power(t) for multiple capacitances at --vin")
    parser.add_argument("--do_3d", action="store_true", help="3D sweep: frequency vs (C, R_load) at fixed --vin")
    parser.add_argument("--do_Rins", action="store_true", help="Resistance in insulating state vs time")
    parser.add_argument(
        "--do_Cpeaks",
        action="store_true",
        help="Sweep capacitance and plot cycle-averaged P_max/P_min vs C (mean ± std)",
    )
    parser.add_argument("--C_start_pF", type=float, default=80.0, help="Capacitance sweep start (pF)")
    parser.add_argument("--C_stop_pF", type=float, default=250.0, help="Capacitance sweep stop (pF)")
    parser.add_argument("--C_step_pF", type=float, default=10.0, help="Capacitance sweep step (pF)")
    parser.add_argument(
        "--auto_Cdomain",
        action="store_true",
        help="Auto-detect oscillatory capacitance band (coarse scan) then run --do_Cpeaks over that band",
    )
    parser.add_argument("--C_coarse_step_pF", type=float, default=10.0, help="Coarse capacitance scan step (pF)")
    parser.add_argument("--C_fine_step_pF", type=float, default=10.0, help="Fine capacitance step (pF) within detected band")

    args = parser.parse_args()

    t_end = args.t_end_us * 1e-6
    dt = args.dt_ns * 1e-9

    resist_params = YuanhangResistParams()
    circuit_params = YuanhangCircuitParams()

    if args.paper:
        # Same as model.py / histogram.py 'paper' overrides
        resist_params.R0 = 5.35882879e-3
        resist_params.Ea_over_k = 5.22047417e3
        resist_params.Rm0 = 262.5
        resist_params.Rm_factor = 4.90025335
        resist_params.w = 7.19357064
        resist_params.Tc_K = 3.32805839e2
        resist_params.beta = 2.52796285e-1
        resist_params.gamma = 9.56269682e-1
        resist_params.width_factor = 1.0
        resist_params.T_min_K = 305.0
        resist_params.T_max_K = 370.0
        resist_params.reversal_threshold_K = 0.01

        circuit_params.R_series_kohm = 12.0
        circuit_params.C_par_pF = 145.34619293
        circuit_params.Cth_mW_ns_per_K = 49.62776831
        circuit_params.Sth_mW_per_K = 0.20558726
        circuit_params.couple_factor = 0.0
        circuit_params.Cth_factor = 1.0
        circuit_params.noise_strength = 0.0
        circuit_params.dimension = 1
        circuit_params.T_base_K = 325.0

    if args.Rload_kohm is not None:
        circuit_params = replace(circuit_params, R_series_kohm=float(args.Rload_kohm))
    if args.Cpar_pF is not None:
        circuit_params = replace(circuit_params, C_par_pF=float(args.Cpar_pF))

    # Single run for time-domain plots
    data_single = simulate_yuanhang(
        Vin=args.vin,
        t_end=t_end,
        dt=dt,
        resist_params=resist_params,
        circuit_params=circuit_params,
    )

    # Optional Vin sweep
    vin_list: List[float] = []
    results: Dict[float, SimOut] = {}
    if args.vin_list.strip():
        vin_list = _parse_csv_floats(args.vin_list)
        for v in vin_list:
            results[v] = simulate_yuanhang(
                Vin=v,
                t_end=t_end,
                dt=dt,
                resist_params=resist_params,
                circuit_params=circuit_params,
                noise_seed=123 + int(v * 10),
            )

    if args.do_baselines:
        plot_baselines_T_and_V_vs_time(data_single)

    if args.do_vmax:
        if not vin_list:
            raise SystemExit("--do_vmax requires --vin_list")
        plot_Vmax_vs_Vin(vin_list, results)

    if args.do_power:
        if not vin_list:
            raise SystemExit("--do_power requires --vin_list")
        plot_power_time_and_peaks_vs_Vin(vin_list, results, example_vin=args.vin)

    if args.do_Cpower:
        if not args.C_sweep_pF.strip():
            raise SystemExit("--do_Cpower requires --C_sweep_pF")
        C_vals = _parse_csv_floats(args.C_sweep_pF)
        plot_capacitance_effect_on_power(
            vin=args.vin,
            C_values_pF=C_vals,
            resist_params=resist_params,
            circuit_params=circuit_params,
            t_end=t_end,
            dt=dt,
        )

    if args.do_3d:
        if not args.C_sweep_pF.strip() or not args.R_sweep_kohm.strip():
            raise SystemExit("--do_3d requires --C_sweep_pF and --R_sweep_kohm")
        C_vals = _parse_csv_floats(args.C_sweep_pF)
        R_vals = _parse_csv_floats(args.R_sweep_kohm)
        plot_frequency_3d_vs_C_and_Rload(
            vin=args.vin,
            C_values_pF=C_vals,
            Rload_values_kohm=R_vals,
            resist_params=resist_params,
            circuit_params=circuit_params,
            t_end=t_end,
            dt=dt,
        )

    if args.do_Rins:
        plot_R_insulating_vs_time(data_single)

    if args.do_Cpeaks:
        if args.auto_Cdomain:
            Cmin, Cmax = _auto_find_oscillatory_C_band(
                vin=args.vin,
                C_start_pF=args.C_start_pF,
                C_stop_pF=args.C_stop_pF,
                C_step_coarse_pF=args.C_coarse_step_pF,
                resist_params=resist_params,
                circuit_params=circuit_params,
                t_end=t_end,
                dt=dt,
            )
            if Cmin is None or Cmax is None:
                raise SystemExit(
                    f"auto_Cdomain: no oscillatory C found between {args.C_start_pF:.1f} and {args.C_stop_pF:.1f} pF"
                )

            print(f"[auto_Cdomain] Coarse sweep done. Detected oscillatory band: {Cmin:.1f}–{Cmax:.1f} pF")
            fine = float(args.C_fine_step_pF)
            C_vals = np.arange(Cmin, Cmax + 0.5 * fine, fine, dtype=float).tolist()
            print(f"[auto_Cdomain] Fine sweep: {len(C_vals)} points in steps of {fine:.1f} pF")

            plot_capacitance_sweep_power_extrema(
                vin=args.vin,
                C_start_pF=Cmin,
                C_stop_pF=Cmax,
                C_step_pF=fine,
                resist_params=resist_params,
                circuit_params=circuit_params,
                t_end=t_end,
                dt=dt,
                C_values_pF=C_vals,
            )
        else:
            plot_capacitance_sweep_power_extrema(
                vin=args.vin,
                C_start_pF=args.C_start_pF,
                C_stop_pF=args.C_stop_pF,
                C_step_pF=args.C_step_pF,
                resist_params=resist_params,
                circuit_params=circuit_params,
                t_end=t_end,
                dt=dt,
            )

    plt.show()


if __name__ == "__main__":
    main()
