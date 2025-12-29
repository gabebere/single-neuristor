

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
from dataclasses import replace
from typing import Dict, Iterable, List, Tuple, Optional

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
