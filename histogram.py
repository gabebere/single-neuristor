"""
Single VO₂ neuristor simulator CLI (Yuanhang Zhang–Qiu–Di Ventra model).

The simulator core lives in model.py; this script provides sweep helpers and plotting.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from model import (
    SimOut,
    YuanhangCircuitParams,
    YuanhangResistParams,
    detect_spike_times,
    find_oscillatory_band_1d,
    is_oscillatory,
    series_first,
    series_mean,
    simulate_yuanhang,
)

# Backwards-compatible aliases for existing CLI code
_series_first = series_first
_series_mean = series_mean
_detect_spike_times = detect_spike_times
_has_oscillations = is_oscillatory


def _simulate_single_neuristor(
    Vin: float | Sequence[float],
    t_end: float = 60e-6,
    dt: float = 10e-9,
    resist_params: YuanhangResistParams | None = None,
    circuit_params: YuanhangCircuitParams | None = None,
    init: Dict[str, float] | None = None,
    start_branch: str = "insulator",
    lattice_shape: Tuple[int, int] = (1, 1),
    noise_seed: int | None = None,
) -> SimOut:
    """Convenience wrapper to keep CLI code unchanged."""
    return simulate_yuanhang(
        Vin=Vin,
        t_end=t_end,
        dt=dt,
        resist_params=resist_params,
        circuit_params=circuit_params,
        init=init,
        start_branch=start_branch,
        lattice_shape=lattice_shape,
        noise_seed=noise_seed,
    )


def _auto_find_oscillatory_band(
    v_start: float,
    v_step: float,
    v_max: float,
    t_end: float,
    dt: float,
    start_branch: str,
    lattice_shape: Tuple[int, int],
    noise_seed: int | None,
    resist_params: YuanhangResistParams | None,
    circuit_params: YuanhangCircuitParams | None,
) -> Tuple[float | None, float | None]:
    """Wrapper around model.find_oscillatory_band_1d to retain the CLI signature."""

    def run_sim(v: float) -> SimOut:
        return simulate_yuanhang(
            Vin=v,
            t_end=t_end,
            dt=dt,
            resist_params=resist_params,
            circuit_params=circuit_params,
            start_branch=start_branch,
            lattice_shape=lattice_shape,
            noise_seed=None if noise_seed is None else noise_seed + int(v * 10),
        )

    return find_oscillatory_band_1d(run_sim, start=v_start, stop=v_max, step=v_step, osc_check=is_oscillatory)


def run_and_save_csv(
    Vin: float | Sequence[float],
    t_end: float = 60e-6,
    dt: float = 10e-9,
    filename: str | None = None,
    lattice_shape: Tuple[int, int] = (1, 1),
) -> str:
    """Run a single-device simulation and write traces to CSV.
    Columns: time_s, V_node, I_load, I_vo2, T_K, R_vo2, g (one row per step)."""
    if lattice_shape != (1, 1):
        raise ValueError("CSV export currently supports only single-device simulations (grid_shape == (1, 1)).")
    if not isinstance(Vin, (int, float)):
        raise ValueError("Scalar Vin is required for CSV export.")
    data = _simulate_single_neuristor(Vin=Vin, t_end=t_end, dt=dt, lattice_shape=lattice_shape)
    if filename is None:
        filename = f"neuristor_yuanhang_Vin_{Vin:.3f}V_60us_10ns.csv".replace("/", "_")
    outpath = os.path.join(os.path.dirname(__file__), filename)

    keys = ["time_s", "V_node", "I_load", "I_vo2", "T_K", "R_vo2", "g"]
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for idx in range(len(data["time_s"])):
            writer.writerow([data[k][idx] for k in keys])
    return outpath


def _run_vin_sweep(
    vins,
    t_end,
    dt,
    start_branch: str = "insulator",
    lattice_shape: Tuple[int, int] = (1, 1),
    noise_seed: int | None = None,
    resist_params: YuanhangResistParams | None = None,
    circuit_params: YuanhangCircuitParams | None = None,
):
    """Run a Vin sweep and return a dict {Vin: SimOut}. Pass-through of parameter objects."""
    results = {}
    total = len(vins)
    for idx, v in enumerate(vins, start=1):
        print(f"[sweep] Simulating Vin = {v:.3f} V ({idx}/{total})")
        results[v] = _simulate_single_neuristor(
            Vin=v,
            t_end=t_end,
            dt=dt,
            resist_params=resist_params,
            circuit_params=circuit_params,
            start_branch=start_branch,
            lattice_shape=lattice_shape,
            noise_seed=None if noise_seed is None else noise_seed + int(v * 10),
        )
    print("[sweep] Completed Vin sweep.")
    return results


if __name__ == "__main__":
    # CLI: use --paper to lock paper constants; --vin_list "9,11,13,15,17"; --save_dir to dump per‑Vin PNGs.
    parser = argparse.ArgumentParser(
        description="Simulate a single VO₂ neuristor (Yuanhang Zhang model). Default sweep: 9,11,13,15,17 V."
    )
    parser.add_argument("--t_end_us", type=float, default=300.0, help="Simulation duration in microseconds")
    parser.add_argument("--dt_ns", type=float, default=10.0, help="Timestep in nanoseconds")
    parser.add_argument("--vin_list", type=str, default="", help="Comma-separated Vin list, e.g. '9,11,13,15,17'")
    parser.add_argument("--save_png", type=str, default="", help="If provided, save combined sweep plot to this path")
    parser.add_argument("--save_dir", type=str, default="", help="If provided, dump per-Vin plots into this directory")
    parser.add_argument("--no_combined", action="store_true", help="Skip the combined sweep figure")
    parser.add_argument("--start_branch", choices=["insulator", "metal"], default="insulator", help="Initial hysteresis bias")
    parser.add_argument("--nx", type=int, default=1, help="Number of devices along x (>=1)")
    parser.add_argument("--ny", type=int, default=1, help="Number of devices along y (>=1)")
    parser.add_argument("--noise_seed", type=int, default=None, help="Seed for the thermal noise RNG")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Use paper-exact preset (Zhang–Qiu–Di Ventra): additive mixing, Table-1 constants, zero coupling, plot I_load with 0–5 mA scale on per-V plots.",
    )
    parser.add_argument(
        "--negative",
        action="store_true",
        help="If set, for the default sweep include both negative and positive Vin (−15.5→−10.5 and 10.5→15.5). For an explicit --vin_list, flip all Vin values to negative.",
    )
    parser.add_argument(
        "--auto_domain",
        action="store_true",
        help="Automatically find the oscillatory Vin band using 0.5 V steps from 0 V, then sweep that band with 0.05 V resolution.",
    )
    args = parser.parse_args()

    t_end = args.t_end_us * 1e-6
    dt = args.dt_ns * 1e-9
    lattice_shape = (max(1, args.nx), max(1, args.ny))

    resist_params = YuanhangResistParams()
    circuit_params = YuanhangCircuitParams()
    if args.paper:
        # Paper-exact constants (matching Methods/Table 1; values already match defaults)
        resist_params.R0 = 5.35882879e-3
        resist_params.Ea_over_k = 5.22047417e3
        resist_params.Rm0 = 262.5
        resist_params.Rm_factor = 4.90025335  # Rm ≈ 1286 Ω
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

    if args.auto_domain:
        # Coarse scan: 0.5 V steps from 0 V up to an upper bound (default 20 V)
        v_start = 0.0
        v_step_coarse = 0.5
        v_max_scan = 50
        v_min_osc, v_max_osc = _auto_find_oscillatory_band(
            v_start=v_start,
            v_step=v_step_coarse,
            v_max=v_max_scan,
            t_end=t_end,
            dt=dt,
            start_branch=args.start_branch,
            lattice_shape=lattice_shape,
            noise_seed=args.noise_seed,
            resist_params=resist_params,
            circuit_params=circuit_params,
        )
        if v_min_osc is None or v_max_osc is None:
            raise SystemExit(
                f"auto_domain: no oscillatory Vin found between {v_start:.2f} V and {v_max_scan:.2f} V. "
                "Try increasing the scan range or adjusting circuit parameters (e.g., R_series)."
            )
        print(
            f"[auto_domain] Coarse sweep done. Detected oscillatory band from "
            f"{v_min_osc:.2f} V to {v_max_osc:.2f} V."
        )
        # Fine sweep within the oscillatory band using 0.05 V steps.
        step_fine = 0.05
        n_steps = int(round((v_max_osc - v_min_osc) / step_fine))
        vins = [v_min_osc + step_fine * i for i in range(n_steps + 1)]
        print(
            f"[auto_domain] Fine sweep: {len(vins)} points from {v_min_osc:.2f} V "
            f"to {v_max_osc:.2f} V in steps of {step_fine:.2f} V."
        )
    else:
        if args.vin_list.strip():
            # User-specified Vin list
            try:
                vins = [float(val.strip()) for val in args.vin_list.split(",") if val.strip()]
            except ValueError:
                raise SystemExit("Could not parse --vin_list. Use comma-separated floats like '9,11,13'.")
            # For an explicit list, --negative flips all Vin values to negative.
            if args.negative:
                vins = [-v for v in vins]
        else:
            # Default: sweep Vin from 10.5 V to 15.5 V in 0.05 V steps.
            v_min, v_max = 10.5, 17.95
            step = 0.05
            n_steps = int(round((v_max - v_min) / step))
            pos_vins = [v_min + step * i for i in range(n_steps + 1)]
            if args.negative:
                # Build a symmetric sweep: negative from -15.5→-10.5, then positive 10.5→15.5.
                neg_vins = list(reversed([-v for v in pos_vins]))
                vins = neg_vins + pos_vins
            else:
                vins = pos_vins

    results = _run_vin_sweep(
        vins,
        t_end=t_end,
        dt=dt,
        start_branch=args.start_branch,
        lattice_shape=lattice_shape,
        noise_seed=args.noise_seed,
        resist_params=resist_params,
        circuit_params=circuit_params,
    )

    multi_device = lattice_shape != (1, 1)
    if multi_device:
        print(
            "Multi-device lattice detected: combined plots show average currents; per-V plots show device (0,0)."
        )

    # Combined time-domain sweep disabled: we only generate the ISI histogram below.

    # Inter-spike-interval (ISI) histogram based on I_vo2 in the 25–300 µs window.
    # For each Vin that actually oscillates, we:
    #   1. Restrict to t ∈ [25, 300] µs.
    #   2. Detect spike times on I_vo2 using a simple local-max detector.
    #   3. Compute Δt between consecutive spikes (ISI, in µs).
    #   4. Plot a histogram of ISI for each Vin, overlaid with transparency.
    fig, ax = plt.subplots(figsize=(9, 5))
    isi_data: Dict[float, np.ndarray] = {}
    for v in vins:
        data = results[v]
        time_s = data["time_s"]
        I_vo2 = _series_first(data["I_vo2"])
        if not time_s or not I_vo2:
            continue
        t_arr = np.asarray(time_s, dtype=float)
        I_arr = np.asarray(I_vo2, dtype=float)
        t_us_arr = t_arr * 1e6
        # Restrict to the 25–300 µs window.
        mask = (t_us_arr >= 25.0) & (t_us_arr <= 300.0)
        if not np.any(mask):
            continue
        t_win = t_arr[mask]
        I_win = I_arr[mask]
        spike_times = _detect_spike_times(t_win.tolist(), I_win.tolist(), threshold_A=1e-3)
        if len(spike_times) < 2:
            continue  # no oscillations for this Vin
        isi_us = np.diff(np.asarray(spike_times)) * 1e6
        if isi_us.size == 0:
            continue
        isi_data[v] = isi_us

    if isi_data:
        # Use a continuous colormap (viridis) so color encodes Vin smoothly.
        vin_values = sorted(isi_data.keys())
        vmin_global = float(min(vin_values))
        vmax_global = float(max(vin_values))
        cmap = plt.cm.viridis
        norm = Normalize(vmin=vmin_global, vmax=vmax_global)
        vin_arr = np.asarray(vin_values, dtype=float)

        # For each Vin, use a single bin that exactly covers that Vin's ISI range.
        # Bar width represents ISI spread; bar height is the number of intervals.
        for v in vin_values:
            isi_us = isi_data[v]
            vmin = float(isi_us.min())
            vmax = float(isi_us.max())
            # If all ISIs are identical, expand the bin very slightly so Matplotlib
            # can render a visible bar.
            if vmin == vmax:
                eps = 0.01 * (abs(vmin) if vmin != 0.0 else 1.0)
                vmin -= eps
                vmax += eps
            bins = [vmin, vmax]
            color = cmap(norm(v))
            ax.hist(isi_us, bins=bins, alpha=0.8, color=color)

        # Add a colorbar to indicate Vin mapping.
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Vin (V)")

        ax.set_xlabel("Inter-spike interval (µs)")
        ax.set_ylabel("Count")
        ax.set_title("Inter-spike interval histogram (I_vo2 spikes, 25–300 µs window)")
        ax.grid(True)

        # --- Mean ISI vs Vin (with spread) ---
        # For each Vin that oscillates, compute the mean and standard deviation of the ISI
        mean_isi_us = []
        std_isi_us = []
        for v in vin_values:
            isi_us = isi_data[v]
            mean_isi_us.append(float(np.mean(isi_us)))
            std_isi_us.append(float(np.std(isi_us)))

        # Convert to arrays and (optionally) split into negative and positive Vin sets.
        mean_isi_arr = np.asarray(mean_isi_us, dtype=float)
        std_isi_arr = np.asarray(std_isi_us, dtype=float)

        # Frequency (MHz) from mean ISI (µs): f_MHz = 1.0 / <ISI_µs>
        freq_MHz_arr = 1.0 / mean_isi_arr

        # When --negative is used and there are both negative and positive Vin values,
        # show separate panels. Otherwise, show a single plot with all Vin (typically positive).
        neg_mask = (vin_arr < 0.0)
        pos_mask = (vin_arr > 0.0)

        if getattr(args, "negative", False) and np.any(neg_mask) and np.any(pos_mask):
            vin_neg = vin_arr[neg_mask]
            vin_pos = vin_arr[pos_mask]
            mean_neg = mean_isi_arr[neg_mask]
            mean_pos = mean_isi_arr[pos_mask]
            std_neg = std_isi_arr[neg_mask]
            std_pos = std_isi_arr[pos_mask]

            freq_neg = freq_MHz_arr[neg_mask]
            freq_pos = freq_MHz_arr[pos_mask]

            # --- Mean ISI vs Vin: negative and positive panels side by side ---
            fig2, (ax2_neg, ax2_pos) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
            ax2_neg.errorbar(vin_neg, mean_neg, yerr=std_neg, fmt="o-", capsize=3)
            ax2_neg.set_title("Mean ISI vs Vin (negative bias)")
            ax2_neg.set_xlabel("Vin (V)")
            ax2_neg.set_ylabel("Mean inter-spike interval (µs)")
            ax2_neg.grid(True)

            ax2_pos.errorbar(vin_pos, mean_pos, yerr=std_pos, fmt="o-", capsize=3)
            ax2_pos.set_title("Mean ISI vs Vin (positive bias)")
            ax2_pos.set_xlabel("Vin (V)")
            ax2_pos.grid(True)
            fig2.suptitle("Mean ISI vs Vin (25–300 µs window)")
        else:
            # Single-panel plot (no negative sweep, or only one sign present)
            fig2, ax2 = plt.subplots(figsize=(9, 4.5))
            ax2.errorbar(vin_arr, mean_isi_arr, yerr=std_isi_arr, fmt="o-", capsize=3)
            ax2.set_xlabel("Vin (V)")
            ax2.set_ylabel("Mean inter-spike interval (µs)")
            ax2.set_title("Mean ISI vs Vin (25–300 µs window)")
            ax2.grid(True)

        # --- Frequency vs Vin ---
        if getattr(args, "negative", False) and np.any(neg_mask) and np.any(pos_mask):
            # Use the same vin_neg/vin_pos, freq_neg/freq_pos as above
            fig3, (ax3_neg, ax3_pos) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
            ax3_neg.plot(vin_neg, freq_neg, "o-")
            ax3_neg.set_title("Frequency vs Vin (negative bias)")
            ax3_neg.set_xlabel("Vin (V)")
            ax3_neg.set_ylabel("Oscillation frequency (MHz)")
            ax3_neg.grid(True)

            ax3_pos.plot(vin_pos, freq_pos, "o-")
            ax3_pos.set_title("Frequency vs Vin (positive bias)")
            ax3_pos.set_xlabel("Vin (V)")
            ax3_pos.grid(True)
            fig3.suptitle("Oscillation frequency vs Vin (25–300 µs window)")
        else:
            fig3, ax3 = plt.subplots(figsize=(9, 4.5))
            ax3.plot(vin_arr, freq_MHz_arr, "o-")
            ax3.set_xlabel("Vin (V)")
            ax3.set_ylabel("Oscillation frequency (MHz)")
            ax3.set_title("Oscillation frequency vs Vin (25–300 µs window)")
            ax3.grid(True)

        # --- Peak temperature vs Vin (steady‑state window) ---
        # For each Vin, compute the maximum temperature in the 25–300 µs window.
        peak_T_K = []
        for v in vin_values:
            data = results[v]
            time_s = data["time_s"]
            T_series = _series_first(data["T_K"])
            if not time_s or not T_series:
                peak_T_K.append(np.nan)
                continue

            t_arr = np.asarray(time_s, dtype=float)
            T_arr = np.asarray(T_series, dtype=float)
            t_us = t_arr * 1e6

            # Restrict to the same "steady‑state" window used for the ISI analysis.
            mask = (t_us >= 25.0) & (t_us <= 300.0)
            if np.any(mask):
                T_win = T_arr[mask]
            else:
                # Fallback to the full trace if the window is empty for some reason.
                T_win = T_arr

            peak_T_K.append(float(np.max(T_win)))

        peak_T_arr = np.asarray(peak_T_K, dtype=float)

        if getattr(args, "negative", False) and np.any(neg_mask) and np.any(pos_mask):
            peak_neg = peak_T_arr[neg_mask]
            peak_pos = peak_T_arr[pos_mask]

            fig4, (ax4_neg, ax4_pos) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
            ax4_neg.plot(vin_neg, peak_neg, "o-")
            ax4_neg.set_title("Peak T vs Vin (negative bias)")
            ax4_neg.set_xlabel("Vin (V)")
            ax4_neg.set_ylabel("Peak temperature (K)")
            ax4_neg.grid(True)

            ax4_pos.plot(vin_pos, peak_pos, "o-")
            ax4_pos.set_title("Peak T vs Vin (positive bias)")
            ax4_pos.set_xlabel("Vin (V)")
            ax4_pos.grid(True)
            fig4.suptitle("Peak device temperature vs Vin (25–300 µs window)")
        else:
            fig4, ax4 = plt.subplots(figsize=(9, 4.5))
            ax4.plot(vin_arr, peak_T_arr, "o-")
            ax4.set_xlabel("Vin (V)")
            ax4.set_ylabel("Peak temperature (K)")
            ax4.set_title("Peak device temperature vs Vin (25–300 µs window)")
            ax4.grid(True)

    plt.show()
