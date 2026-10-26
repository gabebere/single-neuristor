from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import neuristor.model as model
from neuristor.current_domain_search import analyze_current_trace
from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_steps, simulate_current_waveform
from neuristor.model import HysteresisArray, YuanhangResistParams


PRESET_PATH = ROOT / "presets" / "resistance_100425_chip1_gap3.json"
PUBLIC_JOBS = ROOT / "public_jobs"
OUTPUTS = ROOT / "outputs"
PAPER_CURRENT_WINDOW_UA = (200.0, 600.0)
PAPER_FREQ_WINDOW_MHZ = (40.0, 60.0)
RF_REF_OHM = 50.0


def _repo_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _job_id(prefix: str) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{prefix}_{uuid.uuid4().hex[:6]}"


def _load_resistance_preset() -> YuanhangResistParams:
    payload = json.loads(PRESET_PATH.read_text())
    return YuanhangResistParams(**payload["resist_params"])


def _paper_style_params(resist: YuanhangResistParams) -> CurrentDriveParams:
    return CurrentDriveParams(
        dt_s=0.00183e-9,
        t_end_s=600e-9,
        t_pre_s=0.0,
        pulse_on_s=150e-9,
        pulse_off_s=450e-9,
        V_init_V=0.0,
        T0_K=311.21937437938016,
        T_init_K=311.21937437938016,
        C_F=4.0e-12,
        C_th_J_per_K=1.0e-12,
        S_e_W_per_K=0.10e-3,
        sigma_W_sqrt_s=0.0,
        resist_params=resist,
        start_branch="insulator",
    )


def _validated_sample_params(resist: YuanhangResistParams) -> CurrentDriveParams:
    return CurrentDriveParams(
        dt_s=0.01582e-9,
        t_end_s=600e-9,
        t_pre_s=0.0,
        pulse_on_s=150e-9,
        pulse_off_s=450e-9,
        V_init_V=0.0,
        T0_K=311.21937437938016,
        T_init_K=311.21937437938016,
        C_F=25.930953e-12,
        C_th_J_per_K=5.0e-12,
        S_e_W_per_K=0.10e-3,
        sigma_W_sqrt_s=0.0,
        resist_params=resist,
        start_branch="insulator",
    )


def _as_config_params(params: CurrentDriveParams) -> dict:
    data = asdict(params)
    data["resist_params"] = asdict(params.resist_params)
    return data


def _downsample_indices(n: int, max_points: int = 2400) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return np.unique(idx)


def _count_turns(values: np.ndarray) -> int:
    if values.size < 3:
        return 0
    dv = np.diff(values)
    return int(np.sum((dv[:-1] * dv[1:]) < 0.0))


def _input_power_dbm(i_trace_a: np.ndarray) -> float:
    active = np.abs(i_trace_a) > 0.0
    vals = i_trace_a[active] if np.any(active) else i_trace_a
    i_rms = float(np.sqrt(np.mean(vals**2))) if vals.size else 0.0
    p_w = max(i_rms * i_rms * RF_REF_OHM, 1e-18)
    return 10.0 * math.log10(p_w / 1e-3)


def _fft_gain_spectrum(
    t_s: np.ndarray,
    i_a: np.ndarray,
    v_v: np.ndarray,
    *,
    fmin_mhz: float = 1.0,
    fmax_mhz: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if t_s.size < 4:
        return np.array([]), np.array([]), np.array([])
    dt = float(np.median(np.diff(t_s)))
    if dt <= 0.0:
        return np.array([]), np.array([]), np.array([])
    i0 = i_a - float(np.mean(i_a))
    v0 = v_v - float(np.mean(v_v))
    window = np.hanning(i0.size)
    v_fft = np.fft.rfft(v0 * window)
    f_hz = np.fft.rfftfreq(i0.size, d=dt)
    i_rms = float(np.sqrt(np.mean((i_a[np.abs(i_a) > 0.0] if np.any(np.abs(i_a) > 0.0) else i_a) ** 2)))
    if i_rms <= 0.0:
        return np.array([]), np.array([]), np.array([])
    gain_linear = np.abs(v_fft) / max(i_rms * RF_REF_OHM, 1e-18)
    gain_db = 20.0 * np.log10(np.maximum(gain_linear, 1e-12))
    f_mhz = f_hz * 1e-6
    mask = (f_mhz >= fmin_mhz) & (f_mhz <= fmax_mhz)
    return f_mhz[mask], gain_linear[mask], gain_db[mask]


def _threshold_crossings(t_ns: np.ndarray, v_mV: np.ndarray) -> np.ndarray:
    if v_mV.size < 8:
        return np.array([], dtype=int)
    low, high = np.quantile(v_mV, [0.1, 0.9])
    if high - low <= 1e-12:
        return np.array([], dtype=int)
    threshold = 0.5 * (low + high)
    return np.flatnonzero((v_mV[:-1] < threshold) & (v_mV[1:] >= threshold)) + 1


def _cycle_energy_metrics(out: dict, params: CurrentDriveParams) -> dict[str, float]:
    t_ns = out["t"] * 1e9
    active_start = float(params.pulse_on_s * 1e9)
    active_end = float(params.pulse_off_s * 1e9) if params.pulse_off_s is not None else float(t_ns[-1])
    mask = (t_ns >= active_start + 30.0) & (t_ns <= active_end - 10.0)
    if np.sum(mask) < 8:
        return {"cycle_energy_pJ": float("nan"), "cycle_period_ns": float("nan"), "cycle_frequency_MHz": 0.0}
    t_eval = t_ns[mask]
    v_eval = out["V_vo2"][mask] * 1e3
    p_eval = out["P"][mask]
    crossings = _threshold_crossings(t_eval, v_eval)
    if crossings.size < 2:
        return {"cycle_energy_pJ": float("nan"), "cycle_period_ns": float("nan"), "cycle_frequency_MHz": 0.0}
    energies = []
    periods = []
    for start, stop in zip(crossings[:-1], crossings[1:]):
        if stop <= start + 2:
            continue
        t_s = t_eval[start : stop + 1] * 1e-9
        p_w = p_eval[start : stop + 1]
        energies.append(float(np.trapezoid(p_w, t_s) * 1e12))
        periods.append(float(t_eval[stop] - t_eval[start]))
    if not energies or not periods:
        return {"cycle_energy_pJ": float("nan"), "cycle_period_ns": float("nan"), "cycle_frequency_MHz": 0.0}
    period = float(np.mean(periods))
    return {
        "cycle_energy_pJ": float(np.mean(energies)),
        "cycle_period_ns": period,
        "cycle_frequency_MHz": 1000.0 / period if period > 0.0 else 0.0,
    }


def _regime_from_metrics(row: dict) -> str:
    if row["oscillatory"] >= 0.5:
        return "oscillatory"
    if row["R_plateau_ohm"] <= 650.0 and row["V_pp_active_mV"] < 50.0:
        return "metallic lock"
    if row["T_max_K"] >= 330.0 and row["R_min_ohm"] <= 900.0:
        return "transient switch"
    return "insulating"


def _simulate_and_summarize(
    *,
    currents_uA: list[int],
    params: CurrentDriveParams,
    min_cycles: int,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traces = simulate_current_steps([float(i) for i in currents_uA], params=params, seed=0)
    summary_rows = []
    trace_rows = []
    spectra_rows = []

    for i_uA, out in zip(currents_uA, traces):
        t_ns = out["t"] * 1e9
        i_trace_uA = out["I_in"] * 1e6
        v_mV = out["V_vo2"] * 1e3
        active = np.abs(i_trace_uA) > 0.0
        plateau = active & (t_ns >= 250.0) & (t_ns <= 430.0)
        eval_mask = plateau if np.any(plateau) else active
        if not np.any(eval_mask):
            eval_mask = np.ones_like(t_ns, dtype=bool)

        metrics = analyze_current_trace(
            out,
            params=params,
            min_vpp_mV=20.0,
            max_vpp_mV=6000.0,
            min_cycles=min_cycles,
            pulse_on_ns=float(params.pulse_on_s * 1e9),
            pulse_off_ns=None if params.pulse_off_s is None else float(params.pulse_off_s * 1e9),
        )
        energy = _cycle_energy_metrics(out, params)
        row = {
            "I_target_uA": float(i_uA),
            "I_avg_uA": float(np.mean(i_trace_uA[eval_mask])),
            "V_avg_mV": float(np.mean(v_mV[eval_mask])),
            "V_std_mV": float(np.std(v_mV[eval_mask])),
            "V_pp_mV": float(np.ptp(v_mV[eval_mask])),
            "V_pp_active_mV": float(np.ptp(v_mV[plateau])) if np.any(plateau) else 0.0,
            "turn_count": float(_count_turns(v_mV[eval_mask])),
            "input_power_dBm": _input_power_dbm(out["I_in"]),
            "dominant_freq_MHz": float(metrics["dominant_freq_MHz"]),
            "oscillatory": float(metrics["oscillatory"]),
            "n_cycles": float(metrics["n_cycles"]),
            "period_cv": float(metrics["period_cv"]),
            "spectral_purity": float(metrics["spectral_purity"]),
            "R_plateau_ohm": float(np.mean(out["R"][eval_mask])),
            "R_min_ohm": float(np.min(out["R"][eval_mask])),
            "R_max_ohm": float(np.max(out["R"][eval_mask])),
            "T_avg_K": float(np.mean(out["T"][eval_mask])),
            "T_max_K": float(np.max(out["T"][eval_mask])),
            "P_avg_uW": float(np.mean(out["P"][eval_mask]) * 1e6),
            "P_peak_uW": float(np.max(out["P"][eval_mask]) * 1e6),
            **energy,
        }
        row["regime"] = _regime_from_metrics(row)
        summary_rows.append(row)

        keep = _downsample_indices(t_ns.size)
        for idx in keep:
            trace_rows.append(
                {
                    "I_target_uA": float(i_uA),
                    "time_ns": float(t_ns[idx]),
                    "I_in_uA": float(i_trace_uA[idx]),
                    "V_vo2_mV": float(v_mV[idx]),
                    "T_K": float(out["T"][idx]),
                    "R_ohm": float(out["R"][idx]),
                    "P_W": float(out["P"][idx]),
                }
            )

        f_mhz, gain_linear, gain_db = _fft_gain_spectrum(out["t"], out["I_in"], out["V_vo2"])
        for f, gl, gd in zip(f_mhz, gain_linear, gain_db):
            spectra_rows.append(
                {
                    "I_target_uA": float(i_uA),
                    "input_power_dBm": row["input_power_dBm"],
                    "freq_MHz": float(f),
                    "gain_linear": float(gl),
                    "gain_dB": float(gd),
                }
            )

    return traces, pd.DataFrame(trace_rows), pd.DataFrame(summary_rows), pd.DataFrame(spectra_rows)


def _rt_branches(resist: YuanhangResistParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temps = np.linspace(resist.T_min_K, resist.T_max_K, 700, dtype=np.float32)
    heat = HysteresisArray(resist, size=1, start_branch="insulator")
    heat.initialize(np.asarray([temps[0]], dtype=np.float32))
    r_heat = np.asarray([float(heat.evaluate(np.asarray([t], dtype=np.float32))[0][0]) for t in temps])

    cool_t = temps[::-1]
    cool = HysteresisArray(resist, size=1, start_branch="metal")
    cool.initialize(np.asarray([cool_t[0]], dtype=np.float32))
    r_cool_desc = np.asarray([float(cool.evaluate(np.asarray([t], dtype=np.float32))[0][0]) for t in cool_t])
    return temps, r_heat, r_cool_desc[::-1]


def _plot_three_regimes(
    *,
    traces_by_current: dict[int, dict],
    resist: YuanhangResistParams,
    out_path: Path,
    currents: list[int],
) -> None:
    temps, r_heat, r_cool = _rt_branches(resist)
    labels = ["low/no IMT", "self-oscillatory", "high/metallic lock"]
    colors = ["#6b7280", "#7c3aed", "#ef4444"]
    all_v = np.concatenate([traces_by_current[i]["V_vo2"] * 1e3 for i in currents])
    all_i = np.concatenate([traces_by_current[i]["I_in"] * 1e6 for i in currents])
    fig, axes = plt.subplots(len(currents), 2, figsize=(12.5, 10.0), gridspec_kw={"width_ratios": [1.25, 1.0]})
    for row, (i_uA, label, color) in enumerate(zip(currents, labels, colors)):
        out = traces_by_current[i_uA]
        t_ns = out["t"] * 1e9
        ax = axes[row, 0]
        ax2 = ax.twinx()
        ax.plot(t_ns, out["I_in"] * 1e6, color="#16a34a", linewidth=1.7, label="current")
        ax2.plot(t_ns, out["V_vo2"] * 1e3, color="#7e22ce", linewidth=1.5, label="voltage")
        ax.set_xlim(0, 600)
        ax.set_ylim(float(np.min(all_i)) - 50, float(np.max(all_i)) + 150)
        ax2.set_ylim(max(0.0, float(np.min(all_v)) - 150), float(np.max(all_v)) + 250)
        ax.set_ylabel("I (uA)", color="#16a34a")
        ax2.set_ylabel("V (mV)", color="#7e22ce")
        ax.set_title(f"{label}: {i_uA} uA")
        ax.grid(True, alpha=0.25)
        if row == len(currents) - 1:
            ax.set_xlabel("time (ns)")

        rt = axes[row, 1]
        rt.plot(temps, r_cool, color="#3b82f6", linewidth=2.0, label="cooling")
        rt.plot(temps, r_heat, color="#ef4444", linewidth=2.0, label="heating")
        rt.plot(out["T"], out["R"], color=color, alpha=0.60, linewidth=1.4)
        rt.scatter(out["T"][-1], out["R"][-1], color=color, s=42, zorder=5)
        rt.set_yscale("log")
        rt.set_xlim(temps[0], temps[-1])
        rt.set_ylim(10, 1.2 * max(np.nanmax(r_heat), np.nanmax(r_cool)))
        rt.grid(True, alpha=0.25)
        rt.set_ylabel("R (Ohm)")
        if row == 0:
            rt.legend(loc="upper right")
        if row == len(currents) - 1:
            rt.set_xlabel("temperature (K)")
    fig.suptitle("Paper-style current pulse regimes in the Yuanhang current-source model", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_operating_window(summary: pd.DataFrame, out_path: Path, *, title: str) -> None:
    d = summary.sort_values("I_target_uA")
    colors = {
        "insulating": "#6b7280",
        "transient switch": "#f59e0b",
        "oscillatory": "#7c3aed",
        "metallic lock": "#ef4444",
    }
    c = [colors.get(str(r), "#111827") for r in d["regime"]]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True)
    axes[0, 0].scatter(d["I_target_uA"], d["V_pp_active_mV"], c=c, s=48)
    axes[0, 0].set_ylabel("late active Vpp (mV)")
    axes[0, 0].set_title("oscillation amplitude")
    axes[0, 1].scatter(d["I_target_uA"], d["dominant_freq_MHz"], c=c, s=48)
    axes[0, 1].axhspan(PAPER_FREQ_WINDOW_MHZ[0], PAPER_FREQ_WINDOW_MHZ[1], color="#22c55e", alpha=0.13)
    axes[0, 1].set_ylabel("dominant frequency (MHz)")
    axes[0, 1].set_title("frequency window")
    axes[1, 0].scatter(d["I_target_uA"], d["cycle_energy_pJ"], c=c, s=48)
    axes[1, 0].set_ylabel("cycle energy (pJ)")
    axes[1, 0].set_xlabel("model current (uA)")
    axes[1, 0].set_title("energy per oscillation")
    z_eff = 1e3 * d["V_avg_mV"] / d["I_avg_uA"].replace(0.0, np.nan)
    axes[1, 1].scatter(d["I_target_uA"], z_eff, c=c, s=48)
    axes[1, 1].set_ylabel("Vavg/Iavg (Ohm)")
    axes[1, 1].set_xlabel("model current (uA)")
    axes[1, 1].set_title("effective transimpedance")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=8)
        for k, v in colors.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(title, y=1.04, fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_energy_cycle(out: dict, params: CurrentDriveParams, out_path: Path, *, current_uA: int) -> None:
    t_ns = out["t"] * 1e9
    p_uW = out["P"] * 1e6
    active_start = params.pulse_on_s * 1e9 + 30.0
    active_end = (params.pulse_off_s or params.t_end_s) * 1e9 - 10.0
    mask = (t_ns >= active_start) & (t_ns <= active_end)
    crossings = _threshold_crossings(t_ns[mask], out["V_vo2"][mask] * 1e3)
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    ax.plot(t_ns, p_uW, color="#7c3aed", linewidth=1.6)
    ax.set_xlim(120, 470)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("P = V I (uW)")
    ax.set_title(f"Energy per oscillation from simulated power trace ({current_uA} uA)")
    if crossings.size >= 4:
        full_idx = np.flatnonzero(mask)
        start = full_idx[int(crossings[1])]
        stop = full_idx[int(crossings[2])]
        energy_pJ = float(np.trapezoid(out["P"][start : stop + 1], out["t"][start : stop + 1]) * 1e12)
        ax.axvspan(t_ns[start], t_ns[stop], color="#f59e0b", alpha=0.25, label=f"one cycle: {energy_pJ:.2f} pJ")
        ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_pump_probe(params: CurrentDriveParams, out_path: Path) -> pd.DataFrame:
    delays = [50.0, 100.0, 200.0, 500.0]
    rows = []
    fig, axes = plt.subplots(len(delays), 1, figsize=(9.5, 8.2), sharex=True)
    for ax, delay in zip(axes, delays):
        t_grid = np.arange(0.0, 760e-9 + params.dt_s, params.dt_s)
        i_uA = np.zeros_like(t_grid) * 1.0
        pump_on = 100e-9
        pump_off = 150e-9
        probe_on = pump_off + delay * 1e-9
        probe_off = probe_on + 60e-9
        i_uA[(t_grid >= pump_on) & (t_grid < pump_off)] = 2000.0
        i_uA[(t_grid >= probe_on) & (t_grid < probe_off)] = 1250.0
        pp_params = CurrentDriveParams(
            **{
                **_as_config_params(params),
                "t_end_s": float(t_grid[-1]),
                "pulse_on_s": 0.0,
                "pulse_off_s": None,
                "resist_params": params.resist_params,
            }
        )
        out = simulate_current_waveform(i_uA, params=pp_params, waveform_time_s=t_grid, seed=0)
        t_ns = out["t"] * 1e9
        ax2 = ax.twinx()
        ax.plot(t_ns, out["T"], color="#ef4444", linewidth=1.5)
        ax2.plot(t_ns, out["V_vo2"] * 1e3, color="#7e22ce", linewidth=1.0, alpha=0.8)
        ax.axvspan(100, 150, color="#111827", alpha=0.08)
        ax.axvspan(probe_on * 1e9, probe_off * 1e9, color="#16a34a", alpha=0.10)
        ax.set_ylabel("T (K)")
        ax2.set_ylabel("V (mV)")
        ax.set_title(f"pump-probe delay = {delay:.0f} ns")
        probe_idx = int(np.searchsorted(t_ns, probe_on * 1e9))
        rows.append(
            {
                "delay_ns": delay,
                "T_at_probe_K": float(out["T"][probe_idx]),
                "R_at_probe_ohm": float(out["R"][probe_idx]),
                "V_probe_peak_mV": float(np.max(out["V_vo2"][(t_ns >= probe_on * 1e9) & (t_ns <= probe_off * 1e9)]) * 1e3),
            }
        )
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time (ns)")
    fig.suptitle("Deterministic pump-probe recovery in the current-source model", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows)


def _make_sweep_gif(
    *,
    traces_by_current: dict[int, dict],
    resist: YuanhangResistParams,
    summary: pd.DataFrame,
    out_path: Path,
    frame_dir: Path,
    duration_s: float,
) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    temps, r_heat, r_cool = _rt_branches(resist)
    currents = sorted(traces_by_current)
    all_v = np.concatenate([traces_by_current[i]["V_vo2"] * 1e3 for i in currents])
    all_i = np.concatenate([traces_by_current[i]["I_in"] * 1e6 for i in currents])
    frames = []
    regime_by_current = dict(zip(summary["I_target_uA"].astype(int), summary["regime"]))
    freq_by_current = dict(zip(summary["I_target_uA"].astype(int), summary["dominant_freq_MHz"]))
    for idx, i_uA in enumerate(currents):
        out = traces_by_current[i_uA]
        t_ns = out["t"] * 1e9
        fig, (ax_trace, ax_rt) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1.0]})
        ax_v = ax_trace.twinx()
        ax_trace.plot(t_ns, out["I_in"] * 1e6, color="#16a34a", linewidth=1.8)
        ax_v.plot(t_ns, out["V_vo2"] * 1e3, color="#7e22ce", linewidth=1.5)
        ax_trace.set_xlim(0, 600)
        ax_trace.set_ylim(float(np.min(all_i)) - 50, float(np.max(all_i)) + 150)
        ax_v.set_ylim(max(0.0, float(np.min(all_v)) - 150), float(np.max(all_v)) + 250)
        ax_trace.set_xlabel("time (ns)")
        ax_trace.set_ylabel("I (uA)", color="#16a34a")
        ax_v.set_ylabel("V (mV)", color="#7e22ce")
        freq = freq_by_current.get(i_uA, 0.0)
        ax_trace.set_title(f"{i_uA} uA | {regime_by_current.get(i_uA, '')} | {freq:.1f} MHz")
        ax_trace.grid(True, alpha=0.25)

        ax_rt.plot(temps, r_cool, color="#3b82f6", linewidth=2.0, label="cooling")
        ax_rt.plot(temps, r_heat, color="#ef4444", linewidth=2.0, label="heating")
        ax_rt.plot(out["T"], out["R"], color="#7c3aed", alpha=0.55, linewidth=1.4)
        ax_rt.scatter(out["T"][-1], out["R"][-1], color="#dc2626", s=44, zorder=5)
        ax_rt.set_yscale("log")
        ax_rt.set_xlim(temps[0], temps[-1])
        ax_rt.set_ylim(10, 1.2 * max(np.nanmax(r_heat), np.nanmax(r_cool)))
        ax_rt.set_xlabel("temperature (K)")
        ax_rt.set_ylabel("R (Ohm)")
        ax_rt.legend(loc="upper right")
        ax_rt.grid(True, alpha=0.25)
        fig.tight_layout()
        frame = frame_dir / f"frame_{idx:03d}_{i_uA:04d}uA.png"
        fig.savefig(frame, dpi=150)
        plt.close(fig)
        frames.append(frame)
    imageio.mimsave(out_path, [imageio.imread(p) for p in frames], duration=float(duration_s), loop=0)


def _make_time_evolution_gif(
    *,
    out: dict,
    resist: YuanhangResistParams,
    out_path: Path,
    frame_dir: Path,
    total_duration_s: float,
    current_uA: int,
) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    temps, r_heat, r_cool = _rt_branches(resist)
    t_ns = out["t"] * 1e9
    v_mV = out["V_vo2"] * 1e3
    frames = []
    frame_count = 80
    indices = np.linspace(3, len(t_ns) - 1, frame_count, dtype=int)
    for frame_idx, idx in enumerate(indices):
        fig, (ax_trace, ax_rt) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1.0]})
        ax_i = ax_trace
        ax_v = ax_trace.twinx()
        ax_i.plot(t_ns[: idx + 1], out["I_in"][: idx + 1] * 1e6, color="#16a34a", linewidth=1.8)
        ax_v.plot(t_ns[: idx + 1], v_mV[: idx + 1], color="#7e22ce", linewidth=1.5)
        ax_i.set_xlim(0, 600)
        ax_i.set_ylim(-50, max(100.0, float(np.max(out["I_in"] * 1e6)) + 150))
        ax_v.set_ylim(max(0.0, float(np.min(v_mV)) - 150), float(np.max(v_mV)) + 250)
        ax_i.set_xlabel("time (ns)")
        ax_i.set_ylabel("I (uA)", color="#16a34a")
        ax_v.set_ylabel("V (mV)", color="#7e22ce")
        ax_i.set_title(f"time evolution at {current_uA} uA")
        ax_i.grid(True, alpha=0.25)

        ax_rt.plot(temps, r_cool, color="#3b82f6", linewidth=2.0)
        ax_rt.plot(temps, r_heat, color="#ef4444", linewidth=2.0)
        ax_rt.plot(out["T"][: idx + 1], out["R"][: idx + 1], color="#7c3aed", alpha=0.55, linewidth=1.3)
        ax_rt.scatter(out["T"][idx], out["R"][idx], color="#dc2626", s=48, zorder=5)
        ax_rt.set_yscale("log")
        ax_rt.set_xlim(temps[0], temps[-1])
        ax_rt.set_ylim(10, 1.2 * max(np.nanmax(r_heat), np.nanmax(r_cool)))
        ax_rt.set_xlabel("temperature (K)")
        ax_rt.set_ylabel("R (Ohm)")
        ax_rt.grid(True, alpha=0.25)
        fig.tight_layout()
        frame = frame_dir / f"frame_{frame_idx:03d}.png"
        fig.savefig(frame, dpi=150)
        plt.close(fig)
        frames.append(frame)
    imageio.mimsave(out_path, [imageio.imread(p) for p in frames], duration=total_duration_s / frame_count, loop=0)


def _write_public_job(
    *,
    job_dir: Path,
    name: str,
    params: CurrentDriveParams,
    currents: list[int],
    traces_csv: Path,
    summary_csv: Path,
    spectra_csv: Path,
    gif_path: Path,
    extra_outputs: list[tuple[str, Path]],
    conclusion: str,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "type": "current_sweep",
        "job_name": name,
        "job_storage": "public",
        "sample_id": "100425_chip1_gap3",
        "sample_name": "100425 chip1 gap3",
        "sample_source": _repo_path(PRESET_PATH),
        "source_model": "Current Source",
        "I_start_uA": int(min(currents)),
        "I_stop_uA": int(max(currents)),
        "I_step_uA": int(np.gcd.reduce(np.diff(sorted(currents)))) if len(currents) > 1 else 0,
        "frame_duration_s": 1.0,
        "seed": 0,
        "current_params": _as_config_params(params),
        "conclusion": conclusion,
    }
    outputs = [
        {"label": "Current sweep GIF", "path": _repo_path(gif_path)},
        {"label": "Current sweep traces CSV", "path": _repo_path(traces_csv)},
        {"label": "Current sweep summary CSV", "path": _repo_path(summary_csv)},
        {"label": "Current sweep spectra CSV", "path": _repo_path(spectra_csv)},
    ]
    outputs.extend({"label": label, "path": _repo_path(path)} for label, path in extra_outputs)
    job = {
        "id": job_dir.name,
        "name": name,
        "job_storage": "public",
        "type": "current_sweep",
        "sample_id": "100425_chip1_gap3",
        "sample_name": "100425 chip1 gap3",
        "source_model": "Current Source",
        "status": "completed",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": config,
        "outputs": outputs,
        "log_path": _repo_path(job_dir / "log.txt"),
    }
    (job_dir / "job.json").write_text(json.dumps(job, indent=2))
    (job_dir / "log.txt").write_text(
        "[job] generated by scripts/generate_professor_paper_simulations.py\n"
        "[job] completed professor paper current-source package\n"
    )


def _write_report(
    out_dir: Path,
    *,
    analog_summary: pd.DataFrame,
    sample_summary: pd.DataFrame,
    analog_job: Path,
    sample_job: Path,
) -> Path:
    analog_osc = analog_summary[analog_summary["oscillatory"] >= 0.5]
    sample_osc = sample_summary[sample_summary["oscillatory"] >= 0.5]
    lines = [
        "# Professor paper simulation package",
        "",
        "Generated with the hardcoded Yuanhang ideal-current-source model.",
        "",
        "## What was run",
        "",
        "- Paper-frequency analog: one VO2 domain, ideal current source, fitted sample R(T), `C = 4 pF`, `C_th = 1 pJ/K`, `S_e = 0.10 mW/K`, 300 ns current pulse from 150 ns to 450 ns.",
        "- Validated sample-scale control: same equations and R(T), `C = 25.930953 pF`, `C_th = 5 pJ/K`, `S_e = 0.10 mW/K`, same pulse timing.",
        "- No dynamic/quasistatic/multidomain modes were used.",
        "",
        "## Main result",
        "",
    ]
    if not analog_osc.empty:
        lines.append(
            f"- Paper-frequency analog oscillatory window: {analog_osc['I_target_uA'].min():.0f}-"
            f"{analog_osc['I_target_uA'].max():.0f} uA, "
            f"{analog_osc['dominant_freq_MHz'].min():.1f}-"
            f"{analog_osc['dominant_freq_MHz'].max():.1f} MHz."
        )
        analog_energy = analog_osc["cycle_energy_pJ"].dropna()
        if not analog_energy.empty:
            lines.append(
                f"- Paper-frequency analog full-cycle energy: {analog_energy.min():.1f}-"
                f"{analog_energy.max():.1f} pJ. This is larger than the paper's reported "
                "1-2 pJ scale, so the current package supports the oscillatory mechanism and "
                "frequency phenomenology, not an absolute energy calibration."
            )
    if not sample_osc.empty:
        lines.append(
            f"- Validated sample-scale oscillatory window: {sample_osc['I_target_uA'].min():.0f}-"
            f"{sample_osc['I_target_uA'].max():.0f} uA, "
            f"{sample_osc['dominant_freq_MHz'].min():.1f}-"
            f"{sample_osc['dominant_freq_MHz'].max():.1f} MHz."
        )
    lines.extend(
        [
            "",
            "The paper-frequency analog matches the paper's frequency scale, but not its absolute current scale.",
            "That means the clean presentation is a normalized/phenomenological comparison: same low-current, oscillatory, and high-current regimes, with the same hidden R(T) relaxation mechanism.",
            "",
            "## App history jobs",
            "",
            f"- Paper-frequency analog job: `{analog_job.name}`",
            f"- Validated sample-scale job: `{sample_job.name}`",
            "",
            "Open Streamlit History, filter to Public, and open those jobs.",
        ]
    )
    report_path = out_dir / "professor_paper_simulation_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def _copy_outputs_to_job(job_dir: Path, files: Iterable[Path]) -> list[Path]:
    copied = []
    for src in files:
        dst = job_dir / src.name
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
        copied.append(dst)
    return copied


def _copy_output_to_job(job_dir: Path, src: Path, filename: str) -> Path:
    dst = job_dir / filename
    if src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())
    return dst


def run(output_dir: Path) -> dict[str, Path]:
    model._TORCH_HYSTERESIS_AVAILABLE = False
    resist = _load_resistance_preset()
    output_dir.mkdir(parents=True, exist_ok=True)

    analog_params = _paper_style_params(resist)
    sample_params = _validated_sample_params(resist)
    analog_currents = list(range(0, 3001, 100))
    sample_currents = list(range(0, 3001, 100))

    analog_traces, analog_trace_df, analog_summary, analog_spectra = _simulate_and_summarize(
        currents_uA=analog_currents,
        params=analog_params,
        min_cycles=6,
    )
    sample_traces, sample_trace_df, sample_summary, sample_spectra = _simulate_and_summarize(
        currents_uA=sample_currents,
        params=sample_params,
        min_cycles=3,
    )
    analog_by_current = dict(zip(analog_currents, analog_traces))
    sample_by_current = dict(zip(sample_currents, sample_traces))

    analog_summary.to_csv(output_dir / "paper_frequency_summary.csv", index=False)
    analog_trace_df.to_csv(output_dir / "paper_frequency_traces_downsampled.csv", index=False)
    analog_spectra.to_csv(output_dir / "paper_frequency_spectra.csv", index=False)
    sample_summary.to_csv(output_dir / "validated_sample_summary.csv", index=False)
    sample_trace_df.to_csv(output_dir / "validated_sample_traces_downsampled.csv", index=False)
    sample_spectra.to_csv(output_dir / "validated_sample_spectra.csv", index=False)

    _plot_three_regimes(
        traces_by_current=analog_by_current,
        resist=resist,
        out_path=output_dir / "paper_frequency_three_regimes.png",
        currents=[1000, 1750 if 1750 in analog_by_current else 1700, 3000],
    )
    _plot_operating_window(
        analog_summary,
        output_dir / "paper_frequency_operating_window.png",
        title="Paper-frequency analog: operating window",
    )
    _plot_operating_window(
        sample_summary,
        output_dir / "validated_sample_operating_window.png",
        title="Validated sample-scale control: operating window",
    )
    _plot_energy_cycle(
        analog_by_current[1800],
        analog_params,
        output_dir / "paper_frequency_energy_cycle.png",
        current_uA=1800,
    )
    pump_probe_df = _plot_pump_probe(analog_params, output_dir / "paper_frequency_pump_probe_recovery.png")
    pump_probe_df.to_csv(output_dir / "paper_frequency_pump_probe_recovery.csv", index=False)

    _make_sweep_gif(
        traces_by_current={i: analog_by_current[i] for i in range(0, 3001, 100)},
        resist=resist,
        summary=analog_summary,
        out_path=output_dir / "paper_frequency_current_sweep_rt.gif",
        frame_dir=output_dir / "paper_frequency_sweep_frames",
        duration_s=1.0,
    )
    _make_time_evolution_gif(
        out=analog_by_current[1800],
        resist=resist,
        out_path=output_dir / "paper_frequency_1800uA_time_evolution_10s.gif",
        frame_dir=output_dir / "paper_frequency_time_evolution_frames",
        total_duration_s=10.0,
        current_uA=1800,
    )

    analog_job = PUBLIC_JOBS / _job_id("paper_frequency")
    sample_job = PUBLIC_JOBS / _job_id("sample_scale")
    analog_job.mkdir(parents=True, exist_ok=True)
    sample_job.mkdir(parents=True, exist_ok=True)

    analog_gif = _copy_output_to_job(analog_job, output_dir / "paper_frequency_current_sweep_rt.gif", "current_sweep.gif")
    analog_traces_csv = _copy_output_to_job(
        analog_job,
        output_dir / "paper_frequency_traces_downsampled.csv",
        "current_sweep_traces.csv",
    )
    analog_summary_csv = _copy_output_to_job(
        analog_job,
        output_dir / "paper_frequency_summary.csv",
        "current_sweep_summary.csv",
    )
    analog_spectra_csv = _copy_output_to_job(
        analog_job,
        output_dir / "paper_frequency_spectra.csv",
        "current_sweep_spectra.csv",
    )
    analog_job_files = _copy_outputs_to_job(
        analog_job,
        [
            output_dir / "paper_frequency_three_regimes.png",
            output_dir / "paper_frequency_operating_window.png",
            output_dir / "paper_frequency_energy_cycle.png",
            output_dir / "paper_frequency_pump_probe_recovery.png",
            output_dir / "paper_frequency_pump_probe_recovery.csv",
            output_dir / "paper_frequency_1800uA_time_evolution_10s.gif",
        ],
    )
    sample_traces_csv = _copy_output_to_job(
        sample_job,
        output_dir / "validated_sample_traces_downsampled.csv",
        "current_sweep_traces.csv",
    )
    sample_summary_csv = _copy_output_to_job(
        sample_job,
        output_dir / "validated_sample_summary.csv",
        "current_sweep_summary.csv",
    )
    sample_spectra_csv = _copy_output_to_job(
        sample_job,
        output_dir / "validated_sample_spectra.csv",
        "current_sweep_spectra.csv",
    )
    sample_job_files = _copy_outputs_to_job(
        sample_job,
        [output_dir / "validated_sample_operating_window.png"],
    )
    sample_gif = sample_job / "validated_sample_current_sweep_rt.gif"
    _make_sweep_gif(
        traces_by_current=sample_by_current,
        resist=resist,
        summary=sample_summary,
        out_path=sample_gif,
        frame_dir=sample_job / "validated_sample_sweep_frames",
        duration_s=1.0,
    )

    analog_paths = {p.name: p for p in analog_job_files}
    sample_paths = {p.name: p for p in sample_job_files}
    _write_public_job(
        job_dir=analog_job,
        name="Professor paper package - paper-frequency analog",
        params=analog_params,
        currents=analog_currents,
        traces_csv=analog_traces_csv,
        summary_csv=analog_summary_csv,
        spectra_csv=analog_spectra_csv,
        gif_path=analog_gif,
        extra_outputs=[
            ("Three-regime figure", analog_paths["paper_frequency_three_regimes.png"]),
            ("Operating-window figure", analog_paths["paper_frequency_operating_window.png"]),
            ("Energy-per-cycle figure", analog_paths["paper_frequency_energy_cycle.png"]),
            ("Pump-probe recovery figure", analog_paths["paper_frequency_pump_probe_recovery.png"]),
            ("Pump-probe recovery CSV", analog_paths["paper_frequency_pump_probe_recovery.csv"]),
            ("10s time-evolution GIF", analog_paths["paper_frequency_1800uA_time_evolution_10s.gif"]),
        ],
        conclusion=(
            "Paper-frequency phenomenology: this run reaches the 40-60 MHz oscillatory scale "
            "with the same Yuanhang ideal-current-source equations by using a smaller physical "
            "electrical capacitance and thermal mass. The absolute current window remains model/sample-scaled."
        ),
    )
    _write_public_job(
        job_dir=sample_job,
        name="Professor paper package - validated sample-scale control",
        params=sample_params,
        currents=sample_currents,
        traces_csv=sample_traces_csv,
        summary_csv=sample_summary_csv,
        spectra_csv=sample_spectra_csv,
        gif_path=sample_gif,
        extra_outputs=[("Operating-window figure", sample_paths["validated_sample_operating_window.png"])],
        conclusion=(
            "Validated sample-scale control using the same hardcoded Yuanhang current-source model "
            "and the specimen R(T) fit. This preserves the original stable simulation parameters and shows "
            "the same three-regime mechanism at a slower frequency scale."
        ),
    )

    report_path = _write_report(
        output_dir,
        analog_summary=analog_summary,
        sample_summary=sample_summary,
        analog_job=analog_job,
        sample_job=sample_job,
    )
    (analog_job / report_path.name).write_bytes(report_path.read_bytes())
    (sample_job / report_path.name).write_bytes(report_path.read_bytes())

    return {
        "output_dir": output_dir,
        "analog_job": analog_job,
        "sample_job": sample_job,
        "report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate professor-paper current-source simulation package.")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to outputs/professor_paper_simulations_<timestamp>.")
    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUTS / f"professor_paper_simulations_{time.strftime('%Y%m%d_%H%M%S')}"
    result = run(out_dir)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
