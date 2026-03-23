from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step, simulate_current_waveform, stabilize_current_drive_params
from neuristor.current_results_digitizer import count_turns, digitize_directory, reconstruct_current_waveform, smooth_trace
from neuristor.model import YuanhangResistParams


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit current-result images with fixed hysteresis law and free dynamic parameters.")
    parser.add_argument("--image-dir", default="data/Current Results")
    parser.add_argument("--resistance-preset", default="presets/resistance_100425_chip1_gap3.json")
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    parser.add_argument("--forcing-mode", choices=("ideal_step", "digitized_iin"), default="ideal_step")
    parser.add_argument("--fit-pulse-law", action="store_true", help="Also fit a limited pulse-side Tc shift and beta scale.")
    parser.add_argument(
        "--fit-indices",
        default="0,1,2,5,18,35",
        help="Comma-separated frame indices to use during fitting.",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser


def _parse_fit_indices(text: str) -> list[int]:
    vals = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not vals:
        raise ValueError("fit_indices cannot be empty")
    if len(set(vals)) != len(vals):
        raise ValueError("fit_indices must not contain duplicates")
    return vals


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _make_candidate(
    *,
    C_pF: float,
    C_th_mW_ns_per_K: float,
    S_e_mW_per_K: float,
    T0_K: float,
    T_init_K: float,
    Tc_shift_K: float = 0.0,
    beta_scale: float = 1.0,
    R0_scale: float = 1.0,
    Ea_scale: float = 1.0,
) -> dict[str, float]:
    return {
        "C_pF": float(C_pF),
        "C_th_mW_ns_per_K": float(C_th_mW_ns_per_K),
        "S_e_mW_per_K": float(S_e_mW_per_K),
        "T0_K": float(T0_K),
        "T_init_K": float(T_init_K),
        "Tc_shift_K": float(Tc_shift_K),
        "beta_scale": float(beta_scale),
        "R0_scale": float(R0_scale),
        "Ea_scale": float(Ea_scale),
    }


def _seed_candidates(*, fit_pulse_law: bool) -> list[dict[str, float]]:
    return [
        _make_candidate(
            C_pF=145.34619293,
            C_th_mW_ns_per_K=49.62776831,
            S_e_mW_per_K=0.20558726,
            T0_K=325.0,
            T_init_K=324.9,
        ),
        _make_candidate(C_pF=17.0, C_th_mW_ns_per_K=8.0, S_e_mW_per_K=0.20558726, T0_K=325.0, T_init_K=324.9),
        _make_candidate(C_pF=17.0, C_th_mW_ns_per_K=125.0, S_e_mW_per_K=0.20558726, T0_K=325.0, T_init_K=324.9),
        _make_candidate(C_pF=25.0, C_th_mW_ns_per_K=80.0, S_e_mW_per_K=0.4, T0_K=330.0, T_init_K=329.6),
        _make_candidate(C_pF=25.0, C_th_mW_ns_per_K=120.0, S_e_mW_per_K=0.8, T0_K=331.0, T_init_K=330.4),
        _make_candidate(C_pF=35.0, C_th_mW_ns_per_K=60.0, S_e_mW_per_K=0.6, T0_K=329.0, T_init_K=328.5),
        _make_candidate(C_pF=20.0, C_th_mW_ns_per_K=40.0, S_e_mW_per_K=1.2, T0_K=332.0, T_init_K=331.2),
        _make_candidate(
            C_pF=22.0,
            C_th_mW_ns_per_K=12.0,
            S_e_mW_per_K=1.5,
            T0_K=333.0,
            T_init_K=331.8,
            Tc_shift_K=-4.0 if fit_pulse_law else 0.0,
            beta_scale=1.6 if fit_pulse_law else 1.0,
            R0_scale=2.5 if fit_pulse_law else 1.0,
            Ea_scale=1.05 if fit_pulse_law else 1.0,
        ),
        _make_candidate(
            C_pF=28.0,
            C_th_mW_ns_per_K=12.0,
            S_e_mW_per_K=1.8,
            T0_K=334.0,
            T_init_K=332.6,
            Tc_shift_K=-8.0 if fit_pulse_law else 0.0,
            beta_scale=2.2 if fit_pulse_law else 1.0,
            R0_scale=4.0 if fit_pulse_law else 1.0,
            Ea_scale=1.10 if fit_pulse_law else 1.0,
        ),
        _make_candidate(
            C_pF=80.0,
            C_th_mW_ns_per_K=14.0,
            S_e_mW_per_K=0.8,
            T0_K=334.5,
            T_init_K=333.5,
            Tc_shift_K=-7.5 if fit_pulse_law else 0.0,
            beta_scale=0.6 if fit_pulse_law else 1.0,
            R0_scale=3.0 if fit_pulse_law else 1.0,
            Ea_scale=1.08 if fit_pulse_law else 1.0,
        ),
    ]


def _random_candidate(rng: np.random.Generator, *, fit_pulse_law: bool) -> dict[str, float]:
    T0_K = float(rng.uniform(320.0, 338.0))
    delta_init = float(rng.uniform(0.0, 2.5))
    return _make_candidate(
        C_pF=_loguniform(rng, 10.0, 150.0),
        C_th_mW_ns_per_K=_loguniform(rng, 5.0, 200.0),
        S_e_mW_per_K=_loguniform(rng, 0.05, 2.0),
        T0_K=T0_K,
        T_init_K=T0_K - delta_init,
        Tc_shift_K=float(rng.uniform(-12.0, 6.0)) if fit_pulse_law else 0.0,
        beta_scale=_loguniform(rng, 0.5, 3.0) if fit_pulse_law else 1.0,
        R0_scale=_loguniform(rng, 0.3, 12.0) if fit_pulse_law else 1.0,
        Ea_scale=_loguniform(rng, 0.7, 1.45) if fit_pulse_law else 1.0,
    )


def _perturb_candidate(base: dict[str, float], rng: np.random.Generator, *, fit_pulse_law: bool) -> dict[str, float]:
    T0_K = float(base["T0_K"] + rng.uniform(-1.5, 1.5))
    delta_init = max(0.0, float(base["T0_K"] - base["T_init_K"]) + rng.uniform(-0.5, 0.5))
    return _make_candidate(
        C_pF=min(180.0, max(8.0, float(base["C_pF"]) * np.exp(rng.uniform(np.log(0.75), np.log(1.35))))),
        C_th_mW_ns_per_K=min(
            250.0,
            max(3.0, float(base["C_th_mW_ns_per_K"]) * np.exp(rng.uniform(np.log(0.7), np.log(1.4)))),
        ),
        S_e_mW_per_K=min(3.0, max(0.02, float(base["S_e_mW_per_K"]) * np.exp(rng.uniform(np.log(0.65), np.log(1.5))))),
        T0_K=min(340.0, max(318.0, T0_K)),
        T_init_K=min(340.0, max(318.0, T0_K - delta_init)),
        Tc_shift_K=min(8.0, max(-15.0, float(base.get("Tc_shift_K", 0.0)) + (rng.uniform(-2.0, 2.0) if fit_pulse_law else 0.0))),
        beta_scale=min(
            4.0,
            max(
                0.35,
                float(base.get("beta_scale", 1.0))
                * np.exp(rng.uniform(np.log(0.75), np.log(1.35)) if fit_pulse_law else 0.0),
            ),
        ),
        R0_scale=min(
            20.0,
            max(
                0.15,
                float(base.get("R0_scale", 1.0))
                * np.exp(rng.uniform(np.log(0.7), np.log(1.45)) if fit_pulse_law else 0.0),
            ),
        ),
        Ea_scale=min(
            1.8,
            max(
                0.55,
                float(base.get("Ea_scale", 1.0))
                * np.exp(rng.uniform(np.log(0.9), np.log(1.1)) if fit_pulse_law else 0.0),
            ),
        ),
    )


def _apply_pulse_resist_adjustment(base_params: YuanhangResistParams, candidate: dict[str, float]) -> YuanhangResistParams:
    return replace(
        base_params,
        Tc_K=float(base_params.Tc_K + float(candidate.get("Tc_shift_K", 0.0))),
        beta=float(base_params.beta * float(candidate.get("beta_scale", 1.0))),
        R0=float(base_params.R0 * float(candidate.get("R0_scale", 1.0))),
        Ea_over_k=float(base_params.Ea_over_k * float(candidate.get("Ea_scale", 1.0))),
    )


def _trace_metrics(time_ns: np.ndarray, v_mV: np.ndarray) -> dict[str, float]:
    onset = (time_ns >= 0.0) & (time_ns <= 30.0)
    plateau = (time_ns >= 100.0) & (time_ns <= 250.0)
    peak = (time_ns >= 0.0) & (time_ns <= 80.0)
    turnoff = (time_ns >= 300.0) & (time_ns <= 450.0)
    v_smooth = smooth_trace(v_mV, window=5)
    return {
        "slope_0_30_mV_per_ns": float(np.polyfit(time_ns[onset], v_smooth[onset], 1)[0]) if np.sum(onset) >= 3 else float("nan"),
        "plateau_mean_mV": float(np.mean(v_smooth[plateau])) if np.any(plateau) else float("nan"),
        "plateau_vpp_mV": float(np.ptp(v_smooth[plateau])) if np.any(plateau) else float("nan"),
        "peak_0_80_mV": float(np.max(v_smooth[peak])) if np.any(peak) else float("nan"),
        "turnoff_min_300_450_mV": float(np.min(v_smooth[turnoff])) if np.any(turnoff) else float("nan"),
        "turn_count_100_250": float(count_turns(v_smooth[plateau])) if np.any(plateau) else 0.0,
    }


def _simulate_trace(
    current_uA: float,
    forcing_time_ns: np.ndarray | None,
    forcing_i_uA: np.ndarray | None,
    base_resist_params: YuanhangResistParams,
    *,
    C_pF: float,
    C_th_mW_ns_per_K: float,
    S_e_mW_per_K: float,
    T0_K: float,
    T_init_K: float,
    Tc_shift_K: float,
    beta_scale: float,
    R0_scale: float,
    Ea_scale: float,
    dt_s: float,
    i_peak_uA: float,
    forcing_mode: str,
    seed: int,
) -> dict[str, np.ndarray]:
    resist_params = _apply_pulse_resist_adjustment(
        base_resist_params,
        {
            "Tc_shift_K": float(Tc_shift_K),
            "beta_scale": float(beta_scale),
            "R0_scale": float(R0_scale),
            "Ea_scale": float(Ea_scale),
        },
    )
    params = CurrentDriveParams(
        dt_s=dt_s,
        t_end_s=600e-9,
        t_pre_s=200e-9,
        pulse_on_s=0.0,
        pulse_off_s=300e-9,
        V_init_V=0.0,
        T0_K=float(T0_K),
        T_init_K=float(T_init_K),
        C_F=float(C_pF) * 1e-12,
        C_th_J_per_K=float(C_th_mW_ns_per_K) * 1e-12,
        S_e_W_per_K=float(S_e_mW_per_K) * 1e-3,
        sigma_W_sqrt_s=0.0,
        resist_params=resist_params,
        start_branch="insulator",
    )
    params, _ = stabilize_current_drive_params(params, I_peak_uA=float(i_peak_uA))
    if forcing_mode == "digitized_iin":
        if forcing_time_ns is None or forcing_i_uA is None:
            raise ValueError("digitized_iin forcing requires forcing_time_ns and forcing_i_uA.")
        return simulate_current_waveform(
            np.asarray(forcing_i_uA, dtype=float),
            params=params,
            waveform_time_s=np.asarray(forcing_time_ns, dtype=float) * 1e-9,
            seed=seed,
        )
    return simulate_current_step(float(current_uA), params=params, seed=seed)


def _trace_score(
    obs_t_ns: np.ndarray,
    obs_v_mV: np.ndarray,
    sim_t_ns: np.ndarray,
    sim_v_mV: np.ndarray,
    *,
    emphasize_slope: bool,
) -> tuple[float, dict[str, float]]:
    obs_metrics = _trace_metrics(obs_t_ns, obs_v_mV)
    sim_interp = np.interp(obs_t_ns, sim_t_ns, sim_v_mV)
    sim_metrics = _trace_metrics(obs_t_ns, sim_interp)

    compare_mask = (obs_t_ns >= -20.0) & (obs_t_ns <= 450.0)
    onset_mask = (obs_t_ns >= -20.0) & (obs_t_ns <= 80.0)
    turnoff_mask = (obs_t_ns >= 260.0) & (obs_t_ns <= 420.0)
    obs_ds = smooth_trace(obs_v_mV[compare_mask], window=5)[::8]
    sim_ds = smooth_trace(sim_interp[compare_mask], window=5)[::8]
    waveform_rmse = float(np.sqrt(np.mean((sim_ds - obs_ds) ** 2)))
    waveform_scale = max(40.0, float(np.ptp(obs_ds)))
    obs_onset = smooth_trace(obs_v_mV[onset_mask], window=5)[::4]
    sim_onset = smooth_trace(sim_interp[onset_mask], window=5)[::4]
    onset_rmse = float(np.sqrt(np.mean((sim_onset - obs_onset) ** 2))) if obs_onset.size else 0.0
    onset_scale = max(20.0, float(np.ptp(obs_onset))) if obs_onset.size else 20.0
    obs_turnoff = smooth_trace(obs_v_mV[turnoff_mask], window=5)[::4]
    sim_turnoff = smooth_trace(sim_interp[turnoff_mask], window=5)[::4]
    turnoff_rmse = float(np.sqrt(np.mean((sim_turnoff - obs_turnoff) ** 2))) if obs_turnoff.size else 0.0
    turnoff_scale = max(20.0, float(np.ptp(obs_turnoff))) if obs_turnoff.size else 20.0

    score = 0.0
    score += 1.3 * abs(sim_metrics["plateau_mean_mV"] - obs_metrics["plateau_mean_mV"]) / max(
        25.0, abs(obs_metrics["plateau_mean_mV"])
    )
    score += 0.9 * abs(sim_metrics["plateau_vpp_mV"] - obs_metrics["plateau_vpp_mV"]) / max(
        10.0, abs(obs_metrics["plateau_vpp_mV"])
    )
    score += 0.9 * abs(sim_metrics["peak_0_80_mV"] - obs_metrics["peak_0_80_mV"]) / max(
        25.0, abs(obs_metrics["peak_0_80_mV"])
    )
    score += 0.9 * abs(sim_metrics["turnoff_min_300_450_mV"] - obs_metrics["turnoff_min_300_450_mV"]) / max(
        25.0, abs(obs_metrics["turnoff_min_300_450_mV"])
    )
    score += 0.8 * waveform_rmse / waveform_scale
    score += 1.0 * onset_rmse / onset_scale
    score += 1.25 * turnoff_rmse / turnoff_scale
    if emphasize_slope:
        score += 1.0 * abs(sim_metrics["slope_0_30_mV_per_ns"] - obs_metrics["slope_0_30_mV_per_ns"]) / max(
            0.5, abs(obs_metrics["slope_0_30_mV_per_ns"])
        )
    if obs_metrics["turnoff_min_300_450_mV"] < -20.0 and sim_metrics["turnoff_min_300_450_mV"] > -5.0:
        score += 1.5
    if obs_metrics["plateau_mean_mV"] > 40.0 and sim_metrics["plateau_mean_mV"] < 0.5 * obs_metrics["plateau_mean_mV"]:
        score += 1.2
    return score, sim_metrics


def _evaluate_candidate(
    summary_df: pd.DataFrame,
    traces_by_index: dict[int, tuple[np.ndarray, np.ndarray]],
    forcing_by_index: dict[int, tuple[np.ndarray, np.ndarray]],
    resist_params: YuanhangResistParams,
    *,
    candidate: dict[str, float],
    fit_indices: list[int],
    forcing_mode: str,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    i_peak_uA = float(summary_df["current_inferred_uA"].iloc[max(fit_indices)])
    rows: list[dict[str, float]] = []
    total_score = 0.0
    total_weight = 0.0
    max_fit_idx = max(fit_indices)
    for idx in fit_indices:
        obs_t_ns, obs_v_mV = traces_by_index[idx]
        forcing_t_ns, forcing_i_uA = forcing_by_index[idx]
        current_uA = float(summary_df.loc[idx, "current_inferred_uA"])
        out = _simulate_trace(
            current_uA,
            forcing_t_ns,
            forcing_i_uA,
            resist_params,
            C_pF=float(candidate["C_pF"]),
            C_th_mW_ns_per_K=float(candidate["C_th_mW_ns_per_K"]),
            S_e_mW_per_K=float(candidate["S_e_mW_per_K"]),
            T0_K=float(candidate["T0_K"]),
            T_init_K=float(candidate["T_init_K"]),
            Tc_shift_K=float(candidate.get("Tc_shift_K", 0.0)),
            beta_scale=float(candidate.get("beta_scale", 1.0)),
            R0_scale=float(candidate.get("R0_scale", 1.0)),
            Ea_scale=float(candidate.get("Ea_scale", 1.0)),
            dt_s=0.1e-9,
            i_peak_uA=i_peak_uA,
            forcing_mode=forcing_mode,
            seed=seed + idx,
        )
        sim_t_ns = out["t"] * 1e9
        sim_v_mV = out["V_vo2"] * 1e3
        score, sim_metrics = _trace_score(
            obs_t_ns,
            obs_v_mV,
            sim_t_ns,
            sim_v_mV,
            emphasize_slope=idx <= 2,
        )
        weight = 1.0 + 0.8 * (idx / max(max_fit_idx, 1))
        total_score += weight * score
        total_weight += weight
        rows.append(
            {
                "frame_index": float(idx),
                "current_uA": current_uA,
                "candidate_C_pF": float(candidate["C_pF"]),
                "candidate_C_th_mW_ns_per_K": float(candidate["C_th_mW_ns_per_K"]),
                "candidate_S_e_mW_per_K": float(candidate["S_e_mW_per_K"]),
                "candidate_T0_K": float(candidate["T0_K"]),
                "candidate_T_init_K": float(candidate["T_init_K"]),
                "candidate_Tc_shift_K": float(candidate.get("Tc_shift_K", 0.0)),
                "candidate_beta_scale": float(candidate.get("beta_scale", 1.0)),
                "candidate_R0_scale": float(candidate.get("R0_scale", 1.0)),
                "candidate_Ea_scale": float(candidate.get("Ea_scale", 1.0)),
                "trace_weight": float(weight),
                "trace_score": float(score),
                **{f"sim_{k}": float(v) for k, v in sim_metrics.items()},
            }
        )
    return total_score / max(total_weight, 1.0), rows


def _plot_representative_overlays(
    out_dir: Path,
    summary_df: pd.DataFrame,
    traces_by_index: dict[int, tuple[np.ndarray, np.ndarray]],
    forcing_by_index: dict[int, tuple[np.ndarray, np.ndarray]],
    resist_params: YuanhangResistParams,
    *,
    best_candidate: dict[str, float],
    fit_indices: list[int],
    forcing_mode: str,
    seed: int,
) -> None:
    i_peak_uA = float(summary_df["current_inferred_uA"].max())
    ncols = 2
    nrows = int(np.ceil(len(fit_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, max(4.0, 3.2 * nrows)))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, idx in zip(axes_flat, fit_indices):
        obs_t_ns, obs_v_mV = traces_by_index[idx]
        forcing_t_ns, forcing_i_uA = forcing_by_index[idx]
        current_uA = float(summary_df.loc[idx, "current_inferred_uA"])
        out = _simulate_trace(
            current_uA,
            forcing_t_ns,
            forcing_i_uA,
            resist_params,
            C_pF=float(best_candidate["C_pF"]),
            C_th_mW_ns_per_K=float(best_candidate["C_th_mW_ns_per_K"]),
            S_e_mW_per_K=float(best_candidate["S_e_mW_per_K"]),
            T0_K=float(best_candidate["T0_K"]),
            T_init_K=float(best_candidate["T_init_K"]),
            Tc_shift_K=float(best_candidate.get("Tc_shift_K", 0.0)),
            beta_scale=float(best_candidate.get("beta_scale", 1.0)),
            R0_scale=float(best_candidate.get("R0_scale", 1.0)),
            Ea_scale=float(best_candidate.get("Ea_scale", 1.0)),
            dt_s=0.1e-9,
            i_peak_uA=i_peak_uA,
            forcing_mode=forcing_mode,
            seed=seed + idx,
        )
        ax.plot(obs_t_ns, obs_v_mV, label="Digitized", linewidth=2.0)
        ax.plot(out["t"] * 1e9, out["V_vo2"] * 1e3, label="Simulated", linewidth=1.7)
        ax.set_title(f"Frame {idx} ({current_uA:.1f} uA)")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("V_out (mV)")
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(fit_indices):]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "representative_overlays.png", dpi=180)
    plt.close(fig)


def _plot_family_summary(
    out_dir: Path,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key, ylabel in [
        (axes[0, 0], "plateau_mean_mV", "Plateau mean (mV)"),
        (axes[0, 1], "plateau_vpp_mV", "Plateau Vpp (mV)"),
        (axes[1, 0], "peak_0_80_mV", "Peak 0-80 ns (mV)"),
        (axes[1, 1], "turnoff_min_300_450_mV", "Turn-off minimum (mV)"),
    ]:
        ax.plot(comparison_df["current_uA"], comparison_df[f"obs_{key}"], "o-", label="Digitized")
        ax.plot(comparison_df["current_uA"], comparison_df[f"sim_{key}"], "-", label="Simulated")
        ax.set_xlabel("Current (uA)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "family_summary_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    image_dir = Path(args.image_dir)
    preset_path = Path(args.resistance_preset)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"current_results_fit_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(preset_path.read_text())
    resist_params = YuanhangResistParams(**payload["resist_params"])

    bounds, traces, summary_df = digitize_directory(image_dir)
    traces_by_index = {trace.frame_index: (trace.time_ns, trace.v_out_mV) for trace in traces}
    forcing_by_index = {
        trace.frame_index: (
            trace.time_ns,
            reconstruct_current_waveform(trace, float(summary_df.loc[trace.frame_index, "current_inferred_uA"])),
        )
        for trace in traces
    }
    summary_df.to_csv(out_dir / "digitized_summary.csv", index=False)

    fit_indices = _parse_fit_indices(args.fit_indices)
    rng = np.random.default_rng(args.seed)

    candidate_rows: list[dict[str, float]] = []
    best_score = float("inf")
    seed_candidates = _seed_candidates(fit_pulse_law=args.fit_pulse_law)
    best_candidate = seed_candidates[0]
    coarse_candidates = seed_candidates + [_random_candidate(rng, fit_pulse_law=args.fit_pulse_law) for _ in range(24)]
    n_coarse = len(coarse_candidates)
    for coarse_idx, candidate in enumerate(coarse_candidates, start=1):
        msg = (
            "[fit][coarse] "
            f"{coarse_idx}/{n_coarse}: "
            f"C={candidate['C_pF']:.3f} pF, "
            f"C_th={candidate['C_th_mW_ns_per_K']:.3f} mW*ns/K, "
            f"S_e={candidate['S_e_mW_per_K']:.6f} mW/K, "
            f"T0={candidate['T0_K']:.3f} K, "
            f"T_init={candidate['T_init_K']:.3f} K"
        )
        if args.fit_pulse_law:
            msg += (
                f", dTc={candidate['Tc_shift_K']:.3f} K, "
                f"beta_scale={candidate['beta_scale']:.3f}, "
                f"R0_scale={candidate['R0_scale']:.3f}, "
                f"Ea_scale={candidate['Ea_scale']:.3f}"
            )
        print(msg)
        score, _ = _evaluate_candidate(
            summary_df,
            traces_by_index,
            forcing_by_index,
            resist_params,
            candidate=candidate,
            fit_indices=fit_indices,
            forcing_mode=args.forcing_mode,
            seed=args.seed,
        )
        candidate_rows.append(
            {
                "stage": "coarse",
                "score": float(score),
                **{k: float(v) for k, v in candidate.items()},
            }
        )
        if score < best_score:
            best_score = score
            best_candidate = candidate.copy()

    coarse_ranked = pd.DataFrame(candidate_rows).sort_values("score").reset_index(drop=True)
    refine_bases = coarse_ranked.head(5)
    refine_candidates: list[dict[str, float]] = []
    for _, row in refine_bases.iterrows():
        base = _make_candidate(
            C_pF=float(row["C_pF"]),
            C_th_mW_ns_per_K=float(row["C_th_mW_ns_per_K"]),
            S_e_mW_per_K=float(row["S_e_mW_per_K"]),
            T0_K=float(row["T0_K"]),
            T_init_K=float(row["T_init_K"]),
            Tc_shift_K=float(row.get("Tc_shift_K", 0.0)),
            beta_scale=float(row.get("beta_scale", 1.0)),
            R0_scale=float(row.get("R0_scale", 1.0)),
            Ea_scale=float(row.get("Ea_scale", 1.0)),
        )
        refine_candidates.append(base)
        refine_candidates.extend(_perturb_candidate(base, rng, fit_pulse_law=args.fit_pulse_law) for _ in range(4))

    n_refine = len(refine_candidates)
    for refine_idx, candidate in enumerate(refine_candidates, start=1):
        msg = (
            "[fit][refine] "
            f"{refine_idx}/{n_refine}: "
            f"C={candidate['C_pF']:.3f} pF, "
            f"C_th={candidate['C_th_mW_ns_per_K']:.3f} mW*ns/K, "
            f"S_e={candidate['S_e_mW_per_K']:.6f} mW/K, "
            f"T0={candidate['T0_K']:.3f} K, "
            f"T_init={candidate['T_init_K']:.3f} K"
        )
        if args.fit_pulse_law:
            msg += (
                f", dTc={candidate['Tc_shift_K']:.3f} K, "
                f"beta_scale={candidate['beta_scale']:.3f}, "
                f"R0_scale={candidate['R0_scale']:.3f}, "
                f"Ea_scale={candidate['Ea_scale']:.3f}"
            )
        print(msg)
        score, _ = _evaluate_candidate(
            summary_df,
            traces_by_index,
            forcing_by_index,
            resist_params,
            candidate=candidate,
            fit_indices=fit_indices,
            forcing_mode=args.forcing_mode,
            seed=args.seed,
        )
        candidate_rows.append(
            {
                "stage": "refine",
                "score": float(score),
                **{k: float(v) for k, v in candidate.items()},
            }
        )
        if score < best_score:
            best_score = score
            best_candidate = candidate.copy()

    candidate_df = pd.DataFrame(candidate_rows).sort_values("score").reset_index(drop=True)
    candidate_df.to_csv(out_dir / "candidate_scores.csv", index=False)

    full_rows: list[dict[str, float]] = []
    i_peak_uA = float(summary_df["current_inferred_uA"].max())
    for idx, trace in enumerate(traces):
        obs_t_ns = trace.time_ns
        obs_v_mV = trace.v_out_mV
        obs_metrics = _trace_metrics(obs_t_ns, obs_v_mV)
        current_uA = float(summary_df.loc[idx, "current_inferred_uA"])
        forcing_t_ns, forcing_i_uA = forcing_by_index[idx]
        out = _simulate_trace(
            current_uA,
            forcing_t_ns,
            forcing_i_uA,
            resist_params,
            C_pF=float(best_candidate["C_pF"]),
            C_th_mW_ns_per_K=float(best_candidate["C_th_mW_ns_per_K"]),
            S_e_mW_per_K=float(best_candidate["S_e_mW_per_K"]),
            T0_K=float(best_candidate["T0_K"]),
            T_init_K=float(best_candidate["T_init_K"]),
            Tc_shift_K=float(best_candidate.get("Tc_shift_K", 0.0)),
            beta_scale=float(best_candidate.get("beta_scale", 1.0)),
            R0_scale=float(best_candidate.get("R0_scale", 1.0)),
            Ea_scale=float(best_candidate.get("Ea_scale", 1.0)),
            dt_s=0.1e-9,
            i_peak_uA=i_peak_uA,
            forcing_mode=args.forcing_mode,
            seed=args.seed + idx,
        )
        sim_t_ns = out["t"] * 1e9
        sim_v_mV = out["V_vo2"] * 1e3
        sim_metrics = _trace_metrics(obs_t_ns, np.interp(obs_t_ns, sim_t_ns, sim_v_mV))
        full_rows.append(
            {
                "frame_index": float(idx),
                "current_uA": current_uA,
                **{f"obs_{k}": float(v) for k, v in obs_metrics.items()},
                **{f"sim_{k}": float(v) for k, v in sim_metrics.items()},
            }
        )
    comparison_df = pd.DataFrame(full_rows)
    comparison_df.to_csv(out_dir / "best_fit_family_comparison.csv", index=False)

    result = {
        "plot_bounds": {
            "x_left_axis": bounds.x_left_axis,
            "x_right_axis": bounds.x_right_axis,
            "y_top_axis": bounds.y_top_axis,
            "y_bottom_axis": bounds.y_bottom_axis,
        },
        "fit_indices": fit_indices,
        "best_C_pF": float(best_candidate["C_pF"]),
        "best_C_th_mW_ns_per_K": float(best_candidate["C_th_mW_ns_per_K"]),
        "best_S_e_mW_per_K": float(best_candidate["S_e_mW_per_K"]),
        "best_T0_K": float(best_candidate["T0_K"]),
        "best_T_init_K": float(best_candidate["T_init_K"]),
        "best_Tc_shift_K": float(best_candidate.get("Tc_shift_K", 0.0)),
        "best_beta_scale": float(best_candidate.get("beta_scale", 1.0)),
        "best_R0_scale": float(best_candidate.get("R0_scale", 1.0)),
        "best_Ea_scale": float(best_candidate.get("Ea_scale", 1.0)),
        "best_score": best_score,
        "fixed_start_branch": "insulator",
        "forcing_mode": args.forcing_mode,
        "fit_pulse_law": bool(args.fit_pulse_law),
        "notes": [
            "Current sweep inferred from visible green plateaus and extrapolated linearly after clipping.",
            "C, C_th, S_e, T0, and T_init were fit against representative digitized traces.",
            "The resistance preset was held fixed unless fit_pulse_law enabled limited Tc/beta/R0/Ea pulse-side adjustments.",
            "This fit is image-based and should be treated as approximate until raw scope data is available.",
        ],
    }
    (out_dir / "best_fit.json").write_text(json.dumps(result, indent=2))
    _plot_representative_overlays(
        out_dir,
        summary_df,
        traces_by_index,
        forcing_by_index,
        resist_params,
        best_candidate=best_candidate,
        fit_indices=fit_indices,
        forcing_mode=args.forcing_mode,
        seed=args.seed,
    )
    _plot_family_summary(out_dir, summary_df, comparison_df)

    print(f"Wrote fit outputs to: {out_dir}")
    print(
        "Best fit: "
        f"C = {best_candidate['C_pF']:.3f} pF, "
        f"C_th = {best_candidate['C_th_mW_ns_per_K']:.3f} mW*ns/K, "
        f"S_e = {best_candidate['S_e_mW_per_K']:.6f} mW/K, "
        f"T0 = {best_candidate['T0_K']:.3f} K, "
        f"T_init = {best_candidate['T_init_K']:.3f} K, "
        f"Tc_shift = {best_candidate.get('Tc_shift_K', 0.0):.3f} K, "
        f"beta_scale = {best_candidate.get('beta_scale', 1.0):.3f}, "
        f"R0_scale = {best_candidate.get('R0_scale', 1.0):.3f}, "
        f"Ea_scale = {best_candidate.get('Ea_scale', 1.0):.3f}, "
        f"score = {best_score:.6f}"
    )
    print(f"Fit traces: {fit_indices}")


if __name__ == "__main__":
    main()
