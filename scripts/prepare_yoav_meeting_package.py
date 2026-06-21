from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuristor.current_results_digitizer import digitize_directory, reconstruct_current_waveform
from neuristor.model import YuanhangResistParams
from scripts.fit_current_results_dynamic import _simulate_trace, _trace_metrics


OUT_DIR = ROOT / "outputs" / "yoav_meeting_prep"
BASELINE_DIR = ROOT / "outputs" / "current_results_fit_highregime"
DYNAMIC_G_DIR = ROOT / "outputs" / "current_results_fit_highregime_dynamicg"
DOUBLE_THERMAL_DIR = ROOT / "outputs" / "current_results_fit_highregime_double"


def _load_resistance_params() -> YuanhangResistParams:
    payload = json.loads((ROOT / "presets" / "resistance_100425_chip1_gap3.json").read_text())
    return YuanhangResistParams(**payload["resist_params"])


def _candidate_from_fit(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    return {
        "C_pF": float(payload["best_C_pF"]),
        "C_th_mW_ns_per_K": float(payload["best_C_th_mW_ns_per_K"]),
        "S_e_mW_per_K": float(payload["best_S_e_mW_per_K"]),
        "T0_K": float(payload["best_T0_K"]),
        "T_init_K": float(payload["best_T_init_K"]),
        "Tc_shift_K": float(payload.get("best_Tc_shift_K", 0.0)),
        "beta_scale": float(payload.get("best_beta_scale", 1.0)),
        "R0_scale": float(payload.get("best_R0_scale", 1.0)),
        "Ea_scale": float(payload.get("best_Ea_scale", 1.0)),
        "C_sub_factor": float(payload.get("best_C_sub_factor", 1.0)),
        "G_hot_sub_scale": float(payload.get("best_G_hot_sub_scale", 1.0)),
        "tau_g_ns": float(payload.get("best_tau_g_ns", 0.0)),
    }


def _prominent_turn_count(time_ns: np.ndarray, v_mV: np.ndarray, *, lo_ns: float = 100.0, hi_ns: float = 250.0) -> int:
    mask = (time_ns >= lo_ns) & (time_ns <= hi_ns)
    if np.sum(mask) < 8:
        return 0
    values = np.asarray(v_mV[mask], dtype=float)
    if values.size >= 7:
        pad = 3
        padded = np.pad(values, (pad, pad), mode="edge")
        y = np.convolve(padded, np.ones(7, dtype=float) / 7.0, mode="valid")
    else:
        y = values
    if float(np.ptp(y)) < 6.0:
        return 0
    d = np.diff(y)
    signs = np.sign(d)
    signs[signs == 0.0] = np.nan
    if np.any(np.isnan(signs)):
        good = np.where(np.isfinite(signs))[0]
        if good.size < 2:
            return 0
        signs = np.interp(np.arange(signs.size), good, signs[good])
    extrema = np.where((signs[:-1] * signs[1:]) < 0.0)[0] + 1
    if extrema.size == 0:
        return 0
    kept = [int(extrema[0])]
    for idx in extrema[1:]:
        if abs(float(y[idx]) - float(y[kept[-1]])) >= 2.0:
            kept.append(int(idx))
    return int(len(kept))


def _simulate_best(
    summary_df: pd.DataFrame,
    traces_by_index: dict[int, tuple[np.ndarray, np.ndarray]],
    forcing_by_index: dict[int, tuple[np.ndarray, np.ndarray]],
    resist_params: YuanhangResistParams,
    candidate: dict[str, float],
    *,
    thermal_model: str = "single",
    phase_model: str = "quasistatic",
    domain_count: int = 1,
    domain_temperature_span_K: float = 0.0,
    domain_coupling_scale: float = 0.0,
    frame_indices: list[int] | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    i_peak_uA = float(summary_df["current_inferred_uA"].max())
    frames = list(range(len(summary_df))) if frame_indices is None else frame_indices
    outputs: dict[int, dict[str, np.ndarray]] = {}
    for idx in frames:
        forcing_t_ns, forcing_i_uA = forcing_by_index[idx]
        current_uA = float(summary_df.loc[idx, "current_inferred_uA"])
        outputs[idx] = _simulate_trace(
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
            C_sub_factor=float(candidate.get("C_sub_factor", 1.0)),
            G_hot_sub_scale=float(candidate.get("G_hot_sub_scale", 1.0)),
            tau_g_ns=float(candidate.get("tau_g_ns", 0.0)),
            dt_s=0.1e-9,
            i_peak_uA=i_peak_uA,
            forcing_mode="digitized_iin",
            thermal_model=thermal_model,
            phase_model=phase_model,
            domain_count=domain_count,
            domain_temperature_span_K=domain_temperature_span_K,
            domain_coupling_scale=domain_coupling_scale,
            seed=1000 + idx,
        )
    return outputs


def _comparison_from_outputs(
    summary_df: pd.DataFrame,
    traces_by_index: dict[int, tuple[np.ndarray, np.ndarray]],
    outputs: dict[int, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for idx, out in outputs.items():
        obs_t_ns, obs_v_mV = traces_by_index[idx]
        sim_t_ns = out["t"] * 1e9
        sim_v_mV = out["V_vo2"] * 1e3
        sim_interp = np.interp(obs_t_ns, sim_t_ns, sim_v_mV)
        obs_metrics = _trace_metrics(obs_t_ns, obs_v_mV)
        sim_metrics = _trace_metrics(obs_t_ns, sim_interp)
        rows.append(
            {
                "frame_index": float(idx),
                "current_uA": float(summary_df.loc[idx, "current_inferred_uA"]),
                "obs_prominent_turn_count_100_250": float(_prominent_turn_count(obs_t_ns, obs_v_mV)),
                "sim_prominent_turn_count_100_250": float(_prominent_turn_count(obs_t_ns, sim_interp)),
                **{f"obs_{k}": float(v) for k, v in obs_metrics.items()},
                **{f"sim_{k}": float(v) for k, v in sim_metrics.items()},
            }
        )
    return pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)


def _add_threshold_lines(ax: plt.Axes, exp_onset: float | None, sim_onset: float | None) -> None:
    if exp_onset is not None:
        ax.axvline(exp_onset, color="black", linestyle=":", linewidth=1.2, alpha=0.7, label="Exp onset")
    if sim_onset is not None:
        ax.axvline(sim_onset, color="tab:red", linestyle=":", linewidth=1.2, alpha=0.7, label="Sim onset")


def _first_onset(df: pd.DataFrame, col: str, threshold: float) -> float | None:
    hit = df[df[col] >= threshold]
    if hit.empty:
        return None
    return float(hit["current_uA"].iloc[0])


def _plot_overlays(
    traces_by_index: dict[int, tuple[np.ndarray, np.ndarray]],
    outputs: dict[int, dict[str, np.ndarray]],
    summary_df: pd.DataFrame,
) -> None:
    frames = [2, 5, 9, 35]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True, sharey=False)
    for ax, idx in zip(axes.ravel(), frames):
        obs_t, obs_v = traces_by_index[idx]
        out = outputs[idx]
        ax.plot(obs_t, obs_v, color="black", linewidth=1.7, label="Experiment")
        ax.plot(out["t"] * 1e9, out["V_vo2"] * 1e3, color="tab:red", linewidth=1.4, label="Baseline sim")
        ax.axvspan(100.0, 250.0, color="tab:blue", alpha=0.07)
        ax.set_title(f"Frame {idx}, I={summary_df.loc[idx, 'current_inferred_uA']:.0f} uA")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("V_out (mV)")
        ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "overlay_representative_low_mid_high.png", dpi=180)
    plt.close(fig)


def _plot_metric(df: pd.DataFrame, obs_col: str, sim_col: str, ylabel: str, fname: str, *, onset: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(df["current_uA"], df[obs_col], "o-", color="black", label="Experiment")
    ax.plot(df["current_uA"], df[sim_col], "o-", color="tab:red", label="Baseline sim")
    if onset:
        exp_on = _first_onset(df, obs_col, 10.0)
        sim_on = _first_onset(df, sim_col, 10.0)
        _add_threshold_lines(ax, exp_on, sim_on)
    ax.set_xlabel("Inferred current (uA)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=180)
    plt.close(fig)


def _plot_power(power_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(power_df["current_uA"], power_df["exp_switch_power_uW"], "o-", color="black", label="Exp I*peak V")
    ax.plot(power_df["current_uA"], power_df["sim_switch_power_uW"], "o-", color="tab:red", label="Sim I*peak V")
    ax.plot(power_df["current_uA"], power_df["sim_internal_peak_power_0_80_uW"], "--", color="tab:orange", label="Sim V^2/R peak")
    ax.set_xlabel("Inferred current (uA)")
    ax.set_ylabel("Estimated switching power (uW)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "switching_power_vs_current.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(power_df["current_uA"], power_df["exp_hold_power_uW"], "o-", color="black", label="Exp I*plateau V")
    ax.plot(power_df["current_uA"], power_df["sim_hold_power_uW"], "o-", color="tab:red", label="Sim I*plateau V")
    ax.plot(power_df["current_uA"], power_df["sim_internal_hold_power_uW"], "--", color="tab:orange", label="Sim V^2/R plateau")
    ax.set_xlabel("Inferred current (uA)")
    ax.set_ylabel("Estimated hold power (uW)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hold_power_vs_current.png", dpi=180)
    plt.close(fig)


def _plot_internal(outputs: dict[int, dict[str, np.ndarray]], summary_df: pd.DataFrame) -> None:
    frames = [5, 9, 35]
    fig, axes = plt.subplots(len(frames), 4, figsize=(14.5, 7.8), sharex=True)
    for row, idx in enumerate(frames):
        out = outputs[idx]
        t_ns = out["t"] * 1e9
        series = [
            (out["V_vo2"] * 1e3, "V_out (mV)", "tab:purple"),
            (out["T"], "T (K)", "tab:red"),
            (out["R"], "R_eq (ohm)", "tab:blue"),
            (out["g_dyn"], "g", "tab:green"),
        ]
        for col, (values, ylabel, color) in enumerate(series):
            ax = axes[row, col]
            ax.plot(t_ns, values, color=color, linewidth=1.2)
            ax.axvspan(100.0, 250.0, color="tab:blue", alpha=0.07)
            ax.grid(True, alpha=0.22)
            if col == 0:
                ax.set_ylabel(f"Frame {idx}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f"I={summary_df.loc[idx, 'current_inferred_uA']:.0f} uA" if col == 0 else ylabel)
            if row == len(frames) - 1:
                ax.set_xlabel("Time (ns)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "internal_state_baseline_T_R_g.png", dpi=180)
    plt.close(fig)


def _plot_variant_summary(variant_rows: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    metrics = [
        ("plateau_vpp_mV", "Mean plateau Vpp (mV)"),
        ("plateau_mean_mV", "Mean plateau V_out (mV)"),
        ("turnoff_min_300_450_mV", "Mean turn-off minimum (mV)"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        plot_df = variant_rows[variant_rows["metric"] == metric]
        ax.bar(plot_df["label"], plot_df["value"], color=plot_df["color"])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variant_transition_metric_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resist_params = _load_resistance_params()
    _, traces, summary_df = digitize_directory(ROOT / "data" / "Current Results")
    traces_by_index = {trace.frame_index: (trace.time_ns, trace.v_out_mV) for trace in traces}
    forcing_by_index = {
        trace.frame_index: (
            trace.time_ns,
            reconstruct_current_waveform(trace, float(summary_df.loc[trace.frame_index, "current_inferred_uA"])),
        )
        for trace in traces
    }

    baseline_candidate = _candidate_from_fit(BASELINE_DIR / "best_fit.json")
    baseline_outputs = _simulate_best(summary_df, traces_by_index, forcing_by_index, resist_params, baseline_candidate)
    comparison_df = _comparison_from_outputs(summary_df, traces_by_index, baseline_outputs)
    comparison_df.to_csv(OUT_DIR / "baseline_recomputed_comparison.csv", index=False)
    summary_df.to_csv(OUT_DIR / "digitized_summary.csv", index=False)

    power_rows: list[dict[str, float]] = []
    for _, row in comparison_df.iterrows():
        idx = int(row["frame_index"])
        current = float(row["current_uA"])
        out = baseline_outputs[idx]
        t_ns = out["t"] * 1e9
        p_uW = out["P"] * 1e6
        peak_mask = (t_ns >= 0.0) & (t_ns <= 80.0)
        plateau_mask = (t_ns >= 100.0) & (t_ns <= 250.0)
        power_rows.append(
            {
                "frame_index": float(idx),
                "current_uA": current,
                "exp_switch_power_uW": current * float(row["obs_peak_0_80_mV"]) / 1000.0,
                "sim_switch_power_uW": current * float(row["sim_peak_0_80_mV"]) / 1000.0,
                "exp_hold_power_uW": current * float(row["obs_plateau_mean_mV"]) / 1000.0,
                "sim_hold_power_uW": current * float(row["sim_plateau_mean_mV"]) / 1000.0,
                "sim_internal_peak_power_0_80_uW": float(np.max(p_uW[peak_mask])) if np.any(peak_mask) else float("nan"),
                "sim_internal_hold_power_uW": float(np.mean(p_uW[plateau_mask])) if np.any(plateau_mask) else float("nan"),
            }
        )
    power_df = pd.DataFrame(power_rows)
    power_df.to_csv(OUT_DIR / "power_comparison.csv", index=False)

    domain_spot = OUT_DIR / "domain_spot_check.csv"
    variant_rows: list[dict[str, object]] = []
    transition = comparison_df[(comparison_df["frame_index"] >= 5.0) & (comparison_df["frame_index"] <= 13.0)]
    variant_rows.extend(
        [
            {"label": "Experiment", "metric": "plateau_vpp_mV", "value": float(transition["obs_plateau_vpp_mV"].mean()), "color": "black"},
            {"label": "Baseline", "metric": "plateau_vpp_mV", "value": float(transition["sim_plateau_vpp_mV"].mean()), "color": "tab:red"},
            {"label": "Experiment", "metric": "plateau_mean_mV", "value": float(transition["obs_plateau_mean_mV"].mean()), "color": "black"},
            {"label": "Baseline", "metric": "plateau_mean_mV", "value": float(transition["sim_plateau_mean_mV"].mean()), "color": "tab:red"},
            {
                "label": "Experiment",
                "metric": "turnoff_min_300_450_mV",
                "value": float(transition["obs_turnoff_min_300_450_mV"].mean()),
                "color": "black",
            },
            {
                "label": "Baseline",
                "metric": "turnoff_min_300_450_mV",
                "value": float(transition["sim_turnoff_min_300_450_mV"].mean()),
                "color": "tab:red",
            },
        ]
    )
    for label, directory, color in [
        ("Dynamic g", DYNAMIC_G_DIR, "tab:orange"),
        ("Two thermal", DOUBLE_THERMAL_DIR, "tab:blue"),
    ]:
        path = directory / "best_fit_family_comparison.csv"
        if path.exists():
            df = pd.read_csv(path)
            sub = df[(df["frame_index"] >= 5.0) & (df["frame_index"] <= 13.0)]
            variant_rows.append({"label": label, "metric": "plateau_vpp_mV", "value": float(sub["sim_plateau_vpp_mV"].mean()), "color": color})
            variant_rows.append({"label": label, "metric": "plateau_mean_mV", "value": float(sub["sim_plateau_mean_mV"].mean()), "color": color})
            variant_rows.append({"label": label, "metric": "turnoff_min_300_450_mV", "value": float(sub["sim_turnoff_min_300_450_mV"].mean()), "color": color})
    if domain_spot.exists():
        spot = pd.read_csv(domain_spot)
        spot = spot[(spot["domain_count"] == 16) & (spot["span_K"] == 6.0)]
        spot_transition = spot[spot["frame_index"].isin([5, 9])]
        if not spot_transition.empty:
            variant_rows.append(
                {
                    "label": "Domain spot",
                    "metric": "plateau_vpp_mV",
                    "value": float(spot_transition["sim_plateau_vpp_mV"].mean()),
                    "color": "tab:green",
                }
            )
            variant_rows.append(
                {
                    "label": "Domain spot",
                    "metric": "plateau_mean_mV",
                    "value": float(spot_transition["sim_plateau_mean_mV"].mean()),
                    "color": "tab:green",
                }
            )
            variant_rows.append(
                {
                    "label": "Domain spot",
                    "metric": "turnoff_min_300_450_mV",
                    "value": float(spot_transition["sim_turnoff_min_300_450_mV"].mean()),
                    "color": "tab:green",
                }
            )
    variant_df = pd.DataFrame(variant_rows)
    variant_df.to_csv(OUT_DIR / "variant_transition_metrics.csv", index=False)

    _plot_overlays(traces_by_index, baseline_outputs, summary_df)
    _plot_metric(
        comparison_df,
        "obs_prominent_turn_count_100_250",
        "sim_prominent_turn_count_100_250",
        "Prominent turn count, 100-250 ns",
        "oscillation_count_vs_current.png",
        onset=True,
    )
    _plot_metric(comparison_df, "obs_plateau_vpp_mV", "sim_plateau_vpp_mV", "Plateau Vpp (mV)", "plateau_vpp_vs_current.png", onset=True)
    _plot_metric(comparison_df, "obs_plateau_mean_mV", "sim_plateau_mean_mV", "Plateau mean V_out (mV)", "plateau_mean_vout_vs_current.png")
    _plot_metric(
        comparison_df,
        "obs_turnoff_min_300_450_mV",
        "sim_turnoff_min_300_450_mV",
        "Turn-off minimum V_out (mV)",
        "turnoff_min_vs_current.png",
    )
    _plot_power(power_df)
    _plot_internal(baseline_outputs, summary_df)
    _plot_variant_summary(variant_df)

    transition = comparison_df[(comparison_df["frame_index"] >= 5.0) & (comparison_df["frame_index"] <= 13.0)]
    high = comparison_df[comparison_df["current_uA"] >= 500.0]
    exp_onset = _first_onset(comparison_df, "obs_plateau_vpp_mV", 10.0)
    sim_onset = _first_onset(comparison_df, "sim_plateau_vpp_mV", 10.0)
    power_transition = power_df[(power_df["frame_index"] >= 5.0) & (power_df["frame_index"] <= 13.0)]
    summary_payload = {
        "transition_frame_range": "5-13",
        "transition_current_range_uA": [float(transition["current_uA"].min()), float(transition["current_uA"].max())],
        "exp_oscillation_onset_uA_by_vpp10": exp_onset,
        "sim_oscillation_onset_uA_by_vpp10": sim_onset,
        "transition_obs_mean_plateau_vpp_mV": float(transition["obs_plateau_vpp_mV"].mean()),
        "transition_sim_mean_plateau_vpp_mV": float(transition["sim_plateau_vpp_mV"].mean()),
        "transition_obs_mean_prominent_turns": float(transition["obs_prominent_turn_count_100_250"].mean()),
        "transition_sim_mean_prominent_turns": float(transition["sim_prominent_turn_count_100_250"].mean()),
        "transition_obs_mean_plateau_mV": float(transition["obs_plateau_mean_mV"].mean()),
        "transition_sim_mean_plateau_mV": float(transition["sim_plateau_mean_mV"].mean()),
        "high_obs_mean_turnoff_min_mV": float(high["obs_turnoff_min_300_450_mV"].mean()),
        "high_sim_mean_turnoff_min_mV": float(high["sim_turnoff_min_300_450_mV"].mean()),
        "transition_exp_switch_power_uW_mean": float(power_transition["exp_switch_power_uW"].mean()),
        "transition_sim_switch_power_uW_mean": float(power_transition["sim_switch_power_uW"].mean()),
        "transition_exp_hold_power_uW_mean": float(power_transition["exp_hold_power_uW"].mean()),
        "transition_sim_hold_power_uW_mean": float(power_transition["sim_hold_power_uW"].mean()),
    }
    (OUT_DIR / "main_findings.json").write_text(json.dumps(summary_payload, indent=2))

    sim_onset_text = "none through 1.40 mA" if sim_onset is None else f"{sim_onset:.0f} uA"
    summary_md = f"""# Current-Drive VO2 Model Diagnosis

## Data Used

- RT/resistance fit source: `data/experimental/100425_chip1_gap3.tsv`
- Resistance preset: `presets/resistance_100425_chip1_gap3.json`
- Current-drive traces: digitized from `data/Current Results/*.png`
- Baseline fit evaluated here: `outputs/current_results_fit_highregime`

## Direct Conclusion

I could not materially improve the oscillation match. The best available current-drive fit still misses the on-pulse oscillatory regime structurally: experiment turns on large plateau oscillations at about {exp_onset:.0f} uA, while the baseline simulation never reaches 10 mV plateau Vpp through the inferred 1.40 mA sweep ({sim_onset_text}).

## Quantitative Mismatch

Transition/oscillatory frames 5-13 ({summary_payload["transition_current_range_uA"][0]:.0f}-{summary_payload["transition_current_range_uA"][1]:.0f} uA):

- Plateau Vpp: experiment {summary_payload["transition_obs_mean_plateau_vpp_mV"]:.1f} mV; baseline {summary_payload["transition_sim_mean_plateau_vpp_mV"]:.1f} mV.
- Prominent on-pulse turn count: experiment {summary_payload["transition_obs_mean_prominent_turns"]:.1f}; baseline {summary_payload["transition_sim_mean_prominent_turns"]:.1f}.
- Plateau mean V_out: experiment {summary_payload["transition_obs_mean_plateau_mV"]:.1f} mV; baseline {summary_payload["transition_sim_mean_plateau_mV"]:.1f} mV.
- Turn-off undershoot for >=500 uA frames: experiment averages {summary_payload["high_obs_mean_turnoff_min_mV"]:.1f} mV; baseline averages {summary_payload["high_sim_mean_turnoff_min_mV"]:.1f} mV.

## Power Sanity Check

Using the observable current-source estimate P = I * V:

- Switching power over frames 5-13: experiment {summary_payload["transition_exp_switch_power_uW_mean"]:.1f} uW; baseline {summary_payload["transition_sim_switch_power_uW_mean"]:.1f} uW.
- Hold/plateau power over frames 5-13: experiment {summary_payload["transition_exp_hold_power_uW_mean"]:.1f} uW; baseline {summary_payload["transition_sim_hold_power_uW_mean"]:.1f} uW.

This says the simulation is not just missing a small ripple. It reaches a much lower on-pulse voltage and therefore a much lower inferred switching/hold power in the regime where the experiment is visibly oscillatory.

## Model Attempts

- Digitized input-current forcing is already used for the high-regime baseline, including the finite rise/fall timing from the green traces.
- The two-thermal extension did not recover transition-frame Vpp or turn-off undershoot.
- The first-order dynamic phase-fraction lag did not recover transition-frame Vpp.
- I added an opt-in parallel-domain experiment mode (`domain_count`, `domain_temperature_span_K`, `domain_coupling_scale`). A spot check with a large 16-domain, 6 K initial spread can create excess ripple at the highest-current frame, but it still does not reproduce frames 5 and 9 where the experimental oscillations are strongest. I would not present it as a successful model.

## Best Remaining Explanation

The lumped current-source model is settling onto a smooth thermal/electrical trajectory instead of crossing a large resistance swing during the pulse. The missing behavior is likely not a small parameter error in C, C_th, S_e, or a first-order phase lag. The experiment appears to need either spatial filament/domain dynamics with a better-calibrated switching distribution, an unmodeled external circuit/compliance effect, or both. The turn-off undershoot being almost absent in simulation is a second independent sign that the present circuit boundary condition is incomplete.

## Plots To Show

1. `overlay_representative_low_mid_high.png`
2. `plateau_vpp_vs_current.png`
3. `oscillation_count_vs_current.png`
4. `plateau_mean_vout_vs_current.png`
5. `turnoff_min_vs_current.png`
6. `switching_power_vs_current.png`
7. `hold_power_vs_current.png`
8. `internal_state_baseline_T_R_g.png`
9. `variant_transition_metric_summary.png`
"""
    (OUT_DIR / "summary.md").write_text(summary_md)
    print(f"Wrote meeting package to {OUT_DIR}")


if __name__ == "__main__":
    main()
