from __future__ import annotations

import json
import math
import sys
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

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
from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_steps
from neuristor.model import HysteresisArray, YuanhangResistParams


PUBLIC_JOBS = ROOT / "public_jobs"


def _repo_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()
OUTPUTS = ROOT / "outputs"
TABLE_NOISE_W_SQRT_S = 1.0e-6
CURRENT_MAX_UA = 1000
DT_S = 1.0e-9
T_END_S = 30.0e-6
PULSE_ON_S = 1.0e-6
PULSE_OFF_S = 27.0e-6


def _job_id() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_published_table_current_{uuid.uuid4().hex[:6]}"


def _resistance_params() -> YuanhangResistParams:
    return YuanhangResistParams(
        R0=5.36e-3,
        Ea_over_k=5220.0,
        Rm0=1286.0,
        Rm_factor=1.0,
        w=7.19,
        Tc_K=332.8,
        beta=0.253,
        gamma=0.956,
        width_factor=1.0,
        T_min_K=305.0,
        T_max_K=370.0,
        reversal_threshold_K=0.01,
    )


def _params(*, sigma: float) -> CurrentDriveParams:
    return CurrentDriveParams(
        dt_s=DT_S,
        t_end_s=T_END_S,
        t_pre_s=0.0,
        pulse_on_s=PULSE_ON_S,
        pulse_off_s=PULSE_OFF_S,
        V_init_V=0.0,
        T0_K=325.0,
        T_init_K=325.0,
        C_F=145.0e-12,
        C_th_J_per_K=49.6e-12,
        S_e_W_per_K=0.201e-3,
        sigma_W_sqrt_s=sigma,
        resist_params=_resistance_params(),
        start_branch="insulator",
    )


def _smooth(values: np.ndarray, window: int = 25) -> np.ndarray:
    if values.size < window or window <= 1:
        return values.astype(float, copy=True)
    kernel = np.full(window, 1.0 / window)
    return np.convolve(values, kernel, mode="same")


def _late_mask(out: dict[str, np.ndarray]) -> np.ndarray:
    t = out["t"]
    return (t >= 8.0e-6) & (t <= 26.0e-6)


def _crossings(values: np.ndarray) -> np.ndarray:
    if values.size < 20:
        return np.array([], dtype=int)
    low, high = np.quantile(values, [0.10, 0.90])
    if high - low <= 1e-9:
        return np.array([], dtype=int)
    threshold = 0.5 * (low + high)
    return np.flatnonzero((values[:-1] < threshold) & (values[1:] >= threshold)) + 1


def _trace_metrics(current_uA: float, out: dict[str, np.ndarray]) -> dict[str, float | str]:
    mask = _late_mask(out)
    t = out["t"][mask]
    v = out["V_vo2"][mask].astype(float)
    v_smooth = _smooth(v)
    crossings = _crossings(v_smooth)
    periods = np.diff(t[crossings]) if crossings.size >= 2 else np.array([])
    mean_period = float(np.mean(periods)) if periods.size else float("nan")
    period_cv = (
        float(np.std(periods) / max(mean_period, 1e-18))
        if periods.size
        else float("inf")
    )
    frequency_mhz = 1e-6 / mean_period if np.isfinite(mean_period) and mean_period > 0.0 else 0.0
    vpp = float(np.quantile(v_smooth, 0.98) - np.quantile(v_smooth, 0.02))

    half = max(1, v_smooth.size // 2)
    early_vpp = float(np.ptp(v_smooth[:half]))
    late_vpp = float(np.ptp(v_smooth[half:]))
    persistence = late_vpp / max(early_vpp, 1e-12)
    n_cycles = max(0, int(crossings.size - 1))
    oscillatory = (
        n_cycles >= 5
        and vpp >= 0.05
        and period_cv <= 0.20
        and 0.55 <= persistence <= 1.8
    )

    r_mean = float(np.mean(out["R"][mask]))
    r_std = float(np.std(out["R"][mask]))
    t_max = float(np.max(out["T"][mask]))
    if oscillatory:
        regime = "oscillatory"
    elif r_mean <= 2.5 * _resistance_params().Rm and r_std <= 0.15 * max(r_mean, 1.0):
        regime = "metallic lock"
    elif t_max >= _resistance_params().Tc_K:
        regime = "transient/noisy switching"
    else:
        regime = "insulating"

    i_active = out["I_in"][np.abs(out["I_in"]) > 0.0]
    i_avg_uA = float(np.mean(i_active) * 1e6) if i_active.size else 0.0
    v_avg_mV = float(np.mean(v) * 1e3)
    return {
        "I_target_uA": float(current_uA),
        "I_avg_uA": i_avg_uA,
        "V_avg_mV": v_avg_mV,
        "V_pp_mV": vpp * 1e3,
        "dominant_freq_MHz": frequency_mhz,
        "n_cycles": float(n_cycles),
        "period_cv": period_cv,
        "persistence_ratio": float(persistence),
        "T_mean_K": float(np.mean(out["T"][mask])),
        "T_max_K": t_max,
        "R_mean_ohm": r_mean,
        "R_min_ohm": float(np.min(out["R"][mask])),
        "R_max_ohm": float(np.max(out["R"][mask])),
        "oscillatory": float(oscillatory),
        "regime": regime,
    }


def _downsample_trace(
    current_uA: float,
    out: dict[str, np.ndarray],
    *,
    max_points: int = 2500,
) -> pd.DataFrame:
    n = out["t"].size
    idx = np.unique(np.linspace(0, n - 1, min(n, max_points), dtype=int))
    return pd.DataFrame(
        {
            "I_target_uA": float(current_uA),
            "time_ns": out["t"][idx] * 1e9,
            "I_in_uA": out["I_in"][idx] * 1e6,
            "V_vo2_mV": out["V_vo2"][idx] * 1e3,
            "T_K": out["T"][idx],
            "R_ohm": out["R"][idx],
            "P_uW": out["P"][idx] * 1e6,
        }
    )


def _spectrum_rows(
    current_uA: float,
    out: dict[str, np.ndarray],
) -> pd.DataFrame:
    mask = _late_mask(out)
    t = out["t"][mask].astype(float)
    v = _smooth(out["V_vo2"][mask].astype(float))
    if t.size < 16 or current_uA <= 0.0:
        return pd.DataFrame(
            columns=[
                "I_target_uA",
                "input_power_dBm",
                "freq_MHz",
                "gain_linear",
                "gain_dB",
            ]
        )
    dt = float(np.median(np.diff(t)))
    centered = v - float(np.mean(v))
    window = np.hanning(centered.size)
    freq_mhz = np.fft.rfftfreq(centered.size, d=dt) * 1e-6
    spectrum = np.abs(np.fft.rfft(centered * window))
    scale = max(float(current_uA) * 1e-6 * 50.0, 1e-18)
    gain = spectrum / scale
    keep = (freq_mhz >= 0.01) & (freq_mhz <= 500.0)
    input_power_w = (float(current_uA) * 1e-6) ** 2 * 50.0
    input_power_dbm = 10.0 * math.log10(max(input_power_w, 1e-18) / 1e-3)
    stride = max(1, int(np.sum(keep) // 600))
    selected = np.flatnonzero(keep)[::stride]
    return pd.DataFrame(
        {
            "I_target_uA": float(current_uA),
            "input_power_dBm": input_power_dbm,
            "freq_MHz": freq_mhz[selected],
            "gain_linear": gain[selected],
            "gain_dB": 20.0 * np.log10(np.maximum(gain[selected], 1e-12)),
        }
    )


def _rt_branches(resist: YuanhangResistParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temps = np.linspace(resist.T_min_K, resist.T_max_K, 700, dtype=np.float32)
    heat = HysteresisArray(resist, size=1, start_branch="insulator")
    heat.initialize(np.asarray([temps[0]], dtype=np.float32))
    r_heat = np.asarray(
        [float(heat.evaluate(np.asarray([temp], dtype=np.float32))[0][0]) for temp in temps]
    )
    cool_t = temps[::-1]
    cool = HysteresisArray(resist, size=1, start_branch="metal")
    cool.initialize(np.asarray([cool_t[0]], dtype=np.float32))
    r_cool_desc = np.asarray(
        [float(cool.evaluate(np.asarray([temp], dtype=np.float32))[0][0]) for temp in cool_t]
    )
    return temps, r_heat, r_cool_desc[::-1]


def _plot_operating_map(
    deterministic: pd.DataFrame,
    stochastic: pd.DataFrame,
    out_path: Path,
) -> None:
    colors = {
        "insulating": "#6b7280",
        "transient/noisy switching": "#f59e0b",
        "oscillatory": "#7c3aed",
        "metallic lock": "#ef4444",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    for data, label, marker in [
        (deterministic, "deterministic control", "o"),
        (stochastic, "table noise", "s"),
    ]:
        axes[0, 0].plot(
            data["I_target_uA"],
            data["dominant_freq_MHz"],
            marker=marker,
            linewidth=1.5,
            markersize=4.5,
            label=label,
        )
        axes[0, 1].plot(
            data["I_target_uA"],
            data["V_pp_mV"],
            marker=marker,
            linewidth=1.5,
            markersize=4.5,
            label=label,
        )
        axes[1, 0].plot(
            data["I_target_uA"],
            data["T_mean_K"],
            marker=marker,
            linewidth=1.5,
            markersize=4.5,
            label=label,
        )

    stochastic_colors = [colors[str(regime)] for regime in stochastic["regime"]]
    axes[1, 1].scatter(
        stochastic["I_target_uA"],
        stochastic["R_mean_ohm"],
        c=stochastic_colors,
        s=40,
    )
    axes[0, 0].set_ylabel("late-window frequency (MHz)")
    axes[0, 1].set_ylabel("late-window Vpp (mV)")
    axes[1, 0].set_ylabel("mean temperature (K)")
    axes[1, 1].set_ylabel("mean resistance (Ohm)")
    axes[1, 1].set_yscale("log")
    axes[0, 0].set_title("cycle frequency")
    axes[0, 1].set_title("voltage amplitude")
    axes[1, 0].set_title("thermal operating point")
    axes[1, 1].set_title("stochastic-run regime classification")
    for ax in axes.ravel():
        ax.set_xlabel("imposed current (uA)")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="upper left")
    axes[0, 1].legend(loc="upper left")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=key, markerfacecolor=value, markersize=8)
        for key, value in colors.items()
    ]
    axes[1, 1].legend(handles=handles, fontsize=8, loc="best")
    fig.suptitle("Published-table parameters in the ideal-current Yuanhang model", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _pick_representatives(summary: pd.DataFrame) -> list[float]:
    oscillatory = summary[summary["oscillatory"] >= 0.5]
    if oscillatory.empty:
        ranked = summary.sort_values(["n_cycles", "V_pp_mV"], ascending=False)
        middle = float(ranked.iloc[0]["I_target_uA"])
    else:
        middle = float(
            oscillatory.sort_values(["n_cycles", "period_cv"], ascending=[False, True])
            .iloc[0]["I_target_uA"]
        )
    return [100.0, middle, float(CURRENT_MAX_UA)]


def _plot_three_regimes(
    traces: dict[float, dict[str, np.ndarray]],
    currents: list[float],
    resist: YuanhangResistParams,
    out_path: Path,
) -> None:
    temps, r_heat, r_cool = _rt_branches(resist)
    labels = ["low current", "strongest switching candidate", "high current"]
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), gridspec_kw={"width_ratios": [1.3, 1.0]})
    for row, (current, label) in enumerate(zip(currents, labels, strict=True)):
        out = traces[current]
        t_us = out["t"] * 1e6
        left = axes[row, 0]
        right_axis = left.twinx()
        left.plot(t_us, out["I_in"] * 1e6, color="#16a34a", linewidth=1.5)
        right_axis.plot(t_us, out["V_vo2"], color="#7e22ce", linewidth=1.3)
        left.set_ylabel("I (uA)", color="#16a34a")
        right_axis.set_ylabel("V (V)", color="#7e22ce")
        left.set_title(f"{label}: {current:.0f} uA")
        left.grid(True, alpha=0.25)
        if row == 2:
            left.set_xlabel("time (us)")

        rt = axes[row, 1]
        rt.plot(temps, r_cool, color="#3b82f6", linewidth=2.0, label="cooling")
        rt.plot(temps, r_heat, color="#ef4444", linewidth=2.0, label="heating")
        rt.plot(out["T"], out["R"], color="#7c3aed", alpha=0.55, linewidth=1.2)
        rt.set_yscale("log")
        rt.set_xlim(temps[0], temps[-1])
        rt.set_ylim(700.0, 1.25 * max(np.max(r_heat), np.max(r_cool)))
        rt.set_ylabel("R (Ohm)")
        rt.grid(True, alpha=0.25)
        if row == 0:
            rt.legend(loc="upper right")
        if row == 2:
            rt.set_xlabel("temperature (K)")
    fig.suptitle("Exact table noise: current/voltage traces and R(T) trajectories", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_noise_comparison(
    current_uA: float,
    deterministic: dict[str, np.ndarray],
    stochastic: dict[str, np.ndarray],
    resist: YuanhangResistParams,
    out_path: Path,
) -> None:
    temps, r_heat, r_cool = _rt_branches(resist)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))
    for row, (out, label) in enumerate(
        [(deterministic, "sigma = 0 control"), (stochastic, "table sigma = 1 uJ s^-1/2")]
    ):
        mask = (out["t"] >= 8.0e-6) & (out["t"] <= 16.0e-6)
        t_us = out["t"][mask] * 1e6
        axes[row, 0].plot(t_us, out["V_vo2"][mask], color="#7e22ce", linewidth=1.2)
        axes[row, 0].set_ylabel("V (V)")
        axes[row, 0].set_title(label)
        axes[row, 0].grid(True, alpha=0.25)

        axes[row, 1].plot(temps, r_cool, color="#3b82f6", linewidth=1.8)
        axes[row, 1].plot(temps, r_heat, color="#ef4444", linewidth=1.8)
        axes[row, 1].plot(out["T"][mask], out["R"][mask], color="#7c3aed", alpha=0.5, linewidth=1.0)
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_ylim(700.0, 1.25 * max(np.max(r_heat), np.max(r_cool)))
        axes[row, 1].set_ylabel("R (Ohm)")
        axes[row, 1].grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("time (us)")
    axes[1, 1].set_xlabel("temperature (K)")
    fig.suptitle(f"Noise sensitivity at {current_uA:.0f} uA", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _run_validation(
    current_uA: float,
    deterministic_params: CurrentDriveParams,
    stochastic_params: CurrentDriveParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestep_rows = []
    for dt_ns in [1.0, 0.5, 0.25]:
        params = replace(deterministic_params, dt_s=dt_ns * 1e-9)
        out = simulate_current_steps([current_uA], params=params, seed=211)[0]
        metrics = _trace_metrics(current_uA, out)
        timestep_rows.append({"dt_ns": dt_ns, **metrics})

    seed_rows = []
    for seed in range(30, 40):
        out = simulate_current_steps([current_uA], params=stochastic_params, seed=seed)[0]
        metrics = _trace_metrics(current_uA, out)
        seed_rows.append({"seed": seed, **metrics})
    return pd.DataFrame(timestep_rows), pd.DataFrame(seed_rows)


def _plot_validation(
    timestep_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(
        timestep_df["dt_ns"],
        timestep_df["dominant_freq_MHz"],
        color="#7c3aed",
        marker="o",
        linewidth=2.0,
    )
    axes[0].invert_xaxis()
    axes[0].set_xlabel("timestep (ns)")
    axes[0].set_ylabel("deterministic frequency (MHz)")
    axes[0].set_title("timestep refinement")
    axes[0].grid(True, alpha=0.25)

    colors = np.where(seed_df["oscillatory"] >= 0.5, "#7c3aed", "#f59e0b")
    axes[1].scatter(
        seed_df["seed"],
        seed_df["V_pp_mV"],
        c=colors,
        s=55,
    )
    axes[1].set_xlabel("noise seed")
    axes[1].set_ylabel("late-window Vpp (mV)")
    axes[1].set_title("exact-noise seed robustness")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Numerical and stochastic validation at the representative current", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_public_job(
    job_dir: Path,
    *,
    params: CurrentDriveParams,
    outputs: list[tuple[str, Path]],
    oscillatory_currents: list[float],
) -> None:
    config = {
        "type": "current_sweep",
        "job_name": "Published Yuanhang table parameters - ideal-current sweep",
        "job_storage": "public",
        "sample_id": "yuanhang_published_table",
        "sample_name": "Published Yuanhang parameter table",
        "sample_source": "Screenshot supplied 2026-07-09",
        "source_model": "Current Source",
        "I_start_uA": 0,
        "I_stop_uA": CURRENT_MAX_UA,
        "I_step_uA": 50,
        "frame_duration_s": 1.0,
        "seed": 19,
        "current_params": asdict(params),
        "table_parameters_not_active": {
            "R_load_ohm": 12000.0,
            "S_c_W_per_K": 4.11e-6,
            "reason": "Ideal current source has no load term; single device has no neighbor coupling.",
        },
        "deterministic_control": "Same parameters with sigma set to zero.",
        "conclusion": (
            "Simulation of the published parameter table in the hardcoded Yuanhang "
            "ideal-current model. Exact table noise and a sigma=0 control are both saved."
        ),
    }
    output_payload = [{"label": label, "path": _repo_path(path)} for label, path in outputs]
    job = {
        "id": job_dir.name,
        "name": config["job_name"],
        "job_storage": "public",
        "type": "current_sweep",
        "sample_id": config["sample_id"],
        "sample_name": config["sample_name"],
        "source_model": "Current Source",
        "status": "completed",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": config,
        "outputs": output_payload,
        "log_path": _repo_path(job_dir / "log.txt"),
    }
    (job_dir / "job.json").write_text(json.dumps(job, indent=2))
    window_text = (
        "none classified"
        if not oscillatory_currents
        else f"{min(oscillatory_currents):.0f}-{max(oscillatory_currents):.0f} uA"
    )
    (job_dir / "log.txt").write_text(
        "[job] published-table ideal-current simulation\n"
        "[model] R_load ignored by ideal imposed-current equation\n"
        "[model] S_c ignored for one device/domain\n"
        f"[noise] exact table sigma={TABLE_NOISE_W_SQRT_S:.3g} W sqrt(s)\n"
        f"[result] exact-noise oscillatory current range: {window_text}\n"
    )


def run() -> Path:
    model._TORCH_HYSTERESIS_AVAILABLE = False
    output_dir = OUTPUTS / f"published_table_current_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    deterministic_params = _params(sigma=0.0)
    stochastic_params = replace(deterministic_params, sigma_W_sqrt_s=TABLE_NOISE_W_SQRT_S)
    deterministic_currents = [float(value) for value in range(0, CURRENT_MAX_UA + 1, 25)]
    stochastic_currents = [float(value) for value in range(0, CURRENT_MAX_UA + 1, 50)]

    deterministic_traces_list = simulate_current_steps(
        deterministic_currents,
        params=deterministic_params,
        seed=7,
    )
    stochastic_traces_list = simulate_current_steps(
        stochastic_currents,
        params=stochastic_params,
        seed=19,
    )
    deterministic_traces = dict(zip(deterministic_currents, deterministic_traces_list, strict=True))
    stochastic_traces = dict(zip(stochastic_currents, stochastic_traces_list, strict=True))

    deterministic_summary = pd.DataFrame(
        [
            _trace_metrics(current, deterministic_traces[current])
            for current in deterministic_currents
        ]
    )
    stochastic_summary = pd.DataFrame(
        [_trace_metrics(current, stochastic_traces[current]) for current in stochastic_currents]
    )

    deterministic_summary.to_csv(output_dir / "deterministic_control_summary.csv", index=False)
    stochastic_summary.to_csv(output_dir / "current_sweep_summary.csv", index=False)
    pd.concat(
        [_downsample_trace(current, stochastic_traces[current]) for current in stochastic_currents],
        ignore_index=True,
    ).to_csv(output_dir / "current_sweep_traces.csv", index=False)
    spectrum_frames = [
        frame
        for current in stochastic_currents
        if not (frame := _spectrum_rows(current, stochastic_traces[current])).empty
    ]
    pd.concat(spectrum_frames, ignore_index=True).to_csv(
        output_dir / "current_sweep_spectra.csv",
        index=False,
    )

    operating_map = output_dir / "published_table_operating_map.png"
    three_regimes = output_dir / "published_table_three_regimes.png"
    noise_comparison = output_dir / "published_table_noise_comparison.png"
    validation_figure = output_dir / "published_table_validation.png"
    _plot_operating_map(deterministic_summary, stochastic_summary, operating_map)
    representatives = _pick_representatives(stochastic_summary)
    _plot_three_regimes(
        stochastic_traces,
        representatives,
        stochastic_params.resist_params,
        three_regimes,
    )
    selected_current = representatives[1]
    deterministic_selected = simulate_current_steps(
        [selected_current],
        params=deterministic_params,
        seed=101,
    )[0]
    _plot_noise_comparison(
        selected_current,
        deterministic_selected,
        stochastic_traces[selected_current],
        stochastic_params.resist_params,
        noise_comparison,
    )
    timestep_df, seed_df = _run_validation(
        selected_current,
        deterministic_params,
        stochastic_params,
    )
    timestep_df.to_csv(output_dir / "timestep_convergence.csv", index=False)
    seed_df.to_csv(output_dir / "noise_seed_robustness.csv", index=False)
    _plot_validation(timestep_df, seed_df, validation_figure)

    report = {
        "active_model": "Yuanhang ideal-current source",
        "parameter_translation": {
            "C_F": 145.0e-12,
            "C_th_J_per_K": 49.6e-12,
            "S_e_W_per_K": 0.201e-3,
            "T0_K": 325.0,
            "sigma_W_sqrt_s": TABLE_NOISE_W_SQRT_S,
            "R0_ohm": 5.36e-3,
            "Ea_over_k_K": 5220.0,
            "Rm_ohm": 1286.0,
            "w_K": 7.19,
            "Tc_K": 332.8,
            "beta_per_K": 0.253,
            "gamma": 0.956,
        },
        "ignored_for_single_ideal_current_model": {
            "R_load_ohm": 12000.0,
            "S_c_W_per_K": 4.11e-6,
        },
        "numerics": {
            "dt_s": DT_S,
            "t_end_s": T_END_S,
            "pulse_on_s": PULSE_ON_S,
            "pulse_off_s": PULSE_OFF_S,
        },
        "deterministic_oscillatory_currents_uA": deterministic_summary.loc[
            deterministic_summary["oscillatory"] >= 0.5, "I_target_uA"
        ].tolist(),
        "stochastic_oscillatory_currents_uA": stochastic_summary.loc[
            stochastic_summary["oscillatory"] >= 0.5, "I_target_uA"
        ].tolist(),
        "representative_currents_uA": representatives,
        "timestep_validation": timestep_df.to_dict(orient="records"),
        "noise_seed_validation": seed_df.to_dict(orient="records"),
    }
    report_path = output_dir / "published_table_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    job_dir = PUBLIC_JOBS / _job_id()
    job_dir.mkdir(parents=True, exist_ok=True)
    copied_outputs: list[tuple[str, Path]] = []
    labels = {
        "current_sweep_summary.csv": "Exact-noise current sweep summary CSV",
        "current_sweep_traces.csv": "Exact-noise current sweep traces CSV",
        "current_sweep_spectra.csv": "Exact-noise current sweep spectra CSV",
        "deterministic_control_summary.csv": "Deterministic control summary CSV",
        "published_table_operating_map.png": "Operating map",
        "published_table_three_regimes.png": "Representative current traces and R(T)",
        "published_table_noise_comparison.png": "Noise comparison",
        "published_table_validation.png": "Timestep and noise-seed validation",
        "timestep_convergence.csv": "Timestep convergence CSV",
        "noise_seed_robustness.csv": "Noise-seed robustness CSV",
        "published_table_report.json": "Parameter translation and report",
    }
    for filename, label in labels.items():
        source = output_dir / filename
        destination = job_dir / filename
        destination.write_bytes(source.read_bytes())
        copied_outputs.append((label, destination))

    _write_public_job(
        job_dir,
        params=stochastic_params,
        outputs=copied_outputs,
        oscillatory_currents=report["stochastic_oscillatory_currents_uA"],
    )
    return job_dir


if __name__ == "__main__":
    print(run())
