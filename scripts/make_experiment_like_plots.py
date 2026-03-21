from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from current_drive_sim import CurrentDriveParams, simulate_current_step
from model import YuanhangResistParams


RF_REF_OHM = 50.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate experiment-like plots from current-driven simulation data.")
    parser.add_argument(
        "--candidate-json",
        default="",
        help="Best-candidate JSON from current_domain_search. Defaults to the newest one in outputs/.",
    )
    parser.add_argument(
        "--resistance-preset",
        default="presets/resistance_100425_chip1_gap3.json",
        help="Resistance preset JSON used for the current-driven simulation.",
    )
    parser.add_argument("--i-start", type=int, default=50)
    parser.add_argument("--i-stop", type=int, default=2000)
    parser.add_argument("--i-step", type=int, default=50)
    parser.add_argument("--t-pre-ns", type=float, default=200.0)
    parser.add_argument("--t-end-ns", type=float, default=600.0)
    parser.add_argument("--pulse-off-ns", type=float, default=300.0)
    parser.add_argument("--representative-current", type=float, default=50.0)
    parser.add_argument("--avg-start-ns", type=float, default=100.0)
    parser.add_argument("--avg-stop-ns", type=float, default=250.0)
    parser.add_argument(
        "--freq-method",
        choices=["small_signal_linearized", "pulse_fft"],
        default="small_signal_linearized",
        help="How to generate the frequency-domain plots.",
    )
    parser.add_argument("--freq-min-mhz", type=float, default=1.0)
    parser.add_argument("--freq-max-mhz", type=float, default=1000.0)
    parser.add_argument("--freq-points-per-decade", type=int, default=36)
    parser.add_argument(
        "--freq-settle-ns",
        type=float,
        default=1200.0,
        help="Bias-settle time used for local small-signal linearization.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    return parser


def newest_candidate_json() -> Path:
    candidates = sorted(Path("outputs").glob("current_domain_search_*/best_candidate.json"))
    if not candidates:
        raise FileNotFoundError("No current-domain search candidate found in outputs/.")
    return candidates[-1]


def load_resistance_preset(path: Path) -> tuple[YuanhangResistParams, str]:
    payload = json.loads(path.read_text())
    raw = payload.get("resist_params", payload)
    params = YuanhangResistParams(**{f.name: float(raw[f.name]) for f in YuanhangResistParams.__dataclass_fields__.values()})
    start_branch = str(payload.get("start_branch", "insulator")).strip().lower()
    if start_branch not in {"insulator", "metal"}:
        start_branch = "insulator"
    return params, start_branch


def load_candidate(path: Path) -> dict:
    return json.loads(path.read_text())


def candidate_resist_params(base: YuanhangResistParams, candidate: dict) -> YuanhangResistParams:
    params = YuanhangResistParams(**base.__dict__)
    params.Rm_factor = float(base.Rm_factor) * float(candidate.get("Rm_factor_scale", 1.0))
    params.Tc_K = float(base.Tc_K) + float(candidate.get("Tc_shift_K", 0.0))
    params.w = float(base.w) * float(candidate.get("w_scale", 1.0))
    params.beta = float(base.beta) * float(candidate.get("beta_scale", 1.0))
    params.reversal_threshold_K = float(candidate.get("reversal_threshold_K", base.reversal_threshold_K))
    return params


def build_params(args: argparse.Namespace, candidate: dict, resist_params: YuanhangResistParams, start_branch: str) -> CurrentDriveParams:
    return CurrentDriveParams(
        dt_s=float(candidate["dt_ns"]) * 1e-9,
        t_end_s=float(args.t_end_ns) * 1e-9,
        t_pre_s=float(args.t_pre_ns) * 1e-9,
        pulse_on_s=0.0,
        pulse_off_s=float(args.pulse_off_ns) * 1e-9,
        V_init_V=0.0,
        T0_K=float(candidate["T0_K"]),
        T_init_K=float(candidate["T_init_K"]),
        C_F=float(candidate["C_pF"]) * 1e-12,
        C_th_J_per_K=float(candidate["C_th_mW_ns_per_K"]) * 1e-12,
        S_e_W_per_K=float(candidate["S_e_mW_per_K"]) * 1e-3,
        sigma_W_sqrt_s=float(candidate.get("sigma_W_sqrt_s", 0.0)),
        resist_params=resist_params,
        start_branch=start_branch,
    )


def estimate_input_power_dbm(i_in_a: np.ndarray) -> float:
    active = np.abs(i_in_a) > 0.0
    i_rms = float(np.sqrt(np.mean((i_in_a[active] if np.any(active) else i_in_a) ** 2)))
    p_w = max((i_rms**2) * RF_REF_OHM, 1e-18)
    return 10.0 * np.log10(p_w / 1e-3)


def fft_gain_spectrum(t_s: np.ndarray, i_in_a: np.ndarray, v_out_v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if t_s.size < 4:
        return np.array([]), np.array([]), np.array([])
    dt = float(np.median(np.diff(t_s)))
    if dt <= 0.0:
        return np.array([]), np.array([]), np.array([])
    i0 = i_in_a - float(np.mean(i_in_a))
    v0 = v_out_v - float(np.mean(v_out_v))
    window = np.hanning(i0.size)
    v_fft = np.fft.rfft(v0 * window)
    f_hz = np.fft.rfftfreq(i0.size, d=dt)
    v_mag = np.abs(v_fft)
    active = np.abs(i_in_a) > 0.0
    i_rms = float(np.sqrt(np.mean((i_in_a[active] if np.any(active) else i_in_a) ** 2)))
    if i_rms <= 0.0:
        return np.array([]), np.array([]), np.array([])
    gain_linear = v_mag / max(i_rms * RF_REF_OHM, 1e-18)
    gain_db = 20.0 * np.log10(np.maximum(gain_linear, 1e-12))
    f_mhz = f_hz * 1e-6
    mask = (f_mhz >= 1.0) & (f_mhz <= 1000.0)
    return f_mhz[mask], gain_linear[mask], gain_db[mask]


def _estimate_local_drdt(T_K: np.ndarray, R_ohm: np.ndarray) -> float:
    if T_K.size < 3 or R_ohm.size != T_K.size:
        return 0.0
    centered_T = T_K - float(np.mean(T_K))
    centered_R = R_ohm - float(np.mean(R_ohm))
    denom = float(np.dot(centered_T, centered_T))
    if denom <= 1e-18:
        return 0.0
    return float(np.dot(centered_T, centered_R) / denom)


def simulate_bias_trace(i_uA: float, params: CurrentDriveParams, settle_ns: float, seed: int) -> dict:
    bias_params = replace(
        params,
        t_pre_s=0.0,
        t_end_s=float(settle_ns) * 1e-9,
        pulse_on_s=0.0,
        pulse_off_s=None,
    )
    return simulate_current_step(float(i_uA), params=bias_params, seed=seed)


def small_signal_gain_spectrum(
    out: dict,
    params: CurrentDriveParams,
    freq_mhz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    t_s = np.asarray(out["t"], dtype=float)
    V = np.asarray(out["V_vo2"], dtype=float)
    T = np.asarray(out["T"], dtype=float)
    R = np.asarray(out["R"], dtype=float)
    if t_s.size < 20:
        return np.array([]), np.array([]), np.array([]), {}

    tail = max(12, t_s.size // 5)
    V_tail = V[-tail:]
    T_tail = T[-tail:]
    R_tail = np.maximum(R[-tail:], 1e-18)

    V_eq = float(np.mean(V_tail))
    T_eq = float(np.mean(T_tail))
    R_eq = float(np.mean(R_tail))
    dR_dT = _estimate_local_drdt(T_tail, R_tail)

    C = max(float(params.C_F), 1e-18)
    C_th = max(float(params.C_th_J_per_K), 1e-18)
    S_e = float(params.S_e_W_per_K)

    j11 = -1.0 / (C * R_eq)
    j12 = V_eq * dR_dT / (C * R_eq * R_eq)
    j21 = 2.0 * V_eq / (C_th * R_eq)
    j22 = (-(V_eq * V_eq) * dR_dT / (R_eq * R_eq) - S_e) / C_th
    J = np.array([[j11, j12], [j21, j22]], dtype=float)
    eig = np.linalg.eigvals(J)

    b = np.array([1.0 / C, 0.0], dtype=complex)
    c = np.array([1.0, 0.0], dtype=complex)
    H_vals = []
    for f_mhz in freq_mhz:
        omega = 2.0 * np.pi * float(f_mhz) * 1e6
        mat = 1j * omega * np.eye(2, dtype=complex) - J.astype(complex)
        try:
            state = np.linalg.solve(mat, b)
        except np.linalg.LinAlgError:
            H_vals.append(np.nan + 1j * np.nan)
            continue
        H_vals.append(np.dot(c, state))
    H = np.asarray(H_vals, dtype=complex)
    transimpedance_ohm = np.abs(H)
    gain_linear = transimpedance_ohm / RF_REF_OHM
    gain_db = 20.0 * np.log10(np.maximum(gain_linear, 1e-12))
    diag = {
        "V_eq_V": V_eq,
        "T_eq_K": T_eq,
        "R_eq_ohm": R_eq,
        "dR_dT_ohm_per_K": float(dR_dT),
        "eig_1_real_per_s": float(np.real(eig[0])),
        "eig_1_imag_per_s": float(np.imag(eig[0])),
        "eig_2_real_per_s": float(np.real(eig[1])),
        "eig_2_imag_per_s": float(np.imag(eig[1])),
    }
    return freq_mhz, gain_linear, gain_db, diag


def simulate_sweep(args: argparse.Namespace, params: CurrentDriveParams) -> list[dict]:
    traces = []
    currents = list(range(int(args.i_start), int(args.i_stop) + 1, int(args.i_step)))
    for idx, i_uA in enumerate(currents):
        out = simulate_current_step(float(i_uA), params=params, seed=int(args.seed) + idx)
        traces.append({"I_target_uA": float(i_uA), "out": out})
    return traces


def make_linear_gain_plot(
    traces: list[dict],
    out_path: Path,
    *,
    method: str,
    params: CurrentDriveParams,
    settle_ns: float,
    freq_grid_mhz: np.ndarray,
    seed: int,
) -> list[dict]:
    cmap = plt.cm.jet_r
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    powers = [estimate_input_power_dbm(item["out"]["I_in"]) for item in traces]
    p_min = min(powers)
    p_max = max(powers)
    norm = plt.Normalize(vmin=p_min, vmax=p_max)
    diagnostics: list[dict] = []

    for idx, (power_dbm, item) in enumerate(zip(powers, traces)):
        out = item["out"]
        if method == "pulse_fft":
            f_mhz, gain_linear, _ = fft_gain_spectrum(out["t"], out["I_in"], out["V_vo2"])
            diag = {"I_target_uA": float(item["I_target_uA"]), "method": "pulse_fft"}
        else:
            bias_out = simulate_bias_trace(float(item["I_target_uA"]), params=params, settle_ns=settle_ns, seed=seed + idx)
            f_mhz, gain_linear, _, diag = small_signal_gain_spectrum(bias_out, params=params, freq_mhz=freq_grid_mhz)
            diag.update({"I_target_uA": float(item["I_target_uA"]), "method": "small_signal_linearized"})
        if f_mhz.size == 0:
            continue
        ax.plot(f_mhz, gain_linear, color=cmap(norm(power_dbm)), linewidth=2.0)
        diagnostics.append(diag)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Input Power (dBm)")
    ax.set_xscale("log")
    ax.set_xlim(float(freq_grid_mhz[0]), float(freq_grid_mhz[-1]))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Linear Gain")
    title = "Linear Gain vs Frequency"
    if method == "small_signal_linearized":
        title += " (Small-Signal Linearized)"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return diagnostics


def make_db_gain_plot(
    traces: list[dict],
    out_path: Path,
    *,
    method: str,
    params: CurrentDriveParams,
    settle_ns: float,
    freq_grid_mhz: np.ndarray,
    seed: int,
) -> None:
    cmap = plt.cm.jet_r
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    powers = [estimate_input_power_dbm(item["out"]["I_in"]) for item in traces]
    p_min = min(powers)
    p_max = max(powers)
    norm = plt.Normalize(vmin=p_min, vmax=p_max)

    for idx, (power_dbm, item) in enumerate(zip(powers, traces)):
        out = item["out"]
        if method == "pulse_fft":
            f_mhz, _, gain_db = fft_gain_spectrum(out["t"], out["I_in"], out["V_vo2"])
        else:
            bias_out = simulate_bias_trace(float(item["I_target_uA"]), params=params, settle_ns=settle_ns, seed=10_000 + seed + idx)
            f_mhz, _, gain_db, _ = small_signal_gain_spectrum(bias_out, params=params, freq_mhz=freq_grid_mhz)
        if f_mhz.size == 0:
            continue
        ax.plot(f_mhz, gain_db, color=cmap(norm(power_dbm)), linewidth=2.0)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Input Power (dBm)")
    ax.set_xscale("log")
    ax.set_xlim(float(freq_grid_mhz[0]), float(freq_grid_mhz[-1]))
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Gain (dB)")
    title = "Gain vs Frequency"
    if method == "small_signal_linearized":
        title += " (Small-Signal Linearized)"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_pulse_plot(trace: dict, out_path: Path) -> None:
    out = trace["out"]
    t_ns = out["t"] * 1e9
    i_uA = out["I_in"] * 1e6
    v_mV = out["V_vo2"] * 1e3

    fig, ax1 = plt.subplots(figsize=(8.2, 5.8))
    ax2 = ax1.twinx()
    ax1.plot(t_ns, i_uA, color="#00dd00", linewidth=2.2)
    ax2.plot(t_ns, v_mV, color="#8000aa", linewidth=2.0)
    ax1.set_xlim(-200, 600)
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("I_in ($\\mu$A)", color="#00dd00")
    ax2.set_ylabel("V_out (mV)", color="#8000aa")
    ax1.tick_params(axis="y", colors="#00dd00")
    ax2.tick_params(axis="y", colors="#8000aa")
    ax1.set_title(f"{int(round(trace['I_target_uA']))}uA")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_average_iv_plot(traces: list[dict], args: argparse.Namespace, out_path: Path) -> None:
    rows = []
    for item in traces:
        out = item["out"]
        t_ns = out["t"] * 1e9
        mask = (t_ns >= float(args.avg_start_ns)) & (t_ns <= float(args.avg_stop_ns))
        if not np.any(mask):
            continue
        i_avg = float(np.mean(out["I_in"][mask]) * 1e6)
        v_avg = float(np.mean(out["V_vo2"][mask]) * 1e3)
        rows.append((i_avg, v_avg))

    rows.sort(key=lambda x: x[0])
    x = [r[0] for r in rows]
    y = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.scatter(x, y, s=90, color="#f28c18")
    ax.set_xlabel("Average I_in ($\\mu$A)")
    ax.set_ylabel("Average V_out (mV)")
    ax.set_title(f"Average I_in  vs Average V_out    ({int(args.avg_start_ns)}-{int(args.avg_stop_ns)} ns)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    candidate_json = Path(args.candidate_json) if args.candidate_json else newest_candidate_json()
    resistance_preset = Path(args.resistance_preset)
    candidate = load_candidate(candidate_json)
    base_resist_params, start_branch = load_resistance_preset(resistance_preset)
    resist_params = candidate_resist_params(base_resist_params, candidate)
    params = build_params(args, candidate, resist_params, start_branch)
    traces = simulate_sweep(args, params)
    freq_grid_mhz = np.logspace(
        np.log10(float(args.freq_min_mhz)),
        np.log10(float(args.freq_max_mhz)),
        int(max(args.freq_points_per_decade, 4) * max(np.log10(args.freq_max_mhz) - np.log10(args.freq_min_mhz), 1.0)),
    )

    rep_current = min(traces, key=lambda item: abs(item["I_target_uA"] - float(args.representative_current)))

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"experiment_like_plots_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    freq_diagnostics = make_linear_gain_plot(
        traces,
        out_dir / "linear_gain_vs_frequency.png",
        method=args.freq_method,
        params=params,
        settle_ns=float(args.freq_settle_ns),
        freq_grid_mhz=freq_grid_mhz,
        seed=int(args.seed),
    )
    make_db_gain_plot(
        traces,
        out_dir / "gain_db_vs_frequency.png",
        method=args.freq_method,
        params=params,
        settle_ns=float(args.freq_settle_ns),
        freq_grid_mhz=freq_grid_mhz,
        seed=int(args.seed),
    )
    make_pulse_plot(rep_current, out_dir / "pulse_trace.png")
    make_average_iv_plot(traces, args, out_dir / "average_iv.png")

    meta = {
        "candidate_json": str(candidate_json),
        "resistance_preset": str(resistance_preset),
        "representative_current_uA": float(rep_current["I_target_uA"]),
        "dt_ns": float(candidate["dt_ns"]),
        "C_pF": float(candidate["C_pF"]),
        "C_th_mW_ns_per_K": float(candidate["C_th_mW_ns_per_K"]),
        "S_e_mW_per_K": float(candidate["S_e_mW_per_K"]),
        "T0_K": float(candidate["T0_K"]),
        "T_init_K": float(candidate["T_init_K"]),
        "sigma_W_sqrt_s": float(candidate.get("sigma_W_sqrt_s", 0.0)),
        "Rm_factor_scale": float(candidate.get("Rm_factor_scale", 1.0)),
        "Tc_shift_K": float(candidate.get("Tc_shift_K", 0.0)),
        "w_scale": float(candidate.get("w_scale", 1.0)),
        "beta_scale": float(candidate.get("beta_scale", 1.0)),
        "reversal_threshold_K": float(candidate.get("reversal_threshold_K", base_resist_params.reversal_threshold_K)),
        "freq_method": str(args.freq_method),
        "freq_settle_ns": float(args.freq_settle_ns),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "frequency_diagnostics.json").write_text(json.dumps(freq_diagnostics, indent=2))
    print(out_dir)


if __name__ == "__main__":
    main()
