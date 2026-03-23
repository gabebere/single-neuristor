"""
Current-driven VO2 neuristor simulation (separate from voltage-driven workflow).

Quick start:
1) Run the default sweep and create GIF:
   python current_drive_sim.py
2) Programmatic use:
   from current_drive_sim import CurrentDriveParams, simulate_current_step, run_sweep_make_gif
   out = simulate_current_step(500.0, CurrentDriveParams())
   run_sweep_make_gif(params=CurrentDriveParams(), seed=123)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from .model import HysteresisArray, YuanhangResistParams


_EPS = 1e-12
_SIM_DTYPE = np.float32


@dataclass
class CurrentDriveParams:
    """Parameter bundle for single-device current-driven simulation."""

    # Time
    dt_s: float = 10e-9
    t_end_s: float = 600e-9
    t_pre_s: float = 0.0
    pulse_on_s: float = 0.0
    pulse_off_s: Optional[float] = None

    # Initial conditions
    V_init_V: float = 0.0
    T0_K: float = 325.0
    T_init_K: float = 324.9

    # Electrical
    C_F: float = 145.34619293e-12
    # Legacy compatibility field. Current-drive simulation always assumes an ideal
    # imposed source current, so this value is ignored and forced to 0 internally.
    R_out_ohm: float = 0.0

    # Thermal
    C_th_J_per_K: float = 49.62776831e-12
    S_e_W_per_K: float = 0.20558726e-3
    sigma_W_sqrt_s: float = 0.0
    thermal_mode: str = "single"
    C_sub_J_per_K: float = 49.62776831e-12
    G_hot_sub_W_per_K: float = 0.20558726e-3
    T_sub_init_K: Optional[float] = None

    # Hysteresis / resistance
    resist_params: YuanhangResistParams = field(default_factory=YuanhangResistParams)
    start_branch: str = "insulator"

    def __post_init__(self) -> None:
        self.R_out_ohm = 0.0
        mode = self.thermal_mode.lower()
        if mode not in {"single", "double"}:
            raise ValueError("thermal_mode must be 'single' or 'double'.")
        self.thermal_mode = mode


def current_drive_numerics_report(
    params: CurrentDriveParams,
    *,
    I_peak_uA: float,
) -> Dict[str, float]:
    """Estimate fast/initial electrical timescales and step safety for a current-drive run."""

    h = HysteresisArray(params.resist_params, size=1, start_branch=params.start_branch)
    T0 = np.asarray([float(params.T_init_K)], dtype=float)
    h.initialize(T0)
    R_init = float(h.evaluate(T0)[0][0])

    R_metal = max(float(params.resist_params.Rm), _EPS)
    R_eff_init = max(R_init, _EPS)
    R_eff_fast = max(R_metal, _EPS)
    tau_init_s = float(params.C_F) * R_eff_init
    tau_fast_s = float(params.C_F) * R_eff_fast

    I_peak_A = abs(float(I_peak_uA)) * 1e-6
    V_fast_est = I_peak_A * R_eff_fast
    P_fast_est = (V_fast_est * V_fast_est) / max(R_metal, _EPS)
    C_th = max(float(params.C_th_J_per_K), _EPS)
    dt_s = float(params.dt_s)
    dT_step_est_K = (dt_s / C_th) * P_fast_est
    reversal_thr = max(float(params.resist_params.reversal_threshold_K), _EPS)

    return {
        "R_init_ohm": float(R_init),
        "R_metal_ohm": float(R_metal),
        "R_eff_init_ohm": float(R_eff_init),
        "R_eff_fast_ohm": float(R_eff_fast),
        "tau_init_s": float(tau_init_s),
        "tau_fast_s": float(tau_fast_s),
        "dt_over_tau_init": float(dt_s / max(tau_init_s, _EPS)),
        "dt_over_tau_fast": float(dt_s / max(tau_fast_s, _EPS)),
        "I_peak_uA": float(I_peak_uA),
        "dT_step_est_K": float(dT_step_est_K),
        "reversal_threshold_K": float(reversal_thr),
        "dT_step_over_reversal": float(dT_step_est_K / reversal_thr),
    }


def current_drive_report_messages(report: Dict[str, float]) -> List[str]:
    """Human-readable warnings for numerically risky current-drive settings."""

    msgs: List[str] = []
    dt_tau_fast = float(report["dt_over_tau_fast"])
    tau_fast_ns = float(report["tau_fast_s"]) * 1e9
    if dt_tau_fast > 0.1:
        target_ns = 0.1 * tau_fast_ns
        msgs.append(f"dt/tau_fast={dt_tau_fast:.3g} (>0.1). Target dt <= {target_ns:.3g} ns.")
    elif dt_tau_fast > 0.03:
        msgs.append(f"dt/tau_fast={dt_tau_fast:.3g}. This is borderline for accurate waveform shape.")

    dt_tau_init = float(report["dt_over_tau_init"])
    if dt_tau_init > 0.2:
        msgs.append(f"dt/tau_init={dt_tau_init:.3g} (>0.2). Euler artifacts are likely.")

    dT_over_rev = float(report["dT_step_over_reversal"])
    if dT_over_rev > 1.0:
        msgs.append(
            f"Estimated thermal jump per step is {dT_over_rev:.3g}x reversal threshold; "
            "minor-loop reversals may be skipped."
        )
    elif dT_over_rev > 0.2:
        msgs.append(
            f"Estimated thermal jump per step is {dT_over_rev:.3g}x reversal threshold; "
            "hysteresis timing may be dt-sensitive."
        )
    return msgs


def stabilize_current_drive_params(
    params: CurrentDriveParams,
    *,
    I_peak_uA: float,
    dt_tau_target: float = 0.05,
    dT_ratio_target: float = 0.2,
    min_dt_s: float = 1e-13,
) -> tuple[CurrentDriveParams, Dict[str, float]]:
    """Return a copy of params with a safer dt if the current settings are under-resolved."""

    report = current_drive_numerics_report(params, I_peak_uA=I_peak_uA)
    dt_before_s = max(float(params.dt_s), min_dt_s)
    targets = [dt_before_s]

    dt_tau_fast = float(report["dt_over_tau_fast"])
    if dt_tau_fast > dt_tau_target:
        tau_fast_s = float(report["tau_fast_s"])
        targets.append(max(min_dt_s, float(dt_tau_target) * tau_fast_s))

    dT_over_rev = float(report["dT_step_over_reversal"])
    if dT_over_rev > dT_ratio_target:
        targets.append(max(min_dt_s, dt_before_s * (float(dT_ratio_target) / dT_over_rev)))

    dt_after_s = min(targets)
    if dt_after_s < dt_before_s * 0.98:
        params = replace(params, dt_s=float(f"{dt_after_s:.16g}"))
    return params, report


def reference_visual_pulse_params() -> CurrentDriveParams:
    """
    Empirical pulse preset tuned for reference-like waveform shape.

    Note:
    - This is intended for qualitative visual matching of the uploaded pulse plots.
    - Diagnostics may warn about coarse-step Euler stability for this preset.
    """

    rp = YuanhangResistParams()
    rp.Rm_factor = 0.2
    return CurrentDriveParams(
        dt_s=10e-9,
        t_pre_s=200e-9,
        t_end_s=800e-9,
        pulse_on_s=0.0,
        pulse_off_s=300e-9,
        V_init_V=0.0,
        T0_K=342.0,
        T_init_K=341.9,
        C_F=10e-12,
        R_out_ohm=0.0,
        C_th_J_per_K=49.62776831e-12 * 0.005,
        S_e_W_per_K=0.20558726e-3,
        sigma_W_sqrt_s=0.0,
        resist_params=rp,
        start_branch="insulator",
    )


def lab_pulse_current_params() -> CurrentDriveParams:
    """Finite-pulse preset for comparing against lab traces of the single-device current experiment."""

    return CurrentDriveParams(
        dt_s=10e-9,
        t_pre_s=200e-9,
        t_end_s=600e-9,
        pulse_on_s=0.0,
        pulse_off_s=300e-9,
        V_init_V=0.0,
        T0_K=325.0,
        T_init_K=324.9,
        C_F=145.34619293e-12,
        R_out_ohm=0.0,
        C_th_J_per_K=49.62776831e-12,
        S_e_W_per_K=0.20558726e-3,
        sigma_W_sqrt_s=0.0,
        resist_params=YuanhangResistParams(),
        start_branch="insulator",
    )


class HysteresisSingleAdapter:
    """
    Thin scalar wrapper around existing HysteresisArray.

    Provides the required interface:
    - reset(T0)
    - evaluate(T) -> (R_ohm, g)
    - update(T_prev, T_new)
    """

    def __init__(self, resist_params: YuanhangResistParams, start_branch: str = "insulator") -> None:
        self._h = HysteresisArray(resist_params, size=1, start_branch=start_branch)

    def reset(self, T0: float) -> None:
        self._h.initialize(np.asarray([float(T0)], dtype=_SIM_DTYPE))

    def evaluate(self, T: float) -> tuple[float, float]:
        R_arr, g_arr = self._h.evaluate(np.asarray([float(T)], dtype=_SIM_DTYPE))
        return float(R_arr[0]), float(g_arr[0])

    def update(self, T_prev: float, T_new: float) -> None:
        # Keep behavior identical to existing hysteresis logic while honoring
        # the explicit (T_prev, T_new) update contract from the current-drive spec.
        T_prev_f = float(T_prev)
        T_new_f = float(T_new)
        if hasattr(self._h, "_update_reversal"):
            if hasattr(self._h, "T_last"):
                self._h.T_last = np.asarray([T_prev_f], dtype=_SIM_DTYPE)
            self._h._update_reversal(np.asarray([T_new_f], dtype=_SIM_DTYPE))
        else:
            # Fallback path if internals change in the future.
            self._h.evaluate(np.asarray([T_new_f], dtype=_SIM_DTYPE))


def _time_grid(dt_s: float, t_end_s: float, t_pre_s: float) -> np.ndarray:
    total = float(t_pre_s) + float(t_end_s)
    if dt_s <= 0.0 or total <= 0.0:
        raise ValueError("dt_s and (t_pre_s + t_end_s) must be positive.")
    n_steps = int(np.floor(total / dt_s)) + 1
    return np.linspace(-t_pre_s, -t_pre_s + dt_s * (n_steps - 1), n_steps, dtype=_SIM_DTYPE)


def _build_current_waveform(
    t: np.ndarray,
    I_target_A: float,
    pulse_on_s: float = 0.0,
    pulse_off_s: Optional[float] = None,
) -> np.ndarray:
    if pulse_off_s is None:
        return np.where(t >= pulse_on_s, I_target_A, 0.0).astype(_SIM_DTYPE, copy=False)
    return np.where((t >= pulse_on_s) & (t < pulse_off_s), I_target_A, 0.0).astype(_SIM_DTYPE, copy=False)


def _simulate_with_current_trace(
    t: np.ndarray,
    I_in: np.ndarray,
    params: CurrentDriveParams,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Integrate the current-drive model for an explicit source-current trace on grid `t`."""

    if t.ndim != 1 or I_in.ndim != 1 or t.size != I_in.size:
        raise ValueError("t and I_in must be 1D arrays of equal length.")

    n = t.size
    I_in = np.asarray(I_in, dtype=_SIM_DTYPE)

    V = np.zeros(n, dtype=_SIM_DTYPE)
    T = np.zeros(n, dtype=_SIM_DTYPE)
    T_sub = np.zeros(n, dtype=_SIM_DTYPE)
    R = np.zeros(n, dtype=_SIM_DTYPE)
    P = np.zeros(n, dtype=_SIM_DTYPE)

    V[0] = _SIM_DTYPE(params.V_init_V)
    T[0] = _SIM_DTYPE(params.T_init_K)
    T_sub_init = params.T0_K if params.T_sub_init_K is None else params.T_sub_init_K
    T_sub[0] = _SIM_DTYPE(T_sub_init)

    hyst = HysteresisSingleAdapter(params.resist_params, start_branch=params.start_branch)
    hyst.reset(T[0])
    rng = np.random.default_rng(seed)

    C = _SIM_DTYPE(max(float(params.C_F), _EPS))
    C_th = _SIM_DTYPE(max(float(params.C_th_J_per_K), _EPS))
    S_e = _SIM_DTYPE(params.S_e_W_per_K)
    sigma = _SIM_DTYPE(params.sigma_W_sqrt_s)
    dt = _SIM_DTYPE(params.dt_s)
    T0 = _SIM_DTYPE(params.T0_K)
    thermal_mode = params.thermal_mode
    C_sub = _SIM_DTYPE(max(float(params.C_sub_J_per_K), _EPS))
    G_hot_sub = _SIM_DTYPE(max(float(params.G_hot_sub_W_per_K), 0.0))

    for k in range(n - 1):
        R_k, _ = hyst.evaluate(T[k])
        R_k = max(R_k, _EPS)
        R_k_sim = _SIM_DTYPE(R_k)
        P_k = (V[k] * V[k]) / R_k_sim

        dV = (dt / C) * (I_in[k] - V[k] / R_k_sim)
        V_next = V[k] + dV

        if thermal_mode == "double":
            heat_flow = G_hot_sub * (T[k] - T_sub[k])
            dT_det = (dt / C_th) * (P_k - heat_flow)
            dT_sub_det = (dt / C_sub) * (heat_flow - S_e * (T_sub[k] - T0))
            T_sub_next = T_sub[k] + dT_sub_det
        else:
            dT_det = (dt / C_th) * (P_k - S_e * (T[k] - T0))
            T_sub_next = T_sub[k]
        dT_sto = (sigma / C_th) * np.sqrt(dt) * _SIM_DTYPE(rng.standard_normal()) if sigma > 0.0 else _SIM_DTYPE(0.0)
        T_next = T[k] + dT_det + dT_sto

        hyst.update(T_prev=T[k], T_new=T_next)

        R[k] = R_k_sim
        P[k] = P_k
        V[k + 1] = V_next
        T[k + 1] = T_next
        T_sub[k + 1] = T_sub_next

    R_end, _ = hyst.evaluate(T[-1])
    R[-1] = _SIM_DTYPE(max(R_end, _EPS))
    P[-1] = (V[-1] * V[-1]) / R[-1]

    return {
        "t": t,
        "I_in": I_in,
        "V_vo2": V,
        "T": T,
        "T_hot": T,
        "T_sub": T_sub,
        "R": R,
        "P": P,
    }


def simulate_current_waveform(
    I_uA: np.ndarray,
    params: CurrentDriveParams,
    waveform_time_s: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Simulate a current-drive experiment for an arbitrary source-current waveform."""

    t = _time_grid(params.dt_s, params.t_end_s, params.t_pre_s)
    I_trace_uA = np.asarray(I_uA, dtype=float).reshape(-1)
    if waveform_time_s is None:
        if I_trace_uA.size != t.size:
            raise ValueError("I_uA must match the internal time-grid length when waveform_time_s is omitted.")
        I_in = np.asarray(I_trace_uA * 1e-6, dtype=_SIM_DTYPE)
    else:
        t_wave = np.asarray(waveform_time_s, dtype=float).reshape(-1)
        if t_wave.size != I_trace_uA.size:
            raise ValueError("waveform_time_s and I_uA must have the same length.")
        if t_wave.size < 2:
            raise ValueError("waveform_time_s must contain at least two samples.")
        I_interp_uA = np.interp(t.astype(float), t_wave, I_trace_uA, left=I_trace_uA[0], right=I_trace_uA[-1])
        I_in = np.asarray(I_interp_uA * 1e-6, dtype=_SIM_DTYPE)
    return _simulate_with_current_trace(t=t, I_in=I_in, params=params, seed=seed)


def simulate_current_step(I_uA: float, params: CurrentDriveParams, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Simulate a single current-step experiment.

    Uses:
      V_{n+1} = V_n + (dt/C)*(I_in - V/R)
      T_{n+1} = T_n + (dt/C_th)*(V^2/R - S_e*(T-T0)) + (sigma/C_th)*sqrt(dt)*N(0,1)
    """

    t = _time_grid(params.dt_s, params.t_end_s, params.t_pre_s)
    I_target_A = _SIM_DTYPE(float(I_uA) * 1e-6)
    I_in = _build_current_waveform(
        t=t,
        I_target_A=I_target_A,
        pulse_on_s=float(params.pulse_on_s),
        pulse_off_s=params.pulse_off_s,
    )
    return _simulate_with_current_trace(t=t, I_in=I_in, params=params, seed=seed)


def _count_turns(signal: np.ndarray) -> int:
    if signal.size < 3:
        return 0
    d = np.diff(signal)
    return int(np.sum((d[:-1] * d[1:]) < 0.0))


def diagnose_current_step(
    I_uA: float,
    params: CurrentDriveParams,
    seed: Optional[int] = None,
    fit_window_ns: float = 100.0,
    debug_steps: int = 6,
) -> Dict[str, object]:
    """
    Run a simulation and expose quick diagnostics for RC-vs-switching behavior.
    """

    out = simulate_current_step(I_uA=I_uA, params=params, seed=seed)
    t = out["t"]
    V = out["V_vo2"]
    T = out["T"]
    R = out["R"]
    I_in = out["I_in"]
    C = max(float(params.C_F), _EPS)

    fit_mask = (t >= 0.0) & (t <= float(fit_window_ns) * 1e-9)
    if np.sum(fit_mask) >= 2:
        slope = float(np.polyfit(t[fit_mask], V[fit_mask], 1)[0])
    else:
        slope = float("nan")

    I_step_A = float(I_uA) * 1e-6
    I_over_C = I_step_A / C
    ratio = slope / I_over_C if np.isfinite(slope) and abs(I_over_C) > 0.0 else float("nan")

    hyster = HysteresisSingleAdapter(params.resist_params, start_branch=params.start_branch)
    hyster.reset(params.T_init_K)
    R_init, _ = hyster.evaluate(params.T_init_K)
    tau_init_s = C * max(float(R_init), _EPS)

    rows: List[Dict[str, float]] = []
    n = min(int(debug_steps), len(t))
    for k in range(n):
        Vk = float(V[k])
        Rk = max(float(R[k]), _EPS)
        Ik = float(I_in[k])
        I_vo2 = Vk / Rk
        I_cap = Ik - I_vo2
        dVdt = I_cap / C
        rows.append(
            {
                "k": float(k),
                "t_ns": float(t[k] * 1e9),
                "T_K": float(T[k]),
                "R_ohm": Rk,
                "V_V": Vk,
                "I_in_A": Ik,
                "I_vo2_A": I_vo2,
                "I_cap_A": I_cap,
                "dVdt_V_per_s": dVdt,
            }
        )

    turns = _count_turns(V[t >= 0.0])
    warnings: List[str] = []
    if np.isfinite(ratio) and ratio > 0.9 and turns == 0:
        warnings.append("Early-time slope is close to I/C with no turning points: behavior is RC/integrator-like in this window.")
    dt_over_tau = float(params.dt_s) / max(tau_init_s, _EPS)
    if dt_over_tau > 0.2:
        warnings.append(
            "dt is large relative to the initial electrical time constant (dt/tau > 0.2); reduce dt to avoid Euler artifacts."
        )
    return {
        "params": {
            "C_F": C,
            "R_init_ohm": float(R_init),
            "I_step_A": I_step_A,
            "pulse_on_s": float(params.pulse_on_s),
            "pulse_off_s": None if params.pulse_off_s is None else float(params.pulse_off_s),
            "tau_init_s": tau_init_s,
            "dt_over_tau_init": dt_over_tau,
        },
        "slope_check": {
            "fit_window_ns": float(fit_window_ns),
            "slope_V_per_s": slope,
            "I_over_C_V_per_s": I_over_C,
            "ratio_slope_to_I_over_C": ratio,
        },
        "turn_count_from_t_ge_0": turns,
        "warnings": warnings,
        "first_steps": rows,
        "output": out,
    }


def _plot_current_step(out: Dict[str, np.ndarray], I_uA: float, frame_path: Path) -> None:
    t_ns = out["t"] * 1e9
    I_uA_trace = out["I_in"] * 1e6
    V_mV = out["V_vo2"] * 1e3

    fig, ax1 = plt.subplots(figsize=(8.5, 4.6))
    ax2 = ax1.twinx()

    ax1.step(t_ns, I_uA_trace, where="post", color="green", linewidth=2.0, label="I_in")
    ax2.plot(t_ns, V_mV, color="purple", linewidth=2.0, label="V_vo2")

    ax1.set_xlabel("time (ns)")
    ax1.set_ylabel("I_in (uA)", color="green")
    ax2.set_ylabel("V_vo2 (mV)", color="purple")
    ax1.tick_params(axis="y", labelcolor="green")
    ax2.tick_params(axis="y", labelcolor="purple")
    ax1.grid(True, alpha=0.25)
    ax1.set_title(f"I_in = {I_uA:.0f} uA")

    lines = [ax1.lines[0], ax2.lines[0]]
    labels = ["I_in", "V_vo2"]
    ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(frame_path, dpi=180)
    plt.close(fig)


def run_sweep_make_gif(
    params: CurrentDriveParams,
    I_start_uA: int = 50,
    I_stop_uA: int = 2000,
    I_step_uA: int = 50,
    frame_duration_s: float = 0.5,
    frames_dir: str | Path = "outputs/current_sweep_frames",
    gif_path: str | Path = "outputs/current_sweep.gif",
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Run the current sweep, save PNG frames, and create GIF."""

    if I_step_uA <= 0:
        raise ValueError("I_step_uA must be positive.")
    if I_stop_uA < I_start_uA:
        raise ValueError("I_stop_uA must be >= I_start_uA.")

    params_before_dt = float(params.dt_s)
    i_peak_uA = max(abs(int(I_start_uA)), abs(int(I_stop_uA)))
    params, report = stabilize_current_drive_params(params, I_peak_uA=float(i_peak_uA))
    if float(params.dt_s) < params_before_dt * 0.98:
        print(
            "[current_sweep] auto-adjusted dt "
            f"from {params_before_dt * 1e9:.4g} ns to {float(params.dt_s) * 1e9:.4g} ns"
        )
    for msg in current_drive_report_messages(report):
        print(f"[current_sweep][warning] {msg}")

    frame_dir_path = Path(frames_dir)
    gif_path = Path(gif_path)
    frame_dir_path.mkdir(parents=True, exist_ok=True)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    currents = list(range(int(I_start_uA), int(I_stop_uA) + 1, int(I_step_uA)))
    frame_paths: List[Path] = []
    turn_counts: List[int] = []

    for idx, i_uA in enumerate(currents):
        run_seed = None if seed is None else int(seed) + idx
        out = simulate_current_step(float(i_uA), params=params, seed=run_seed)
        frame_path = frame_dir_path / f"frame_{idx:03d}_I{i_uA:04d}uA.png"
        _plot_current_step(out, float(i_uA), frame_path)
        frame_paths.append(frame_path)
        turn_counts.append(_count_turns(out["V_vo2"][out["t"] >= 0.0]))

    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, duration=float(frame_duration_s), loop=0)

    print(f"Saved {len(frame_paths)} frames to: {frame_dir_path}")
    print(f"Saved GIF to: {gif_path}")

    return {
        "currents_uA": np.asarray(currents, dtype=int),
        "frame_paths": [str(p) for p in frame_paths],
        "gif_path": str(gif_path),
        "turn_counts": np.asarray(turn_counts, dtype=int),
    }


def main() -> None:
    results = run_sweep_make_gif(
        params=CurrentDriveParams(),
        I_start_uA=50,
        I_stop_uA=2000,
        I_step_uA=50,
        frame_duration_s=0.5,
        frames_dir="outputs/current_sweep_frames",
        gif_path="outputs/current_sweep.gif",
        seed=123,
    )
    print("Done.")
    print(f"GIF: {results['gif_path']}")


if __name__ == "__main__":
    main()
