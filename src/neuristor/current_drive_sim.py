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

from .model import HysteresisArray, YuanhangResistParams, resistance_from_semiconducting_weight


_EPS = 1e-12
_SIM_DTYPE = np.float32
LEGACY_CURRENT_DRIVE_PARAM_KEYS = frozenset(
    {
        "thermal_mode",
        "C_sub_J_per_K",
        "G_hot_sub_W_per_K",
        "T_sub_init_K",
        "phase_mode",
        "tau_g_s",
        "domain_count",
        "domain_temperature_span_K",
        "domain_coupling_W_per_K",
        "hysteresis_reversal_mode",
    }
)


def sanitize_current_drive_params(payload: Dict[str, object]) -> Dict[str, object]:
    """Drop obsolete current-source mode fields from a persisted job/preset payload."""

    cleaned = dict(payload)
    for key in LEGACY_CURRENT_DRIVE_PARAM_KEYS:
        cleaned.pop(key, None)
    return cleaned


@dataclass
class CurrentDriveParams:
    """Parameters for the supported Yuanhang ideal-current-source model."""

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

    # Hysteresis / resistance
    resist_params: YuanhangResistParams = field(default_factory=YuanhangResistParams)
    start_branch: str = "insulator"
    # Experimental normalized hot-filament/cold-surroundings conductance split.
    # One is the original single-region model, with its original arithmetic.
    filament_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.filament_fraction) or not 0 < self.filament_fraction <= 1:
            raise ValueError("filament_fraction must be finite and in (0, 1]")
        self.R_out_ohm = 0.0
        if float(self.C_F) < 0.0:
            raise ValueError("C_F must be non-negative; use C_F=0 for the algebraic RC=0 limit.")
        if float(self.C_th_J_per_K) <= 0.0:
            raise ValueError("C_th_J_per_K must be positive.")
        if float(self.S_e_W_per_K) < 0.0:
            raise ValueError("S_e_W_per_K must be non-negative.")


def _advance_voltage_exact(
    V: np.ndarray | np.float32,
    I_in: np.ndarray | np.float32,
    R: np.ndarray | np.float32,
    *,
    dt_s: float,
    C_F: float | np.ndarray,
) -> np.ndarray | np.float32:
    """Advance ``C dV/dt = I - V/R`` exactly while I and R are frozen."""

    target = np.asarray(I_in, dtype=_SIM_DTYPE) * np.asarray(R, dtype=_SIM_DTYPE)
    if np.ndim(C_F) == 0 and float(C_F) == 0.0:
        return target
    capacitance = np.asarray(C_F, dtype=_SIM_DTYPE)
    tau = capacitance * np.asarray(R, dtype=_SIM_DTYPE)
    with np.errstate(divide="ignore"):
        alpha = -np.expm1(-_SIM_DTYPE(float(dt_s)) / tau)
    return np.asarray(V, dtype=_SIM_DTYPE) + alpha * (target - np.asarray(V, dtype=_SIM_DTYPE))


def _advance_temperature_exact(
    T: np.ndarray | np.float32,
    P: np.ndarray | np.float32,
    *,
    T0_K: float,
    dt_s: float,
    C_th_J_per_K: float | np.ndarray,
    S_e_W_per_K: float | np.ndarray,
) -> np.ndarray | np.float32:
    """Advance the linear cooling equation exactly while Joule power is frozen."""

    T_arr = np.asarray(T, dtype=_SIM_DTYPE)
    P_arr = np.asarray(P, dtype=_SIM_DTYPE)
    C_th = np.asarray(C_th_J_per_K, dtype=_SIM_DTYPE)
    S_e = np.asarray(S_e_W_per_K, dtype=_SIM_DTYPE)
    dt = _SIM_DTYPE(float(dt_s))
    if np.ndim(S_e) == 0 and float(S_e) == 0.0:
        return T_arr + (dt / C_th) * P_arr
    alpha = -np.expm1(-(dt * S_e) / C_th)
    if np.ndim(S_e) == 0:
        target = _SIM_DTYPE(float(T0_K)) + P_arr / S_e
        return T_arr + alpha * (target - T_arr)
    # The zero-conductance columns retain their exact no-cooling limit.
    positive = S_e > 0
    target = _SIM_DTYPE(float(T0_K)) + P_arr / np.where(positive, S_e, _SIM_DTYPE(1.0))
    return np.where(positive, T_arr + alpha * (target - T_arr), T_arr + (dt / C_th) * P_arr)


def current_drive_numerics_report(
    params: CurrentDriveParams,
    *,
    I_peak_uA: float,
) -> Dict[str, float]:
    """Estimate fast/initial electrical timescales and step safety for a current-drive run."""

    if params.filament_fraction != 1:
        raise ValueError("Single-region operating estimates do not apply to the experimental filament model")

    h = HysteresisArray(
        params.resist_params,
        size=1,
        start_branch=params.start_branch,
    )
    T0 = np.asarray([float(params.T_init_K)], dtype=float)
    h.initialize(T0)
    R_init = float(h.evaluate(T0)[0][0])

    R_metal = max(float(params.resist_params.Rm), _EPS)
    R_eff_init = max(R_init, _EPS)
    R_eff_fast = max(R_metal, _EPS)
    algebraic_voltage = float(params.C_F) == 0.0
    tau_init_s = float(params.C_F) * R_eff_init
    tau_fast_s = float(params.C_F) * R_eff_fast

    I_peak_A = abs(float(I_peak_uA)) * 1e-6
    V_fast_est = I_peak_A * R_eff_fast
    P_fast_est = (V_fast_est * V_fast_est) / max(R_metal, _EPS)
    # CurrentDriveParams already requires C_th > 0. The specimen estimate is
    # below 1e-12 J/K, so a generic denominator floor would hide thermal jumps.
    C_th = float(params.C_th_J_per_K)
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
        "dt_over_tau_init": 0.0 if algebraic_voltage else float(dt_s / max(tau_init_s, _EPS)),
        "dt_over_tau_fast": 0.0 if algebraic_voltage else float(dt_s / max(tau_fast_s, _EPS)),
        "algebraic_voltage": float(1.0 if algebraic_voltage else 0.0),
        "tau_thermal_s": (
            float(params.C_th_J_per_K) / float(params.S_e_W_per_K)
            if float(params.S_e_W_per_K) > 0.0
            else float("inf")
        ),
        "V_metal_floor_V": float(I_peak_A * R_metal),
        "I_peak_uA": float(I_peak_uA),
        "dT_step_est_K": float(dT_step_est_K),
        "reversal_threshold_K": float(reversal_thr),
        "dT_step_over_reversal": float(dT_step_est_K / reversal_thr),
    }


def current_drive_operating_estimates(params: CurrentDriveParams, *, I_uA: float) -> Dict[str, float]:
    """Return interpretable timescale, voltage-floor, and thermal-only window estimates.

    The current bounds use the centers of the heating and cooling transitions
    (where the corresponding major-branch semiconducting fraction is 1/2).
    They are diagnostics, not replacements for a time-domain sweep.
    """

    if params.filament_fraction != 1:
        raise ValueError("Single-region operating estimates do not apply to the experimental filament model")
    rp = params.resist_params
    T_heat = float(rp.Tc_K + 0.5 * rp.w_eff)
    T_cool = float(rp.Tc_K - 0.5 * rp.w_eff)
    R_heat = float(rp.Rm + 0.5 * rp.R0 * np.exp(rp.Ea_over_k / T_heat))
    R_cool = float(rp.Rm + 0.5 * rp.R0 * np.exp(rp.Ea_over_k / T_cool))
    S_e = float(params.S_e_W_per_K)
    T0 = float(params.T0_K)

    heat_power = max(S_e * (T_heat - T0), 0.0)
    cool_power = max(S_e * (T_cool - T0), 0.0)
    I_heat_A = float(np.sqrt(heat_power / max(R_heat, _EPS)))
    I_cool_A = float(np.sqrt(cool_power / max(R_cool, _EPS)))
    I_A = abs(float(I_uA)) * 1e-6
    tau_th_s = float(params.C_th_J_per_K / S_e) if S_e > 0.0 else float("inf")

    return {
        "T_heating_transition_K": T_heat,
        "T_cooling_transition_K": T_cool,
        "R_heating_transition_ohm": R_heat,
        "R_cooling_transition_ohm": R_cool,
        "thermal_only_lower_current_uA": I_heat_A * 1e6,
        "thermal_only_upper_current_uA": I_cool_A * 1e6,
        "thermal_only_window_exists": float(1.0 if I_heat_A < I_cool_A else 0.0),
        "tau_thermal_s": tau_th_s,
        "tau_metal_s": float(params.C_F) * float(rp.Rm),
        "tau_metal_over_tau_thermal": (
            float(params.C_F) * float(rp.Rm) / tau_th_s
            if np.isfinite(tau_th_s) and tau_th_s > 0.0
            else float("nan")
        ),
        "V_metal_floor_V": I_A * float(rp.Rm),
    }


def current_drive_report_messages(report: Dict[str, float]) -> List[str]:
    """Human-readable warnings for numerically risky current-drive settings."""

    msgs: List[str] = []
    if float(report.get("algebraic_voltage", 0.0)) >= 0.5:
        msgs.append("C=0 selects the algebraic limit V=I_in*R(T); there is no electrical RC state.")

    dt_tau_fast = float(report["dt_over_tau_fast"])
    tau_fast_ns = float(report["tau_fast_s"]) * 1e9
    if dt_tau_fast > 0.1:
        target_ns = 0.1 * tau_fast_ns
        msgs.append(f"dt/tau_fast={dt_tau_fast:.3g} (>0.1). Resolve the coupled waveform with dt <= {target_ns:.3g} ns.")
    elif dt_tau_fast > 0.03:
        msgs.append(f"dt/tau_fast={dt_tau_fast:.3g}. This is borderline for accurate waveform shape.")

    dt_tau_init = float(report["dt_over_tau_init"])
    if dt_tau_init > 0.2:
        msgs.append(f"dt/tau_init={dt_tau_init:.3g} (>0.2). The initial RC transient is under-resolved.")

    dT_over_rev = float(report["dT_step_over_reversal"])
    if dT_over_rev > 1.0:
        msgs.append(
            f"Estimated thermal jump per step is {dT_over_rev:.3g}x the hysteresis deadband; "
            "transition timing may be under-resolved."
        )
    elif dT_over_rev > 0.2:
        msgs.append(
            f"Estimated thermal jump per step is {dT_over_rev:.3g}x the hysteresis deadband; "
            "use a smaller dt for accurate transition timing."
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
    """

    def __init__(
        self,
        resist_params: YuanhangResistParams,
        start_branch: str = "insulator",
    ) -> None:
        self._h = HysteresisArray(
            resist_params,
            size=1,
            start_branch=start_branch,
        )

    def reset(self, T0: float) -> None:
        self._h.initialize(np.asarray([float(T0)], dtype=_SIM_DTYPE))

    def evaluate(self, T: float) -> tuple[float, float]:
        R_arr, g_arr = self._h.evaluate(np.asarray([float(T)], dtype=_SIM_DTYPE))
        return float(R_arr[0]), float(g_arr[0])


def filament_effective_resistance(hot_reference_ohm, cold_reference_ohm, fraction):
    """Normalize parallel paths to the measured uniform-temperature R(T) law.

    G = f/R_fit(T_hot,H_hot) + (1-f)/R_fit(T0,H0). The weight is an
    effective conductance share, not an independently measured area fraction.
    The cold branch is held at ambient by an ideal thermal reservoir. Its
    Joule heat is not deposited in the filament. Equal reference resistances
    recover the measured R(T), and f=1 exactly preserves the original model.
    """
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("filament fraction must be in (0, 1]")
    hot, cold = np.asarray(hot_reference_ohm), np.asarray(cold_reference_ohm)
    if np.any(~np.isfinite(hot)) or np.any(hot <= 0) or np.any(~np.isfinite(cold)) or np.any(cold <= 0):
        raise ValueError("Reference resistances must be finite and positive")
    return _filament_resistance(hot, cold, fraction)


def _filament_resistance(hot, cold, fraction):
    # Validation is done once in the dataclass/public boundary, not each step.
    if fraction == 1.0:
        return hot
    return hot / (_SIM_DTYPE(fraction) + _SIM_DTYPE(1 - fraction) * hot / cold)


def filament_cold_reference(params):
    """Freeze ambient R(T0,H0) using an independent copy of the initial branch."""
    cold = HysteresisSingleAdapter(params.resist_params, params.start_branch)
    cold.reset(params.T0_K)
    return _SIM_DTYPE(cold.evaluate(params.T0_K)[0])


def _filament_channels(voltage, hot, cold, fraction, ambient):
    """Expose branch currents/powers; P remains total electrical dissipation."""
    hot_current = voltage * _SIM_DTYPE(fraction) / hot
    cold_current = voltage * _SIM_DTYPE(1 - fraction) / cold
    return dict(R_reference_hot=hot, R_reference_cold=np.full_like(hot, cold),
                I_filament=hot_current, I_surrounding=cold_current,
                P_filament=voltage * hot_current, P_surrounding=voltage * cold_current,
                T_surrounding=np.full_like(hot, ambient))


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
    """Integrate the Yuanhang ideal-current-source model on grid `t`."""

    if t.ndim != 1 or I_in.ndim != 1 or t.size != I_in.size:
        raise ValueError("t and I_in must be 1D arrays of equal length.")

    n = t.size
    I_in = np.asarray(I_in, dtype=_SIM_DTYPE)
    V = np.zeros(n, dtype=_SIM_DTYPE)
    T = np.zeros(n, dtype=_SIM_DTYPE)
    g_eq = np.zeros(n, dtype=_SIM_DTYPE)
    R = np.zeros(n, dtype=_SIM_DTYPE)
    P = np.zeros(n, dtype=_SIM_DTYPE)

    V[0] = _SIM_DTYPE(params.V_init_V)
    T[0] = _SIM_DTYPE(params.T_init_K)

    hyst = HysteresisSingleAdapter(
        params.resist_params,
        start_branch=params.start_branch,
    )
    hyst.reset(T[0])
    _, g0 = hyst.evaluate(T[0])
    g_eq[0] = _SIM_DTYPE(g0)
    rng = np.random.default_rng(seed)

    C = float(params.C_F)
    C_th = float(params.C_th_J_per_K)
    S_e = float(params.S_e_W_per_K)
    sigma = _SIM_DTYPE(params.sigma_W_sqrt_s)
    dt = _SIM_DTYPE(params.dt_s)
    T0 = _SIM_DTYPE(params.T0_K)
    fraction = float(params.filament_fraction)
    cold = filament_cold_reference(params) if fraction < 1 else None
    hot_reference = np.empty_like(R) if fraction < 1 else None

    for k in range(n - 1):
        R_eq_k, g_eq_k = hyst.evaluate(T[k])
        g_eq_k = float(np.clip(g_eq_k, 0.0, 1.0))
        g_eq[k] = _SIM_DTYPE(g_eq_k)
        R_k_sim = _SIM_DTYPE(max(float(R_eq_k), _EPS))
        if fraction < 1:
            hot_reference[k] = R_k_sim
            R_k_sim = _SIM_DTYPE(_filament_resistance(R_k_sim, cold, fraction))
        if C == 0.0:
            V[k] = _SIM_DTYPE(I_in[k] * R_k_sim)
        P_k = (V[k] * V[k]) / R_k_sim
        filament_power = P_k if fraction == 1 else V[k] * V[k] * _SIM_DTYPE(fraction) / hot_reference[k]

        V_next = _advance_voltage_exact(V[k], I_in[k], R_k_sim, dt_s=float(dt), C_F=C)
        T_next = _advance_temperature_exact(
            T[k],
            filament_power,
            T0_K=float(T0),
            dt_s=float(dt),
            C_th_J_per_K=C_th,
            S_e_W_per_K=S_e,
        )
        dT_sto = (sigma / C_th) * np.sqrt(dt) * _SIM_DTYPE(rng.standard_normal()) if sigma > 0.0 else _SIM_DTYPE(0.0)
        T_next = T_next + dT_sto

        R[k] = R_k_sim
        P[k] = P_k
        V[k + 1] = V_next
        T[k + 1] = T_next

    R_end_eq, g_end_eq = hyst.evaluate(T[-1])
    g_eq[-1] = _SIM_DTYPE(float(np.clip(g_end_eq, 0.0, 1.0)))
    R[-1] = _SIM_DTYPE(max(float(R_end_eq), _EPS))
    if fraction < 1:
        hot_reference[-1] = R[-1]
        R[-1] = _SIM_DTYPE(_filament_resistance(R[-1], cold, fraction))
    if C == 0.0:
        V[-1] = _SIM_DTYPE(I_in[-1] * R[-1])
    P[-1] = (V[-1] * V[-1]) / R[-1]

    return {
        "t": t,
        "I_in": I_in,
        "V_vo2": V,
        "T": T,
        "T_hot": T,
        "T_sub": T.copy(),
        "g_eq": g_eq,
        "g_dyn": g_eq.copy(),
        "R": R,
        "P": P,
        **(_filament_channels(V, hot_reference, cold, fraction, T0) if fraction < 1 else {}),
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

    Uses the same continuous equations as the Euler form below, but advances
    each frozen-coefficient RC/cooling subproblem with its exact exponential:

      C dV/dt = I_in - V/R
      C_th dT/dt = V^2/R - S_e*(T-T0) + noise

    ``C_F=0`` selects the algebraic limit ``V=I_in*R(T)``.
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


def _simulate_with_current_matrix(
    t: np.ndarray,
    I_in: np.ndarray,
    params: CurrentDriveParams,
    *, audit_proximity: bool = False,
    parameter_columns: Dict[str, np.ndarray] | None = None,
) -> List[Dict[str, np.ndarray]]:
    """Integrate independent deterministic waveforms in one vectorized pass."""

    if I_in.ndim != 2 or I_in.shape[0] != t.size:
        raise ValueError("I_in must have shape (time, independent waveforms)")
    if float(params.sigma_W_sqrt_s) != 0.0:
        raise ValueError("Vectorized arbitrary waveforms require zero thermal noise")
    n_steps, n_currents = I_in.shape
    if n_currents == 0:
        return []
    I_in = np.asarray(I_in, dtype=_SIM_DTYPE)
    shape = (n_steps, n_currents)
    V = np.zeros(shape, dtype=_SIM_DTYPE)
    T = np.zeros(shape, dtype=_SIM_DTYPE)
    g_eq = np.zeros(shape, dtype=_SIM_DTYPE)
    R = np.zeros(shape, dtype=_SIM_DTYPE)
    P = np.zeros(shape, dtype=_SIM_DTYPE)
    V[0, :] = _SIM_DTYPE(params.V_init_V)
    T[0, :] = _SIM_DTYPE(params.T_init_K)

    hyst = HysteresisArray(
        params.resist_params,
        size=n_currents,
        start_branch=params.start_branch,
        independent_anchors=True,
    )
    hyst.initialize(T[0, :])
    C = float(params.C_F)
    C_th = float(params.C_th_J_per_K)
    S_e = float(params.S_e_W_per_K)
    if parameter_columns is not None:
        C = np.asarray(parameter_columns["C_F"], dtype=_SIM_DTYPE)
        C_th = np.asarray(parameter_columns["C_th_J_per_K"], dtype=_SIM_DTYPE)
        S_e = np.asarray(parameter_columns["S_e_W_per_K"], dtype=_SIM_DTYPE)
        if any(a.shape != (n_currents,) for a in (C, C_th, S_e)):
            raise ValueError("Parameter columns must match the number of waveforms")
    algebraic = C == 0.0
    dt = _SIM_DTYPE(params.dt_s)
    T0 = _SIM_DTYPE(params.T0_K)
    minimum_proximity_slope = np.ones(n_currents)
    fraction = float(params.filament_fraction)
    cold = filament_cold_reference(params) if fraction < 1 else None
    hot_reference = np.empty_like(R) if fraction < 1 else None

    for k in range(n_steps - 1):
        R_k, g_k = hyst.evaluate(T[k, :])
        if audit_proximity:
            minimum_proximity_slope = np.minimum(
                minimum_proximity_slope, hyst.proximity_temperature_slope(T[k, :]))
        R_k = np.maximum(np.asarray(R_k, dtype=_SIM_DTYPE), _SIM_DTYPE(_EPS))
        if fraction < 1:
            hot_reference[k, :] = R_k
            R_k = _filament_resistance(R_k, cold, fraction)
        g_k = np.clip(np.asarray(g_k, dtype=_SIM_DTYPE), 0.0, 1.0)
        if np.any(algebraic):
            V[k, :] = np.where(algebraic, I_in[k, :] * R_k, V[k, :])
        P_k = (V[k, :] * V[k, :]) / R_k
        filament_power = P_k if fraction == 1 else V[k, :] * V[k, :] * _SIM_DTYPE(fraction) / hot_reference[k, :]
        V[k + 1, :] = _advance_voltage_exact(V[k, :], I_in[k, :], R_k, dt_s=float(dt), C_F=C)
        T[k + 1, :] = _advance_temperature_exact(
            T[k, :],
            filament_power,
            T0_K=float(T0),
            dt_s=float(dt),
            C_th_J_per_K=C_th,
            S_e_W_per_K=S_e,
        )
        g_eq[k, :] = g_k
        R[k, :] = R_k
        P[k, :] = P_k

    R_end, g_end = hyst.evaluate(T[-1, :])
    if audit_proximity:
        minimum_proximity_slope = np.minimum(
            minimum_proximity_slope, hyst.proximity_temperature_slope(T[-1, :]))
    g_eq[-1, :] = np.clip(np.asarray(g_end, dtype=_SIM_DTYPE), 0.0, 1.0)
    R[-1, :] = np.maximum(np.asarray(R_end, dtype=_SIM_DTYPE), _SIM_DTYPE(_EPS))
    if fraction < 1:
        hot_reference[-1, :] = R[-1, :]
        R[-1, :] = _filament_resistance(R[-1, :], cold, fraction)
    if np.any(algebraic):
        V[-1, :] = np.where(algebraic, I_in[-1, :] * R[-1, :], V[-1, :])
    P[-1, :] = (V[-1, :] * V[-1, :]) / R[-1, :]
    return [
        {
            "t": t,
            "I_in": I_in[:, idx].copy(),
            "V_vo2": V[:, idx].copy(),
            "T": T[:, idx].copy(),
            "T_hot": T[:, idx].copy(),
            "T_sub": T[:, idx].copy(),
            "g_eq": g_eq[:, idx].copy(),
            "g_dyn": g_eq[:, idx].copy(),
            "R": R[:, idx].copy(),
            "P": P[:, idx].copy(),
            **(_filament_channels(V[:, idx], hot_reference[:, idx], cold, fraction, T0) if fraction < 1 else {}),
            **({"minimum_proximity_slope": np.asarray(minimum_proximity_slope[idx])}
               if audit_proximity else {}),
        }
        for idx in range(n_currents)
    ]


def simulate_current_parameter_grid(
    I_uA: float,
    params: CurrentDriveParams,
    *,
    C_F: np.ndarray,
    C_th_J_per_K: np.ndarray,
    S_e_W_per_K: np.ndarray,
) -> List[Dict[str, np.ndarray]]:
    """Batch independent C/Cth/Se points through the authoritative time updates.

    All columns share resistance parameters, ambient temperature, initial state,
    timestep and imposed current. Independent hysteresis anchors prevent one
    grid point from modifying another point's reversal history.
    """
    columns = {"C_F": np.asarray(C_F, float).reshape(-1),
               "C_th_J_per_K": np.asarray(C_th_J_per_K, float).reshape(-1),
               "S_e_W_per_K": np.asarray(S_e_W_per_K, float).reshape(-1)}
    n = len(columns["C_F"])
    if any(len(a) != n or not np.isfinite(a).all() for a in columns.values()):
        raise ValueError("Grid columns must have equal lengths and finite values")
    if np.any(columns["C_F"] < 0) or np.any(columns["C_th_J_per_K"] <= 0) or np.any(columns["S_e_W_per_K"] < 0):
        raise ValueError("Require C >= 0, Cth > 0 and Se >= 0")
    if n == 0:
        return []
    t = _time_grid(params.dt_s, params.t_end_s, params.t_pre_s)
    current = _build_current_waveform(t, I_uA * 1e-6, params.pulse_on_s, params.pulse_off_s)
    return _simulate_with_current_matrix(t, np.repeat(current[:, None], n, axis=1),
        params, audit_proximity=True, parameter_columns=columns)


def simulate_current_waveforms(
    currents_uA: np.ndarray,
    params: CurrentDriveParams,
    *,
    waveform_time_s: np.ndarray,
    audit_proximity: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """Simulate many measured current records with shared times and parameters.

    Columns are independent devices/traces. This is mathematically identical to
    repeated ``simulate_current_waveform`` calls when thermal noise is disabled,
    but makes global parameter inference computationally practical.
    """

    currents = np.asarray(currents_uA, dtype=float)
    if currents.ndim == 1:
        currents = currents[:, np.newaxis]
    times = np.asarray(waveform_time_s, dtype=float).reshape(-1)
    if currents.ndim != 2 or currents.shape[0] != times.size:
        raise ValueError("currents_uA must have shape (waveform time, traces)")
    if times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("waveform_time_s must increase strictly")
    t = _time_grid(params.dt_s, params.t_end_s, params.t_pre_s)
    interpolated = np.column_stack(
        [
            np.interp(t.astype(float), times, currents[:, idx], left=currents[0, idx], right=currents[-1, idx])
            for idx in range(currents.shape[1])
        ]
    )
    return _simulate_with_current_matrix(t, interpolated * 1e-6, params,
                                         audit_proximity=audit_proximity)


@dataclass(frozen=True)
class ExperimentalCurrentModel:
    """Opt-in material hypotheses, separate from the supported default model.

    For parallel regions primary_fraction is a conductance weight; for series
    regions it partitions the uniform-temperature resistance. Secondary thermal
    parameters are physical SI values, not automatically identified geometry.
    The primary thermal capacity/cooling are supplied in CurrentDriveParams.
    """

    kind: str = "baseline"
    primary_fraction: float = 0.75
    C_secondary_J_per_K: float = 1e-13
    S_secondary_W_per_K: float = 1e-6
    coupling_W_per_K: float = 0.0
    phase_tau_s: float = 0.0

    def __post_init__(self):
        if self.kind not in {"baseline", "warm_parallel", "series_bottleneck", "phase_lag"}:
            raise ValueError("Unknown experimental current model")
        values = [self.primary_fraction, self.C_secondary_J_per_K, self.S_secondary_W_per_K,
                  self.coupling_W_per_K, self.phase_tau_s]
        if not np.isfinite(values).all():
            raise ValueError("Experimental model parameters must be finite")
        if not 0 < self.primary_fraction < 1 or self.C_secondary_J_per_K <= 0:
            raise ValueError("Require 0 < primary fraction < 1 and positive secondary heat capacity")
        if min(self.S_secondary_W_per_K, self.coupling_W_per_K, self.phase_tau_s) < 0:
            raise ValueError("Cooling, coupling and phase time must be nonnegative")


def two_region_electrical_channels(voltage_V, primary_reference_ohm, secondary_reference_ohm,
                                   fraction, topology):
    """Partition current, internal voltage and Joule heat without double counting.

    Equal reference resistances recover the measured whole-device R(T) in both
    topologies. Series elements are regions inside the measured device: no
    external resistor or change to an imposed ideal current is introduced.
    """
    v = np.asarray(voltage_V)
    hot, secondary = np.asarray(primary_reference_ohm), np.asarray(secondary_reference_ohm)
    if (not 0 < fraction < 1 or not np.isfinite(v).all() or not np.isfinite(hot).all()
            or not np.isfinite(secondary).all() or np.any(hot <= 0) or np.any(secondary <= 0)):
        raise ValueError("Require positive reference resistances and 0 < fraction < 1")
    if topology == "warm_parallel":
        resistance = _filament_resistance(hot, secondary, fraction)
        primary_current = v * _SIM_DTYPE(fraction) / hot
        secondary_current = v * _SIM_DTYPE(1-fraction) / secondary
        primary_voltage = secondary_voltage = v
    elif topology == "series_bottleneck":
        r1 = _SIM_DTYPE(fraction)*hot
        r2 = _SIM_DTYPE(1-fraction)*secondary
        resistance = r1+r2
        primary_current = secondary_current = v/resistance
        primary_voltage = primary_current*r1
        secondary_voltage = secondary_current*r2
    else:
        raise ValueError("Choose a two-region electrical topology")
    return dict(R=resistance, I_device=v/resistance, I_primary=primary_current,
                I_secondary=secondary_current, V_primary=primary_voltage, V_secondary=secondary_voltage,
                P_primary=primary_voltage*primary_current, P_secondary=secondary_voltage*secondary_current)


def coupled_thermal_propagator(*, C_primary_J_per_K, C_secondary_J_per_K,
                               S_primary_W_per_K, S_secondary_W_per_K, coupling_W_per_K, dt_s):
    """Exact frozen-power step for two passive coupled thermal nodes.

    Mutual heat is K(Tprimary-Tsecondary), with opposite signs at the two nodes.
    An augmented matrix exponential handles zero bath cooling without a singular
    inverse. Compute these coefficients once per integration, not at each step.
    """
    from scipy.linalg import expm

    values = [C_primary_J_per_K, C_secondary_J_per_K, S_primary_W_per_K,
              S_secondary_W_per_K, coupling_W_per_K, dt_s]
    if not np.isfinite(values).all() or min(values[:2]) <= 0 or dt_s <= 0 or min(values[2:5]) < 0:
        raise ValueError("Invalid thermal capacities, conductances or timestep")
    c1, c2, s1, s2, k, step = values
    matrix = np.zeros((4, 4))
    matrix[:2, :2] = [[-(s1+k)/c1, k/c1], [k/c2, -(s2+k)/c2]]
    matrix[0, 2] = 1/c1
    matrix[1, 3] = 1/c2
    evolution = expm(step*matrix)
    return evolution[:2, :2], evolution[:2, 2:]


def advance_coupled_temperatures(primary_K, secondary_K, primary_power_W, secondary_power_W,
                                 *, T0_K, propagator):
    """Advance both temperatures with frozen powers using energy-conserving coupling."""
    evolution, heating = propagator
    a, b = np.asarray(primary_K)-T0_K, np.asarray(secondary_K)-T0_K
    return (T0_K+evolution[0, 0]*a+evolution[0, 1]*b+heating[0, 0]*primary_power_W+heating[0, 1]*secondary_power_W,
            T0_K+evolution[1, 0]*a+evolution[1, 1]*b+heating[1, 0]*primary_power_W+heating[1, 1]*secondary_power_W)


def simulate_experimental_current_waveforms(currents_uA, params, experiment, *, waveform_time_s,
                                            audit_proximity=False):
    """Integrate one experimental mechanism with the same ideal-current KCL.

    Baseline and zero phase time delegate exactly to the original solver. Other
    modes preserve the original float32 hysteresis target and use exact frozen
    RC, thermal and phase substeps. Their coupling still requires refinement.
    Phase-lag hysteresis anchors follow the original quasi-static target; the
    actual semiconducting weight relaxes toward it and determines resistance.
    """
    if experiment.kind == "baseline" or (experiment.kind == "phase_lag" and experiment.phase_tau_s == 0):
        return simulate_current_waveforms(currents_uA, params, waveform_time_s=waveform_time_s,
                                         audit_proximity=audit_proximity)
    if params.filament_fraction != 1 or params.sigma_W_sqrt_s != 0:
        raise ValueError("Do not combine mechanisms with the old cold branch or thermal noise")
    currents = np.asarray(currents_uA, float)
    if currents.ndim == 1:
        currents = currents[:, None]
    times = np.asarray(waveform_time_s, float)
    if currents.ndim != 2 or times.ndim != 1 or len(times) < 2 or len(times) != len(currents):
        raise ValueError("Require matching waveform times and current columns")
    if not np.isfinite(times).all() or not np.isfinite(currents).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Waveforms must be finite with increasing times")
    if not currents.shape[1]:
        return []
    t = _time_grid(params.dt_s, params.t_end_s, params.t_pre_s)
    source = np.column_stack([np.interp(t.astype(float), times, currents[:, i]) for i in range(currents.shape[1])])
    source = np.asarray(source*1e-6, dtype=_SIM_DTYPE)
    shape = source.shape
    two_regions = experiment.kind in {"warm_parallel", "series_bottleneck"}
    names = ["V_vo2", "T", "g_eq", "g_dyn", "R", "P", "I_device", "P_primary", "P_secondary",
             "I_primary", "I_secondary", "V_primary", "V_secondary", "R_reference_hot"]
    if two_regions:
        names += ["T_secondary", "g_secondary", "R_reference_secondary"]
    arrays = {key: np.zeros(shape, dtype=_SIM_DTYPE) for key in names}
    arrays["V_vo2"][0] = params.V_init_V
    arrays["T"][0] = params.T_init_K
    targets = []
    for _ in range(2 if two_regions else 1):
        hysteresis = HysteresisArray(params.resist_params, size=shape[1], start_branch=params.start_branch,
                                     independent_anchors=True)
        hysteresis.initialize(np.full(shape[1], params.T_init_K, dtype=_SIM_DTYPE))
        targets.append(hysteresis)
    minimum = np.ones(shape[1])
    if two_regions:
        arrays["T_secondary"][0] = params.T_init_K
        propagation = coupled_thermal_propagator(C_primary_J_per_K=params.C_th_J_per_K,
            C_secondary_J_per_K=experiment.C_secondary_J_per_K, S_primary_W_per_K=params.S_e_W_per_K,
            S_secondary_W_per_K=experiment.S_secondary_W_per_K,
            coupling_W_per_K=experiment.coupling_W_per_K, dt_s=params.dt_s)
        # Preserve float32 state while computing stable substeps on temperature
        # differences in float64, avoiding a spurious accumulated bath offset.
    else:
        alpha = _SIM_DTYPE(-np.expm1(-params.dt_s/experiment.phase_tau_s))
        phase = None
    for index in range(len(t)):
        temperature = arrays["T"][index]
        reference, equilibrium = targets[0].evaluate(temperature)
        if audit_proximity:
            minimum = np.minimum(minimum, targets[0].proximity_temperature_slope(temperature))
        arrays["g_eq"][index] = equilibrium
        if two_regions:
            reference2, equilibrium2 = targets[1].evaluate(arrays["T_secondary"][index])
            arrays["g_secondary"][index] = equilibrium2
            arrays["R_reference_secondary"][index] = reference2
            if audit_proximity:
                minimum = np.minimum(minimum, targets[1].proximity_temperature_slope(arrays["T_secondary"][index]))
            arrays["g_dyn"][index] = equilibrium
            resistance = (_filament_resistance(reference, reference2, experiment.primary_fraction)
                          if experiment.kind == "warm_parallel" else
                          _SIM_DTYPE(experiment.primary_fraction)*reference+_SIM_DTYPE(1-experiment.primary_fraction)*reference2)
        else:
            if phase is None:
                phase = equilibrium.copy()
            arrays["g_dyn"][index] = phase
            reference = resistance_from_semiconducting_weight(temperature, phase, params.resist_params)
            resistance = reference
        arrays["R"][index] = resistance
        arrays["R_reference_hot"][index] = reference
        if params.C_F == 0:
            arrays["V_vo2"][index] = source[index]*resistance
        voltage = arrays["V_vo2"][index]
        arrays["P"][index] = voltage*voltage/resistance
        if two_regions:
            channels = two_region_electrical_channels(voltage, reference, reference2,
                                                      experiment.primary_fraction, experiment.kind)
            for key, value in channels.items():
                arrays[key][index] = value
        else:
            arrays["I_device"][index] = voltage/resistance
            arrays["I_primary"][index] = voltage/resistance
            arrays["V_primary"][index] = voltage
            arrays["P_primary"][index] = arrays["P"][index]
        if index == len(t)-1:
            break
        arrays["V_vo2"][index+1] = _advance_voltage_exact(voltage, source[index], resistance,
                                                         dt_s=params.dt_s, C_F=params.C_F)
        if two_regions:
            next_primary, next_secondary = advance_coupled_temperatures(
                temperature.astype(float), arrays["T_secondary"][index].astype(float),
                arrays["P_primary"][index].astype(float), arrays["P_secondary"][index].astype(float),
                T0_K=params.T0_K, propagator=propagation)
            arrays["T"][index+1] = next_primary
            arrays["T_secondary"][index+1] = next_secondary
        else:
            arrays["T"][index+1] = _advance_temperature_exact(temperature, arrays["P"][index],
                T0_K=params.T0_K, dt_s=params.dt_s, C_th_J_per_K=params.C_th_J_per_K,
                S_e_W_per_K=params.S_e_W_per_K)
            phase = np.clip(phase+alpha*(equilibrium-phase), 0., 1.)
    return [dict(t=t, I_in=source[:, i], **{name: value[:, i].copy() for name, value in arrays.items()},
                 **({"minimum_proximity_slope": np.asarray(minimum[i])} if audit_proximity else {}))
            for i in range(shape[1])]


def simulate_current_steps(
    currents_uA: List[float],
    params: CurrentDriveParams,
    seed: Optional[int] = None,
) -> List[Dict[str, np.ndarray]]:
    """Simulate an independent Yuanhang current-source sweep."""

    currents = np.asarray(currents_uA, dtype=float).reshape(-1)
    if currents.size == 0:
        return []
    if float(params.sigma_W_sqrt_s) != 0.0:
        return [
            simulate_current_step(
                float(current),
                params=params,
                seed=None if seed is None else int(seed) + idx,
            )
            for idx, current in enumerate(currents)
        ]

    t = _time_grid(params.dt_s, params.t_end_s, params.t_pre_s)
    n_steps = t.size
    n_currents = currents.size
    active = t >= float(params.pulse_on_s)
    if params.pulse_off_s is not None:
        active &= t < float(params.pulse_off_s)
    I_in = np.zeros((n_steps, n_currents), dtype=_SIM_DTYPE)
    currents_A = np.asarray(currents * 1e-6, dtype=_SIM_DTYPE)
    I_in[active, :] = currents_A[np.newaxis, :]

    return _simulate_with_current_matrix(t, I_in, params)


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
    C = float(params.C_F)

    fit_mask = (t >= 0.0) & (t <= float(fit_window_ns) * 1e-9)
    if np.sum(fit_mask) >= 2:
        slope = float(np.polyfit(t[fit_mask], V[fit_mask], 1)[0])
    else:
        slope = float("nan")

    I_step_A = float(I_uA) * 1e-6
    I_over_C = I_step_A / C if C > 0.0 else float("nan")
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
        dVdt = I_cap / C if C > 0.0 else float("nan")
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
    dt_over_tau = 0.0 if C == 0.0 else float(params.dt_s) / max(tau_init_s, _EPS)
    if C == 0.0:
        warnings.append("C=0: voltage is algebraic (V=I_in*R), so no I/C charging slope exists.")
    if dt_over_tau > 0.2:
        warnings.append(
            "dt is large relative to the initial electrical time constant (dt/tau > 0.2); "
            "the exponential substep is stable, but the coupled transition is under-resolved."
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
    frames_dir: str | Path = "runs/scratch/current_sweep_frames",
    gif_path: str | Path = "runs/scratch/current_sweep.gif",
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
    traces = simulate_current_steps([float(current) for current in currents], params=params, seed=seed)

    for idx, (i_uA, out) in enumerate(zip(currents, traces)):
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
        "traces": traces,
    }


def main() -> None:
    results = run_sweep_make_gif(
        params=CurrentDriveParams(),
        I_start_uA=50,
        I_stop_uA=2000,
        I_step_uA=50,
        frame_duration_s=0.5,
        frames_dir="runs/scratch/current_sweep_frames",
        gif_path="runs/scratch/current_sweep.gif",
        seed=123,
    )
    print("Done.")
    print(f"GIF: {results['gif_path']}")


if __name__ == "__main__":
    main()
