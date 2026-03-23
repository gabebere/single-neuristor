"""
Methodical oscillation-domain search for the ideal current-driven VO2 model.

This module is intentionally separate from the Streamlit UI. It provides a
search backend that can be reused later in the app without inheriting the
ad-hoc assumptions from the old domain explorer.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .current_drive_sim import CurrentDriveParams, simulate_current_step
from .model import HysteresisArray, YuanhangResistParams


_EPS = 1e-12


@dataclass(frozen=True)
class SearchRange:
    low: float
    high: float
    scale: str = "linear"

    def sample(self, rng: np.random.Generator) -> float:
        lo = float(self.low)
        hi = float(self.high)
        if hi <= lo:
            return lo
        if self.scale == "log":
            return float(10 ** rng.uniform(math.log10(lo), math.log10(hi)))
        return float(rng.uniform(lo, hi))

    def perturb(self, center: float, rng: np.random.Generator, rel_width: float = 0.35) -> float:
        lo = float(self.low)
        hi = float(self.high)
        c = float(center)
        if self.scale == "log":
            c = min(max(c, lo), hi)
            log_c = math.log10(c)
            span = rel_width * max(math.log10(hi) - math.log10(lo), 1e-9)
            value = 10 ** rng.uniform(log_c - span, log_c + span)
        else:
            span = rel_width * max(hi - lo, 1e-9)
            value = rng.uniform(c - span, c + span)
        return float(min(max(value, lo), hi))


@dataclass(frozen=True)
class DomainSearchConfig:
    resistance_preset_path: str
    current_start_branch: str = "insulator"
    current_start_uA: int = 50
    current_stop_uA: int = 2000
    coarse_current_step_uA: int = 100
    refine_current_step_uA: int = 25
    refine_half_window_uA: int = 200
    t_end_ns: float = 600.0
    t_pre_ns: float = 0.0
    pulse_on_ns: float = 0.0
    pulse_off_ns: Optional[float] = None
    V_init_mV: float = 0.0
    t0_K: SearchRange = SearchRange(298.0, 298.0, "linear")
    c_pF: SearchRange = SearchRange(5.0, 500.0, "log")
    c_th_mW_ns_per_K: SearchRange = SearchRange(2.0, 500.0, "log")
    s_e_mW_per_K: SearchRange = SearchRange(0.02, 5.0, "log")
    t_init_K: SearchRange = SearchRange(298.0, 338.0, "linear")
    sigma_W_sqrt_s: SearchRange = SearchRange(0.0, 0.0, "linear")
    rm_factor_scale: SearchRange = SearchRange(1.0, 1.0, "linear")
    tc_shift_K: SearchRange = SearchRange(0.0, 0.0, "linear")
    w_scale: SearchRange = SearchRange(1.0, 1.0, "linear")
    beta_scale: SearchRange = SearchRange(1.0, 1.0, "linear")
    reversal_threshold_K: SearchRange = SearchRange(0.01, 0.01, "linear")
    n_random_candidates: int = 18
    top_k_for_refine: int = 4
    refine_samples_per_top: int = 4
    dt_tau_target: float = 0.03
    dT_ratio_target: float = 0.2
    min_dt_ns: float = 0.005
    max_dt_ns: float = 0.5
    max_steps_per_trace: int = 25_000
    min_vpp_mV: float = 20.0
    max_vpp_mV: float = 1_500.0
    min_cycles: int = 3
    seed: int = 1


@dataclass(frozen=True)
class CandidatePoint:
    T0_K: float
    C_pF: float
    C_th_mW_ns_per_K: float
    S_e_mW_per_K: float
    T_init_K: float
    sigma_W_sqrt_s: float
    Rm_factor_scale: float
    Tc_shift_K: float
    w_scale: float
    beta_scale: float
    reversal_threshold_K: float

    def to_resist_params(self, base: YuanhangResistParams) -> YuanhangResistParams:
        params = YuanhangResistParams(**asdict(base))
        params.Rm_factor = float(base.Rm_factor) * float(self.Rm_factor_scale)
        params.Tc_K = float(base.Tc_K) + float(self.Tc_shift_K)
        params.w = float(base.w) * float(self.w_scale)
        params.beta = float(base.beta) * float(self.beta_scale)
        params.reversal_threshold_K = float(self.reversal_threshold_K)
        return params

    def to_params(
        self,
        *,
        cfg: DomainSearchConfig,
        base_resist_params: YuanhangResistParams,
        start_branch: str,
        dt_ns: float,
    ) -> CurrentDriveParams:
        resist_params = self.to_resist_params(base_resist_params)
        pulse_off_s = None if cfg.pulse_off_ns is None else float(cfg.pulse_off_ns) * 1e-9
        return CurrentDriveParams(
            dt_s=float(dt_ns) * 1e-9,
            t_end_s=float(cfg.t_end_ns) * 1e-9,
            t_pre_s=float(cfg.t_pre_ns) * 1e-9,
            pulse_on_s=float(cfg.pulse_on_ns) * 1e-9,
            pulse_off_s=pulse_off_s,
            V_init_V=float(cfg.V_init_mV) * 1e-3,
            T0_K=float(self.T0_K),
            T_init_K=float(self.T_init_K),
            C_F=float(self.C_pF) * 1e-12,
            C_th_J_per_K=float(self.C_th_mW_ns_per_K) * 1e-12,
            S_e_W_per_K=float(self.S_e_mW_per_K) * 1e-3,
            sigma_W_sqrt_s=float(self.sigma_W_sqrt_s),
            resist_params=resist_params,
            start_branch=start_branch,
        )


def _sanitize_candidate(cfg: DomainSearchConfig, point: CandidatePoint) -> CandidatePoint:
    t0 = min(max(float(point.T0_K), cfg.t0_K.low), cfg.t0_K.high)
    t_init = min(max(float(point.T_init_K), max(cfg.t_init_K.low, t0)), cfg.t_init_K.high)
    return CandidatePoint(
        T0_K=float(t0),
        C_pF=min(max(float(point.C_pF), cfg.c_pF.low), cfg.c_pF.high),
        C_th_mW_ns_per_K=min(max(float(point.C_th_mW_ns_per_K), cfg.c_th_mW_ns_per_K.low), cfg.c_th_mW_ns_per_K.high),
        S_e_mW_per_K=min(max(float(point.S_e_mW_per_K), cfg.s_e_mW_per_K.low), cfg.s_e_mW_per_K.high),
        T_init_K=float(t_init),
        sigma_W_sqrt_s=min(max(float(point.sigma_W_sqrt_s), cfg.sigma_W_sqrt_s.low), cfg.sigma_W_sqrt_s.high),
        Rm_factor_scale=min(max(float(point.Rm_factor_scale), cfg.rm_factor_scale.low), cfg.rm_factor_scale.high),
        Tc_shift_K=min(max(float(point.Tc_shift_K), cfg.tc_shift_K.low), cfg.tc_shift_K.high),
        w_scale=min(max(float(point.w_scale), cfg.w_scale.low), cfg.w_scale.high),
        beta_scale=min(max(float(point.beta_scale), cfg.beta_scale.low), cfg.beta_scale.high),
        reversal_threshold_K=min(
            max(float(point.reversal_threshold_K), cfg.reversal_threshold_K.low),
            cfg.reversal_threshold_K.high,
        ),
    )


def load_resistance_preset(path: str | Path) -> Tuple[YuanhangResistParams, str, Dict[str, float]]:
    payload = json.loads(Path(path).read_text())
    raw = payload.get("resist_params", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid resistance preset at {path}")
    kwargs = {f.name: float(raw[f.name]) for f in YuanhangResistParams.__dataclass_fields__.values()}
    start_branch = str(payload.get("start_branch", "insulator")).strip().lower()
    if start_branch not in {"insulator", "metal"}:
        start_branch = "insulator"
    metrics = payload.get("fit_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return YuanhangResistParams(**kwargs), start_branch, {str(k): float(v) for k, v in metrics.items()}


def _inclusive_range(start: int, stop: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be >= start")
    return list(range(int(start), int(stop) + 1, int(step)))


def _count_turns(values: np.ndarray) -> int:
    if values.size < 3:
        return 0
    d = np.diff(values)
    return int(np.sum((d[:-1] * d[1:]) < 0.0))


def _looks_like_zigzag(values: np.ndarray) -> bool:
    if values.size < 8:
        return False
    d = np.diff(values)
    signs = np.sign(d)
    signs = signs[signs != 0.0]
    if signs.size < 6:
        return False
    return bool(float(np.mean(signs[:-1] * signs[1:] < 0.0)) > 0.92)


def _extrema_indices(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if values.size < 5:
        return np.array([], dtype=int), np.array([], dtype=int)
    smooth = np.convolve(values, np.array([0.25, 0.5, 0.25]), mode="same")
    d = np.diff(smooth)
    s = np.sign(d)
    peaks = np.where((s[:-1] > 0.0) & (s[1:] < 0.0))[0] + 1
    troughs = np.where((s[:-1] < 0.0) & (s[1:] > 0.0))[0] + 1
    return peaks.astype(int), troughs.astype(int)


def _spectral_purity(t_ns: np.ndarray, values_mV: np.ndarray) -> Tuple[float, float]:
    if values_mV.size < 8:
        return 0.0, 0.0
    dt_s = float(np.median(np.diff(t_ns))) * 1e-9
    if dt_s <= 0.0:
        return 0.0, 0.0
    centered = values_mV - float(np.mean(values_mV))
    window = np.hanning(centered.size)
    spec = np.fft.rfft(centered * window)
    freq_hz = np.fft.rfftfreq(centered.size, d=dt_s)
    mag = np.abs(spec)
    if mag.size <= 1:
        return 0.0, 0.0
    mag[0] = 0.0
    total = float(np.sum(mag))
    if total <= 0.0:
        return 0.0, 0.0
    idx = int(np.argmax(mag))
    purity = float(mag[idx] / total)
    return purity, float(freq_hz[idx] * 1e-6)


def _estimate_local_dRdT(T_K: np.ndarray, R_ohm: np.ndarray) -> float:
    if T_K.size < 3 or R_ohm.size != T_K.size:
        return 0.0
    centered_T = T_K - float(np.mean(T_K))
    centered_R = R_ohm - float(np.mean(R_ohm))
    denom = float(np.dot(centered_T, centered_T))
    if denom <= 1e-18:
        return 0.0
    return float(np.dot(centered_T, centered_R) / denom)


def _plateau_linearization_metrics(
    out: Dict[str, np.ndarray],
    *,
    params: CurrentDriveParams,
    plateau_mask: np.ndarray,
) -> Dict[str, float]:
    if not np.any(plateau_mask):
        return {
            "plateau_dRdT_ohm_per_K": 0.0,
            "plateau_trace_per_s": float("nan"),
            "plateau_det_per_s2": float("nan"),
            "plateau_discriminant_per_s2": float("nan"),
            "plateau_underdamped": 0.0,
            "plateau_ring_freq_MHz": 0.0,
            "plateau_damping_ratio": float("nan"),
            "plateau_tau_env_ns": float("nan"),
        }

    idx = np.flatnonzero(plateau_mask)
    mid = int(idx[len(idx) // 2])
    lo = max(int(idx[0]), mid - 5)
    hi = min(int(idx[-1]), mid + 5)

    T_seg = np.asarray(out["T"][lo : hi + 1], dtype=float)
    R_seg = np.asarray(out["R"][lo : hi + 1], dtype=float)
    dR_dT = _estimate_local_dRdT(T_seg, R_seg)

    V = float(out["V_vo2"][mid])
    R = max(float(out["R"][mid]), _EPS)
    C = max(float(params.C_F), _EPS)
    C_th = max(float(params.C_th_J_per_K), _EPS)
    S_e = float(params.S_e_W_per_K)

    j11 = -1.0 / (C * R)
    j12 = V * dR_dT / (C * R * R)
    j21 = 2.0 * V / (C_th * R)
    j22 = (-(V * V) * dR_dT / (R * R) - S_e) / C_th
    J = np.array([[j11, j12], [j21, j22]], dtype=float)
    eig = np.linalg.eigvals(J)
    trace = float(np.trace(J))
    det = float(np.linalg.det(J))
    disc = float(trace * trace - 4.0 * det)

    underdamped = bool(np.any(np.abs(np.imag(eig)) > 0.0))
    ring_freq_hz = float(np.max(np.abs(np.imag(eig))) / (2.0 * math.pi)) if underdamped else 0.0
    if det > 0.0:
        damping_ratio = float(-trace / (2.0 * math.sqrt(det)))
    else:
        damping_ratio = float("nan")
    tau_env_ns = float(-1e9 / np.max(np.real(eig))) if np.max(np.real(eig)) < 0.0 else float("nan")

    return {
        "plateau_dRdT_ohm_per_K": float(dR_dT),
        "plateau_trace_per_s": trace,
        "plateau_det_per_s2": det,
        "plateau_discriminant_per_s2": disc,
        "plateau_underdamped": float(1.0 if underdamped else 0.0),
        "plateau_ring_freq_MHz": float(ring_freq_hz * 1e-6),
        "plateau_damping_ratio": damping_ratio,
        "plateau_tau_env_ns": tau_env_ns,
    }


def analyze_current_trace(
    out: Dict[str, np.ndarray],
    *,
    params: Optional[CurrentDriveParams] = None,
    min_vpp_mV: float,
    max_vpp_mV: float,
    min_cycles: int,
    pulse_on_ns: float = 0.0,
    pulse_off_ns: Optional[float] = None,
) -> Dict[str, float]:
    t_ns_full = out["t"] * 1e9
    i_uA_full = out["I_in"] * 1e6
    v_mV_full = out["V_vo2"] * 1e3

    active = np.abs(i_uA_full) > 0.0
    if np.any(active):
        active_idx = np.flatnonzero(active)
        t0 = float(t_ns_full[active_idx[0]])
        t1 = float(t_ns_full[active_idx[-1]])
        skip_ns = min(20.0, 0.08 * max(t1 - t0, 0.0))
        eval_mask = active & (t_ns_full >= (t0 + skip_ns))
        if not np.any(eval_mask):
            eval_mask = active
    else:
        eval_mask = np.ones_like(t_ns_full, dtype=bool)

    t_ns = t_ns_full[eval_mask]
    v_mV = v_mV_full[eval_mask]
    if v_mV.size < 8:
        return {
            "turn_count": 0.0,
            "n_cycles": 0.0,
            "V_pp_mV": 0.0,
            "V_std_mV": 0.0,
            "dominant_freq_MHz": 0.0,
            "spectral_purity": 0.0,
            "period_cv": float("inf"),
            "amplitude_cv": float("inf"),
            "late_to_early_vpp": 0.0,
            "unstable": 1.0,
            "zigzag": 0.0,
            "oscillatory": 0.0,
            "trace_score": 0.0,
        }

    unstable = float(np.max(np.abs(v_mV)) > 2_000.0)
    zigzag = float(_looks_like_zigzag(v_mV))
    turns = float(_count_turns(v_mV))
    peaks, troughs = _extrema_indices(v_mV)
    n_cycles = float(max(0, min(len(peaks), len(troughs))))
    v_pp = float(np.ptp(v_mV))
    v_std = float(np.std(v_mV))

    spectral_purity, dominant_freq = _spectral_purity(t_ns, v_mV)

    periods: List[float] = []
    if len(peaks) >= 2:
        periods.extend(np.diff(t_ns[peaks]).tolist())
    if len(troughs) >= 2:
        periods.extend(np.diff(t_ns[troughs]).tolist())
    if periods:
        period_arr = np.asarray(periods, dtype=float)
        period_mean = float(np.mean(period_arr))
        period_cv = float(np.std(period_arr) / max(period_mean, _EPS))
    else:
        period_cv = float("inf")

    extrema = np.sort(np.concatenate([peaks, troughs])) if len(peaks) or len(troughs) else np.array([], dtype=int)
    if extrema.size >= 2:
        amps = np.abs(np.diff(v_mV[extrema]))
        amp_mean = float(np.mean(amps))
        amplitude_cv = float(np.std(amps) / max(amp_mean, _EPS))
    else:
        amplitude_cv = float("inf")

    half = max(1, v_mV.size // 2)
    early_vpp = float(np.ptp(v_mV[:half])) if half >= 2 else 0.0
    late_vpp = float(np.ptp(v_mV[half:])) if (v_mV.size - half) >= 2 else 0.0
    late_to_early = float(late_vpp / max(early_vpp, _EPS)) if early_vpp > 0.0 else 0.0

    oscillatory = (
        (not unstable)
        and (n_cycles >= float(min_cycles))
        and (v_pp >= float(min_vpp_mV))
        and (v_pp <= float(max_vpp_mV))
        and (period_cv <= 1.0)
        and (spectral_purity >= 0.08)
        and not (zigzag and spectral_purity < 0.20 and n_cycles < float(min_cycles) + 2.0)
    )

    cycle_score = min(n_cycles / max(float(min_cycles) + 3.0, 1.0), 1.0)
    amp_score = min(max((v_pp - float(min_vpp_mV)) / 250.0, 0.0), 1.0)
    regularity_score = 0.0 if not np.isfinite(period_cv) else 1.0 / (1.0 + period_cv)
    persistence_score = min(late_to_early, 1.0)
    purity_score = min(max(spectral_purity / 0.25, 0.0), 1.0)
    raw_score = (
        0.30 * cycle_score
        + 0.20 * amp_score
        + 0.20 * regularity_score
        + 0.15 * persistence_score
        + 0.15 * purity_score
    )
    if not oscillatory:
        raw_score *= 0.45
    if unstable:
        raw_score = 0.0
    elif zigzag:
        raw_score *= 0.65

    pre_mask = t_ns_full < float(pulse_on_ns)
    if np.any(pre_mask):
        baseline_mean = float(np.mean(v_mV_full[pre_mask]))
    else:
        n_head = max(3, min(20, v_mV_full.size // 10))
        baseline_mean = float(np.mean(v_mV_full[:n_head]))

    if np.any(active):
        active_idx = np.flatnonzero(active)
        active_start = float(t_ns_full[active_idx[0]])
        active_end = float(t_ns_full[active_idx[-1]])
    else:
        active_start = float(pulse_on_ns)
        active_end = float(t_ns_full[-1]) if t_ns_full.size else float(pulse_on_ns)

    active_duration = max(active_end - active_start, 1.0)
    plateau_start = active_start + 0.33 * active_duration
    plateau_end = active_start + 0.83 * active_duration
    if pulse_off_ns is not None:
        plateau_end = min(plateau_end, float(pulse_off_ns) - 10.0)
    plateau_mask = active & (t_ns_full >= plateau_start) & (t_ns_full <= plateau_end)
    if not np.any(plateau_mask):
        plateau_mask = active
    plateau_values = v_mV_full[plateau_mask] if np.any(plateau_mask) else v_mV
    plateau_mean = float(np.mean(plateau_values)) if plateau_values.size else 0.0
    plateau_std = float(np.std(plateau_values)) if plateau_values.size else 0.0
    plateau_vpp = float(np.ptp(plateau_values)) if plateau_values.size >= 2 else 0.0
    plateau_turns = float(_count_turns(plateau_values)) if plateau_values.size >= 3 else 0.0
    plateau_current_uA = float(np.mean(i_uA_full[plateau_mask])) if np.any(plateau_mask) else 0.0
    plateau_res_ohm = 0.0
    if abs(plateau_current_uA) > 0.0:
        plateau_res_ohm = float(1e3 * plateau_mean / plateau_current_uA)
    if params is not None:
        plateau_lin = _plateau_linearization_metrics(out, params=params, plateau_mask=plateau_mask)
    else:
        plateau_lin = {
            "plateau_dRdT_ohm_per_K": 0.0,
            "plateau_trace_per_s": float("nan"),
            "plateau_det_per_s2": float("nan"),
            "plateau_discriminant_per_s2": float("nan"),
            "plateau_underdamped": 0.0,
            "plateau_ring_freq_MHz": 0.0,
            "plateau_damping_ratio": float("nan"),
            "plateau_tau_env_ns": float("nan"),
        }

    onset_end = active_start + min(0.22 * active_duration, 80.0)
    onset_mask = active & (t_ns_full >= active_start) & (t_ns_full <= onset_end)
    onset_values = v_mV_full[onset_mask] if np.any(onset_mask) else np.array([], dtype=float)
    onset_overshoot = float(np.max(onset_values) - plateau_mean) if onset_values.size else 0.0
    onset_undershoot = float(plateau_mean - np.min(onset_values)) if onset_values.size else 0.0

    if pulse_off_ns is not None:
        off_start = float(pulse_off_ns)
        off_end = off_start + min(0.25 * active_duration, 120.0)
        off_mask = (t_ns_full >= off_start) & (t_ns_full <= off_end)
        off_values = v_mV_full[off_mask] if np.any(off_mask) else np.array([], dtype=float)
        turnoff_undershoot = float(baseline_mean - np.min(off_values)) if off_values.size else 0.0
        if off_values.size:
            turnoff_undershoot = max(turnoff_undershoot, float(plateau_mean - np.min(off_values)))
    else:
        turnoff_undershoot = 0.0

    pulse_turns = float(_count_turns(v_mV_full[active])) if np.any(active) else turns

    return {
        "turn_count": turns,
        "pulse_turn_count": pulse_turns,
        "n_cycles": n_cycles,
        "V_pp_mV": v_pp,
        "V_std_mV": v_std,
        "baseline_mean_mV": baseline_mean,
        "plateau_mean_mV": plateau_mean,
        "plateau_std_mV": plateau_std,
        "plateau_vpp_mV": plateau_vpp,
        "plateau_turn_count": plateau_turns,
        "plateau_res_ohm": plateau_res_ohm,
        **plateau_lin,
        "onset_overshoot_mV": onset_overshoot,
        "onset_undershoot_mV": onset_undershoot,
        "turnoff_undershoot_mV": float(max(turnoff_undershoot, 0.0)),
        "dominant_freq_MHz": float(dominant_freq),
        "spectral_purity": float(spectral_purity),
        "period_cv": float(period_cv),
        "amplitude_cv": float(amplitude_cv),
        "late_to_early_vpp": float(late_to_early),
        "unstable": float(1.0 if unstable else 0.0),
        "zigzag": float(1.0 if zigzag else 0.0),
        "oscillatory": float(1.0 if oscillatory else 0.0),
        "trace_score": float(raw_score),
    }


def _candidate_dt_ns(
    point: CandidatePoint,
    *,
    cfg: DomainSearchConfig,
    base_resist_params: YuanhangResistParams,
    start_branch: str,
    i_peak_uA: float,
) -> Dict[str, float]:
    resist_params = point.to_resist_params(base_resist_params)
    h = HysteresisArray(resist_params, size=1, start_branch=start_branch)
    T_init = np.asarray([float(point.T_init_K)], dtype=float)
    h.initialize(T_init)
    R_init = float(h.evaluate(T_init)[0][0])
    R_fast = max(float(resist_params.Rm), _EPS)

    C_F = float(point.C_pF) * 1e-12
    C_th = float(point.C_th_mW_ns_per_K) * 1e-12
    tau_init_ns = C_F * max(R_init, _EPS) * 1e9
    tau_fast_ns = C_F * R_fast * 1e9
    dt_tau_ns = float(cfg.dt_tau_target) * tau_fast_ns

    i_peak_A = abs(float(i_peak_uA)) * 1e-6
    p_fast_W = i_peak_A * i_peak_A * R_fast
    reversal_K = max(float(resist_params.reversal_threshold_K), _EPS)
    if p_fast_W <= 0.0:
        dt_thermal_ns = float(cfg.max_dt_ns)
    else:
        dt_thermal_ns = float(cfg.dT_ratio_target) * reversal_K * C_th / p_fast_W * 1e9

    recommended_dt_ns = min(float(cfg.max_dt_ns), dt_tau_ns, dt_thermal_ns)
    total_ns = float(cfg.t_pre_ns) + float(cfg.t_end_ns)
    feasible = (
        recommended_dt_ns >= float(cfg.min_dt_ns)
        and total_ns / max(recommended_dt_ns, _EPS) <= float(cfg.max_steps_per_trace)
    )
    chosen_dt_ns = min(max(recommended_dt_ns, float(cfg.min_dt_ns)), float(cfg.max_dt_ns))

    return {
        "R_init_ohm": float(R_init),
        "R_fast_ohm": float(R_fast),
        "tau_init_ns": float(tau_init_ns),
        "tau_fast_ns": float(tau_fast_ns),
        "Rm_eff_ohm": float(resist_params.Rm),
        "Tc_eff_K": float(resist_params.Tc_K),
        "w_eff_K": float(resist_params.w_eff),
        "beta_eff_per_K": float(resist_params.beta),
        "reversal_threshold_K": float(resist_params.reversal_threshold_K),
        "recommended_dt_ns": float(recommended_dt_ns),
        "dt_ns": float(chosen_dt_ns),
        "dt_over_tau_fast": float(chosen_dt_ns / max(tau_fast_ns, _EPS)),
        "dt_over_tau_init": float(chosen_dt_ns / max(tau_init_ns, _EPS)),
        "dT_step_over_reversal": float((chosen_dt_ns * 1e-9) * p_fast_W / max(C_th * reversal_K, _EPS)),
        "estimated_steps": float(total_ns / max(chosen_dt_ns, _EPS)),
        "feasible": float(1.0 if feasible else 0.0),
    }


def _band_metrics(currents: Sequence[int], good_mask: Sequence[bool]) -> Dict[str, float]:
    if not currents:
        return {
            "n_good_currents": 0.0,
            "band_fraction": 0.0,
            "first_good_uA": float("nan"),
            "last_good_uA": float("nan"),
            "longest_band_uA": 0.0,
        }
    good_currents = [int(i) for i, good in zip(currents, good_mask) if good]
    if not good_currents:
        return {
            "n_good_currents": 0.0,
            "band_fraction": 0.0,
            "first_good_uA": float("nan"),
            "last_good_uA": float("nan"),
            "longest_band_uA": 0.0,
        }

    longest = 1
    run = 1
    step = None
    for a, b in zip(good_currents[:-1], good_currents[1:]):
        current_step = b - a
        if step is None:
            step = current_step
        if current_step == step:
            run += 1
        else:
            longest = max(longest, run)
            run = 1
            step = current_step
    longest = max(longest, run)
    effective_step = step if step is not None else 0
    longest_span = 0.0 if longest <= 1 else float((longest - 1) * effective_step)

    return {
        "n_good_currents": float(len(good_currents)),
        "band_fraction": float(len(good_currents) / max(len(currents), 1)),
        "first_good_uA": float(good_currents[0]),
        "last_good_uA": float(good_currents[-1]),
        "longest_band_uA": float(longest_span),
    }


def _experiment_target_score(detail_df: pd.DataFrame) -> Dict[str, float]:
    if detail_df.empty:
        return {
            "target_score": 0.0,
            "plateau_mean_mV_target": float("nan"),
            "plateau_std_mV_target": float("nan"),
            "low_current_peak_mV": float("nan"),
            "low_current_peak_uA": float("nan"),
            "low_current_bump_mV": float("nan"),
            "rep_plateau_mV": float("nan"),
            "rep_overshoot_mV": float("nan"),
            "rep_turnoff_undershoot_mV": float("nan"),
            "rep_plateau_vpp_mV": float("nan"),
            "rep_plateau_turn_count": float("nan"),
            "rep_plateau_ring_freq_MHz": float("nan"),
            "rep_plateau_damping_ratio": float("nan"),
            "rep_plateau_underdamped": float("nan"),
        }

    curve = (
        detail_df[
            [
                "I_target_uA",
                "plateau_mean_mV",
                "plateau_std_mV",
                "plateau_vpp_mV",
                "plateau_turn_count",
                "plateau_res_ohm",
                "plateau_underdamped",
                "plateau_ring_freq_MHz",
                "plateau_damping_ratio",
                "onset_overshoot_mV",
                "turnoff_undershoot_mV",
                "pulse_turn_count",
            ]
        ]
        .sort_values("I_target_uA")
        .reset_index(drop=True)
    )

    high_mask = curve["I_target_uA"] >= 300.0
    high = curve[high_mask]
    if high.empty:
        high = curve.tail(min(5, len(curve)))
    plateau_mean = float(high["plateau_mean_mV"].mean()) if not high.empty else 0.0
    plateau_std = float(high["plateau_mean_mV"].std(ddof=0)) if len(high) >= 2 else 0.0
    if len(high) >= 2:
        slope = abs(float(np.polyfit(high["I_target_uA"], high["plateau_mean_mV"], 1)[0]))
    else:
        slope = float("inf")

    low = curve[(curve["I_target_uA"] >= 50.0) & (curve["I_target_uA"] <= 300.0)]
    if low.empty:
        low = curve.head(min(5, len(curve)))
    low_peak_idx = int(low["plateau_mean_mV"].idxmax())
    low_peak = low.loc[low_peak_idx]
    low_bump = float(low_peak["plateau_mean_mV"] - plateau_mean)

    rep = curve.iloc[0]
    rep_plateau = float(rep["plateau_mean_mV"])
    rep_overshoot = float(max(rep["onset_overshoot_mV"], 0.0))
    rep_turnoff = float(max(rep["turnoff_undershoot_mV"], 0.0))
    rep_turns = float(rep["pulse_turn_count"])
    rep_flat = float(rep["plateau_std_mV"] / max(abs(rep_plateau), 1e-9))
    rep_plateau_vpp = float(rep["plateau_vpp_mV"])
    rep_plateau_turns = float(rep["plateau_turn_count"])
    rep_ring_freq = float(rep["plateau_ring_freq_MHz"])
    rep_damping_ratio = float(rep["plateau_damping_ratio"])
    rep_underdamped = float(rep["plateau_underdamped"])

    plateau_level_score = math.exp(-((plateau_mean - 190.0) / 35.0) ** 2)
    plateau_flat_score = 1.0 / (1.0 + plateau_std / 18.0)
    plateau_slope_score = 1.0 / (1.0 + slope / 0.04) if np.isfinite(slope) else 0.0
    bump_score = min(max(low_bump / 90.0, 0.0), 1.0)

    if rep_plateau <= 0.0:
        rep_pulse_score = 0.0
    else:
        overshoot_ratio = rep_overshoot / max(abs(rep_plateau), 1e-9)
        turnoff_ratio = rep_turnoff / max(abs(rep_plateau), 1e-9)
        overshoot_score = min(max(overshoot_ratio / 0.18, 0.0), 1.0)
        turnoff_score = min(max(turnoff_ratio / 0.20, 0.0), 1.0)
        flat_score = 1.0 / (1.0 + rep_flat / 0.08)
        turn_score = 1.0 / (1.0 + abs(rep_turns - 2.0))
        ripple_amp_score = math.exp(-((rep_plateau_vpp - 6.0) / 5.0) ** 2)
        ripple_turn_score = 1.0 / (1.0 + abs(rep_plateau_turns - 2.0))
        if rep_underdamped > 0.5 and np.isfinite(rep_damping_ratio) and rep_ring_freq > 0.0:
            ring_freq_score = math.exp(-((math.log10(rep_ring_freq) - math.log10(12.0)) / 0.45) ** 2)
            damping_score = math.exp(-((rep_damping_ratio - 0.45) / 0.30) ** 2)
            linear_ring_score = 0.5 * ring_freq_score + 0.5 * damping_score
        else:
            linear_ring_score = 0.0
        rep_pulse_score = (
            0.22 * overshoot_score
            + 0.18 * turnoff_score
            + 0.18 * flat_score
            + 0.08 * turn_score
            + 0.14 * ripple_amp_score
            + 0.10 * ripple_turn_score
            + 0.10 * linear_ring_score
        )

    target_score = (
        0.30 * plateau_level_score
        + 0.20 * plateau_flat_score
        + 0.15 * plateau_slope_score
        + 0.20 * bump_score
        + 0.15 * rep_pulse_score
    )
    return {
        "target_score": float(target_score),
        "plateau_mean_mV_target": float(plateau_mean),
        "plateau_std_mV_target": float(plateau_std),
        "low_current_peak_mV": float(low_peak["plateau_mean_mV"]),
        "low_current_peak_uA": float(low_peak["I_target_uA"]),
        "low_current_bump_mV": float(low_bump),
        "rep_plateau_mV": float(rep_plateau),
        "rep_overshoot_mV": float(rep_overshoot),
        "rep_turnoff_undershoot_mV": float(rep_turnoff),
        "rep_plateau_vpp_mV": float(rep_plateau_vpp),
        "rep_plateau_turn_count": float(rep_plateau_turns),
        "rep_plateau_ring_freq_MHz": float(rep_ring_freq),
        "rep_plateau_damping_ratio": float(rep_damping_ratio),
        "rep_plateau_underdamped": float(rep_underdamped),
    }


def evaluate_candidate(
    point: CandidatePoint,
    *,
    cfg: DomainSearchConfig,
    base_resist_params: YuanhangResistParams,
    start_branch: str,
    base_seed: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    coarse_currents = _inclusive_range(cfg.current_start_uA, cfg.current_stop_uA, cfg.coarse_current_step_uA)
    dt_info = _candidate_dt_ns(
        point,
        cfg=cfg,
        base_resist_params=base_resist_params,
        start_branch=start_branch,
        i_peak_uA=max(abs(cfg.current_start_uA), abs(cfg.current_stop_uA)),
    )

    summary: Dict[str, float] = {
        "C_pF": float(point.C_pF),
        "C_th_mW_ns_per_K": float(point.C_th_mW_ns_per_K),
        "S_e_mW_per_K": float(point.S_e_mW_per_K),
        "T0_K": float(point.T0_K),
        "T_init_K": float(point.T_init_K),
        "sigma_W_sqrt_s": float(point.sigma_W_sqrt_s),
        "Rm_factor_scale": float(point.Rm_factor_scale),
        "Tc_shift_K": float(point.Tc_shift_K),
        "w_scale": float(point.w_scale),
        "beta_scale": float(point.beta_scale),
        **dt_info,
    }
    if dt_info["feasible"] < 0.5:
        summary.update(
            {
                "candidate_score": 0.0,
                "best_trace_score": 0.0,
                "best_current_uA": float("nan"),
                "best_turn_count": 0.0,
                "best_vpp_mV": 0.0,
                "best_freq_MHz": 0.0,
                "n_good_currents": 0.0,
                "band_fraction": 0.0,
                "first_good_uA": float("nan"),
                "last_good_uA": float("nan"),
                "longest_band_uA": 0.0,
            }
        )
        return summary, pd.DataFrame()

    params = point.to_params(
        cfg=cfg,
        base_resist_params=base_resist_params,
        start_branch=start_branch,
        dt_ns=dt_info["dt_ns"],
    )
    results: Dict[int, Dict[str, float]] = {}

    def _run_current(i_uA: int, seed_offset: int) -> None:
        if i_uA in results:
            return
        out = simulate_current_step(float(i_uA), params=params, seed=base_seed + seed_offset)
        results[i_uA] = analyze_current_trace(
            out,
            params=params,
            min_vpp_mV=cfg.min_vpp_mV,
            max_vpp_mV=cfg.max_vpp_mV,
            min_cycles=cfg.min_cycles,
            pulse_on_ns=cfg.pulse_on_ns,
            pulse_off_ns=cfg.pulse_off_ns,
        )

    for idx, i_uA in enumerate(coarse_currents):
        _run_current(i_uA, idx)

    best_coarse = max(coarse_currents, key=lambda i: results[i]["trace_score"])
    if results[best_coarse]["trace_score"] >= 0.25 and cfg.refine_current_step_uA > 0:
        r0 = max(cfg.current_start_uA, best_coarse - cfg.refine_half_window_uA)
        r1 = min(cfg.current_stop_uA, best_coarse + cfg.refine_half_window_uA)
        refine_currents = _inclusive_range(r0, r1, cfg.refine_current_step_uA)
        for idx, i_uA in enumerate(refine_currents, start=len(coarse_currents)):
            _run_current(i_uA, idx)

    ordered_currents = sorted(results.keys())
    detail_rows: List[Dict[str, float]] = []
    for i_uA in ordered_currents:
        row = {
            "I_target_uA": float(i_uA),
            **summary,
            **results[i_uA],
        }
        detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows).sort_values("I_target_uA").reset_index(drop=True)
    if detail_df.empty:
        summary.update(
            {
                "candidate_score": 0.0,
                "best_trace_score": 0.0,
                "best_current_uA": float("nan"),
                "best_turn_count": 0.0,
                "best_vpp_mV": 0.0,
                "best_freq_MHz": 0.0,
                "n_good_currents": 0.0,
                "band_fraction": 0.0,
                "first_good_uA": float("nan"),
                "last_good_uA": float("nan"),
                "longest_band_uA": 0.0,
            }
        )
        return summary, detail_df

    best_idx = int(detail_df["trace_score"].idxmax())
    best_row = detail_df.loc[best_idx]
    good_mask = detail_df["oscillatory"] > 0.5
    band = _band_metrics(detail_df["I_target_uA"].astype(int).tolist(), good_mask.tolist())
    target = _experiment_target_score(detail_df)
    current_span = max(float(cfg.current_stop_uA - cfg.current_start_uA), 1.0)
    generic_score = (
        0.50 * float(best_row["trace_score"])
        + 0.30 * float(band["band_fraction"])
        + 0.20 * min(float(band["longest_band_uA"]) / current_span, 1.0)
    )
    # For the present task, reproducing the measured pulse/IV shape matters more
    # than finding the largest generic oscillation window.
    candidate_score = 0.30 * generic_score + 0.70 * float(target["target_score"])

    summary.update(
        {
            "candidate_score": float(candidate_score),
            "generic_score": float(generic_score),
            "best_trace_score": float(best_row["trace_score"]),
            "best_current_uA": float(best_row["I_target_uA"]),
            "best_turn_count": float(best_row["turn_count"]),
            "best_vpp_mV": float(best_row["V_pp_mV"]),
            "best_freq_MHz": float(best_row["dominant_freq_MHz"]),
            **band,
            **target,
        }
    )
    return summary, detail_df


def _seed_candidates(cfg: DomainSearchConfig, resist_params: YuanhangResistParams) -> List[CandidatePoint]:
    tc = float(resist_params.Tc_K)
    w = float(resist_params.w_eff)
    return [
        _sanitize_candidate(
            cfg,
            CandidatePoint(
                T0_K=cfg.t0_K.low,
                C_pF=145.34619293,
                C_th_mW_ns_per_K=49.62776831,
                S_e_mW_per_K=0.20558726,
                T_init_K=max(cfg.t0_K.low, min(cfg.t_init_K.high, cfg.t0_K.low)),
                sigma_W_sqrt_s=cfg.sigma_W_sqrt_s.low,
                Rm_factor_scale=1.0,
                Tc_shift_K=0.0,
                w_scale=1.0,
                beta_scale=1.0,
                reversal_threshold_K=resist_params.reversal_threshold_K,
            ),
        ),
        _sanitize_candidate(
            cfg,
            CandidatePoint(
                T0_K=min(max(tc - 0.75 * w, cfg.t0_K.low), cfg.t0_K.high),
                C_pF=min(max(20.0, cfg.c_pF.low), cfg.c_pF.high),
                C_th_mW_ns_per_K=min(max(5.0, cfg.c_th_mW_ns_per_K.low), cfg.c_th_mW_ns_per_K.high),
                S_e_mW_per_K=min(max(0.2, cfg.s_e_mW_per_K.low), cfg.s_e_mW_per_K.high),
                T_init_K=min(max(tc - 0.25 * w, cfg.t_init_K.low), cfg.t_init_K.high),
                sigma_W_sqrt_s=cfg.sigma_W_sqrt_s.low,
                Rm_factor_scale=min(max(4.0, cfg.rm_factor_scale.low), cfg.rm_factor_scale.high),
                Tc_shift_K=0.0,
                w_scale=1.0,
                beta_scale=1.0,
                reversal_threshold_K=resist_params.reversal_threshold_K,
            ),
        ),
        _sanitize_candidate(
            cfg,
            CandidatePoint(
                T0_K=min(max(tc + 0.10 * w, cfg.t0_K.low), cfg.t0_K.high),
                C_pF=min(max(50.0, cfg.c_pF.low), cfg.c_pF.high),
                C_th_mW_ns_per_K=min(max(20.0, cfg.c_th_mW_ns_per_K.low), cfg.c_th_mW_ns_per_K.high),
                S_e_mW_per_K=min(max(0.5, cfg.s_e_mW_per_K.low), cfg.s_e_mW_per_K.high),
                T_init_K=min(max(tc + 0.10 * w, cfg.t_init_K.low), cfg.t_init_K.high),
                sigma_W_sqrt_s=cfg.sigma_W_sqrt_s.low,
                Rm_factor_scale=min(max(10.0, cfg.rm_factor_scale.low), cfg.rm_factor_scale.high),
                Tc_shift_K=min(max(2.0, cfg.tc_shift_K.low), cfg.tc_shift_K.high),
                w_scale=min(max(1.1, cfg.w_scale.low), cfg.w_scale.high),
                beta_scale=min(max(1.1, cfg.beta_scale.low), cfg.beta_scale.high),
                reversal_threshold_K=max(cfg.reversal_threshold_K.low, min(0.005, cfg.reversal_threshold_K.high)),
            ),
        ),
        _sanitize_candidate(
            cfg,
            CandidatePoint(
                T0_K=min(max(tc + 0.15 * w, cfg.t0_K.low), cfg.t0_K.high),
                C_pF=min(max(80.0, cfg.c_pF.low), cfg.c_pF.high),
                C_th_mW_ns_per_K=min(max(40.0, cfg.c_th_mW_ns_per_K.low), cfg.c_th_mW_ns_per_K.high),
                S_e_mW_per_K=min(max(0.1, cfg.s_e_mW_per_K.low), cfg.s_e_mW_per_K.high),
                T_init_K=min(max(tc + 0.20 * w, cfg.t_init_K.low), cfg.t_init_K.high),
                sigma_W_sqrt_s=cfg.sigma_W_sqrt_s.high,
                Rm_factor_scale=min(max(8.0, cfg.rm_factor_scale.low), cfg.rm_factor_scale.high),
                Tc_shift_K=min(max(4.0, cfg.tc_shift_K.low), cfg.tc_shift_K.high),
                w_scale=min(max(1.2, cfg.w_scale.low), cfg.w_scale.high),
                beta_scale=min(max(1.2, cfg.beta_scale.low), cfg.beta_scale.high),
                reversal_threshold_K=max(cfg.reversal_threshold_K.low, min(0.003, cfg.reversal_threshold_K.high)),
            ),
        ),
    ]


def _random_candidate(cfg: DomainSearchConfig, rng: np.random.Generator) -> CandidatePoint:
    return _sanitize_candidate(
        cfg,
        CandidatePoint(
            T0_K=cfg.t0_K.sample(rng),
            C_pF=cfg.c_pF.sample(rng),
            C_th_mW_ns_per_K=cfg.c_th_mW_ns_per_K.sample(rng),
            S_e_mW_per_K=cfg.s_e_mW_per_K.sample(rng),
            T_init_K=cfg.t_init_K.sample(rng),
            sigma_W_sqrt_s=cfg.sigma_W_sqrt_s.sample(rng),
            Rm_factor_scale=cfg.rm_factor_scale.sample(rng),
            Tc_shift_K=cfg.tc_shift_K.sample(rng),
            w_scale=cfg.w_scale.sample(rng),
            beta_scale=cfg.beta_scale.sample(rng),
            reversal_threshold_K=cfg.reversal_threshold_K.sample(rng),
        ),
    )


def _perturb_candidate(cfg: DomainSearchConfig, base: CandidatePoint, rng: np.random.Generator) -> CandidatePoint:
    return _sanitize_candidate(
        cfg,
        CandidatePoint(
            T0_K=cfg.t0_K.perturb(base.T0_K, rng, rel_width=0.12),
            C_pF=cfg.c_pF.perturb(base.C_pF, rng),
            C_th_mW_ns_per_K=cfg.c_th_mW_ns_per_K.perturb(base.C_th_mW_ns_per_K, rng),
            S_e_mW_per_K=cfg.s_e_mW_per_K.perturb(base.S_e_mW_per_K, rng),
            T_init_K=cfg.t_init_K.perturb(base.T_init_K, rng, rel_width=0.12),
            sigma_W_sqrt_s=cfg.sigma_W_sqrt_s.perturb(base.sigma_W_sqrt_s, rng, rel_width=0.35),
            Rm_factor_scale=cfg.rm_factor_scale.perturb(base.Rm_factor_scale, rng, rel_width=0.25),
            Tc_shift_K=cfg.tc_shift_K.perturb(base.Tc_shift_K, rng, rel_width=0.20),
            w_scale=cfg.w_scale.perturb(base.w_scale, rng, rel_width=0.20),
            beta_scale=cfg.beta_scale.perturb(base.beta_scale, rng, rel_width=0.20),
            reversal_threshold_K=cfg.reversal_threshold_K.perturb(base.reversal_threshold_K, rng, rel_width=0.30),
        ),
    )


def search_current_domain(cfg: DomainSearchConfig) -> Dict[str, Any]:
    base_resist_params, fit_start_branch, fit_metrics = load_resistance_preset(cfg.resistance_preset_path)
    start_branch = str(cfg.current_start_branch).strip().lower()
    if start_branch not in {"insulator", "metal"}:
        start_branch = fit_start_branch
    rng = np.random.default_rng(int(cfg.seed))

    candidate_points: List[CandidatePoint] = _seed_candidates(cfg, base_resist_params)
    for _ in range(int(cfg.n_random_candidates)):
        candidate_points.append(_random_candidate(cfg, rng))

    summaries: List[Dict[str, float]] = []
    details: List[pd.DataFrame] = []
    for idx, point in enumerate(candidate_points):
        summary, detail = evaluate_candidate(
            point,
            cfg=cfg,
            base_resist_params=base_resist_params,
            start_branch=start_branch,
            base_seed=int(cfg.seed) + idx * 10_000,
        )
        summaries.append(summary)
        if not detail.empty:
            details.append(detail)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["candidate_score", "best_trace_score", "n_good_currents"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    if summary_df.empty:
        detail_df = pd.DataFrame()
        return {
            "config": asdict(cfg),
            "fit_metrics": fit_metrics,
            "start_branch": start_branch,
            "fit_start_branch": fit_start_branch,
            "summary_df": summary_df,
            "detail_df": detail_df,
        }

    top_points: List[CandidatePoint] = []
    for _, row in summary_df.head(int(cfg.top_k_for_refine)).iterrows():
        top_points.append(
            CandidatePoint(
                T0_K=float(row["T0_K"]),
                C_pF=float(row["C_pF"]),
                C_th_mW_ns_per_K=float(row["C_th_mW_ns_per_K"]),
                S_e_mW_per_K=float(row["S_e_mW_per_K"]),
                T_init_K=float(row["T_init_K"]),
                sigma_W_sqrt_s=float(row.get("sigma_W_sqrt_s", 0.0)),
                Rm_factor_scale=float(row.get("Rm_factor_scale", 1.0)),
                Tc_shift_K=float(row.get("Tc_shift_K", 0.0)),
                w_scale=float(row.get("w_scale", 1.0)),
                beta_scale=float(row.get("beta_scale", 1.0)),
                reversal_threshold_K=float(row.get("reversal_threshold_K", base_resist_params.reversal_threshold_K)),
            )
        )

    refine_points: List[CandidatePoint] = []
    for point in top_points:
        for _ in range(int(cfg.refine_samples_per_top)):
            refine_points.append(_perturb_candidate(cfg, point, rng))

    for idx, point in enumerate(refine_points, start=len(candidate_points)):
        summary, detail = evaluate_candidate(
            point,
            cfg=cfg,
            base_resist_params=base_resist_params,
            start_branch=start_branch,
            base_seed=int(cfg.seed) + idx * 10_000,
        )
        summaries.append(summary)
        if not detail.empty:
            details.append(detail)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["candidate_score", "best_trace_score", "n_good_currents"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()

    return {
        "config": asdict(cfg),
        "fit_metrics": fit_metrics,
        "start_branch": start_branch,
        "fit_start_branch": fit_start_branch,
        "summary_df": summary_df,
        "detail_df": detail_df,
    }


def save_search_results(results: Dict[str, Any], output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "search_config.json"
    summary_path = out_dir / "candidate_summary.csv"
    detail_path = out_dir / "trace_detail.csv"
    best_path = out_dir / "best_candidate.json"
    best_target_path = out_dir / "best_by_target_score.json"
    best_generic_path = out_dir / "best_by_generic_score.json"

    payload = {
        "config": results["config"],
        "fit_metrics": results.get("fit_metrics", {}),
        "start_branch": results.get("start_branch", "insulator"),
        "fit_start_branch": results.get("fit_start_branch", "insulator"),
    }
    config_path.write_text(json.dumps(payload, indent=2))

    summary_df: pd.DataFrame = results["summary_df"]
    detail_df: pd.DataFrame = results["detail_df"]
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    if not summary_df.empty:
        best_path.write_text(json.dumps(summary_df.iloc[0].to_dict(), indent=2))
        if "target_score" in summary_df.columns:
            best_target_row = summary_df.sort_values(["target_score", "candidate_score"], ascending=[False, False]).iloc[0]
            best_target_path.write_text(json.dumps(best_target_row.to_dict(), indent=2))
        if "generic_score" in summary_df.columns:
            best_generic_row = summary_df.sort_values(["generic_score", "candidate_score"], ascending=[False, False]).iloc[0]
            best_generic_path.write_text(json.dumps(best_generic_row.to_dict(), indent=2))

    return out_dir
