"""
Fit specimen-specific VO2 resistance/hysteresis parameters from experimental R(T) data.

This tool uses the same hysteresis evaluator used by the simulators (`model.HysteresisArray`)
and optimizes a major-loop parameter set against a measured temperature trajectory.

Example:
  python resistance_custom_analysis.py \
    --data data/experimental/100425_chip1_gap3.tsv \
    --save-json presets/resistance_100425_chip1_gap3.json \
    --save-plot outputs/resistance_fit_100425_chip1_gap3.png
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import HysteresisArray, YuanhangResistParams


_EPS = 1e-12


@dataclass
class ResistanceFitResult:
    params: YuanhangResistParams
    start_branch: str
    rmse_log10: float
    rmse_log10_cooling: float
    rmse_log10_heating: float
    source_data: str
    n_samples: int

    def to_jsonable(self) -> Dict[str, object]:
        return {
            "generated_at_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "source_data": self.source_data,
            "n_samples": int(self.n_samples),
            "start_branch": self.start_branch,
            "fit_metrics": {
                "rmse_log10": float(self.rmse_log10),
                "rmse_log10_cooling": float(self.rmse_log10_cooling),
                "rmse_log10_heating": float(self.rmse_log10_heating),
            },
            "resist_params": dataclasses.asdict(self.params),
        }


def load_experimental_rt(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Experimental data file not found: {p}")

    df = pd.read_csv(p, sep="\t", engine="python", skip_blank_lines=True)
    df.columns = [str(c).strip() for c in df.columns]

    # Common format in lab exports: second row stores units (K, Ohm, sec, ...).
    if len(df) > 0 and str(df.iloc[0, 0]).strip().upper() in {"K", "TEMP", "TEMPERATURE"}:
        df = df.iloc[1:].reset_index(drop=True)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    need = ["Temperature", "Resistance"]
    if not all(c in df.columns for c in need):
        raise ValueError(f"Expected columns {need}, got {list(df.columns)}")

    df = df.dropna(subset=need)
    if "Time" in df.columns:
        df = df.dropna(subset=["Time"]).sort_values("Time")

    df = df.reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Not enough valid samples after parsing.")
    return df


def _predict_resistance_from_temperature(
    temperatures_K: np.ndarray,
    params: YuanhangResistParams,
    *,
    start_branch: str,
) -> np.ndarray:
    h = HysteresisArray(params, size=1, start_branch=start_branch)
    h.initialize(np.asarray([float(temperatures_K[0])], dtype=float))
    pred = np.zeros_like(temperatures_K, dtype=float)
    for i, t in enumerate(temperatures_K):
        r, _ = h.evaluate(np.asarray([float(t)], dtype=float))
        pred[i] = float(r[0])
    return pred


def _predict_g_from_temperature(
    temperatures_K: np.ndarray,
    params: YuanhangResistParams,
    *,
    start_branch: str,
) -> np.ndarray:
    h = HysteresisArray(params, size=1, start_branch=start_branch)
    h.initialize(np.asarray([float(temperatures_K[0])], dtype=float))
    g_out = np.zeros_like(temperatures_K, dtype=float)
    for i, t in enumerate(temperatures_K):
        _, g = h.evaluate(np.asarray([float(t)], dtype=float))
        g_out[i] = float(g[0])
    return g_out


def _vector_to_params(
    x: np.ndarray,
    *,
    t_min: float,
    t_max: float,
) -> YuanhangResistParams:
    # x = [log10_R0, Ea_over_k, log10_Rm, w, Tc_K, log10_beta, gamma]
    p = YuanhangResistParams()
    p.R0 = float(10.0 ** x[0])
    p.Ea_over_k = float(x[1])
    rm = float(10.0 ** x[2])
    p.Rm0 = rm
    p.Rm_factor = 1.0
    p.w = float(x[3])
    p.Tc_K = float(x[4])
    p.beta = float(10.0 ** x[5])
    p.gamma = float(x[6])
    p.width_factor = 1.0
    p.T_min_K = float(t_min)
    p.T_max_K = float(t_max)
    p.reversal_threshold_K = 0.01
    return p


def _rmse_log10(pred_ohm: np.ndarray, target_ohm: np.ndarray) -> float:
    e = np.log10(np.maximum(pred_ohm, _EPS)) - np.log10(np.maximum(target_ohm, _EPS))
    return float(np.sqrt(np.mean(e * e)))


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    e = pred - target
    return float(np.sqrt(np.mean(e * e)))


def _estimate_r0_ea_rm_from_extremes(
    t_k: np.ndarray,
    r_ohm: np.ndarray,
) -> tuple[float, float, float]:
    # 2582_1 guidance: estimate Rm from metallic side; estimate (R0, Ea) from semiconducting side.
    # We use robust quantile windows and a 1D search over Rm.
    t = np.asarray(t_k, dtype=float)
    r = np.asarray(r_ohm, dtype=float)
    q_low = np.quantile(t, 0.22)
    low_mask = t <= q_low
    if np.sum(low_mask) < 8:
        idx = np.argsort(t)[: max(8, min(24, t.size))]
        low_mask = np.zeros_like(t, dtype=bool)
        low_mask[idx] = True

    t_low = t[low_mask]
    r_low = r[low_mask]
    x = 1.0 / np.maximum(t_low, 1e-9)

    r_min = float(np.min(r))
    rm_grid = np.linspace(max(0.05 * r_min, 1e-6), max(2.2 * r_min, r_min + 1.0), 280)
    best_err = float("inf")
    best_r0 = 5.35882879e-3
    best_ea = 5.22047417e3
    best_rm = max(0.9 * r_min, 1e-6)
    for rm in rm_grid:
        y = r_low - rm
        if np.any(y <= 0.0):
            continue
        logy = np.log(y)
        slope, intercept = np.polyfit(x, logy, 1)
        fit = slope * x + intercept
        err = _rmse(fit, logy)
        if err < best_err:
            best_err = err
            best_ea = float(slope)
            best_r0 = float(np.exp(intercept))
            best_rm = float(rm)
    return best_r0, best_ea, best_rm


def _compute_g_experimental(
    t_k: np.ndarray,
    r_ohm: np.ndarray,
    *,
    r0: float,
    ea_over_k: float,
    rm: float,
) -> np.ndarray:
    denom = max(float(r0), _EPS) * np.exp(float(ea_over_k) / np.maximum(t_k, 1e-9))
    g = (r_ohm - float(rm)) / np.maximum(denom, _EPS)
    return np.clip(g, 0.0, 1.0)


def _estimate_major_w_tc_from_g(
    t_k: np.ndarray,
    g_exp: np.ndarray,
) -> tuple[float, float]:
    dT = np.diff(t_k, prepend=t_k[0])
    cool = dT < 0.0
    heat = dT > 0.0

    def _closest_half(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float(np.median(t_k))
        idx = np.argmin(np.abs(g_exp[mask] - 0.5))
        return float(t_k[mask][idx])

    tc_cool = _closest_half(cool)
    tc_heat = _closest_half(heat)
    w = abs(tc_heat - tc_cool)
    tc = 0.5 * (tc_heat + tc_cool)
    return max(float(w), 0.6), float(tc)


def _fit_hysteresis_from_g(
    t_k: np.ndarray,
    g_exp: np.ndarray,
    *,
    t_min_fit: float,
    t_max_fit: float,
    beta_init: float,
    gamma_init: float,
    w_init: float,
    tc_init: float,
    start_branch_hint: str,
    seed: int,
    random_iters: int,
    local_passes: int,
) -> tuple[YuanhangResistParams, str, np.ndarray, float]:
    lb = np.array([0.5, t_min_fit + 2.0, -3.0, 0.2], dtype=float)
    ub = np.array([25.0, t_max_fit - 2.0, 1.0, 2.0], dtype=float)
    x0 = np.array(
        [
            float(w_init),
            float(tc_init),
            float(math.log10(max(beta_init, 1e-4))),
            float(gamma_init),
        ],
        dtype=float,
    )
    x0 = np.clip(x0, lb, ub)
    rng = np.random.default_rng(int(seed))
    step = np.array([0.4, 0.45, 0.07, 0.06], dtype=float)

    def _params_from_x(x: np.ndarray) -> YuanhangResistParams:
        p = YuanhangResistParams()
        # Resistance terms are not used for g-only fit, but keep finite values.
        p.R0 = 1.0
        p.Ea_over_k = 1.0
        p.Rm0 = 1.0
        p.Rm_factor = 1.0
        p.w = float(x[0])
        p.Tc_K = float(x[1])
        p.beta = float(10.0 ** x[2])
        p.gamma = float(x[3])
        p.width_factor = 1.0
        p.T_min_K = float(t_min_fit)
        p.T_max_K = float(t_max_fit)
        p.reversal_threshold_K = 0.01
        return p

    def _score(x: np.ndarray, branch: str) -> tuple[float, np.ndarray]:
        p = _params_from_x(x)
        g_model = _predict_g_from_temperature(t_k, p, start_branch=branch)
        return _rmse(g_model, g_exp), g_model

    best_x = x0.copy()
    best_branch = start_branch_hint
    best_score, best_g = _score(best_x, best_branch)
    for br in ("insulator", "metal"):
        s, g = _score(best_x, br)
        if s < best_score:
            best_score, best_branch, best_g = s, br, g

    for i in range(int(random_iters)):
        if i % 3 == 0:
            proposal = best_x + rng.normal(scale=np.array([1.0, 2.0, 0.2, 0.2]))
            x = np.clip(proposal, lb, ub)
        else:
            x = lb + rng.random(lb.shape) * (ub - lb)
        br = best_branch if rng.random() < 0.65 else ("metal" if best_branch == "insulator" else "insulator")
        s, g = _score(x, br)
        if s < best_score:
            best_score, best_x, best_branch, best_g = s, x.copy(), br, g

    for _ in range(int(local_passes)):
        improved = False
        for j in range(best_x.size):
            for direction in (-1.0, 1.0):
                cand = best_x.copy()
                cand[j] = np.clip(cand[j] + direction * step[j], lb[j], ub[j])
                s, g = _score(cand, best_branch)
                if s < best_score:
                    best_score, best_x, best_g = s, cand, g
                    improved = True
        if not improved:
            step *= 0.72
            if float(np.max(step)) < 1e-4:
                break

    p_out = _params_from_x(best_x)
    # Restore default resistance terms for caller to overwrite.
    p_out.R0 = 5.35882879e-3
    p_out.Ea_over_k = 5.22047417e3
    p_out.Rm0 = 262.5
    p_out.Rm_factor = 1.0
    return p_out, best_branch, best_g, best_score


def _fit_re_terms_given_g(
    t_k: np.ndarray,
    r_ohm: np.ndarray,
    g_model: np.ndarray,
    *,
    ea_center: float,
) -> tuple[float, float, float, np.ndarray, float]:
    # Fit R = R0*exp(Ea/T)*g + Rm by scanning Ea and solving linear least squares for (R0, Rm).
    lo = max(800.0, 0.5 * float(ea_center))
    hi = min(12000.0, 1.6 * float(ea_center))
    ea_grid = np.linspace(lo, hi, 420)
    best = (float("inf"), 5.35882879e-3, ea_center, 100.0, np.zeros_like(r_ohm))
    for ea in ea_grid:
        a = np.exp(ea / np.maximum(t_k, 1e-9)) * g_model
        x = np.column_stack((a, np.ones_like(a)))
        try:
            theta, *_ = np.linalg.lstsq(x, r_ohm, rcond=None)
        except np.linalg.LinAlgError:
            continue
        r0 = float(theta[0])
        rm = float(theta[1])
        if r0 <= 0.0 or rm <= 0.0:
            continue
        pred = x @ theta
        s = _rmse_log10(pred, r_ohm)
        if s < best[0]:
            best = (s, r0, float(ea), rm, pred.copy())

    # small refinement around best Ea
    ea_best = best[2]
    ea_grid2 = np.linspace(max(800.0, ea_best * 0.92), min(12000.0, ea_best * 1.08), 220)
    for ea in ea_grid2:
        a = np.exp(ea / np.maximum(t_k, 1e-9)) * g_model
        x = np.column_stack((a, np.ones_like(a)))
        try:
            theta, *_ = np.linalg.lstsq(x, r_ohm, rcond=None)
        except np.linalg.LinAlgError:
            continue
        r0 = float(theta[0])
        rm = float(theta[1])
        if r0 <= 0.0 or rm <= 0.0:
            continue
        pred = x @ theta
        s = _rmse_log10(pred, r_ohm)
        if s < best[0]:
            best = (s, r0, float(ea), rm, pred.copy())
    s, r0, ea, rm, pred = best
    return r0, ea, rm, pred, s


def fit_resistance_params(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    random_iters: int = 12000,
    local_passes: int = 180,
    fit_gamma: bool = False,
    gamma_fixed: float = 9.56269682e-1,
    g_weight: float = 0.2,
) -> Tuple[ResistanceFitResult, np.ndarray]:
    t = df["Temperature"].to_numpy(dtype=float)
    r = df["Resistance"].to_numpy(dtype=float)
    if np.nanmedian(t) < 200.0:
        # Accept Celsius input if provided; convert to Kelvin.
        t = t + 273.15

    dT = np.diff(t, prepend=t[0])
    cooling_mask = dT < 0.0
    heating_mask = dT > 0.0

    t_min_fit = float(np.min(t) - 5.0)
    t_max_fit = float(np.max(t) + 5.0)
    r0_est, ea_est, rm_est = _estimate_r0_ea_rm_from_extremes(t, r)
    g_exp_seed = _compute_g_experimental(t, r, r0=r0_est, ea_over_k=ea_est, rm=rm_est)
    w_est, tc_est = _estimate_major_w_tc_from_g(t, g_exp_seed)

    nonzero = dT[np.abs(dT) > 1e-12]
    start_branch_hint = "metal" if (nonzero.size > 0 and nonzero[0] < 0.0) else "insulator"

    # Variables:
    # x = [log10_R0, Ea_over_k, log10_Rm, w, Tc_K, log10_beta, (optional gamma)]
    if fit_gamma:
        lb = np.array([-8.0, 1200.0, 0.7, 2.0, t_min_fit + 8.0, -2.0, 0.5], dtype=float)
        ub = np.array([0.3, 9000.0, 2.4, 20.0, t_max_fit - 8.0, 0.6, 1.4], dtype=float)
        x_default = np.array(
            [
                math.log10(5.35882879e-3),
                5.22047417e3,
                math.log10(262.5 * 4.90025335),
                7.19357064,
                3.32805839e2,
                math.log10(2.52796285e-1),
                float(gamma_fixed),
            ],
            dtype=float,
        )
        x_seed = np.array(
            [
                math.log10(max(r0_est, 1e-12)),
                float(ea_est),
                math.log10(max(rm_est, 1e-12)),
                float(w_est),
                float(tc_est),
                math.log10(2.52796285e-1),
                float(gamma_fixed),
            ],
            dtype=float,
        )
        step = np.array([0.12, 130.0, 0.05, 0.42, 0.42, 0.04, 0.035], dtype=float)
    else:
        lb = np.array([-8.0, 1200.0, 0.7, 2.0, t_min_fit + 8.0, -2.0], dtype=float)
        ub = np.array([0.3, 9000.0, 2.4, 20.0, t_max_fit - 8.0, 0.6], dtype=float)
        x_default = np.array(
            [
                math.log10(5.35882879e-3),
                5.22047417e3,
                math.log10(262.5 * 4.90025335),
                7.19357064,
                3.32805839e2,
                math.log10(2.52796285e-1),
            ],
            dtype=float,
        )
        x_seed = np.array(
            [
                math.log10(max(r0_est, 1e-12)),
                float(ea_est),
                math.log10(max(rm_est, 1e-12)),
                float(w_est),
                float(tc_est),
                math.log10(2.52796285e-1),
            ],
            dtype=float,
        )
        step = np.array([0.12, 130.0, 0.05, 0.42, 0.42, 0.04], dtype=float)

    x_default = np.clip(x_default, lb, ub)
    x_seed = np.clip(x_seed, lb, ub)
    rng = np.random.default_rng(int(seed))

    def _params_from_x(x: np.ndarray) -> YuanhangResistParams:
        p = YuanhangResistParams()
        p.R0 = float(10.0 ** x[0])
        p.Ea_over_k = float(x[1])
        p.Rm0 = float(10.0 ** x[2])
        p.Rm_factor = 1.0
        p.w = float(x[3])
        p.Tc_K = float(x[4])
        p.beta = float(10.0 ** x[5])
        p.gamma = float(x[6]) if fit_gamma else float(gamma_fixed)
        p.width_factor = 1.0
        p.T_min_K = float(t_min_fit)
        p.T_max_K = float(t_max_fit)
        p.reversal_threshold_K = 0.01
        return p

    def _score(x: np.ndarray, branch: str) -> tuple[float, float, float, np.ndarray]:
        p = _params_from_x(x)
        r_pred = _predict_resistance_from_temperature(t, p, start_branch=branch)
        g_pred = _predict_g_from_temperature(t, p, start_branch=branch)
        g_exp = _compute_g_experimental(t, r, r0=p.R0, ea_over_k=p.Ea_over_k, rm=p.Rm)
        err_r = _rmse_log10(r_pred, r)
        err_g = _rmse(g_pred, g_exp)
        penalty = 0.0
        if fit_gamma:
            # Avoid boundary-hugging gamma when minor-loop information is limited.
            penalty = 0.01 * abs(p.gamma - float(gamma_fixed))
        return err_r + float(g_weight) * err_g + penalty, err_r, err_g, r_pred

    candidates = [
        (x_default, "insulator"),
        (x_default, "metal"),
        (x_seed, start_branch_hint),
        (x_seed, "metal" if start_branch_hint == "insulator" else "insulator"),
    ]
    best_score = float("inf")
    best_x = x_seed.copy()
    best_branch = start_branch_hint
    best_r_pred = np.zeros_like(r)
    for x, br in candidates:
        s, _, _, r_pred = _score(x, br)
        if s < best_score:
            best_score = s
            best_x = x.copy()
            best_branch = br
            best_r_pred = r_pred

    for i in range(int(random_iters)):
        if i % 3 == 0:
            prop_scale = np.array([0.18, 300.0, 0.08, 0.8, 1.0, 0.07] + ([0.07] if fit_gamma else []))
            proposal = best_x + rng.normal(scale=prop_scale)
            x = np.clip(proposal, lb, ub)
        else:
            x = lb + rng.random(lb.shape) * (ub - lb)
        br = best_branch if rng.random() < 0.65 else ("metal" if best_branch == "insulator" else "insulator")
        s, _, _, r_pred = _score(x, br)
        if s < best_score:
            best_score = s
            best_x = x.copy()
            best_branch = br
            best_r_pred = r_pred

    for _ in range(int(local_passes)):
        improved = False
        for j in range(best_x.size):
            for direction in (-1.0, 1.0):
                cand = best_x.copy()
                cand[j] = np.clip(cand[j] + direction * step[j], lb[j], ub[j])
                s, _, _, r_pred = _score(cand, best_branch)
                if s < best_score:
                    best_score = s
                    best_x = cand
                    best_r_pred = r_pred
                    improved = True
        if not improved:
            step *= 0.72
            if float(np.max(step)) < 1e-4:
                break

    # Final branch check.
    branch_candidates = []
    for br in ("insulator", "metal"):
        s, err_r, err_g, r_pred = _score(best_x, br)
        branch_candidates.append((s, err_r, err_g, br, r_pred))
    branch_candidates.sort(key=lambda t2: t2[0])
    _, score_all, _, branch_all, pred_all = branch_candidates[0]
    params_all = _params_from_x(best_x)

    rmse_cool = float("nan")
    rmse_heat = float("nan")
    if np.any(cooling_mask):
        rmse_cool = _rmse_log10(pred_all[cooling_mask], r[cooling_mask])
    if np.any(heating_mask):
        rmse_heat = _rmse_log10(pred_all[heating_mask], r[heating_mask])

    result = ResistanceFitResult(
        params=params_all,
        start_branch=branch_all,
        rmse_log10=score_all,
        rmse_log10_cooling=rmse_cool,
        rmse_log10_heating=rmse_heat,
        source_data="",
        n_samples=int(len(df)),
    )
    return result, pred_all


def save_fit_json(result: ResistanceFitResult, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result.to_jsonable(), indent=2))


def _save_fit_plot(df: pd.DataFrame, pred: np.ndarray, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = df["Temperature"].to_numpy(dtype=float)
    r = df["Resistance"].to_numpy(dtype=float)
    dT = np.diff(t, prepend=t[0])
    cooling = dT < 0.0
    heating = dT > 0.0

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.semilogy(t[cooling], r[cooling], ".", alpha=0.75, label="Data (cooling)")
    ax.semilogy(t[heating], r[heating], ".", alpha=0.75, label="Data (heating)")
    ax.semilogy(t, pred, "-", linewidth=1.8, label="Model fit")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Resistance (Ohm)")
    ax.set_title("Custom Resistance Fit")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(p, dpi=220)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit YuanhangResistParams from experimental R(T) data.")
    parser.add_argument("--data", required=True, help="Path to TSV-like experimental data file.")
    parser.add_argument(
        "--save-json",
        default="presets/resistance_100425_chip1_gap3.json",
        help="Where to write fitted parameter preset JSON.",
    )
    parser.add_argument(
        "--save-plot",
        default="outputs/resistance_fit_100425_chip1_gap3.png",
        help="Where to write fit-vs-data plot PNG.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for fitting reproducibility.")
    parser.add_argument("--random-iters", type=int, default=12000, help="Random search iterations.")
    parser.add_argument("--local-passes", type=int, default=180, help="Local coordinate-descent passes.")
    parser.add_argument(
        "--fit-gamma",
        action="store_true",
        help="Also fit gamma. Leave off for major-loop-only datasets where gamma is weakly identifiable.",
    )
    parser.add_argument("--gamma-fixed", type=float, default=9.56269682e-1, help="Fixed gamma when --fit-gamma is off.")
    parser.add_argument("--g-weight", type=float, default=0.2, help="Weight of g(T) consistency term in fitting objective.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_path = Path(args.data)
    df = load_experimental_rt(data_path)
    result, pred = fit_resistance_params(
        df,
        seed=int(args.seed),
        random_iters=int(args.random_iters),
        local_passes=int(args.local_passes),
        fit_gamma=bool(args.fit_gamma),
        gamma_fixed=float(args.gamma_fixed),
        g_weight=float(args.g_weight),
    )
    result.source_data = str(data_path)
    save_fit_json(result, args.save_json)
    _save_fit_plot(df, pred, args.save_plot)

    print(f"Fitted start branch: {result.start_branch}")
    print(f"RMSE log10 (overall): {result.rmse_log10:.6f}")
    print(f"RMSE log10 (cooling): {result.rmse_log10_cooling:.6f}")
    print(f"RMSE log10 (heating): {result.rmse_log10_heating:.6f}")
    print("Fitted resistance parameters:")
    for f in dataclasses.fields(YuanhangResistParams):
        print(f"  {f.name}: {getattr(result.params, f.name)}")
    print(f"Wrote preset JSON: {args.save_json}")
    print(f"Wrote plot PNG: {args.save_plot}")


if __name__ == "__main__":
    main()
