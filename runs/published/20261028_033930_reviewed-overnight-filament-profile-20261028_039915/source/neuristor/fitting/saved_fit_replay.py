"""Replay saved resistance fits with their recorded measurement detector.

This is used to verify stored fit metrics. Time-domain integration always uses
model.HysteresisArray directly; this module does not own a second solver or a
sample database. The saved detector and endpoint clipping remain byte-faithful
at the level of the numerical function bodies.
"""
from __future__ import annotations

import dataclasses
import io
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from neuristor.model import HysteresisArray, YuanhangResistParams

_EPS = 1e-12
_FIT_DTYPE = np.float32


def parse_experimental_rt_bytes(data: bytes, filename: str = "uploaded.tsv") -> pd.DataFrame:
    """Parse a lab R-T export containing Temperature and Resistance columns."""
    if not data:
        raise ValueError("Uploaded file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", skip_blank_lines=True)
    except Exception:
        df = pd.read_csv(io.BytesIO(data), sep=r"\s+", engine="python", skip_blank_lines=True)
    df.columns = [str(c).strip() for c in df.columns]

    # Lab exports commonly use the second row for units (K, Ohm, sec, ...).
    if len(df) > 0 and str(df.iloc[0, 0]).strip().upper() in {"K", "TEMP", "TEMPERATURE"}:
        df = df.iloc[1:].reset_index(drop=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["Temperature", "Resistance"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"{filename} must contain Temperature and Resistance columns.")

    df = df.dropna(subset=required).copy()
    if "Time" in df.columns:
        df = df.dropna(subset=["Time"]).sort_values("Time")
    df = df.reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("Not enough valid Temperature/Resistance samples after parsing.")

    if float(np.nanmedian(df["Temperature"].to_numpy(dtype=float))) < 200.0:
        df["Temperature"] = df["Temperature"].astype(float) + 273.15
    return df


def load_experimental_rt_path(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return parse_experimental_rt_bytes(p.read_bytes(), filename=p.name)


class SampleFitHysteresisArray(HysteresisArray):
    """Replay measured R(T) fits with the detector used when samples were saved.

    The time-domain simulator uses :class:`HysteresisArray` directly. Saved
    sample presets, however, were calibrated with a point-to-point deadband
    replay of the measured temperature sweep. Keeping that replay here verifies
    the fit actually recorded in each preset without changing the simulator's
    convergence-tested hysteresis update.
    """

    def _solve_Tpr(self, delta: np.ndarray, gr: np.ndarray, Tr: np.ndarray) -> np.ndarray:
        """Retain the 1e-6 endpoint clip used when the saved sample fits were made."""

        gr_legacy = np.clip(np.asarray(gr, dtype=_FIT_DTYPE), 1e-6, 1.0 - 1e-6)
        return super()._solve_Tpr(delta, gr_legacy, Tr)

    def _update_reversal(self, T_clamped: np.ndarray) -> None:
        params = self.params
        T_arr = np.asarray(T_clamped, dtype=_FIT_DTYPE)
        dT = T_arr - self.T_last
        mask = np.abs(dT) > float(params.reversal_threshold_K)
        if not np.any(mask):
            self.T_last = T_arr.copy()
            return
        delta_new = np.sign(dT).astype(_FIT_DTYPE, copy=False)
        delta_new[delta_new == 0.0] = self.delta[delta_new == 0.0]
        reversal_mask = mask & (delta_new != self.delta)
        if np.any(reversal_mask):
            g_at_detection = self.g(T_arr)
            self.gr[reversal_mask] = g_at_detection[reversal_mask]
            self.delta[reversal_mask] = delta_new[reversal_mask]
            self.reversed[reversal_mask] = _FIT_DTYPE(1.0)
            self.Tr[reversal_mask] = T_arr[reversal_mask]
            self.Tpr[reversal_mask] = self._solve_Tpr(
                self.delta[reversal_mask], self.gr[reversal_mask], self.Tr[reversal_mask]
            )
        self.T_last = T_arr.copy()


def predict_resistance_trace(
    temperatures_K: Iterable[float],
    params: YuanhangResistParams,
    *,
    start_branch: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate R(T) and g(T) over the measured temperature trajectory."""
    t = np.asarray(list(temperatures_K), dtype=float)
    if t.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    h = SampleFitHysteresisArray(params, size=1, start_branch=start_branch)
    h.initialize(np.asarray([float(t[0])], dtype=float))
    r_pred = np.zeros_like(t, dtype=float)
    g_pred = np.zeros_like(t, dtype=float)
    for idx, temp in enumerate(t):
        r, g = h.evaluate(np.asarray([float(temp)], dtype=float))
        r_pred[idx] = float(r[0])
        g_pred[idx] = float(g[0])
    return r_pred, g_pred


def compute_rt_fit_metrics(
    df: pd.DataFrame,
    params: YuanhangResistParams,
    *,
    start_branch: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Compute log-resistance fit metrics and the model overlay arrays."""
    t = df["Temperature"].to_numpy(dtype=float)
    target = np.maximum(df["Resistance"].to_numpy(dtype=float), _EPS)
    pred, g_pred = predict_resistance_trace(t, params, start_branch=start_branch)
    pred_safe = np.maximum(pred, _EPS)
    err = np.log10(pred_safe) - np.log10(target)
    dT = np.diff(t, prepend=t[0])
    cooling = dT < 0.0
    heating = dT > 0.0

    def _rmse(mask: np.ndarray | None = None) -> float:
        vals = err if mask is None else err[mask]
        if vals.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(vals * vals)))

    target_log = np.log10(target)
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((target_log - float(np.mean(target_log))) ** 2))
    metrics = {
        "rmse_log10": _rmse(),
        "rmse_log10_cooling": _rmse(cooling),
        "rmse_log10_heating": _rmse(heating),
        "r2_log10": float(1.0 - ss_res / ss_tot) if ss_tot > _EPS else float("nan"),
        "mean_log10_error": float(np.mean(err)) if err.size else float("nan"),
        "max_abs_log10_error": float(np.max(np.abs(err))) if err.size else float("nan"),
        "n_samples": float(len(df)),
    }
    return metrics, pred, g_pred


def params_from_dict(raw: Dict[str, Any]) -> YuanhangResistParams:
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(YuanhangResistParams):
        if f.name == 'proximity_function':
            kwargs[f.name] = str(raw.get(f.name, 'yuanhang'))
            continue
        if f.name not in raw:
            raise ValueError(f"Missing resistance parameter: {f.name}")
        kwargs[f.name] = float(raw[f.name])
    return YuanhangResistParams(**kwargs)
