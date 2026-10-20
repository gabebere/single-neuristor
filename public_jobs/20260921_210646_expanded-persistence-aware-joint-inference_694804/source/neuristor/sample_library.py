"""Sample preset library and R-T calibration helpers.

This module keeps the Streamlit app's sample-first workflow separate from the
time-domain simulators.  A sample is a measured R(T) dataset plus a saved
YuanhangResistParams hysteresis preset that downstream simulations can reuse.
"""
from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import functools
import io
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .model import HysteresisArray, YuanhangResistParams


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_LIBRARY_DIR = ROOT / "presets" / "samples"
LEGACY_SPECIMEN_PRESET_PATH = ROOT / "presets" / "resistance_100425_chip1_gap3.json"
DEFAULT_EXPERIMENTAL_DIR = ROOT / "data" / "experimental"

_EPS = 1e-12
_FIT_DTYPE = np.float32


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")
    return slug or "sample"


def make_sample_id(display_name: str) -> str:
    return f"{_slugify(display_name)}_{uuid.uuid4().hex[:8]}"


def sample_path(sample_id: str) -> Path:
    return SAMPLE_LIBRARY_DIR / f"{_slugify(sample_id)}.json"


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
    replay of the measured temperature sweep. Keeping that replay here makes
    the Samples tab show the fit that is actually recorded in each preset
    without changing the simulator's convergence-tested hysteresis update.
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


def params_to_dict(params: YuanhangResistParams) -> Dict[str, float]:
    return {f.name: float(getattr(params, f.name)) for f in dataclasses.fields(YuanhangResistParams)}


def params_from_dict(raw: Dict[str, Any]) -> YuanhangResistParams:
    kwargs: Dict[str, float] = {}
    for f in dataclasses.fields(YuanhangResistParams):
        if f.name not in raw:
            raise ValueError(f"Missing resistance parameter: {f.name}")
        kwargs[f.name] = float(raw[f.name])
    return YuanhangResistParams(**kwargs)


def build_sample_payload(
    *,
    display_name: str,
    source_filename: str,
    source_path: str,
    notes: str,
    params: YuanhangResistParams,
    start_branch: str,
    fit_metrics: Dict[str, float],
    sample_id: str | None = None,
    created_at_utc: str | None = None,
) -> Dict[str, Any]:
    sid = sample_id or make_sample_id(display_name)
    return {
        "schema_version": 1,
        "sample_id": sid,
        "display_name": str(display_name).strip() or sid,
        "created_at_utc": created_at_utc or _utc_now(),
        "updated_at_utc": _utc_now(),
        "source_filename": str(source_filename),
        "source_path": str(source_path),
        "notes": str(notes),
        "start_branch": str(start_branch).strip().lower() if start_branch else "insulator",
        "fit_metrics": {str(k): float(v) for k, v in fit_metrics.items() if np.isfinite(float(v))},
        "resist_params": params_to_dict(params),
    }


def normalize_sample_payload(payload: Dict[str, Any], *, path: Path | None = None, legacy: bool = False) -> Dict[str, Any]:
    raw_params = payload.get("resist_params", payload)
    params = params_from_dict(raw_params)
    display_name = str(payload.get("display_name") or payload.get("name") or (path.stem if path else "Sample"))
    sample_id = str(payload.get("sample_id") or _slugify(display_name))
    source_path = str(payload.get("source_path") or payload.get("source_data") or "")
    source_filename = str(payload.get("source_filename") or (Path(source_path).name if source_path else ""))
    if legacy and source_filename:
        local_candidate = DEFAULT_EXPERIMENTAL_DIR / source_filename
        if local_candidate.exists():
            source_path = str(local_candidate)
    return {
        "schema_version": int(payload.get("schema_version", 1)),
        "sample_id": sample_id,
        "display_name": display_name,
        "created_at_utc": str(payload.get("created_at_utc") or payload.get("generated_at_utc") or ""),
        "updated_at_utc": str(payload.get("updated_at_utc") or payload.get("generated_at_utc") or ""),
        "source_filename": source_filename,
        "source_path": source_path,
        "notes": str(payload.get("notes", "")),
        "start_branch": str(payload.get("start_branch", "insulator")).lower(),
        "fit_metrics": dict(payload.get("fit_metrics", {})),
        "resist_params": params_to_dict(params),
        "_path": "" if path is None else str(path),
        "_legacy": bool(legacy),
    }


def load_sample_json(path: str | Path, *, legacy: bool = False) -> Dict[str, Any]:
    p = Path(path)
    return normalize_sample_payload(json.loads(p.read_text()), path=p, legacy=legacy)


def _sample_library_signature() -> Tuple[Tuple[str, int, int], ...]:
    """Return a cheap cache key that changes whenever a sample file changes."""
    SAMPLE_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    return tuple(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(SAMPLE_LIBRARY_DIR.glob("*.json"))
    )


@functools.lru_cache(maxsize=4)
def _list_samples_cached(signature: Tuple[Tuple[str, int, int], ...]) -> Tuple[Dict[str, Any], ...]:
    samples: List[Dict[str, Any]] = []
    for name, _, _ in signature:
        path = SAMPLE_LIBRARY_DIR / name
        try:
            samples.append(load_sample_json(path))
        except Exception:
            continue
    ordered = sorted(samples, key=lambda s: (bool(s.get("_legacy", False)), str(s.get("display_name", "")).lower()))
    return tuple(ordered)


def list_samples() -> List[Dict[str, Any]]:
    """List sample presets without reparsing unchanged JSON files on every rerun."""
    return copy.deepcopy(list(_list_samples_cached(_sample_library_signature())))


def get_sample(sample_id: str) -> Dict[str, Any] | None:
    for sample in list_samples():
        if str(sample.get("sample_id")) == str(sample_id):
            return sample
    return None


def save_sample(payload: Dict[str, Any]) -> Path:
    SAMPLE_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    normalized = normalize_sample_payload(payload)
    normalized.pop("_path", None)
    normalized.pop("_legacy", None)
    normalized["updated_at_utc"] = _utc_now()
    path = sample_path(str(normalized["sample_id"]))
    path.write_text(json.dumps(normalized, indent=2))
    return path


def delete_sample(sample_id: str) -> None:
    sample = get_sample(sample_id)
    if not sample or sample.get("_legacy"):
        raise ValueError("Only saved custom samples can be deleted.")
    path = Path(str(sample.get("_path", "")))
    if path.exists():
        path.unlink()


def duplicate_sample(sample_id: str, new_name: str | None = None) -> Dict[str, Any]:
    sample = get_sample(sample_id)
    if not sample:
        raise ValueError("Sample not found.")
    display = str(new_name or f"{sample['display_name']} copy")
    payload = dict(sample)
    payload.pop("_path", None)
    payload.pop("_legacy", None)
    payload["sample_id"] = make_sample_id(display)
    payload["display_name"] = display
    payload["created_at_utc"] = _utc_now()
    payload["updated_at_utc"] = _utc_now()
    save_sample(payload)
    return payload


def rename_sample(sample_id: str, new_name: str) -> Dict[str, Any]:
    sample = get_sample(sample_id)
    if not sample:
        raise ValueError("Sample not found.")
    if sample.get("_legacy"):
        raise ValueError("Legacy sample cannot be renamed; duplicate it first.")
    sample["display_name"] = str(new_name).strip() or str(sample["display_name"])
    sample["updated_at_utc"] = _utc_now()
    save_sample(sample)
    return sample
