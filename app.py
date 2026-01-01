"""
Streamlit app with job system for neuristor simulations.

Features:
- Job persistence on disk (single/sweep1d/sweep2d)
- Batch submit
- Interactive Plotly charts with click-to-inspect values
- Clear, human-readable parameter labels
"""
from __future__ import annotations

import csv
import io
import shutil
import dataclasses
import json
import os
import time
import uuid
import queue
import threading
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

import plots
from model import (
    SimulationCancelled,
    YuanhangCircuitParams,
    YuanhangResistParams,
    series_first,
    simulate_yuanhang,
)

try:
    from streamlit_plotly_events import plotly_events

    _HAS_PLOTLY_EVENTS = True
except Exception:
    _HAS_PLOTLY_EVENTS = False


JOB_ROOT = Path(__file__).with_name("jobs")
JOB_ROOT.mkdir(exist_ok=True)

MPL_FIGSIZE_WIDE = (16, 9)
MPL_DPI = 320
MPL_LINEWIDTH = 1.6
MPL_TITLE_SIZE = 16
MPL_LABEL_SIZE = 12
MPL_TICK_SIZE = 11


class JobCancelled(Exception):
    """Raised when a job is cancelled by the user."""


PARAM_LABELS = {
    "Vin": "Input Voltage (Vin) [V]",
    "R_series_kohm": "Series Resistance (R_series) [kOhm]",
    "C_par_pF": "Parasitic Capacitance (C_par) [pF]",
    "Cth_mW_ns_per_K": "Thermal Capacitance (C_th) [mW*ns/K]",
    "Sth_mW_per_K": "Thermal Conductance (S_th) [mW/K]",
    "couple_factor": "Thermal Coupling Factor",
    "Cth_factor": "Thermal Capacitance Factor",
    "noise_strength": "Thermal Noise Strength",
    "T_base_K": "Base Temperature (T_base) [K]",
    "R0": "Pre-exponential Resistance (R0) [Ohm]",
    "Ea_over_k": "Activation Energy / kB (Ea/k) [K]",
    "Rm0": "Metallic Resistance Base (Rm0) [Ohm]",
    "Rm_factor": "Metallic Resistance Factor (Rm_factor)",
    "w": "Hysteresis Width (w) [K]",
    "Tc_K": "Critical Temperature (Tc) [K]",
    "beta": "Hysteresis Sharpness (beta) [1/K]",
    "gamma": "Hysteresis Window (gamma)",
    "width_factor": "Width Factor",
    "T_min_K": "Min Temperature Clamp (T_min) [K]",
    "T_max_K": "Max Temperature Clamp (T_max) [K]",
    "reversal_threshold_K": "Reversal Threshold (dT) [K]",
    "t_end_us": "Simulation Duration [us]",
    "dt_ns": "Time Step [ns]",
    "t_start_us": "Steady-State Start [us]",
    "t_end_window_us": "Steady-State End [us]",
    "threshold_A": "Spike Threshold [A]",
    "start_branch": "Initial Branch",
    "nx": "Lattice Nx",
    "ny": "Lattice Ny",
}


def _label(name: str) -> str:
    return PARAM_LABELS.get(name, name)


def _xy_key(x: float, y: float, digits: int = 12) -> tuple[float, float]:
    return (round(float(x), digits), round(float(y), digits))


def _get_removed_store() -> Dict[str, Any]:
    return st.session_state.setdefault("removed_points", {})


def _get_job_removals(job_id: str) -> Dict[str, Any]:
    store = _get_removed_store()
    if job_id not in store:
        store[job_id] = {"sweep1d": set(), "sweep2d": set(), "time_traces": {}}
    return store[job_id]


def _click_signature(point: Dict[str, Any]) -> tuple:
    return (
        point.get("x"),
        point.get("y"),
        point.get("curveNumber"),
        point.get("pointIndex"),
        point.get("pointNumber"),
    )


def _consume_click(key: str, points: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not points:
        return None
    sig = _click_signature(points[0])
    store = st.session_state.setdefault("last_click_sig", {})
    if store.get(key) == sig:
        return None
    store[key] = sig
    return points[0]


def _build_single_job_from_params(
    base_params: Dict[str, Any],
    overrides: Dict[str, float],
    name_suffix: str,
) -> Dict[str, Any]:
    config = {
        "type": "single",
        "job_name": "",
        "vin": float(base_params.get("vin", 0.0)),
        "vin_list": [],
        "t_end": float(base_params["t_end"]),
        "dt": float(base_params["dt"]),
        "t_start_us": float(base_params["t_start_us"]),
        "t_end_us": float(base_params["t_end_us"]),
        "threshold_A": float(base_params["threshold_A"]),
        "noise_seed": base_params.get("noise_seed"),
        "start_branch": base_params.get("start_branch", "insulator"),
        "lattice_shape": base_params.get("lattice_shape", (1, 1)),
        "resist_params": dict(base_params["resist_params"]),
        "circuit_params": dict(base_params["circuit_params"]),
    }
    for param, value in overrides.items():
        if param == "Vin":
            config["vin"] = float(value)
        elif param in config["resist_params"]:
            config["resist_params"][param] = float(value)
        elif param in config["circuit_params"]:
            config["circuit_params"][param] = float(value)
        else:
            raise ValueError(f"Unknown parameter: {param}")

    base_name = base_params.get("job_name") or ""
    job_name = base_name if base_name else "Point simulation"
    if name_suffix:
        job_name = f"{job_name} | {name_suffix}"
    config["job_name"] = job_name
    return config


def _enqueue_single_from_click(job: Dict[str, Any], overrides: Dict[str, float], label: str) -> str:
    config = _build_single_job_from_params(job["params"], overrides, label)
    new_job = _create_job(config)
    _enqueue_job(new_job["id"])
    return new_job["id"]


def _sweep_value_from_point(point: Dict[str, Any], values: pd.Series) -> Optional[float]:
    x_val = point.get("x")
    if x_val is not None:
        try:
            return float(x_val)
        except (TypeError, ValueError):
            pass
    custom = point.get("customdata")
    if custom is not None:
        try:
            return float(custom)
        except (TypeError, ValueError):
            pass
    idx = point.get("pointIndex")
    if idx is not None:
        try:
            return float(values.iloc[int(idx)])
        except (IndexError, ValueError, TypeError):
            return None
    return None


def _toggle_remove_index(indices: set[int], idx: int) -> bool:
    if idx in indices:
        indices.remove(idx)
        return False
    indices.add(idx)
    return True


def _toggle_remove_xy(points: set[tuple[float, float]], x: float, y: float) -> bool:
    key = _xy_key(x, y)
    if key in points:
        points.remove(key)
        return False
    points.add(key)
    return True


def _apply_time_trace_removals(df: pd.DataFrame, removals: Dict[int, set[int]]) -> pd.DataFrame:
    if not removals:
        return df
    df2 = df.copy()
    col_map = {0: "V_vo2", 1: "V_load", 2: "T_K", 3: "I_vo2", 4: "P_vo2", 5: "R_vo2"}
    max_idx = len(df2) - 1
    for trace_idx, idxs in removals.items():
        col = col_map.get(trace_idx)
        if col is None or col not in df2.columns:
            continue
        safe = [i for i in idxs if 0 <= i <= max_idx]
        if safe:
            df2.loc[safe, col] = np.nan
    return df2


def _apply_sweep1d_removals(df: pd.DataFrame, removed_idx: set[int]) -> pd.DataFrame:
    if not removed_idx:
        return df
    df2 = df.copy()
    max_idx = len(df2) - 1
    safe = [i for i in removed_idx if 0 <= i <= max_idx]
    if not safe:
        return df2
    for col in ["Vmax", "Pmax", "Pmin", "Tmax", "Tmin", "freq_MHz", "ISI_mean_us"]:
        if col in df2.columns:
            df2.loc[safe, col] = np.nan
    return df2


def _apply_sweep2d_removals(df: pd.DataFrame, removed_xy: set[tuple[float, float]], x_label: str, y_label: str) -> pd.DataFrame:
    if not removed_xy:
        return df
    df2 = df.copy()
    keys = [_xy_key(x, y) for x, y in zip(df2[x_label], df2[y_label])]
    mask = [key in removed_xy for key in keys]
    df2.loc[mask, "freq_MHz"] = np.nan
    return df2


def _chunked(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _num_input(label: str, key: str, value: float | None = None, **kwargs):
    if key in st.session_state:
        return st.number_input(label, key=key, **kwargs)
    if value is None:
        return st.number_input(label, key=key, **kwargs)
    return st.number_input(label, key=key, value=value, **kwargs)


def _int_input(label: str, key: str, value: int | None = None, **kwargs):
    if key in st.session_state:
        return st.number_input(label, key=key, step=1, **kwargs)
    if value is None:
        return st.number_input(label, key=key, step=1, **kwargs)
    return st.number_input(label, key=key, value=value, step=1, **kwargs)


def _text_input(label: str, key: str, value: str | None = None, **kwargs):
    if key in st.session_state:
        return st.text_input(label, key=key, **kwargs)
    if value is None:
        return st.text_input(label, key=key, **kwargs)
    return st.text_input(label, key=key, value=value, **kwargs)


def _toggle_input(label: str, key: str, value: bool = False, **kwargs):
    if key in st.session_state:
        return st.toggle(label, key=key, **kwargs)
    return st.toggle(label, key=key, value=value, **kwargs)


def _render_input_grid(keys: List[str], input_fn, columns: int = 4) -> None:
    for row in _chunked(keys, columns):
        cols = st.columns(columns)
        for col, key in zip(cols, row):
            with col:
                input_fn(_label(key), key=key)

def _paper_params() -> tuple[YuanhangResistParams, YuanhangCircuitParams]:
    resist = YuanhangResistParams()
    circuit = YuanhangCircuitParams()
    resist.R0 = 5.35882879e-3
    resist.Ea_over_k = 5.22047417e3
    resist.Rm0 = 262.5
    resist.Rm_factor = 4.90025335
    resist.w = 7.19357064
    resist.Tc_K = 3.32805839e2
    resist.beta = 2.52796285e-1
    resist.gamma = 9.56269682e-1
    resist.width_factor = 1.0
    resist.T_min_K = 305.0
    resist.T_max_K = 370.0
    resist.reversal_threshold_K = 0.01

    circuit.R_series_kohm = 12.0
    circuit.C_par_pF = 145.34619293
    circuit.Cth_mW_ns_per_K = 49.62776831
    circuit.Sth_mW_per_K = 0.20558726
    circuit.couple_factor = 0.0
    circuit.Cth_factor = 1.0
    circuit.noise_strength = 0.0
    circuit.dimension = 1
    circuit.T_base_K = 325.0
    return resist, circuit


# -----------------------------
# Job persistence
# -----------------------------


def _job_id() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def _job_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id


def _job_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _load_job(job_id: str) -> Dict[str, Any]:
    with open(_job_path(job_id), "r") as f:
        return json.load(f)


def _save_job(job: Dict[str, Any]) -> None:
    path = _job_path(job["id"])
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(job, f, indent=2)
    os.replace(tmp_path, path)


def _append_job_log(job: Dict[str, Any], line: str) -> None:
    log_path = Path(job["log_path"])
    with open(log_path, "a") as f:
        f.write(line + "\n")


def _list_jobs() -> List[Dict[str, Any]]:
    jobs = []
    for job_dir in sorted(JOB_ROOT.iterdir(), reverse=True):
        if not job_dir.is_dir():
            continue
        job_file = job_dir / "job.json"
        if not job_file.exists():
            continue
        with open(job_file, "r") as f:
            jobs.append(json.load(f))
    return jobs


def _job_status(job_id: str) -> str:
    try:
        return _load_job(job_id).get("status", "")
    except Exception:
        return ""


def _job_cancel_requested(job_id: str) -> bool:
    return _job_status(job_id) in {"cancel_requested", "cancelled"}


def _recover_pending_jobs() -> List[str]:
    pending = []
    for job in _list_jobs():
        if job.get("status") in {"queued", "running"}:
            job["status"] = "queued"
            _save_job(job)
            pending.append(job["id"])
    return pending


def _job_progress_cb(job: Dict[str, Any]):
    def _cb(msg: str) -> None:
        _append_job_log(job, msg)
        if _job_cancel_requested(job["id"]):
            raise JobCancelled("cancelled")

    return _cb


def _job_worker_loop(job_queue: "queue.Queue[str]") -> None:
    while True:
        job_id = job_queue.get()
        if job_id is None:
            job_queue.task_done()
            break
        job_path = _job_path(job_id)
        if not job_path.exists():
            job_queue.task_done()
            continue
        try:
            job = _load_job(job_id)
        except Exception:
            job_queue.task_done()
            continue
        if job.get("status") in {"cancel_requested", "cancelled"}:
            job["status"] = "cancelled"
            _append_job_log(job, "[job] cancelled before start")
            _save_job(job)
            job_queue.task_done()
            continue
        if job.get("status") not in {"queued", "running"}:
            job_queue.task_done()
            continue
        job["status"] = "running"
        _save_job(job)
        _append_job_log(job, f"[job] starting {job['type']}")
        try:
            _run_job_core(job, progress_cb=_job_progress_cb(job))
            if job.get("status") == "cancel_requested":
                raise JobCancelled("cancelled")
            job["status"] = "completed"
            _append_job_log(job, f"[job] completed {job['type']}")
        except (JobCancelled, SimulationCancelled):
            job["status"] = "cancelled"
            _append_job_log(job, "[job] cancelled")
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            _append_job_log(job, f"[job] error: {exc}")
        _save_job(job)
        job_queue.task_done()


@st.cache_resource(show_spinner=False)
def _get_worker_queue() -> "queue.Queue[str]":
    job_queue: "queue.Queue[str]" = queue.Queue()
    worker = threading.Thread(target=_job_worker_loop, args=(job_queue,), daemon=True)
    worker.start()
    pending = _recover_pending_jobs()
    for job_id in pending:
        job_queue.put(job_id)
    return job_queue


def _ensure_worker() -> "queue.Queue[str]":
    return _get_worker_queue()


def _enqueue_job(job_id: str) -> None:
    _ensure_worker().put(job_id)


# -----------------------------
# Helpers
# -----------------------------


def _param_names() -> List[str]:
    resist = {f.name for f in dataclasses.fields(YuanhangResistParams)}
    circuit = {f.name for f in dataclasses.fields(YuanhangCircuitParams)}
    names = ["Vin"] + sorted(list(resist | circuit))
    return names


def _param_label_options() -> List[str]:
    return [_label(name) for name in _param_names()]


def _param_name_from_label(label: str) -> str:
    for name in _param_names():
        if _label(name) == label:
            return name
    return label


def _init_defaults() -> None:
    required_keys = {
        "vin",
        "vin_list",
        "t_end_us",
        "dt_ns",
        "t_start_us",
        "t_end_window_us",
        "threshold_A",
        "start_branch",
        "noise_seed",
        "nx",
        "ny",
        "batch_jobs",
        "terminal_log",
        "enable_point_removal",
        "removed_points",
        "last_click_sig",
        "job_name_single",
        "job_name_sweep1d",
        "job_name_sweep2d",
        "sweep_start",
        "sweep_stop",
        "coarse_step",
        "fine_step",
        "x_start",
        "x_stop",
        "x_step",
        "y_start",
        "y_stop",
        "y_step",
    }
    missing = [k for k in required_keys if k not in st.session_state]
    if st.session_state.get("_init_done") and not missing:
        return
    resist, circuit = _paper_params()
    for f in dataclasses.fields(YuanhangResistParams):
        if f.name not in st.session_state:
            st.session_state[f.name] = getattr(resist, f.name)
    for f in dataclasses.fields(YuanhangCircuitParams):
        if f.name not in st.session_state:
            st.session_state[f.name] = getattr(circuit, f.name)
    st.session_state.setdefault("vin", 14.5)
    st.session_state.setdefault("vin_list", "")
    st.session_state.setdefault("t_end_us", 300.0)
    st.session_state.setdefault("dt_ns", 10.0)
    st.session_state.setdefault("t_start_us", 25.0)
    st.session_state.setdefault("t_end_window_us", 300.0)
    st.session_state.setdefault("threshold_A", 1e-3)
    st.session_state.setdefault("start_branch", "insulator")
    st.session_state.setdefault("noise_seed", "")
    st.session_state.setdefault("nx", 1)
    st.session_state.setdefault("ny", 1)
    st.session_state.setdefault("batch_jobs", [])
    st.session_state.setdefault("terminal_log", "")
    st.session_state.setdefault("enable_point_removal", False)
    st.session_state.setdefault("removed_points", {})
    st.session_state.setdefault("last_click_sig", {})
    st.session_state.setdefault("job_name_single", "")
    st.session_state.setdefault("job_name_sweep1d", "")
    st.session_state.setdefault("job_name_sweep2d", "")
    st.session_state.setdefault("sweep_start", 0.0)
    st.session_state.setdefault("sweep_stop", 20.0)
    st.session_state.setdefault("coarse_step", 0.5)
    st.session_state.setdefault("fine_step", 0.05)
    st.session_state.setdefault("x_start", "")
    st.session_state.setdefault("x_stop", "")
    st.session_state.setdefault("x_step", 0.5)
    st.session_state.setdefault("y_start", "")
    st.session_state.setdefault("y_stop", "")
    st.session_state.setdefault("y_step", 10.0)
    st.session_state["_init_done"] = True


def _build_params() -> Tuple[YuanhangResistParams, YuanhangCircuitParams, Tuple[int, int]]:
    resist = YuanhangResistParams(
        **{f.name: float(st.session_state[f.name]) for f in dataclasses.fields(YuanhangResistParams)}
    )
    circuit = YuanhangCircuitParams(
        **{f.name: float(st.session_state[f.name]) for f in dataclasses.fields(YuanhangCircuitParams)}
    )
    nx = max(1, int(st.session_state["nx"]))
    ny = max(1, int(st.session_state["ny"]))
    circuit.dimension = 1 if (nx == 1 or ny == 1) else 2
    return resist, circuit, (nx, ny)


def _apply_preset(paper: bool) -> None:
    resist, circuit = _paper_params() if paper else (YuanhangResistParams(), YuanhangCircuitParams())
    for f in dataclasses.fields(YuanhangResistParams):
        st.session_state[f.name] = getattr(resist, f.name)
    for f in dataclasses.fields(YuanhangCircuitParams):
        st.session_state[f.name] = getattr(circuit, f.name)
    st.session_state["t_end_us"] = 300.0
    st.session_state["dt_ns"] = 10.0
    st.session_state["t_start_us"] = 25.0
    st.session_state["t_end_window_us"] = 300.0
    st.session_state["threshold_A"] = 1e-3


def _update_terminal(line: str, placeholder) -> None:
    st.session_state["terminal_log"] += line + "\n"
    placeholder.code(st.session_state["terminal_log"])


def _sim_to_csv(simout: Dict[str, Any], outpath: Path) -> None:
    keys = ["time_s", "V_node", "I_load", "I_vo2", "T_K", "R_vo2", "g"]
    series = {k: series_first(simout[k]) for k in keys if k != "time_s"}
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i, t in enumerate(simout["time_s"]):
            row = [t] + [series[k][i] for k in keys if k != "time_s"]
            writer.writerow(row)


def _csv_outputs(outputs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        o
        for o in outputs
        if o["path"].endswith(".csv") or o["path"].endswith(".pcsv")
    ]


# -----------------------------
# Plotly helpers
# -----------------------------


def _plot_time_traces(df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df["time_us"], y=df["V_vo2"], name="V_vo2"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time_us"], y=df["V_load"], name="V_load"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time_us"], y=df["T_K"], name="T_vo2"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["time_us"], y=df["I_vo2"], name="I_vo2"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["time_us"], y=df["P_vo2"], name="P_vo2"), row=4, col=1)
    if "R_vo2" in df.columns:
        fig.add_trace(go.Scatter(x=df["time_us"], y=df["R_vo2"], name="R_vo2"), row=5, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (K)", row=2, col=1)
    fig.update_yaxes(title_text="Current (A)", row=3, col=1)
    fig.update_yaxes(title_text="Power (W)", row=4, col=1)
    fig.update_yaxes(title_text="Resistance (Ω)", row=5, col=1)
    fig.update_xaxes(title_text="time (us)", row=5, col=1)
    fig.update_layout(height=980, title=title, legend=dict(orientation="h"))
    return fig


def _plot_sweep_metrics(df: pd.DataFrame, free_label: str) -> List[go.Figure]:
    figs: List[go.Figure] = []
    figs.append(
        go.Figure(
            data=[
                go.Scatter(
                    x=df["value"],
                    y=df["Vmax"],
                    mode="lines+markers",
                    customdata=df["value"],
                )
            ]
        )
    )
    figs[-1].update_layout(title=f"Vmax vs {free_label}", xaxis_title=free_label, yaxis_title="Vmax (V)")

    fig_p = go.Figure()
    fig_p.add_trace(
        go.Scatter(x=df["value"], y=df["Pmax"], mode="lines+markers", name="Pmax", customdata=df["value"])
    )
    fig_p.add_trace(
        go.Scatter(x=df["value"], y=df["Pmin"], mode="lines+markers", name="Pmin", customdata=df["value"])
    )
    fig_p.update_layout(title=f"Pmax/Pmin vs {free_label}", xaxis_title=free_label, yaxis_title="Power (W)")
    figs.append(fig_p)

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=df["value"], y=df["Tmax"], mode="lines+markers", name="Tmax", customdata=df["value"]))
    fig_t.add_trace(go.Scatter(x=df["value"], y=df["Tmin"], mode="lines+markers", name="Tmin", customdata=df["value"]))
    fig_t.update_layout(title=f"Tmax/Tmin vs {free_label}", xaxis_title=free_label, yaxis_title="Temperature (K)")
    figs.append(fig_t)

    fig_f = go.Figure(
        data=[go.Scatter(x=df["value"], y=df["freq_MHz"], mode="lines+markers", customdata=df["value"])]
    )
    fig_f.update_layout(title=f"Frequency vs {free_label}", xaxis_title=free_label, yaxis_title="Frequency (MHz)")
    figs.append(fig_f)

    fig_isi = go.Figure(
        data=[go.Scatter(x=df["value"], y=df["ISI_mean_us"], mode="lines+markers", customdata=df["value"])]
    )
    fig_isi.update_layout(title=f"Mean ISI vs {free_label}", xaxis_title=free_label, yaxis_title="Mean ISI (us)")
    figs.append(fig_isi)
    return figs


def _plot_frequency_2d(
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    removed_xy: set[tuple[float, float]] | None = None,
) -> Tuple[go.Figure, go.Figure]:
    pivot = df.pivot(index="y", columns="x", values="freq_MHz")
    x_vals = pivot.columns.values
    y_vals = pivot.index.values
    z = pivot.values
    if removed_xy:
        for yi, y in enumerate(y_vals):
            for xi, x in enumerate(x_vals):
                if _xy_key(x, y) in removed_xy:
                    z[yi, xi] = np.nan
    finite = z[np.isfinite(z)]
    if finite.size:
        zmin = float(np.nanmin(finite))
        zmax = float(np.nanmax(finite))
    else:
        zmin, zmax = 0.0, 1.0
    colorscale = "Viridis"

    heatmap = go.Figure(
        data=go.Heatmap(
            x=x_vals,
            y=y_vals,
            z=z,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Frequency (MHz)"),
            colorscale=colorscale,
        )
    )
    heatmap.update_layout(
        title=f"Frequency heatmap: {x_label} vs {y_label}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=650,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    xs, ys, zs = [], [], []
    for yi, y in enumerate(y_vals):
        for xi, x in enumerate(x_vals):
            if np.isfinite(z[yi, xi]):
                xs.append(x)
                ys.append(y)
                zs.append(z[yi, xi])

    scatter3d = go.Figure(
        data=[
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(
                    size=4,
                    color=zs,
                    colorscale=colorscale,
                    cmin=zmin,
                    cmax=zmax,
                    colorbar=dict(title="Frequency (MHz)"),
                ),
            )
        ]
    )
    scatter3d.update_layout(
        title=f"Frequency 3D: {x_label} vs {y_label}",
        scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title="Frequency (MHz)"),
        height=650,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return heatmap, scatter3d


def _show_plotly_with_click(
    fig,
    label: str,
    key: str | None = None,
    use_events: bool = False,
    show_click: bool = True,
) -> None:
    if use_events and _HAS_PLOTLY_EVENTS:
        height = int(fig.layout.height) if fig.layout.height else 450
        points = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            key=key,
            override_height=height,
            override_width="100%",
        )
        if points and show_click:
            st.write(f"{label} click:", points[0])
        return points
    st.plotly_chart(fig, use_container_width=True)
    return []


def _rerun() -> None:
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=MPL_DPI)
    buf.seek(0)
    return buf.read()


def _launch_matplotlib_viewer(csv_path: str, x_label: str, y_label: str, title: str) -> None:
    script_path = Path(__file__).with_name("matplotlib_viewer.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Missing matplotlib_viewer.py at {script_path}")
    cmd = [
        sys.executable,
        str(script_path),
        "--csv",
        csv_path,
        "--x-label",
        x_label,
        "--y-label",
        y_label,
        "--title",
        title,
    ]
    subprocess.Popen(cmd, cwd=str(Path(__file__).parent))


def _matplotlib_time_traces_figure(df: pd.DataFrame, title: str):
    fig, axes = plt.subplots(5, 1, figsize=MPL_FIGSIZE_WIDE, sharex=True)
    axes[0].plot(df["time_us"], df["V_vo2"], label="V_vo2", linewidth=MPL_LINEWIDTH)
    axes[0].plot(df["time_us"], df["V_load"], label="V_load", linewidth=MPL_LINEWIDTH)
    axes[0].set_ylabel("Voltage (V)", fontsize=MPL_LABEL_SIZE)
    axes[0].legend(loc="best", fontsize=MPL_TICK_SIZE)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["time_us"], df["T_K"], color="tab:red", linewidth=MPL_LINEWIDTH)
    axes[1].set_ylabel("T_vo2 (K)", fontsize=MPL_LABEL_SIZE)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["time_us"], df["I_vo2"], color="tab:green", linewidth=MPL_LINEWIDTH)
    axes[2].set_ylabel("I_vo2 (A)", fontsize=MPL_LABEL_SIZE)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(df["time_us"], df["P_vo2"], color="tab:purple", linewidth=MPL_LINEWIDTH)
    axes[3].set_ylabel("P_vo2 (W)", fontsize=MPL_LABEL_SIZE)
    axes[3].grid(True, alpha=0.3)

    if "R_vo2" in df.columns:
        axes[4].plot(df["time_us"], df["R_vo2"], color="tab:orange", linewidth=MPL_LINEWIDTH)
    axes[4].set_ylabel("R_vo2 (Ω)", fontsize=MPL_LABEL_SIZE)
    axes[4].set_xlabel("time (us)", fontsize=MPL_LABEL_SIZE)
    axes[4].grid(True, alpha=0.3)

    for ax in axes:
        ax.tick_params(labelsize=MPL_TICK_SIZE)

    fig.suptitle(title, fontsize=MPL_TITLE_SIZE, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def _matplotlib_sweep1d_figs(df: pd.DataFrame, free_label: str) -> List[Tuple[str, Any]]:
    figs: List[Tuple[str, Any]] = []

    fig_v, ax_v = plt.subplots(figsize=MPL_FIGSIZE_WIDE)
    ax_v.plot(df["value"], df["Vmax"], "o-", linewidth=MPL_LINEWIDTH, markersize=4)
    ax_v.set_xlabel(free_label, fontsize=MPL_LABEL_SIZE)
    ax_v.set_ylabel("Vmax (V)", fontsize=MPL_LABEL_SIZE)
    ax_v.set_title(f"Vmax vs {free_label}", fontsize=MPL_TITLE_SIZE)
    ax_v.grid(True, alpha=0.3)
    ax_v.tick_params(labelsize=MPL_TICK_SIZE)
    fig_v.tight_layout()
    figs.append(("vmax", fig_v))

    fig_p, ax_p = plt.subplots(figsize=MPL_FIGSIZE_WIDE)
    ax_p.plot(df["value"], df["Pmax"], "o-", label="Pmax", linewidth=MPL_LINEWIDTH, markersize=4)
    ax_p.plot(df["value"], df["Pmin"], "o-", label="Pmin", linewidth=MPL_LINEWIDTH, markersize=4)
    ax_p.set_xlabel(free_label, fontsize=MPL_LABEL_SIZE)
    ax_p.set_ylabel("Power (W)", fontsize=MPL_LABEL_SIZE)
    ax_p.set_title(f"Pmax/Pmin vs {free_label}", fontsize=MPL_TITLE_SIZE)
    ax_p.legend(loc="best", fontsize=MPL_TICK_SIZE)
    ax_p.grid(True, alpha=0.3)
    ax_p.tick_params(labelsize=MPL_TICK_SIZE)
    fig_p.tight_layout()
    figs.append(("power", fig_p))

    fig_t, ax_t = plt.subplots(figsize=MPL_FIGSIZE_WIDE)
    ax_t.plot(df["value"], df["Tmax"], "o-", label="Tmax", linewidth=MPL_LINEWIDTH, markersize=4)
    ax_t.plot(df["value"], df["Tmin"], "o-", label="Tmin", linewidth=MPL_LINEWIDTH, markersize=4)
    ax_t.set_xlabel(free_label, fontsize=MPL_LABEL_SIZE)
    ax_t.set_ylabel("Temperature (K)", fontsize=MPL_LABEL_SIZE)
    ax_t.set_title(f"Tmax/Tmin vs {free_label}", fontsize=MPL_TITLE_SIZE)
    ax_t.legend(loc="best", fontsize=MPL_TICK_SIZE)
    ax_t.grid(True, alpha=0.3)
    ax_t.tick_params(labelsize=MPL_TICK_SIZE)
    fig_t.tight_layout()
    figs.append(("temp", fig_t))

    fig_f, ax_f = plt.subplots(figsize=MPL_FIGSIZE_WIDE)
    ax_f.plot(df["value"], df["freq_MHz"], "o-", linewidth=MPL_LINEWIDTH, markersize=4)
    ax_f.set_xlabel(free_label, fontsize=MPL_LABEL_SIZE)
    ax_f.set_ylabel("Frequency (MHz)", fontsize=MPL_LABEL_SIZE)
    ax_f.set_title(f"Frequency vs {free_label}", fontsize=MPL_TITLE_SIZE)
    ax_f.grid(True, alpha=0.3)
    ax_f.tick_params(labelsize=MPL_TICK_SIZE)
    fig_f.tight_layout()
    figs.append(("freq", fig_f))

    fig_isi, ax_isi = plt.subplots(figsize=MPL_FIGSIZE_WIDE)
    ax_isi.plot(df["value"], df["ISI_mean_us"], "o-", linewidth=MPL_LINEWIDTH, markersize=4)
    ax_isi.set_xlabel(free_label, fontsize=MPL_LABEL_SIZE)
    ax_isi.set_ylabel("Mean ISI (us)", fontsize=MPL_LABEL_SIZE)
    ax_isi.set_title(f"Mean ISI vs {free_label}", fontsize=MPL_TITLE_SIZE)
    ax_isi.grid(True, alpha=0.3)
    ax_isi.tick_params(labelsize=MPL_TICK_SIZE)
    fig_isi.tight_layout()
    figs.append(("isi", fig_isi))

    return figs


def _edges_from_centers(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)
    edges = np.zeros(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def _matplotlib_sweep2d_figs(df: pd.DataFrame, x_label: str, y_label: str) -> List[Tuple[str, Any]]:
    pivot = df.pivot(index="y", columns="x", values="freq_MHz")
    x_vals = pivot.columns.values
    y_vals = pivot.index.values
    z = pivot.values
    mask = ~np.isfinite(z)
    masked = np.ma.array(z, mask=mask)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    fig_h, ax_h = plt.subplots(figsize=MPL_FIGSIZE_WIDE)
    x_edges = _edges_from_centers(x_vals)
    y_edges = _edges_from_centers(y_vals)
    img = ax_h.pcolormesh(x_edges, y_edges, masked, shading="auto", cmap=cmap)
    fig_h.colorbar(img, ax=ax_h, label="Frequency (MHz)")
    ax_h.set_xlabel(x_label, fontsize=MPL_LABEL_SIZE)
    ax_h.set_ylabel(y_label, fontsize=MPL_LABEL_SIZE)
    ax_h.set_title(f"Frequency heatmap: {x_label} vs {y_label}", fontsize=MPL_TITLE_SIZE)
    ax_h.tick_params(labelsize=MPL_TICK_SIZE)
    fig_h.tight_layout()
    figs = [("heatmap", fig_h)]

    fig_3d = plt.figure(figsize=MPL_FIGSIZE_WIDE)
    ax3d = fig_3d.add_subplot(111, projection="3d")
    xs, ys, zs = [], [], []
    for yi, y in enumerate(y_vals):
        for xi, x in enumerate(x_vals):
            if np.isfinite(z[yi, xi]):
                xs.append(x)
                ys.append(y)
                zs.append(z[yi, xi])
    ax3d.scatter(xs, ys, zs, c=zs, cmap="viridis", s=12)
    ax3d.set_xlabel(x_label, fontsize=MPL_LABEL_SIZE)
    ax3d.set_ylabel(y_label, fontsize=MPL_LABEL_SIZE)
    ax3d.set_zlabel("Frequency (MHz)", fontsize=MPL_LABEL_SIZE)
    ax3d.set_title(f"Frequency 3D: {x_label} vs {y_label}", fontsize=MPL_TITLE_SIZE)
    fig_3d.tight_layout()
    figs.append(("scatter3d", fig_3d))
    return figs


# -----------------------------
# Job execution
# -----------------------------


def _create_job(config: Dict[str, Any]) -> Dict[str, Any]:
    job_id = _job_id()
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "id": job_id,
        "name": config.get("job_name", ""),
        "type": config["type"],
        "status": "queued",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": config,
        "outputs": [],
        "log_path": str(job_dir / "log.txt"),
    }
    _save_job(job)
    return job


def _run_job_core(job: Dict[str, Any], progress_cb=None) -> None:
    config = job["params"]
    resist = YuanhangResistParams(**config["resist_params"])
    circuit = YuanhangCircuitParams(**config["circuit_params"])
    lattice_shape = tuple(config.get("lattice_shape", (1, 1)))

    if job["type"] == "single":
        vins = config["vin_list"] if config["vin_list"] else [config["vin"]]
        results = {}

        counter = {"i": 0}

        def cancel_cb() -> bool:
            counter["i"] += 1
            if counter["i"] % 200 != 0:
                return False
            return _job_cancel_requested(job["id"])

        for v in vins:
            if _job_cancel_requested(job["id"]):
                raise JobCancelled("cancelled")
            sim = simulate_yuanhang(
                Vin=v,
                t_end=config["t_end"],
                dt=config["dt"],
                resist_params=resist,
                circuit_params=circuit,
                start_branch=config["start_branch"],
                lattice_shape=lattice_shape,
                noise_seed=config["noise_seed"],
                cancel_cb=cancel_cb,
            )
            results[v] = sim

        for v, sim in results.items():
            vin_label = f"{v:.3f}".replace(".", "p")
            outpath = _job_dir(job["id"]) / f"sim_Vin_{vin_label}.csv"
            _sim_to_csv(sim, outpath)
            job["outputs"].append({"label": f"Simulation CSV (Vin={v})", "path": str(outpath)})
            _append_job_log(job, f"[single] wrote {outpath.name}")
            _save_job(job)
        return

    if job["type"] == "sweep1d":
        cb = progress_cb
        result = plots.sweep_free_variable_coarse_fine(
            param_name=config["param"],
            start=config["start"],
            stop=config["stop"],
            coarse_step=config["coarse_step"],
            fine_step=config["fine_step"],
            Vin=config["vin"],
            t_end=config["t_end"],
            dt=config["dt"],
            resist_params=resist,
            circuit_params=circuit,
            start_branch=config["start_branch"],
            lattice_shape=lattice_shape,
            noise_seed=config["noise_seed"],
            t_start_us=config["t_start_us"],
            t_end_us=config["t_end_us"],
            threshold_A=config["threshold_A"],
            progress_cb=cb,
        )
        if result["fine_results"] is None:
            _append_job_log(job, "[sweep1d] no oscillatory band found")
            _save_job(job)
            return

        outpath = _job_dir(job["id"]) / "sweep1d_results.csv"
        keys = ["values", "Vmax", "Pmax", "Pmin", "Tmax", "Tmin", "ISI_mean_us", "freq_MHz"]
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["value"] + keys[1:])
            for i, val in enumerate(result["fine_results"]["values"]):
                writer.writerow([val] + [result["fine_results"][k][i] for k in keys[1:]])
        job["outputs"].append({"label": "Sweep1D results CSV", "path": str(outpath)})
        _append_job_log(job, f"[sweep1d] wrote {outpath.name}")
        _save_job(job)
        return

    if job["type"] == "sweep2d":
        cb = progress_cb
        result = plots.sweep_frequency_2d(
            param_x=config["param_x"],
            param_y=config["param_y"],
            x_start=config["x_start"],
            x_stop=config["x_stop"],
            x_step=config["x_step"],
            y_start=config["y_start"],
            y_stop=config["y_stop"],
            y_step=config["y_step"],
            Vin=config["vin"],
            t_end=config["t_end"],
            dt=config["dt"],
            resist_params=resist,
            circuit_params=circuit,
            start_branch=config["start_branch"],
            lattice_shape=lattice_shape,
            noise_seed=config["noise_seed"],
            t_start_us=config["t_start_us"],
            t_end_us=config["t_end_us"],
            threshold_A=config["threshold_A"],
            row_early_stop=config["row_early_stop"],
            col_early_stop=config["col_early_stop"],
            progress_cb=cb,
        )
        outpath = _job_dir(job["id"]) / "sweep2d_frequency.csv"
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([config["param_x"], config["param_y"], "freq_MHz"])
            for yi, y in enumerate(result["y_values"]):
                for xi, x in enumerate(result["x_values"]):
                    writer.writerow([x, y, result["freq_MHz"][yi][xi]])
        job["outputs"].append({"label": "Sweep2D frequency CSV", "path": str(outpath)})
        _append_job_log(job, f"[sweep2d] wrote {outpath.name}")
        _save_job(job)
        return


# -----------------------------
# UI rendering
# -----------------------------


def _render_top_bar() -> None:
    col_text, _ = st.columns([6, 1])
    with col_text:
        st.markdown("## Quantum Materials for Neuromorphic Computation")
        st.markdown("VO2 Simulations")


def _render_sidebar() -> None:
    st.sidebar.markdown("")
    choice = st.sidebar.radio(
        "Select mode",
        ["Single Simulation", "Sweep over Free Variable", "2D Frequency Sweep", "Jobs"],
    )
    st.session_state["mode"] = choice


def _render_terminal() -> None:
    st.subheader("Job Terminal")
    terminal_placeholder = st.empty()
    terminal_placeholder.code(st.session_state.get("terminal_log", ""))


def _inputs_common() -> None:
    st.markdown("### Circuit / Thermal")
    circuit_keys = [f.name for f in dataclasses.fields(YuanhangCircuitParams)]
    _render_input_grid(circuit_keys, _num_input, columns=4)

    with st.expander("Time / Window", expanded=False):
        time_keys = ["t_end_us", "dt_ns", "t_start_us", "t_end_window_us", "threshold_A"]
        _render_input_grid(time_keys, _num_input, columns=4)

    with st.expander("Lattice", expanded=False):
        _render_input_grid(["nx", "ny"], _int_input, columns=4)

    with st.expander("Hysteresis / Resistance", expanded=False):
        resist_keys = [f.name for f in dataclasses.fields(YuanhangResistParams)]
        _render_input_grid(resist_keys, _num_input, columns=4)


def _build_job_config_single() -> Dict[str, Any]:
    resist, circuit, lattice = _build_params()
    vin_list = [float(x.strip()) for x in st.session_state["vin_list"].split(",") if x.strip()]
    return {
        "type": "single",
        "job_name": st.session_state.get("job_name_single", "").strip(),
        "vin": float(st.session_state["vin"]),
        "vin_list": vin_list,
        "t_end": float(st.session_state["t_end_us"]) * 1e-6,
        "dt": float(st.session_state["dt_ns"]) * 1e-9,
        "t_start_us": float(st.session_state["t_start_us"]),
        "t_end_us": float(st.session_state["t_end_window_us"]),
        "threshold_A": float(st.session_state["threshold_A"]),
        "noise_seed": None if st.session_state["noise_seed"] == "" else int(st.session_state["noise_seed"]),
        "start_branch": st.session_state["start_branch"],
        "lattice_shape": lattice,
        "resist_params": dataclasses.asdict(resist),
        "circuit_params": dataclasses.asdict(circuit),
    }


def _build_job_config_sweep1d(param_label: str) -> Dict[str, Any]:
    resist, circuit, lattice = _build_params()
    return {
        "type": "sweep1d",
        "job_name": st.session_state.get("job_name_sweep1d", "").strip(),
        "param": _param_name_from_label(param_label),
        "start": float(st.session_state["sweep_start"]),
        "stop": float(st.session_state["sweep_stop"]),
        "coarse_step": float(st.session_state["coarse_step"]),
        "fine_step": float(st.session_state["fine_step"]),
        "vin": float(st.session_state["vin"]),
        "t_end": float(st.session_state["t_end_us"]) * 1e-6,
        "dt": float(st.session_state["dt_ns"]) * 1e-9,
        "t_start_us": float(st.session_state["t_start_us"]),
        "t_end_us": float(st.session_state["t_end_window_us"]),
        "threshold_A": float(st.session_state["threshold_A"]),
        "noise_seed": None if st.session_state["noise_seed"] == "" else int(st.session_state["noise_seed"]),
        "start_branch": st.session_state["start_branch"],
        "lattice_shape": lattice,
        "resist_params": dataclasses.asdict(resist),
        "circuit_params": dataclasses.asdict(circuit),
    }


def _parse_optional_float(text: str) -> Optional[float]:
    if text.strip() == "":
        return None
    return float(text)


def _build_job_config_sweep2d(param_x_label: str, param_y_label: str) -> Dict[str, Any]:
    resist, circuit, lattice = _build_params()
    return {
        "type": "sweep2d",
        "job_name": st.session_state.get("job_name_sweep2d", "").strip(),
        "param_x": _param_name_from_label(param_x_label),
        "param_y": _param_name_from_label(param_y_label),
        "x_start": _parse_optional_float(st.session_state["x_start"]),
        "x_stop": _parse_optional_float(st.session_state["x_stop"]),
        "x_step": float(st.session_state["x_step"]),
        "y_start": _parse_optional_float(st.session_state["y_start"]),
        "y_stop": _parse_optional_float(st.session_state["y_stop"]),
        "y_step": float(st.session_state["y_step"]),
        "vin": float(st.session_state["vin"]),
        "t_end": float(st.session_state["t_end_us"]) * 1e-6,
        "dt": float(st.session_state["dt_ns"]) * 1e-9,
        "t_start_us": float(st.session_state["t_start_us"]),
        "t_end_us": float(st.session_state["t_end_window_us"]),
        "threshold_A": float(st.session_state["threshold_A"]),
        "noise_seed": None if st.session_state["noise_seed"] == "" else int(st.session_state["noise_seed"]),
        "start_branch": st.session_state["start_branch"],
        "lattice_shape": lattice,
        "resist_params": dataclasses.asdict(resist),
        "circuit_params": dataclasses.asdict(circuit),
        "row_early_stop": True,
        "col_early_stop": True,
    }


def _validate_job_config(config: Dict[str, Any]) -> List[str]:
    errors = []
    if config["dt"] <= 0:
        errors.append("Time step (dt) must be > 0.")
    if config["t_end"] <= 0:
        errors.append("Simulation duration (t_end) must be > 0.")
    if config["t_end"] <= config["dt"]:
        errors.append("Simulation duration (t_end) must be larger than dt.")
    if config["t_end_us"] <= config["t_start_us"]:
        errors.append("Steady-state window end must be greater than start.")
    circuit = config["circuit_params"]
    resist = config["resist_params"]
    if circuit["R_series_kohm"] <= 0:
        errors.append("R_series must be > 0.")
    if circuit["C_par_pF"] <= 0:
        errors.append("C_par must be > 0.")
    if circuit["Cth_mW_ns_per_K"] <= 0:
        errors.append("C_th must be > 0.")
    if circuit["Cth_factor"] <= 0:
        errors.append("C_th factor must be > 0.")
    if circuit["T_base_K"] <= 0:
        errors.append("Base temperature must be > 0 K.")
    if resist["R0"] <= 0:
        errors.append("R0 must be > 0.")
    if resist["Rm0"] <= 0:
        errors.append("Rm0 must be > 0.")
    if resist["w"] <= 0:
        errors.append("Hysteresis width (w) must be > 0.")
    if resist["beta"] == 0:
        errors.append("Hysteresis beta must be non-zero.")
    if resist["Tc_K"] <= 0:
        errors.append("Critical temperature must be > 0 K.")
    if resist["gamma"] == 0:
        errors.append("Hysteresis gamma must be non-zero.")
    if resist["width_factor"] <= 0:
        errors.append("Width factor must be > 0.")
    if resist["T_min_K"] <= 0 or resist["T_max_K"] <= 0 or resist["T_max_K"] <= resist["T_min_K"]:
        errors.append("Temperature clamp range must be positive and T_max > T_min.")
    if config["type"] == "sweep1d":
        if config["coarse_step"] <= 0:
            errors.append("Coarse step must be > 0.")
        if config["fine_step"] <= 0:
            errors.append("Fine step must be > 0.")
        if config["start"] >= config["stop"]:
            errors.append("Sweep start must be less than stop.")
    if config["type"] == "sweep2d":
        if config["x_step"] <= 0 or config["y_step"] <= 0:
            errors.append("Sweep steps must be > 0.")
    return errors


def _process_form_actions(
    submitted_add: bool,
    submitted_run: bool,
    job_config: Dict[str, Any],
    terminal_placeholder,
) -> None:
    if submitted_add:
        errors = _validate_job_config(job_config)
        if errors:
            st.error("Please fix these inputs before adding to batch:")
            for msg in errors:
                st.write(f"- {msg}")
            return
        st.session_state["batch_jobs"].append(job_config)
        _update_terminal(f"[batch] added {job_config['type']} job", terminal_placeholder)
    if submitted_run:
        errors = _validate_job_config(job_config)
        if errors:
            st.error("Please fix these inputs before running:")
            for msg in errors:
                st.write(f"- {msg}")
            return
        st.session_state["terminal_log"] = ""
        terminal_placeholder.code("")
        job = _create_job(job_config)
        _enqueue_job(job["id"])
        _update_terminal(f"[job] queued {job['type']} ({job['id']})", terminal_placeholder)


def _render_batch_runner(terminal_placeholder) -> None:
    if not st.session_state["batch_jobs"]:
        return
    col1, col2 = st.columns([1, 3])
    if col1.button("Run batch"):
        st.session_state["terminal_log"] = ""
        terminal_placeholder.code("")
        for idx, cfg in enumerate(st.session_state["batch_jobs"], start=1):
            job = _create_job(cfg)
            _enqueue_job(job["id"])
            _update_terminal(f"[batch] queued {idx}/{len(st.session_state['batch_jobs'])}: {cfg['type']}", terminal_placeholder)
        st.session_state["batch_jobs"] = []
        _update_terminal("[batch] queued", terminal_placeholder)
    col2.markdown(f"**Batch size:** {len(st.session_state['batch_jobs'])}")


def _render_single() -> None:
    st.header("Single Simulation")
    with st.form("single_form"):
        _text_input("Simulation name (optional)", key="job_name_single")
        st.markdown("### Inputs")
        cols = st.columns(4)
        with cols[0]:
            _num_input(_label("Vin"), key="vin")
        with cols[1]:
            _text_input("Vin list (comma-separated)", key="vin_list")
        with cols[2]:
            st.selectbox(_label("start_branch"), ["insulator", "metal"], key="start_branch")
        with cols[3]:
            _text_input("Noise seed (optional)", key="noise_seed")

        _inputs_common()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            submitted_add = st.form_submit_button("Add to batch")
        with btn_col2:
            submitted_run = st.form_submit_button("Run now")

    terminal_placeholder = st.empty()
    terminal_placeholder.code(st.session_state.get("terminal_log", ""))

    if submitted_add or submitted_run:
        job_config = _build_job_config_single()
        _process_form_actions(submitted_add, submitted_run, job_config, terminal_placeholder)

    _render_batch_runner(terminal_placeholder)


def _render_sweep1d() -> None:
    st.header("Sweep Over Free Variable")
    with st.form("sweep1d_form"):
        _text_input("Simulation name (optional)", key="job_name_sweep1d")
        st.markdown("### Inputs")
        cols = st.columns(4)
        with cols[0]:
            param_label = st.selectbox("Free variable", _param_label_options())
        with cols[1]:
            _num_input("Start", key="sweep_start")
        with cols[2]:
            _num_input("Stop", key="sweep_stop")
        with cols[3]:
            _num_input("Coarse step", key="coarse_step")

        cols = st.columns(4)
        with cols[0]:
            _num_input("Fine step", key="fine_step")
        with cols[1]:
            _num_input(_label("Vin"), key="vin")
        with cols[2]:
            st.selectbox(_label("start_branch"), ["insulator", "metal"], key="start_branch")
        with cols[3]:
            _text_input("Noise seed (optional)", key="noise_seed")

        _inputs_common()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            submitted_add = st.form_submit_button("Add to batch")
        with btn_col2:
            submitted_run = st.form_submit_button("Run now")

    terminal_placeholder = st.empty()
    terminal_placeholder.code(st.session_state.get("terminal_log", ""))

    if submitted_add or submitted_run:
        job_config = _build_job_config_sweep1d(param_label)
        _process_form_actions(submitted_add, submitted_run, job_config, terminal_placeholder)

    _render_batch_runner(terminal_placeholder)


def _render_sweep2d() -> None:
    st.header("2D Free Variable vs Oscillation Frequency")
    with st.form("sweep2d_form"):
        _text_input("Simulation name (optional)", key="job_name_sweep2d")
        st.markdown("### Inputs")
        cols = st.columns(4)
        with cols[0]:
            param_x_label = st.selectbox("X variable", _param_label_options(), key="param_x_label")
        with cols[1]:
            _text_input("X start (optional)", key="x_start")
        with cols[2]:
            _text_input("X stop (optional)", key="x_stop")
        with cols[3]:
            _num_input("X step", key="x_step")

        cols = st.columns(4)
        with cols[0]:
            param_y_label = st.selectbox("Y variable", _param_label_options(), key="param_y_label")
        with cols[1]:
            _text_input("Y start (optional)", key="y_start")
        with cols[2]:
            _text_input("Y stop (optional)", key="y_stop")
        with cols[3]:
            _num_input("Y step", key="y_step")

        cols = st.columns(4)
        with cols[0]:
            _num_input(_label("Vin"), key="vin")
        with cols[1]:
            st.selectbox(_label("start_branch"), ["insulator", "metal"], key="start_branch")
        with cols[2]:
            _text_input("Noise seed (optional)", key="noise_seed")

        _inputs_common()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            submitted_add = st.form_submit_button("Add to batch")
        with btn_col2:
            submitted_run = st.form_submit_button("Run now")

    terminal_placeholder = st.empty()
    terminal_placeholder.code(st.session_state.get("terminal_log", ""))

    if submitted_add or submitted_run:
        job_config = _build_job_config_sweep2d(param_x_label, param_y_label)
        _process_form_actions(submitted_add, submitted_run, job_config, terminal_placeholder)

    _render_batch_runner(terminal_placeholder)


def _render_jobs_view() -> None:
    st.header("Jobs")
    loading = st.progress(0.0)
    use_events = False
    jobs = _list_jobs()
    if not jobs:
        loading.empty()
        st.info("No jobs found.")
        return

    total = len(jobs)
    for idx, job in enumerate(jobs, start=1):
        loading.progress(idx / total)
        job_name = job.get("name", "").strip()
        summary_cols = st.columns([4, 2, 2, 1, 1])
        name_key = f"name_{job['id']}"
        with summary_cols[0]:
            new_name = st.text_input(
                "Job name",
                value=job_name,
                key=name_key,
                placeholder=job["id"],
                label_visibility="collapsed",
            )
            if new_name != job_name:
                job["name"] = new_name.strip()
                job_name = job["name"]
                _save_job(job)
        summary_cols[1].write(job["type"])
        summary_cols[2].write(job["status"])
        with summary_cols[3]:
            open_job = st.toggle("Open", key=f"open_{job['id']}")
        with summary_cols[4]:
            if job.get("status") not in {"queued", "running", "cancel_requested"}:
                if st.button("🗑", key=f"delete_{job['id']}", help="Delete job"):
                    shutil.rmtree(_job_dir(job["id"]), ignore_errors=True)
                    _rerun()
        st.caption(f"Created: {job.get('created_at', '')} | Outputs: {len(job.get('outputs', []))}")
        if not open_job:
            st.markdown("---")
            continue

        if job.get("status") in {"queued", "running", "cancel_requested"}:
            if st.button("Cancel job", key=f"cancel_{job['id']}"):
                job["status"] = "cancel_requested" if job.get("status") == "running" else "cancelled"
                _append_job_log(job, "[job] cancel requested")
                _save_job(job)
                _rerun()

        job_removals = _get_job_removals(job["id"])
        st.json(job["params"], expanded=False)
        if os.path.exists(job["log_path"]):
            with open(job["log_path"], "r") as f:
                log_text = f.read().strip()
            expanded = job.get("status") in {"queued", "running", "cancel_requested"}
            with st.expander("Job log", expanded=expanded):
                st.code(log_text or "(no logs)")

        for out in job.get("outputs", []):
            path = out["path"]
            if os.path.exists(path):
                with open(path, "rb") as f:
                    st.download_button(
                        out["label"],
                        data=f,
                        file_name=os.path.basename(path),
                        key=f"dl_{job['id']}_{os.path.basename(path)}",
                    )

        # Preview plots
        if job["type"] == "single":
            csvs = _csv_outputs(job.get("outputs", []))
            if csvs:
                tabs = st.tabs([o["label"] for o in csvs])
                for tab, selected in zip(tabs, csvs):
                    with tab:
                        df = pd.read_csv(selected["path"])
                        df["time_us"] = df["time_s"] * 1e6
                        df["V_vo2"] = df["V_node"]
                        df["I_vo2"] = df["I_vo2"]
                        df["T_K"] = df["T_K"]
                        df["P_vo2"] = df["V_node"] * df["I_vo2"]
                        df["V_load"] = df["I_load"] * job["params"]["circuit_params"]["R_series_kohm"] * 1e3
                        trace_key = f"time_traces:{os.path.basename(selected['path'])}"
                        trace_removals = job_removals["time_traces"].setdefault(trace_key, {})
                        df_filtered = _apply_time_trace_removals(df, trace_removals)
                        remove_key = f"remove_time_{job['id']}_{os.path.basename(selected['path'])}"
                        remove_points = bool(st.session_state.get(remove_key, False))
                        fig = _plot_time_traces(df_filtered, "Time traces")
                        points = _show_plotly_with_click(
                            fig,
                            "Time traces",
                            key=f"plot_{job['id']}_{os.path.basename(selected['path'])}",
                            use_events=(use_events or remove_points),
                            show_click=not remove_points,
                        )
                        point = _consume_click(
                            f"click_time_{job['id']}_{os.path.basename(selected['path'])}",
                            points,
                        )
                        if remove_points and point:
                            p0 = point
                            idx = p0.get("pointIndex")
                            trace_idx = p0.get("curveNumber")
                            if idx is not None and trace_idx is not None:
                                trace_set = trace_removals.setdefault(int(trace_idx), set())
                                _toggle_remove_index(trace_set, int(idx))
                                st.session_state[remove_key] = False
                                _rerun()
                        _toggle_input(
                            "Remove points (click to remove/restore)",
                            key=remove_key,
                            help="Use Plotly click events to remove points for this session.",
                        )
                        mat_fig = _matplotlib_time_traces_figure(df_filtered, "Time traces")
                        png = _fig_to_png_bytes(mat_fig)
                        plt.close(mat_fig)
                        st.download_button(
                            "Download PNG (matplotlib)",
                            data=png,
                            file_name=f"{job['id']}_time_traces.png",
                            key=f"png_{job['id']}_{os.path.basename(selected['path'])}",
                        )

        if job["type"] == "sweep1d":
            csvs = _csv_outputs(job.get("outputs", []))
            if csvs:
                df = pd.read_csv(csvs[0]["path"])
                df.rename(columns={"value": "value"}, inplace=True)
                removed_idx = job_removals["sweep1d"]
                df_filtered = _apply_sweep1d_removals(df, removed_idx)
                figs = _plot_sweep_metrics(df_filtered, job["params"]["param"])
                mat_figs = _matplotlib_sweep1d_figs(df_filtered, job["params"]["param"])
                for idx, fig in enumerate(figs):
                    remove_key = f"remove_sweep1d_{job['id']}_{idx}"
                    run_key = f"run_sweep1d_{job['id']}_{idx}"
                    remove_points = bool(st.session_state.get(remove_key, False))
                    run_points = bool(st.session_state.get(run_key, False))
                    if run_points and remove_points:
                        remove_points = False
                        st.session_state[remove_key] = False
                    if run_points and not _HAS_PLOTLY_EVENTS:
                        st.error("Click-to-run requires `streamlit-plotly-events` to be installed.")
                    points = _show_plotly_with_click(
                        fig,
                        "Sweep1D",
                        key=f"plot_{job['id']}_sweep1d_{idx}",
                        use_events=(use_events or remove_points or run_points),
                        show_click=not (remove_points or run_points),
                    )
                    point = _consume_click(f"click_sweep1d_{job['id']}_{idx}", points)
                    if remove_points and point:
                        p0 = point
                        p_idx = p0.get("pointIndex")
                        if p_idx is not None:
                            _toggle_remove_index(removed_idx, int(p_idx))
                            st.session_state[remove_key] = False
                            _rerun()
                    elif run_points and point:
                        p0 = point
                        try:
                            x_val = _sweep_value_from_point(p0, df["value"])
                            if x_val is None:
                                raise ValueError(f"Missing x-axis value in click payload: {p0}")
                            param = job["params"]["param"]
                            overrides = {param: float(x_val)}
                            new_job_id = _enqueue_single_from_click(job, overrides, f"{param}={float(x_val):.4g}")
                            st.session_state[f"open_{new_job_id}"] = True
                            st.success("Queued single simulation from clicked point.")
                            st.session_state[run_key] = False
                            _rerun()
                        except Exception as exc:
                            _append_job_log(job, f"[click] error: {exc}")
                            st.error(f"Click-to-run failed: {exc}")
                            st.session_state[run_key] = False
                    elif run_points and not point:
                        st.warning("Click a data point to run a single simulation.")
                    _toggle_input(
                        "Remove points (click to remove/restore)",
                        key=remove_key,
                        help="Use Plotly click events to remove points for this session.",
                    )
                    _toggle_input(
                        "Run single simulation from click",
                        key=run_key,
                        value=run_points,
                        help="Click a point to queue a single simulation from this selection.",
                    )
                    tag, mat_fig = mat_figs[idx]
                    png = _fig_to_png_bytes(mat_fig)
                    plt.close(mat_fig)
                    st.download_button(
                        f"Download PNG ({tag})",
                        data=png,
                        file_name=f"{job['id']}_sweep1d_{tag}.png",
                        key=f"png_{job['id']}_sweep1d_{tag}",
                    )

        if job["type"] == "sweep2d":
            csvs = _csv_outputs(job.get("outputs", []))
            if csvs:
                if st.button("Open in Matplotlib (local)", key=f"mpl_{job['id']}"):
                    try:
                        _launch_matplotlib_viewer(
                            csvs[0]["path"],
                            job["params"]["param_x"],
                            job["params"]["param_y"],
                            job.get("name", "") or job["id"],
                        )
                        st.success("Opened matplotlib window.")
                    except Exception as exc:
                        st.error(f"Failed to launch matplotlib: {exc}")
                df = pd.read_csv(csvs[0]["path"])
                df.rename(columns={job["params"]["param_x"]: "x", job["params"]["param_y"]: "y"}, inplace=True)
                removed_xy = job_removals["sweep2d"]
                df_filtered = _apply_sweep2d_removals(df, removed_xy, "x", "y")
                heatmap, scatter3d = _plot_frequency_2d(
                    df_filtered,
                    job["params"]["param_x"],
                    job["params"]["param_y"],
                    removed_xy=removed_xy,
                )
                remove_key_hm = f"remove_sweep2d_heatmap_{job['id']}"
                run_key_hm = f"run_sweep2d_heatmap_{job['id']}"
                remove_points_hm = bool(st.session_state.get(remove_key_hm, False))
                run_points_hm = bool(st.session_state.get(run_key_hm, False))
                if run_points_hm and remove_points_hm:
                    remove_points_hm = False
                    st.session_state[remove_key_hm] = False
                points = _show_plotly_with_click(
                    heatmap,
                    "Heatmap",
                    key=f"plot_{job['id']}_sweep2d_heatmap",
                    use_events=(use_events or remove_points_hm or run_points_hm),
                    show_click=not (remove_points_hm or run_points_hm),
                )
                point = _consume_click(f"click_sweep2d_heatmap_{job['id']}", points)
                if remove_points_hm and point:
                    p0 = point
                    px = p0.get("x")
                    py = p0.get("y")
                    if px is not None and py is not None:
                        _toggle_remove_xy(removed_xy, float(px), float(py))
                        st.session_state[remove_key_hm] = False
                        _rerun()
                elif run_points_hm and point:
                    p0 = point
                    px = p0.get("x")
                    py = p0.get("y")
                    if px is not None and py is not None:
                        overrides = {job["params"]["param_x"]: float(px), job["params"]["param_y"]: float(py)}
                        label = f"{job['params']['param_x']}={float(px):.4g}, {job['params']['param_y']}={float(py):.4g}"
                        new_job_id = _enqueue_single_from_click(job, overrides, label)
                        st.session_state[f"open_{new_job_id}"] = True
                        st.success("Queued single simulation from clicked point.")
                        st.session_state[run_key_hm] = False
                        _rerun()
                _toggle_input(
                    "Remove points (click to remove/restore)",
                    key=remove_key_hm,
                    help="Use Plotly click events to remove points for this session.",
                )
                _toggle_input(
                    "Run single simulation from click",
                    key=run_key_hm,
                    value=run_points_hm,
                    help="Click a point to queue a single simulation from this selection.",
                )
                remove_key_3d = f"remove_sweep2d_3d_{job['id']}"
                run_key_3d = f"run_sweep2d_3d_{job['id']}"
                remove_points_3d = bool(st.session_state.get(remove_key_3d, False))
                run_points_3d = bool(st.session_state.get(run_key_3d, False))
                if run_points_3d and remove_points_3d:
                    remove_points_3d = False
                    st.session_state[remove_key_3d] = False
                points = _show_plotly_with_click(
                    scatter3d,
                    "3D",
                    key=f"plot_{job['id']}_sweep2d_3d",
                    use_events=(use_events or remove_points_3d or run_points_3d),
                    show_click=not (remove_points_3d or run_points_3d),
                )
                point = _consume_click(f"click_sweep2d_3d_{job['id']}", points)
                if remove_points_3d and point:
                    p0 = point
                    px = p0.get("x")
                    py = p0.get("y")
                    if px is not None and py is not None:
                        _toggle_remove_xy(removed_xy, float(px), float(py))
                        st.session_state[remove_key_3d] = False
                        _rerun()
                elif run_points_3d and point:
                    p0 = point
                    px = p0.get("x")
                    py = p0.get("y")
                    if px is not None and py is not None:
                        overrides = {job["params"]["param_x"]: float(px), job["params"]["param_y"]: float(py)}
                        label = f"{job['params']['param_x']}={float(px):.4g}, {job['params']['param_y']}={float(py):.4g}"
                        new_job_id = _enqueue_single_from_click(job, overrides, label)
                        st.session_state[f"open_{new_job_id}"] = True
                        st.success("Queued single simulation from clicked point.")
                        st.session_state[run_key_3d] = False
                        _rerun()
                _toggle_input(
                    "Remove points (click to remove/restore)",
                    key=remove_key_3d,
                    help="Use Plotly click events to remove points for this session.",
                )
                _toggle_input(
                    "Run single simulation from click",
                    key=run_key_3d,
                    value=run_points_3d,
                    help="Click a point to queue a single simulation from this selection.",
                )
                mat_figs = _matplotlib_sweep2d_figs(df_filtered, job["params"]["param_x"], job["params"]["param_y"])
                for tag, mat_fig in mat_figs:
                    png = _fig_to_png_bytes(mat_fig)
                    plt.close(mat_fig)
                    st.download_button(
                        f"Download PNG ({tag})",
                        data=png,
                        file_name=f"{job['id']}_sweep2d_{tag}.png",
                        key=f"png_{job['id']}_sweep2d_{tag}",
                    )
        st.markdown("---")

    loading.empty()
    running = any(job.get("status") in {"queued", "running"} for job in jobs)
    if running:
        st.caption("Auto-refreshing while jobs are running…")
        time.sleep(1.5)
        _rerun()


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    st.set_page_config(page_title="Quantum Materials for Neuromorphic Computation", layout="wide")
    _init_defaults()
    _ensure_worker()

    _render_top_bar()

    _render_sidebar()

    with st.sidebar:
        st.markdown("---")
        if st.button("Load paper preset"):
            _apply_preset(True)

    current_mode = st.session_state["mode"]
    last_mode = st.session_state.get("last_mode")
    if last_mode is None:
        st.session_state["last_mode"] = current_mode
    elif last_mode != current_mode:
        _apply_preset(True)
        st.session_state["last_mode"] = current_mode

    content = st.empty()
    mode = st.session_state["mode"]
    with content.container():
        if mode == "Single Simulation":
            _render_single()
        elif mode == "Sweep over Free Variable":
            _render_sweep1d()
        elif mode == "2D Frequency Sweep":
            _render_sweep2d()
        else:
            _render_jobs_view()


if __name__ == "__main__":
    main()
