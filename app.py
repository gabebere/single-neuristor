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

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

import neuristor.plots as plots
from neuristor.model import (
    HysteresisArray,
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


JOB_ROOT = ROOT / "jobs"
JOB_ROOT.mkdir(exist_ok=True)
SPECIMEN_RESIST_PRESET_PATH = ROOT / "presets" / "resistance_100425_chip1_gap3.json"

MPL_FIGSIZE_WIDE = (16, 9)
MPL_DPI = 320
MPL_LINEWIDTH = 1.6
MPL_TITLE_SIZE = 16
MPL_LABEL_SIZE = 12
MPL_TICK_SIZE = 11
_CURRENT_RF_REF_OHM = 50.0
_CURRENT_AVG_WINDOW_NS = (100.0, 250.0)
_CURRENT_FFT_RANGE_MHZ = (1.0, 1000.0)
_CURRENT_DOMAIN_SCAN_PARAM_MAP: Dict[str, Tuple[str, float]] = {
    "cd_C_pF": ("C_F", 1e-12),
    "cd_Cth_mW_ns_per_K": ("C_th_J_per_K", 1e-12),
    "cd_S_e_mW_per_K": ("S_e_W_per_K", 1e-3),
    # Legacy key kept for backward compatibility with old jobs; in the
    # ideal-current-source model, this does not affect VO2 dynamics.
    "cd_dt_ns": ("dt_s", 1e-9),
    "cd_T0_K": ("T0_K", 1.0),
    "cd_T_init_K": ("T_init_K", 1.0),
    "cd_sigma": ("sigma_W_sqrt_s", 1.0),
}
_CURRENT_DOMAIN_STANDARD_CRITERIA: Dict[str, float] = {
    "cd_scan_min_turns": 6.0,
    "cd_scan_min_vpp_mV": 20.0,
    "cd_scan_max_vpp_mV": 700.0,
}
_CURRENT_DOMAIN_CRITERIA_VERSION = 1
_SIMULATION_MODES = (
    "Single Simulation",
    "Sweep over Free Variable",
    "2D Frequency Sweep",
    "Current-Driven Sweep",
)
_RESIST_PARAM_KEYS = {f.name for f in dataclasses.fields(YuanhangResistParams)}
_CIRCUIT_PARAM_KEYS = {f.name for f in dataclasses.fields(YuanhangCircuitParams)}
_SAMPLE_DERIVED_RESIST_KEYS = {f.name for f in dataclasses.fields(YuanhangResistParams)}
_SAMPLE_DERIVED_KEYS = set(_SAMPLE_DERIVED_RESIST_KEYS) | {"start_branch", "cd_start_branch"}
_NEUTRAL_INPUT_KEYS = {
    "job_name_single",
    "job_name_sweep1d",
    "job_name_sweep2d",
    "job_name_current_drive",
    "job_name_current_domain",
    "vin_list",
    "noise_seed",
    "cd_seed",
    "cd_pulse_off_ns",
    "x_start",
    "x_stop",
    "y_start",
    "y_stop",
    "cd_scan_param_key",
    "cd_scan_start",
    "cd_scan_stop",
    "cd_scan_step",
    "cd_scan_min_turns",
    "cd_scan_min_vpp_mV",
    "cd_scan_max_vpp_mV",
    "cd_frame_duration_s",
}
_HIGHLIGHT_COLORS = {
    "sample_derived": "#16a34a",
    "assumed": "#dc2626",
    "conflict": "#facc15",
}


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
    "noise_strength": "Thermal Noise Strength [K/ns]",
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
    "cd_i_start_uA": "Current Start [uA]",
    "cd_i_stop_uA": "Current Stop [uA]",
    "cd_i_step_uA": "Current Step [uA]",
    "cd_dt_ns": "Current Sim dt [ns]",
    "cd_t_end_ns": "Current Sim t_end [ns]",
    "cd_t_pre_ns": "Current Sim t_pre [ns]",
    "cd_pulse_on_ns": "Current Sim Pulse On [ns]",
    "cd_pulse_off_ns": "Current Sim Pulse Off [ns]",
    "cd_C_pF": "Current Sim C [pF]",
    "cd_Cth_mW_ns_per_K": "Current Sim C_th [mW*ns/K]",
    "cd_S_e_mW_per_K": "Current Sim S_e [mW/K]",
    "cd_T0_K": "Current Sim T0 [K]",
    "cd_T_init_K": "Current Sim T_init [K]",
    "cd_V_init_mV": "Current Sim V_init [mV]",
    "cd_sigma": "Current Sim sigma [W*sqrt(s)]",
    "cd_start_branch": "Current Sim Initial Branch",
    "cd_frame_duration_s": "GIF Frame Duration [s]",
    "cd_seed": "Current Sim Seed",
    "job_name_current_domain": "Current Domain Scan Name",
    "cd_scan_param_key": "Domain Scan Parameter",
    "cd_scan_start": "Domain Scan Start",
    "cd_scan_stop": "Domain Scan Stop",
    "cd_scan_step": "Domain Scan Step",
    "cd_scan_min_turns": "Domain Min Turns",
    "cd_scan_min_vpp_mV": "Domain Min Vpp [mV]",
    "cd_scan_max_vpp_mV": "Domain Max Vpp [mV]",
}


FIELD_HELP = {
    "job_name_single": "Optional name used to identify this single-simulation job in the Jobs view.",
    "job_name_sweep1d": "Optional name used to identify this 1D sweep job in the Jobs view.",
    "job_name_sweep2d": "Optional name used to identify this 2D sweep job in the Jobs view.",
    "Vin": "Input bias voltage applied to the neuristor circuit (volts).",
    "vin": "Input bias voltage applied to the neuristor circuit (volts).",
    "vin_list": "Optional comma-separated Vin values. If provided, runs one simulation per Vin.",
    "R_series_kohm": "Series/load resistor in kOhm. Higher values limit current and change oscillation conditions.",
    "C_par_pF": "Parasitic/electrical capacitance in pF. Sets the electrical RC timescale.",
    "Cth_mW_ns_per_K": "Thermal capacitance of the device. Larger values slow temperature changes.",
    "Sth_mW_per_K": "Thermal conductance to the environment. Larger values increase cooling.",
    "couple_factor": "Neighbor thermal-coupling factor (used for Nx×Ny lattices).",
    "Cth_factor": "Global scaling factor on the thermal equation (including thermal noise).",
    "noise_strength": "Additive thermal noise amplitude in K/ns (paper-style term in dT/dt).",
    "T_base_K": "Base/environment temperature in Kelvin for cooling dynamics.",
    "R0": "Pre-exponential resistance factor in the activated insulating-resistance term.",
    "Ea_over_k": "Activation-energy over Boltzmann constant (K), controlling insulating resistance temperature dependence.",
    "Rm0": "Base metallic resistance in Ohm.",
    "Rm_factor": "Multiplier applied to Rm0 to obtain effective metallic resistance.",
    "w": "Hysteresis width parameter (K).",
    "Tc_K": "Critical-transition temperature center (K).",
    "beta": "Hysteresis sharpness (1/K).",
    "gamma": "Minor-loop window-shape parameter in the hysteresis model.",
    "width_factor": "Scaling factor applied to hysteresis width w.",
    "T_min_K": "Lower temperature clamp used in resistance/hysteresis evaluation.",
    "T_max_K": "Upper temperature clamp used in resistance/hysteresis evaluation.",
    "reversal_threshold_K": "Minimum |dT| needed to register a hysteresis branch reversal.",
    "t_end_us": "Total simulation duration in microseconds.",
    "dt_ns": "Integration timestep in nanoseconds.",
    "t_start_us": "Start of the analysis window (steady-state) in microseconds.",
    "t_end_window_us": "End of the analysis window (steady-state) in microseconds.",
    "threshold_A": "Current spike threshold (A) for frequency/ISI detection.",
    "start_branch": "Initial hysteresis branch at t=0: insulator (heating branch) or metal (cooling branch).",
    "noise_seed": "Optional random seed. Same seed reproduces the same noise realization.",
    "nx": "Number of devices along lattice X dimension.",
    "ny": "Number of devices along lattice Y dimension.",
    "param_label": "Parameter selected as the free variable for 1D sweep.",
    "sweep_start": "Start value of the coarse sweep range.",
    "sweep_stop": "Stop value of the coarse sweep range (must be > start).",
    "coarse_step": "Step size for coarse sweep used to locate oscillatory band.",
    "fine_step": "Step size for fine sweep inside detected oscillatory band.",
    "param_x_label": "Parameter selected for the X axis in 2D sweep.",
    "param_y_label": "Parameter selected for the Y axis in 2D sweep.",
    "x_start": "Optional X sweep start. Leave blank for automatic start behavior.",
    "x_stop": "Optional X sweep stop. Leave blank for automatic stop behavior.",
    "x_step": "Step size for X parameter sweep.",
    "y_start": "Optional Y sweep start. Leave blank for automatic start behavior.",
    "y_stop": "Optional Y sweep stop. Leave blank for automatic stop behavior.",
    "y_step": "Step size for Y parameter sweep.",
    "job_name_current_drive": "Optional name shown above the current-driven sweep results.",
    "cd_i_start_uA": "Start of current sweep in microamps.",
    "cd_i_stop_uA": "Stop of current sweep in microamps.",
    "cd_i_step_uA": "Sweep increment in microamps.",
    "cd_dt_ns": "Current-driven simulation timestep in ns.",
    "cd_t_end_ns": "Current-driven simulation duration after the step in ns.",
    "cd_t_pre_ns": "Optional pre-step duration in ns (I_in=0 before t=0).",
    "cd_pulse_on_ns": "Time in ns at which the current pulse turns on (typically 0).",
    "cd_pulse_off_ns": "Optional time in ns at which the pulse turns off. Leave blank to keep current on.",
    "cd_C_pF": "Capacitance C used in current-driven electrical dynamics.",
    "cd_Cth_mW_ns_per_K": "Thermal capacitance C_th for current-driven simulation.",
    "cd_S_e_mW_per_K": "Thermal cooling coefficient S_e to ambient.",
    "cd_T0_K": "Ambient/base temperature T0 in Kelvin.",
    "cd_T_init_K": "Initial device temperature at simulation start.",
    "cd_V_init_mV": "Initial device voltage across the VO2||C node in mV. This is not a source-side voltage.",
    "cd_sigma": "Thermal-noise intensity sigma used in Euler-Maruyama term.",
    "cd_start_branch": "Initial hysteresis branch for the 2582_1 model.",
    "cd_frame_duration_s": "Frame display duration in the output GIF.",
    "cd_seed": "Optional base RNG seed for deterministic sweep outputs.",
    "job_name_current_domain": "Optional name shown above the domain-scan results in Jobs.",
    "cd_scan_param_key": "Current-driven model input to scan over a numeric range.",
    "cd_scan_start": "Start value for the selected domain-scan parameter.",
    "cd_scan_stop": "Stop value for the selected domain-scan parameter (must be > start).",
    "cd_scan_step": "Step for the selected domain-scan parameter sweep.",
    "cd_scan_min_turns": "Minimum turning points required to count a trace as oscillatory.",
    "cd_scan_min_vpp_mV": "Minimum peak-to-peak V_vo2 amplitude (mV) to count as oscillatory.",
    "cd_scan_max_vpp_mV": "Maximum peak-to-peak V_vo2 amplitude (mV) allowed for oscillatory classification.",
}


def _label(name: str) -> str:
    return PARAM_LABELS.get(name, name)


def _help(name: str) -> str | None:
    if name.startswith("cd_res_"):
        base = name[len("cd_res_") :]
        if base in FIELD_HELP:
            return f"Current-driven simulation: {FIELD_HELP[base]}"
        if base in PARAM_LABELS:
            return f"Current-driven simulation parameter: {PARAM_LABELS[base]}."
    if name in FIELD_HELP:
        return FIELD_HELP[name]
    if name in PARAM_LABELS:
        return f"Model parameter: {PARAM_LABELS[name]}."
    return None


def _cd_res_key(name: str) -> str:
    return f"cd_res_{name}"


def _mode_profile(mode: str | None = None) -> str:
    m = str(mode or st.session_state.get("mode", "Single Simulation"))
    store = st.session_state.setdefault("preset_profile_by_mode", {})
    return str(store.get(m, "paper"))


def _set_mode_profile(profile: str, mode: str | None = None) -> None:
    m = str(mode or st.session_state.get("mode", "Single Simulation"))
    store = st.session_state.setdefault("preset_profile_by_mode", {})
    store[m] = str(profile)


def _sample_status_for_key(key: str) -> str | None:
    if _mode_profile() != "sample":
        return None
    if key in _NEUTRAL_INPUT_KEYS or key.startswith("job_name_"):
        return None
    base = key[len("cd_res_") :] if key.startswith("cd_res_") else key
    if base in _SAMPLE_DERIVED_KEYS:
        return "sample_derived"

    mode = str(st.session_state.get("mode", ""))
    if mode == "Current-Driven Sweep":
        if key.startswith("cd_") and not key.startswith("cd_scan_"):
            return "assumed"
        return None
    if mode in {"Single Simulation", "Sweep over Free Variable", "2D Frequency Sweep"}:
        if key in {"Vin", "vin", "t_end_us", "dt_ns", "t_start_us", "t_end_window_us", "threshold_A", "nx", "ny"}:
            return "assumed"
        if key in _CIRCUIT_PARAM_KEYS:
            return "assumed"
        if key in _RESIST_PARAM_KEYS:
            return "sample_derived" if key in _SAMPLE_DERIVED_RESIST_KEYS else "assumed"
    return None


def _conflict_entry_for_key(key: str) -> Dict[str, Any] | None:
    if str(st.session_state.get("mode", "")) != "Current-Driven Sweep":
        return None
    conflicts = st.session_state.get("cd_diag_conflicts", {})
    if not isinstance(conflicts, dict):
        return None
    entry = conflicts.get(str(key))
    return entry if isinstance(entry, dict) else None


def _input_visual_state(key: str) -> Dict[str, str] | None:
    sample_status = _sample_status_for_key(key)
    if sample_status is None:
        return None
    if sample_status == "sample_derived":
        return {
            "kind": "sample_derived",
            "color": _HIGHLIGHT_COLORS["sample_derived"],
            "message": "Loaded from specimen fit data.",
        }
    if sample_status == "assumed":
        return {
            "kind": "assumed",
            "color": _HIGHLIGHT_COLORS["assumed"],
            "message": "Assumed/default (not directly extracted from specimen fit).",
        }
    return None


def _merge_help(base_help: str | None, extra_help: str | None) -> str | None:
    b = (base_help or "").strip()
    e = (extra_help or "").strip()
    if b and e:
        return f"{b}\n\n{e}"
    if b:
        return b
    if e:
        return e
    return None


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


def _render_marker(color: str, tooltip: str | None = None) -> None:
    tip = (tooltip or "").replace('"', "&quot;")
    st.markdown(
        (
            f"<div title=\"{tip}\" "
            f"style=\"height:2.35rem;margin-top:0.55rem;border-left:6px solid {color};\"></div>"
        ),
        unsafe_allow_html=True,
    )


def _num_input(label: str, key: str, value: float | None = None, **kwargs):
    # Allow high-precision float entry while keeping display compact (no forced trailing zeros).
    kwargs.setdefault("format", "%.16g")
    kwargs.setdefault("step", 1e-12)
    if "help" not in kwargs:
        h = _help(key)
        if h is not None:
            kwargs["help"] = h
    state = _input_visual_state(key)
    kwargs["help"] = _merge_help(kwargs.get("help"), None if state is None else state.get("message"))

    def _draw() -> float:
        if key in st.session_state:
            return float(st.number_input(label, key=key, **kwargs))
        if value is None:
            return float(st.number_input(label, key=key, **kwargs))
        return float(st.number_input(label, key=key, value=value, **kwargs))

    if state is None:
        return _draw()
    marker_col, input_col = st.columns([0.08, 0.92], gap="small")
    with marker_col:
        _render_marker(state["color"], tooltip=state.get("message"))
    with input_col:
        return _draw()


def _int_input(label: str, key: str, value: int | None = None, **kwargs):
    if "help" not in kwargs:
        h = _help(key)
        if h is not None:
            kwargs["help"] = h
    state = _input_visual_state(key)
    kwargs["help"] = _merge_help(kwargs.get("help"), None if state is None else state.get("message"))

    def _draw() -> int:
        if key in st.session_state:
            return int(st.number_input(label, key=key, step=1, **kwargs))
        if value is None:
            return int(st.number_input(label, key=key, step=1, **kwargs))
        return int(st.number_input(label, key=key, value=value, step=1, **kwargs))

    if state is None:
        return _draw()
    marker_col, input_col = st.columns([0.08, 0.92], gap="small")
    with marker_col:
        _render_marker(state["color"], tooltip=state.get("message"))
    with input_col:
        return _draw()


def _text_input(label: str, key: str, value: str | None = None, **kwargs):
    if "help" not in kwargs:
        h = _help(key)
        if h is not None:
            kwargs["help"] = h
    state = _input_visual_state(key)
    kwargs["help"] = _merge_help(kwargs.get("help"), None if state is None else state.get("message"))

    def _draw() -> str:
        if key in st.session_state:
            return str(st.text_input(label, key=key, **kwargs))
        if value is None:
            return str(st.text_input(label, key=key, **kwargs))
        return str(st.text_input(label, key=key, value=value, **kwargs))

    if state is None:
        return _draw()
    marker_col, input_col = st.columns([0.08, 0.92], gap="small")
    with marker_col:
        _render_marker(state["color"], tooltip=state.get("message"))
    with input_col:
        return _draw()


def _selectbox_input(label: str, options: List[str], key: str, **kwargs):
    if "help" not in kwargs:
        h = _help(key)
        if h is not None:
            kwargs["help"] = h
    state = _input_visual_state(key)
    kwargs["help"] = _merge_help(kwargs.get("help"), None if state is None else state.get("message"))

    def _draw() -> str:
        return str(st.selectbox(label, options, key=key, **kwargs))

    if state is None:
        return _draw()
    marker_col, input_col = st.columns([0.08, 0.92], gap="small")
    with marker_col:
        _render_marker(state["color"], tooltip=state.get("message"))
    with input_col:
        return _draw()


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


def _load_resistance_preset(path: Path) -> tuple[YuanhangResistParams, str, Dict[str, float]]:
    payload = json.loads(path.read_text())
    raw = payload.get("resist_params", payload)
    if not isinstance(raw, dict):
        raise ValueError("Invalid resistance preset format: missing resist_params object.")
    kwargs = {}
    for f in dataclasses.fields(YuanhangResistParams):
        if f.name not in raw:
            raise ValueError(f"Invalid resistance preset format: missing field {f.name}.")
        kwargs[f.name] = float(raw[f.name])
    start_branch = str(payload.get("start_branch", "insulator")).strip().lower()
    if start_branch not in {"insulator", "metal"}:
        start_branch = "insulator"
    metrics_raw = payload.get("fit_metrics", {})
    metrics: Dict[str, float] = {}
    if isinstance(metrics_raw, dict):
        for k, v in metrics_raw.items():
            try:
                metrics[str(k)] = float(v)
            except Exception:
                pass
    return YuanhangResistParams(**kwargs), start_branch, metrics


def _apply_specimen_resistance_preset(path: Path = SPECIMEN_RESIST_PRESET_PATH) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Specimen resistance preset not found: {path}"
    try:
        resist, start_branch, metrics = _load_resistance_preset(path)
    except Exception as exc:
        return False, f"Failed to load specimen resistance preset: {exc}"

    for f in dataclasses.fields(YuanhangResistParams):
        value = getattr(resist, f.name)
        st.session_state[f.name] = value
        st.session_state[_cd_res_key(f.name)] = value
    st.session_state["start_branch"] = start_branch
    st.session_state["cd_start_branch"] = start_branch
    rmse = metrics.get("rmse_log10")
    if rmse is not None:
        return (
            True,
            f"Loaded specimen resistance preset ({path.name}), start_branch={start_branch}, rmse_log10={rmse:.4f}.",
        )
    return True, f"Loaded specimen resistance preset ({path.name}), start_branch={start_branch}."


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


def _current_scan_param_options() -> List[str]:
    return list(_CURRENT_DOMAIN_SCAN_PARAM_MAP.keys())


def _inclusive_range(start: float, stop: float, step: float) -> List[float]:
    start_f = float(start)
    stop_f = float(stop)
    step_f = float(step)
    if step_f <= 0.0:
        raise ValueError("step must be > 0")
    if stop_f < start_f:
        raise ValueError("stop must be >= start")
    n = int(np.floor((stop_f - start_f) / step_f + 1e-12)) + 1
    vals = [start_f + i * step_f for i in range(max(n, 1))]
    if vals and vals[-1] < stop_f - 1e-12:
        vals.append(stop_f)
    return vals


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_optional_ns(text: str) -> Optional[float]:
    s = str(text).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _guided_current_domain_presets() -> Dict[str, Dict[str, float]]:
    """
    Produce compact, physics-guided scan ranges for unknown current-drive parameters.

    Returns mapping:
      key -> {"start","stop","step","n"}
    where key is one of current-domain scan parameter keys.
    """
    presets: Dict[str, Dict[str, float]] = {}
    try:
        resist = YuanhangResistParams(
            **{f.name: _safe_float(st.session_state[_cd_res_key(f.name)]) for f in dataclasses.fields(YuanhangResistParams)}
        )
        i_peak_uA = max(
            abs(_safe_float(st.session_state["cd_i_start_uA"])),
            abs(_safe_float(st.session_state["cd_i_stop_uA"])),
        )
        report = _current_drive_numerics_report(
            dt_s=_safe_float(st.session_state["cd_dt_ns"]) * 1e-9,
            C_F=_safe_float(st.session_state["cd_C_pF"]) * 1e-12,
            C_th_J_per_K=_safe_float(st.session_state["cd_Cth_mW_ns_per_K"]) * 1e-12,
            T_init_K=_safe_float(st.session_state["cd_T_init_K"]),
            I_peak_uA=i_peak_uA,
            resist_params=resist,
            start_branch=str(st.session_state["cd_start_branch"]),
        )
        dt_s = max(_safe_float(st.session_state["cd_dt_ns"]) * 1e-9, 1e-13)
        c_cur_pF = max(_safe_float(st.session_state["cd_C_pF"]), 1e-9)
        r_eff_fast = max(float(report["R_eff_fast_ohm"]), 1e-12)
        c_min_stable_pF = (dt_s / (0.1 * r_eff_fast)) * 1e12
        c_nom_pF = max(c_cur_pF, 1.3 * c_min_stable_pF)
        c_start = max(1e-6, 0.55 * c_nom_pF)
        c_stop = max(c_start * 1.2, 2.0 * c_nom_pF)
        c_step = (c_stop - c_start) / 6.0
        presets["cd_C_pF"] = {"start": c_start, "stop": c_stop, "step": c_step, "n": 7.0}

        cth_cur = max(_safe_float(st.session_state["cd_Cth_mW_ns_per_K"]), 1e-9)
        dT_over_rev = max(float(report["dT_step_over_reversal"]), 1e-6)
        cth_target = max(cth_cur, cth_cur * (dT_over_rev / 0.2))
        cth_start = max(1e-9, 0.5 * cth_target)
        cth_stop = max(cth_start * 1.2, 2.0 * cth_target)
        cth_step = (cth_stop - cth_start) / 5.0
        presets["cd_Cth_mW_ns_per_K"] = {"start": cth_start, "stop": cth_stop, "step": cth_step, "n": 6.0}

        se_cur = max(_safe_float(st.session_state["cd_S_e_mW_per_K"]), 1e-9)
        t_end_ns = max(_safe_float(st.session_state["cd_t_end_ns"]), 1e-6)
        pulse_off_ns = _parse_optional_ns(st.session_state.get("cd_pulse_off_ns", ""))
        if pulse_off_ns is None:
            relax_ns = max(20.0, 0.5 * t_end_ns)
        else:
            relax_ns = max(20.0, t_end_ns - pulse_off_ns)
        tau_target_s = max(relax_ns * 1e-9 / 3.0, 1e-12)
        cth_target_J = cth_target * 1e-12
        se_from_tau_mW = (cth_target_J / tau_target_s) * 1e3
        se_nom = float(np.sqrt(max(se_cur, 1e-12) * max(se_from_tau_mW, 1e-12)))
        se_start = max(1e-6, 0.45 * se_nom)
        se_stop = max(se_start * 1.2, 1.8 * se_nom)
        se_step = (se_stop - se_start) / 5.0
        presets["cd_S_e_mW_per_K"] = {"start": se_start, "stop": se_stop, "step": se_step, "n": 6.0}
    except Exception:
        return {}
    return presets


def _apply_current_domain_preset(param_key: str, preset: Dict[str, float]) -> None:
    st.session_state["cd_scan_param_key"] = str(param_key)
    st.session_state["cd_scan_start"] = float(f"{float(preset['start']):.16g}")
    st.session_state["cd_scan_stop"] = float(f"{float(preset['stop']):.16g}")
    st.session_state["cd_scan_step"] = float(f"{float(preset['step']):.16g}")


def _apply_current_domain_standard_criteria(force: bool = False) -> None:
    for key, default_value in _CURRENT_DOMAIN_STANDARD_CRITERIA.items():
        if force:
            st.session_state[key] = float(default_value)
            continue
        raw = st.session_state.get(key)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float("nan")
        if raw is None or not np.isfinite(value):
            st.session_state[key] = float(default_value)

    min_turns = float(st.session_state["cd_scan_min_turns"])
    min_vpp = float(st.session_state["cd_scan_min_vpp_mV"])
    max_vpp = float(st.session_state["cd_scan_max_vpp_mV"])

    if min_turns < 1.0:
        st.session_state["cd_scan_min_turns"] = _CURRENT_DOMAIN_STANDARD_CRITERIA["cd_scan_min_turns"]
    if min_vpp < 0.0:
        st.session_state["cd_scan_min_vpp_mV"] = _CURRENT_DOMAIN_STANDARD_CRITERIA["cd_scan_min_vpp_mV"]
    if max_vpp <= 0.0:
        st.session_state["cd_scan_max_vpp_mV"] = _CURRENT_DOMAIN_STANDARD_CRITERIA["cd_scan_max_vpp_mV"]
    if float(st.session_state["cd_scan_max_vpp_mV"]) <= float(st.session_state["cd_scan_min_vpp_mV"]):
        st.session_state["cd_scan_min_vpp_mV"] = _CURRENT_DOMAIN_STANDARD_CRITERIA["cd_scan_min_vpp_mV"]
        st.session_state["cd_scan_max_vpp_mV"] = _CURRENT_DOMAIN_STANDARD_CRITERIA["cd_scan_max_vpp_mV"]


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
        "job_name_current_drive",
        "cd_i_start_uA",
        "cd_i_stop_uA",
        "cd_i_step_uA",
        "cd_dt_ns",
        "cd_t_end_ns",
        "cd_t_pre_ns",
        "cd_pulse_on_ns",
        "cd_pulse_off_ns",
        "cd_C_pF",
        "cd_Cth_mW_ns_per_K",
        "cd_S_e_mW_per_K",
        "cd_T0_K",
        "cd_T_init_K",
        "cd_V_init_mV",
        "cd_sigma",
        "cd_start_branch",
        "cd_frame_duration_s",
        "cd_seed",
        "cd_last_result",
        "cd_last_diag",
        "job_name_current_domain",
        "cd_scan_param_key",
        "cd_scan_start",
        "cd_scan_stop",
        "cd_scan_step",
        "cd_scan_min_turns",
        "cd_scan_min_vpp_mV",
        "cd_scan_max_vpp_mV",
        "_cd_scan_criteria_defaults_version",
        "preset_profile_by_mode",
        "cd_diag_conflicts",
    }
    required_keys.update({_cd_res_key(f.name) for f in dataclasses.fields(YuanhangResistParams)})
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
    st.session_state.setdefault("job_name_current_drive", "")
    st.session_state.setdefault("cd_i_start_uA", 50.0)
    st.session_state.setdefault("cd_i_stop_uA", 2000.0)
    st.session_state.setdefault("cd_i_step_uA", 50.0)
    st.session_state.setdefault("cd_dt_ns", 10.0)
    st.session_state.setdefault("cd_t_end_ns", 600.0)
    st.session_state.setdefault("cd_t_pre_ns", 0.0)
    st.session_state.setdefault("cd_pulse_on_ns", 0.0)
    st.session_state.setdefault("cd_pulse_off_ns", "")
    st.session_state.setdefault("cd_C_pF", circuit.C_par_pF)
    st.session_state.setdefault("cd_Cth_mW_ns_per_K", circuit.Cth_mW_ns_per_K)
    st.session_state.setdefault("cd_S_e_mW_per_K", circuit.Sth_mW_per_K)
    st.session_state.setdefault("cd_T0_K", circuit.T_base_K)
    st.session_state.setdefault("cd_T_init_K", circuit.T_base_K - 0.1)
    st.session_state.setdefault("cd_V_init_mV", 0.0)
    st.session_state.setdefault("cd_sigma", 0.0)
    st.session_state.setdefault("cd_start_branch", "insulator")
    st.session_state.setdefault("cd_frame_duration_s", 0.5)
    st.session_state.setdefault("cd_seed", "")
    st.session_state.setdefault("cd_last_result", None)
    st.session_state.setdefault("cd_last_diag", None)
    st.session_state.setdefault("cd_diag_conflicts", {})
    st.session_state.setdefault("preset_profile_by_mode", {m: "paper" for m in _SIMULATION_MODES})
    for m in _SIMULATION_MODES:
        if m not in st.session_state["preset_profile_by_mode"]:
            st.session_state["preset_profile_by_mode"][m] = "paper"
    st.session_state.setdefault("job_name_current_domain", "")
    st.session_state.setdefault("cd_scan_param_key", "cd_C_pF")
    st.session_state.setdefault("cd_scan_start", 20.0)
    st.session_state.setdefault("cd_scan_stop", 300.0)
    st.session_state.setdefault("cd_scan_step", 10.0)
    st.session_state.setdefault("cd_scan_min_turns", _CURRENT_DOMAIN_STANDARD_CRITERIA["cd_scan_min_turns"])
    st.session_state.setdefault("cd_scan_min_vpp_mV", _CURRENT_DOMAIN_STANDARD_CRITERIA["cd_scan_min_vpp_mV"])
    st.session_state.setdefault("cd_scan_max_vpp_mV", _CURRENT_DOMAIN_STANDARD_CRITERIA["cd_scan_max_vpp_mV"])
    if str(st.session_state.get("cd_scan_param_key")) not in _current_scan_param_options():
        st.session_state["cd_scan_param_key"] = "cd_C_pF"
    st.session_state.setdefault("_cd_scan_criteria_defaults_version", 0)
    if int(st.session_state.get("_cd_scan_criteria_defaults_version", 0)) < _CURRENT_DOMAIN_CRITERIA_VERSION:
        needs_migration = False
        try:
            needs_migration = (
                float(st.session_state["cd_scan_min_turns"]) <= 0.0
                or float(st.session_state["cd_scan_min_vpp_mV"]) <= 0.0
                or float(st.session_state["cd_scan_max_vpp_mV"]) <= 0.0
                or float(st.session_state["cd_scan_max_vpp_mV"]) <= float(st.session_state["cd_scan_min_vpp_mV"])
            )
        except Exception:
            needs_migration = True
        _apply_current_domain_standard_criteria(force=needs_migration)
        st.session_state["_cd_scan_criteria_defaults_version"] = _CURRENT_DOMAIN_CRITERIA_VERSION
    else:
        _apply_current_domain_standard_criteria(force=False)
    for f in dataclasses.fields(YuanhangResistParams):
        key = _cd_res_key(f.name)
        st.session_state.setdefault(key, getattr(resist, f.name))
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
    if paper:
        # Paper baseline used in the original code (main.py): noise_strength = 0.001, Cth_factor = 1.0
        circuit.noise_strength = 1e-3
        circuit.Cth_factor = 1.0
    for f in dataclasses.fields(YuanhangResistParams):
        st.session_state[f.name] = getattr(resist, f.name)
        st.session_state[_cd_res_key(f.name)] = getattr(resist, f.name)
    for f in dataclasses.fields(YuanhangCircuitParams):
        st.session_state[f.name] = getattr(circuit, f.name)
    # Paper runs do not require a fixed RNG seed; clear any previous manual seed.
    st.session_state["noise_seed"] = ""
    st.session_state["t_end_us"] = 300.0
    st.session_state["dt_ns"] = 10.0
    st.session_state["t_start_us"] = 25.0
    st.session_state["t_end_window_us"] = 300.0
    st.session_state["threshold_A"] = 1e-3
    st.session_state["cd_dt_ns"] = 10.0
    st.session_state["cd_t_end_ns"] = 600.0
    st.session_state["cd_t_pre_ns"] = 0.0
    st.session_state["cd_pulse_on_ns"] = 0.0
    st.session_state["cd_pulse_off_ns"] = ""
    st.session_state["cd_C_pF"] = circuit.C_par_pF
    st.session_state["cd_Cth_mW_ns_per_K"] = circuit.Cth_mW_ns_per_K
    st.session_state["cd_S_e_mW_per_K"] = circuit.Sth_mW_per_K
    st.session_state["cd_T0_K"] = circuit.T_base_K
    st.session_state["cd_T_init_K"] = circuit.T_base_K - 0.1
    st.session_state["cd_V_init_mV"] = 0.0
    st.session_state["cd_sigma"] = 0.0
    st.session_state["cd_start_branch"] = "insulator"
    st.session_state["cd_i_start_uA"] = 50.0
    st.session_state["cd_i_stop_uA"] = 2000.0
    st.session_state["cd_i_step_uA"] = 50.0
    st.session_state["cd_frame_duration_s"] = 0.5
    st.session_state["cd_seed"] = ""
    st.session_state["cd_last_result"] = None


def _apply_current_drive_reference_preset() -> None:
    from neuristor.current_drive_sim import reference_visual_pulse_params

    p = reference_visual_pulse_params()
    st.session_state["job_name_current_drive"] = "Reference Pulse Preset (Visual)"
    st.session_state["cd_i_start_uA"] = 50.0
    st.session_state["cd_i_stop_uA"] = 800.0
    st.session_state["cd_i_step_uA"] = 50.0
    st.session_state["cd_dt_ns"] = p.dt_s * 1e9
    st.session_state["cd_t_end_ns"] = p.t_end_s * 1e9
    st.session_state["cd_t_pre_ns"] = p.t_pre_s * 1e9
    st.session_state["cd_pulse_on_ns"] = p.pulse_on_s * 1e9
    st.session_state["cd_pulse_off_ns"] = "" if p.pulse_off_s is None else f"{p.pulse_off_s * 1e9:.16g}"
    st.session_state["cd_C_pF"] = p.C_F * 1e12
    st.session_state["cd_Cth_mW_ns_per_K"] = p.C_th_J_per_K * 1e12
    st.session_state["cd_S_e_mW_per_K"] = p.S_e_W_per_K * 1e3
    st.session_state["cd_T0_K"] = p.T0_K
    st.session_state["cd_T_init_K"] = p.T_init_K
    st.session_state["cd_V_init_mV"] = p.V_init_V * 1e3
    st.session_state["cd_sigma"] = p.sigma_W_sqrt_s
    st.session_state["cd_start_branch"] = p.start_branch
    st.session_state["cd_frame_duration_s"] = 0.5
    st.session_state["cd_seed"] = "1"
    for f in dataclasses.fields(YuanhangResistParams):
        st.session_state[_cd_res_key(f.name)] = getattr(p.resist_params, f.name)
    st.session_state["cd_last_result"] = None


def _apply_current_drive_paper_preset() -> None:
    # Keep this scoped to current-drive controls so voltage-driven workflows are unaffected.
    resist, circuit = _paper_params()
    st.session_state["job_name_current_drive"] = "Paper Current Preset"
    st.session_state["cd_i_start_uA"] = 50.0
    st.session_state["cd_i_stop_uA"] = 2000.0
    st.session_state["cd_i_step_uA"] = 50.0
    st.session_state["cd_dt_ns"] = 10.0
    st.session_state["cd_t_end_ns"] = 600.0
    st.session_state["cd_t_pre_ns"] = 0.0
    st.session_state["cd_pulse_on_ns"] = 0.0
    st.session_state["cd_pulse_off_ns"] = ""
    st.session_state["cd_C_pF"] = circuit.C_par_pF
    st.session_state["cd_Cth_mW_ns_per_K"] = circuit.Cth_mW_ns_per_K
    st.session_state["cd_S_e_mW_per_K"] = circuit.Sth_mW_per_K
    st.session_state["cd_T0_K"] = 298.0
    st.session_state["cd_T_init_K"] = 297.9
    st.session_state["cd_V_init_mV"] = 0.0
    # From collective-dynamics table: sigma = 1 microJ * s^(-1/2).
    st.session_state["cd_sigma"] = 1.0e-6
    st.session_state["cd_start_branch"] = "insulator"
    st.session_state["cd_frame_duration_s"] = 0.5
    st.session_state["cd_seed"] = ""
    for f in dataclasses.fields(YuanhangResistParams):
        st.session_state[_cd_res_key(f.name)] = getattr(resist, f.name)
    st.session_state["cd_last_result"] = None


def _apply_current_drive_professor_preset() -> tuple[bool, str]:
    """
    Apply the sample-oriented current-mode baseline in one step:
    - current/thermal settings from the paper-current preset
    - specimen RT-fitted resistance/hysteresis preset
    """

    _apply_current_drive_paper_preset()
    ok, msg = _apply_specimen_resistance_preset()
    st.session_state["job_name_current_drive"] = "Sample Current + Specimen RT Preset"
    if not ok:
        return False, f"Loaded paper current preset, but specimen RT preset failed: {msg}"

    # Auto-stabilize dt for sample preset so diagnostics do not start in a numerically invalid regime.
    # This keeps the preset immediately runnable while preserving all other parameters.
    dt_before_ns = float(st.session_state["cd_dt_ns"])
    dt_after_ns = dt_before_ns
    try:
        resist = YuanhangResistParams(
            **{f.name: float(st.session_state[_cd_res_key(f.name)]) for f in dataclasses.fields(YuanhangResistParams)}
        )
        i_peak_uA = max(
            abs(float(st.session_state["cd_i_start_uA"])),
            abs(float(st.session_state["cd_i_stop_uA"])),
        )
        report = _current_drive_numerics_report(
            dt_s=dt_before_ns * 1e-9,
            C_F=float(st.session_state["cd_C_pF"]) * 1e-12,
            C_th_J_per_K=float(st.session_state["cd_Cth_mW_ns_per_K"]) * 1e-12,
            T_init_K=float(st.session_state["cd_T_init_K"]),
            I_peak_uA=i_peak_uA,
            resist_params=resist,
            start_branch=str(st.session_state["cd_start_branch"]),
        )

        min_dt_ns = 1e-4
        targets = [dt_before_ns]

        dt_tau_fast = float(report["dt_over_tau_fast"])
        if dt_tau_fast > 0.1:
            tau_fast_ns = float(report["tau_fast_s"]) * 1e9
            targets.append(max(min_dt_ns, 0.05 * tau_fast_ns))

        dT_over_rev = float(report["dT_step_over_reversal"])
        if dT_over_rev > 0.2:
            targets.append(max(min_dt_ns, dt_before_ns * (0.2 / dT_over_rev)))

        dt_after_ns = min(targets)
        if dt_after_ns < dt_before_ns * 0.98:
            st.session_state["cd_dt_ns"] = float(f"{dt_after_ns:.16g}")
    except Exception:
        # If diagnostics fail for any reason, keep the loaded preset and avoid hard failure.
        dt_after_ns = dt_before_ns

    if dt_after_ns < dt_before_ns * 0.98:
        return True, (
            "Loaded sample preset: paper current/thermal defaults "
            "(T0=298 K, sigma=1e-6, ideal current source) + specimen RT resistance fit, "
            f"and auto-adjusted dt from {dt_before_ns:.4g} ns to {dt_after_ns:.4g} ns for stability."
        )
    return True, (
        "Loaded sample preset: paper current/thermal defaults "
        "(T0=298 K, sigma=1e-6, ideal current source) + specimen RT resistance fit."
    )


def _apply_mode_scoped_preset(kind: str) -> tuple[bool, str]:
    mode = str(st.session_state.get("mode", "Single Simulation"))
    if mode not in _SIMULATION_MODES:
        return False, "Preset buttons apply to simulation modes only."

    if kind == "paper":
        if mode == "Current-Driven Sweep":
            _apply_current_drive_paper_preset()
        else:
            _apply_preset(True)
        _set_mode_profile("paper", mode=mode)
        return True, f"Loaded paper parameters for {mode}."

    if kind == "sample":
        if mode == "Current-Driven Sweep":
            ok, msg = _apply_current_drive_professor_preset()
            if ok:
                _set_mode_profile("sample", mode=mode)
            return ok, msg
        ok, msg = _apply_specimen_resistance_preset()
        if ok:
            _set_mode_profile("sample", mode=mode)
        return ok, msg

    return False, f"Unknown preset kind: {kind}"


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


def _count_turns(values: np.ndarray) -> int:
    if values.size < 3:
        return 0
    d = np.diff(values)
    return int(np.sum((d[:-1] * d[1:]) < 0.0))


def _current_drive_numerics_report(
    *,
    dt_s: float,
    C_F: float,
    C_th_J_per_K: float,
    T_init_K: float,
    I_peak_uA: float,
    resist_params: YuanhangResistParams,
    start_branch: str,
) -> Dict[str, float]:
    eps = 1e-12
    dt = float(dt_s)
    C = max(float(C_F), eps)
    C_th = max(float(C_th_J_per_K), eps)

    # Use the same hysteresis evaluator as simulation to estimate initial branch resistance.
    h = HysteresisArray(resist_params, size=1, start_branch=start_branch)
    T0 = np.asarray([float(T_init_K)], dtype=float)
    h.initialize(T0)
    R_init = float(h.evaluate(T0)[0][0])

    R_metal = max(float(resist_params.Rm), eps)
    R_eff_init = max(R_init, eps)
    R_eff_fast = max(R_metal, eps)
    tau_init_s = C * R_eff_init
    tau_fast_s = C * R_eff_fast

    I_peak_A = abs(float(I_peak_uA)) * 1e-6
    V_fast_est = I_peak_A * R_eff_fast
    P_fast_est = (V_fast_est * V_fast_est) / max(R_metal, eps)
    dT_step_est_K = (dt / C_th) * P_fast_est
    reversal_thr = max(float(resist_params.reversal_threshold_K), eps)

    return {
        "R_init_ohm": R_init,
        "R_metal_ohm": R_metal,
        "R_eff_init_ohm": R_eff_init,
        "R_eff_fast_ohm": R_eff_fast,
        "tau_init_s": tau_init_s,
        "tau_fast_s": tau_fast_s,
        "dt_over_tau_init": dt / max(tau_init_s, eps),
        "dt_over_tau_fast": dt / max(tau_fast_s, eps),
        "I_peak_uA": float(I_peak_uA),
        "dT_step_est_K": dT_step_est_K,
        "reversal_threshold_K": reversal_thr,
        "dT_step_over_reversal": dT_step_est_K / reversal_thr,
    }


def _current_drive_report_messages(report: Dict[str, float]) -> List[str]:
    msgs: List[str] = []
    dt_tau_fast = float(report["dt_over_tau_fast"])
    tau_fast_ns = float(report["tau_fast_s"]) * 1e9
    if dt_tau_fast > 0.1:
        target_ns = 0.1 * tau_fast_ns
        msgs.append(
            f"dt/tau_fast={dt_tau_fast:.3g} (>0.1). Fast RC is under-resolved; target dt <= {target_ns:.3g} ns."
        )
    elif dt_tau_fast > 0.03:
        msgs.append(f"dt/tau_fast={dt_tau_fast:.3g}. This is borderline for accurate waveform shape.")

    dt_tau_init = float(report["dt_over_tau_init"])
    if dt_tau_init > 0.2:
        msgs.append(f"dt/tau_init={dt_tau_init:.3g} (>0.2). Euler artifacts are likely.")

    dT_over_rev = float(report["dT_step_over_reversal"])
    if dT_over_rev > 1.0:
        msgs.append(
            f"Conservative per-step thermal jump estimate is {dT_over_rev:.3g}x reversal threshold; "
            "minor-loop reversal points may be skipped."
        )
    elif dT_over_rev > 0.2:
        msgs.append(
            f"Conservative per-step thermal jump estimate is {dT_over_rev:.3g}x reversal threshold; "
            "hysteresis timing may be sensitive to dt."
        )
    return msgs


def _current_drive_recommendations(
    report: Dict[str, float],
    *,
    dt_ns: float,
    c_th_mW_ns_per_K: float,
) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    min_dt_ns = 1e-4
    dt_ns = max(float(dt_ns), min_dt_ns)
    c_th = max(float(c_th_mW_ns_per_K), 1e-12)

    dt_tau_fast = float(report["dt_over_tau_fast"])
    tau_fast_ns = float(report["tau_fast_s"]) * 1e9
    if dt_tau_fast > 0.1:
        target_dt_ns = max(min_dt_ns, 0.05 * tau_fast_ns)
        recs.append(
            {
                "id": "dt_fast",
                "severity": "error",
                "title": "Fast electrical RC is under-resolved",
                "problem": (
                    f"`dt/tau_fast = {dt_tau_fast:.3g}` (> 0.1). "
                    "This can suppress oscillations or produce non-physical waveforms."
                ),
                "change": f"Reduce `{_label('cd_dt_ns')}` to `<= {target_dt_ns:.4g}` ns.",
                "actions": [
                    {
                        "label": f"Set dt = {target_dt_ns:.4g} ns",
                        "updates": {"cd_dt_ns": float(f"{target_dt_ns:.16g}")},
                    }
                ],
                "why": "Smaller dt better resolves the fastest VO2 RC state.",
            }
        )
    elif dt_tau_fast > 0.03:
        target_dt_ns = max(min_dt_ns, 0.03 * tau_fast_ns)
        recs.append(
            {
                "id": "dt_fast_borderline",
                "severity": "warning",
                "title": "Fast electrical RC is borderline",
                "problem": f"`dt/tau_fast = {dt_tau_fast:.3g}`. Waveform shape can drift.",
                "change": f"Try lowering `{_label('cd_dt_ns')}` toward `{target_dt_ns:.4g}` ns.",
                "actions": [
                    {
                        "label": f"Set dt = {target_dt_ns:.4g} ns",
                        "updates": {"cd_dt_ns": float(f"{target_dt_ns:.16g}")},
                    }
                ],
                "why": "This helps recover peak/valley timing and oscillation amplitude.",
            }
        )

    dt_tau_init = float(report["dt_over_tau_init"])
    if dt_tau_init > 0.2 and dt_tau_fast <= 0.03:
        tau_init_ns = float(report["tau_init_s"]) * 1e9
        target_dt_ns = max(min_dt_ns, 0.1 * tau_init_ns)
        recs.append(
            {
                "id": "dt_init",
                "severity": "warning",
                "title": "Initial electrical RC is under-resolved",
                "problem": f"`dt/tau_init = {dt_tau_init:.3g}` (> 0.2). Euler artifacts are likely.",
                "change": f"Reduce `{_label('cd_dt_ns')}` to `<= {target_dt_ns:.4g}` ns.",
                "actions": [
                    {
                        "label": f"Set dt = {target_dt_ns:.4g} ns",
                        "updates": {"cd_dt_ns": float(f"{target_dt_ns:.16g}")},
                    }
                ],
                "why": "The pre-switch charging transient is currently too coarse.",
            }
        )

    dT_over_rev = float(report["dT_step_over_reversal"])
    if dT_over_rev > 0.2:
        target_ratio = 0.2
        target_dt_ns = max(min_dt_ns, dt_ns * (target_ratio / dT_over_rev))
        target_c_th = c_th * (dT_over_rev / target_ratio)
        actions: List[Dict[str, Any]] = []
        if target_dt_ns < dt_ns * 0.98:
            actions.append(
                {
                    "label": f"Set dt = {target_dt_ns:.4g} ns",
                    "updates": {"cd_dt_ns": float(f"{target_dt_ns:.16g}")},
                }
            )
        if target_c_th > c_th * 1.02:
            actions.append(
                {
                    "label": f"Set C_th = {target_c_th:.4g}",
                    "updates": {"cd_Cth_mW_ns_per_K": float(f"{target_c_th:.16g}")},
                }
            )
        recs.append(
            {
                "id": "thermal_jump",
                "severity": "warning" if dT_over_rev <= 1.0 else "error",
                "title": "Thermal update is too large per step",
                "problem": (
                    f"`dT_step/reversal = {dT_over_rev:.3g}`. "
                    "Hysteresis reversal points can be skipped."
                ),
                "change": (
                    f"Lower `{_label('cd_dt_ns')}` and/or increase `{_label('cd_Cth_mW_ns_per_K')}` "
                    "until this ratio is below ~0.2."
                ),
                "actions": actions,
                "why": "Smaller per-step thermal jumps preserve branch switching timing.",
            }
        )
    return recs


def _build_current_conflict_map(recommendations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    conflicts: Dict[str, Dict[str, Any]] = {}
    for rec in recommendations:
        title = str(rec.get("title", "Diagnostic recommendation")).strip()
        problem = str(rec.get("problem", "")).strip()
        change = str(rec.get("change", "")).strip()
        msg_parts = [p for p in [title, problem, change] if p]
        message = " ".join(msg_parts).strip()
        actions = rec.get("actions", [])
        if not isinstance(actions, list):
            continue

        for action in actions:
            if not isinstance(action, dict):
                continue
            updates = action.get("updates", {})
            if not isinstance(updates, dict) or not updates:
                continue
            for key in updates.keys():
                k = str(key)
                entry = conflicts.setdefault(k, {"message": message, "actions": []})
                if message and not str(entry.get("message", "")).strip():
                    entry["message"] = message
                action_signature = (str(action.get("label", "")), tuple(sorted(updates.items())))
                existing_sigs = {
                    (str(a.get("label", "")), tuple(sorted(dict(a.get("updates", {})).items())))
                    for a in entry["actions"]
                    if isinstance(a, dict)
                }
                if action_signature not in existing_sigs:
                    entry["actions"].append(action)
    return conflicts


def _render_current_drive_recommendations(
    report: Dict[str, float],
    recommendations: List[Dict[str, Any]],
) -> None:
    cols = st.columns(4)
    cols[0].metric("dt/tau_fast", f"{float(report['dt_over_tau_fast']):.3g}")
    cols[1].metric("dt/tau_init", f"{float(report['dt_over_tau_init']):.3g}")
    cols[2].metric("dT step/reversal", f"{float(report['dT_step_over_reversal']):.3g}")
    cols[3].metric("R_fast [Ohm]", f"{float(report['R_eff_fast_ohm']):.4g}")

    if not recommendations:
        st.success(
            "No high-risk numerical flags. If behavior still mismatches reference, "
            "tune current pulse shape/window and resistance hysteresis parameters."
        )
        return

    conflict_map = _build_current_conflict_map(recommendations)
    if conflict_map:
        st.warning("Yellow-marked inputs above have active diagnostics conflicts.")
        st.markdown("Conflict Fix Popovers")
        keys = sorted(conflict_map.keys())
        cols = st.columns(min(3, max(1, len(keys))))
        for idx, key in enumerate(keys):
            col = cols[idx % len(cols)]
            entry = conflict_map.get(key, {})
            label = _label(key)
            with col:
                with st.popover(f"⚠ {_label(key)}"):
                    msg = str(entry.get("message", "")).strip()
                    if msg:
                        st.write(msg)
                    actions = entry.get("actions", [])
                    if isinstance(actions, list) and actions:
                        for a_idx, action in enumerate(actions):
                            action_label = str(action.get("label", "Apply"))
                            if st.button(action_label, key=f"cd_conflict_fix_{key}_{a_idx}"):
                                updates = action.get("updates", {})
                                if isinstance(updates, dict):
                                    for u_key, u_val in updates.items():
                                        st.session_state[str(u_key)] = u_val
                                    st.success(f"Applied update for {_label(key)}.")
                                    _rerun()
                    else:
                        st.caption("No automatic update available for this conflict.")
                st.caption(label)

    with st.expander("Detailed Diagnostic Recommendations", expanded=False):
        st.markdown("### Recommended Parameter Changes")
        for rec in recommendations:
            severity = str(rec.get("severity", "warning"))
            title = str(rec.get("title", "Recommendation"))
            if severity == "error":
                st.error(title)
            elif severity == "info":
                st.info(title)
            else:
                st.warning(title)
            st.write(str(rec.get("problem", "")))
            st.write(f"Change: {rec.get('change', '')}")
            why = str(rec.get("why", "")).strip()
            if why:
                st.caption(why)
            actions = rec.get("actions", [])
            if actions:
                action_cols = st.columns(len(actions))
                for idx, action in enumerate(actions):
                    with action_cols[idx]:
                        label = str(action.get("label", "Apply"))
                        if st.button(label, key=f"cd_fix_{rec.get('id','rec')}_{idx}"):
                            updates = action.get("updates", {})
                            if isinstance(updates, dict):
                                for k, v in updates.items():
                                    st.session_state[str(k)] = v
                                st.success("Applied recommended value(s).")
                                _rerun()


def _render_current_drive_tuning_guide() -> None:
    with st.expander("Symptom -> parameters to tune", expanded=False):
        guide_rows = [
            {
                "Symptom": "No oscillation, smooth ramp/flat response",
                "Change first": "cd_i_stop_uA, cd_C_pF, cd_T_init_K",
                "How to change": (
                    "Increase current range, reduce C if charge dynamics are too slow, and start T_init closer to transition "
                    "(typically high-320s to low-330s K for paper-like parameters)."
                ),
            },
            {
                "Symptom": "Sawtooth/zigzag numerical-looking waveform",
                "Change first": "cd_dt_ns, cd_Cth_mW_ns_per_K",
                "How to change": "Decrease dt and/or increase C_th until dt/tau_fast < 0.1 and dT_step/reversal < 0.2.",
            },
            {
                "Symptom": "Oscillation appears then quickly dies",
                "Change first": "cd_i_stop_uA, cd_S_e_mW_per_K, cd_Cth_mW_ns_per_K",
                "How to change": (
                    "Reduce peak current or cooling S_e if overheating dominates; increase C_th to slow thermal runaway."
                ),
            },
            {
                "Symptom": "Output too random/noisy",
                "Change first": "cd_sigma, cd_seed, cd_dt_ns",
                "How to change": "Lower sigma, set a fixed seed for reproducibility, and reduce dt to avoid noise magnification.",
            },
            {
                "Symptom": "Switching window shifted vs expected experiment",
                "Change first": "cd_C_pF, cd_Cth_mW_ns_per_K, cd_S_e_mW_per_K, resistance fit",
                "How to change": (
                    "Re-fit RT resistance parameters first, then tune C/C_th/S_e jointly so electrical and thermal time scales "
                    "match the measured oscillation onset and damping."
                ),
            },
        ]
        st.dataframe(pd.DataFrame(guide_rows), hide_index=True, use_container_width=True)


def _estimate_input_power_dbm(i_trace_a: np.ndarray, r_ref_ohm: float = _CURRENT_RF_REF_OHM) -> float:
    mask = np.abs(i_trace_a) > 0.0
    if np.any(mask):
        i_rms = float(np.sqrt(np.mean(i_trace_a[mask] ** 2)))
    else:
        i_rms = float(np.sqrt(np.mean(i_trace_a**2)))
    p_w = max((i_rms**2) * float(r_ref_ohm), 1e-18)
    return 10.0 * np.log10(p_w / 1e-3)


def _fft_gain_spectrum(
    t_s: np.ndarray,
    i_in_a: np.ndarray,
    v_vo2_v: np.ndarray,
    fmin_mhz: float = _CURRENT_FFT_RANGE_MHZ[0],
    fmax_mhz: float = _CURRENT_FFT_RANGE_MHZ[1],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use full waveform so pulse edges are captured in the input spectrum.
    if t_s.size < 4:
        return np.array([]), np.array([]), np.array([])

    dt = float(np.median(np.diff(t_s)))
    if dt <= 0.0:
        return np.array([]), np.array([]), np.array([])

    i0 = i_in_a - float(np.mean(i_in_a))
    v0 = v_vo2_v - float(np.mean(v_vo2_v))
    window = np.hanning(i0.size)
    v_fft = np.fft.rfft(v0 * window)
    f_hz = np.fft.rfftfreq(i0.size, d=dt)

    v_mag = np.abs(v_fft)
    active = np.abs(i_in_a) > 0.0
    i_rms = float(np.sqrt(np.mean((i_in_a[active] if np.any(active) else i_in_a) ** 2)))
    if i_rms <= 0.0:
        return np.array([]), np.array([]), np.array([])

    # Pseudo-gain normalization for pulse-driven sweeps:
    # normalize output spectrum by RMS input current times 50 Ohm.
    gain_linear = v_mag / max(i_rms * _CURRENT_RF_REF_OHM, 1e-18)
    gain_db = 20.0 * np.log10(np.maximum(gain_linear, 1e-12))
    f_mhz = f_hz * 1e-6

    mask = (f_mhz >= float(fmin_mhz)) & (f_mhz <= float(fmax_mhz))
    return f_mhz[mask], gain_linear[mask], gain_db[mask]


def _estimate_threshold_current_uA(df_summary: pd.DataFrame) -> Optional[float]:
    if df_summary.empty or "I_avg_uA" not in df_summary.columns or "V_avg_mV" not in df_summary.columns:
        return None
    d = df_summary.sort_values("I_avg_uA").reset_index(drop=True)
    x = d["I_avg_uA"].to_numpy(dtype=float)
    y = d["V_avg_mV"].to_numpy(dtype=float)
    if x.size < 3:
        return None
    # First local maximum before 40% of sweep range; fallback to global maximum.
    x_lim = x[0] + 0.4 * (x[-1] - x[0])
    for i in range(1, x.size - 1):
        if x[i] <= x_lim and y[i] > y[i - 1] and y[i] >= y[i + 1]:
            return float(x[i])
    return float(x[int(np.nanargmax(y))])


def _build_current_summary_and_spectra(
    currents_uA: List[int],
    traces: List[Dict[str, np.ndarray]],
    avg_window_ns: Tuple[float, float] = _CURRENT_AVG_WINDOW_NS,
    fft_range_mhz: Tuple[float, float] = _CURRENT_FFT_RANGE_MHZ,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, float]] = []
    trace_rows: List[Dict[str, float]] = []
    spectra_rows: List[Dict[str, float]] = []

    for i_uA, out in zip(currents_uA, traces):
        t_ns = out["t"] * 1e9
        i_uA_trace = out["I_in"] * 1e6
        v_mV = out["V_vo2"] * 1e3
        t_k = out["T"]
        r_ohm = out["R"]
        p_w = out["P"]

        m_avg = (t_ns >= float(avg_window_ns[0])) & (t_ns <= float(avg_window_ns[1]))
        if np.any(m_avg):
            i_avg_uA = float(np.mean(i_uA_trace[m_avg]))
            v_avg_mV = float(np.mean(v_mV[m_avg]))
            v_std_mV = float(np.std(v_mV[m_avg]))
            v_pp_mV = float(np.max(v_mV[m_avg]) - np.min(v_mV[m_avg]))
            turn_count = _count_turns(v_mV[m_avg])
        else:
            i_avg_uA = float(np.mean(i_uA_trace))
            v_avg_mV = float(np.mean(v_mV))
            v_std_mV = float(np.std(v_mV))
            v_pp_mV = float(np.max(v_mV) - np.min(v_mV))
            turn_count = _count_turns(v_mV)

        p_dbm = _estimate_input_power_dbm(out["I_in"])
        summary_rows.append(
            {
                "I_target_uA": float(i_uA),
                "I_avg_uA": i_avg_uA,
                "V_avg_mV": v_avg_mV,
                "V_std_mV": v_std_mV,
                "V_pp_mV": v_pp_mV,
                "turn_count": float(turn_count),
                "input_power_dBm": p_dbm,
            }
        )

        for idx in range(t_ns.size):
            trace_rows.append(
                {
                    "I_target_uA": float(i_uA),
                    "time_ns": float(t_ns[idx]),
                    "I_in_uA": float(i_uA_trace[idx]),
                    "V_vo2_mV": float(v_mV[idx]),
                    "T_K": float(t_k[idx]),
                    "R_ohm": float(r_ohm[idx]),
                    "P_W": float(p_w[idx]),
                }
            )

        f_mhz, gain_linear, gain_db = _fft_gain_spectrum(
            out["t"],
            out["I_in"],
            out["V_vo2"],
            fmin_mhz=float(fft_range_mhz[0]),
            fmax_mhz=float(fft_range_mhz[1]),
        )
        for f, gl, gd in zip(f_mhz, gain_linear, gain_db):
            spectra_rows.append(
                {
                    "I_target_uA": float(i_uA),
                    "input_power_dBm": p_dbm,
                    "freq_MHz": float(f),
                    "gain_linear": float(gl),
                    "gain_dB": float(gd),
                }
            )

    return pd.DataFrame(trace_rows), pd.DataFrame(summary_rows), pd.DataFrame(spectra_rows)


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


def _plot_current_avg_iv(summary_df: pd.DataFrame) -> go.Figure:
    d = summary_df.sort_values("I_avg_uA")
    fig = go.Figure(
        data=[
            go.Scatter(
                x=d["I_avg_uA"],
                y=d["V_avg_mV"],
                mode="markers",
                marker=dict(size=10, color="#f28e1c"),
                name="Sweep points",
            )
        ]
    )
    fig.update_layout(
        title="Average I_in vs Average V_out (100-250 ns)",
        xaxis_title="Average I_in (uA)",
        yaxis_title="Average V_out (mV)",
        height=560,
    )
    thr = _estimate_threshold_current_uA(d)
    if thr is not None:
        y_thr = float(d.loc[(d["I_avg_uA"] - thr).abs().idxmin(), "V_avg_mV"])
        fig.add_annotation(
            x=thr,
            y=y_thr,
            text="Threshold Current",
            showarrow=True,
            arrowhead=2,
            arrowwidth=2,
            arrowcolor="red",
            ax=120,
            ay=-10,
            font=dict(color="red", size=22),
        )
    return fig


def _color_for_value(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        ratio = 0.5
    else:
        ratio = (float(value) - vmin) / (vmax - vmin)
    ratio = min(1.0, max(0.0, ratio))
    rgba = plt.get_cmap("jet")(ratio)
    return f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})"


def _plot_current_gain_spectra(
    spectra_df: pd.DataFrame,
    y_col: str,
    y_title: str,
    title: str,
) -> go.Figure:
    if spectra_df.empty:
        return go.Figure()
    d = spectra_df.sort_values(["I_target_uA", "freq_MHz"])
    pmin = float(d["input_power_dBm"].min())
    pmax = float(d["input_power_dBm"].max())

    fig = go.Figure()
    for i_uA, g in d.groupby("I_target_uA", sort=True):
        p_dbm = float(g["input_power_dBm"].iloc[0])
        color = _color_for_value(p_dbm, pmin, pmax)
        fig.add_trace(
            go.Scatter(
                x=g["freq_MHz"],
                y=g[y_col],
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{int(round(i_uA))} uA",
                showlegend=False,
                hovertemplate=(
                    "I_target=%{meta[0]:.0f} uA<br>"
                    "P_in=%{meta[1]:.2f} dBm<br>"
                    "f=%{x:.2f} MHz<br>"
                    f"{y_title}=%{{y:.3g}}<extra></extra>"
                ),
                meta=[float(i_uA), p_dbm],
            )
        )

    # Dummy marker for continuous colorbar
    fig.add_trace(
        go.Scatter(
            x=[None, None],
            y=[None, None],
            mode="markers",
            marker=dict(
                size=0.01,
                color=[pmin, pmax],
                cmin=pmin,
                cmax=pmax,
                colorscale="Jet",
                showscale=True,
                colorbar=dict(title="Input Power (dBm)"),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (MHz)",
        yaxis_title=y_title,
        xaxis_type="log",
        height=620,
    )
    return fig


def _plot_current_time_trace(trace_df: pd.DataFrame, current_uA: float) -> go.Figure:
    d = trace_df[trace_df["I_target_uA"] == current_uA].sort_values("time_ns")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=d["time_ns"], y=d["I_in_uA"], name="I_in", line=dict(color="green", width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=d["time_ns"], y=d["V_vo2_mV"], name="V_vo2", line=dict(color="purple", width=2)),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"Current-Driven Trace (I_target={int(round(current_uA))} uA)",
        xaxis_title="time (ns)",
        height=460,
    )
    fig.update_yaxes(title_text="I_in (uA)", secondary_y=False)
    fig.update_yaxes(title_text="V_vo2 (mV)", secondary_y=True)
    return fig


def _plot_current_domain_summary(summary_df: pd.DataFrame, param_label: str) -> go.Figure:
    d = summary_df.sort_values("scan_value")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=d["scan_value"],
            y=100.0 * d["osc_fraction"],
            mode="lines+markers",
            name="Oscillatory fraction (%)",
            line=dict(color="#16a34a", width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=d["scan_value"],
            y=d["best_current_uA"],
            mode="lines+markers",
            name="Best current (uA)",
            line=dict(color="#7e22ce", width=2),
        ),
        secondary_y=True,
    )
    fig.update_layout(title=f"Current-Domain Scan: {param_label}", height=500)
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Oscillatory points (%)", secondary_y=False)
    fig.update_yaxes(title_text="Best current (uA)", secondary_y=True)
    return fig


def _plot_current_domain_heatmap(detail_df: pd.DataFrame, param_label: str) -> go.Figure:
    if detail_df.empty:
        return go.Figure()
    pivot = detail_df.pivot(index="I_target_uA", columns="scan_value", values="turn_count")
    x_vals = pivot.columns.values
    y_vals = pivot.index.values
    z = pivot.values
    fig = go.Figure(
        data=go.Heatmap(
            x=x_vals,
            y=y_vals,
            z=z,
            colorscale="Viridis",
            colorbar=dict(title="Turn count"),
        )
    )
    osc = detail_df[detail_df["oscillatory"] > 0.5]
    if not osc.empty:
        fig.add_trace(
            go.Scatter(
                x=osc["scan_value"],
                y=osc["I_target_uA"],
                mode="markers",
                marker=dict(color="white", size=6, line=dict(color="black", width=1)),
                name="Oscillatory points",
            )
        )
    fig.update_layout(
        title=f"Turn-count map vs {param_label}",
        xaxis_title=param_label,
        yaxis_title="I_target (uA)",
        height=560,
    )
    return fig


def _looks_like_nyquist_zigzag(v_mV: np.ndarray) -> bool:
    """Detect alternating sample-to-sample artifacts that render as a ribbon."""
    if v_mV.size < 8:
        return False
    dv = np.diff(v_mV)
    if dv.size < 2:
        return False
    sign = np.sign(dv)
    alt_ratio = float(np.mean((sign[1:] * sign[:-1]) < 0.0))
    return alt_ratio > 0.75 and float(np.ptp(v_mV)) > 20.0


def _looks_numerically_unstable(v_mV: np.ndarray) -> bool:
    if v_mV.size == 0:
        return False
    return float(np.nanmax(np.abs(v_mV))) > 2_000.0


def _classify_current_trace_oscillation(
    *,
    t_ns: np.ndarray,
    i_in_uA: np.ndarray,
    v_mV: np.ndarray,
    min_turns: int,
    min_vpp_mV: float,
    max_vpp_mV: float,
) -> Dict[str, float]:
    active = np.abs(i_in_uA) > 0.0
    if np.any(active):
        v_eval = v_mV[active]
    else:
        v_eval = v_mV
    if v_eval.size == 0:
        v_eval = v_mV
    turns = _count_turns(v_eval)
    v_pp = float(np.ptp(v_eval)) if v_eval.size else 0.0
    v_std = float(np.std(v_eval)) if v_eval.size else 0.0
    v_avg = float(np.mean(v_eval)) if v_eval.size else 0.0
    unstable = _looks_numerically_unstable(v_eval)
    zigzag = _looks_like_nyquist_zigzag(v_eval)
    oscillatory = (
        (not unstable)
        and (not zigzag)
        and (turns >= int(min_turns))
        and (v_pp >= float(min_vpp_mV))
        and (v_pp <= float(max_vpp_mV))
    )
    return {
        "turn_count": float(turns),
        "V_avg_mV": v_avg,
        "V_std_mV": v_std,
        "V_pp_mV": v_pp,
        "unstable": float(1.0 if unstable else 0.0),
        "zigzag": float(1.0 if zigzag else 0.0),
        "oscillatory": float(1.0 if oscillatory else 0.0),
    }


def _apply_current_scan_override(current_params: Dict[str, Any], param_key: str, value: float) -> None:
    if param_key not in _CURRENT_DOMAIN_SCAN_PARAM_MAP:
        raise ValueError(f"Unsupported current-domain parameter key: {param_key}")
    target_field, scale = _CURRENT_DOMAIN_SCAN_PARAM_MAP[param_key]
    current_params[target_field] = float(value) * float(scale)


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
    script_path = Path(__file__).resolve().parent / "scripts" / "matplotlib_viewer.py"
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
    if job["type"] == "current_domain_scan":
        from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step

        cp_base = dict(config["current_params"])
        cp_base["resist_params"] = dict(cp_base["resist_params"])

        scan_key = str(config["scan_param_key"])
        scan_values = _inclusive_range(
            float(config["scan_start"]),
            float(config["scan_stop"]),
            float(config["scan_step"]),
        )
        currents_uA = list(
            range(
                int(config["I_start_uA"]),
                int(config["I_stop_uA"]) + 1,
                int(config["I_step_uA"]),
            )
        )
        min_turns = int(config.get("min_turns", 6))
        min_vpp_mV = float(config.get("min_vpp_mV", 20.0))
        max_vpp_mV = float(config.get("max_vpp_mV", 700.0))
        base_seed = config.get("seed")

        summary_rows: List[Dict[str, float]] = []
        detail_rows: List[Dict[str, float]] = []

        total_runs = max(1, len(scan_values) * len(currents_uA))
        run_idx = 0
        for s_idx, scan_val in enumerate(scan_values):
            cp = dict(cp_base)
            cp["resist_params"] = YuanhangResistParams(**cp_base["resist_params"])
            _apply_current_scan_override(cp, scan_key, float(scan_val))
            params = CurrentDriveParams(**cp)

            scan_detail: List[Dict[str, float]] = []
            for i_idx, i_uA in enumerate(currents_uA):
                run_seed = None if base_seed is None else int(base_seed) + s_idx * 100_000 + i_idx
                out = simulate_current_step(float(i_uA), params=params, seed=run_seed)
                t_ns = out["t"] * 1e9
                i_trace_uA = out["I_in"] * 1e6
                v_trace_mV = out["V_vo2"] * 1e3
                cls = _classify_current_trace_oscillation(
                    t_ns=t_ns,
                    i_in_uA=i_trace_uA,
                    v_mV=v_trace_mV,
                    min_turns=min_turns,
                    min_vpp_mV=min_vpp_mV,
                    max_vpp_mV=max_vpp_mV,
                )
                row = {
                    "scan_param_key": scan_key,
                    "scan_value": float(scan_val),
                    "I_target_uA": float(i_uA),
                    **cls,
                }
                scan_detail.append(row)
                detail_rows.append(row)
                run_idx += 1
                if progress_cb and (run_idx % 10 == 0 or run_idx == total_runs):
                    progress_cb(f"[current_domain_scan] {run_idx}/{total_runs}")

            dscan = pd.DataFrame(scan_detail)
            osc = dscan[dscan["oscillatory"] > 0.5].sort_values("I_target_uA")
            n_osc = int(len(osc))
            frac = float(n_osc / max(1, len(dscan)))
            if not osc.empty:
                score = 0.6 * osc["turn_count"] + 0.4 * (osc["V_pp_mV"] / max(min_vpp_mV, 1e-9))
                best_idx = int(score.idxmax())
                best_row = osc.loc[best_idx]
                first_i = float(osc["I_target_uA"].min())
                last_i = float(osc["I_target_uA"].max())
                best_i = float(best_row["I_target_uA"])
                best_turns = float(best_row["turn_count"])
                best_vpp = float(best_row["V_pp_mV"])
            else:
                first_i = float("nan")
                last_i = float("nan")
                best_i = float("nan")
                best_turns = 0.0
                best_vpp = 0.0
            summary_rows.append(
                {
                    "scan_param_key": scan_key,
                    "scan_value": float(scan_val),
                    "n_currents": float(len(dscan)),
                    "n_oscillatory": float(n_osc),
                    "osc_fraction": frac,
                    "first_osc_uA": first_i,
                    "last_osc_uA": last_i,
                    "best_current_uA": best_i,
                    "best_turn_count": best_turns,
                    "best_vpp_mV": best_vpp,
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values("scan_value").reset_index(drop=True)
        detail_df = pd.DataFrame(detail_rows).sort_values(["scan_value", "I_target_uA"]).reset_index(drop=True)

        summary_csv = _job_dir(job["id"]) / "current_domain_scan_summary.csv"
        detail_csv = _job_dir(job["id"]) / "current_domain_scan_detail.csv"
        summary_df.to_csv(summary_csv, index=False)
        detail_df.to_csv(detail_csv, index=False)
        job["outputs"].append({"label": "Current domain scan summary CSV", "path": str(summary_csv)})
        job["outputs"].append({"label": "Current domain scan detail CSV", "path": str(detail_csv)})
        _append_job_log(job, f"[current_domain_scan] wrote {summary_csv.name}")
        _append_job_log(job, f"[current_domain_scan] wrote {detail_csv.name}")
        _save_job(job)
        if progress_cb:
            progress_cb("[current_domain_scan] done")
        return

    if job["type"] == "current_sweep":
        from neuristor.current_drive_sim import CurrentDriveParams, run_sweep_make_gif, simulate_current_step

        cp = dict(config["current_params"])
        cp["resist_params"] = YuanhangResistParams(**cp["resist_params"])
        params = CurrentDriveParams(**cp)

        i_start = int(config["I_start_uA"])
        i_stop = int(config["I_stop_uA"])
        i_step = int(config["I_step_uA"])
        seed = config.get("seed")
        i_peak = max(abs(i_start), abs(i_stop))

        report = _current_drive_numerics_report(
            dt_s=params.dt_s,
            C_F=params.C_F,
            C_th_J_per_K=params.C_th_J_per_K,
            T_init_K=params.T_init_K,
            I_peak_uA=float(i_peak),
            resist_params=params.resist_params,
            start_branch=params.start_branch,
        )
        _append_job_log(
            job,
            (
                "[current_sweep] numerics "
                f"dt/tau_init={report['dt_over_tau_init']:.3g}, "
                f"dt/tau_fast={report['dt_over_tau_fast']:.3g}, "
                f"dT_step/reversal~{report['dT_step_over_reversal']:.3g}"
            ),
        )
        for msg in _current_drive_report_messages(report):
            _append_job_log(job, f"[current_sweep][warning] {msg}")

        if progress_cb:
            progress_cb(f"[current_sweep] running {i_start}..{i_stop} uA (step {i_step})")

        frame_dir = _job_dir(job["id"]) / "current_sweep_frames"
        gif_path = _job_dir(job["id"]) / "current_sweep.gif"
        result = run_sweep_make_gif(
            params=params,
            I_start_uA=i_start,
            I_stop_uA=i_stop,
            I_step_uA=i_step,
            frame_duration_s=float(config["frame_duration_s"]),
            frames_dir=frame_dir,
            gif_path=gif_path,
            seed=seed,
        )
        job["outputs"].append({"label": "Current sweep GIF", "path": str(gif_path)})
        _append_job_log(job, f"[current_sweep] wrote {gif_path.name}")
        _save_job(job)

        if progress_cb:
            progress_cb("[current_sweep] extracting sweep summary/spectra")

        currents_uA = [int(v) for v in result["currents_uA"]]
        traces: List[Dict[str, np.ndarray]] = []
        for idx, i_uA in enumerate(currents_uA):
            run_seed = None if seed is None else int(seed) + idx
            traces.append(simulate_current_step(float(i_uA), params=params, seed=run_seed))

        traces_df, summary_df, spectra_df = _build_current_summary_and_spectra(
            currents_uA=currents_uA,
            traces=traces,
            avg_window_ns=_CURRENT_AVG_WINDOW_NS,
            fft_range_mhz=_CURRENT_FFT_RANGE_MHZ,
        )

        traces_csv = _job_dir(job["id"]) / "current_sweep_traces.csv"
        summary_csv = _job_dir(job["id"]) / "current_sweep_summary.csv"
        spectra_csv = _job_dir(job["id"]) / "current_sweep_spectra.csv"
        traces_df.to_csv(traces_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        spectra_df.to_csv(spectra_csv, index=False)

        job["outputs"].append({"label": "Current sweep traces CSV", "path": str(traces_csv)})
        job["outputs"].append({"label": "Current sweep summary CSV", "path": str(summary_csv)})
        job["outputs"].append({"label": "Current sweep spectra CSV", "path": str(spectra_csv)})
        _append_job_log(job, f"[current_sweep] wrote {traces_csv.name}")
        _append_job_log(job, f"[current_sweep] wrote {summary_csv.name}")
        _append_job_log(job, f"[current_sweep] wrote {spectra_csv.name}")
        _save_job(job)
        if progress_cb:
            progress_cb("[current_sweep] done")
        return

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
        [*_SIMULATION_MODES, "Jobs"],
    )
    st.session_state["mode"] = choice
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Parameter Presets")
    st.sidebar.caption(f"Apply to: {choice}")
    if choice in _SIMULATION_MODES:
        c1, c2 = st.sidebar.columns(2)
        if c1.button("Paper Parameters", key="sidebar_paper_params", use_container_width=True):
            ok, msg = _apply_mode_scoped_preset("paper")
            if ok:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)
        if c2.button("Sample Parameters", key="sidebar_sample_params", use_container_width=True):
            ok, msg = _apply_mode_scoped_preset("sample")
            if ok:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)

        profile = _mode_profile(mode=choice)
        st.sidebar.caption(f"Current preset profile: {profile}")
        if profile == "sample":
            st.sidebar.markdown(
                """
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                  <div style="border-left:6px solid #16a34a;height:18px;"></div>
                  <span>Sample-derived parameter</span>
                </div>
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                  <div style="border-left:6px solid #dc2626;height:18px;"></div>
                  <span>Assumed / not extracted</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.caption("Choose a simulation mode to apply presets.")


def _render_terminal() -> None:
    st.subheader("Job Terminal")
    terminal_placeholder = st.empty()
    terminal_placeholder.code(st.session_state.get("terminal_log", ""))


def _inject_global_styles() -> None:
    """Apply app-wide UI overrides."""
    st.markdown(
        """
        <style>
        /* Remove +/- stepper controls from all Streamlit number inputs. */
        div[data-testid="stNumberInput"] button {
            display: none !important;
        }
        div[data-testid="stNumberInput"] input[type="number"] {
            -moz-appearance: textfield;
        }
        div[data-testid="stNumberInput"] input[type="number"]::-webkit-outer-spin-button,
        div[data-testid="stNumberInput"] input[type="number"]::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def _build_job_config_current_drive() -> Dict[str, Any]:
    pulse_off_ns = _parse_optional_float(str(st.session_state["cd_pulse_off_ns"]))
    resist_kwargs = {
        f.name: float(st.session_state[_cd_res_key(f.name)])
        for f in dataclasses.fields(YuanhangResistParams)
    }
    current_params = {
        "dt_s": float(st.session_state["cd_dt_ns"]) * 1e-9,
        "t_end_s": float(st.session_state["cd_t_end_ns"]) * 1e-9,
        "t_pre_s": float(st.session_state["cd_t_pre_ns"]) * 1e-9,
        "pulse_on_s": float(st.session_state["cd_pulse_on_ns"]) * 1e-9,
        "pulse_off_s": None if pulse_off_ns is None else float(pulse_off_ns) * 1e-9,
        "V_init_V": float(st.session_state["cd_V_init_mV"]) * 1e-3,
        "T0_K": float(st.session_state["cd_T0_K"]),
        "T_init_K": float(st.session_state["cd_T_init_K"]),
        "C_F": float(st.session_state["cd_C_pF"]) * 1e-12,
        "C_th_J_per_K": float(st.session_state["cd_Cth_mW_ns_per_K"]) * 1e-12,
        "S_e_W_per_K": float(st.session_state["cd_S_e_mW_per_K"]) * 1e-3,
        "sigma_W_sqrt_s": float(st.session_state["cd_sigma"]),
        "resist_params": resist_kwargs,
        "start_branch": st.session_state["cd_start_branch"],
    }
    seed_text = str(st.session_state["cd_seed"]).strip()
    return {
        "type": "current_sweep",
        "job_name": st.session_state.get("job_name_current_drive", "").strip(),
        "I_start_uA": int(round(float(st.session_state["cd_i_start_uA"]))),
        "I_stop_uA": int(round(float(st.session_state["cd_i_stop_uA"]))),
        "I_step_uA": int(round(float(st.session_state["cd_i_step_uA"]))),
        "frame_duration_s": float(st.session_state["cd_frame_duration_s"]),
        "seed": None if seed_text == "" else int(seed_text),
        "current_params": current_params,
    }


def _build_job_config_current_domain_scan() -> Dict[str, Any]:
    base = _build_job_config_current_drive()
    return {
        "type": "current_domain_scan",
        "job_name": st.session_state.get("job_name_current_domain", "").strip(),
        "I_start_uA": int(base["I_start_uA"]),
        "I_stop_uA": int(base["I_stop_uA"]),
        "I_step_uA": int(base["I_step_uA"]),
        "seed": base["seed"],
        "current_params": base["current_params"],
        "scan_param_key": str(st.session_state["cd_scan_param_key"]),
        "scan_start": float(st.session_state["cd_scan_start"]),
        "scan_stop": float(st.session_state["cd_scan_stop"]),
        "scan_step": float(st.session_state["cd_scan_step"]),
        "min_turns": int(round(float(st.session_state["cd_scan_min_turns"]))),
        "min_vpp_mV": float(st.session_state["cd_scan_min_vpp_mV"]),
        "max_vpp_mV": float(st.session_state["cd_scan_max_vpp_mV"]),
    }


def _validate_current_domain_inputs() -> List[str]:
    errors = _validate_current_drive_inputs()
    key = str(st.session_state["cd_scan_param_key"])
    if key not in _CURRENT_DOMAIN_SCAN_PARAM_MAP:
        errors.append("Selected domain parameter is not supported.")
    start = float(st.session_state["cd_scan_start"])
    stop = float(st.session_state["cd_scan_stop"])
    step = float(st.session_state["cd_scan_step"])
    if step <= 0:
        errors.append("Domain scan step must be > 0.")
    if stop <= start:
        errors.append("Domain scan stop must be greater than start.")
    if float(st.session_state["cd_scan_min_turns"]) < 1:
        errors.append("Domain min turns must be >= 1.")
    min_vpp = float(st.session_state["cd_scan_min_vpp_mV"])
    max_vpp = float(st.session_state["cd_scan_max_vpp_mV"])
    if min_vpp < 0.0:
        errors.append("Domain min Vpp must be >= 0.")
    if max_vpp <= min_vpp:
        errors.append("Domain max Vpp must be greater than min Vpp.")
    try:
        vals = _inclusive_range(start, stop, step)
    except Exception as exc:
        errors.append(f"Invalid domain scan range: {exc}")
        vals = []
    i_start = int(round(float(st.session_state["cd_i_start_uA"])))
    i_stop = int(round(float(st.session_state["cd_i_stop_uA"])))
    i_step = int(round(float(st.session_state["cd_i_step_uA"])))
    n_currents = int(np.floor((i_stop - i_start) / max(i_step, 1))) + 1 if i_stop >= i_start and i_step > 0 else 0
    n_values = len(vals)
    n_sims = n_currents * n_values
    if n_values > 120:
        errors.append("Domain scan has too many parameter points (>120). Increase scan step.")
    if n_sims > 2500:
        errors.append(
            f"Domain scan would run {n_sims} simulations. Reduce parameter/current range or increase step sizes."
        )
    return errors


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
            _selectbox_input(
                _label("start_branch"),
                ["insulator", "metal"],
                key="start_branch",
            )
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
            param_label = st.selectbox(
                "Free variable",
                _param_label_options(),
                key="param_label",
                help=_help("param_label"),
            )
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
            _selectbox_input(
                _label("start_branch"),
                ["insulator", "metal"],
                key="start_branch",
            )
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
            param_x_label = st.selectbox(
                "X variable",
                _param_label_options(),
                key="param_x_label",
                help=_help("param_x_label"),
            )
        with cols[1]:
            _text_input("X start (optional)", key="x_start")
        with cols[2]:
            _text_input("X stop (optional)", key="x_stop")
        with cols[3]:
            _num_input("X step", key="x_step")

        cols = st.columns(4)
        with cols[0]:
            param_y_label = st.selectbox(
                "Y variable",
                _param_label_options(),
                key="param_y_label",
                help=_help("param_y_label"),
            )
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
            _selectbox_input(
                _label("start_branch"),
                ["insulator", "metal"],
                key="start_branch",
            )
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


def _validate_current_drive_inputs() -> List[str]:
    errors: List[str] = []
    i_start = float(st.session_state["cd_i_start_uA"])
    i_stop = float(st.session_state["cd_i_stop_uA"])
    i_step = float(st.session_state["cd_i_step_uA"])
    i_start_i = int(round(i_start))
    i_stop_i = int(round(i_stop))
    i_step_i = int(round(i_step))

    if i_step <= 0:
        errors.append("Current step must be > 0.")
    if i_stop < i_start:
        errors.append("Current stop must be >= current start.")
    if i_step_i <= 0:
        errors.append("Current step rounds to zero in integer uA units; increase step.")
    if i_stop_i < i_start_i:
        errors.append("Current stop must remain >= current start after integer rounding.")
    if float(st.session_state["cd_dt_ns"]) <= 0:
        errors.append("Current-driven dt must be > 0.")
    if float(st.session_state["cd_t_end_ns"]) <= 0:
        errors.append("Current-driven t_end must be > 0.")
    if float(st.session_state["cd_t_pre_ns"]) < 0:
        errors.append("Current-driven t_pre must be >= 0.")
    if float(st.session_state["cd_pulse_on_ns"]) < 0:
        errors.append("Current-driven pulse-on time must be >= 0.")
    pulse_off_text = str(st.session_state["cd_pulse_off_ns"]).strip()
    pulse_off_ns = None
    if pulse_off_text != "":
        try:
            pulse_off_ns = float(pulse_off_text)
        except ValueError:
            errors.append("Current-driven pulse-off time must be numeric or empty.")
    if pulse_off_ns is not None and pulse_off_ns <= float(st.session_state["cd_pulse_on_ns"]):
        errors.append("Pulse-off time must be greater than pulse-on time.")
    if float(st.session_state["cd_C_pF"]) <= 0:
        errors.append("Current-driven C must be > 0.")
    if float(st.session_state["cd_Cth_mW_ns_per_K"]) <= 0:
        errors.append("Current-driven C_th must be > 0.")
    if float(st.session_state["cd_S_e_mW_per_K"]) <= 0:
        errors.append("Current-driven S_e must be > 0.")
    if float(st.session_state["cd_frame_duration_s"]) <= 0:
        errors.append("GIF frame duration must be > 0.")

    seed_text = str(st.session_state["cd_seed"]).strip()
    if seed_text != "":
        try:
            int(seed_text)
        except ValueError:
            errors.append("Current-driven seed must be an integer or empty.")

    for f in dataclasses.fields(YuanhangResistParams):
        val = float(st.session_state[_cd_res_key(f.name)])
        if f.name in {"R0", "Rm0", "w", "width_factor"} and val <= 0:
            errors.append(f"Current-driven resistance parameter {f.name} must be > 0.")
        if f.name in {"beta", "gamma"} and val == 0:
            errors.append(f"Current-driven resistance parameter {f.name} must be non-zero.")
    return errors


def _current_drive_input_diagnostics() -> Optional[Dict[str, Any]]:
    try:
        resist = YuanhangResistParams(
            **{f.name: float(st.session_state[_cd_res_key(f.name)]) for f in dataclasses.fields(YuanhangResistParams)}
        )
        i_peak_uA = max(
            abs(float(st.session_state["cd_i_start_uA"])),
            abs(float(st.session_state["cd_i_stop_uA"])),
        )
        report = _current_drive_numerics_report(
            dt_s=float(st.session_state["cd_dt_ns"]) * 1e-9,
            C_F=float(st.session_state["cd_C_pF"]) * 1e-12,
            C_th_J_per_K=float(st.session_state["cd_Cth_mW_ns_per_K"]) * 1e-12,
            T_init_K=float(st.session_state["cd_T_init_K"]),
            I_peak_uA=i_peak_uA,
            resist_params=resist,
            start_branch=str(st.session_state["cd_start_branch"]),
        )
        recommendations = _current_drive_recommendations(
            report,
            dt_ns=float(st.session_state["cd_dt_ns"]),
            c_th_mW_ns_per_K=float(st.session_state["cd_Cth_mW_ns_per_K"]),
        )
        return {
            "report": report,
            "messages": _current_drive_report_messages(report),
            "recommendations": recommendations,
            "conflicts": _build_current_conflict_map(recommendations),
        }
    except Exception:
        return None


def _render_current_drive() -> None:
    st.header("Current-Driven Sweep")
    st.caption(
        "Current-drive model uses an ideal current source at the VO2 node: "
        "dV/dt = (I_in - V/R_vo2)/C. External/source series resistance is not part of this model."
    )
    st.session_state["cd_diag_conflicts"] = {}

    with st.form("current_drive_form"):
        _text_input("Simulation name (optional)", key="job_name_current_drive")

        st.markdown("### Current Sweep")
        _render_input_grid(
            ["cd_i_start_uA", "cd_i_stop_uA", "cd_i_step_uA", "cd_frame_duration_s"],
            _num_input,
            columns=4,
        )

        st.markdown("### Waveform and Time")
        _render_input_grid(["cd_dt_ns", "cd_t_end_ns", "cd_t_pre_ns", "cd_C_pF"], _num_input, columns=4)
        cols = st.columns(4)
        with cols[0]:
            _num_input(_label("cd_pulse_on_ns"), key="cd_pulse_on_ns")
        with cols[1]:
            _text_input("Current Sim Pulse Off [ns] (optional)", key="cd_pulse_off_ns")

        st.markdown("### Thermal and Initial State")
        _render_input_grid(["cd_Cth_mW_ns_per_K", "cd_S_e_mW_per_K", "cd_sigma"], _num_input, columns=4)
        _render_input_grid(["cd_T0_K", "cd_T_init_K", "cd_V_init_mV"], _num_input, columns=4)

        cols = st.columns(4)
        with cols[0]:
            _selectbox_input(
                _label("cd_start_branch"),
                ["insulator", "metal"],
                key="cd_start_branch",
            )
        with cols[1]:
            _text_input("Seed (optional)", key="cd_seed")

        with st.expander("Current-Driven Resistance / Hysteresis", expanded=False):
            resist_keys = [f.name for f in dataclasses.fields(YuanhangResistParams)]
            for row in _chunked(resist_keys, 4):
                cols = st.columns(4)
                for col, name in zip(cols, row):
                    with col:
                        _num_input(
                            _label(name),
                            key=_cd_res_key(name),
                            help=_help(_cd_res_key(name)),
                        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            submitted_add = st.form_submit_button("Add to batch")
        with btn_col2:
            submitted_run = st.form_submit_button("Run now")

    terminal_placeholder = st.empty()
    terminal_placeholder.code(st.session_state.get("terminal_log", ""))

    if submitted_add or submitted_run:
        errors = _validate_current_drive_inputs()
        if errors:
            st.error("Please fix these inputs before queuing current-driven sweep:")
            for msg in errors:
                st.write(f"- {msg}")
        else:
            job_config = _build_job_config_current_drive()
            if submitted_add:
                st.session_state["batch_jobs"].append(job_config)
                _update_terminal("[batch] added current_sweep job", terminal_placeholder)
            if submitted_run:
                st.session_state["terminal_log"] = ""
                terminal_placeholder.code("")
                job = _create_job(job_config)
                _enqueue_job(job["id"])
                _update_terminal(f"[job] queued {job['type']} ({job['id']})", terminal_placeholder)

    st.caption("Current-driven sweeps are saved and rendered in the Jobs view.")
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
        if job["type"] == "current_sweep":
            outputs = job.get("outputs", [])
            summary_path = next((o["path"] for o in outputs if o["path"].endswith("current_sweep_summary.csv")), None)
            spectra_path = next((o["path"] for o in outputs if o["path"].endswith("current_sweep_spectra.csv")), None)
            traces_path = next((o["path"] for o in outputs if o["path"].endswith("current_sweep_traces.csv")), None)

            if summary_path and os.path.exists(summary_path):
                df_summary = pd.read_csv(summary_path)
                fig_avg = _plot_current_avg_iv(df_summary)
                st.plotly_chart(fig_avg, use_container_width=True)
                thr = _estimate_threshold_current_uA(df_summary)
                if thr is not None:
                    st.caption(f"Estimated threshold current: {thr:.1f} uA")
            else:
                st.info("Current sweep summary CSV not found yet.")

            if spectra_path and os.path.exists(spectra_path):
                df_spectra = pd.read_csv(spectra_path)
                fig_db = _plot_current_gain_spectra(
                    df_spectra,
                    y_col="gain_dB",
                    y_title="Gain (dB)",
                    title="Gain (dB) vs Frequency",
                )
                fig_linear = _plot_current_gain_spectra(
                    df_spectra,
                    y_col="gain_linear",
                    y_title="Linear Gain",
                    title="Linear Gain vs Frequency",
                )
                st.plotly_chart(fig_db, use_container_width=True)
                st.plotly_chart(fig_linear, use_container_width=True)
            else:
                st.info("Current sweep spectra CSV not found yet.")

            if traces_path and os.path.exists(traces_path):
                df_traces = pd.read_csv(traces_path)
                currents = sorted(df_traces["I_target_uA"].unique().tolist())
                if currents:
                    selected_current = st.select_slider(
                        "Current trace to inspect (uA)",
                        options=currents,
                        value=currents[0],
                        key=f"current_trace_slider_{job['id']}",
                    )
                    d_sel = df_traces[df_traces["I_target_uA"] == float(selected_current)].sort_values("time_ns")
                    v_sel = d_sel["V_vo2_mV"].to_numpy(dtype=float)
                    i_sel = d_sel["I_in_uA"].to_numpy(dtype=float)
                    if _looks_numerically_unstable(v_sel):
                        st.warning(
                            "This trace exceeds +/-2 V equivalent range in mV units, which usually indicates "
                            "numerical instability for this preset/current."
                        )
                    v_check = v_sel[np.abs(i_sel) > 0.0] if np.any(np.abs(i_sel) > 0.0) else v_sel
                    if _looks_like_nyquist_zigzag(v_check):
                        st.warning(
                            "This trace looks like an alternating-sample numerical artifact "
                            "(zigzag/Nyquist ribbon), not a physical oscillation. "
                            "Use a smaller integration step or a different preset."
                        )
                    fig_trace = _plot_current_time_trace(df_traces, float(selected_current))
                    st.plotly_chart(fig_trace, use_container_width=True)
            else:
                st.info("Current sweep trace CSV not found yet.")
        if job["type"] == "current_domain_scan":
            outputs = job.get("outputs", [])
            summary_path = next((o["path"] for o in outputs if o["path"].endswith("current_domain_scan_summary.csv")), None)
            detail_path = next((o["path"] for o in outputs if o["path"].endswith("current_domain_scan_detail.csv")), None)
            param_key = str(job["params"].get("scan_param_key", "cd_C_pF"))
            param_label = _label(param_key)

            if summary_path and os.path.exists(summary_path):
                df_summary = pd.read_csv(summary_path)
                if not df_summary.empty:
                    d_rank = df_summary.sort_values(
                        ["osc_fraction", "best_turn_count"],
                        ascending=[False, False],
                    ).head(12)
                    st.dataframe(d_rank, hide_index=True, use_container_width=True)
                    fig_summary = _plot_current_domain_summary(df_summary, param_label)
                    st.plotly_chart(fig_summary, use_container_width=True)
                else:
                    st.info("Domain summary CSV is empty.")
            else:
                st.info("Current domain summary CSV not found yet.")

            if detail_path and os.path.exists(detail_path):
                df_detail = pd.read_csv(detail_path)
                if not df_detail.empty:
                    fig_map = _plot_current_domain_heatmap(df_detail, param_label)
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.info("Domain detail CSV is empty.")
            else:
                st.info("Current domain detail CSV not found yet.")
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
    _inject_global_styles()
    _init_defaults()
    _ensure_worker()

    _render_top_bar()

    _render_sidebar()

    content = st.empty()
    mode = st.session_state["mode"]
    with content.container():
        if mode == "Single Simulation":
            _render_single()
        elif mode == "Sweep over Free Variable":
            _render_sweep1d()
        elif mode == "2D Frequency Sweep":
            _render_sweep2d()
        elif mode == "Current-Driven Sweep":
            _render_current_drive()
        else:
            _render_jobs_view()


if __name__ == "__main__":
    main()
