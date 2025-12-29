"""
Streamlit GUI for the VO₂ neuristor simulator and analyses.

Implements the specification in IMPLEMENTATION_GUIDE.md:
- Uses model.py as the single source of truth for simulation and helpers.
- Exposes PI-requested analyses (baselines, Vmax vs Vin, power plots, C sweeps, 3D freq, R_ins).
- Supports single runs, 1D/2D sweeps, oscillation-domain finding, exports, and progress/logging UX.
"""
from __future__ import annotations

import dataclasses
import io
import json
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import plots as analysis
from model import (
    SimOut,
    YuanhangCircuitParams,
    YuanhangResistParams,
    detect_spike_times,
    find_oscillatory_band_1d,
    is_oscillatory,
    series_first,
    simulate_yuanhang,
)


# ---------------------------------------------------------------------------
# Streamlit / session setup
# ---------------------------------------------------------------------------


st.set_page_config(page_title="VO₂ Neuristor Simulator", layout="wide")

if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "status" not in st.session_state:
    st.session_state["status"] = {"state": "Idle", "mode": "", "detail": ""}
if "cancel_requested" not in st.session_state:
    st.session_state["cancel_requested"] = False
if "last_sim" not in st.session_state:
    st.session_state["last_sim"] = None
if "last_config" not in st.session_state:
    st.session_state["last_config"] = None
if "last_sweep_df" not in st.session_state:
    st.session_state["last_sweep_df"] = None
if "last_metrics" not in st.session_state:
    st.session_state["last_metrics"] = None
if "inputs_initialized" not in st.session_state:
    st.session_state["inputs_initialized"] = False


# ---------------------------------------------------------------------------
# Presets and parameter management
# ---------------------------------------------------------------------------


def _paper_params() -> tuple[YuanhangResistParams, YuanhangCircuitParams]:
    """Return paper preset for convenience."""
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


def _default_params() -> tuple[YuanhangResistParams, YuanhangCircuitParams]:
    return YuanhangResistParams(), YuanhangCircuitParams()


INPUT_KEYS = [
    "vin",
    "vin_list",
    "example_vin",
    "start_branch",
    "t_end_us",
    "dt_ns",
    "t_start_us",
    "t_end_window_us",
    "threshold_A",
    "noise_seed",
    "nx",
    "ny",
    "R_series_kohm",
    "C_par_pF",
    "T_base_K",
    "Cth_mW_ns_per_K",
    "Sth_mW_per_K",
    "Cth_factor",
    "couple_factor",
    "noise_strength",
    "R0",
    "Ea_over_k",
    "Rm0",
    "Rm_factor",
    "w",
    "Tc_K",
    "beta",
    "gamma",
    "width_factor",
    "T_min_K",
    "T_max_K",
    "reversal_threshold_K",
    "c_list",
    "c_start",
    "c_stop",
    "c_step",
    "r_list",
]


def apply_preset(name: str) -> None:
    """Apply preset values into session_state."""
    if name == "Paper preset":
        resist, circuit = _paper_params()
    else:
        resist, circuit = _default_params()

    defaults = {
        "vin": 14.5,
        "vin_list": "10.5,12.5,14.5,16.5",
        "example_vin": "14.5",
        "start_branch": "insulator",
        "t_end_us": 300.0,
        "dt_ns": 10.0,
        "t_start_us": 25.0,
        "t_end_window_us": 300.0,
        "threshold_A": "1e-3",
        "noise_seed": "",
        "nx": 1,
        "ny": 1,
        "R_series_kohm": circuit.R_series_kohm,
        "C_par_pF": circuit.C_par_pF,
        "T_base_K": circuit.T_base_K,
        "Cth_mW_ns_per_K": circuit.Cth_mW_ns_per_K,
        "Sth_mW_per_K": circuit.Sth_mW_per_K,
        "Cth_factor": circuit.Cth_factor,
        "couple_factor": circuit.couple_factor,
        "noise_strength": circuit.noise_strength,
        "R0": resist.R0,
        "Ea_over_k": resist.Ea_over_k,
        "Rm0": resist.Rm0,
        "Rm_factor": resist.Rm_factor,
        "w": resist.w,
        "Tc_K": resist.Tc_K,
        "beta": resist.beta,
        "gamma": resist.gamma,
        "width_factor": resist.width_factor,
        "T_min_K": resist.T_min_K,
        "T_max_K": resist.T_max_K,
        "reversal_threshold_K": resist.reversal_threshold_K,
        "c_list": "100,150,200",
        "c_start": 80.0,
        "c_stop": 250.0,
        "c_step": 10.0,
        "r_list": "10,12,14,16",
    }
    for key, val in defaults.items():
        st.session_state[key] = val
    st.session_state["inputs_initialized"] = True


def ensure_defaults_initialized() -> None:
    if not st.session_state["inputs_initialized"]:
        apply_preset("Default")


# ---------------------------------------------------------------------------
# Logging and status helpers
# ---------------------------------------------------------------------------


def log_line(msg: str) -> None:
    st.session_state["logs"].append(msg)


def reset_logs() -> None:
    st.session_state["logs"] = []


def update_status(state: str, mode: str, detail: str) -> None:
    st.session_state["status"] = {"state": state, "mode": mode, "detail": detail}


def render_status_and_logs() -> None:
    status = st.session_state["status"]
    with st.container():
        st.subheader("Run Status")
        st.info(f"State: {status['state']} | Mode: {status['mode']} | Detail: {status['detail']}")
    with st.expander("Logs", expanded=False):
        if st.session_state["logs"]:
            st.text("\n".join(st.session_state["logs"]))
        else:
            st.text("No logs yet.")


# ---------------------------------------------------------------------------
# Parameter building
# ---------------------------------------------------------------------------


def build_resist_params_from_state() -> YuanhangResistParams:
    return YuanhangResistParams(
        R0=float(st.session_state["R0"]),
        Ea_over_k=float(st.session_state["Ea_over_k"]),
        Rm0=float(st.session_state["Rm0"]),
        Rm_factor=float(st.session_state["Rm_factor"]),
        w=float(st.session_state["w"]),
        Tc_K=float(st.session_state["Tc_K"]),
        beta=float(st.session_state["beta"]),
        gamma=float(st.session_state["gamma"]),
        width_factor=float(st.session_state["width_factor"]),
        T_min_K=float(st.session_state["T_min_K"]),
        T_max_K=float(st.session_state["T_max_K"]),
        reversal_threshold_K=float(st.session_state["reversal_threshold_K"]),
    )


def build_circuit_params_from_state(nx: int, ny: int) -> YuanhangCircuitParams:
    circuit = YuanhangCircuitParams(
        R_series_kohm=float(st.session_state["R_series_kohm"]),
        C_par_pF=float(st.session_state["C_par_pF"]),
        Cth_mW_ns_per_K=float(st.session_state["Cth_mW_ns_per_K"]),
        Sth_mW_per_K=float(st.session_state["Sth_mW_per_K"]),
        couple_factor=float(st.session_state["couple_factor"]),
        Cth_factor=float(st.session_state["Cth_factor"]),
        noise_strength=float(st.session_state["noise_strength"]),
        dimension=1 if (nx == 1 or ny == 1) else 2,
        T_base_K=float(st.session_state["T_base_K"]),
    )
    return circuit


def to_dict(obj: Any) -> Dict[str, Any]:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError("Unsupported type for to_dict")


# ---------------------------------------------------------------------------
# Simulation + metrics
# ---------------------------------------------------------------------------


def _local_extrema_indices(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if y.size < 3:
        return np.array([], dtype=int), np.array([], dtype=int)
    y0 = y[:-2]
    y1 = y[1:-1]
    y2 = y[2:]
    max_mask = (y0 < y1) & (y1 >= y2)
    min_mask = (y0 > y1) & (y1 <= y2)
    max_idx = np.where(max_mask)[0] + 1
    min_idx = np.where(min_mask)[0] + 1
    return min_idx, max_idx


def compute_metrics(simout: SimOut, t_start_us: float, t_end_us: float, threshold_A: float) -> Dict[str, Any]:
    t = np.asarray(simout["time_s"], dtype=float)
    V = np.asarray(series_first(simout["V_node"]), dtype=float)
    I = np.asarray(series_first(simout["I_vo2"]), dtype=float)
    T = np.asarray(series_first(simout["T_K"]), dtype=float)
    P = V * I
    mask = (t * 1e6 >= t_start_us) & (t * 1e6 <= t_end_us)
    if np.any(mask):
        tw, Vw, Iw, Pw, Tw = t[mask], V[mask], I[mask], P[mask], T[mask]
    else:
        tw, Vw, Iw, Pw, Tw = t, V, I, P, T
    spikes = detect_spike_times(tw.tolist(), Iw.tolist(), threshold_A=threshold_A)
    spike_count = len(spikes)
    if spike_count >= 2:
        isi_us = np.diff(np.asarray(spikes)) * 1e6
        freq_mhz = float(1.0 / float(np.mean(isi_us))) if isi_us.size > 0 else float("nan")
    else:
        freq_mhz = float("nan")
    min_idx, _ = _local_extrema_indices(Vw)
    V_baseline = float(np.mean(Vw[min_idx])) if min_idx.size else float(np.min(Vw))
    min_idx_T, _ = _local_extrema_indices(Tw)
    T_baseline = float(np.mean(Tw[min_idx_T])) if min_idx_T.size else float(np.min(Tw))
    return {
        "oscillatory": is_oscillatory(simout, t_start_us=t_start_us, t_end_us=t_end_us, threshold_A=threshold_A),
        "spike_count": spike_count,
        "frequency_MHz": freq_mhz,
        "Vmax": float(np.max(Vw)),
        "Pmax": float(np.max(Pw)),
        "Pmin": float(np.min(Pw)),
        "V_baseline": V_baseline,
        "T_baseline": T_baseline,
    }


@st.cache_data(show_spinner=False)
def run_sim_cached(
    config: Dict[str, Any], resist_dict: Dict[str, Any], circuit_dict: Dict[str, Any]
) -> SimOut:  # pragma: no cover - executed via Streamlit
    resist = YuanhangResistParams(**resist_dict)
    circuit = YuanhangCircuitParams(**circuit_dict)
    return simulate_yuanhang(
        Vin=config["Vin"],
        t_end=config["t_end"],
        dt=config["dt"],
        resist_params=resist,
        circuit_params=circuit,
        lattice_shape=config["lattice_shape"],
        start_branch=config["start_branch"],
        noise_seed=config["noise_seed"],
    )


def current_sim_config() -> Dict[str, Any]:
    nx = int(st.session_state["nx"])
    ny = int(st.session_state["ny"])
    return {
        "Vin": float(st.session_state["vin"]),
        "t_end": float(st.session_state["t_end_us"]) * 1e-6,
        "dt": float(st.session_state["dt_ns"]) * 1e-9,
        "start_branch": st.session_state["start_branch"],
        "lattice_shape": (max(1, nx), max(1, ny)),
        "noise_seed": None if str(st.session_state["noise_seed"]).strip() == "" else int(st.session_state["noise_seed"]),
    }


def run_single_simulation() -> SimOut:
    config = current_sim_config()
    resist = build_resist_params_from_state()
    circuit = build_circuit_params_from_state(config["lattice_shape"][0], config["lattice_shape"][1])
    simout = run_sim_cached(config, to_dict(resist), to_dict(circuit))
    st.session_state["last_sim"] = simout
    st.session_state["last_config"] = {"config": config, "resist": to_dict(resist), "circuit": to_dict(circuit)}
    return simout


def sweep_parameter(
    base_config: Dict[str, Any],
    resist: YuanhangResistParams,
    circuit: YuanhangCircuitParams,
    param_name: str,
    values: Iterable[float],
    t_start_us: float,
    t_end_us: float,
    threshold_A: float,
    progress_placeholder,
    table_placeholder,
    plot_placeholder,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    vals = list(values)
    progress = progress_placeholder.progress(0.0)
    for idx, val in enumerate(vals):
        if st.session_state.get("cancel_requested", False):
            log_line("Sweep cancelled by user.")
            break
        cfg = dict(base_config)
        res = dataclasses.replace(resist)
        circ = dataclasses.replace(circuit)
        if param_name == "Vin":
            cfg["Vin"] = val
        elif hasattr(circ, param_name):
            setattr(circ, param_name, val)
        elif hasattr(res, param_name):
            setattr(res, param_name, val)
        else:
            log_line(f"[sweep] Unknown parameter {param_name}, skipping.")
            continue
        sim = run_sim_cached(cfg, to_dict(res), to_dict(circ))
        metrics = compute_metrics(sim, t_start_us, t_end_us, threshold_A)
        row = {
            "sweep_value": val,
            "oscillatory": metrics["oscillatory"],
            "spike_count": metrics["spike_count"],
            "frequency_MHz": metrics["frequency_MHz"],
            "Vmax": metrics["Vmax"],
            "Pmax": metrics["Pmax"],
            "Pmin": metrics["Pmin"],
        }
        rows.append(row)
        df = pd.DataFrame(rows)
        table_placeholder.dataframe(df)
        if (idx + 1) % 3 == 0:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(df["sweep_value"], df["frequency_MHz"], "o-")
            ax.set_xlabel(param_name)
            ax.set_ylabel("Frequency (MHz)")
            ax.grid(True)
            plot_placeholder.pyplot(fig)
            plt.close(fig)
        progress.progress((idx + 1) / len(vals))
        log_line(f"[sweep] {idx+1}/{len(vals)}: {param_name}={val} → osc={metrics['oscillatory']}, f={metrics['frequency_MHz']:.3g} MHz")
    df = pd.DataFrame(rows)
    st.session_state["last_sweep_df"] = df
    return df


def domain_find(
    param_name: str,
    values: List[float],
    base_config: Dict[str, Any],
    resist: YuanhangResistParams,
    circuit: YuanhangCircuitParams,
    t_start_us: float,
    t_end_us: float,
    threshold_A: float,
    progress_placeholder,
    table_placeholder,
) -> Tuple[float | None, float | None, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    progress = progress_placeholder.progress(0.0)
    for idx, val in enumerate(values):
        if st.session_state.get("cancel_requested", False):
            log_line("Domain finder cancelled by user.")
            break
        cfg = dict(base_config)
        res = dataclasses.replace(resist)
        circ = dataclasses.replace(circuit)
        if param_name == "Vin":
            cfg["Vin"] = val
        elif hasattr(circ, param_name):
            setattr(circ, param_name, val)
        elif hasattr(res, param_name):
            setattr(res, param_name, val)
        sim = run_sim_cached(cfg, to_dict(res), to_dict(circ))
        metrics = compute_metrics(sim, t_start_us, t_end_us, threshold_A)
        rows.append({"value": val, "oscillatory": metrics["oscillatory"], "frequency_MHz": metrics["frequency_MHz"]})
        progress.progress((idx + 1) / len(values))
        table_placeholder.dataframe(pd.DataFrame(rows))
        log_line(f"[domain] {idx+1}/{len(values)} {param_name}={val} → osc={metrics['oscillatory']}")
    df = pd.DataFrame(rows)
    if not df.empty and df["oscillatory"].any():
        osc_values = df[df["oscillatory"]]["value"]
        return float(osc_values.min()), float(osc_values.max()), df
    return None, None, df


def capture_new_figures(fn) -> List[plt.Figure]:
    """Capture figures created by fn()."""
    before = set(plt.get_fignums())
    fn()
    after = set(plt.get_fignums())
    new_nums = sorted(after - before)
    return [plt.figure(num) for num in new_nums]


def render_figures(figs: List[plt.Figure]) -> None:
    for fig in figs:
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------------------------


def sidebar_inputs() -> None:
    ensure_defaults_initialized()
    st.sidebar.header("Simulation Setup")
    preset = st.sidebar.selectbox("Preset", ["Default", "Paper preset"])
    colp1, colp2 = st.sidebar.columns(2)
    if colp1.button("Load preset"):
        apply_preset(preset)
    if colp2.button("Reset all"):
        apply_preset("Default")

    st.sidebar.markdown("#### Experiment mode")
    st.sidebar.radio("Mode", ["Single run", "1D sweep", "2D sweep", "Oscillation domain"], key="exp_mode")

    st.sidebar.markdown("#### Time settings")
    st.sidebar.number_input("t_end (µs)", value=st.session_state["t_end_us"], key="t_end_us")
    st.sidebar.number_input("dt (ns)", value=st.session_state["dt_ns"], key="dt_ns", min_value=0.001)
    st.sidebar.number_input("Steady-state start (µs)", value=st.session_state["t_start_us"], key="t_start_us")
    st.sidebar.number_input("Steady-state end (µs)", value=st.session_state["t_end_window_us"], key="t_end_window_us")
    st.sidebar.text_input("Spike threshold (A)", value=str(st.session_state.get("threshold_A", "1e-3")), key="threshold_A")

    st.sidebar.markdown("#### Electrical")
    st.sidebar.number_input("Vin (V)", value=st.session_state["vin"], key="vin")
    st.sidebar.text_input("Vin list (csv)", value=st.session_state["vin_list"], key="vin_list")
    st.sidebar.text_input("Example Vin (for power plots)", value=st.session_state["example_vin"], key="example_vin")
    st.sidebar.number_input("R_series (kΩ)", value=st.session_state["R_series_kohm"], key="R_series_kohm")
    st.sidebar.number_input("C_par (pF)", value=st.session_state["C_par_pF"], key="C_par_pF")
    st.sidebar.radio("Start branch", ["insulator", "metal"], key="start_branch")

    st.sidebar.markdown("#### Thermal")
    st.sidebar.number_input("T_base (K)", value=st.session_state["T_base_K"], key="T_base_K")
    st.sidebar.number_input("Cth (mW*ns/K)", value=st.session_state["Cth_mW_ns_per_K"], key="Cth_mW_ns_per_K")
    st.sidebar.number_input("Sth (mW/K)", value=st.session_state["Sth_mW_per_K"], key="Sth_mW_per_K")
    st.sidebar.number_input("Cth_factor", value=st.session_state["Cth_factor"], key="Cth_factor")
    st.sidebar.number_input("couple_factor", value=st.session_state["couple_factor"], key="couple_factor")
    st.sidebar.number_input("noise_strength", value=st.session_state["noise_strength"], key="noise_strength")

    st.sidebar.markdown("#### Lattice")
    st.sidebar.number_input("Nx", value=st.session_state["nx"], key="nx", min_value=1, step=1)
    st.sidebar.number_input("Ny", value=st.session_state["ny"], key="ny", min_value=1, step=1)

    with st.sidebar.expander("Hysteresis / resistance (advanced)"):
        st.number_input("R0 (Ω)", value=st.session_state["R0"], key="R0")
        st.number_input("Ea_over_k (K)", value=st.session_state["Ea_over_k"], key="Ea_over_k")
        st.number_input("Rm0 (Ω)", value=st.session_state["Rm0"], key="Rm0")
        st.number_input("Rm_factor", value=st.session_state["Rm_factor"], key="Rm_factor")
        st.number_input("w (K)", value=st.session_state["w"], key="w")
        st.number_input("Tc_K (K)", value=st.session_state["Tc_K"], key="Tc_K")
        st.number_input("beta (1/K)", value=st.session_state["beta"], key="beta")
        st.number_input("gamma", value=st.session_state["gamma"], key="gamma")
        st.number_input("width_factor", value=st.session_state["width_factor"], key="width_factor")
        st.number_input("T_min_K", value=st.session_state["T_min_K"], key="T_min_K")
        st.number_input("T_max_K", value=st.session_state["T_max_K"], key="T_max_K")
        st.number_input("reversal_threshold_K", value=st.session_state["reversal_threshold_K"], key="reversal_threshold_K")

    st.sidebar.markdown("#### Sweeps")
    st.sidebar.text_input("C list (pF)", value=st.session_state["c_list"], key="c_list")
    st.sidebar.number_input("C start (pF)", value=st.session_state["c_start"], key="c_start")
    st.sidebar.number_input("C stop (pF)", value=st.session_state["c_stop"], key="c_stop")
    st.sidebar.number_input("C step (pF)", value=st.session_state["c_step"], key="c_step")
    st.sidebar.text_input("R_load list (kΩ)", value=st.session_state["r_list"], key="r_list")
    st.sidebar.text_input("Noise seed (blank = none)", value=st.session_state["noise_seed"], key="noise_seed")

    st.sidebar.markdown("#### Control")
    if st.sidebar.button("Stop current run"):
        st.session_state["cancel_requested"] = True
        log_line("User requested cancellation.")


# ---------------------------------------------------------------------------
# UI Tabs
# ---------------------------------------------------------------------------


def render_run_tab():
    st.header("Run")
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        if st.button("Run simulation"):
            reset_logs()
            st.session_state["cancel_requested"] = False
            update_status("Running", "Single", "Simulating...")
            sim = run_single_simulation()
            metrics = compute_metrics(
                sim,
                float(st.session_state["t_start_us"]),
                float(st.session_state["t_end_window_us"]),
                float(st.session_state.get("threshold_A", 1e-3)),
            )
            st.session_state["last_metrics"] = metrics
            update_status("Completed", "Single", f"Oscillatory={metrics['oscillatory']}, f={metrics['frequency_MHz']:.3g} MHz")
            log_line("Single simulation completed.")
    with col_status:
        render_status_and_logs()

        if st.session_state["last_sim"]:
            sim = st.session_state["last_sim"]
            metrics = st.session_state.get("last_metrics", {})
            st.subheader("Summary metrics")
            st.write(metrics)

            fig = plt.figure()
            # Use analysis helper for legible time traces including V_load.
            analysis.plot_time_traces(
                sim,
                R_series_kohm=float(st.session_state["R_series_kohm"]),
                title="V_vo2/V_load, T_vo2, I_vo2, P_vo2 vs time",
            )
            st.pyplot(fig)
            plt.close(fig)


def render_pi_tab():
    st.header("PI Analyses")
    checks = {
        "baselines": st.checkbox("Baselines of T and V vs time", value=False),
        "vmax": st.checkbox("Vmax vs Vin", value=False),
        "power": st.checkbox("Power vs time and power peaks vs Vin", value=False),
        "c_overlay": st.checkbox("Capacitance sweep power overlay", value=False),
        "freq3d": st.checkbox("3D frequency vs (C, R_load)", value=False),
        "r_ins": st.checkbox("Resistance in insulating state vs time", value=False),
    }
    if st.button("Run selected analyses"):
        reset_logs()
        st.session_state["cancel_requested"] = False
        update_status("Running", "PI analyses", "Executing analyses...")
        figs: List[plt.Figure] = []
        config = current_sim_config()
        resist = build_resist_params_from_state()
        circuit = build_circuit_params_from_state(config["lattice_shape"][0], config["lattice_shape"][1])

        if checks["baselines"]:
            sim = run_single_simulation()
            figs += capture_new_figures(
                lambda: analysis.plot_baselines_T_and_V_vs_time(
                    sim,
                    t_start_us=float(st.session_state["t_start_us"]),
                    t_end_us=float(st.session_state["t_end_window_us"]),
                )
            )
            log_line("Baselines plotted.")

        if checks["vmax"] or checks["power"]:
            vin_list = [float(x.strip()) for x in st.session_state["vin_list"].split(",") if x.strip()]
            results: Dict[float, SimOut] = {}
            progress = st.progress(0.0)
            for idx, v in enumerate(vin_list):
                cfg = dict(config)
                cfg["Vin"] = v
                results[v] = run_sim_cached(cfg, to_dict(resist), to_dict(circuit))
                progress.progress((idx + 1) / len(vin_list))
                log_line(f"[analysis sweep] Vin={v}")
            if checks["vmax"]:
                figs += capture_new_figures(
                    lambda: analysis.plot_Vmax_vs_Vin(
                        vin_list,
                        results,
                        t_start_us=float(st.session_state["t_start_us"]),
                        t_end_us=float(st.session_state["t_end_window_us"]),
                    )
                )
            if checks["power"]:
                example_v = float(st.session_state["example_vin"])
                figs += capture_new_figures(
                    lambda: analysis.plot_power_time_and_peaks_vs_Vin(
                        vin_list,
                        results,
                        example_vin=example_v,
                        t_start_us=float(st.session_state["t_start_us"]),
                        t_end_us=float(st.session_state["t_end_window_us"]),
                    )
                )

        if checks["c_overlay"]:
            C_vals = [float(x.strip()) for x in st.session_state["c_list"].split(",") if x.strip()]
            figs += capture_new_figures(
                lambda: analysis.plot_capacitance_effect_on_power(
                    vin=float(st.session_state["vin"]),
                    C_values_pF=C_vals,
                    resist_params=resist,
                    circuit_params=circuit,
                    t_end=float(st.session_state["t_end_us"]) * 1e-6,
                    dt=float(st.session_state["dt_ns"]) * 1e-9,
                    t_start_us=float(st.session_state["t_start_us"]),
                    t_end_us=float(st.session_state["t_end_window_us"]),
                )
            )
            log_line("Capacitance overlay plotted.")

        if checks["freq3d"]:
            C_vals = [float(x.strip()) for x in st.session_state["c_list"].split(",") if x.strip()]
            R_vals = [float(x.strip()) for x in st.session_state["r_list"].split(",") if x.strip()]
            figs += capture_new_figures(
                lambda: analysis.plot_frequency_3d_vs_C_and_Rload(
                    vin=float(st.session_state["vin"]),
                    C_values_pF=C_vals,
                    Rload_values_kohm=R_vals,
                    resist_params=resist,
                    circuit_params=circuit,
                    t_end=float(st.session_state["t_end_us"]) * 1e-6,
                    dt=float(st.session_state["dt_ns"]) * 1e-9,
                    threshold_A=float(st.session_state.get("threshold_A", 1e-3)),
                )
            )
            log_line("Frequency 3D plotted.")

        if checks["r_ins"]:
            sim = run_single_simulation()
            figs += capture_new_figures(
                lambda: analysis.plot_R_insulating_vs_time(
                    sim,
                    t_start_us=float(st.session_state["t_start_us"]),
                    t_end_us=float(st.session_state["t_end_window_us"]),
                )
            )
            log_line("R_ins vs time plotted.")

        render_figures(figs)
        update_status("Completed", "PI analyses", "Analyses done.")


def render_sweeps_tab():
    st.header("Sweeps")
    st.markdown("### 1D Sweep")
    param_options = [
        "Vin",
        "R_series_kohm",
        "C_par_pF",
        "T_base_K",
        "Cth_factor",
        "noise_strength",
        "couple_factor",
        "Rm_factor",
    ]
    col1, col2, col3, col4 = st.columns(4)
    param = col1.selectbox("Parameter", param_options)
    start = col2.number_input("Start", value=10.0)
    stop = col3.number_input("Stop", value=16.0)
    step = col4.number_input("Step", value=0.5, min_value=0.001)
    if st.button("Run 1D sweep"):
        reset_logs()
        st.session_state["cancel_requested"] = False
        update_status("Running", "1D sweep", f"{param} sweep")
        vals = np.arange(start, stop + 0.5 * step, step, dtype=float).tolist()
        config = current_sim_config()
        resist = build_resist_params_from_state()
        circuit = build_circuit_params_from_state(config["lattice_shape"][0], config["lattice_shape"][1])
        progress_placeholder = st.empty()
        table_placeholder = st.empty()
        plot_placeholder = st.empty()
        df = sweep_parameter(
            config,
            resist,
            circuit,
            param,
            vals,
            float(st.session_state["t_start_us"]),
            float(st.session_state["t_end_window_us"]),
            float(st.session_state.get("threshold_A", 1e-3)),
            progress_placeholder,
            table_placeholder,
            plot_placeholder,
        )
        st.dataframe(df)
        update_status("Completed", "1D sweep", f"Completed {len(df)} points")

    st.markdown("---")
    st.markdown("### 2D Sweep")
    param_a = st.selectbox("Parameter A", param_options, index=0, key="param_a")
    param_b = st.selectbox("Parameter B", param_options, index=1, key="param_b")
    cols = st.columns(4)
    a_start = cols[0].number_input("A start", value=10.0, key="a_start")
    a_stop = cols[1].number_input("A stop", value=16.0, key="a_stop")
    a_step = cols[2].number_input("A step", value=0.5, key="a_step")
    b_start = cols[0].number_input("B start", value=100.0, key="b_start")
    b_stop = cols[1].number_input("B stop", value=200.0, key="b_stop")
    b_step = cols[2].number_input("B step", value=25.0, key="b_step")
    if st.button("Run 2D sweep"):
        reset_logs()
        st.session_state["cancel_requested"] = False
        update_status("Running", "2D sweep", f"{param_a} vs {param_b}")
        vals_a = np.arange(a_start, a_stop + 0.5 * a_step, a_step, dtype=float).tolist()
        vals_b = np.arange(b_start, b_stop + 0.5 * b_step, b_step, dtype=float).tolist()
        config = current_sim_config()
        resist = build_resist_params_from_state()
        circuit = build_circuit_params_from_state(config["lattice_shape"][0], config["lattice_shape"][1])
        progress = st.progress(0.0)
        rows: List[Dict[str, Any]] = []
        total = len(vals_a) * len(vals_b)
        count = 0
        for va in vals_a:
            for vb in vals_b:
                if st.session_state.get("cancel_requested", False):
                    log_line("2D sweep cancelled by user.")
                    break
                cfg = dict(config)
                res = dataclasses.replace(resist)
                circ = dataclasses.replace(circuit)
                for name, value in [(param_a, va), (param_b, vb)]:
                    if name == "Vin":
                        cfg["Vin"] = value
                    elif hasattr(circ, name):
                        setattr(circ, name, value)
                    elif hasattr(res, name):
                        setattr(res, name, value)
                sim = run_sim_cached(cfg, to_dict(res), to_dict(circ))
                metrics = compute_metrics(
                    sim,
                    float(st.session_state["t_start_us"]),
                    float(st.session_state["t_end_window_us"]),
                    float(st.session_state.get("threshold_A", 1e-3)),
                )
                rows.append(
                    {
                        "param_a": va,
                        "param_b": vb,
                        "oscillatory": metrics["oscillatory"],
                        "frequency_MHz": metrics["frequency_MHz"],
                    }
                )
                count += 1
                progress.progress(count / total)
                log_line(f"[2D] {count}/{total}: {param_a}={va}, {param_b}={vb}")
            if st.session_state.get("cancel_requested", False):
                break
        df = pd.DataFrame(rows)
        st.dataframe(df)
        if not df.empty:
            pivot = df.pivot(index="param_a", columns="param_b", values="frequency_MHz")
            fig, ax = plt.subplots(figsize=(6, 4))
            c = ax.imshow(pivot, aspect="auto", origin="lower")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v:.2g}" for v in pivot.columns], rotation=45)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:.2g}" for v in pivot.index])
            ax.set_xlabel(param_b)
            ax.set_ylabel(param_a)
            fig.colorbar(c, ax=ax, label="Frequency (MHz)")
            st.pyplot(fig)
            plt.close(fig)
        st.session_state["last_sweep_df"] = df
        update_status("Completed", "2D sweep", f"Completed {len(df)} points")


def render_domain_tab():
    st.header("Oscillation Domain Finder")
    param_options = [
        "Vin",
        "R_series_kohm",
        "C_par_pF",
        "T_base_K",
        "Cth_factor",
        "noise_strength",
        "couple_factor",
    ]
    param = st.selectbox("Parameter", param_options, key="domain_param")
    start = st.number_input("Start", value=0.0, key="domain_start")
    stop = st.number_input("Stop", value=20.0, key="domain_stop")
    step = st.number_input("Step", value=0.5, min_value=0.001, key="domain_step")
    fine_step = st.number_input("Fine step (optional)", value=0.05, min_value=0.001, key="domain_fine_step")
    if st.button("Find oscillatory band"):
        reset_logs()
        st.session_state["cancel_requested"] = False
        update_status("Running", "Domain finder", "Coarse sweep")
        vals = np.arange(start, stop + 0.5 * step, step, dtype=float).tolist()
        config = current_sim_config()
        resist = build_resist_params_from_state()
        circuit = build_circuit_params_from_state(config["lattice_shape"][0], config["lattice_shape"][1])
        progress_placeholder = st.empty()
        table_placeholder = st.empty()
        vmin, vmax, df = domain_find(
            param,
            vals,
            config,
            resist,
            circuit,
            float(st.session_state["t_start_us"]),
            float(st.session_state["t_end_window_us"]),
            float(st.session_state.get("threshold_A", 1e-3)),
            progress_placeholder,
            table_placeholder,
        )
        st.dataframe(df)
        if vmin is None or vmax is None:
            st.warning("No oscillatory region found in coarse sweep.")
            update_status("Completed", "Domain finder", "No oscillatory band found")
        else:
            st.success(f"Coarse oscillatory band: {vmin:.3g} – {vmax:.3g}")
            log_line(f"[domain] coarse band {vmin} – {vmax}")
            update_status("Running", "Domain finder", "Fine sweep")
            fine_vals = np.arange(vmin, vmax + 0.5 * fine_step, fine_step, dtype=float).tolist()
            progress_fine = st.progress(0.0)
            table_fine = st.empty()
            vmin_f, vmax_f, df_f = domain_find(
                param,
                fine_vals,
                config,
                resist,
                circuit,
                float(st.session_state["t_start_us"]),
                float(st.session_state["t_end_window_us"]),
                float(st.session_state.get("threshold_A", 1e-3)),
                progress_fine,
                table_fine,
            )
            st.dataframe(df_f)
            st.success(f"Fine band: {vmin_f} – {vmax_f}")
            update_status("Completed", "Domain finder", f"Band {vmin_f} – {vmax_f}")


def render_export_tab():
    st.header("Export")
    if st.session_state["last_sim"]:
        sim = st.session_state["last_sim"]
        buf = io.StringIO()
        series_keys = ["V_node", "I_load", "I_vo2", "T_K", "R_vo2", "g"]
        series_data = {k: series_first(sim[k]) for k in series_keys}
        header = ["time_s"] + series_keys
        buf.write(",".join(header) + "\n")
        n = len(sim["time_s"])
        for i in range(n):
            row = [sim["time_s"][i]] + [series_data[k][i] for k in series_keys]
            buf.write(",".join(str(x) for x in row) + "\n")
        st.download_button("Download last simulation CSV", data=buf.getvalue(), file_name="simulation.csv")
    else:
        st.info("Run a simulation to enable export.")

    if st.session_state["last_sweep_df"] is not None:
        df = st.session_state["last_sweep_df"]
        st.download_button(
            "Download last sweep CSV",
            data=df.to_csv(index=False),
            file_name="sweep.csv",
            mime="text/csv",
        )

    if st.session_state["last_config"]:
        st.download_button(
            "Download config JSON",
            data=json.dumps(st.session_state["last_config"], indent=2),
            file_name="config.json",
            mime="application/json",
        )


# ---------------------------------------------------------------------------
# App entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    sidebar_inputs()
    tab_run, tab_pi, tab_sweeps, tab_domain, tab_export = st.tabs(
        ["Run", "PI Analyses", "Sweeps", "Oscillation Domain Finder", "Export"]
    )
    with tab_run:
        render_run_tab()
    with tab_pi:
        render_pi_tab()
    with tab_sweeps:
        render_sweeps_tab()
    with tab_domain:
        render_domain_tab()
    with tab_export:
        render_export_tab()


if __name__ == "__main__":
    main()
