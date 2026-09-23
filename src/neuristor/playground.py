"""Editable Streamlit laboratory replay; physics and saved runs live in workflows."""
from pathlib import Path
import copy
import dataclasses
import json

import pandas as pd
import streamlit as st

from neuristor.config import load_toml, deep_get, deep_set
from neuristor.joint_inference import NAMES, joint_model
from neuristor.replay_plots import comparison_figure
from neuristor.workflows import _current_params_from_config, run_lab_replay

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "experiments/current/specimen_lab_replay.toml"
st.set_page_config(page_title="VO₂ Simulation Lab", layout="wide")
st.title("VO₂ Simulation Lab")
st.caption("One shared set of device parameters, replayed against every measured current history.")


def presets():
    """Find saved joint candidates without executing a simulation."""
    options = {"Original specimen parameters": None}
    for table in sorted((ROOT / "public_jobs").glob("*/parameters.csv"), reverse=True):
        frame = pd.read_csv(table)
        if set(NAMES).issubset(frame.columns) and "candidate" in frame:
            for _, row in frame.iterrows():
                if row.candidate in ("static_preserving", "dynamic_diagnostic"):
                    options[f"{table.parent.name[:15]} · {row.candidate}"] = row
    return options


if "lab_config" not in st.session_state:
    st.session_state.lab_config = load_toml(TEMPLATE)
choices = presets()
with st.sidebar:
    st.header("Starting parameters")
    choice = st.selectbox("Saved fit", list(choices))
    if st.button("Load parameters"):
        config = load_toml(TEMPLATE)
        row = choices[choice]
        if row is not None:
            base, *_ = _current_params_from_config(config)
            p = joint_model(row[list(NAMES)].to_numpy(float), base, config["time"]["dt_ns"])
            config["resistance"]["parameters"] = dataclasses.asdict(p.resist_params)
            config["electrical"]["C_pF"] = p.C_F * 1e12
            config["thermal"] = dict(C_th_pJ_per_K=p.C_th_J_per_K*1e12,
                                     S_e_mW_per_K=p.S_e_W_per_K*1e3, T0_K=p.T0_K)
            config["initial"]["temperature_K"] = p.T_init_K
        st.session_state.lab_config = config
        for key in list(st.session_state):
            if key.startswith("param:"):
                st.session_state[key] = float(deep_get(config, key.removeprefix("param:")))
        st.rerun()
    saved = sorted((ROOT / "runs").glob("*/comparison.html"), reverse=True)
    if saved:
        previous = st.selectbox("Saved comparison", saved, format_func=lambda p: p.parent.name)
        if st.button("Open saved results"):
            st.session_state.lab_result = str(previous.parent)
    st.caption("Measured voltage is compared as exported. Its relation to device terminal voltage is still unconfirmed without the amplifier circuit.")

config = copy.deepcopy(st.session_state.lab_config)

def number(label, path, minimum=None):
    value = st.number_input(label, value=None if "param:"+path in st.session_state else float(deep_get(config, path)), min_value=minimum,
                            format="%.8g", key="param:"+path)
    if value is None:
        st.error(f"Enter a value for {label}")
        st.stop()
    deep_set(config, path, value)

with st.form("parameters"):
    resistance, other = st.columns(2)
    with resistance:
        st.subheader("Resistance law")
        for label, name, minimum in [
            ("R₀ (Ω), Arrhenius prefactor", "R0", 0.0),
            ("Activation energy / k (K)", "Ea_over_k", 0.000001),
            ("Metallic resistance Rm₀ (Ω)", "Rm0", 0.000001),
            ("Transition temperature Tc (K)", "Tc_K", 1.0),
            ("Hysteresis width w (K)", "w", 0.000001),
            ("Transition slope β (K⁻¹)", "beta", 0.000001),
            ("Minor-loop shape γ", "gamma", 0.000001)]:
            number(label, "resistance.parameters."+name, minimum)
    with other:
        st.subheader("Electrical and thermal parameters")
        for label, path, minimum in [
            ("Capacitance C (pF); zero gives algebraic limit", "electrical.C_pF", 0.0),
            ("Heat capacity Cth (pJ/K)", "thermal.C_th_pJ_per_K", 0.000001),
            ("Thermal conductance Se (mW/K)", "thermal.S_e_mW_per_K", 0.000001),
            ("Ambient temperature T₀ (K)", "thermal.T0_K", 1.0)]:
            number(label, path, minimum)
        tie_initial = st.checkbox("Start at ambient temperature", value=True)
        number("Initial temperature (K), used if unticked", "initial.temperature_K", 1.0)
        number("Initial voltage (V)", "initial.voltage_V")
        number("Integration timestep (ns)", "time.dt_ns", 0.001)
        st.caption("0.025 ns is an exploratory setting. Repeat at 0.0125 and 0.00625 ns before interpreting a promising fit. Smaller steps cost more time.")
    with st.expander("Other resistance settings"):
        for label, name, minimum in [
            ("Metallic resistance multiplier", "Rm_factor", 0.000001),
            ("Width multiplier", "width_factor", 0.000001),
            ("Calibration minimum temperature (K)", "T_min_K", 1.0),
            ("Calibration maximum temperature (K)", "T_max_K", 1.0),
            ("Reversal detection threshold (K)", "reversal_threshold_K", 0.000001)]:
            number(label, "resistance.parameters."+name, minimum)
        config["resistance"]["start_branch"] = st.selectbox("Initial branch", ["insulator", "metal"])
    config["replay"]["generate_gifs"] = st.checkbox("Generate GIF for every current", value=True)
    submitted = st.form_submit_button("Run all measured currents", type="primary")
if submitted:
    if tie_initial:
        config["initial"]["temperature_K"] = config["thermal"]["T0_K"]
    st.session_state.lab_config = config
    try:
        with st.spinner("Simulating all current histories and saving comparisons…"):
            bundle = run_lab_replay(config, output_root=ROOT / "runs")
        st.session_state.lab_result = str(bundle.root)
    except Exception as exc:
        st.error(str(exc))

if "lab_result" in st.session_state:
    run = Path(st.session_state.lab_result)
    st.subheader("Saved run results")
    st.caption(f"{run.name} · These curves use the saved parameters; press Run to apply edits.")
    metrics = json.loads((run / "metrics.json").read_text())
    columns = st.columns(3)
    columns[0].metric("Sustained oscillations recovered", f"{metrics['recovered_oscillators']} / {metrics['measured_oscillators']}")
    columns[1].metric("Late mean voltage RMSE", f"{metrics['late_mean_RMSE_mV']:.1f} mV")
    columns[2].metric("False oscillation predictions", metrics["false_positives"])
    if metrics["outside_calibration"]:
        st.warning("Some simulated temperatures leave the measured R(T) calibration range.")
    traces = pd.read_csv(run / "traces.csv")
    summary = pd.read_csv(run / "summary.csv")
    drives = summary.source_mV.tolist()
    key = "current:"+run.name
    selected = st.session_state.get(key, drives[0])
    frame = traces[traces.source_mV == selected]
    st.plotly_chart(comparison_figure(frame), use_container_width=True, theme=None)
    selected = st.select_slider("Measured current (µA)", options=drives, key=key,
        format_func=lambda d: f"{summary.loc[summary.source_mV == d, 'current_step_uA'].iloc[0]:.1f}")
    gif = run / "animations" / f"{selected:g}mV.gif"
    if gif.exists() and st.checkbox("Show animated time cursor"):
        st.image(str(gif))
    with st.expander("Downloads and all-current metrics"):
        for name, label, mime in [("comparison.html", "Interactive comparison with current slider", "text/html"),
            ("animations.zip", "All current GIFs", "application/zip"),
            ("recipe.toml", "Parameters / reproducible recipe", "application/toml"),
            ("summary.csv", "All-current metrics", "text/csv")]:
            path = run / name
            if path.exists():
                st.download_button(label, path.read_bytes(), file_name=name, mime=mime)
        if gif.exists():
            st.download_button("Selected current GIF", gif.read_bytes(), file_name=gif.name, mime="image/gif")
        st.dataframe(summary, use_container_width=True)
else:
    st.info("Edit the parameters, then Run to compare all measured currents. Previous comparisons can be opened in the sidebar.")
