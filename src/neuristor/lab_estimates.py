"""Identifiability-aware estimates from digitized current-drive traces.

These calculations intentionally keep direct estimates separate from assumed
quantities.  Screenshot data can estimate the cold electrical capacitance, but
thermal capacitance still requires an independently measured recovery time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabParameterEstimates:
    """Tables and switching bracket produced by the lab-estimation workflow."""

    electrical_capacitance: pd.DataFrame
    thermal_conductance: pd.DataFrame
    thermal_capacitance: pd.DataFrame
    effective_resistance: pd.DataFrame
    pre_switch: pd.Series
    post_switch: pd.Series


def estimate_lab_parameters(
    summary: pd.DataFrame,
    *,
    transition_temperature_K: float,
    ambient_temperatures_K: Sequence[float],
    thermal_times_ns: Sequence[float],
    ripple_threshold_mV: float = 20.0,
) -> LabParameterEstimates:
    """Estimate C, S_e scenarios, C_th scenarios, and plateau V/I.

    ``C = I/(dV/dt)`` uses pre-switch cold edges. ``S_e = P/delta_T``
    is evaluated for every supplied ambient temperature. ``C_th = S_e*tau``
    remains a scenario until a thermal recovery time is independently known.
    """

    required = {
        "frame_index",
        "current_inferred_uA",
        "v_plateau_mean_mV",
        "v_plateau_vpp_mV",
        "v_slope_0_30_mV_per_ns",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Lab summary is missing columns: {', '.join(missing)}")
    ordered = summary.sort_values("current_inferred_uA").reset_index(drop=True)
    switched = np.flatnonzero(ordered["v_plateau_vpp_mV"].to_numpy(dtype=float) >= ripple_threshold_mV)
    if switched.size == 0 or int(switched[0]) == 0:
        raise ValueError("Could not bracket switching onset from plateau ripple")
    high_index = int(switched[0])
    pre_switch = ordered.iloc[high_index - 1]
    post_switch = ordered.iloc[high_index]

    capacitance = ordered.copy()
    capacitance["C_slope_pF"] = capacitance["current_inferred_uA"] / capacitance["v_slope_0_30_mV_per_ns"]
    valid_capacitance = (
        (capacitance["current_inferred_uA"] > 0.0)
        & (capacitance["current_inferred_uA"] <= float(pre_switch["current_inferred_uA"]))
        & (capacitance["v_slope_0_30_mV_per_ns"] > 1.0)
        & (capacitance["v_plateau_vpp_mV"] < ripple_threshold_mV)
        & np.isfinite(capacitance["C_slope_pF"])
    )
    capacitance = capacitance.loc[
        valid_capacitance,
        ["frame_index", "current_inferred_uA", "v_slope_0_30_mV_per_ns", "C_slope_pF"],
    ].reset_index(drop=True)
    if capacitance.empty:
        raise ValueError("No valid pre-switch slope points were found")

    current_uA = float(pre_switch["current_inferred_uA"])
    voltage_mV = float(pre_switch["v_plateau_mean_mV"])
    switching_power_mW = current_uA * voltage_mV * 1e-6
    conductance_rows = []
    for ambient_K in ambient_temperatures_K:
        delta_temperature = float(transition_temperature_K) - float(ambient_K)
        conductance_rows.append(
            {
                "T0_K": float(ambient_K),
                "T_switch_K": float(transition_temperature_K),
                "Delta_T_K": delta_temperature,
                "I_pre_switch_uA": current_uA,
                "V_pre_switch_mV": voltage_mV,
                "P_pre_switch_mW": switching_power_mW,
                "S_e_mW_per_K": switching_power_mW / delta_temperature if delta_temperature > 0.0 else float("nan"),
            }
        )
    conductance = pd.DataFrame(conductance_rows)

    thermal_rows = []
    for item in conductance.to_dict("records"):
        for tau_ns in thermal_times_ns:
            thermal_rows.append(
                {
                    "T0_K": item["T0_K"],
                    "S_e_mW_per_K": item["S_e_mW_per_K"],
                    "assumed_tau_th_ns": float(tau_ns),
                    # mW/K * ns is numerically pJ/K.
                    "C_th_pJ_per_K": float(item["S_e_mW_per_K"]) * float(tau_ns),
                }
            )
    thermal_capacitance = pd.DataFrame(thermal_rows)

    resistance = ordered[["frame_index", "current_inferred_uA", "v_plateau_mean_mV", "v_plateau_vpp_mV"]].copy()
    resistance["R_effective_ohm"] = 1000.0 * resistance["v_plateau_mean_mV"] / resistance["current_inferred_uA"]
    return LabParameterEstimates(
        electrical_capacitance=capacitance,
        thermal_conductance=conductance,
        thermal_capacitance=thermal_capacitance,
        effective_resistance=resistance,
        pre_switch=pre_switch,
        post_switch=post_switch,
    )
