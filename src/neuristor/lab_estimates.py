"""Estimate environmental thermal conductance from numerical TIA waveforms.

The estimate deliberately uses the closest *settled, non-oscillating* trace
below oscillation onset. Its baseline-corrected resistance is mapped to a
temperature through the specimen's fitted heating branch. Only after the
resistance drift confirms that the trace is quasi-steady do we set ``dT/dt``
to zero and evaluate ``S_e = P / (T - T0)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .model import YuanhangResistParams


@dataclass(frozen=True)
class EnvironmentalConductanceEstimate:
    """Numerical evidence and uncertainty for one conductance estimate."""

    result: pd.DataFrame
    analyzed_trace: pd.DataFrame
    bootstrap: pd.DataFrame
    pre_switch: pd.Series
    post_switch: pd.Series


def heating_branch_resistance_ohm(
    temperature_K: float | np.ndarray,
    params: YuanhangResistParams,
) -> np.ndarray:
    """Evaluate the fitted major heating branch in double precision."""

    temperature = np.asarray(temperature_K, dtype=float)
    fraction = 0.5 + 0.5 * np.tanh(
        float(params.beta)
        * (0.5 * float(params.w_eff) + float(params.Tc_K) - temperature)
    )
    return (
        float(params.Rm)
        + float(params.R0)
        * np.exp(float(params.Ea_over_k) / np.maximum(temperature, 1e-12))
        * fraction
    )


def infer_heating_temperature_K(
    resistance_ohm: float,
    params: YuanhangResistParams,
) -> float:
    """Invert the monotonic fitted heating branch for one resistance value."""

    lower = float(params.T_min_K)
    upper = float(params.T_max_K)
    resistance = float(resistance_ohm)
    cold = float(heating_branch_resistance_ohm(lower, params))
    hot = float(heating_branch_resistance_ohm(upper, params))
    if not hot <= resistance <= cold:
        raise ValueError(
            f"Resistance {resistance:.6g} ohm lies outside the fitted heating branch "
            f"[{hot:.6g}, {cold:.6g}] ohm"
        )
    return float(
        brentq(
            lambda temperature: float(heating_branch_resistance_ohm(temperature, params))
            - resistance,
            lower,
            upper,
        )
    )


def _params_from_bootstrap_row(
    row: pd.Series,
    template: YuanhangResistParams,
) -> YuanhangResistParams:
    """Translate one resistance-fit bootstrap record to model parameters."""

    return YuanhangResistParams(
        R0=float(row["R0_ohm"]),
        Ea_over_k=float(row["Ea_over_k_K"]),
        Rm0=float(row["Rm_ohm"]),
        Rm_factor=1.0,
        w=float(row["w_K"]),
        Tc_K=float(row["Tc_K"]),
        beta=float(row["beta_per_K"]),
        gamma=float(template.gamma),
        width_factor=1.0,
        T_min_K=float(template.T_min_K),
        T_max_K=float(template.T_max_K),
        reversal_threshold_K=float(template.reversal_threshold_K),
    )


def _circular_block_indices(
    size: int,
    *,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return paired circular-block indices for autocorrelated waveform samples."""

    blocks = int(np.ceil(size / block_size))
    starts = rng.integers(0, size, size=blocks)
    return np.concatenate(
        [(start + np.arange(block_size)) % size for start in starts]
    )[:size]


def estimate_environmental_conductance(
    traces: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    resistance: YuanhangResistParams,
    ambient_temperature_K: float,
    ambient_interval_K: tuple[float, float] | None = None,
    resistance_bootstrap: pd.DataFrame | None = None,
    baseline_window_ns: tuple[float, float] = (-200.0, -50.0),
    steady_window_ns: tuple[float, float] = (100.0, 250.0),
    max_resistance_drift_fraction: float = 0.01,
    bootstrap_samples: int = 1000,
    block_size: int = 10,
    seed: int = 20260817,
) -> EnvironmentalConductanceEstimate:
    """Estimate ``S_e`` from the last settled trace before oscillation onset.

    Current and voltage are offset by their pre-pulse medians. The selected
    trace must precede the first coherently oscillating trace and its effective
    resistance may drift by at most ``max_resistance_drift_fraction`` across
    the settled window. The returned percentile interval propagates paired
    waveform block resampling, the supplied R(T)-fit bootstrap, and the stated
    ambient-temperature interval. It remains conditional on the TIA and R(T)
    measurements describing the same device.
    """

    required_summary = {"source_file", "current_step_uA", "oscillation_detected"}
    missing_summary = sorted(required_summary - set(summary.columns))
    if missing_summary:
        raise ValueError(f"Lab summary is missing columns: {', '.join(missing_summary)}")
    required_trace = {
        "source_file",
        "time_ns",
        "input_current_uA",
        "output_voltage_mV",
    }
    missing_trace = sorted(required_trace - set(traces.columns))
    if missing_trace:
        raise ValueError(f"Lab traces are missing columns: {', '.join(missing_trace)}")

    ordered = summary.sort_values("current_step_uA").reset_index(drop=True)
    oscillating = np.flatnonzero(ordered["oscillation_detected"].astype(bool).to_numpy())
    if oscillating.size == 0 or int(oscillating[0]) == 0:
        raise ValueError("Could not bracket oscillation onset with a preceding stable trace")
    onset_index = int(oscillating[0])
    pre_switch = ordered.iloc[onset_index - 1]
    post_switch = ordered.iloc[onset_index]
    source_file = str(pre_switch["source_file"])

    selected = traces.loc[traces["source_file"].astype(str) == source_file].copy()
    if selected.empty:
        raise ValueError(f"Selected trace {source_file} is missing from numerical waveforms")
    selected = selected.sort_values("time_ns").reset_index(drop=True)
    time_ns = selected["time_ns"].to_numpy(dtype=float)
    current_uA = selected["input_current_uA"].to_numpy(dtype=float)
    voltage_mV = selected["output_voltage_mV"].to_numpy(dtype=float)
    baseline_mask = (time_ns >= baseline_window_ns[0]) & (time_ns <= baseline_window_ns[1])
    steady_mask = (time_ns >= steady_window_ns[0]) & (time_ns <= steady_window_ns[1])
    if int(np.sum(baseline_mask)) < 20 or int(np.sum(steady_mask)) < 20:
        raise ValueError("Selected trace does not span the baseline and settled windows")

    current_baseline_uA = float(np.median(current_uA[baseline_mask]))
    voltage_baseline_mV = float(np.median(voltage_mV[baseline_mask]))
    selected["current_corrected_uA"] = current_uA - current_baseline_uA
    selected["voltage_corrected_mV"] = voltage_mV - voltage_baseline_mV
    corrected_current = selected["current_corrected_uA"].to_numpy(dtype=float)
    corrected_voltage = selected["voltage_corrected_mV"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        selected["effective_resistance_ohm"] = np.where(
            corrected_current > 0.0,
            1000.0 * corrected_voltage / corrected_current,
            np.nan,
        )
    selected["corrected_power_uW"] = corrected_current * corrected_voltage * 1e-3

    steady = selected.loc[steady_mask]
    valid = steady["effective_resistance_ohm"].notna()
    if int(np.sum(valid)) < 20:
        raise ValueError("Too few valid resistance samples in the settled window")
    steady_time = steady.loc[valid, "time_ns"].to_numpy(dtype=float)
    steady_resistance = steady.loc[valid, "effective_resistance_ohm"].to_numpy(dtype=float)
    steady_power = steady.loc[valid, "corrected_power_uW"].to_numpy(dtype=float)
    selected_current_uA = float(np.median(steady.loc[valid, "current_corrected_uA"]))
    selected_voltage_mV = float(np.median(steady.loc[valid, "voltage_corrected_mV"]))
    selected_resistance_ohm = float(np.median(steady_resistance))
    selected_power_uW = float(np.median(steady_power))
    window_duration_ns = float(steady_window_ns[1] - steady_window_ns[0])
    resistance_slope_ohm_per_ns = float(np.polyfit(steady_time, steady_resistance, 1)[0])
    power_slope_uW_per_ns = float(np.polyfit(steady_time, steady_power, 1)[0])
    resistance_drift_fraction = (
        resistance_slope_ohm_per_ns * window_duration_ns / selected_resistance_ohm
    )
    power_drift_fraction = power_slope_uW_per_ns * window_duration_ns / selected_power_uW
    if abs(resistance_drift_fraction) > float(max_resistance_drift_fraction):
        raise ValueError(
            "Closest non-oscillating trace is not quasi-steady: "
            f"resistance drift is {100.0 * resistance_drift_fraction:.3g}%"
        )

    inferred_temperature_K = infer_heating_temperature_K(selected_resistance_ohm, resistance)
    delta_temperature_K = inferred_temperature_K - float(ambient_temperature_K)
    if delta_temperature_K <= 0.0:
        raise ValueError("Inferred device temperature must exceed ambient temperature")
    conductance_uW_per_K = selected_power_uW / delta_temperature_K

    ambient_bounds = ambient_interval_K or (
        float(ambient_temperature_K),
        float(ambient_temperature_K),
    )
    ambient_low, ambient_high = sorted(map(float, ambient_bounds))
    if not ambient_low <= float(ambient_temperature_K) <= ambient_high:
        raise ValueError("Ambient estimate must lie inside ambient_interval_K")
    if bootstrap_samples < 1 or block_size < 1:
        raise ValueError("bootstrap_samples and block_size must be positive")

    required_bootstrap = {
        "R0_ohm",
        "Ea_over_k_K",
        "Rm_ohm",
        "w_K",
        "Tc_K",
        "beta_per_K",
    }
    if resistance_bootstrap is not None:
        missing_bootstrap = sorted(required_bootstrap - set(resistance_bootstrap.columns))
        if missing_bootstrap:
            raise ValueError(
                f"Resistance bootstrap is missing columns: {', '.join(missing_bootstrap)}"
            )

    baseline_current = current_uA[baseline_mask]
    baseline_voltage = voltage_mV[baseline_mask]
    plateau_current = current_uA[steady_mask]
    plateau_voltage = voltage_mV[steady_mask]
    rng = np.random.default_rng(seed)
    bootstrap_rows: list[dict[str, float]] = []
    for sample_index in range(int(bootstrap_samples)):
        baseline_indices = _circular_block_indices(
            len(baseline_current), block_size=block_size, rng=rng
        )
        plateau_indices = _circular_block_indices(
            len(plateau_current), block_size=block_size, rng=rng
        )
        sampled_current_baseline = float(np.median(baseline_current[baseline_indices]))
        sampled_voltage_baseline = float(np.median(baseline_voltage[baseline_indices]))
        sampled_current = plateau_current[plateau_indices] - sampled_current_baseline
        sampled_voltage = plateau_voltage[plateau_indices] - sampled_voltage_baseline
        positive = sampled_current > 0.0
        sampled_resistance = float(
            np.median(1000.0 * sampled_voltage[positive] / sampled_current[positive])
        )
        sampled_power = float(
            np.median(sampled_current[positive] * sampled_voltage[positive] * 1e-3)
        )
        if resistance_bootstrap is None or resistance_bootstrap.empty:
            sampled_params = resistance
        else:
            bootstrap_row = resistance_bootstrap.iloc[
                int(rng.integers(0, len(resistance_bootstrap)))
            ]
            sampled_params = _params_from_bootstrap_row(bootstrap_row, resistance)
        try:
            sampled_temperature = infer_heating_temperature_K(
                sampled_resistance, sampled_params
            )
        except ValueError:
            continue
        sampled_ambient = float(rng.uniform(ambient_low, ambient_high))
        sampled_delta = sampled_temperature - sampled_ambient
        if sampled_delta <= 0.0:
            continue
        bootstrap_rows.append(
            {
                "bootstrap_sample": float(sample_index),
                "effective_resistance_ohm": sampled_resistance,
                "power_uW": sampled_power,
                "inferred_temperature_K": sampled_temperature,
                "ambient_temperature_K": sampled_ambient,
                "delta_temperature_K": sampled_delta,
                "S_e_uW_per_K": sampled_power / sampled_delta,
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    if len(bootstrap) < max(20, int(0.8 * bootstrap_samples)):
        raise ValueError("Too many conductance bootstrap samples were invalid")

    lower_temperature, upper_temperature = np.percentile(
        bootstrap["inferred_temperature_K"], [2.5, 97.5]
    )
    lower_conductance, upper_conductance = np.percentile(
        bootstrap["S_e_uW_per_K"], [2.5, 97.5]
    )

    raw_current = steady.loc[valid, "input_current_uA"].to_numpy(dtype=float)
    raw_voltage = steady.loc[valid, "output_voltage_mV"].to_numpy(dtype=float)
    raw_resistance = float(np.median(1000.0 * raw_voltage / raw_current))
    raw_power = float(np.median(raw_current * raw_voltage * 1e-3))
    raw_temperature = infer_heating_temperature_K(raw_resistance, resistance)
    raw_conductance = raw_power / (raw_temperature - float(ambient_temperature_K))

    result = pd.DataFrame(
        [
            {
                "selected_trace": source_file,
                "first_oscillating_trace": str(post_switch["source_file"]),
                "baseline_start_ns": float(baseline_window_ns[0]),
                "baseline_stop_ns": float(baseline_window_ns[1]),
                "steady_start_ns": float(steady_window_ns[0]),
                "steady_stop_ns": float(steady_window_ns[1]),
                "current_baseline_uA": current_baseline_uA,
                "voltage_baseline_mV": voltage_baseline_mV,
                "current_corrected_uA": selected_current_uA,
                "voltage_corrected_mV": selected_voltage_mV,
                "effective_resistance_ohm": selected_resistance_ohm,
                "power_uW": selected_power_uW,
                "resistance_drift_fraction": resistance_drift_fraction,
                "power_drift_fraction": power_drift_fraction,
                "inferred_temperature_K": inferred_temperature_K,
                "inferred_temperature_ci95_lower_K": float(lower_temperature),
                "inferred_temperature_ci95_upper_K": float(upper_temperature),
                "ambient_temperature_K": float(ambient_temperature_K),
                "ambient_ci_lower_K": ambient_low,
                "ambient_ci_upper_K": ambient_high,
                "delta_temperature_K": delta_temperature_K,
                "S_e_uW_per_K": conductance_uW_per_K,
                "S_e_mW_per_K": conductance_uW_per_K * 1e-3,
                "S_e_ci95_lower_mW_per_K": float(lower_conductance) * 1e-3,
                "S_e_ci95_upper_mW_per_K": float(upper_conductance) * 1e-3,
                "S_e_without_baseline_correction_mW_per_K": raw_conductance * 1e-3,
                "bootstrap_samples": float(len(bootstrap)),
            }
        ]
    )
    return EnvironmentalConductanceEstimate(
        result=result,
        analyzed_trace=selected,
        bootstrap=bootstrap,
        pre_switch=pre_switch,
        post_switch=post_switch,
    )
