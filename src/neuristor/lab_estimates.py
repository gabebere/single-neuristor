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
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import savgol_filter

from .model import YuanhangResistParams


@dataclass(frozen=True)
class EnvironmentalConductanceEstimate:
    """Numerical evidence and uncertainty for one conductance estimate."""

    result: pd.DataFrame
    analyzed_trace: pd.DataFrame
    bootstrap: pd.DataFrame
    pre_switch: pd.Series
    post_switch: pd.Series


@dataclass(frozen=True)
class ThermalCapacitanceEstimate:
    """Thermal time-constant fit reconstructed from nonswitching heating edges."""

    result: pd.DataFrame
    trajectories: pd.DataFrame
    trace_fits: pd.DataFrame
    bootstrap: pd.DataFrame


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


def _infer_heating_temperature_array_K(
    resistance_ohm: np.ndarray,
    params: YuanhangResistParams,
) -> np.ndarray:
    """Invert the monotonic fitted heating branch for an array of resistances."""

    temperature_grid = np.linspace(
        float(params.T_min_K),
        float(params.T_max_K),
        4096,
    )
    resistance_grid = heating_branch_resistance_ohm(temperature_grid, params)
    return np.interp(
        np.asarray(resistance_ohm, dtype=float),
        resistance_grid[::-1],
        temperature_grid[::-1],
        left=np.nan,
        right=np.nan,
    )


def _thermal_temperature_from_power_K(
    time_ns: np.ndarray,
    power_mW: np.ndarray,
    *,
    C_th_pJ_per_K: float,
    S_e_mW_per_K: float,
    ambient_temperature_K: float,
) -> np.ndarray:
    """Integrate the linear thermal balance using a piecewise-linear power trace."""

    if C_th_pJ_per_K <= 0.0 or S_e_mW_per_K <= 0.0:
        raise ValueError("C_th and S_e must be positive for a thermal transient fit")
    time = np.asarray(time_ns, dtype=float)
    power = np.asarray(power_mW, dtype=float)
    if time.shape != power.shape or time.size < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("Thermal integration requires aligned, increasing time and power arrays")
    tau_ns = float(C_th_pJ_per_K) / float(S_e_mW_per_K)
    temperature_rise = np.zeros_like(power)
    for index in range(time.size - 1):
        dt_ns = float(time[index + 1] - time[index])
        one_minus_decay = -np.expm1(-dt_ns / tau_ns)
        decay = 1.0 - one_minus_decay
        power_slope = float(power[index + 1] - power[index]) / dt_ns
        temperature_rise[index + 1] = (
            decay * temperature_rise[index]
            + (
                float(power[index]) * one_minus_decay
                + power_slope * (dt_ns - tau_ns * one_minus_decay)
            )
            / float(S_e_mW_per_K)
        )
    return float(ambient_temperature_K) + temperature_rise


def _prepare_thermal_trace(
    frame: pd.DataFrame,
    *,
    electrical_capacitance_pF: float,
    baseline_window_ns: tuple[float, float],
    integration_window_ns: tuple[float, float],
    smoothing_window: int,
) -> pd.DataFrame:
    """Prepare the resistive current, resistance, and power used by the thermal fit.

    In the trace units, ``C_pF * dV_mV_per_ns/dt`` is directly in microamperes.
    This keeps the current split explicit before resistance and Joule power are
    reconstructed.
    """

    ordered = frame.sort_values("time_ns").reset_index(drop=True)
    time_ns = ordered["time_ns"].to_numpy(dtype=float)
    current_uA = ordered["input_current_uA"].to_numpy(dtype=float)
    voltage_mV = ordered["output_voltage_mV"].to_numpy(dtype=float)
    baseline = (time_ns >= baseline_window_ns[0]) & (time_ns <= baseline_window_ns[1])
    if int(np.sum(baseline)) < 20:
        raise ValueError("Thermal trace does not span the requested baseline window")
    current_corrected = current_uA - float(np.median(current_uA[baseline]))
    voltage_corrected = voltage_mV - float(np.median(voltage_mV[baseline]))
    keep = (time_ns >= integration_window_ns[0]) & (time_ns <= integration_window_ns[1])
    time_ns = time_ns[keep]
    current_corrected = current_corrected[keep]
    voltage_corrected = voltage_corrected[keep]
    if time_ns.size < smoothing_window or smoothing_window < 5 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be an odd integer supported by the trace")
    if electrical_capacitance_pF < 0.0:
        raise ValueError("electrical_capacitance_pF must be non-negative")
    current_smooth = savgol_filter(current_corrected, smoothing_window, 2)
    voltage_smooth = savgol_filter(voltage_corrected, smoothing_window, 2)
    voltage_slew_mV_per_ns = np.gradient(voltage_smooth, time_ns)
    capacitive_current_uA = float(electrical_capacitance_pF) * voltage_slew_mV_per_ns
    resistive_current_uA = current_smooth - capacitive_current_uA
    with np.errstate(divide="ignore", invalid="ignore"):
        resistance_ohm = np.where(
            resistive_current_uA > 0.0,
            1000.0 * voltage_smooth / resistive_current_uA,
            np.nan,
        )
    return pd.DataFrame(
        {
            "source_file": str(ordered["source_file"].iloc[0]),
            "nominal_drive_mV": float(ordered["nominal_drive_mV"].iloc[0]),
            "time_ns": time_ns,
            "current_corrected_uA": current_corrected,
            "voltage_corrected_mV": voltage_corrected,
            "current_smoothed_uA": current_smooth,
            "voltage_smoothed_mV": voltage_smooth,
            "voltage_slew_mV_per_ns": voltage_slew_mV_per_ns,
            "capacitive_current_uA": capacitive_current_uA,
            "resistive_current_uA": resistive_current_uA,
            "effective_resistance_ohm": resistance_ohm,
            "power_mW": resistive_current_uA * voltage_smooth * 1e-6,
        }
    )


def _fit_thermal_capacitance(
    prepared: list[pd.DataFrame],
    trace_indices: np.ndarray,
    *,
    resistance: YuanhangResistParams,
    S_e_mW_per_K: float,
    ambient_temperature_K: float,
    fit_window_ns: tuple[float, float],
    capacitance_bounds_pJ_per_K: tuple[float, float],
) -> tuple[float, float]:
    """Fit one shared thermal capacitance to selected reconstructed temperatures."""

    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for index in trace_indices:
        trace = prepared[int(index)]
        time_ns = trace["time_ns"].to_numpy(dtype=float)
        power_mW = trace["power_mW"].to_numpy(dtype=float)
        inferred_temperature = _infer_heating_temperature_array_K(
            trace["effective_resistance_ohm"].to_numpy(dtype=float),
            resistance,
        )
        fit = (
            (time_ns >= fit_window_ns[0])
            & (time_ns <= fit_window_ns[1])
            & np.isfinite(inferred_temperature)
        )
        if int(np.sum(fit)) < 10:
            raise ValueError("Too few valid inferred temperatures in the thermal fit window")
        observations.append((time_ns, power_mW, inferred_temperature, fit))

    def objective(C_th_pJ_per_K: float) -> float:
        residuals: list[np.ndarray] = []
        for time_ns, power_mW, inferred_temperature, fit in observations:
            modeled = _thermal_temperature_from_power_K(
                time_ns,
                power_mW,
                C_th_pJ_per_K=C_th_pJ_per_K,
                S_e_mW_per_K=S_e_mW_per_K,
                ambient_temperature_K=ambient_temperature_K,
            )
            residuals.append(modeled[fit] - inferred_temperature[fit])
        combined = np.concatenate(residuals)
        return float(np.sqrt(np.mean(combined**2)))

    lower, upper = sorted(map(float, capacitance_bounds_pJ_per_K))
    if lower <= 0.0 or upper <= lower:
        raise ValueError("capacitance_bounds_pJ_per_K must be positive and increasing")
    optimum = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-9},
    )
    return float(optimum.x), float(optimum.fun)


def estimate_thermal_capacitance(
    traces: pd.DataFrame,
    *,
    resistance: YuanhangResistParams,
    S_e_mW_per_K: float,
    ambient_temperature_K: float,
    electrical_capacitance_pF: float = 0.0,
    selected_drives_mV: tuple[float, ...] = (100.0, 150.0, 200.0),
    near_transition_check_mV: float | None = 250.0,
    resistance_bootstrap: pd.DataFrame | None = None,
    conductance_bootstrap: pd.DataFrame | None = None,
    baseline_window_ns: tuple[float, float] = (-200.0, -50.0),
    integration_window_ns: tuple[float, float] = (-50.0, 80.0),
    fit_window_ns: tuple[float, float] = (15.0, 35.0),
    smoothing_window: int = 9,
    capacitance_bounds_pJ_per_K: tuple[float, float] = (0.002, 0.2),
    bootstrap_samples: int = 1000,
    fit_window_jitter_ns: int = 2,
    seed: int = 20260817,
) -> ThermalCapacitanceEstimate:
    """Fit ``tau_th`` and ``C_th`` from moderate nonswitching heating edges.

    The resistive current is reconstructed as ``I_R=I_in-C*dV/dt`` for the
    supplied non-negative electrical capacitance. The ratio ``V/I_R`` is mapped
    to temperature through the specimen's major heating branch, and ``V*I_R``
    then drives
    ``C_th dT/dt = P(t) - S_e (T-T0)``. The primary fit excludes the closest
    trace below switching because its near-transition overshoot and reversal
    violate the single heating-branch approximation. The returned percentile
    interval is a conditional robustness interval: it combines trace
    resampling, supplied R(T) and conductance bootstraps, and small fit-window
    shifts; it is not a device-to-device confidence interval.
    """

    required = {
        "source_file",
        "nominal_drive_mV",
        "time_ns",
        "input_current_uA",
        "output_voltage_mV",
    }
    missing = sorted(required - set(traces.columns))
    if missing:
        raise ValueError(f"Lab traces are missing columns: {', '.join(missing)}")
    if S_e_mW_per_K <= 0.0 or bootstrap_samples < 1 or fit_window_jitter_ns < 0:
        raise ValueError("S_e and bootstrap_samples must be positive; window jitter cannot be negative")
    if electrical_capacitance_pF < 0.0:
        raise ValueError("electrical_capacitance_pF must be non-negative")
    if len(selected_drives_mV) < 1:
        raise ValueError("At least one nonswitching drive must be selected")

    prepared: list[pd.DataFrame] = []
    for drive_mV in selected_drives_mV:
        match = traces.loc[np.isclose(traces["nominal_drive_mV"], float(drive_mV))]
        if match.empty:
            raise ValueError(f"No waveform found for nominal drive {drive_mV:g} mV")
        prepared.append(
            _prepare_thermal_trace(
                match,
                electrical_capacitance_pF=electrical_capacitance_pF,
                baseline_window_ns=baseline_window_ns,
                integration_window_ns=integration_window_ns,
                smoothing_window=smoothing_window,
            )
        )

    check_trace: pd.DataFrame | None = None
    if near_transition_check_mV is not None:
        check_match = traces.loc[
            np.isclose(traces["nominal_drive_mV"], float(near_transition_check_mV))
        ]
        if not check_match.empty:
            check_trace = _prepare_thermal_trace(
                check_match,
                electrical_capacitance_pF=electrical_capacitance_pF,
                baseline_window_ns=baseline_window_ns,
                integration_window_ns=integration_window_ns,
                smoothing_window=smoothing_window,
            )

    selected_indices = np.arange(len(prepared), dtype=int)
    C_th_pJ_per_K, fit_rmse_K = _fit_thermal_capacitance(
        prepared,
        selected_indices,
        resistance=resistance,
        S_e_mW_per_K=S_e_mW_per_K,
        ambient_temperature_K=ambient_temperature_K,
        fit_window_ns=fit_window_ns,
        capacitance_bounds_pJ_per_K=capacitance_bounds_pJ_per_K,
    )
    tau_th_ns = C_th_pJ_per_K / float(S_e_mW_per_K)

    trace_fit_rows: list[dict[str, float | str | bool]] = []
    for index, trace in enumerate(prepared):
        trace_C, trace_rmse = _fit_thermal_capacitance(
            prepared,
            np.asarray([index]),
            resistance=resistance,
            S_e_mW_per_K=S_e_mW_per_K,
            ambient_temperature_K=ambient_temperature_K,
            fit_window_ns=fit_window_ns,
            capacitance_bounds_pJ_per_K=capacitance_bounds_pJ_per_K,
        )
        trace_fit_rows.append(
            {
                "source_file": str(trace["source_file"].iloc[0]),
                "nominal_drive_mV": float(trace["nominal_drive_mV"].iloc[0]),
                "included_in_primary_fit": True,
                "C_th_pJ_per_K": trace_C,
                "tau_th_ns": trace_C / float(S_e_mW_per_K),
                "fit_rmse_K": trace_rmse,
            }
        )
    check_C = float("nan")
    check_rmse = float("nan")
    if check_trace is not None:
        check_C, check_rmse = _fit_thermal_capacitance(
            [check_trace],
            np.asarray([0]),
            resistance=resistance,
            S_e_mW_per_K=S_e_mW_per_K,
            ambient_temperature_K=ambient_temperature_K,
            fit_window_ns=fit_window_ns,
            capacitance_bounds_pJ_per_K=capacitance_bounds_pJ_per_K,
        )
        trace_fit_rows.append(
            {
                "source_file": str(check_trace["source_file"].iloc[0]),
                "nominal_drive_mV": float(check_trace["nominal_drive_mV"].iloc[0]),
                "included_in_primary_fit": False,
                "C_th_pJ_per_K": check_C,
                "tau_th_ns": check_C / float(S_e_mW_per_K),
                "fit_rmse_K": check_rmse,
            }
        )
    trace_fits = pd.DataFrame(trace_fit_rows)

    required_resistance = {
        "R0_ohm",
        "Ea_over_k_K",
        "Rm_ohm",
        "w_K",
        "Tc_K",
        "beta_per_K",
    }
    if resistance_bootstrap is not None:
        missing_resistance = sorted(required_resistance - set(resistance_bootstrap.columns))
        if missing_resistance:
            raise ValueError(
                f"Resistance bootstrap is missing columns: {', '.join(missing_resistance)}"
            )
    required_conductance = {"S_e_uW_per_K", "ambient_temperature_K"}
    if conductance_bootstrap is not None:
        missing_conductance = sorted(required_conductance - set(conductance_bootstrap.columns))
        if missing_conductance:
            raise ValueError(
                f"Conductance bootstrap is missing columns: {', '.join(missing_conductance)}"
            )

    rng = np.random.default_rng(seed)
    bootstrap_rows: list[dict[str, float | str]] = []
    for sample_index in range(int(bootstrap_samples)):
        sampled_resistance = resistance
        if resistance_bootstrap is not None and not resistance_bootstrap.empty:
            sampled_resistance = _params_from_bootstrap_row(
                resistance_bootstrap.iloc[int(rng.integers(0, len(resistance_bootstrap)))],
                resistance,
            )
        sampled_conductance = float(S_e_mW_per_K)
        sampled_ambient = float(ambient_temperature_K)
        if conductance_bootstrap is not None and not conductance_bootstrap.empty:
            conductance_row = conductance_bootstrap.iloc[
                int(rng.integers(0, len(conductance_bootstrap)))
            ]
            sampled_conductance = float(conductance_row["S_e_uW_per_K"]) * 1e-3
            sampled_ambient = float(conductance_row["ambient_temperature_K"])
        trace_indices = rng.integers(0, len(prepared), size=len(prepared))
        low_shift = int(rng.integers(-fit_window_jitter_ns, fit_window_jitter_ns + 1))
        high_shift = int(rng.integers(-fit_window_jitter_ns, fit_window_jitter_ns + 1))
        sampled_window = (
            float(fit_window_ns[0] + low_shift),
            float(fit_window_ns[1] + high_shift),
        )
        if sampled_window[1] - sampled_window[0] < 10.0:
            continue
        try:
            sampled_C, sampled_rmse = _fit_thermal_capacitance(
                prepared,
                trace_indices,
                resistance=sampled_resistance,
                S_e_mW_per_K=sampled_conductance,
                ambient_temperature_K=sampled_ambient,
                fit_window_ns=sampled_window,
                capacitance_bounds_pJ_per_K=capacitance_bounds_pJ_per_K,
            )
        except ValueError:
            continue
        bootstrap_rows.append(
            {
                "bootstrap_sample": float(sample_index),
                "selected_trace_indices": ",".join(str(int(value)) for value in trace_indices),
                "fit_start_ns": sampled_window[0],
                "fit_stop_ns": sampled_window[1],
                "ambient_temperature_K": sampled_ambient,
                "S_e_mW_per_K": sampled_conductance,
                "C_th_pJ_per_K": sampled_C,
                "tau_th_ns": sampled_C / sampled_conductance,
                "fit_rmse_K": sampled_rmse,
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    if len(bootstrap) < max(20, int(0.8 * bootstrap_samples)):
        raise ValueError("Too many thermal-capacitance bootstrap samples were invalid")
    C_lower, C_upper = np.percentile(bootstrap["C_th_pJ_per_K"], [2.5, 97.5])
    tau_lower, tau_upper = np.percentile(bootstrap["tau_th_ns"], [2.5, 97.5])

    trajectory_frames: list[pd.DataFrame] = []
    all_traces = [*prepared, *([check_trace] if check_trace is not None else [])]
    for index, trace in enumerate(all_traces):
        trajectory = trace.copy()
        trajectory["temperature_inferred_K"] = _infer_heating_temperature_array_K(
            trajectory["effective_resistance_ohm"].to_numpy(dtype=float),
            resistance,
        )
        trajectory["temperature_model_K"] = _thermal_temperature_from_power_K(
            trajectory["time_ns"].to_numpy(dtype=float),
            trajectory["power_mW"].to_numpy(dtype=float),
            C_th_pJ_per_K=C_th_pJ_per_K,
            S_e_mW_per_K=S_e_mW_per_K,
            ambient_temperature_K=ambient_temperature_K,
        )
        trajectory["fit_window"] = (
            (trajectory["time_ns"] >= fit_window_ns[0])
            & (trajectory["time_ns"] <= fit_window_ns[1])
        )
        trajectory["included_in_primary_fit"] = index < len(prepared)
        trajectory_frames.append(trajectory)
    trajectories = pd.concat(trajectory_frames, ignore_index=True)

    result = pd.DataFrame(
        [
            {
                "selected_traces": ",".join(
                    str(trace["source_file"].iloc[0]) for trace in prepared
                ),
                "near_transition_check_trace": (
                    str(check_trace["source_file"].iloc[0]) if check_trace is not None else ""
                ),
                "baseline_start_ns": float(baseline_window_ns[0]),
                "baseline_stop_ns": float(baseline_window_ns[1]),
                "integration_start_ns": float(integration_window_ns[0]),
                "integration_stop_ns": float(integration_window_ns[1]),
                "fit_start_ns": float(fit_window_ns[0]),
                "fit_stop_ns": float(fit_window_ns[1]),
                "ambient_temperature_K": float(ambient_temperature_K),
                "S_e_mW_per_K": float(S_e_mW_per_K),
                "electrical_capacitance_pF": float(electrical_capacitance_pF),
                "C_th_pJ_per_K": C_th_pJ_per_K,
                "C_th_ci95_lower_pJ_per_K": float(C_lower),
                "C_th_ci95_upper_pJ_per_K": float(C_upper),
                "tau_th_ns": tau_th_ns,
                "tau_th_ci95_lower_ns": float(tau_lower),
                "tau_th_ci95_upper_ns": float(tau_upper),
                "fit_rmse_K": fit_rmse_K,
                "near_transition_check_C_th_pJ_per_K": check_C,
                "near_transition_check_rmse_K": check_rmse,
                "bootstrap_samples": float(len(bootstrap)),
            }
        ]
    )
    return ThermalCapacitanceEstimate(
        result=result,
        trajectories=trajectories,
        trace_fits=trace_fits,
        bootstrap=bootstrap,
    )
