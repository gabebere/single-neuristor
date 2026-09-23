"""Blind validation of the specimen current-drive model against lab waveforms.

The functions in this module never tune a parameter to an individual trace.
They replay the measured, baseline-corrected current records through one frozen
parameter set and apply the same oscillation detector to measured and predicted
voltages.  A separate grid explores electrical and thermal capacitance without
changing the fitted resistance, ambient temperature, or thermal conductance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .current_drive_sim import CurrentDriveParams, simulate_current_steps, simulate_current_waveform
from .experimental_waveforms import BASELINE_WINDOW_NS, PLATEAU_WINDOW_NS, oscillation_metrics


@dataclass(frozen=True)
class ModelValidationResult:
    """Measured-versus-predicted summaries and sampled time traces."""

    comparison: pd.DataFrame
    traces: pd.DataFrame
    convergence: pd.DataFrame


@dataclass(frozen=True)
class CapacitanceSensitivityResult:
    """Frequency/classification grid for C, Cth, and measured current."""

    summary: pd.DataFrame


def _mask(time_ns: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return (time_ns >= bounds[0]) & (time_ns <= bounds[1])


def _electrical_summary(
    time_ns: np.ndarray,
    current_uA: np.ndarray,
    voltage_mV: np.ndarray,
) -> dict[str, float | bool]:
    """Summarize a baseline-corrected trace on the fixed plateau window."""

    plateau = _mask(time_ns, PLATEAU_WINDOW_NS)
    if int(np.sum(plateau)) < 9:
        raise ValueError("Trace does not span the 50--250 ns comparison window")
    metrics = oscillation_metrics(time_ns[plateau], voltage_mV[plateau])
    mean_power_uW = float(np.mean(current_uA[plateau] * voltage_mV[plateau] * 1e-3))
    frequency_MHz = float(metrics["oscillation_frequency_MHz"])
    energy_per_cycle_pJ = (
        mean_power_uW / frequency_MHz
        if bool(metrics["oscillation_detected"]) and np.isfinite(frequency_MHz)
        else float("nan")
    )
    return {
        "current_step_uA": float(np.median(current_uA[plateau])),
        "voltage_mean_mV": float(np.mean(voltage_mV[plateau])),
        "voltage_min_mV": float(np.min(voltage_mV[plateau])),
        "voltage_max_mV": float(np.max(voltage_mV[plateau])),
        "voltage_vpp_mV": float(np.ptp(voltage_mV[plateau])),
        "mean_power_uW": mean_power_uW,
        "energy_per_cycle_pJ": energy_per_cycle_pJ,
        **metrics,
    }


def _baseline_corrected_trace(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_ns = frame["time_ns"].to_numpy(dtype=float)
    current_uA = frame["input_current_uA"].to_numpy(dtype=float)
    voltage_mV = frame["output_voltage_mV"].to_numpy(dtype=float)
    baseline = _mask(time_ns, BASELINE_WINDOW_NS)
    if int(np.sum(baseline)) < 3:
        raise ValueError("Trace does not span the documented baseline window")
    return (
        time_ns,
        current_uA - float(np.median(current_uA[baseline])),
        voltage_mV - float(np.median(voltage_mV[baseline])),
    )


def _simulate_sampled_waveform(
    time_ns: np.ndarray,
    current_uA: np.ndarray,
    params: CurrentDriveParams,
) -> dict[str, np.ndarray]:
    local = replace(
        params,
        t_pre_s=max(0.0, -float(time_ns[0]) * 1e-9),
        t_end_s=max(0.0, float(time_ns[-1]) * 1e-9),
        V_init_V=0.0,
    )
    raw = simulate_current_waveform(
        current_uA,
        local,
        waveform_time_s=time_ns * 1e-9,
        seed=0,
    )
    internal_time = raw["t"].astype(float)
    sample_time = time_ns * 1e-9
    return {
        "voltage_mV": np.interp(sample_time, internal_time, raw["V_vo2"].astype(float)) * 1e3,
        "temperature_K": np.interp(sample_time, internal_time, raw["T"].astype(float)),
        "resistance_ohm": np.interp(sample_time, internal_time, raw["R"].astype(float)),
        "power_uW": np.interp(sample_time, internal_time, raw["P"].astype(float)) * 1e6,
    }


def compare_model_to_lab(
    traces: pd.DataFrame,
    params: CurrentDriveParams,
    *,
    convergence_dt_ns: float | None = None,
    convergence_drives_mV: tuple[float, ...] = (300.0, 500.0, 800.0),
) -> ModelValidationResult:
    """Replay every measured current waveform through one frozen model.

    Model voltage is sampled at the original 1 ns laboratory times before the
    common detector is applied.  This gives both sides the same analysis window
    and effective sampling bandwidth.  The optional smaller-step reruns check
    representative low, middle, and high currents without changing physics.
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
        raise ValueError(f"Laboratory traces are missing columns: {', '.join(missing)}")

    comparison_rows: list[dict[str, object]] = []
    trace_rows: list[pd.DataFrame] = []
    convergence_rows: list[dict[str, object]] = []
    ordered_files = (
        traces[["source_file", "nominal_drive_mV"]]
        .drop_duplicates()
        .sort_values("nominal_drive_mV")
    )
    for record in ordered_files.itertuples(index=False):
        frame = traces.loc[traces["source_file"] == record.source_file].sort_values("time_ns")
        time_ns, current_uA, measured_voltage_mV = _baseline_corrected_trace(frame)
        predicted = _simulate_sampled_waveform(time_ns, current_uA, params)
        measured = _electrical_summary(time_ns, current_uA, measured_voltage_mV)
        model = _electrical_summary(time_ns, current_uA, predicted["voltage_mV"])
        plateau = _mask(time_ns, PLATEAU_WINDOW_NS)
        row: dict[str, object] = {
            "source_file": str(record.source_file),
            "nominal_drive_mV": float(record.nominal_drive_mV),
            **{f"measured_{key}": value for key, value in measured.items()},
            **{f"predicted_{key}": value for key, value in model.items()},
            "classification_match": bool(measured["oscillation_detected"]) == bool(model["oscillation_detected"]),
            "plateau_voltage_rmse_mV": float(
                np.sqrt(np.mean((predicted["voltage_mV"][plateau] - measured_voltage_mV[plateau]) ** 2))
            ),
            "plateau_voltage_mean_error_mV": float(model["voltage_mean_mV"]) - float(measured["voltage_mean_mV"]),
            "predicted_temperature_min_K": float(np.min(predicted["temperature_K"][plateau])),
            "predicted_temperature_max_K": float(np.max(predicted["temperature_K"][plateau])),
        }
        comparison_rows.append(row)
        trace_rows.append(
            pd.DataFrame(
                {
                    "source_file": str(record.source_file),
                    "nominal_drive_mV": float(record.nominal_drive_mV),
                    "time_ns": time_ns,
                    "measured_current_uA": current_uA,
                    "measured_voltage_mV": measured_voltage_mV,
                    "predicted_voltage_mV": predicted["voltage_mV"],
                    "predicted_temperature_K": predicted["temperature_K"],
                    "predicted_resistance_ohm": predicted["resistance_ohm"],
                    "predicted_power_uW": predicted["power_uW"],
                }
            )
        )

        if convergence_dt_ns is not None and any(
            np.isclose(float(record.nominal_drive_mV), drive) for drive in convergence_drives_mV
        ):
            refined = _simulate_sampled_waveform(
                time_ns,
                current_uA,
                replace(params, dt_s=float(convergence_dt_ns) * 1e-9),
            )
            refined_summary = _electrical_summary(time_ns, current_uA, refined["voltage_mV"])
            convergence_rows.append(
                {
                    "nominal_drive_mV": float(record.nominal_drive_mV),
                    "current_step_uA": float(measured["current_step_uA"]),
                    "coarse_dt_ns": float(params.dt_s) * 1e9,
                    "fine_dt_ns": float(convergence_dt_ns),
                    "coarse_voltage_mean_mV": float(model["voltage_mean_mV"]),
                    "fine_voltage_mean_mV": float(refined_summary["voltage_mean_mV"]),
                    "absolute_mean_difference_mV": abs(
                        float(model["voltage_mean_mV"]) - float(refined_summary["voltage_mean_mV"])
                    ),
                    "coarse_oscillation_detected": bool(model["oscillation_detected"]),
                    "fine_oscillation_detected": bool(refined_summary["oscillation_detected"]),
                }
            )

    return ModelValidationResult(
        comparison=pd.DataFrame(comparison_rows).sort_values("measured_current_step_uA").reset_index(drop=True),
        traces=pd.concat(trace_rows, ignore_index=True),
        convergence=pd.DataFrame(convergence_rows),
    )


def capacitance_sensitivity(
    currents_uA: list[float] | np.ndarray,
    params: CurrentDriveParams,
    *,
    capacitances_pF: tuple[float, ...],
    thermal_capacitances_pJ_per_K: tuple[float, ...],
    pulse_duration_ns: float = 300.0,
    pre_duration_ns: float = 200.0,
    post_duration_ns: float = 200.0,
    detector_sample_step_ns: float = 1.0,
) -> CapacitanceSensitivityResult:
    """Map ideal-step predictions over C and Cth at measured currents.

    The sensitivity grid deliberately changes only the two capacitances.  The
    voltage is downsampled to the laboratory's 1 ns spacing before applying the
    common oscillation detector, preventing a finer simulation grid from being
    mistaken for extra experimental bandwidth.
    """

    currents = np.asarray(currents_uA, dtype=float).reshape(-1)
    if currents.size == 0 or np.any(currents < 0.0):
        raise ValueError("currents_uA must contain nonnegative values")
    sample_time_ns = np.arange(
        -float(pre_duration_ns),
        float(pulse_duration_ns) + float(post_duration_ns) + 0.5 * detector_sample_step_ns,
        float(detector_sample_step_ns),
    )
    plateau = _mask(sample_time_ns, PLATEAU_WINDOW_NS)
    rows: list[dict[str, object]] = []
    for thermal_capacitance in thermal_capacitances_pJ_per_K:
        for capacitance in capacitances_pF:
            local = replace(
                params,
                C_F=float(capacitance) * 1e-12,
                C_th_J_per_K=float(thermal_capacitance) * 1e-12,
                t_pre_s=float(pre_duration_ns) * 1e-9,
                t_end_s=(float(pulse_duration_ns) + float(post_duration_ns)) * 1e-9,
                pulse_on_s=0.0,
                pulse_off_s=float(pulse_duration_ns) * 1e-9,
                V_init_V=0.0,
            )
            outputs = simulate_current_steps(currents.tolist(), local, seed=0)
            for current, output in zip(currents, outputs):
                voltage_mV = np.interp(
                    sample_time_ns * 1e-9,
                    output["t"].astype(float),
                    output["V_vo2"].astype(float),
                ) * 1e3
                metrics = oscillation_metrics(sample_time_ns[plateau], voltage_mV[plateau])
                rows.append(
                    {
                        "current_uA": float(current),
                        "electrical_capacitance_pF": float(capacitance),
                        "thermal_capacitance_pJ_per_K": float(thermal_capacitance),
                        "oscillation_detected": bool(metrics["oscillation_detected"]),
                        "oscillation_frequency_MHz": float(metrics["oscillation_frequency_MHz"]),
                        "voltage_mean_mV": float(np.mean(voltage_mV[plateau])),
                        "voltage_vpp_mV": float(np.ptp(voltage_mV[plateau])),
                        "temperature_max_K": float(np.max(output["T"])),
                    }
                )
    return CapacitanceSensitivityResult(summary=pd.DataFrame(rows))
