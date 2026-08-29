"""Global inverse fitting of the current-drive model to measured waveforms.

One parameter vector is shared across all fitted current traces. The objective
combines phase-tolerant waveform mismatch with mean voltage, amplitude, spectrum,
frequency, onset classification, sustained periodic amplitude, edge shape, and
optional independent-parameter priors. Differential evolution is used because
hysteretic switching makes the objective discontinuous; a bounded Powell pass
refines the best population member.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from .current_drive_sim import CurrentDriveParams, simulate_current_waveforms
from .experimental_waveforms import BASELINE_WINDOW_NS, PLATEAU_WINDOW_NS, oscillation_metrics


PARAMETER_NAMES = (
    "C_pF",
    "C_th_pJ_per_K",
    "S_e_mW_per_K",
    "T0_K",
    "Tc_K",
    "w_K",
    "beta_per_K",
    "gamma",
)


@dataclass(frozen=True)
class InferenceDataset:
    """Common-time measured currents and baseline-corrected voltages."""

    time_ns: np.ndarray
    current_uA: np.ndarray
    voltage_mV: np.ndarray
    source_files: tuple[str, ...]
    nominal_drives_mV: np.ndarray
    measured_summary: pd.DataFrame
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class FitParameter:
    """One fitted quantity with hard bounds and an optional Gaussian prior."""

    name: str
    lower: float
    upper: float
    prior_center: float | None = None
    prior_scale: float | None = None
    initial_value: float | None = None


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for interpretable components of the shared objective."""

    waveform: float = 0.20
    mean_voltage: float = 0.20
    amplitude: float = 0.15
    spectrum: float = 0.15
    frequency: float = 0.15
    classification: float = 0.10
    false_positive: float = 0.0
    sustain: float = 0.0
    edge: float = 0.05
    prior: float = 0.05


@dataclass(frozen=True)
class ParameterFit:
    """Best parameters, optimizer history, and termination details."""

    mode: str
    values: np.ndarray
    objective: Mapping[str, float]
    history: pd.DataFrame
    differential_evolution_message: str
    local_message: str
    evaluations: int


@dataclass(frozen=True)
class PredictionEvaluation:
    """Fine-grid predictions and per-trace objective evidence."""

    objective: Mapping[str, float]
    trace_metrics: pd.DataFrame
    traces: pd.DataFrame


def _mask(time_ns: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return (time_ns >= bounds[0]) & (time_ns <= bounds[1])


def _summarize_voltage(time_ns: np.ndarray, current_uA: np.ndarray, voltage_mV: np.ndarray) -> dict[str, object]:
    plateau = _mask(time_ns, PLATEAU_WINDOW_NS)
    metrics = oscillation_metrics(time_ns[plateau], voltage_mV[plateau])
    return {
        "current_step_uA": float(np.median(current_uA[plateau])),
        "voltage_mean_mV": float(np.mean(voltage_mV[plateau])),
        "voltage_vpp_mV": float(np.ptp(voltage_mV[plateau])),
        **metrics,
    }


def prepare_inference_dataset(
    traces: pd.DataFrame,
    *,
    holdout_drives_mV: Sequence[float],
) -> InferenceDataset:
    """Build a common-time matrix and a current-stratified holdout set."""

    required = {"source_file", "nominal_drive_mV", "time_ns", "input_current_uA", "output_voltage_mV"}
    missing = sorted(required - set(traces.columns))
    if missing:
        raise ValueError(f"Laboratory traces are missing columns: {', '.join(missing)}")
    records = traces[["source_file", "nominal_drive_mV"]].drop_duplicates().sort_values("nominal_drive_mV")
    times: np.ndarray | None = None
    currents: list[np.ndarray] = []
    voltages: list[np.ndarray] = []
    summaries: list[dict[str, object]] = []
    files: list[str] = []
    drives: list[float] = []
    for record in records.itertuples(index=False):
        frame = traces.loc[traces["source_file"] == record.source_file].sort_values("time_ns")
        time = frame["time_ns"].to_numpy(dtype=float)
        if times is None:
            times = time
        elif time.shape != times.shape or not np.allclose(time, times, rtol=0.0, atol=1e-9):
            raise ValueError("All fitted oscilloscope traces must share the same time samples")
        baseline = _mask(time, BASELINE_WINDOW_NS)
        current = frame["input_current_uA"].to_numpy(dtype=float)
        voltage = frame["output_voltage_mV"].to_numpy(dtype=float)
        current = current - float(np.median(current[baseline]))
        voltage = voltage - float(np.median(voltage[baseline]))
        currents.append(current)
        voltages.append(voltage)
        summaries.append(_summarize_voltage(time, current, voltage))
        files.append(str(record.source_file))
        drives.append(float(record.nominal_drive_mV))
    assert times is not None
    drive_array = np.asarray(drives, dtype=float)
    test = np.asarray(
        [index for index, drive in enumerate(drive_array) if any(np.isclose(drive, value) for value in holdout_drives_mV)],
        dtype=int,
    )
    train = np.asarray([index for index in range(len(files)) if index not in set(test.tolist())], dtype=int)
    if train.size == 0 or test.size == 0:
        raise ValueError("Inference requires nonempty training and held-out current sets")
    summary = pd.DataFrame(summaries)
    summary.insert(0, "source_file", files)
    summary.insert(1, "nominal_drive_mV", drive_array)
    return InferenceDataset(
        time_ns=times,
        current_uA=np.column_stack(currents),
        voltage_mV=np.column_stack(voltages),
        source_files=tuple(files),
        nominal_drives_mV=drive_array,
        measured_summary=summary,
        train_indices=train,
        test_indices=test,
    )


def parameter_vector_from_model(params: CurrentDriveParams) -> np.ndarray:
    """Return the eight fitted quantities in documented user units."""

    rp = params.resist_params
    return np.asarray(
        [
            params.C_F * 1e12,
            params.C_th_J_per_K * 1e12,
            params.S_e_W_per_K * 1e3,
            params.T0_K,
            rp.Tc_K,
            rp.w,
            rp.beta,
            rp.gamma,
        ],
        dtype=float,
    )


def model_from_parameter_vector(
    values: Sequence[float],
    base: CurrentDriveParams,
    *,
    dt_ns: float,
) -> CurrentDriveParams:
    """Apply one shared inference vector without changing the measured R(T) scale."""

    vector = np.asarray(values, dtype=float)
    if vector.size != len(PARAMETER_NAMES):
        raise ValueError(f"Expected {len(PARAMETER_NAMES)} fitted parameters")
    rp = replace(
        base.resist_params,
        Tc_K=float(vector[4]),
        w=float(vector[5]),
        beta=float(vector[6]),
        gamma=float(vector[7]),
    )
    return replace(
        base,
        dt_s=float(dt_ns) * 1e-9,
        C_F=float(vector[0]) * 1e-12,
        C_th_J_per_K=float(vector[1]) * 1e-12,
        S_e_W_per_K=float(vector[2]) * 1e-3,
        T0_K=float(vector[3]),
        T_init_K=float(vector[3]),
        V_init_V=0.0,
        resist_params=rp,
    )


def _sample_predictions(
    dataset: InferenceDataset,
    params: CurrentDriveParams,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local = replace(
        params,
        t_pre_s=max(0.0, -float(dataset.time_ns[0]) * 1e-9),
        t_end_s=max(0.0, float(dataset.time_ns[-1]) * 1e-9),
    )
    outputs = simulate_current_waveforms(
        dataset.current_uA[:, indices],
        local,
        waveform_time_s=dataset.time_ns * 1e-9,
    )
    voltage = np.column_stack(
        [
            np.interp(dataset.time_ns * 1e-9, output["t"].astype(float), output["V_vo2"].astype(float)) * 1e3
            for output in outputs
        ]
    )
    temperature = np.column_stack(
        [
            np.interp(dataset.time_ns * 1e-9, output["t"].astype(float), output["T"].astype(float))
            for output in outputs
        ]
    )
    return voltage, temperature


def _phase_tolerant_mse(measured: np.ndarray, predicted: np.ndarray, *, max_shift_samples: int = 12) -> float:
    measured_centered = measured - np.mean(measured)
    predicted_centered = predicted - np.mean(predicted)
    return float(
        min(
            np.mean((measured_centered - np.roll(predicted_centered, shift)) ** 2)
            for shift in range(-max_shift_samples, max_shift_samples + 1)
        )
    )


def _spectrum_distance(measured: np.ndarray, predicted: np.ndarray) -> float:
    measured_amplitude = np.abs(np.fft.rfft(measured - np.mean(measured)))[1:]
    predicted_amplitude = np.abs(np.fft.rfft(predicted - np.mean(predicted)))[1:]
    measured_norm = measured_amplitude / max(float(np.linalg.norm(measured_amplitude)), 1e-12)
    predicted_norm = predicted_amplitude / max(float(np.linalg.norm(predicted_amplitude)), 1e-12)
    return float(np.mean((measured_norm - predicted_norm) ** 2) * measured_norm.size)


def _harmonic_amplitude(time_ns: np.ndarray, voltage_mV: np.ndarray, frequency_MHz: float) -> float:
    """Return the least-squares sinusoidal amplitude at one physical frequency."""

    phase = 2.0 * np.pi * float(frequency_MHz) * np.asarray(time_ns, dtype=float) * 1e-3
    design = np.column_stack((np.sin(phase), np.cos(phase), np.ones_like(phase)))
    coefficients, *_ = np.linalg.lstsq(design, np.asarray(voltage_mV, dtype=float), rcond=None)
    return float(np.hypot(coefficients[0], coefficients[1]))


def _sustained_oscillation_loss(
    time_ns: np.ndarray,
    measured_mV: np.ndarray,
    predicted_mV: np.ndarray,
    measured_frequency_MHz: float,
) -> tuple[float, float, float]:
    """Compare periodic amplitude throughout the plateau, not just at turn-on.

    Four separate time segments must retain the measured-frequency component. A
    transient can therefore match at most one segment and cannot masquerade as a
    sustained oscillation. The 25 mV scale makes a missing experimental cycle a
    deliberately substantial but bounded contribution to the global objective.
    """

    if not np.isfinite(measured_frequency_MHz) or time_ns.size < 32:
        return 0.0, float("nan"), float("nan")
    segments = [segment for segment in np.array_split(np.arange(time_ns.size), 4) if segment.size >= 8]
    measured_amplitudes = np.asarray(
        [_harmonic_amplitude(time_ns[segment], measured_mV[segment], measured_frequency_MHz) for segment in segments]
    )
    predicted_amplitudes = np.asarray(
        [_harmonic_amplitude(time_ns[segment], predicted_mV[segment], measured_frequency_MHz) for segment in segments]
    )
    loss = float(np.mean(np.minimum(((predicted_amplitudes - measured_amplitudes) / 25.0) ** 2, 16.0)))
    return loss, float(np.mean(measured_amplitudes)), float(np.mean(predicted_amplitudes))


def _prior_penalty(values: np.ndarray, parameters: Sequence[FitParameter]) -> float:
    terms = [
        ((float(value) - float(parameter.prior_center)) / float(parameter.prior_scale)) ** 2
        for value, parameter in zip(values, parameters)
        if parameter.prior_center is not None and parameter.prior_scale is not None and parameter.prior_scale > 0.0
    ]
    return float(np.mean(terms)) if terms else 0.0


def score_predictions(
    dataset: InferenceDataset,
    predicted_voltage_mV: np.ndarray,
    indices: np.ndarray,
    *,
    weights: ObjectiveWeights,
    prior_penalty: float = 0.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Calculate the documented multi-trace objective and per-trace evidence."""

    plateau = _mask(dataset.time_ns, PLATEAU_WINDOW_NS)
    edge = _mask(dataset.time_ns, (0.0, 50.0))
    rows: list[dict[str, object]] = []
    waveform_terms: list[float] = []
    mean_terms: list[float] = []
    amplitude_terms: list[float] = []
    spectrum_terms: list[float] = []
    frequency_terms: list[float] = []
    classification_terms: list[float] = []
    false_positive_terms: list[float] = []
    sustain_terms: list[float] = []
    edge_terms: list[float] = []
    for local_index, global_index in enumerate(indices):
        measured = dataset.voltage_mV[:, global_index]
        predicted = predicted_voltage_mV[:, local_index]
        measured_summary = dataset.measured_summary.iloc[int(global_index)]
        predicted_summary = _summarize_voltage(dataset.time_ns, dataset.current_uA[:, global_index], predicted)
        measured_osc = bool(measured_summary["oscillation_detected"])
        predicted_osc = bool(predicted_summary["oscillation_detected"])
        waveform_rmse_mV = np.sqrt(
            _phase_tolerant_mse(measured[plateau], predicted[plateau])
            if measured_osc
            else float(np.mean((measured[plateau] - predicted[plateau]) ** 2))
        )
        waveform_loss = float(waveform_rmse_mV / 50.0)
        mean_mse = ((float(predicted_summary["voltage_mean_mV"]) - float(measured_summary["voltage_mean_mV"])) / 50.0) ** 2
        amplitude_mse = ((float(predicted_summary["voltage_vpp_mV"]) - float(measured_summary["voltage_vpp_mV"])) / 50.0) ** 2
        spectrum_mse = _spectrum_distance(measured[plateau], predicted[plateau]) if measured_osc else 0.0
        measured_frequency = float(measured_summary["oscillation_frequency_MHz"])
        predicted_frequency = float(predicted_summary["oscillation_frequency_MHz"])
        frequency_mse = (
            ((predicted_frequency - measured_frequency) / 10.0) ** 2
            if measured_osc and predicted_osc and np.isfinite(predicted_frequency)
            else (4.0 if measured_osc != predicted_osc else 0.0)
        )
        classification_mismatch = float(measured_osc != predicted_osc)
        false_positive = float(predicted_osc and not measured_osc)
        sustain_loss, measured_harmonic_amplitude, predicted_harmonic_amplitude = _sustained_oscillation_loss(
            dataset.time_ns[plateau],
            measured[plateau],
            predicted[plateau],
            measured_frequency,
        )
        edge_rmse_mV = float(np.sqrt(np.mean((measured[edge] - predicted[edge]) ** 2)))
        edge_loss = edge_rmse_mV / 100.0
        waveform_terms.append(waveform_loss)
        mean_terms.append(mean_mse)
        amplitude_terms.append(amplitude_mse)
        spectrum_terms.append(spectrum_mse)
        frequency_terms.append(frequency_mse)
        classification_terms.append(classification_mismatch)
        false_positive_terms.append(false_positive)
        sustain_terms.append(sustain_loss)
        edge_terms.append(edge_loss)
        rows.append(
            {
                "trace_index": int(global_index),
                "source_file": dataset.source_files[int(global_index)],
                "nominal_drive_mV": float(dataset.nominal_drives_mV[int(global_index)]),
                "current_step_uA": float(measured_summary["current_step_uA"]),
                "measured_oscillation": measured_osc,
                "predicted_oscillation": predicted_osc,
                "measured_frequency_MHz": measured_frequency,
                "predicted_frequency_MHz": predicted_frequency,
                "measured_target_harmonic_amplitude_mV": measured_harmonic_amplitude,
                "predicted_target_harmonic_amplitude_mV": predicted_harmonic_amplitude,
                "measured_mean_mV": float(measured_summary["voltage_mean_mV"]),
                "predicted_mean_mV": float(predicted_summary["voltage_mean_mV"]),
                "measured_vpp_mV": float(measured_summary["voltage_vpp_mV"]),
                "predicted_vpp_mV": float(predicted_summary["voltage_vpp_mV"]),
                "phase_tolerant_plateau_rmse_mV": float(waveform_rmse_mV),
                "edge_rmse_mV": edge_rmse_mV,
                "waveform_component": waveform_loss,
                "mean_voltage_component": mean_mse,
                "amplitude_component": amplitude_mse,
                "spectrum_component": spectrum_mse,
                "frequency_component": frequency_mse,
                "classification_component": classification_mismatch,
                "false_positive_component": false_positive,
                "sustain_component": sustain_loss,
                "edge_component": edge_loss,
            }
        )
    components = {
        "waveform": float(np.mean(waveform_terms)),
        "mean_voltage": float(np.mean(mean_terms)),
        "amplitude": float(np.mean(amplitude_terms)),
        "spectrum": float(np.mean(spectrum_terms)),
        "frequency": float(np.mean(frequency_terms)),
        "classification": float(np.mean(classification_terms)),
        "false_positive": float(np.mean(false_positive_terms)),
        "sustain": float(np.mean(sustain_terms)),
        "edge": float(np.mean(edge_terms)),
        "prior": float(prior_penalty),
    }
    total = sum(float(getattr(weights, name)) * value for name, value in components.items())
    return {"total": float(total), **components}, pd.DataFrame(rows)


def evaluate_parameter_vector(
    values: Sequence[float],
    parameters: Sequence[FitParameter],
    base: CurrentDriveParams,
    dataset: InferenceDataset,
    indices: np.ndarray,
    *,
    dt_ns: float,
    weights: ObjectiveWeights,
    include_prior: bool,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, np.ndarray]:
    """Simulate and score one candidate parameter vector."""

    vector = np.asarray(values, dtype=float)
    model = model_from_parameter_vector(vector, base, dt_ns=dt_ns)
    voltage, temperature = _sample_predictions(dataset, model, indices)
    prior = _prior_penalty(vector, parameters) if include_prior else 0.0
    objective, metrics = score_predictions(dataset, voltage, indices, weights=weights, prior_penalty=prior)
    return objective, metrics, voltage, temperature


def fit_parameter_set(
    mode: str,
    parameters: Sequence[FitParameter],
    base: CurrentDriveParams,
    dataset: InferenceDataset,
    *,
    search_dt_ns: float,
    weights: ObjectiveWeights,
    include_prior: bool,
    seed: int,
    maxiter: int,
    popsize: int,
    local_max_evaluations: int,
) -> ParameterFit:
    """Run deterministic global optimization followed by bounded refinement."""

    if tuple(parameter.name for parameter in parameters) != PARAMETER_NAMES:
        raise ValueError(f"Inference parameters must be ordered as {PARAMETER_NAMES}")
    bounds = [(parameter.lower, parameter.upper) for parameter in parameters]
    base_values = parameter_vector_from_model(base)
    initial = np.clip(
        [
            float(parameter.initial_value) if parameter.initial_value is not None else float(base_values[index])
            for index, parameter in enumerate(parameters)
        ],
        [bound[0] for bound in bounds],
        [bound[1] for bound in bounds],
    )
    history: list[dict[str, float | int | str]] = []
    cache: dict[tuple[float, ...], tuple[dict[str, float], pd.DataFrame]] = {}
    evaluations = 0

    def objective_function(values: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        key = tuple(np.round(np.asarray(values, dtype=float), 12))
        cached = cache.get(key)
        if cached is None:
            try:
                objective, metrics, _, _ = evaluate_parameter_vector(
                    values,
                    parameters,
                    base,
                    dataset,
                    dataset.train_indices,
                    dt_ns=search_dt_ns,
                    weights=weights,
                    include_prior=include_prior,
                )
            except (FloatingPointError, OverflowError, ValueError):
                objective = {"total": 1e6, **{name: 1e6 for name in weights.__dataclass_fields__}}
                metrics = pd.DataFrame()
            cache[key] = (objective, metrics)
        else:
            objective, _ = cached
        history.append(
            {
                "evaluation": evaluations,
                "stage": "global" if not history or history[-1]["stage"] == "global" else "local",
                **{name: float(value) for name, value in zip(PARAMETER_NAMES, values)},
                **{f"objective_{name}": float(value) for name, value in objective.items()},
            }
        )
        return float(objective["total"])

    global_result = differential_evolution(
        objective_function,
        bounds=bounds,
        x0=initial,
        seed=int(seed),
        maxiter=int(maxiter),
        popsize=int(popsize),
        polish=False,
        updating="immediate",
        workers=1,
        tol=0.01,
        atol=1e-4,
    )
    global_evaluations = evaluations

    def local_objective(values: np.ndarray) -> float:
        value = objective_function(values)
        history[-1]["stage"] = "local"
        return value

    local_result = minimize(
        local_objective,
        np.asarray(global_result.x, dtype=float),
        method="Powell",
        bounds=bounds,
        options={"maxfev": int(local_max_evaluations), "xtol": 1e-4, "ftol": 1e-4},
    )
    candidates = [
        (float(global_result.fun), np.asarray(global_result.x, dtype=float)),
        (float(local_result.fun), np.asarray(local_result.x, dtype=float)),
    ]
    _, best = min(candidates, key=lambda item: item[0])
    best_objective, _, _, _ = evaluate_parameter_vector(
        best,
        parameters,
        base,
        dataset,
        dataset.train_indices,
        dt_ns=search_dt_ns,
        weights=weights,
        include_prior=include_prior,
    )
    history_frame = pd.DataFrame(history)
    history_frame["best_total_so_far"] = history_frame["objective_total"].cummin()
    history_frame["global_evaluations"] = global_evaluations
    return ParameterFit(
        mode=mode,
        values=best,
        objective=best_objective,
        history=history_frame,
        differential_evolution_message=str(global_result.message),
        local_message=str(local_result.message),
        evaluations=evaluations,
    )


def evaluate_fitted_model(
    mode: str,
    values: Sequence[float],
    parameters: Sequence[FitParameter],
    base: CurrentDriveParams,
    dataset: InferenceDataset,
    *,
    dt_ns: float,
    weights: ObjectiveWeights,
    include_prior: bool,
) -> PredictionEvaluation:
    """Evaluate a selected fit on every trace and return long-form evidence."""

    indices = np.arange(len(dataset.source_files), dtype=int)
    objective, metrics, voltage, temperature = evaluate_parameter_vector(
        values,
        parameters,
        base,
        dataset,
        indices,
        dt_ns=dt_ns,
        weights=weights,
        include_prior=include_prior,
    )
    split = np.full(len(indices), "train", dtype=object)
    split[dataset.test_indices] = "test"
    metrics.insert(1, "split", split)
    frames: list[pd.DataFrame] = []
    for index in indices:
        frames.append(
            pd.DataFrame(
                {
                    "fit_mode": mode,
                    "split": split[index],
                    "source_file": dataset.source_files[index],
                    "nominal_drive_mV": float(dataset.nominal_drives_mV[index]),
                    "time_ns": dataset.time_ns,
                    "measured_current_uA": dataset.current_uA[:, index],
                    "measured_voltage_mV": dataset.voltage_mV[:, index],
                    "predicted_voltage_mV": voltage[:, index],
                    "predicted_temperature_K": temperature[:, index],
                }
            )
        )
    return PredictionEvaluation(objective=objective, trace_metrics=metrics, traces=pd.concat(frames, ignore_index=True))
