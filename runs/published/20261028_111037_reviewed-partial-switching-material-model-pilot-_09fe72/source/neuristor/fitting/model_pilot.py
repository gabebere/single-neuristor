"""Prespecified material-model pilot: shared starts, feature ranking and units."""

from __future__ import annotations

from dataclasses import asdict, replace
import copy
import hashlib
import json

import numpy as np
import pandas as pd

from neuristor.current_drive_sim import ExperimentalCurrentModel, simulate_experimental_current_waveforms
from neuristor.fitting.joint_inference import NAMES
from neuristor.fitting.steady_inference import evaluate_settled


def realize_model(params, specification):
    """Translate declared dimensionless thermal ratios to explicit SI parameters.

    Two-region reference budgets split as w and 1-w. Secondary capacity/cooling
    ratios then introduce asymmetry; this is a diagnostic assumption, not an
    identification of thermal volume from a conductance/resistance partition.
    """
    kind = specification["kind"]
    if kind in ["warm_parallel", "series_bottleneck"]:
        weight = float(specification["weight"])
        experiment = ExperimentalCurrentModel(
            kind=kind,
            primary_fraction=weight,
            C_secondary_J_per_K=(1 - weight) * specification["capacity_ratio"] * params.C_th_J_per_K,
            S_secondary_W_per_K=(1 - weight) * specification["cooling_ratio"] * params.S_e_W_per_K,
            coupling_W_per_K=specification["coupling_ratio"] * params.S_e_W_per_K,
        )
        primary = replace(params, C_th_J_per_K=weight * params.C_th_J_per_K, S_e_W_per_K=weight * params.S_e_W_per_K)
        return primary, experiment
    return params, ExperimentalCurrentModel(kind=kind, phase_tau_s=float(specification.get("phase_tau_ns", 0.0)) * 1e-9)


def model_simulator(specification):
    """Bind a pure physics hypothesis for the existing measured-waveform evaluator."""

    def simulate(currents, params, **kwargs):
        primary, experiment = realize_model(params, specification)
        return simulate_experimental_current_waveforms(currents, primary, experiment, **kwargs)

    return simulate


def initial_cases(settings):
    """Form the declared grid; no per-current fitting or random population is used."""
    cases = []

    def add(anchor, specification):
        key = hashlib.sha256(json.dumps([anchor["values"], specification], sort_keys=True).encode()).hexdigest()[:12]
        cases.append(
            dict(
                id=specification["kind"] + "_" + key,
                anchor=anchor["name"],
                values=anchor["values"],
                specification=specification,
                stage="declared_grid",
            )
        )

    for anchor in settings["anchors"]:
        add(anchor, dict(kind="baseline"))
        for tau in settings["phase_tau_ns"]:
            add(anchor, dict(kind="phase_lag", phase_tau_ns=tau))
        for kind in ["warm_parallel", "series_bottleneck"]:
            for weight in settings["weights"]:
                for capacity in settings["secondary_capacity_ratios"]:
                    for coupling in settings["coupling_ratios"]:
                        add(
                            anchor,
                            dict(
                                kind=kind,
                                weight=weight,
                                capacity_ratio=capacity,
                                cooling_ratio=settings["secondary_cooling_ratio"],
                                coupling_ratio=coupling,
                            ),
                        )
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate declared pilot cases")
    return cases


def refinement_cases(winners, settings, bounds):
    """Eight one-coordinate probes per arm; static R(T) and ambient stay fixed."""
    cases = []
    seen = set()
    for winner in winners:
        for coordinate in [0, 1, 2, 7]:
            proposals = (
                [winner["values"][coordinate] + delta for delta in [-0.05, 0.05]]
                if coordinate == 7
                else [winner["values"][coordinate] * factor for factor in settings["refinement_factors"]]
            )
            for value in proposals:
                case = copy.deepcopy(winner)
                vector = np.array(case["values"], float)
                clipped = float(np.clip(value, *bounds[NAMES[coordinate]]))
                vector[coordinate] = clipped
                if np.array_equal(vector, case["values"]):
                    continue
                key = hashlib.sha256(
                    json.dumps([vector.tolist(), case["specification"]], sort_keys=True).encode()
                ).hexdigest()[:12]
                if key in seen:
                    continue
                seen.add(key)
                case.update(
                    id="refine_" + case["specification"]["kind"] + "_" + key,
                    values=vector.tolist(),
                    stage="shared_dynamic_refinement",
                    parent=winner["id"],
                    changed_coordinate=NAMES[coordinate],
                    bound_clamped=clipped != value,
                )
                cases.append(case)
    return cases


def pilot_metrics(rows, target_labels, control_labels, settings):
    """Rank feasibility first; missing-cycle diagnostics are never accepted losses."""
    frame = pd.DataFrame(rows)
    target = frame[frame.source_label_mV.isin(target_labels)]
    control = frame[frame.source_label_mV.isin(control_labels)]
    if not len(target):
        raise ValueError("Pilot requires target records")
    physical = bool(frame.temperature_valid.all() and frame.proximity_admissible.all())
    invalid = int((~target.frequency_valid).sum())
    ferr = pd.to_numeric(target.frequency_relative_error, errors="coerce")
    aerr = pd.to_numeric(target.robust_amplitude_relative_error, errors="coerce")
    scale = float(settings["voltage_scale_mV"])
    mean_error = (frame.predicted_mean_mV - frame.measured_mean_mV) / scale
    shape = []
    for field in ["peak_mV", "trough_mV"]:
        delta = (target["predicted_" + field] - target["measured_" + field]) / scale
        shape.append(delta)
    partial = float(np.mean(aerr**2) + np.mean(mean_error**2))
    if ferr.notna().any():
        partial += float(np.mean(ferr.dropna() ** 2))
    defined_shape = np.concatenate([x.dropna().to_numpy(float) for x in shape])
    if len(defined_shape):
        partial += float(np.mean(defined_shape**2))
    excessive = int(
        (
            control.sustained.astype(bool)
            & (
                control.predicted_robust_vpp_mV
                > np.maximum(
                    settings["control_amplitude_floor_mV"],
                    settings["control_amplitude_ratio"] * control.measured_robust_vpp_mV,
                )
            )
        ).sum()
    )
    accepted = bool(physical and invalid == 0 and excessive == 0 and len(defined_shape) == 2 * len(target))
    return dict(
        rank=[int(not physical), invalid, excessive, partial],
        diagnostic_feature_score=partial,
        accepted_feature_loss=partial if accepted else None,
        physical=physical,
        passing_targets=int(target.frequency_valid.sum()),
        targets=len(target),
        defined_frequencies=int(ferr.notna().sum()),
        frequency_MAPE_defined=float(ferr.abs().mean()) if ferr.notna().any() else None,
        amplitude_MAPE_all=float(aerr.abs().mean()),
        mean_voltage_RMSE_mV=float(np.sqrt(np.mean(mean_error**2)) * scale),
        shape_RMSE_defined_mV=float(np.sqrt(np.mean(defined_shape**2)) * scale) if len(defined_shape) else None,
        excessive_control_oscillations=excessive,
        maximum_temperature_K=float(frame.temperature_max_K.max()),
    )


def evaluate_pilot_case(context, case, dt_ns, indices, *, native_callback=None):
    """Use the canonical measured prehistory, sampling and extrema extraction."""
    rows, voltage, outside = evaluate_settled(
        context["dataset"],
        case["values"],
        context["base"],
        np.asarray(indices, int),
        context["config"]["steady"],
        dt_ns,
        native_callback=native_callback,
        simulator=model_simulator(case["specification"]),
    )
    return rows, voltage, outside


def realized_units(params, specification):
    """Expose actual primary/secondary SI values beside the reference coordinates."""
    primary, experiment = realize_model(params, specification)
    active = asdict(experiment)
    if experiment.kind not in ["warm_parallel", "series_bottleneck"]:
        for field in ["primary_fraction", "C_secondary_J_per_K", "S_secondary_W_per_K", "coupling_W_per_K"]:
            active.pop(field)
    if experiment.kind != "phase_lag":
        active.pop("phase_tau_s")
    return dict(
        C_F=primary.C_F,
        C_primary_J_per_K=primary.C_th_J_per_K,
        S_primary_W_per_K=primary.S_e_W_per_K,
        **active,
    )
