"""Search design and acceptance rules for a fixed-fraction filament profile.

No circuit equations live here. Geometry-scaled thermal intervals are explicit
diagnostic assumptions, separate from the inherited presentation intervals.
"""

from __future__ import annotations

import copy
import numpy as np

from neuristor.fitting.joint_inference import NAMES, natural_vector


def profile_bounds(settings, family, fraction):
    """Keep tau=Cth/Se fixed in range while optionally scaling the Se interval."""
    bounds = copy.deepcopy(settings["bounds"])
    if family not in ["inherited", "volume_scaled"] or not 0 < fraction <= 1:
        raise ValueError("Unknown thermal family or invalid fraction")
    if family == "volume_scaled":
        bounds["S_e_mW_per_K"] = [fraction * float(v) for v in bounds["S_e_mW_per_K"]]
    return bounds


def profile_seeds(base, records, bounds, fraction):
    """Include direct and coupled-volume starts, recording every bound clamp.

    Scaling Se and Cth together preserves the initial thermal time. It is a
    starting family, not a forced relation between geometry and heat flow.
    Circuit capacitance is not multiplied by the filament weight.
    """
    fixed = natural_vector(base)
    static = [3, 4, 5, 6, 8, 9, 10]
    lows, highs = np.array([bounds[name] for name in NAMES], float).T
    result = []
    seen = set()
    for record in records:
        for coupled in [False, True]:
            vector = np.array(record["values"], float).copy()
            if coupled:
                vector[1] *= fraction / float(record.get("filament_fraction", 1.0))
            vector[static] = fixed[static]
            clipped = np.clip(vector, lows, highs)
            key = tuple(np.round(clipped, 12))
            if key in seen:
                continue
            seen.add(key)
            result.append(
                dict(
                    values=clipped.tolist(),
                    source=record.get("source", "declared"),
                    coupled_thermal_start=coupled,
                    clamped=bool(np.any(vector != clipped)),
                    source_fraction=float(record.get("filament_fraction", 1.0)),
                    filament_fraction=fraction,
                )
            )
    if not result:
        raise ValueError("At least one declared seed is required")
    return result


def acceptance(finest, convergence, settings):
    """Prespecified numerical/observable gates; never turn a missing cycle into zero error."""
    targets = finest[finest.target].copy()
    ferror = targets.frequency_relative_error.astype(float).abs()
    aerror = targets.robust_amplitude_relative_error.astype(float).abs()
    means = finest.predicted_mean_mV - finest.measured_mean_mV
    physical = bool(finest.temperature_valid.all() and finest.proximity_admissible.all())
    defined = int(ferror.notna().sum())
    frequency = float(ferror.mean()) if defined else None
    amplitude = float(aerror.mean())
    mean_rms = float(np.sqrt(np.mean(means**2)))
    quiet = (
        ~finest.target
        & ~finest.measured_sustained.fillna(False).astype(bool)
        & (finest.measured_robust_vpp_mV <= settings["quiet_measured_vpp_mV"])
    )
    false_positive = int(
        (
            quiet
            & finest.sustained.astype(bool)
            & (finest.predicted_robust_vpp_mV > settings["quiet_simulated_vpp_mV"])
        ).sum()
    )
    target_conv = convergence[convergence.target]
    conv = bool(len(target_conv) and target_conv.features_converged.all() and target_conv.waveform_converged.all())
    flags = dict(
        all_record_physical=physical,
        all_target_cycles=bool(targets.frequency_valid.all()),
        all_target_frequencies_defined=defined == len(targets),
        frequency_accuracy=bool(
            defined == len(targets)
            and frequency <= settings["frequency_MAPE"]
            and ferror.max() <= settings["maximum_frequency_error"]
        ),
        amplitude_accuracy=bool(
            aerror.notna().all()
            and amplitude <= settings["amplitude_MAPE"]
            and aerror.max() <= settings["maximum_amplitude_error"]
        ),
        mean_voltage_accuracy=mean_rms <= settings["mean_voltage_RMSE_mV"],
        quiet_controls=false_positive == 0,
        numerically_converged=conv,
    )
    return dict(
        success=all(flags.values()),
        gates=flags,
        failed_gates=[k for k, v in flags.items() if not v],
        target_records=len(targets),
        passing_targets=int(targets.frequency_valid.sum()),
        defined_frequencies=defined,
        frequency_MAPE_defined=frequency,
        amplitude_MAPE_all=amplitude,
        mean_voltage_RMSE_mV=mean_rms,
        quiet_control_count=int(quiet.sum()),
        false_positive_controls=false_positive,
    )


def diagnostic_rank(result):
    """Select refinement seeds using a common primary objective and explicit failures."""
    metrics = result.get("acceptance", {})
    summary = result.get("all_target", {})
    gates = metrics.get("gates", {})
    return (
        not gates.get("all_record_physical", False),
        int(summary.get("invalid_count", 1000)),
        not metrics.get("success", False),
        float(summary.get("dynamic_loss") if summary.get("dynamic_loss") is not None else 1e12),
        float(metrics.get("mean_voltage_RMSE_mV", 1e12)),
    )
