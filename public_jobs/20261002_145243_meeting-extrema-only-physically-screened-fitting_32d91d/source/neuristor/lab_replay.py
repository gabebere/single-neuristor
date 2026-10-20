"""All-current manual comparisons using measured input histories and shared physics."""
from dataclasses import replace

import numpy as np
import pandas as pd

from .current_drive_sim import simulate_current_waveforms
from .model_validation import _baseline_corrected_trace
from .oscillation_audit import audit_voltage


def replay_sweep(traces, params):
    """Return sampled voltage/resistance/temperature and separate per-current metrics.

    Both measured channels are pre-pulse baseline corrected. The full measured
    current prehistory drives one shared parameter vector; R is simulated device
    resistance, not the potentially misleading measured TIA voltage/current ratio.
    """
    groups = list(traces.groupby("nominal_drive_mV", sort=True))
    if not groups:
        raise ValueError("No laboratory traces found")
    prepared = [_baseline_corrected_trace(g) for _, g in groups]
    time = prepared[0][0]
    if any(t.shape != time.shape or not np.allclose(t, time) for t, _, _ in prepared):
        raise ValueError("Measured records must share a time grid")
    inputs = np.column_stack([i for _, i, _ in prepared])
    local = replace(params, t_pre_s=max(0, -time[0]*1e-9), t_end_s=max(0, time[-1]*1e-9))
    outputs = simulate_current_waveforms(inputs, local, waveform_time_s=time*1e-9)
    rows, metrics = [], []
    plateau = (time >= 50) & (time <= 250)
    for (drive, _), (_, current, measured), out in zip(groups, prepared, outputs):
        sampled = {name: np.interp(time*1e-9, out["t"].astype(float), out[key].astype(float))
                   for name, key in [("simulated_voltage_mV", "V_vo2"), ("resistance_ohm", "R"), ("temperature_K", "T")]}
        sampled["simulated_voltage_mV"] *= 1000
        if not all(np.isfinite(v).all() for v in sampled.values()):
            raise ValueError("Simulation produced non-finite values; reduce timestep or check parameters")
        step = float(np.mean(current[plateau]))
        rows.append(pd.DataFrame(dict(source_mV=drive, current_step_uA=step, time_ns=time,
                    input_current_uA=current, measured_voltage_mV=measured, **sampled)))
        m, _ = audit_voltage(time, measured)
        p, _ = audit_voltage(time, sampled["simulated_voltage_mV"])
        metrics.append(dict(source_mV=drive, current_step_uA=step,
            waveform_RMSE_mV=float(np.sqrt(np.mean((sampled["simulated_voltage_mV"][plateau]-measured[plateau])**2))),
            temperature_outside_calibration=bool(np.min(out["T"]) < params.resist_params.T_min_K or np.max(out["T"]) > params.resist_params.T_max_K),
            **{f"measured_{k}": v for k, v in m.items()}, **{f"simulated_{k}": v for k, v in p.items()}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(metrics)


def major_branches(resist_params, points=500):
    """Evaluate both major branches with the authoritative hysteresis engine.

    Each temperature has an independent, unreversed state. This avoids replacing
    the model's branch law with plotting-only equations or a minor-loop sweep.
    """
    from .model import HysteresisArray
    temperature = np.linspace(resist_params.T_min_K, resist_params.T_max_K, points)
    result = dict(temperature_K=temperature)
    for branch, label in [('insulator', 'heating_ohm'), ('metal', 'cooling_ohm')]:
        state = HysteresisArray(resist_params, size=points, start_branch=branch)
        state.initialize(temperature)
        result[label] = state.evaluate(temperature)[0]
    return pd.DataFrame(result)
