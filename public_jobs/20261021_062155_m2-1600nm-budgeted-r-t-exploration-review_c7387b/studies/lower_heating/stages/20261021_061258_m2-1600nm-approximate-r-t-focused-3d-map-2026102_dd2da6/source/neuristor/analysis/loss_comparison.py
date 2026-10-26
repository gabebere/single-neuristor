"""Common feature summaries for completed, matched steady-search experiments."""
from __future__ import annotations

import pandas as pd


def verification_summary(table):
    """Summarize each timestep without treating missing frequencies as zeros.

    The two scientific objectives are defined only when all target traces pass
    eligibility. Partial frequency diagnostics carry their own denominator.
    Controls are the records excluded from the explicit measured-current range.
    """
    rows = []
    for dt, frame in table.groupby('dt_ns'):
        targets = frame[frame.target.astype(bool)]
        controls = frame[~frame.target.astype(bool)]
        ferr = targets.frequency_relative_error
        aerr = targets.robust_amplitude_relative_error
        valid = targets.frequency_valid.astype(bool)
        proximity = bool(targets.proximity_admissible.all()) if 'proximity_admissible' in targets else True
        eligible = bool(valid.all() and not targets.outside_domain.any() and proximity)
        rows.append(dict(dt_ns=float(dt), target_count=len(targets),
            defined_frequency_count=int(ferr.notna().sum()),
            sustained_count=int(targets.sustained.sum()), screened_target_count=int(valid.sum()),
            control_oscillation_count=int(controls.predicted_sustained.sum()),
            frequency_MAPE_defined=float(ferr.abs().mean()) if ferr.notna().any() else None,
            frequency_MSE_defined=float((ferr**2).mean()) if ferr.notna().any() else None,
            frequency_amplitude_MSE_defined=float((ferr**2+aerr**2).mean()) if ferr.notna().any() else None,
            amplitude_MAPE=float(aerr.abs().mean()), eligible=eligible,
            L_frequency=float((ferr**2).mean()) if eligible else None,
            L_frequency_amplitude=float((ferr**2+aerr**2).mean()) if eligible else None,
            outside_domain=bool(frame.outside_domain.any()), proximity_admissible=proximity))
    return pd.DataFrame(rows)


def timestep_comparison(table):
    """Compare every current to its next coarser replay, retaining gate changes."""
    steps = sorted(table.dt_ns.unique(), reverse=True)
    rows = []
    for coarse, fine in zip(steps, steps[1:]):
        a = table[table.dt_ns == coarse].set_index('source_label_mV')
        b = table[table.dt_ns == fine].set_index('source_label_mV')
        for label in b.index:
            x, y = a.loc[label], b.loc[label]
            f0, f1 = x.predicted_frequency_MHz, y.predicted_frequency_MHz
            rows.append(dict(source_label_mV=label, measured_current_A=y.measured_current_A,
                coarse_dt_ns=coarse, fine_dt_ns=fine, target=bool(y.target),
                frequency_change_fraction=abs(f0/f1-1) if pd.notna(f0) and pd.notna(f1) and f1 > 0 else None,
                amplitude_change_mV=abs(x.predicted_extrema_vpp_mV-y.predicted_extrema_vpp_mV),
                mean_change_mV=abs(x.predicted_mean_mV-y.predicted_mean_mV),
                sustained_changed=bool(x.predicted_sustained != y.predicted_sustained),
                eligibility_changed=bool(x.frequency_valid != y.frequency_valid)))
    return pd.DataFrame(rows)


def representative_records(table):
    """Choose low, middle and high measured-current targets before inspecting fit errors."""
    fine = table[(table.dt_ns == table.dt_ns.min()) & table.target.astype(bool)]
    ordered = fine.sort_values('measured_current_A')
    if len(ordered) < 3:
        raise ValueError('At least three target currents are required')
    return ordered.iloc[[0, len(ordered)//2, len(ordered)-1]].copy()
