"""Finite-difference evidence for a sampled, potentially discontinuous loss."""
from __future__ import annotations

import numpy as np
import pandas as pd


def finite_difference_stencils(samples):
    """Compare forward/backward/centered slopes without inventing missing losses.

    Coordinates are normalized log search coordinates, not SI increments. A
    derivative of the finite feasibility score is retained separately from a
    derivative of the eligible all-target scientific loss.
    """
    rows=[]
    centers = samples[samples.coordinate == 'center']
    for (name,dt), block in samples.groupby(['center_name','dt_ns']):
        base=centers[(centers.center_name==name)&(centers.dt_ns==dt)].iloc[0]
        perturbed=block[block.coordinate!='center']
        for (coordinate,h), group in perturbed.groupby(['coordinate','step']):
            low=group[group.direction==-1].iloc[0]
            high=group[group.direction==1].iloc[0]
            forward=(high.score-base.score)/h
            backward=(base.score-low.score)/h
            valid=bool(base.eligible and low.eligible and high.eligible)
            denom=abs(forward)+abs(backward)
            rows.append(dict(center_name=name,dt_ns=dt,coordinate=coordinate,step=h,
                score_center=base.score,score_forward_slope=forward,score_backward_slope=backward,
                score_centered_slope=(high.score-low.score)/(2*h),
                score_slope_disagreement=abs(forward-backward)/denom if denom else 0.,
                score_plateau=bool(denom==0),
                eligibility_changed=bool(low.eligible!=base.eligible or high.eligible!=base.eligible),
                invalid_count_changed=bool(low.invalid_count!=base.invalid_count or high.invalid_count!=base.invalid_count),
                scientific_derivative_defined=valid,
                objective_forward_slope=(high.objective_loss-base.objective_loss)/h if valid else None,
                objective_backward_slope=(base.objective_loss-low.objective_loss)/h if valid else None,
                objective_centered_slope=(high.objective_loss-low.objective_loss)/(2*h) if valid else None))
    return pd.DataFrame(rows)


def derivative_timestep_changes(stencils):
    """Show slope dependence on timestep; null scientific derivatives stay null."""
    rows=[]
    for (name,coordinate,h), group in stencils.groupby(['center_name','coordinate','step']):
        group=group.sort_values('dt_ns',ascending=False)
        for a,b in zip(group.iloc[:-1].itertuples(),group.iloc[1:].itertuples()):
            rows.append(dict(center_name=name,coordinate=coordinate,step=h,
                coarse_dt_ns=a.dt_ns,fine_dt_ns=b.dt_ns,
                score_slope_change=abs(a.score_centered_slope-b.score_centered_slope),
                score_slope_sign_changed=bool(np.sign(a.score_centered_slope)!=np.sign(b.score_centered_slope)),
                scientific_both_defined=bool(a.scientific_derivative_defined and b.scientific_derivative_defined),
                objective_slope_change=abs(a.objective_centered_slope-b.objective_centered_slope)
                    if a.scientific_derivative_defined and b.scientific_derivative_defined else None))
    return pd.DataFrame(rows)
