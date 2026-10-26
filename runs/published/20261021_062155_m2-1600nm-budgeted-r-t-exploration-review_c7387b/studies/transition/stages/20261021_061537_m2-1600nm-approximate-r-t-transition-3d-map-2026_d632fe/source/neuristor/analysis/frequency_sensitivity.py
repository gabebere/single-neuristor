"""Per-input frequency derivatives from saved parameter perturbation experiments."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def frequency_stencils(samples,evaluations):
    """Keep f_i slopes where measurable even if another input invalidates full L.

    Missing frequencies remain undefined. Slopes are MHz per normalized log
    coordinate. Forward/backward disagreement can include curvature and is not,
    by itself, a proof of nondifferentiability.
    """
    def key(name,dt,unit):
        return (name,float(dt),*np.round(unit,12))
    records={key(r['center_name'],r['dt_ns'],r['unit']):r for r in evaluations}
    def inputs(sample):
        r=records[key(sample.center_name,sample.dt_ns,json.loads(sample.unit_json))]
        return {x['source_label_mV']:x for x in r['per_current']}
    rows=[]
    for (name,dt),block in samples.groupby(['center_name','dt_ns']):
        center=inputs(block[block.coordinate=='center'].iloc[0])
        for (coordinate,h),points in block[block.coordinate!='center'].groupby(['coordinate','step']):
            low=inputs(points[points.direction==-1].iloc[0])
            high=inputs(points[points.direction==1].iloc[0])
            for label,base in center.items():
                a,b=low[label],high[label]
                fm,f0,fp=(x['predicted_frequency_MHz'] for x in (a,base,b))
                defined=all(value is not None and np.isfinite(value) for value in (fm,f0,fp))
                gf=(fp-f0)/h if defined else None
                gb=(f0-fm)/h if defined else None
                denom=abs(gf)+abs(gb) if defined else None
                rows.append(dict(center_name=name,dt_ns=float(dt),coordinate=coordinate,step=float(h),
                    source_label_mV=label,measured_current_A=base['measured_current_A'],
                    frequency_low_MHz=fm,frequency_center_MHz=f0,frequency_high_MHz=fp,
                    frequency_derivative_defined=defined,
                    all_three_cycle_frequency_checks=all(x['frequency_valid'] for x in (a,base,b)),
                    frequency_forward_slope=gf,frequency_backward_slope=gb,
                    frequency_centered_slope=(fp-fm)/(2*h) if defined else None,
                    frequency_plateau=bool(defined and gf==0 and gb==0),
                    frequency_slope_disagreement=abs(gf-gb)/denom if defined and denom else (0. if defined else None)))
    return pd.DataFrame(rows)


def frequency_profile_points(samples,evaluations):
    """Return measured-current frequency traces for every saved profile point."""
    records={(r['center_name'],float(r['dt_ns']),*np.round(r['unit'],12)):r for r in evaluations}
    rows=[]
    for sample in samples.itertuples():
        record=records[(sample.center_name,float(sample.dt_ns),*np.round(json.loads(sample.unit_json),12))]
        for r in record['per_current']:
            rows.append(dict(center_name=sample.center_name,dt_ns=sample.dt_ns,coordinate=sample.coordinate,
                offset=sample.step*sample.direction,measured_current_A=r['measured_current_A'],
                source_label_mV=r['source_label_mV'],predicted_frequency_MHz=r['predicted_frequency_MHz'],
                measured_frequency_MHz=r['measured_frequency_MHz'],frequency_valid=r['frequency_valid']))
    return pd.DataFrame(rows)
