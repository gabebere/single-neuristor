"""New controlled tests of capacitance scaling and conditional parameter trends.

The evaluator calls the authoritative ideal-current solver and the existing
two-window boundary classifier. It never turns absent cycles into zero frequency.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd

from neuristor.analysis.boundary_search import classify_trajectory
from neuristor.current_drive_sim import simulate_current_parameter_grid


def histories_agree(rows, tolerance):
    """Require matching resolved classes and adjacent feature agreement."""
    if len(rows) < 2 or len({r['classification'] for r in rows}) != 1:
        return False
    if rows[-1]['classification'] == 'nonoscillatory':
        return True
    if rows[-1]['classification'] != 'oscillatory':
        return False
    return all(abs(a[k]/b[k]-1) <= tolerance
               for a,b in zip(rows[:-1],rows[1:])
               for k in ('frequency_MHz','extrema_vpp_mV'))


class TrendEvaluator:
    """Cache independent integrations while retaining every logical comparison.

    Each request carries SI-valued parameters, timestep and duration. Batching is
    allowed only at identical current, ambient, gamma, timestep and duration;
    the existing solver supplies independent hysteresis memories per column.
    Saved trajectory samples are interpolation samples for clock comparisons,
    not smoothed or projected physical trajectories.
    """
    def __init__(self, params, settings, checkpoint=lambda rows:None):
        if params.sigma_W_sqrt_s or params.t_pre_s or params.pulse_on_s or params.pulse_off_s is not None:
            raise ValueError('Trend tests require deterministic constant current from t=0')
        self.params,self.settings,self.checkpoint=params,settings,checkpoint
        self.cache={};self.records=[];self.paths={}

    @staticmethod
    def key(case,dt_ns,duration_ns):
        return tuple(float(case[k]) for k in ('C_F','C_th_J_per_K','S_e_W_per_K','gamma','current_uA','T0_K'))+(float(dt_ns),float(duration_ns))

    def evaluate(self, requests):
        """Evaluate (SI case, physical dt ns, physical horizon ns) requests."""
        pending={self.key(c,dt,end):(c,dt,end) for c,dt,end in requests
                 if self.key(c,dt,end) not in self.cache}
        groups={}
        for key,(case,dt,end) in pending.items():
            group=(case['current_uA'],case['T0_K'],case['gamma'],dt,end)
            groups.setdefault(group,[]).append((key,case))
        for (current,ambient,gamma,dt,end),items in groups.items():
            if end/dt+2 > self.settings['max_steps_per_simulation']:
                raise ValueError('Trend simulation exceeds declared step budget')
            p=replace(self.params,T0_K=ambient,T_init_K=ambient,dt_s=dt*1e-9,
                      t_end_s=end*1e-9,resist_params=replace(self.params.resist_params,gamma=gamma))
            batch=min(self.settings['batch_size'],max(1,int(self.settings.get('max_batch_samples',2500000)/(end/dt+2))))
            for start in range(0,len(items),batch):
                part=items[start:start+batch]
                outputs=simulate_current_parameter_grid(current,p,
                    C_F=np.array([c['C_F'] for _,c in part]),
                    C_th_J_per_K=np.array([c['C_th_J_per_K'] for _,c in part]),
                    S_e_W_per_K=np.array([c['S_e_W_per_K'] for _,c in part]))
                for (key,c),out in zip(part,outputs):
                    actual=replace(p,C_F=c['C_F'],C_th_J_per_K=c['C_th_J_per_K'],S_e_W_per_K=c['S_e_W_per_K'])
                    row=classify_trajectory(out,actual,self.settings)
                    row.update(integration_id=len(self.records),dt_ns=dt,requested_duration_ns=end,
                               C_F=c['C_F'],C_th_J_per_K=c['C_th_J_per_K'],S_e_W_per_K=c['S_e_W_per_K'],
                               current_uA=current,T0_K=ambient,gamma=gamma)
                    self.cache[key]=row;self.records.append(row)
                    t=np.asarray(out['t'],float)*1e9
                    sample_t=np.linspace(t[0],t[-1],self.settings['path_samples'])
                    self.paths[row['integration_id']]={
                        'time_ns':sample_t,
                        'voltage_V':np.interp(sample_t,t,out['V_vo2']),
                        'temperature_K':np.interp(sample_t,t,out['T']),
                        'resistance_ohm':np.interp(sample_t,t,out['R'])}
                self.checkpoint(self.records)
        return [self.cache[self.key(c,dt,end)].copy() for c,dt,end in requests]


def summarize_cases(cases, measurements, settings):
    """Retain horizon and timestep disagreements rather than hiding exclusions."""
    table=pd.DataFrame(measurements);result=[]
    for case in cases:
        part=table[table.case_id==case['case_id']]
        steps=[]
        for _,by_dt in part.groupby('nominal_dt_ns',sort=False):
            by_dt=by_dt.sort_values('normalized_duration_ns')
            last=by_dt.iloc[-1].to_dict()
            stable=histories_agree(by_dt.iloc[-2:].to_dict('records'),settings['convergence_fraction'])
            last['horizon_verified']=stable
            if not stable and last['classification'] in ('oscillatory','nonoscillatory'):
                last['classification']='time_unresolved'
            steps.append(last)
        steps.sort(key=lambda r:r['nominal_dt_ns'],reverse=True)
        final=steps[-1].copy()
        verified=len(steps)>=3 and histories_agree(steps[-3:],settings['convergence_fraction'])
        if not verified and final['classification'] in ('oscillatory','nonoscillatory'):
            final['classification']='numerical_unresolved'
        final.update(timestep_verified=verified,verification_steps_ns=[r['dt_ns'] for r in steps[-3:]],
                     horizon_verified=bool(final['horizon_verified']))
        for metric in ('frequency_MHz','extrema_vpp_mV'):
            vals=[r.get(metric) for r in steps[-3:]]
            final[metric+'_max_step_change']=max(abs(a/b-1) for a,b in zip(vals[:-1],vals[1:])) if all(v is not None and np.isfinite(v) and v>0 for v in vals) else None
        final.update({k:v for k,v in case.items() if k not in final})
        result.append(final)
    return pd.DataFrame(result)


def controlled_trend_pairs(summary, settings):
    """Compare only adjacent prescribed factors, keeping every unresolved pair."""
    rows=[]
    for keys,part in summary[summary.family.isin(['local','gamma'])].groupby(['family','anchor','scenario','axis']):
        part=part.sort_values('factor')
        for low,high in zip(part.iloc[:-1].to_dict('records'),part.iloc[1:].to_dict('records')):
            usable=all(r['classification']=='oscillatory' and r['timestep_verified'] and r['horizon_verified'] for r in (low,high))
            row=dict(zip(['family','anchor','scenario','axis'],keys),low_case=low['case_id'],high_case=high['case_id'],
                     low_factor=low['factor'],high_factor=high['factor'],usable=usable,
                     low_classification=low['classification'],high_classification=high['classification'])
            for metric in ('frequency_MHz','extrema_vpp_mV'):
                change=high[metric]/low[metric]-1 if usable else None
                # A conservative numerical envelope from each endpoint's last
                # three timesteps. It is a resolution diagnostic, not a CI.
                uncertainty=max(low[metric+'_max_step_change'] or 0,high[metric+'_max_step_change'] or 0)
                threshold=max(settings['direction_threshold_fraction'],2*uncertainty)
                row[metric+'_change']=change
                row[metric+'_direction_threshold']=threshold
                row[metric+'_direction']=('unresolved' if change is None else
                    'increase' if change>threshold else 'decrease' if change < -threshold else 'small_or_unresolved')
            rows.append(row)
    return pd.DataFrame(rows)


def scaling_comparisons(measurements, paths, tolerance):
    """Test f*s and Vpp against s=1 at equal normalized step and horizon.

    Fixed-physical-dt comparisons are deliberately separate from co-scaled-dt
    comparisons. The latter test the discrete solver covariance; the former
    expose timestep error. No phase alignment is applied to the sampled paths.
    """
    table=pd.DataFrame(measurements)
    rows=[]
    for keys,part in table[table.family=='scale'].groupby(['anchor','protocol','nominal_dt_ns','normalized_duration_ns']):
        reference=part[np.isclose(part.scale,1)].iloc[0]
        for _,r in part[~np.isclose(part.scale,1)].iterrows():
            valid=r.classification==reference.classification=='oscillatory'
            ferr=r.frequency_MHz*r.scale/reference.frequency_MHz-1 if valid else None
            aerr=r.extrema_vpp_mV/reference.extrema_vpp_mV-1 if valid else None
            a,b=paths[int(reference.integration_id)],paths[int(r.integration_id)]
            u=a['time_ns']; ub=b['time_ns']/r.scale
            v=np.interp(u,ub,b['voltage_V']);temp=np.interp(u,ub,b['temperature_K'])
            span=max(float(np.ptp(a['voltage_V'])),1e-12)
            row=dict(zip(['anchor','protocol','nominal_dt_ns','normalized_duration_ns'],keys),
                     scale=float(r.scale),reference_integration=int(reference.integration_id),
                     scaled_integration=int(r.integration_id),both_oscillatory=valid,
                     reference_classification=reference.classification,scaled_classification=r.classification,
                     frequency_scaling_error=ferr,amplitude_scaling_error=aerr,
                     voltage_path_normalized_rmse=float(np.sqrt(np.mean((v-a['voltage_V'])**2))/span),
                     temperature_path_rmse_K=float(np.sqrt(np.mean((temp-a['temperature_K'])**2))),
                     feature_scaling_pass=bool(valid and abs(ferr)<=tolerance and abs(aerr)<=tolerance))
            rows.append(row)
    return pd.DataFrame(rows)


def verified_scaling_summary(summary,tolerance):
    """Compare final scale endpoints only after horizon and timestep verification."""
    rows=[]
    for (anchor,protocol),part in summary[summary.family=='scale'].groupby(['anchor','protocol']):
        base=part[np.isclose(part.scale,1)].iloc[0]
        for _,r in part[~np.isclose(part.scale,1)].iterrows():
            valid=all(x.classification=='oscillatory' and x.timestep_verified and x.horizon_verified for x in [base,r])
            ferr=float(r.frequency_MHz*r.scale/base.frequency_MHz-1) if valid else None
            aerr=float(r.extrema_vpp_mV/base.extrema_vpp_mV-1) if valid else None
            rows.append(dict(anchor=anchor,protocol=protocol,scale=float(r.scale),
                reference_case=base.case_id,scaled_case=r.case_id,both_verified_oscillatory=valid,
                reference_classification=base.classification,scaled_classification=r.classification,
                reference_final_dt_ns=base.dt_ns,scaled_final_dt_ns=r.dt_ns,
                frequency_scaling_error=ferr,amplitude_scaling_error=aerr,
                within_tolerance=bool(valid and abs(ferr)<=tolerance and abs(aerr)<=tolerance)))
    return pd.DataFrame(rows)


def rising_crossing_frequency(time_ns,voltage_V):
    """Independent diagnostic frequency from rising mid-level crossings.

    Use the last half of the saved sampled path, no smoothing or peak detector.
    A quantile mid-level avoids dependence on one sampled extreme. Three rising
    crossings are required; this does not replace the persistence classifier.
    Finite path sampling and linear crossing interpolation limit its accuracy.
    """
    t=np.asarray(time_ns,float);v=np.asarray(voltage_V,float)
    if t.ndim!=1 or t.shape!=v.shape or len(t)<9 or not np.isfinite(t).all() or not np.isfinite(v).all() or np.any(np.diff(t)<=0):
        raise ValueError('Crossing paths must be finite, matching and increasing')
    keep=t>=.5*(t[0]+t[-1]);t=t[keep];v=v[keep]
    low,high=np.quantile(v,[.05,.95])
    if high-low<1e-9:return None
    level=.5*(low+high)
    indices=np.flatnonzero((v[:-1]<=level)&(v[1:]>level))
    if len(indices)<3:return None
    crossings=t[indices]+(level-v[indices])/(v[indices+1]-v[indices])*(t[indices+1]-t[indices])
    return float(1000/np.median(np.diff(crossings)))


def independent_frequency_checks(summary,paths,verified_scaling):
    """Cross-check final scale pairs with a second frequency measurement."""
    cases=summary.set_index('case_id');rows=[]
    for r in verified_scaling.to_dict('records'):
        base=cases.loc[r['reference_case']];scaled=cases.loc[r['scaled_case']]
        a=paths[int(base.integration_id)];b=paths[int(scaled.integration_id)]
        fa=rising_crossing_frequency(a['time_ns'],a['voltage_V'])
        fb=rising_crossing_frequency(b['time_ns'],b['voltage_V'])
        valid=r['both_verified_oscillatory'] and fa is not None and fb is not None
        rows.append(dict(anchor=r['anchor'],protocol=r['protocol'],scale=r['scale'],
            both_verified_oscillatory=r['both_verified_oscillatory'],reference_frequency_MHz=fa,
            scaled_frequency_MHz=fb,crossing_scaling_error=fb*r['scale']/fa-1 if valid else None,
            reference_crossing_vs_peak_error=fa/base.frequency_MHz-1 if valid else None,
            scaled_crossing_vs_peak_error=fb/scaled.frequency_MHz-1 if valid else None))
    return pd.DataFrame(rows)
