"""Controlled fresh integrations for the documented full-map tendencies."""
from __future__ import annotations

import copy
from dataclasses import asdict
import time
import numpy as np
import pandas as pd

from neuristor.config import validate_config,resolved_copy
from neuristor.provenance import snapshot_package
from neuristor.runs import RunBundle
from neuristor.analysis.trend_validation import (
    TrendEvaluator,summarize_cases,controlled_trend_pairs,scaling_comparisons,
)


def plan_cases(settings):
    """Prespecify anchors, operating conditions and finite parameter changes.

    SI conversion happens here at the workflow boundary. Repeated center cases
    retain their logical roles but share cached integrations, never fitted values.
    """
    cases=[]
    def add(anchor,scenario,axis,factor,family='local',protocol='fixed_dt',scale=1):
        row=dict(anchor=anchor['name'],scenario=scenario['name'],axis=axis,factor=factor,
                 family=family,protocol=protocol,scale=scale,
                 current_uA=float(scenario['current_uA']),T0_K=float(scenario['T0_K']),
                 gamma=float(settings['gamma']),
                 C_pF=float(anchor['C_pF']),C_th_pJ_per_K=float(anchor['C_th_pJ_per_K']),
                 S_e_mW_per_K=float(anchor['S_e_mW_per_K']))
        if family=='local':row[axis]*=factor
        elif family=='gamma':row['gamma']=factor
        elif family=='scale':
            row['C_pF']*=scale;row['C_th_pJ_per_K']*=scale
        row.update(C_F=row['C_pF']*1e-12,C_th_J_per_K=row['C_th_pJ_per_K']*1e-12,
                   S_e_W_per_K=row['S_e_mW_per_K']*1e-3,case_id=f'case_{len(cases):03d}')
        cases.append(row)
    for anchor in settings['anchors']:
        for scenario in settings['scenarios']:
            for axis in ('C_pF','C_th_pJ_per_K','S_e_mW_per_K'):
                for factor in settings['local_factors']:add(anchor,scenario,axis,float(factor))
        baseline=settings['scenarios'][0]
        for gamma in settings['gamma_values']:add(anchor,baseline,'gamma',float(gamma),family='gamma')
        for protocol in ['fixed_dt','scaled_dt']:
            for scale in settings['scale_factors']:
                add(anchor,baseline,'common_scale',float(scale),family='scale',protocol=protocol,scale=float(scale))
    return cases


def validate_settings(s):
    """Validate numerical design before creating a bundle or starting the solver."""
    steps=s['dt_ns']
    if len(steps)!=3 or any(not np.isfinite(v) or v<=0 for v in steps) or any(b>=a for a,b in zip(steps[:-1],steps[1:])):
        raise ValueError('Require three successively smaller dt_ns')
    if not 0<s['convergence_fraction']<1 or not 0<s['direction_threshold_fraction']<1:
        raise ValueError('Require positive fractional convergence/direction tolerances')
    if not 0<s['initial_duration_ns'] or s['maximum_duration_ns']<2*s['initial_duration_ns']:
        raise ValueError('Require at least two observation horizons')
    for key in ['local_factors','scale_factors']:
        a=np.asarray(s[key],float)
        if len(a)<3 or not np.isfinite(a).all() or np.any(a<=0) or np.any(np.diff(a)<=0) or not np.any(a==1):
            raise ValueError(f'{key} must increase, be positive, and include 1')
    for key in ['batch_size','path_samples','max_steps_per_simulation','max_batch_samples']:
        if int(s[key])!=s[key] or s[key]<1:raise ValueError(f'{key} must be a positive integer')
    if s['path_samples']<32:raise ValueError('Require at least 32 path samples')
    if int(s['maximum_timestep_refinements'])!=s['maximum_timestep_refinements'] or s['maximum_timestep_refinements']<0:
        raise ValueError('maximum_timestep_refinements must be nonnegative')
    if not s['anchors'] or not s['scenarios']:raise ValueError('Require anchors and scenarios')
    if len({a['name'] for a in s['anchors']})!=len(s['anchors']) or len({a['name'] for a in s['scenarios']})!=len(s['scenarios']):
        raise ValueError('Anchor and scenario names must be unique')
    for a in s['anchors']:
        if any(not np.isfinite(a[k]) or a[k]<=0 for k in ('C_pF','C_th_pJ_per_K','S_e_mW_per_K')):
            raise ValueError('Anchor parameters must be finite and positive')
    for a in s['scenarios']:
        if any(not np.isfinite(a[k]) or a[k]<=0 for k in ('current_uA','T0_K')):
            raise ValueError('Scenario current and ambient must be positive')
    if not np.isfinite(s['gamma']) or any(not np.isfinite(g) or g<0 for g in s['gamma_values']):
        raise ValueError('Gamma must be finite and nonnegative')


def run_trend_validation(config,*,output_root=None,command='neuristor analyze validate-trends'):
    """Run new simulations, keep every exception, and produce a standard bundle."""
    validate_config(config)
    from neuristor.workflows import _current_params_from_config
    params,*_=_current_params_from_config(config)
    if config['model']!='current':raise ValueError('Trend tests require current model')
    s=copy.deepcopy(config['trend_validation']);validate_settings(s)
    cases=plan_cases(s)
    bundle=RunBundle.create(name=config['name'],model='current',kind='analysis',config=resolved_copy(config),
        output_root=output_root or config.get('output',{}).get('root','runs'),command=command)
    print(f'RUN_DIRECTORY={bundle.root}',flush=True)
    start=time.monotonic();measurements=[]
    try:
        snapshot_package(bundle)
        bundle.write_json('physics_parameters.json',asdict(params),label='Base SI physics; case table records overrides')
        pd.DataFrame(cases).to_csv(bundle.add_artifact('planned_cases.csv',label='Prespecified cases including repeated comparison centers'),index=False)
        def checkpoint(rows):
            pd.DataFrame(rows).to_csv(bundle.add_artifact('integrations.csv',label='Every fresh integration including rejected cases'),index=False)
            bundle.write_json('status.json',dict(state='integrating',unique_integrations=len(rows),elapsed_s=time.monotonic()-start),label='Progress')
            print(f"Fresh integrations: {len(rows)} | elapsed {time.monotonic()-start:.1f}s",flush=True)
        evaluator=TrendEvaluator(params,s,checkpoint)
        def integrate(selected,steps):
            # Extend all unresolved cases; scale tests always retain common
            # normalized horizons for a like-for-like clock comparison.
            for dt in steps:
                pending=list(selected);previous={};duration=s['initial_duration_ns']
                while pending:
                    requests=[]
                    for c in pending:
                        scale=c['scale'] if c['family']=='scale' else 1.
                        step=dt*(scale if c['protocol']=='scaled_dt' else 1.)
                        requests.append((c,step,duration*scale))
                    results=evaluator.evaluate(requests);again=[]
                    for c,row in zip(pending,results):
                        measurements.append(dict(row,**{k:v for k,v in c.items() if k not in row},
                            nominal_dt_ns=dt,normalized_duration_ns=duration))
                        old=previous.get(c['case_id'])
                        from neuristor.analysis.trend_validation import histories_agree
                        stable=old is not None and histories_agree([old,row],s['convergence_fraction'])
                        minimum_pair=duration>=2*s['initial_duration_ns']
                        if duration<s['maximum_duration_ns'] and (not minimum_pair or not stable):
                            previous[c['case_id']]=row;again.append(c)
                    pending=again;duration=min(2*duration,s['maximum_duration_ns'])
        integrate(cases,s['dt_ns'])
        for refinement in range(s['maximum_timestep_refinements']):
            summary=summarize_cases(cases,measurements,s)
            pending_ids=set(summary.loc[summary.classification=='numerical_unresolved','case_id'])
            if not pending_ids:break
            # Scaling comparison families need a matching s=1 reference at the
            # extra step. Refine all scale cases together if any needs it.
            if any(c['case_id'] in pending_ids and c['family']=='scale' for c in cases):
                pending_ids.update(c['case_id'] for c in cases if c['family']=='scale')
            integrate([c for c in cases if c['case_id'] in pending_ids],[s['dt_ns'][-1]/2**(refinement+1)])
        summary=summarize_cases(cases,measurements,s)
        pairs=controlled_trend_pairs(summary,s)
        scaling=scaling_comparisons(measurements,evaluator.paths,s['convergence_fraction'])
        for name,frame,label in [('measurements.csv',pd.DataFrame(measurements),'Every logical case step and horizon'),
            ('case_summary.csv',summary,'Final classifications and explicit convergence'),
            ('trend_pairs.csv',pairs,'Adjacent prescribed factors with unresolved endpoints'),
            ('scaling_checks.csv',scaling,'Feature and unaligned sampled-path clock checks')]:
            frame.to_csv(bundle.add_artifact(name,label=label),index=False)
        paths=[]
        ids=set(scaling.reference_integration)|set(scaling.scaled_integration)
        for ident in sorted(ids):
            paths.append(pd.DataFrame(evaluator.paths[int(ident)]).assign(integration_id=int(ident)))
        pd.concat(paths).to_csv(bundle.add_artifact('scaling_paths.csv',label='Sampled physical-time paths; no phase alignment'),index=False)
        metrics=dict(logical_cases=len(cases),unique_integrations=len(evaluator.records),
            classifications=summary.classification.value_counts().to_dict(),
            verified_oscillatory=int(((summary.classification=='oscillatory')&summary.timestep_verified).sum()),
            local_pairs=len(pairs),usable_local_pairs=int(pairs.usable.sum()),scaling_comparisons=len(scaling),
            elapsed_s=time.monotonic()-start,resimulated=True)
        bundle.write_json('metrics.json',metrics,label='Counts and completeness')
        from neuristor.plotting.trend_validation import plot_trend_validation
        plot_trend_validation(summary,pairs,scaling,bundle)
        report='''# Fresh tests of the parameter-map trends

These are new independent integrations, not saved-grid reanalysis or optimization.
The versioned recipe and planned_cases.csv specify anchors, scenarios and all
factor changes before simulation. No physical equation or hysteresis rule changed.

Common-capacitance factors are tested with fixed physical timestep and with a
timestep multiplied by the same factor. Observation horizons are multiplied by
that factor in both protocols, preserving equal dimensionless startup/settling
time. Original-step comparisons and two halvings are retained. Unresolved
timestep cases may receive the declared additional halving. Every case starts
with V=0, T=T0 and insulating hysteresis memory. Both late windows and successive
observation horizons must agree, using the existing boundary classifier.

The ratio scaling law follows by replacing t with t/s in BOTH deterministic
equations. Its assumptions are constant ideal current, fixed Se/gamma/R(T)/ambient,
identical initial state, no independent memory clock and zero noise. It does not
apply unchanged to a fixed-duration measured pulse. Scaled dt tests discrete
solver covariance; fixed dt tests numerical convergence independently. The
feature extractor has fixed physical 0.5 ns smoothing, so extracted Vpp/f can
deviate even if state paths scale. Sampled path errors use the entire history,
without phase alignment, on the declared common normalized-time grid. They are
finite-resolution comparisons, not maximum-error bounds between sample times.

All local C/Cth/Se changes keep the other parameters fixed. Separate gamma
values, a different current and the adopted 314.4 K ambient probe transfer of
the old map tendencies. Censored, nonoscillatory and unresolved cases remain
in all tables and do not become zero-frequency points. No 110 MHz ceiling enters
classification. Temperature must stay inside the measured resistance interval;
this is stricter than the historical map's padded interval.

Direction labels require two verified oscillatory endpoints. A change must exceed
both the declared minimum direction threshold and twice the larger endpoint's
last-three-step change. That envelope is a numerical-resolution diagnostic,
not a statistical confidence interval. Repeated comparisons and centers are
correlated. Aggregate results must retain scenario and endpoint denominators.
This bounded design does not prove universal monotonicity, global stability,
complete boundaries, unique identifiability or specimen calibration.

See the numerical tables and figures for completed results. The research note
documents the reviewed conclusions and exceptions after all checks finish.
'''
        bundle.write_text('report.md',report,label='Methods, scope and numerical interpretation')
        bundle.write_json('status.json',dict(state='completed',**metrics),label='Progress')
        bundle.complete(summary=metrics)
    except Exception as exc:
        bundle.fail(exc);raise
    return bundle
