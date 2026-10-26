"""Reproducible local derivative, timestep and L-BFGS-B experiments."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from neuristor.analysis.experimental_waveforms import load_converted_sweep
from neuristor.analysis.optimizer_diagnostics import finite_difference_stencils, derivative_timestep_changes
from neuristor.config import resolved_copy, source_directory
from neuristor.fitting.joint_inference import NAMES, encode, natural_vector
from neuristor.fitting.optimizer_audit import ACTIVE, COORDINATES, CachedEvaluator, lbfgsb_score_trial
from neuristor.fitting.parameter_inference import prepare_inference_dataset
from neuristor.fitting.steady_inference import evaluate_settled, measured_current_targets
from neuristor.provenance import snapshot_package, package_source_files
from neuristor.runs import RunBundle


def run_optimizer_audit(config,base,*,output_root=None,command='neuristor analyze audit-optimizer'):
    """Audit existing candidates; keep all old bundles and the scientific loss unchanged."""
    settings=copy.deepcopy(config['steady']); audit=config['optimizer_audit']
    steps=list(map(float,audit['normalized_steps']))
    dts=list(map(float,audit['dt_ns']))
    if not steps or any(h<=0 or h>=.5 for h in steps) or not dts or any(dt<=0 for dt in dts):
        raise ValueError('Positive timestep and normalized perturbation sizes are required')
    def resolve(value):
        path=Path(value)
        return path if path.is_absolute() else (source_directory(config)/path).resolve()
    data=resolve(settings['data_directory'])
    raw,_=load_converted_sweep(data)
    labels=sorted(raw.nominal_drive_mV.unique())
    provisional=prepare_inference_dataset(raw,holdout_drives_mV=[labels[0]])
    indices=measured_current_targets(provisional,settings['target_current_range_uA'])
    targets=provisional.nominal_drives_mV[indices]
    dataset=prepare_inference_dataset(raw,holdout_drives_mV=[x for x in labels if x not in targets])
    centers=[]
    for value in audit['sources']:
        source=resolve(value)
        manifest=json.loads((source/'run.json').read_text())
        if manifest['status']!='completed':
            raise ValueError('Audit parents must be completed')
        parent=json.loads((source/'resolved_config.json').read_text())
        parent_settings=parent['steady']
        # Only the amplitude weight differs between the two controlled centers.
        for key in ('bounds','window_ns','target_current_range_uA','search_dt_ns'):
            if parent_settings[key]!=settings[key]:
                raise ValueError(f'Parent and audit protocols disagree: {key}')
        table=pd.read_csv(source/'parameters.csv')
        selected=table[table.candidate.str.startswith(('diagnostic_','search_'))]
        if len(selected)!=1:
            raise ValueError('Expected one searched candidate in each parent')
        values=selected.iloc[0][list(NAMES)].to_numpy(float)
        fixed=[3,4,5,6,8,9,10]
        np.testing.assert_allclose(values[fixed],natural_vector(base)[fixed],rtol=1e-12,atol=1e-12)
        mode='frequency_only' if parent_settings['amplitude_weight']==0 else 'frequency_amplitude'
        centers.append((mode,source,values,parent_settings))
    if len({n for n,*_ in centers})!=len(centers):
        raise ValueError('Audit requires distinct objective centers')
    bundle=RunBundle.create(name=config['name'],model='optimizer-audit',kind='analysis',
        config=resolved_copy(config),output_root=output_root or config['output']['root'],command=command)
    print(f'RUN_DIRECTORY={bundle.root}',flush=True)
    start=monotonic(); deadline=start+float(audit['total_seconds'])
    status_path=bundle.add_artifact('status.json',label='Live audit phase')
    history=bundle.add_artifact('evaluations.jsonl',label='Unique cached evaluations including per-current features')
    stop=bundle.path('STOP')
    state={'phase':'initializing','evaluations':0}
    def update(**extra):
        state.update(extra); state['elapsed_s']=monotonic()-start
        temp=status_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(status_path)
    def check():
        if stop.exists() or monotonic()>deadline:
            raise TimeoutError('Audit stopped or reached its declared wall-time cap')
    samples=[]; trials=[]; evaluators={}; last_print=[0.]
    try:
        snapshot_package(bundle)
        files=[Path(config['_source']),*data.glob('*_converted.csv'),*package_source_files()]
        for _,source,_,_ in centers:
            files.extend(source/name for name in ['run.json','resolved_config.json','parameters.csv','verification.csv'])
        bundle.write_json('source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},label='Input and source hashes')
        with history.open('a') as out:
            for name,source,values,parent_settings in centers:
                def callback(record):
                    out.write(json.dumps(dict(record,center_name=name,phase=state['phase']),allow_nan=False)+'\n');out.flush()
                    update(evaluations=state['evaluations']+1)
                    if monotonic()-last_print[0]>30:
                        print(f"{state['phase']} | {name} | evaluations {state['evaluations']}",flush=True)
                        last_print[0]=monotonic()
                evaluator=CachedEvaluator(dataset,base,parent_settings,values,callback)
                evaluators[name]=evaluator
                center=encode(values,evaluator.bounds)[ACTIVE]
                update(phase='profiles',center=name)
                for dt in dts:
                    points=[('center',0.,0,center)]
                    for j,coordinate in enumerate(COORDINATES):
                        for h in steps:
                            for direction in [-1,1]:
                                unit=center.copy();unit[j]+=direction*h
                                if not 0<=unit[j]<=1:
                                    raise ValueError('A requested centered stencil crosses a parameter bound')
                                points.append((coordinate,h,direction,unit))
                    for coordinate,h,direction,unit in points:
                        check();r=evaluator.evaluate(unit,dt)
                        row={k:v for k,v in r.items() if k not in ('per_current','values','unit')}
                        row.update(center_name=name,coordinate=coordinate,step=h,direction=direction,
                                   values_json=json.dumps(r['values']),unit_json=json.dumps(r['unit']))
                        samples.append(row)
                        pd.DataFrame(samples).to_csv(bundle.add_artifact('profile_samples.csv',label='All coordinate perturbations and timesteps'),index=False)
                update(phase='gradient_trials',center=name)
                feasible=[r for r in evaluator.cache.values()
                          if r['dt_ns']==float(audit['gradient_dt_ns']) and r['eligible']]
                gradient_start=np.asarray(min(feasible,key=lambda r:r['score'])['unit']) if feasible else center
                start_kind='eligible_profile_point' if feasible else 'infeasible_parent_center'
                for h in audit['gradient_steps']:
                    check()
                    result=lbfgsb_score_trial(evaluator,gradient_start,dt_ns=float(audit['gradient_dt_ns']),step=float(h),
                        max_evaluations=int(audit['gradient_max_evaluations']),
                        max_iterations=int(audit['gradient_max_iterations']),max_line_search=int(audit['gradient_max_line_search']))
                    result.update(center_name=name,start_kind=start_kind)
                    trials.append(result)
                    bundle.write_json('gradient_trials.json',trials,label='L-BFGS-B terminations, steps and best sampled candidates')
            # Finish callbacks while the history stream is still open.
        samples=pd.DataFrame(samples)
        stencils=finite_difference_stencils(samples)
        stencils.to_csv(bundle.add_artifact('finite_differences.csv',label='Forward/backward/centered derivatives'),index=False)
        derivative_timestep_changes(stencils).to_csv(bundle.add_artifact('derivative_timestep_changes.csv',label='Derivative dependence on timestep'),index=False)
        compact=[{k:v for k,v in r.items() if k not in ('best','values','best_unit')} for r in trials]
        pd.DataFrame(compact).to_csv(bundle.add_artifact('gradient_summary.csv',label='Gradient trial summary'),index=False)
        update(phase='verifying')
        verification=[]; parameters=[]; traces=[]
        for name,source,values,parent_settings in centers:
            chosen=min((r for r in trials if r['center_name']==name),key=lambda r:r['best_score'])
            candidates={f'{name}_baseline':values,f'{name}_gradient':np.asarray(chosen['values'])}
            for candidate,vector in candidates.items():
                parameters.append(dict(candidate=candidate,**dict(zip(NAMES,vector)),C_th_pJ_per_K=float(vector[1]*vector[2])))
                for dt in dts:
                    check();rows,voltage,outside=evaluate_settled(dataset,vector,base,np.arange(len(labels)),parent_settings,dt)
                    time=dataset.time_ns[dataset.time_ns<=parent_settings['window_ns'][1]]
                    for i,row in enumerate(rows):
                        row.update(candidate=candidate,mode=name,dt_ns=dt,target=row['source_label_mV'] in targets,outside_domain=outside)
                        verification.append(row)
                        if dt==min(dts):
                            traces.append(pd.DataFrame(dict(time_ns=time,measured_mV=dataset.voltage_mV[:len(time),i],
                                predicted_mV=voltage[:,i],candidate=candidate,dt_ns=dt,source_label_mV=row['source_label_mV'])))
                    pd.DataFrame(verification).to_csv(bundle.add_artifact('verification.csv',label='All-current gradient/baseline timestep checks'),index=False)
                    print(f'Verified {candidate} dt={dt} ns',flush=True)
        pd.DataFrame(parameters).to_csv(bundle.add_artifact('parameters.csv',label='Baseline and gradient-selected physical parameters'),index=False)
        pd.concat(traces).to_csv(bundle.add_artifact('verification_traces.csv',label='Finest saved voltage samples on all currents'),index=False)
        from neuristor.analysis.loss_comparison import verification_summary,timestep_comparison
        summaries=[]; changes=[]
        for candidate,frame in pd.DataFrame(verification).groupby('candidate'):
            summaries.append(verification_summary(frame).assign(candidate=candidate))
            changes.append(timestep_comparison(frame).assign(candidate=candidate))
        pd.concat(summaries).to_csv(bundle.add_artifact('summary.csv',label='Scientific losses, eligibility and partial diagnostics'),index=False)
        pd.concat(changes).to_csv(bundle.add_artifact('convergence.csv',label='Both finer-step comparisons'),index=False)
        from neuristor.plotting.optimizer_diagnostics import plot_profiles
        plot_profiles(samples,bundle.add_artifact('figures/coordinate_profiles.png',label='Loss-score profiles at three timesteps'))
        metrics=dict(unique_evaluations=state['evaluations'],profile_points=len(samples),stencils=len(stencils),
            scientific_derivatives_defined=int(stencils.scientific_derivative_defined.sum()),gradient_trials=len(trials),
            verified_batches=len(verification)//len(labels),elapsed_s=monotonic()-start,
            normalized_coordinates=COORDINATES,dt_ns=dts)
        bundle.write_json('metrics.json',metrics,label='Audit completion and derivative validity')
        text='''# Parameter perturbations, timestep sensitivity and gradient-based trials

Physical equations, measured input waveforms, feature extraction, loss weights,
bounds and eligibility screens are unchanged. Four normalized log coordinates
are C, Se, Cth/Se and gamma. A normalized increment is a fraction of that
coordinate's log-bound range; it is not an increment in SI units. In particular,
changing Se at fixed Cth/Se changes Cth too. The six R(T) values and T0 stay fixed.

Each center is tested at positive/negative coordinate perturbations, multiple
step sizes, and three timesteps. Forward/backward/centered slopes of the finite
optimizer score are separate from derivatives of an eligible all-target loss.
A missing frequency makes the latter undefined. A score plateau can mean peak
quantization or the rejection tie-breaker cap; it is not proof of a physical
stationary point. Slope disagreement alone is not a proof of nondifferentiability.

L-BFGS-B uses the identical finite optimizer score at two declared difference
steps. If its starting point is infeasible, it is an exploratory test of score
refinement, not a gradient calculation on a defined all-target scientific loss.
The score ranks failed-target count before its bounded partial error term.
Optimizer convergence messages are recorded and do not constitute a fit.
The best sampled candidate per objective is replayed on all currents at three
steps, alongside the original center. Neither gradient trial is a global search.

Frequency features are extracted on the original acquisition sample grid after
interpolating simulated voltage. The smoothing rule targets 0.5 ns but has a
minimum of five samples; the actual 1 ns data grid therefore uses five samples.
Peak times, median periods and rejection decisions can change discretely.
No smoother proxy loss or changed physical model is introduced in this audit.
'''
        text+='\n## Completion\n'+json.dumps(metrics,indent=2)+'\n\nSee gradient_summary.csv, finite_differences.csv and summary.csv for outcomes.\n'
        bundle.write_text('report.md',text,label='Methods, interpretations and completion')
        update(phase='completed',**metrics)
        bundle.complete(summary=metrics)
    except Exception as exc:
        update(phase='failed',error=str(exc));bundle.fail(exc);raise
    return bundle
