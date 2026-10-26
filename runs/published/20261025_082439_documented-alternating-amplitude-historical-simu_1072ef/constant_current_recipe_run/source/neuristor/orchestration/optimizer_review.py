"""Review completed derivative and expanded-search experiments on a common basis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from neuristor.analysis.frequency_sensitivity import frequency_stencils, frequency_profile_points
from neuristor.analysis.loss_comparison import verification_summary, timestep_comparison
from neuristor.plotting.frequency_sensitivity import plot_frequency_profiles
from neuristor.provenance import snapshot_package
from neuristor.runs import RunBundle


def review_optimizer_runs(audit_run,frequency_run,combined_run,*,output_root=None,command='neuristor analyze review-optimizer'):
    """Compare old centers, expanded DE/Powell and saved L-BFGS-B outcomes."""
    sources={'audit':Path(audit_run).resolve(),'frequency_only':Path(frequency_run).resolve(),
             'frequency_amplitude':Path(combined_run).resolve()}
    configs={}
    for name,source in sources.items():
        if json.loads((source/'run.json').read_text())['status']!='completed':
            raise ValueError('All optimizer-review sources must be completed')
        configs[name]=json.loads((source/'resolved_config.json').read_text())
    reference_settings=configs['audit']['steady']
    for name in ['frequency_only','frequency_amplitude']:
        settings=configs[name]['steady']
        for key in ['bounds','target_current_range_uA','window_ns','search_dt_ns','fixed_parameters','feature_method','loss_function']:
            if settings[key]!=reference_settings[key]:
                raise ValueError(f'Optimizer-review protocols disagree: {key}')
        if settings['amplitude_weight']!=(0. if name=='frequency_only' else 1.):
            raise ValueError('Wrong expanded objective supplied')
        if not json.loads((sources[name]/'metrics.json').read_text())['verification_complete']:
            raise ValueError('Expanded verification is incomplete')
    bundle=RunBundle.create(name='Optimizer budget derivative and gradient review',model='optimizer-review',kind='analysis',
        config={'sources':{n:str(p) for n,p in sources.items()}},output_root=output_root or 'runs',command=command)
    try:
        snapshot_package(bundle)
        audit=sources['audit']
        samples=pd.read_csv(audit/'profile_samples.csv')
        evaluations=[json.loads(line) for line in (audit/'evaluations.jsonl').read_text().splitlines()]
        derivatives=frequency_stencils(samples,evaluations)
        derivatives.to_csv(bundle.add_artifact('frequency_derivatives.csv',label='Per-input finite-difference frequency slopes'),index=False)
        points=frequency_profile_points(samples,evaluations)
        points.to_csv(bundle.add_artifact('frequency_profile_points.csv',label='Extracted frequencies at every perturbation'),index=False)
        plot_frequency_profiles(points,bundle.add_artifact('figures/frequency_profiles.png',label='Direct frequency profiles for the middle measured input'))
        all_features=[];parameters=[];manifest={}
        verification=pd.read_csv(audit/'verification.csv')
        audit_parameters=pd.read_csv(audit/'parameters.csv')
        for (candidate,mode),frame in verification.groupby(['candidate','mode']):
            method='baseline' if candidate.endswith('_baseline') else 'gradient'
            all_features.append(frame.assign(method=method))
            parameters.append(audit_parameters[audit_parameters.candidate==candidate].assign(mode=mode,method=method))
        for mode in ['frequency_only','frequency_amplitude']:
            source=sources[mode]
            params=pd.read_csv(source/'parameters.csv')
            selected=params[params.candidate.str.startswith(('diagnostic_','search_'))]
            if len(selected)!=1:
                raise ValueError('Expected one expanded searched candidate per objective')
            candidate=selected.candidate.iloc[0]
            selected=selected.copy(); selected['candidate']=f'{mode}_expanded'
            parameters.append(selected.assign(mode=mode,method='expanded'))
            frame=pd.read_csv(source/'verification.csv')
            frame=frame[frame.candidate==candidate].copy()
            frame['candidate']=f'{mode}_expanded'
            all_features.append(frame.assign(mode=mode,method='expanded'))
        combined=pd.concat(all_features,ignore_index=True)
        # All methods must have identical measured inputs, target group and features.
        expected=None
        for candidate,frame in combined.groupby('candidate'):
            measured=frame[frame.dt_ns==frame.dt_ns.min()].sort_values('source_label_mV')
            measured=measured[['source_label_mV','measured_current_A','measured_frequency_MHz','measured_extrema_vpp_mV','target']].reset_index(drop=True)
            if expected is not None:pd.testing.assert_frame_equal(expected,measured)
            expected=measured
        summary=[];changes=[]
        for (method,mode,candidate),frame in combined.groupby(['method','mode','candidate']):
            summary.append(verification_summary(frame).assign(method=method,mode=mode,candidate=candidate))
            changes.append(timestep_comparison(frame).assign(method=method,mode=mode,candidate=candidate))
        table=pd.concat(summary,ignore_index=True)
        table.to_csv(bundle.add_artifact('method_summary.csv',label='Common all-target and partial diagnostics for all methods'),index=False)
        pd.concat(changes).to_csv(bundle.add_artifact('convergence.csv',label='Method-by-method timestep comparisons'),index=False)
        combined.to_csv(bundle.add_artifact('verification.csv',label='Unchanged per-current verification features'),index=False)
        pd.concat(parameters).to_csv(bundle.add_artifact('parameters.csv',label='All compared physical parameter sets'),index=False)
        for name,source in sources.items():
            files=['run.json','resolved_config.json','verification.csv','parameters.csv','metrics.json']
            if name=='audit':files+=['profile_samples.csv','evaluations.jsonl','gradient_summary.csv']
            else:files+=['optimization_history.jsonl','search_result.json']
            manifest[name]={'path':str(source),'hashes':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in files}}
        bundle.write_json('numerical_provenance.json',manifest,label='Immutable inputs and hashes')
        fine=table[table.dt_ns==table.dt_ns.min()]
        metrics=dict(method_candidates=len(fine),frequency_stencils=len(derivatives),
            frequency_stencils_defined=int(derivatives.frequency_derivative_defined.sum()),
            frequency_stencil_plateaus=int(derivatives.frequency_plateau.sum()),
            defined_disagreement_over_half=int(((derivatives.frequency_slope_disagreement>.5)&derivatives.frequency_derivative_defined).sum()),
            eligible_finest_candidates=int(fine.eligible.sum()),dt_ns=sorted(table.dt_ns.unique().tolist()))
        bundle.write_json('metrics.json',metrics,label='Review completion and derivative-availability counts')
        report='''# Do a larger search or gradient methods improve the fit?

This review introduces no new simulations or altered objectives. It joins saved
all-current checks for the earlier centers, expanded DE/Powell searches, and
L-BFGS-B score trials. The sources retain complete histories and termination
reasons. Original data, fixed static fit/T0, bounds and relative losses match.

frequency_derivatives.csv retains per-input df_i/du_j wherever three neighboring
frequencies are defined. That can hold even if another input makes the full
scientific loss undefined. Missing frequencies are not zero. A zero finite slope
can be a sampling plateau; forward/backward disagreement can include curvature.
The figure shows the middle input by measured-current rank and capacitance
perturbations chosen before seeing residuals. No smoother frequency estimator
was introduced.

method_summary.csv uses the same complete-cycle definition for every method.
Full accepted losses remain null on failed eligibility; partial frequency
errors carry their denominators. A lower partial error is not an all-target fit.
The existing outside-domain flag aggregates each complete 22-record replay.
Gradient methods were warm-started using the profile experiment, while expanded
searches received larger populations and more starts. These are practical tests,
not a compute-matched benchmark proving one algorithm universally superior.

Parameter and numerical convergence checks remain essential. Optimizer success
or loss reduction alone does not establish specimen calibration or prove that
no better model parameter set exists.
'''
        report+='\n## Review metrics\n'+json.dumps(metrics,indent=2)+'\n'
        bundle.write_text('report.md',report,label='Scope and interpretation of the optimizer comparison')
        bundle.complete(summary=metrics)
    except Exception as exc:
        bundle.fail(exc);raise
    return bundle
