"""Audit and compactly archive the completed material-model pilot, without rerunning it."""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

from neuristor.fitting.steady_inference import settled_features
from neuristor.orchestration.model_pilot import _context
from neuristor.runs import RunBundle, RunRegistry

project=Path('/Users/gabrielberezovsky/Documents/Projects/single-neuristor')
source=project/'runs/local/20261028_105307_partial-switching-material-model-comparison-pilo_fdbd33'
def read(path):return json.loads(path.read_text())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def relative(a,b):return float(np.max(abs(a-b)/np.maximum(np.maximum(abs(a),abs(b)),1e-30)))

manifest=read(source/'run.json');assert manifest['status']=='completed'
metrics=read(source/'metrics.json')
hashes=read(source/'artifact_hashes.json')
for name,expected in hashes.items():assert digest(source/name)==expected,name
for name,expected in read(source/'source_hashes.json').items():
    assert digest(source/name)==expected
    assert digest(project/'src/neuristor'/Path(name).relative_to('source/neuristor'))==expected
context=_context(read(source/'execution_config.json'))
times=context['dataset'].time_ns[context['dataset'].time_ns<=250]
native_audit=[];points=[];halvings=[]
for path in sorted((source/'verification').glob('*/result.json')):
    result=read(path);folder=path.parent
    all_steps=pd.read_csv(folder/'verification.csv');fine=all_steps[all_steps.dt_ns==all_steps.dt_ns.min()]
    conv=pd.read_csv(folder/'convergence.csv')
    for pair,group in conv[conv.target].groupby(['coarse_dt_ns','fine_dt_ns']):
        halvings.append(dict(case=folder.name,kind=result['specification']['kind'],role=result['role'],
            coarse_dt_ns=pair[0],fine_dt_ns=pair[1],feature_pairs_passing=int(group.features_converged.sum()),
            waveform_pairs_passing=int(group.waveform_converged.sum()),records=len(group)))
    if result['role']!='candidate':continue
    for row in fine[fine.pilot_target].itertuples():
        points.append(dict(kind=result['specification']['kind'],current_uA=row.measured_current_A*1e6,
            measured_frequency_MHz=row.measured_frequency_MHz,predicted_frequency_MHz=row.predicted_frequency_MHz,
            measured_Vpp_mV=row.measured_robust_vpp_mV,predicted_Vpp_mV=row.predicted_robust_vpp_mV,
            measured_mean_mV=row.measured_mean_mV,predicted_mean_mV=row.predicted_mean_mV,
            valid_cycles=row.frequency_valid,temperature_valid=row.temperature_valid))
    for path in sorted((folder/'native').glob('*.npz')):
        label=float(path.stem.removeprefix('current_'));row=fine[fine.source_label_mV==label].iloc[0]
        with np.load(path) as out:
            assert all(np.isfinite(out[key]).all() for key in out.files)
            v=out['V_vo2'].astype(float);r=out['R'].astype(float)
            heat=out['P_primary'].astype(float)+out['P_secondary'].astype(float) if 'P_primary' in out.files else out['P'].astype(float)
            error=relative(heat,v*v/r);assert error<1e-6
            current_error=voltage_error=None
            if result['specification']['kind']=='warm_parallel':
                current_error=relative(out['I_primary'].astype(float)+out['I_secondary'].astype(float),v/r)
                assert current_error<1e-6
            elif result['specification']['kind']=='series_bottleneck':
                current_error=relative(out['I_primary'].astype(float),out['I_secondary'].astype(float))
                voltage_error=relative(out['V_primary'].astype(float)+out['V_secondary'].astype(float),v)
                assert current_error<1e-6 and voltage_error<1e-6
            elif result['specification']['kind']=='phase_lag':
                assert out['g_dyn'].min()>=0 and out['g_dyn'].max()<=1
                current_error=relative(out['I_device'].astype(float),v/r);assert current_error<1e-6
            voltage=np.interp(times*1e-9,out['t'].astype(float),v)*1e3
            features=settled_features(times,voltage,[150.,250.],method='extrema')
            for field in ['frequency_MHz','robust_vpp_mV','mean_mV']:
                expected=row['predicted_'+field];actual=features[field]
                if actual is None:assert pd.isna(expected)
                else:assert np.isclose(actual,expected,rtol=1e-10,atol=1e-8)
            native_audit.append(dict(case=folder.name,kind=result['specification']['kind'],source_label_mV=label,
                samples=len(v),heat_partition_error=error,current_partition_error=current_error,
                series_voltage_error=voltage_error,features_reconstructed=True))
assert len(native_audit)==20
audit=dict(original_hashes_verified=len(hashes),native_histories_audited=len(native_audit),
    native_features_reconstructed=True,maximum_relative_heat_partition_error=max(x['heat_partition_error'] for x in native_audit),
    source_commit=manifest['provenance']['commit'],original_run_id=source.name,calibration_adopted=False,
    scope='No new numerical integrations; independent saved-channel identities and feature reconstruction')

bundle=RunBundle.create(name='Reviewed partial-switching material-model pilot 20261028',
    model='reviewed-material-model-pilot',kind='analysis',config=read(source/'resolved_config.json'),
    output_root=project/'runs/local',command='Audit saved pilot outputs and curate a separate reviewed bundle')
preserved={};mapping={}
aliases={'run.json':'original_run.json','report.md':'original_report.md','START_HERE.html':'original_START_HERE.html',
         'status.json':'original_status.json','artifact_hashes.json':'original_artifact_hashes.json'}
for path in sorted(source.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.npz':continue
    relative=path.relative_to(source).as_posix();destination=aliases.get(relative,relative)
    shutil.copy2(path,bundle.add_artifact(destination,label='Unmodified pilot evidence '+relative))
    preserved[destination]=digest(path);mapping[relative]=destination
pd.DataFrame(native_audit).to_csv(bundle.add_artifact('native_audit.csv',label='Independent saved-channel accounting and feature audit'),index=False)
pd.DataFrame(points).to_csv(bundle.add_artifact('pilot_points.csv',label='Three prespecified currents across all four mechanisms'),index=False)
pd.DataFrame(halvings).to_csv(bundle.add_artifact('timestep_audit.csv',label='Classifications and convergence at every halving'),index=False)
bundle.write_json('review_metrics.json',audit,label='Independent audit; original metrics remain unchanged')
bundle.write_json('preserved_artifact_hashes.json',preserved,label='Byte-identical original evidence inventory')
inventory={}
for path in sorted(source.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts:
        relative=path.relative_to(source).as_posix()
        inventory[relative]=dict(sha256=digest(path),bytes=path.stat().st_size,
            included_in_review=relative in mapping,review_path=mapping.get(relative))
bundle.write_json('full_local_inventory.json',inventory,label='Full numerical run hashes and explicit native-array exclusions')
comparison=pd.read_csv(source/'comparison.csv');main=comparison[comparison.role=='candidate']
table='| Model | Valid targets | Defined frequencies | Frequency MAPE | Vpp MAPE | Mean RMSE (mV) | Max T (K) |\n|---|---:|---:|---:|---:|---:|---:|\n'
for row in main.itertuples():
    table+=f'| {row.label} | {row.passing_targets}/11 | {row.defined_frequencies}/11 | {row.frequency_MAPE_percent:.2f}% | {row.amplitude_MAPE_percent:.1f}% | {row.mean_voltage_RMSE_mV:.2f} | {row.maximum_temperature_K:.2f} |\n'
long=[]
for path in sorted((source/'long_controls').glob('*/result.json')):
    result=read(path);frame=pd.read_csv(path.parent/'features.csv')
    long.append(dict(kind=result['kind'],joint_pass=result['persistent_physical_and_converged'],
        sustained_rows=int(frame.sustained.sum()),physical_rows=int((frame.temperature_valid&frame.proximity_admissible).sum()),
        rows=len(frame),maximum_temperature_K=float(frame.maximum_temperature_K.max())))
bundle.write_json('long_control_review.json',long,label='Preserved long-control classifications and physical exclusions')
report=f'''# Reviewed material-model pilot — 28 October 2026

**The three optional mechanisms did not produce a validated fit in this bounded pilot.**
No selected candidate reproduces the decreasing-amplitude trend while retaining all three
pilot target oscillations, and none passes the complete all-current acceptance criteria.
No calibration is adopted. This sparse comparison does not rule out spatial localization,
finite phase dynamics or a different thermal/geometry mapping.

## Coverage and final comparisons

All {metrics['total_screened_cases']} screen cases completed: {metrics['declared_cases']} declared
grid points and {metrics['refinement_cases']} shared one-coordinate probes. Seven retained
cases (four candidates plus three exact-reference-vector mechanism-off controls) replay all
22 inputs at four timesteps. Four 2 µs constant-current cases use three pilot currents and
the three finest timesteps. Elapsed numerical time: {metrics['elapsed_s']/60:.2f} minutes.

Final values use 0.003125 ns. Frequency MAPE uses the stated defined denominator; Vpp
uses all 11 targets, and mean-voltage RMSE all 22 records. The configured temperature
domain is 284.8545–375.1625 K. All rows fail full waveform convergence and shape acceptance.

{table}

The series bottleneck has smaller Vpp errors partly because it loses oscillations:
at the pilot currents, predicted Vpp is 0, 312.6 and 0 mV versus measured 101.8, 34.8
and 9.6 mV. It retains only four of 11 target records. This is quenching, not a fit.
Warm parallel regions retain 10 targets and frequency MAPE ~15.1%, but their swings
remain too large and temperatures exceed calibration. A 6 ns finite phase response
reduces some swings but slows the oscillations, retains nine targets and also exceeds
the temperature domain. These are conditional trade-offs, not accepted improvements.

The pilot's baseline is a different bounded selection from the previous overnight
f=1 diagnostic (10/11 targets; frequency MAPE 16.59%; Vpp MAPE 1437.28%; mean RMSE
35.52 mV). Screening here prioritizes physical trajectories on five records at
0.025 ns before cycles and feature accuracy; the selected baseline sacrifices cycles.
Do not replace the earlier retained reference with this seven-target pilot selection
or interpret the reduced Vpp error as evidence of a better overall fit. Coarse exclusions
may change under refinement; the pilot is not a complete scan of fine-step feasibility.

## Physical model and comparison limits

The original ideal-current solver is unchanged. The spatial variants separately test
warm parallel paths and a series VO₂ bottleneck/access path, with correct branch-current,
internal-voltage and heat allocation. Uniform-temperature resistances recover the same
measured R(T). Two passive thermal nodes exchange equal/opposite heat. Finite phase lag
is a separate phenomenological state relaxing toward the original hysteresis target;
no separate latent-heat model is added. The zero-lag limit calls the exact original solver.

The actual weights tested are 0.25, 0.5, 0.75, secondary capacity ratios 10/100 and
coupling ratios 0/1; no continuous geometry optimization was performed. Phase times
are 0.2, 0.6, 2, 6 ns. The retained lag reaches the upper trial value; this does not
identify a phase time. Static R(T) and ambient are fixed. Reference-budget partitions,
secondary thermal ratios and local constitutive-law transfer are diagnostic hypotheses.
No parameter is fitted separately for each current. All records and historical anchors
were seen before, so there is no blind validation claim. See original_report.md for
the full equations, feature score and long-control protocol.

## Integrity and independent checks

Every original artifact hash ({len(hashes)}) and source-file hash passes verification.
All 20 saved finest-step native histories reconstruct the saved measured-grid voltage
features. Joule-heat partitions agree with V²/R within maximum relative error
{audit['maximum_relative_heat_partition_error']:.3g}; saved parallel currents and series
voltage/current identities pass independently. Both regional temperatures are included
in physical-domain checks. Independent analytic RC/thermal/phase controls and 116
pre-change batched-waveform arrays verify the optional implementation and preservation
of existing output. Software checks pass 246 tests and validation of 51 recipes.

## Next step

This pilot supplies no robust reason to commit to one of these material extensions.
Obtain the actual TIA circuit/probe-node and conversion/feedback-current information
before another large fit. The previous fixed-gain/offset audit already failed to
explain the sweep; a circuit hypothesis must represent actual feedback/load dynamics.
If continuing with current data, use narrowly justified geometry or minor-loop/phase
measurements to constrain an extension rather than adding freely varying f(t).
Finer-step checks and all-current controls remain mandatory.

## Preservation

Original metrics/configuration, input data, full executable source, all 119 case
tables/specifications, all seven replay/ablation tables, long-control tables, figures
and offline viewer are byte-preserved separately from this review. The active manifest
contains only files actually included. Twenty bulky native NPZ arrays stay in the
full local numerical bundle; their hashes, sizes and exclusion are recorded in
full_local_inventory.json. Those omitted bytes are not backed up to GitHub by this
compact review. Source numerical commit: {manifest['provenance']['commit']}.
The source and review bundles are distinct, and no existing public bundle was edited.
'''
bundle.write_text('report.md',report,label='Reviewed outcomes, trade-offs, numerical limits and next step')
page=(source/'START_HERE.html').read_text()
notice='''<div class="note"><strong>Reviewed outcome: no validated fit.</strong>
All 119 cases and the full-current/long-control checks completed. The series model reduces swings
partly by losing oscillations; warm surroundings and phase lag retain oversized swings and exceed
the calibrated temperature domain. The selected pilot baseline is a different bounded selection from
the earlier overnight reference. <a href="report.md">Read the reviewed interpretation and limitations</a>.
Large native arrays remain local, with explicit hashes and exclusion flags.</div>'''
page=page.replace('<h1>Partial-switching material-model pilot</h1>','<h1>Reviewed material-model pilot</h1>'+notice)
bundle.write_text('START_HERE.html',page,label='Reviewed offline all-current comparisons',media_type='text/html')
shutil.copy2(__file__,bundle.add_artifact('review_audit.py',label='Independent audit and archival review generation'))
curated={path.relative_to(bundle.root).as_posix():digest(path) for path in bundle.root.rglob('*')
    if path.is_file() and path.name not in ['run.json','artifact_hashes.json']}
bundle.write_json('artifact_hashes.json',curated,label='Included compact-review artifact hashes')
bundle.complete(summary=dict(metrics,reviewed=True,source_run_id=source.name,native_histories_audited=20))
for name,expected in preserved.items():assert digest(bundle.root/name)==expected,name
record=RunRegistry(project).publish(bundle.id)
(project/'runs/scratch/material_pilot_review_path.txt').write_text(str(record.root)+'\n')
print(json.dumps(dict(published_review=str(record.root),review_metrics=audit,long_controls=long),indent=2))
