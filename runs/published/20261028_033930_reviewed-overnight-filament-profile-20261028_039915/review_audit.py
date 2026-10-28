"""Post-run numerical audit and compact archival copy; never mutates the source run."""
from pathlib import Path
import copy
from dataclasses import replace
import hashlib
import html
import json
import shutil

import numpy as np
import pandas as pd

from neuristor.current_drive_sim import filament_cold_reference
from neuristor.fitting.filament_profile import acceptance, diagnostic_rank
from neuristor.fitting.joint_inference import joint_model
from neuristor.fitting.steady_inference import settled_features
from neuristor.orchestration.filament_profile import _context, _verify_frozen, _verify_completed
from neuristor.orchestration.filament_study import summarize
from neuristor.runs import RunBundle

PROJECT=Path('/Users/gabrielberezovsky/Documents/Projects/single-neuristor')
SOURCE=PROJECT/'runs/local/20261027_193458_overnight-fixed-f-filament-profile-and-cross-cur_5e8844'
def read(path):
    return json.loads(path.read_text())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()
def relative_error(a,b):
    return float(np.max(np.abs(a-b)/np.maximum(np.maximum(np.abs(a),np.abs(b)),1e-30)))

_verify_frozen(SOURCE)
state=read(SOURCE/'state.json');_verify_completed(SOURCE,state)
metrics=read(SOURCE/'metrics.json');assert metrics['coverage_complete'] and metrics['state']=='completed'
context=_context(read(SOURCE/'execution_config.json'))
profile=context['config']['profile_search']
rows=[];native_rows=[];profile_results=[];controls=[]
for stage,entry in state['completed'].items():
    result=entry['result']
    if result['kind']=='verification':
        frame=pd.read_csv(SOURCE/'stages'/stage/'verification.csv')
        finest=frame[frame.dt_ns==frame.dt_ns.min()]
        convergence=pd.read_csv(SOURCE/'stages'/stage/'convergence.csv')
        repeated=acceptance(finest,convergence,profile['success'])
        assert repeated['gates']==result['acceptance']['gates']
        target=finest[finest.target]
        # Normalize NaN losses introduced by DataFrame round trips. The frozen
        # helper recognizes None, and never assigns zero error to a missing cycle.
        records=target.to_dict('records')
        for record in records:
            if pd.isna(record.get('loss')):
                record['loss']=None
        audited=summarize(records)
        corrected=dict(result,all_target=audited)
        if not stage.startswith('final_'):
            profile_results.append(corrected)
        c=result['candidate'];a=result['acceptance'];conv=convergence[convergence.target]
        rows.append(dict(stage=stage,family=c['family'],fraction=c['fraction'],objective=c['objective_label'],
            fit_records=len(c['training']),passing_targets=a['passing_targets'],defined_frequencies=a['defined_frequencies'],
            frequency_MAPE_percent=None if a['frequency_MAPE_defined'] is None else 100*a['frequency_MAPE_defined'],
            amplitude_MAPE_percent=100*a['amplitude_MAPE_all'],mean_voltage_RMSE_mV=a['mean_voltage_RMSE_mV'],
            mean_voltage_RMSE_targets_mV=float(np.sqrt(np.mean((target.predicted_mean_mV-target.measured_mean_mV)**2))),
            all_record_maximum_temperature_K=float(finest.temperature_max_K.max()),
            outside_temperature_records=int((~finest.temperature_valid).sum()),
            inadmissible_proximity_records=int((~finest.proximity_admissible).sum()),
            false_positive_quiet_controls=a['false_positive_controls'],
            feature_halving_pairs_passing=int(conv.features_converged.sum()),
            waveform_halving_pairs_passing=int(conv.waveform_converged.sum()),halving_pairs=len(conv),
            original_partial_loss=result['all_target']['dynamic_loss'],audited_partial_loss=audited['dynamic_loss'],
            audited_partial_denominator=int(target.loss.notna().sum()),accepted_loss=audited['objective_loss'],
            every_prespecified_short_gate=a['success'],failed_gates=', '.join(a['failed_gates'])))
        if not stage.startswith('final_'):
            continue
        params=joint_model(c['values'],replace(context['base'],filament_fraction=c['fraction']),min(result['steps']))
        cold=float(filament_cold_reference(params))
        times=context['dataset'].time_ns[context['dataset'].time_ns<=250.]
        for path in sorted((SOURCE/'stages'/stage/'native').glob('*.npz')):
            label=float(path.stem.removeprefix('current_'));saved=finest[finest.source_label_mV==label].iloc[0]
            with np.load(path) as native:
                v=native['voltage_V'];hot=native['reference_hot_resistance_ohm'];effective=native['effective_resistance_ohm']
                i_fil=native['filament_current_A'];i_sur=native['surrounding_current_A']
                p_fil=native['filament_power_W'];p_sur=native['surrounding_power_W']
                assert all(np.isfinite(native[key]).all() for key in native.files)
                i_balance=relative_error(i_fil+i_sur,v/effective)
                p_balance=relative_error(p_fil+p_sur,v*v/effective)
                hot_balance=relative_error(i_fil,np.float32(c['fraction'])*v/hot)
                cold_balance=relative_error(i_sur,np.float32(1-c['fraction'])*v/cold)
                assert max(i_balance,p_balance,hot_balance,cold_balance)<6e-7
                volts=np.interp(times,native['time_ns'],v*1e3)
                features=settled_features(times,volts,context['config']['steady']['window_ns'],method='extrema')
                errors=[]
                for key in ['frequency_MHz','robust_vpp_mV','mean_mV']:
                    expected=saved['predicted_'+key];actual=features[key]
                    if actual is None:
                        assert pd.isna(expected)
                    else:
                        errors.append(abs(actual-expected));assert np.isclose(actual,expected,rtol=1e-10,atol=1e-8)
                assert features['sustained']==saved.predicted_sustained
                native_rows.append(dict(stage=stage,source_label_mV=label,samples=len(v),
                    current_balance_relative_error=i_balance,power_balance_relative_error=p_balance,
                    hot_branch_relative_error=hot_balance,cold_branch_relative_error=cold_balance,
                    maximum_recomputed_feature_difference=max(errors,default=0.)))
    elif result['kind']=='constant-control':
        frame=pd.read_csv(SOURCE/'stages'/stage/'constant_current.csv')
        for current,group in frame.groupby('current_uA'):
            group=group.sort_values('dt_ns',ascending=False)
            feature_ok=True
            changes={}
            for field in ['frequency_MHz','robust_vpp_mV']:
                values=group[field].to_numpy(float)
                valid=bool(np.isfinite(values).all() and np.all(values>0))
                change=float(np.max(np.abs(np.diff(values))/values[1:])) if valid else None
                changes[field+'_maximum_halving_change']=change
                feature_ok &= change is not None and change<=profile['success']['convergence_fraction']
            controls.append(dict(stage=stage,current_uA=current,sustained_all_steps=bool(group.sustained.all()),
                physical_all_steps=bool(group.temperature_valid.all() and group.proximity_admissible.all()),
                features_converged=bool(feature_ok),**changes,
                passing=bool(group.sustained.all() and group.temperature_valid.all() and group.proximity_admissible.all() and feature_ok)))

table=pd.DataFrame(rows);control_table=pd.DataFrame(controls);native_table=pd.DataFrame(native_rows)
assert len(native_table)==132
selection={}
for family in ['inherited','volume_scaled']:
    best=min([r for r in profile_results if r['candidate']['family']==family and r['candidate']['fraction']<1],key=diagnostic_rank)
    actual=read(SOURCE/'stages'/('refine_'+family+'_primary')/'task.json')['seed_records'][0]['stage']
    selection[family]=dict(actual_candidate=actual,audited_candidate=best['candidate']['stage'],unchanged=actual==best['candidate']['stage'])
    assert selection[family]['unchanged']

vectors=set();scoped=set();count=0
for path in sorted((SOURCE/'stages').glob('*/history.jsonl')):
    task=read(path.parent/'task.json')
    for line in path.open():
        record=json.loads(line);count+=1
        key=(float(task['fraction']),*map(float,record['values']))
        vectors.add(key);scoped.add((task['dt_ns'],task['mean_weight'],tuple(task['training']),*key))
assert count==metrics['total_unique_candidate_evaluations']
audit=dict(frozen_file_hashes_verified=len(read(SOURCE/'frozen_hashes.json')),completed_stage_hashes_verified=len(state['completed']),
    numerical_source_commit=read(SOURCE/'run.json')['provenance']['commit'],attempts=state['attempts'],
    native_histories_audited=len(native_table),stage_unique_evaluations=count,
    distinct_saved_physical_vectors=len(vectors),distinct_vectors_with_objective_dt_and_target_scope=len(scoped),
    native_balance_maximum_relative_error=float(native_table.iloc[:,3:7].max().max()),
    recomputed_acceptance_matches_original=True,all_final_native_features_match_saved_tables=True,
    summary_nan_issue='Missing-cycle losses became NaN in DataFrame summaries, erasing the partial loss diagnostic. Accepted objectives remain undefined. Finite-loss post-processing restores the diagnostic without changing either selected refinement seed.',
    audited_refinement_selection=selection,calibration_adopted=False)

bundle=RunBundle.create(name='Reviewed overnight filament profile 20261028',model='reviewed-overnight-filament-profile',kind='analysis',
    config=read(SOURCE/'resolved_config.json'),output_root=PROJECT/'runs/local',command='Post-run frozen-source audit and compact archival review')
copied={};original_to_review={}
def preserve(source,relative=None):
    relative=relative or source.relative_to(SOURCE).as_posix()
    shutil.copy2(source,bundle.add_artifact(relative,label='Unmodified original evidence: '+relative))
    copied[relative]=digest(source)
    if source.is_relative_to(SOURCE):
        original_to_review[source.relative_to(SOURCE).as_posix()]=relative

for name in ['metrics.json','resolved_config.json','execution_config.json','plan.json','current_split.csv','environment.json',
    'source_hashes.json','frozen_hashes.json','search_parameters.csv','verified_candidates.csv','adaptive_plan.json','refinement_plan.json']:
    preserve(SOURCE/name)
for name in ['run.json','state.json','status.json','report.md','process.json','launch.json']:
    preserve(SOURCE/name,'original_'+name)
for lane in ['source','inputs','figures']:
    for path in sorted((SOURCE/lane).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            preserve(path)
for folder in sorted((SOURCE/'stages').iterdir()):
    for name in ['task.json','result.json','seed_vectors.json','timing.json','verification.csv','convergence.csv','constant_current.csv']:
        if (folder/name).is_file():
            preserve(folder/name)
    if folder.name.startswith('final_'):
        preserve(folder/'traces.csv')
preserve(Path(__file__).resolve(),'review_audit.py')
table.to_csv(bundle.add_artifact('audited_verification.csv',label='Audit with correct partial losses and all-record maxima'),index=False)
table[table.stage.str.startswith('final_')].to_csv(bundle.add_artifact('final_comparison.csv',label='Six matched fine-step final comparisons'),index=False)
control_table.to_csv(bundle.add_artifact('long_control_audit.csv',label='Current-by-current persistence, domain and feature convergence'),index=False)
native_table.to_csv(bundle.add_artifact('native_conservation_audit.csv',label='Independent native branch-balance and feature replay audit'),index=False)
bundle.write_json('review_metrics.json',audit,label='Independent hash, native feature and selection audit')
bundle.write_json('preserved_artifact_hashes.json',copied,label='Byte-identical original evidence inventory')
inventory={}
for path in sorted(SOURCE.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts and path.name!='supervisor.lock':
        relative=path.relative_to(SOURCE).as_posix()
        inventory[relative]=dict(sha256=digest(path),bytes=path.stat().st_size,
            included_in_review=relative in original_to_review,review_path=original_to_review.get(relative))
bundle.write_json('full_local_inventory.json',inventory,label='Hashes and explicit exclusions for the full local numerical run')
(PROJECT/'runs/scratch/overnight_review_path.txt').write_text(str(bundle.root)+'\n')
print(json.dumps(dict(review_path=str(bundle.root),review_metrics=audit),indent=2))
