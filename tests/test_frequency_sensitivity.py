"""Per-record derivatives must remain usable without inventing a full-data loss."""
import json

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from neuristor.analysis.frequency_sensitivity import frequency_stencils
from neuristor.cli import app


def profiles(modes=('frequency_only',),missing_first=True):
    samples=[];evaluations=[]
    for mode in modes:
        for dt in [.025,.0125,.00625]:
            for direction in [-1,0,1]:
                unit=[.5+direction*.01,.5,.5,.5]
                samples.append(dict(center_name=mode,dt_ns=dt,coordinate='center' if direction==0 else 'C_pF',
                    step=0. if direction==0 else .01,direction=direction,unit_json=json.dumps(unit)))
                records=[]
                for i,current in enumerate([230.,420.,605.]):
                    f=None if missing_first and i==0 else 55.+2*(unit[0]-.5)
                    records.append(dict(source_label_mV=150+i*50,measured_current_A=current*1e-6,
                        predicted_frequency_MHz=f,measured_frequency_MHz=50.,frequency_valid=f is not None))
                evaluations.append(dict(center_name=mode,dt_ns=dt,unit=unit,per_current=records))
    return pd.DataFrame(samples),evaluations


def test_frequency_derivatives_keep_defined_inputs_when_another_input_is_missing():
    samples,evaluations=profiles()
    result=frequency_stencils(samples,evaluations)
    good=result[result.source_label_mV!=150]
    np.testing.assert_allclose(good.frequency_centered_slope,2.)
    assert good.frequency_derivative_defined.all()
    absent=result[result.source_label_mV==150]
    assert not absent.frequency_derivative_defined.any()
    assert absent.frequency_centered_slope.isna().all()


def test_optimizer_review_cli_saved_artifact_smoke(tmp_path):
    common=dict(bounds={},target_current_range_uA=[225.,610.],window_ns=[150.,250.],
                search_dt_ns=.025,fixed_parameters=['T0_K'],feature_method='extrema',loss_function='relative_squared')
    rows=[]
    for dt in [.025,.0125,.00625]:
        for i,current in enumerate([40.,230.,420.,605.]):
            rows.append(dict(dt_ns=dt,source_label_mV=100+i*50,measured_current_A=current*1e-6,
                target=i>0,measured_frequency_MHz=50. if i else np.nan,measured_extrema_vpp_mV=20. if i else 0.,
                predicted_frequency_MHz=55. if i else np.nan,predicted_extrema_vpp_mV=40. if i else 0.,
                predicted_mean_mV=200.,frequency_relative_error=.1 if i else np.nan,
                robust_amplitude_relative_error=1. if i else np.nan,frequency_valid=i>0,sustained=i>0,
                predicted_sustained=i>0,outside_domain=False,proximity_admissible=True))
    data=pd.DataFrame(rows)
    audit=tmp_path/'audit';audit.mkdir()
    (audit/'run.json').write_text(json.dumps({'status':'completed'}))
    (audit/'resolved_config.json').write_text(json.dumps({'steady':common}))
    (audit/'metrics.json').write_text('{}')
    samples,evaluations=profiles(('frequency_only','frequency_amplitude'),False)
    samples.to_csv(audit/'profile_samples.csv',index=False)
    (audit/'evaluations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in evaluations))
    pd.DataFrame([dict(center_name='frequency_only')]).to_csv(audit/'gradient_summary.csv',index=False)
    all_frames=[];all_params=[]
    for mode in ['frequency_only','frequency_amplitude']:
        for method in ['baseline','gradient']:
            candidate=f'{mode}_{method}'
            all_frames.append(data.assign(candidate=candidate,mode=mode))
            all_params.append(dict(candidate=candidate,gamma=.8))
    pd.concat(all_frames).to_csv(audit/'verification.csv',index=False)
    pd.DataFrame(all_params).to_csv(audit/'parameters.csv',index=False)
    parents=[]
    for mode,weight in [('frequency_only',0.),('frequency_amplitude',1.)]:
        source=tmp_path/mode;source.mkdir();parents.append(source)
        for file,value in [('run.json',{'status':'completed'}),('resolved_config.json',{'steady':dict(common,amplitude_weight=weight)}),
                           ('metrics.json',{'verification_complete':True}),('search_result.json',{})]:
            (source/file).write_text(json.dumps(value))
        data.assign(candidate='search_1').to_csv(source/'verification.csv',index=False)
        pd.DataFrame([dict(candidate='search_1',gamma=.8)]).to_csv(source/'parameters.csv',index=False)
        (source/'optimization_history.jsonl').write_text('{}\n')
    result=CliRunner().invoke(app,['analyze','review-optimizer','--audit-run',str(audit),
        '--frequency-run',str(parents[0]),'--combined-run',str(parents[1]),'--output-root',str(tmp_path/'runs/scratch/documents')])
    assert result.exit_code==0,result.output+repr(result.exception)
    bundle=next((tmp_path/'runs/scratch/documents').glob('*/run.json')).parent
    assert json.loads((bundle/'metrics.json').read_text())['method_candidates']==6
    assert len(pd.read_csv(bundle/'method_summary.csv'))==18
    assert (bundle/'figures/frequency_profiles.png').is_file()
