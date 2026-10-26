"""Cross-objective reporting must keep missing frequencies and source identities honest."""
import json

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from neuristor.analysis.loss_comparison import representative_records, verification_summary
from neuristor.cli import app


def sample_verification():
    rows=[]
    for dt in [.025,.0125,.00625]:
        for i,current in enumerate([40.,230.,420.,605.]):
            rows.append(dict(candidate='search_1',dt_ns=dt,source_label_mV=100+i*50,
                measured_current_A=current*1e-6,target=i>0,measured_frequency_MHz=50. if i else np.nan,
                measured_extrema_vpp_mV=20. if i else 0.,predicted_frequency_MHz=55. if i else np.nan,
                predicted_extrema_vpp_mV=40. if i else 0.,predicted_mean_mV=200.,
                frequency_relative_error=.1 if i else np.nan,robust_amplitude_relative_error=1. if i else np.nan,
                frequency_valid=i>0,sustained=i>0,predicted_sustained=i>0,
                outside_domain=False,proximity_admissible=True))
    return pd.DataFrame(rows)


def test_common_objectives_require_all_target_frequencies():
    data=sample_verification()
    summary=verification_summary(data)
    assert summary.eligible.all()
    np.testing.assert_allclose(summary.L_frequency,.01)
    np.testing.assert_allclose(summary.L_frequency_amplitude,1.01)
    data.loc[data.source_label_mV==150,'frequency_relative_error']=np.nan
    data.loc[data.source_label_mV==150,'frequency_valid']=False
    summary=verification_summary(data)
    assert not summary.eligible.any()
    assert summary.L_frequency.isna().all()
    assert (summary.defined_frequency_count==2).all()
    np.testing.assert_allclose(summary.frequency_MSE_defined,.01)
    np.testing.assert_allclose(summary.frequency_amplitude_MSE_defined,1.01)
    assert len(representative_records(data))==3


def test_compare_steady_cli_uses_saved_samples_and_current_titles(tmp_path):
    data=sample_verification()
    parents=[]
    for mode,aw in [('frequency',0.),('combined',1.)]:
        root=tmp_path/mode;root.mkdir();parents.append(root)
        payloads={'run.json':{'status':'completed'},
            'metrics.json':dict(verification_complete=True,unique_candidates=12,search_elapsed_s=2.),
            'resolved_config.json':dict(steady=dict(loss_function='relative_squared',frequency_weight=1.,amplitude_weight=aw)),
            'search_result.json':{'best':{'score':.1}}}
        for name,value in payloads.items():(root/name).write_text(json.dumps(value))
        data.to_csv(root/'verification.csv',index=False)
        pd.DataFrame([dict(candidate='search_1',gamma=.8)]).to_csv(root/'parameters.csv',index=False)
        traces=[]
        for row in data.itertuples():
            t=np.arange(0.,251.)
            traces.append(pd.DataFrame(dict(candidate=row.candidate,dt_ns=row.dt_ns,
                source_label_mV=row.source_label_mV,time_ns=t,
                measured_mV=200+10*np.sin(t*.3),predicted_mV=200+20*np.sin(t*.33))))
        pd.concat(traces).to_csv(root/'verification_traces.csv',index=False)
    result=CliRunner().invoke(app,['analyze','compare-steady','--frequency-run',str(parents[0]),
        '--combined-run',str(parents[1]),'--output-root',str(tmp_path/'output')])
    assert result.exit_code==0,result.output+repr(result.exception)
    bundle=next((tmp_path/'output').glob('*/run.json')).parent
    assert json.loads((bundle/'run.json').read_text())['status']=='completed'
    scope=json.loads((bundle/'metrics.json').read_text())
    np.testing.assert_allclose(scope['representative_current_uA'],[230.,420.,605.])
    assert (bundle/'figures/frequency_only_representatives.png').is_file()
    assert (bundle/'numerical_provenance.json').is_file()
