import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.steady_inference import settled_features, settled_loss

ROOT=Path(__file__).resolve().parents[1]
SETTINGS=dict(frequency_tolerance_fraction=.025,amplitude_tolerance_fraction=.075)


def test_settled_loss_ignores_startup_and_dc_but_not_frequency_or_amplitude():
    t=np.arange(-200.,301.)
    y=200+20*np.sin(2*np.pi*.05*t)
    m=settled_features(t,y)
    changed=y+500
    changed[t<150]+=10000*np.exp(-abs(t[t<150])/30)
    p=settled_features(t,changed)
    assert settled_loss(m,p,SETTINGS)['loss']<1e-10
    wrong_amp=settled_features(t,200+40*np.sin(2*np.pi*.05*t))
    wrong_frequency=settled_features(t,200+20*np.sin(2*np.pi*.08*t))
    assert settled_loss(m,wrong_amp,SETTINGS)['loss']>10
    assert settled_loss(m,wrong_frequency,SETTINGS)['loss']>10
    flat=settled_features(t,np.full_like(t,200.))
    assert not settled_loss(m,flat,SETTINGS)['frequency_valid']
    assert settled_loss(m,flat,SETTINGS)['loss']>100


def test_settled_loss_penalizes_decay():
    t=np.arange(-200.,301.)
    m=settled_features(t,200+20*np.sin(2*np.pi*.05*t))
    p=settled_features(t,200+20*np.sin(2*np.pi*.05*t)*np.exp(-np.maximum(t-150,0)/40))
    assert p['retention']<.5
    assert not settled_loss(m,p,SETTINGS)['near_target']


def test_steady_cli_small_budget_checkpoint_and_verification(tmp_path):
    result=CliRunner().invoke(app,['analyze','fit-steady','--config',str(ROOT/'experiments/current/specimen_steady_multistart.toml'),
        '--set','steady.restarts=1','--set','steady.population=5','--set','steady.maximum_generations=0',
        '--set','steady.local_evaluations=1','--set','steady.search_seconds=30.0',
        '--set','steady.total_seconds=180.0','--set','steady.search_dt_ns=0.5',
        '--set','steady.verification_dt_ns=[0.5]','--set','steady.verification_candidates=0',
        '--output-root',str(tmp_path)])
    assert result.exit_code==0,result.output+repr(result.exception)
    root=next(tmp_path.glob('*/run.json')).parent
    assert json.loads((root/'run.json').read_text())['status']=='completed'
    assert json.loads((root/'status.json').read_text())['state']=='completed'
    checkpoint=json.loads((root/'checkpoint.json').read_text())
    assert checkpoint['unique_candidates']>=5
    assert checkpoint['best']['static_rmse_log10']<=.05
    assert (root/'optimization_history.jsonl').is_file()
    assert (root/'summary.csv').is_file()
    assert json.loads((root/'metrics.json').read_text())['verification_complete']
