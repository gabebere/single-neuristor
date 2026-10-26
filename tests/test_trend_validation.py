"""Physical clock covariance and honest unresolved-comparison contracts."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.config import load_toml
from neuristor.workflows import _current_params_from_config
from neuristor.current_drive_sim import simulate_current_parameter_grid
from neuristor.orchestration.trend_validation import plan_cases,validate_settings
from neuristor.analysis.trend_validation import histories_agree,controlled_trend_pairs,scaling_comparisons,verified_scaling_summary,rising_crossing_frequency

ROOT=Path(__file__).resolve().parents[1]
RECIPE=ROOT/'experiments/current/validate_parameter_trends_20261020.toml'


def test_prespecified_plan_holds_other_parameters_fixed():
    s=load_toml(RECIPE)['trend_validation'];validate_settings(s)
    cases=plan_cases(s)
    assert len(cases)==108
    for c in cases:
        anchor=next(a for a in s['anchors'] if a['name']==c['anchor'])
        if c['family']=='local':
            for axis in ('C_pF','C_th_pJ_per_K','S_e_mW_per_K'):
                np.testing.assert_allclose(c[axis],anchor[axis]*(c['factor'] if axis==c['axis'] else 1))
        if c['family']=='scale':
            assert np.isclose(c['C_pF']/c['C_th_pJ_per_K'],anchor['C_pF']/anchor['C_th_pJ_per_K'])


def test_actual_solver_binary_common_scale_preserves_coupled_history():
    config=load_toml(RECIPE);p,*_=_current_params_from_config(config)
    p=replace(p,t_end_s=100e-9,dt_s=.0125e-9)
    def run(q,scale):
        return simulate_current_parameter_grid(500,q,C_F=np.array([3e-12*scale]),
            C_th_J_per_K=np.array([.072e-12*scale]),S_e_W_per_K=np.array([.006e-3]))[0]
    a=run(p,1);b=run(replace(p,dt_s=p.dt_s*2,t_end_s=p.t_end_s*2),2)
    assert len(a['t'])==len(b['t'])
    for key in ['V_vo2','T','R']:
        np.testing.assert_allclose(a[key],b[key],rtol=1e-6,atol=1e-6)


def test_unresolved_horizons_do_not_become_a_direction():
    assert not histories_agree([{'classification':'time_unresolved'}]*2,.02)
    rows=[]
    for factor,kind in [(.75,'oscillatory'),(1.,'calibration_censored'),(1.25,'oscillatory')]:
        rows.append(dict(family='local',anchor='A',scenario='s',axis='C_pF',factor=factor,
            case_id=str(factor),classification=kind,timestep_verified=kind=='oscillatory',horizon_verified=True,
            frequency_MHz=20/factor,extrema_vpp_mV=100,
            frequency_MHz_max_step_change=.001,extrema_vpp_mV_max_step_change=.001))
    pairs=controlled_trend_pairs(pd.DataFrame(rows),{'direction_threshold_fraction':.02})
    assert len(pairs)==2 and not pairs.usable.any()
    assert set(pairs.frequency_MHz_direction)=={'unresolved'}


def test_scaling_checks_distinguish_features_and_sampled_clock():
    rows=[];paths={}
    for i,scale in enumerate([.5,1,2]):
        rows.append(dict(family='scale',anchor='A',protocol='fixed_dt',nominal_dt_ns=.01,
            normalized_duration_ns=100,scale=scale,classification='oscillatory',
            integration_id=i,frequency_MHz=20/scale,extrema_vpp_mV=100))
        t=np.linspace(0,100,128)
        paths[i]={'time_ns':t*scale,'voltage_V':np.sin(t),'temperature_K':300+np.cos(t)}
    checks=scaling_comparisons(rows,paths,.02)
    assert len(checks)==2 and checks.feature_scaling_pass.all()
    assert (checks.voltage_path_normalized_rmse<1e-12).all()
    summary=pd.DataFrame(rows).assign(case_id=['half','unit','double'],timestep_verified=True,horizon_verified=True,dt_ns=.01)
    final=verified_scaling_summary(summary,.02)
    assert final.within_tolerance.all()
    summary.loc[summary.scale==.5,'horizon_verified']=False
    final=verified_scaling_summary(summary,.02)
    assert not final[final.scale==.5].within_tolerance.any()
    assert final[final.scale==.5].frequency_scaling_error.isna().all()


def test_validate_trends_cli_low_current_bundle(tmp_path):
    recipe=RECIPE.read_text()
    # Keep one anchor and scenario; low current gives a resolved equilibrium.
    recipe=recipe[:recipe.index('[[trend_validation.anchors]]')]+'''
[[trend_validation.anchors]]
name = "A"
C_pF = 3.0
C_th_pJ_per_K = 0.072
S_e_mW_per_K = 0.006
[[trend_validation.scenarios]]
name = "low_current_control"
current_uA = 100.0
T0_K = 298.0
'''
    recipe=recipe.replace('../../data/parameters/',str(ROOT/'data/parameters')+'/')
    recipe=recipe.replace('dt_ns = [0.0125, 0.00625, 0.003125]','dt_ns = [0.5, 0.25, 0.125]')
    path=tmp_path/'smoke.toml';path.write_text(recipe)
    result=CliRunner().invoke(app,['analyze','validate-trends','--config',str(path),'--output-root',str(tmp_path/'runs/local')])
    assert result.exit_code==0,result.output+str(result.exception)
    bundle=next((tmp_path/'runs/local').iterdir())
    assert json.loads((bundle/'run.json').read_text())['status']=='completed'
    table=pd.read_csv(bundle/'case_summary.csv')
    assert (table.classification=='nonoscillatory').all()
    assert table.timestep_verified.all() and table.horizon_verified.all()
    assert not pd.read_csv(bundle/'trend_pairs.csv').usable.any()
    original=(bundle/'integrations.csv').read_bytes()
    extended=tmp_path/'extended.toml'
    extended.write_text(recipe.replace('maximum_duration_ns = 1200.0','maximum_duration_ns = 2400.0'))
    followup=CliRunner().invoke(app,['analyze','validate-trends','--config',str(extended),
        '--reuse-integrations',str(bundle),'--output-root',str(tmp_path/'extended_runs')])
    assert followup.exit_code==0,followup.output+str(followup.exception)
    fresh=next((tmp_path/'extended_runs').iterdir())
    metrics=json.loads((fresh/'metrics.json').read_text())
    assert metrics['reused_integrations']>0 and metrics['new_integrations']==0
    assert (bundle/'integrations.csv').read_bytes()==original
    changed=tmp_path/'changed.toml'
    changed.write_text(extended.read_text().replace('gamma = 0.956269682','gamma = 0.8'))
    rejected=CliRunner().invoke(app,['analyze','validate-trends','--config',str(changed),
        '--reuse-integrations',str(bundle),'--output-root',str(tmp_path/'rejected_runs')])
    assert rejected.exit_code!=0 and 'protocol differs' in str(rejected.exception)


def test_independent_crossings_recover_known_frequency_and_keep_missing_values():
    t=np.arange(0,1000,.5);v=.4+.1*np.sin(2*np.pi*t/50)
    assert np.isclose(rising_crossing_frequency(t,v),20,rtol=1e-4)
    assert rising_crossing_frequency(t,np.ones(len(t))) is None
    assert rising_crossing_frequency(t,.4+.1*np.sin(2*np.pi*t/400)) is None
