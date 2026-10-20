"""Prevent finite grids, horizons and operating screens from masquerading as bounds."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from neuristor.analysis.boundary_search import (
    AXES, BoundaryEvaluator, classify_trajectory, reconstruct_capacitance_scales,
    search_boundary_domain, validate_boundary_settings,
)
from neuristor.cli import app
from neuristor.config import load_toml
from neuristor.current_drive_sim import simulate_current_parameter_grid
from neuristor.workflows import _current_params_from_config

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / 'experiments/current/adaptive_boundary_maps.toml'


def settings():
    s = load_toml(RECIPE)['boundary']
    s['seed_points'] = []
    return s


def synthetic_evaluator(points, stage):
    # A bounded ratio band and Se band. Seed range deliberately clips both bands.
    rows = []
    for ratio, se in points:
        oscillatory = 2 < ratio < 30 and .002 < se < .014
        rows.append(dict(zip(AXES, (ratio, se))) | dict(classification=(
            'oscillatory' if oscillatory else 'nonoscillatory'),
            frequency_MHz=200. if oscillatory else None, extrema_vpp_mV=20. if oscillatory else 0.,
            duration_ns=1200., stage=stage, timestep_verified=stage=='fine'))
    return rows


def test_coarse_expansion_then_fine_brackets_and_frequency_screen_separation():
    s = settings()
    s.update(coarse_count=3, fine_count=5, maximum_expansions=5, max_coarse_points=500)
    s['axes'][AXES[0]].update(low=3., high=10., hard_low=.1, hard_high=100.)
    coarse, fine, bounds, result = search_boundary_domain(s, synthetic_evaluator)
    assert coarse[AXES[0]].min() < 2 and coarse[AXES[0]].max() > 30
    assert all(f['status'] == 'sampled_nonoscillatory' for f in result['coarse_faces'])
    assert len(bounds) and set(bounds.low_class) | set(bounds.high_class) == {'oscillatory','nonoscillatory'}
    assert not result['complete_physical_boundary']
    assert result['common_capacitance_scale'] == 'open_by_model_symmetry'
    assert fine[fine.classification == 'oscillatory'].frequency_MHz.min() > s['maximum_frequency_MHz']
    scaled = reconstruct_capacitance_scales(fine, s)
    assert not scaled.frequency_screen_pass.all()
    assert scaled.scale_boundary.eq('display_only_open').all()
    row = scaled.iloc[0]
    assert row.C_pF == row.reference_ratio * row.C_th_pJ_per_K
    assert np.isclose(row.frequency_MHz * row.scale, 200.)


def test_hard_caps_and_point_budgets_remain_open():
    s = settings()
    s.update(coarse_count=3, fine_count=3, max_coarse_points=9, max_fine_points=9)
    def all_oscillatory(points, stage):
        return [dict(zip(AXES, p)) | dict(classification='oscillatory',
            frequency_MHz=20., extrema_vpp_mV=100., duration_ns=600., stage=stage) for p in points]
    _, fine, bounds, result = search_boundary_domain(s, all_oscillatory)
    assert result['budget_exhausted']
    assert all(f['status']=='oscillatory_open' for f in result['coarse_faces'])
    assert len(fine)==9 and not len(bounds)
    # Returned face coordinates belong to evaluated points, never an unrun shell.
    assert result['coarse_faces'][0]['value'] == s['axes'][AXES[0]]['low']


def test_finer_discovery_expands_an_incorrect_coarse_envelope():
    s = settings(); s.update(coarse_count=3, fine_count=5, max_fine_points=500)
    s['axes'][AXES[0]].update(low=3., high=300., hard_low=.1, hard_high=1000.)
    s['axes'][AXES[1]].update(low=.001, high=.1, hard_low=.0001, hard_high=1.)
    def evaluate(points, stage):
        rows = []
        for p in points:
            # The initial 3x3 coarse grid sees the center only. Better integration
            # discovers an oscillator on the fine low-ratio face (3), so it must
            # sample farther outward to locate the real transition near ratio 2.
            osc = (p[0]==30. and p[1]==.01) if stage=='coarse' else (2 < p[0] < 80 and .002 < p[1] < .04)
            rows.append(dict(zip(AXES,p)) | dict(classification='oscillatory' if osc else 'nonoscillatory',
                frequency_MHz=20.,extrema_vpp_mV=100.,duration_ns=600.,stage=stage))
        return rows
    _, fine, _, result = search_boundary_domain(s,evaluate)
    assert fine[AXES[0]].min() < 2
    assert any(r['faces'][0]['status']=='oscillatory_open' for r in result['fine_rounds'])
    assert all(f['status']=='sampled_nonoscillatory' for f in result['fine_faces'])


def test_censored_midpoint_never_certifies_a_transition():
    s = settings(); s.update(coarse_count=3, fine_count=3)
    def evaluate(points, stage):
        out = []
        for p in points:
            ratio = p[0]
            cls = 'oscillatory' if ratio < 30 else 'nonoscillatory'
            # Midpoint 53.35 between 30 and 94.87 becomes a calibration exclusion.
            if 40 < ratio < 70:
                cls = 'calibration_censored'
            out.append(dict(zip(AXES,p)) | dict(classification=cls, frequency_MHz=20.,
                extrema_vpp_mV=100., duration_ns=600., stage=stage))
        return out
    _, _, bounds, result = search_boundary_domain(s, evaluate)
    assert any(bounds.status == 'calibration_censored')
    assert not bounds.loc[bounds.status=='calibration_censored', 'tolerance_reached'].any()


@pytest.mark.parametrize('dt_ns', [.05, .025, .0125])
def test_common_capacitance_scaling_preserves_physics_at_three_steps(dt_ns):
    params, current, *_ = _current_params_from_config(load_toml(RECIPE))
    params = replace(params, dt_s=dt_ns*1e-9, t_end_s=600e-9)
    def run(p, scale):
        return simulate_current_parameter_grid(current, p,
            C_F=np.array([3e-12*scale]), C_th_J_per_K=np.array([.072e-12*scale]),
            S_e_W_per_K=np.array([.006e-3]))[0]
    a = run(params,1)
    b = run(replace(params,dt_s=params.dt_s*4,t_end_s=params.t_end_s*4),4)
    for key in ('V_vo2','T','R'):
        np.testing.assert_allclose(a[key],b[key],rtol=2e-6,atol=1e-5)
    np.testing.assert_allclose(a['t']*4,b['t'],rtol=2e-6,atol=1e-14)


def test_absent_peaks_with_unbalanced_slow_transient_stays_unresolved():
    params, _, *_ = _current_params_from_config(load_toml(RECIPE))
    t = np.linspace(0,600,6001)*1e-9
    out = dict(t=t,V_vo2=np.full_like(t,.2),T=np.full_like(t,330),R=np.full_like(t,1000),
               I_in=np.full_like(t,.0005),minimum_proximity_slope=1.)
    assert classify_trajectory(out,params,settings())['classification']=='time_unresolved'
    out['T'][:] = 2494
    assert classify_trajectory(out,params,settings())['classification']=='calibration_censored'


def test_time_extension_and_nonoscillating_neighbors_verified(monkeypatch):
    import neuristor.analysis.boundary_search as module
    params, current, *_ = _current_params_from_config(load_toml(RECIPE))
    s=settings(); calls=[]
    monkeypatch.setattr(module, 'simulate_current_parameter_grid', lambda i,p,**kw: [
        dict(duration=p.t_end_s*1e9,dt=p.dt_s*1e9) for _ in kw['C_F']])
    def classification(out, p, s):
        calls.append((out['duration'],out['dt']))
        return dict(classification='nonoscillatory' if out['duration']>=600 else 'time_unresolved',
                    duration_ns=out['duration'],dt_ns=out['dt'],frequency_MHz=None,extrema_vpp_mV=0.)
    monkeypatch.setattr(module,'classify_trajectory',classification)
    evaluator=BoundaryEvaluator(params,current,s)
    point=(40.,.006)
    row=evaluator([point],'fine')[0]
    assert row['classification']=='nonoscillatory' and row['timestep_verified']
    assert len(calls)==9  # 300,600,1200 at all three steps; both longer horizons agree.
    evaluator([point],'fine')
    assert len(calls)==9


def test_unsafe_boundary_settings_rejected():
    s=settings();s['verification_dt_ns']=[.025,.1]
    with pytest.raises(ValueError,match='smaller'):
        validate_boundary_settings(s)


def test_disagreeing_classifications_require_three_agreeing_finer_steps(monkeypatch):
    params, current, *_ = _current_params_from_config(load_toml(RECIPE))
    s = settings(); evaluator = BoundaryEvaluator(params, current, s)
    def at_step(points, dt, stage):
        cls = 'nonoscillatory' if dt >= .02 else 'oscillatory'
        return {p: dict(classification=cls, frequency_MHz=20., extrema_vpp_mV=100.,
                        duration_ns=600., dt_ns=dt) for p in points}
    monkeypatch.setattr(evaluator, '_at_step', at_step)
    row = evaluator([(40., .006)], 'fine')[0]
    assert row['classification']=='oscillatory' and row['timestep_verified']
    assert row['verification_steps_ns']==[.0125,.00625,.003125]


def test_step_budget_produces_unresolved_rows_instead_of_a_false_equilibrium():
    params, current, *_ = _current_params_from_config(load_toml(RECIPE))
    s=settings(); s['max_steps_per_simulation']=100
    evaluator=BoundaryEvaluator(params,current,s)
    row=evaluator([(40.,.006)],'fine')[0]
    assert row['classification']=='step_budget' and not row['timestep_verified']
    assert row['frequency_MHz'] is None
    assert len(evaluator.verification)==3


def test_boundary_cli_bundle_handles_empty_oscillatory_domain(tmp_path):
    args=['analyze','boundary-maps','--config',str(RECIPE),'--output-root',str(tmp_path)]
    for setting in ['input.amplitude_uA=0', 'boundary.coarse_count=3','boundary.fine_count=3',
                    'boundary.initial_duration_ns=30','boundary.maximum_duration_ns=60',
                    'boundary.coarse_dt_ns=0.1','boundary.fine_dt_ns=0.1',
                    'boundary.verification_dt_ns=[0.05,0.025]', 'boundary.seed_points=[]']:
        args+=['--set',setting]
    result=CliRunner().invoke(app,args)
    assert result.exit_code==0,result.output+repr(result.exception)
    root=next(tmp_path.glob('*/run.json')).parent
    manifest=json.loads((root/'run.json').read_text())
    assert manifest['status']=='completed'
    for a in manifest['artifacts']:
        assert not Path(a['path']).is_absolute() and (root/a['path']).is_file()
    summary=json.loads((root/'metrics.json').read_text())
    assert summary['classifications']=={'nonoscillatory':9}
    assert summary['transition_brackets']==0 and not summary['complete_physical_boundary']
    assert 'display limits' in (root/'parameter_maps.html').read_text()
    # Exact reuse performs no new integrations and keeps original evidence separate.
    reused=CliRunner().invoke(app,args+['--reuse-exploration',str(root)])
    assert reused.exit_code==0,reused.output+repr(reused.exception)
    second=[p.parent for p in tmp_path.glob('*/run.json') if p.parent != root][0]
    reused_summary=json.loads((second/'metrics.json').read_text())
    assert reused_summary['integration_attempts']==0 and reused_summary['reused_evaluations']==18
    assert (second/'original_evaluations/verification_checkpoint.csv').read_bytes()==(root/'verification_checkpoint.csv').read_bytes()
    assert (second/'original_numerical_source/neuristor/current_drive_sim.py').is_file()
    chain=CliRunner().invoke(app,args+['--reuse-exploration',str(second)])
    assert chain.exit_code==0,chain.output+repr(chain.exception)
    third=[p.parent for p in tmp_path.glob('*/run.json') if p.parent not in (root,second)][0]
    assert (third/'reused_history/original_evaluations/verification_checkpoint.csv').read_bytes()==(root/'verification_checkpoint.csv').read_bytes()
    assert (third/'reused_history/parent_provenance.json').is_file()
    mismatch=CliRunner().invoke(app,args+['--reuse-exploration',str(root),'--set','input.amplitude_uA=1'])
    assert mismatch.exit_code != 0 and 'imposed current' in str(mismatch.exception)
