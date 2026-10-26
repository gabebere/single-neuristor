"""The derivative audit must separate scientific losses from rejection scores."""
import numpy as np
import pandas as pd

from neuristor.analysis.optimizer_diagnostics import finite_difference_stencils, derivative_timestep_changes
from neuristor.fitting.optimizer_audit import lbfgsb_score_trial
from neuristor.fitting.steady_inference import relative_candidate_score


def test_relative_score_retains_scientific_constraint_and_exact_legacy_sentinel():
    row=dict(frequency_valid=True,loss=.25,proximity_admissible=True)
    result=relative_candidate_score([row])
    assert result['score']==.25 and result['objective_loss']==.25
    failed=dict(row,frequency_valid=False,loss=None)
    result=relative_candidate_score([row,failed])
    assert result['score']==1001000.25 and result['objective_loss'] is None
    assert relative_candidate_score([row],True)['score']==1000000.25
    assert relative_candidate_score([dict(row,proximity_admissible=False)])['objective_loss'] is None


def test_smooth_stencil_and_undefined_scientific_derivative_stay_distinct():
    rows=[]
    for dt in [.1,.05,.025]:
        rows.append(dict(center_name='test',dt_ns=dt,coordinate='center',step=0.,direction=0,
                         score=1.,objective_loss=1.,eligible=True,invalid_count=0))
        for h in [.001,.01]:
            for sign in [-1,1]:
                value=(1+sign*h)**2
                rows.append(dict(center_name='test',dt_ns=dt,coordinate='C_pF',step=h,direction=sign,
                                 score=value,objective_loss=value,eligible=True,invalid_count=0))
    table=pd.DataFrame(rows)
    stencils=finite_difference_stencils(table)
    np.testing.assert_allclose(stencils.objective_centered_slope,2.)
    assert stencils.scientific_derivative_defined.all()
    assert (derivative_timestep_changes(stencils).score_slope_change<1e-12).all()
    table['eligible']=False;table['objective_loss']=np.nan;table['score']=1e6;table['invalid_count']=1
    stencils=finite_difference_stencils(table)
    assert not stencils.scientific_derivative_defined.any()
    assert stencils.objective_centered_slope.isna().all()
    assert stencils.score_plateau.all()


def test_gradient_trial_moves_on_a_smooth_score_and_respects_hard_budget():
    class Quadratic:
        def evaluate(self,unit,dt):
            loss=float(np.sum((np.asarray(unit)-.25)**2))
            return dict(score=loss,eligible=True,unit=list(unit),values=list(unit),invalid_count=0)
    start=np.full(4,.7)
    result=lbfgsb_score_trial(Quadratic(),start,dt_ns=.025,step=.001,max_evaluations=40)
    assert result['best_score']<result['starting_score']/100
    assert result['objective_requests']<=40
    limited=lbfgsb_score_trial(Quadratic(),start,dt_ns=.025,step=.001,max_evaluations=2)
    assert limited['objective_requests']==2
    assert not limited['optimizer_success']


def test_optimizer_audit_cli_bundle_smoke(tmp_path):
    import json
    from pathlib import Path
    from typer.testing import CliRunner
    from neuristor.cli import app
    root=Path(__file__).resolve().parents[1]
    args=['analyze','audit-optimizer','--config',str(root/'experiments/current/specimen_optimizer_audit_20261020.toml'),
          '--output-root',str(tmp_path)]
    for override in ['optimizer_audit.normalized_steps=[0.005]', 'optimizer_audit.dt_ns=[0.5,0.25]',
                     'optimizer_audit.gradient_steps=[0.02]', 'optimizer_audit.gradient_dt_ns=0.5',
                     'optimizer_audit.gradient_max_evaluations=2', 'optimizer_audit.gradient_max_iterations=1',
                     'optimizer_audit.total_seconds=180.0']:
        args.extend(['--set',override])
    result=CliRunner().invoke(app,args)
    assert result.exit_code==0,result.output+repr(result.exception)
    bundle=next(tmp_path.glob('*/run.json')).parent
    assert json.loads((bundle/'run.json').read_text())['status']=='completed'
    metrics=json.loads((bundle/'metrics.json').read_text())
    assert metrics['profile_points']==36
    assert metrics['verified_batches']==8
    stencils=pd.read_csv(bundle/'finite_differences.csv')
    assert len(stencils)==16
    assert (bundle/'figures/coordinate_profiles.png').is_file()
    assert (bundle/'gradient_summary.csv').is_file()
