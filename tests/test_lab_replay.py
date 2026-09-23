import json
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.config import load_toml
from neuristor.joint_inference import feature_loss, voltage_features, combined_feature_loss

ROOT = Path(__file__).resolve().parents[1]


def test_persistence_loss_rejects_flat_trace():
    t = np.arange(-200., 251.)
    measured = voltage_features(t, 180+15*np.sin(2*np.pi*.05*t))
    good = feature_loss(measured, measured, missing_frequency_penalty=1)
    flat = feature_loss(measured, voltage_features(t, np.full_like(t, 180)), missing_frequency_penalty=1)
    assert flat['persistence'] > 0
    assert flat['frequency'] > 0
    assert combined_feature_loss(flat, {'persistence_weight': 12}) > combined_feature_loss(flat, {})
    assert good['persistence'] == 0
    # The missing-frequency regularizer retains a small smooth gate penalty
    # even on a self-match; it is not a pure squared-error likelihood.
    assert combined_feature_loss(good, {'persistence_weight': 12}) < combined_feature_loss(flat, {'persistence_weight': 12})


def test_replay_cli_bundle_and_recipe(tmp_path):
    result = CliRunner().invoke(app, ['analyze', 'replay-lab', '--config',
        str(ROOT/'experiments/current/specimen_lab_replay.toml'),
        '--set', 'time.dt_ns=0.5', '--set', 'replay.generate_gifs=false', '--output-root', str(tmp_path)])
    assert result.exit_code == 0, result.output + repr(result.exception)
    run = next(tmp_path.glob('*/run.json')).parent
    assert json.loads((run/'metrics.json').read_text())['currents'] == 22
    cfg = load_toml(run/'recipe.toml')
    assert cfg['time']['dt_ns'] == .5
    assert Path(cfg['replay']['data_directory']).is_absolute()
    assert (run/'comparison.html').is_file()
    import pandas as pd
    from neuristor.replay_plots import comparison_figure, write_current_gif
    from PIL import Image
    traces = pd.read_csv(run/'traces.csv')
    fig = comparison_figure(traces, all_currents=True)
    assert len(fig.layout.sliders[0].steps) == 22
    frame = traces[traces.source_mV == traces.source_mV.iloc[0]]
    write_current_gif(frame, tmp_path/'sample.gif', frames=3)
    with Image.open(tmp_path/'sample.gif') as gif:
        assert gif.n_frames == 3


def test_playground_loads_and_groups_parameters():
    ui = AppTest.from_file(str(ROOT/'src/neuristor/playground.py')).run(timeout=20)
    assert not ui.exception
    assert any(n.label.startswith('Ambient temperature') for n in ui.number_input)
    assert any(b.label == 'Run all measured currents' for b in ui.button)

    fitted = next(o for o in ui.sidebar.selectbox[0].options if "static_preserving" in o)
    ui.sidebar.selectbox[0].select(fitted).run()
    ui.sidebar.button[0].click().run()
    assert not ui.exception
    capacitance = next(n.value for n in ui.number_input if n.label.startswith('Capacitance'))
    assert capacitance == ui.session_state.lab_config['electrical']['C_pF']
    assert capacitance != .39


def test_major_branches_and_progressive_scope(tmp_path):
    import pandas as pd
    from PIL import Image, ImageChops
    from neuristor.lab_replay import major_branches
    from neuristor.workflows import _current_params_from_config
    from neuristor.replay_plots import write_scope_gif, scope_gallery
    p, *_ = _current_params_from_config(load_toml(ROOT/'experiments/current/specimen_lab_replay.toml'))
    branches = major_branches(p.resist_params)
    assert (branches.heating_ohm >= branches.cooling_ohm).all()
    assert np.isfinite(branches.to_numpy()).all()
    t = np.linspace(-30, 350, 200)
    frame = pd.DataFrame(dict(time_ns=t, current_step_uA=100., measured_voltage_mV=100+np.sin(t),
        simulated_voltage_mV=100+5*np.sin(t), input_current_uA=np.where(t>0,100,0),
        temperature_K=320+np.sin(t), resistance_ohm=1000+100*np.sin(t)))
    write_scope_gif(frame, branches, tmp_path/'scope.gif', frames=3, snapshot_path=tmp_path/'scope.png', stacked_amperes=True, original_label_uA=350)
    with Image.open(tmp_path/'scope.gif') as im:
        assert im.n_frames == 3
        first = im.convert('RGB')
        im.seek(2)
        assert ImageChops.difference(first, im.convert('RGB')).getbbox() is not None
    assert (tmp_path/'scope.png').is_file()
    assert 'type="range"' in scope_gallery([dict(label='100 µA',gif='scope.gif',png='scope.png')])


def test_saved_scope_cli_keeps_exact_samples(tmp_path):
    import pandas as pd
    source = tmp_path/'source'
    source.mkdir()
    # Build a small self-contained replay fixture; the workflow must not simulate.
    cfg = load_toml(ROOT/'experiments/current/specimen_lab_replay.toml')
    cfg.pop('_source')
    cfg['resistance']['preset'] = 'yuanhang'
    (source/'resolved_config.json').write_text(json.dumps(cfg))
    t = np.linspace(-30,350,80)
    pd.DataFrame(dict(source_mV=100.,time_ns=t,current_step_uA=100.,input_current_uA=100.,
        measured_voltage_mV=100+np.sin(t),simulated_voltage_mV=100+2*np.sin(t),
        temperature_K=320+np.sin(t),resistance_ohm=1000+100*np.sin(t))).to_csv(source/'traces.csv',index=False)
    for name, text in [('summary.csv','current_step_uA\n100\n'),('metrics.json','{}'),
                       ('comparison.html','<html></html>'),('recipe.toml','schema_version = 1')]:
        (source/name).write_text(text)
    output = tmp_path/'out'
    result = CliRunner().invoke(app,['analyze','export-scope','--source',str(source),
        '--output-root',str(output),'--frames','2','--original-labels-uA'])
    assert result.exit_code == 0, result.output + repr(result.exception)
    run = next(output.glob('*/run.json')).parent
    assert (run/'traces.csv').read_bytes() == (source/'traces.csv').read_bytes()
    assert (run/'GIFs/current_0.000100000_A.gif').is_file()
    assert (run/'Figures/current_0.000100000_A.png').is_file()
    assert json.loads((run/'source_provenance.json').read_text())['dt_ns'] == .025


    assert json.loads((run/'source_provenance.json').read_text())['original_labels_uA'] is True
    assert '100 µA' in (run/'START_HERE.html').read_text()
