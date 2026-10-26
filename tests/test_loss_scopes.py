"""Check replay fidelity and distinct acquisition/native grids in the GIF export."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.config import load_toml, resolved_copy
from neuristor.current_drive_sim import simulate_current_waveforms
from neuristor.fitting.joint_inference import NAMES, natural_vector, joint_model
from neuristor.orchestration import loss_scopes
from neuristor.workflows import _current_params_from_config


def parents_and_data(tmp_path, monkeypatch):
    recipe = Path('experiments/current/specimen_frequency_only_20261020.toml').resolve()
    text = recipe.read_text().replace('[150.0,250.0]', '[1.0,3.0]')
    text = text.replace('../../data/parameters/resistance_100425_chip1_gap3.json',
                        str(Path('data/parameters/resistance_100425_chip1_gap3.json').resolve()))
    recipe = tmp_path/'recipe.toml'
    recipe.write_text(text)
    config = load_toml(recipe)
    base, *_ = _current_params_from_config(config)
    # Real low-current integrations on a much finer grid than the acquisition.
    time = np.array([-1., 0., 1., 2., 3., 50., 250.])
    currents = np.tile([20., 40.], (len(time), 1))
    data = SimpleNamespace(time_ns=time, current_uA=currents,
        voltage_mV=np.zeros_like(currents), nominal_drives_mV=np.array([50., 100.]))
    monkeypatch.setattr(loss_scopes, 'load_converted_sweep', lambda _: (pd.DataFrame({'nominal_drive_mV':[50.,100.]}), None))
    monkeypatch.setattr(loss_scopes, 'prepare_inference_dataset', lambda *a, **kw: data)
    values = natural_vector(base)
    p = replace(joint_model(values, base, .025), t_pre_s=1e-9, t_end_s=3e-9)
    outputs = simulate_current_waveforms(currents, p, waveform_time_s=time*1e-9, audit_proximity=True)
    parents = []
    for name, weight in [('frequency', 0.), ('combined', 1.)]:
        root = tmp_path/name
        root.mkdir()
        parents.append(root)
        cfg = resolved_copy(config)
        cfg['steady']['amplitude_weight'] = weight
        for filename, value in [('resolved_config.json',cfg), ('run.json',{'status':'completed'}),
                                ('metrics.json',{'verification_complete':True})]:
            (root/filename).write_text(json.dumps(value))
        pd.DataFrame([dict(candidate='diagnostic_1', **dict(zip(NAMES, values)))]).to_csv(root/'parameters.csv',index=False)
        pd.DataFrame(dict(candidate='diagnostic_1', dt_ns=.025, source_label_mV=[50.,100.],
                          measured_current_A=[20e-6,40e-6])).to_csv(root/'verification.csv',index=False)
        traces = []
        for label, out in zip(data.nominal_drives_mV, outputs):
            t = time[time <= 3.]
            v = np.interp(t*1e-9, out['t'].astype(float), out['V_vo2'].astype(float))*1e3
            traces.append(pd.DataFrame(dict(candidate='diagnostic_1', dt_ns=.025, source_label_mV=label,
                                             time_ns=t, predicted_mV=v)))
        pd.concat(traces).to_csv(root/'verification_traces.csv',index=False)
    return recipe, parents


def test_loss_scopes_cli_real_replay_and_gif(tmp_path, monkeypatch):
    recipe, parents = parents_and_data(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ['analyze','loss-scopes','--config',str(recipe),
        '--frequency-run',str(parents[0]),'--combined-run',str(parents[1]),
        '--frames','3','--output-root',str(tmp_path/'out')])
    assert result.exit_code == 0, result.output + repr(result.exception)
    root = next((tmp_path/'out').glob('*/run.json')).parent
    index = pd.read_csv(root/'animation_index.csv')
    assert len(index) == 4
    assert index.replay_max_abs_error_mV.max() < 1e-6
    for row in index.itertuples():
        with Image.open(root/row.gif) as gif:
            assert gif.n_frames == 3
            assert gif.height > gif.width
        samples = np.load((root/row.gif).with_suffix('.npz'))
        assert len(samples['time_ns']) > len(samples['acquisition_time_ns'])
        assert samples['input_current_uA'].shape == samples['acquisition_time_ns'].shape
    assert '20.000 µA' in (root/'START_HERE.html').read_text()
    assert json.loads((root/'run.json').read_text())['status'] == 'completed'


def test_loss_scopes_rejects_changed_saved_voltage_before_render(tmp_path, monkeypatch):
    recipe, parents = parents_and_data(tmp_path, monkeypatch)
    path = parents[0]/'verification_traces.csv'
    frame = pd.read_csv(path)
    frame.predicted_mV += .1
    frame.to_csv(path,index=False)
    config = load_toml(recipe)
    base, *_ = _current_params_from_config(config)
    with pytest.raises(ValueError, match='differs from saved voltage'):
        loss_scopes.export_loss_scopes(config, base, *parents, frames=3, output_root=tmp_path/'out')
    assert not list((tmp_path/'out').rglob('*.gif'))
