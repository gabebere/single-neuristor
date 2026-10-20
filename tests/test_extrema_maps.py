"""Scientific checks for extrema measurements and independently batched maps."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_parameter_grid, simulate_current_step
from neuristor.extrema import voltage_extrema, extrema_loss
from neuristor.physical_maps import screen_trajectory


def test_simple_loss_equal_relative_errors_and_separate_rejection():
    measured = dict(frequency_MHz=60., extrema_vpp_mV=100.)
    settings = dict(loss_function='relative_squared', maximum_frequency_MHz=110.)
    predicted = dict(frequency_MHz=66., extrema_vpp_mV=110., sustained=True)
    result = extrema_loss(measured, predicted, settings)
    assert np.isclose(result['loss'], .02)
    assert result['frequency_valid']
    predicted['sustained'] = False
    rejected = extrema_loss(measured, predicted, settings)
    assert not rejected['frequency_valid']
    assert rejected['loss'] == result['loss']  # No hidden scientific penalty.
    predicted['frequency_MHz'] = None
    missing = extrema_loss(measured, predicted, settings)
    assert missing['loss'] is None and missing['frequency_relative_error'] is None


def test_simple_loss_perfect_match_and_frequency_ceiling():
    measured = dict(frequency_MHz=60., extrema_vpp_mV=100.)
    settings = dict(loss_function='relative_squared', maximum_frequency_MHz=110.)
    predicted = dict(**measured, sustained=True)
    assert extrema_loss(measured, predicted, settings)['loss'] == 0
    predicted['frequency_MHz'] = 120.
    result = extrema_loss(measured, predicted, settings)
    assert result['loss'] == 1 and not result['frequency_valid']


def test_slice_normalization_uses_only_visible_screened_points():
    import itertools
    import pandas as pd
    from neuristor.physical_maps import AXES, slice_heatmap_values
    axes = [np.array([1., 2.]), np.array([.003, .006]), np.array([1., 3.])]
    rows = []
    for cth, se, c in itertools.product(*axes):
        shown = cth == 1 and se == .003
        f = 18 + c if shown else 1000
        rows.append(dict(zip(AXES, [cth, se, c])) | dict(included=shown,
            frequency_MHz=f, extrema_vpp_mV=f*10))
    table = pd.DataFrame(rows)
    table['S_e_mW_per_K'] += 1e-17  # Mimic a CSV floating-point round-trip.
    plane = slice_heatmap_values(table, axes, AXES[0], 1.)
    assert plane['count'] == 2
    assert (plane['metrics'][0]['low'], plane['metrics'][0]['high']) == (19, 21)
    assert (plane['metrics'][1]['low'], plane['metrics'][1]['high']) == (190, 210)
    assert np.isnan(plane['metrics'][0]['z'][:, 1]).all()
    assert np.isfinite(plane['metrics'][0]['z']).sum() == plane['count']
    empty = slice_heatmap_values(table, axes, AXES[0], 2.)
    assert empty['count'] == 0 and not empty['metrics'][0]['has_values']


def test_extrema_recover_known_frequency_amplitude_and_ignore_phase_dc_startup():
    t = np.arange(0, 301, .1)
    settings = dict(frequency_tolerance_fraction=.025, amplitude_tolerance_fraction=.075)
    measured = voltage_extrema(t, 200 + 20 * np.cos(2 * np.pi * .06 * t))
    changed = 700 + 20 * np.cos(2 * np.pi * .06 * t + .8)
    changed[t < 150] += 500
    predicted = voltage_extrema(t, changed)
    assert measured['sustained'] and predicted['sustained']
    assert abs(measured['frequency_MHz'] - 60) < .15
    assert abs(measured['extrema_vpp_mV'] - 40) < .01
    assert extrema_loss(measured, predicted, settings)['loss'] < .02
    flat = voltage_extrema(t, np.full_like(t, 200))
    assert not flat['sustained'] and flat['frequency_MHz'] is None
    assert extrema_loss(measured, flat, settings)['loss'] >= 10000
    artifact = dict(measured, extrema_vpp_mV=-2.0)
    assert np.isfinite(extrema_loss(measured, artifact, settings)['loss'])
    assert not extrema_loss(measured, artifact, settings)['frequency_valid']
    decaying = voltage_extrema(t, 200 + 20 * np.cos(2*np.pi*.06*t) * np.exp(-(t-150)/30))
    assert not decaying['sustained']


def test_frequency_ceiling_does_not_replace_temperature_check():
    t = np.arange(0, 301, .1)
    params = CurrentDriveParams()
    settings = dict(window_ns=[150, 300], maximum_frequency_MHz=110)
    out = dict(t=t*1e-9, V_vo2=(200+20*np.sin(2*np.pi*.06*t))*1e-3,
               T=np.full_like(t, 2494), minimum_proximity_slope=1)
    result = screen_trajectory(out, params, settings)
    assert result['frequency_valid'] and not result['temperature_valid'] and not result['eligible']
    out['T'][:] = 340
    out['V_vo2'] = (200+20*np.sin(2*np.pi*.12*t))*1e-3
    result = screen_trajectory(out, params, settings)
    assert result['temperature_valid'] and not result['frequency_valid']


def test_parameter_batch_matches_independent_runs_and_refines():
    base = CurrentDriveParams(dt_s=.5e-9, t_end_s=30e-9, T_init_K=325,
                             C_F=1e-12, C_th_J_per_K=1e-12, S_e_W_per_K=.003e-3)
    c = np.array([0., 1., 5.])*1e-12
    cth = np.array([1., 2., 3.])*1e-12
    se = np.array([0., .003, .008])*1e-3
    batch = simulate_current_parameter_grid(100, base, C_F=c, C_th_J_per_K=cth, S_e_W_per_K=se)
    for j, out in enumerate(batch):
        single = simulate_current_step(100, replace(base, C_F=c[j], C_th_J_per_K=cth[j], S_e_W_per_K=se[j]))
        np.testing.assert_allclose(out['V_vo2'], single['V_vo2'], rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(out['T'], single['T'], rtol=2e-6, atol=1e-4)
    refined = [simulate_current_parameter_grid(100, replace(base, dt_s=dt*1e-9),
               C_F=c, C_th_J_per_K=cth, S_e_W_per_K=se) for dt in [.5, .25, .125]]
    for j in range(3):
        coarse_error = np.max(np.abs(refined[0][j]['V_vo2'] - refined[1][j]['V_vo2'][::2]))
        fine_error = np.max(np.abs(refined[1][j]['V_vo2'] - refined[2][j]['V_vo2'][::2]))
        assert fine_error <= coarse_error + 1e-6


def test_switching_parameter_grid_extrema_converge_at_two_smaller_steps():
    from neuristor.current_domain_search import load_resistance_preset
    root = Path(__file__).resolve().parents[1]
    rp, _, _ = load_resistance_preset(root / 'presets/resistance_100425_chip1_gap3.json')
    base = CurrentDriveParams(t_end_s=600e-9, T0_K=298, T_init_K=298, resist_params=rp)
    features = []
    for dt in [.05, .025, .0125]:
        out = simulate_current_parameter_grid(500, replace(base, dt_s=dt*1e-9),
            C_F=np.array([3e-12]), C_th_J_per_K=np.array([.072e-12]),
            S_e_W_per_K=np.array([.006e-3]))[0]
        f = voltage_extrema(out['t'].astype(float)*1e9, out['V_vo2'].astype(float)*1e3, [300, 600])
        assert f['sustained']
        features.append(f)
    for a, b in zip(features[:-1], features[1:]):
        for field in ['frequency_MHz', 'extrema_vpp_mV']:
            assert abs(a[field] / b[field] - 1) < .05


def test_physical_map_cli_bundle_with_empty_screened_domain(tmp_path):
    config = Path(__file__).resolve().parents[1] / 'experiments/current/physical_parameter_maps.toml'
    args = ['analyze', 'physical-maps', '--config', str(config), '--output-root', str(tmp_path)]
    for value in ['time.duration_us=0.03', 'time.dt_ns=0.5', 'maps.window_ns=[15.0,30.0]',
                  'maps.verification_dt_ns=[0.25,0.125]']:
        args += ['--set', value]
    for name, fixed in [('C_pF', 16.4), ('C_th_pJ_per_K', .047873236), ('S_e_mW_per_K', .0036751265)]:
        for key, value in [('low', fixed), ('high', fixed), ('count', 1), ('extra_values', [])]:
            args += ['--set', f'maps.axes.{name}.{key}={json.dumps(value)}']
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    root = next(tmp_path.glob('*/run.json')).parent
    assert json.loads((root/'run.json').read_text())['status'] == 'completed'
    assert (root/'parameter_maps.html').is_file()
    html = (root/'parameter_maps.html').read_text().replace(r'\u002f', '/')
    for label in ['Locked Cth (pJ/K)', 'Locked Se (mW/K)', 'Locked C (pF)']:
        assert label in html
    assert (root/'grid.csv').is_file()
    reuse = CliRunner().invoke(app, args + ['--reuse-numerics', str(root)])
    assert reuse.exit_code == 0, reuse.output + repr(reuse.exception)
    rendered = [p.parent for p in tmp_path.glob('*/run.json') if p.parent != root][0]
    assert (root/'grid.csv').read_bytes() == (rendered/'grid.csv').read_bytes()
    assert (rendered/'numerical_provenance.json').is_file()
    mismatch = CliRunner().invoke(app, args + ['--reuse-numerics', str(root), '--set', 'input.amplitude_uA=400.0'])
    assert mismatch.exit_code != 0
