"""Render both saved loss candidates without repeating their optimization."""
from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuristor.config import resolved_copy, source_directory
from neuristor.current_drive_sim import simulate_current_waveforms
from neuristor.analysis.experimental_waveforms import load_converted_sweep
from neuristor.analysis.lab_replay import major_branches
from neuristor.fitting.parameter_inference import prepare_inference_dataset
from neuristor.fitting.joint_inference import NAMES, joint_model
from neuristor.plotting.replay_plots import write_scope_gif
from neuristor.plotting.map_animations import resolved_rt_path
from neuristor.runs import RunBundle


def loss_scope_gallery(rows):
    """Offline selector loads one of the complete GIFs at a time."""
    options = ''.join(f'<option value="{i}">{html.escape(r["label"])}</option>' for i, r in enumerate(rows))
    data = json.dumps(rows).replace('<', '\\u003c')
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Loss-function comparisons: 44 animations</title>
<style>body{{font:17px system-ui;max-width:1050px;margin:30px auto;padding:0 20px;color:#203040}}select{{font:inherit;width:100%;padding:10px}}img{{max-width:100%;width:680px}}a{{color:#6040b0}}</style></head><body>
<h1>Loss-function comparisons</h1><p>22 measured inputs for each objective: 44 animations.</p>
<p>Top: measured input current. Middle: measured output and simulated device voltage, without phase alignment. Bottom: simulated R–T history and model major branches.</p>
<select id="pick">{options}</select><p><button id="prev">Previous</button> <button id="next">Next</button> · <a id="gif">GIF</a> · <a id="png">Complete-trace image</a> · <a href="report.md">Methods and limitations</a></p>
<p id="detail"></p><img id="animation" alt="Three synchronized panels"><p>These are diagnostic candidates, not accepted specimen fits. Playback is slowed down. Broken R–T segments mark large native-step jumps; R and T are simulated, not measured.</p>
<script>const rows={data};const p=document.getElementById('pick');function show(){{const r=rows[+p.value];document.getElementById('animation').src=r.gif;document.getElementById('gif').href=r.gif;document.getElementById('png').href=r.png;document.getElementById('detail').textContent=r.detail;}}p.onchange=show;document.getElementById('prev').onclick=()=>{{p.selectedIndex=(p.selectedIndex+rows.length-1)%rows.length;show()}};document.getElementById('next').onclick=()=>{{p.selectedIndex=(p.selectedIndex+1)%rows.length;show()}};show();</script></body></html>'''


def export_loss_scopes(config, base, frequency_run, combined_run, *, output_root=None,
                       frames=96, command='neuristor analyze loss-scopes'):
    """Replay selected vectors, require saved-voltage agreement, then render all inputs.

    The original recipe supplies path resolution and fixed initial/physics settings.
    Native R/T endpoints are retained; acquisition channels stay on their own grid.
    No objective, parameter search, or scientific acceptance gate is changed.
    """
    if frames < 2:
        raise ValueError('At least two frames required')
    def fixed(c):
        return {k: v for k, v in resolved_copy(c).items() if k not in ('name', 'steady')}
    parents = {}
    for mode, source, weight in [('frequency_only', frequency_run, 0.), ('frequency_amplitude', combined_run, 1.)]:
        source = Path(source).resolve()
        cfg = json.loads((source/'resolved_config.json').read_text())
        if fixed(cfg) != fixed(config):
            raise ValueError('Original recipe must match both saved runs fixed settings')
        if (json.loads((source/'run.json').read_text())['status'] != 'completed'
            or not json.loads((source/'metrics.json').read_text())['verification_complete']):
            raise ValueError('Completed verification required')
        if cfg['steady']['amplitude_weight'] != weight:
            raise ValueError('Supply frequency-only first and combined second')
        pars = pd.read_csv(source/'parameters.csv')
        pars = pars[pars.candidate.str.startswith(('search_', 'diagnostic_'))]
        if len(pars) != 1:
            raise ValueError('Exactly one selected diagnostic/search candidate required')
        verification = pd.read_csv(source/'verification.csv')
        verification = verification[verification.candidate == pars.candidate.iloc[0]]
        dt = float(verification.dt_ns.min())
        parents[mode] = (source, cfg, pars.iloc[0], verification[verification.dt_ns == dt], dt)
    data_path = Path(config['steady']['data_directory'])
    if not data_path.is_absolute():
        data_path = source_directory(config)/data_path
    raw, _ = load_converted_sweep(data_path)
    dataset = prepare_inference_dataset(raw, holdout_drives_mV=[float(raw.nominal_drive_mV.min())])
    bundle = RunBundle.create(name='Loss function comparison all 44 stacked GIFs', model='loss-scopes',
        kind='analysis', config=dict(recipe=resolved_copy(config), sources={k:str(v[0]) for k,v in parents.items()}, frames=frames),
        output_root=output_root or 'runs', command=command)
    print(f'RUN_DIRECTORY={bundle.root}', flush=True)
    try:
        from neuristor.provenance import snapshot_package
        snapshot_package(bundle)
        rows, provenance = [], {}
        for mode, (source, cfg, pars, verification, dt) in parents.items():
            p = joint_model(pars[list(NAMES)].to_numpy(float), base, dt)
            p = replace(p, t_pre_s=max(0., -dataset.time_ns[0]*1e-9), t_end_s=cfg['steady']['window_ns'][1]*1e-9)
            outputs = simulate_current_waveforms(dataset.current_uA, p, waveform_time_s=dataset.time_ns*1e-9,
                audit_proximity=bool(cfg['steady'].get('require_monotone_proximity', False)))
            saved = pd.read_csv(source/'verification_traces.csv')
            saved = saved[(saved.candidate == pars.candidate) & (saved.dt_ns == dt)]
            branches = major_branches(p.resist_params)
            branches.to_csv(bundle.add_artifact(f'{mode}/major_branches.csv', label='Model major branches'), index=False)
            checks = []
            # Check every full saved trace before spending time rendering any GIF.
            for i, out in enumerate(outputs):
                prior = saved[saved.source_label_mV == dataset.nominal_drives_mV[i]].sort_values('time_ns')
                if prior.empty:
                    raise ValueError('Missing saved current record')
                replay = np.interp(prior.time_ns.to_numpy()*1e-9, out['t'].astype(float), out['V_vo2'].astype(float))*1e3
                error = float(np.max(np.abs(replay-prior.predicted_mV.to_numpy())))
                if error > 1e-6:
                    raise ValueError(f'Replay differs from saved voltage by {error} mV')
                checks.append(error)
            print(f'{mode}: all {len(outputs)} saved voltages reproduced; rendering', flush=True)
            for i, (out, error) in enumerate(zip(outputs, checks)):
                label = float(dataset.nominal_drives_mV[i])
                check = verification[verification.source_label_mV == label].iloc[0]
                current = float(check.measured_current_A)
                plateau = (dataset.time_ns >= 50) & (dataset.time_ns <= 250)
                np.testing.assert_allclose(current, np.mean(dataset.current_uA[plateau, i])*1e-6, rtol=1e-12)
                stop = float(cfg['steady']['window_ns'][1])
                time_ns = out['t'].astype(float)*1e9
                keep = (time_ns >= -30.) & (time_ns <= stop)
                t = time_ns[keep]
                acquired = pd.DataFrame(dict(time_ns=dataset.time_ns, input_current_uA=dataset.current_uA[:,i],
                    measured_voltage_mV=dataset.voltage_mV[:,i]))
                frame = pd.DataFrame(dict(time_ns=t, temperature_K=out['T'][keep], resistance_ohm=out['R'][keep],
                    simulated_voltage_mV=out['V_vo2'][keep].astype(float)*1e3, current_step_uA=current*1e6,
                    input_current_uA=np.interp(t, dataset.time_ns, dataset.current_uA[:,i]),
                    measured_voltage_mV=np.interp(t, dataset.time_ns, dataset.voltage_mV[:,i])))
                stem = f'{mode}/current_{current*1e6:09.3f}_uA'
                gif, png = stem+'.gif', stem+'.png'
                _, _, _, jumps = resolved_rt_path(t, frame.temperature_K, frame.resistance_ohm)
                outside = bool(out['T'].min() < p.resist_params.T_min_K or out['T'].max() > p.resist_params.T_max_K)
                detail = f'{"Frequency only" if mode == "frequency_only" else "Frequency + amplitude"} · {current*1e6:.3f} µA · dt = {dt:g} ns'
                if outside:
                    detail += ' · leaves measured R–T temperature range'
                write_scope_gif(frame, branches, bundle.add_artifact(gif, label=detail), frames=frames,
                    snapshot_path=bundle.add_artifact(png, label='Complete trace'), stacked_amperes=True,
                    time_window_ns=(-30., stop), trajectory_breaks=True,
                    parameter_caption=detail, acquisition_frame=acquired)
                np.savez_compressed(bundle.add_artifact(stem+'.npz', label='Native simulation and acquired channels'),
                    time_ns=t, simulated_voltage_mV=frame.simulated_voltage_mV.to_numpy(),
                    temperature_K=frame.temperature_K.to_numpy(), resistance_ohm=frame.resistance_ohm.to_numpy(),
                    acquisition_time_ns=dataset.time_ns, measured_voltage_mV=dataset.voltage_mV[:,i], input_current_uA=dataset.current_uA[:,i])
                rows.append(dict(mode=mode, label=f'{detail}', detail=detail, gif=gif, png=png,
                    measured_current_A=current, source_label_mV=label, candidate=pars.candidate,
                    dt_ns=dt, replay_max_abs_error_mV=error, rt_jump_count=jumps,
                    outside_calibration=outside, native_samples=len(t)))
                print(f'{mode}: {i+1}/{len(outputs)} · {current*1e6:.3f} µA', flush=True)
            provenance[mode] = dict(source=str(source), candidate=pars.candidate, dt_ns=dt,
                hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
                    ['run.json', 'resolved_config.json', 'parameters.csv', 'verification.csv', 'verification_traces.csv']})
            pars.to_frame().T.to_csv(bundle.add_artifact(f'{mode}/parameters.csv', label='Replayed parameter vector'), index=False)
        pd.DataFrame(rows).to_csv(bundle.add_artifact('animation_index.csv', label='All animations and replay checks'), index=False)
        bundle.write_json('source_provenance.json', provenance, label='Parent hashes')
        bundle.write_text('START_HERE.html', loss_scope_gallery(rows), label='All-current animation selector', media_type='text/html')
        metrics = dict(gif_count=len(rows), records_per_objective={mode:sum(r['mode']==mode for r in rows) for mode in parents},
            max_saved_voltage_error_mV=max(r['replay_max_abs_error_mV'] for r in rows), frames=frames)
        bundle.write_json('metrics.json', metrics, label='Rendering scope and fidelity checks')
        bundle.write_text('report.md', '''# All-input loss-function animations

One GIF for every experimental input under each retained loss candidate. No optimization was repeated.
The original finest verification timestep and full measured current prehistory are replayed.
Every replay must agree with its saved voltage to within 0.000001 mV over the complete saved acquisition grid.
Parent files are unchanged; their hashes and the exact parameter vectors are included.

The top panel is the baseline-corrected measured current, not the acquisition filename number.
Titles use its 50–250 ns average. The middle panel overlays baseline-corrected measured output and
simulated device voltage without phase alignment or rescaling. Both measured channels keep their
original acquisition times; native solver voltage is shown without temporal decimation.
Playback shows −30 to 250 ns; earlier measured current prehistory still drives the integration.

The bottom panel keeps every native simulated resistance/temperature endpoint, with heating and
cooling major branches as guides. Large per-step jumps (>0.25 K or >0.08 in ln R) are broken rather
than connected by invented chords; counts appear in animation_index.csv. Branch guides stop at the
measured temperature domain. Some trajectories leave that domain, explicitly flagged in the viewer.
R and T are simulated, not inferred from measured V/I. Compressed NPZ files retain native plotted
simulation samples and both complete acquisition channels. Full-trace PNGs accompany every GIF.

These are diagnostic candidates, not accepted specimen calibrations. Neither objective produced
a satisfactory all-current fit; displaying all 22 records includes failures and stable controls.
GIF axes fit each record, so use the labeled scales when comparing amplitudes between records.
The animations slow physical time; frame count changes playback only, never integration.
''', label='Methods and limitations')
        bundle.complete(summary=metrics)
    except Exception as exc:
        bundle.fail(exc)
        raise
    return bundle
