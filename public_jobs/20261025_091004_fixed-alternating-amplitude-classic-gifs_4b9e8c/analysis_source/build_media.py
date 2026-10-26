"""Frozen-vector current sweep, classic scope GIFs and electrothermal diagnostics."""
from pathlib import Path
from dataclasses import replace, asdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import hashlib
import html
import json
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neuristor.config import load_toml, resolved_copy
from neuristor.workflows import _current_params_from_config
from neuristor.current_drive_sim import simulate_current_waveforms
from neuristor.analysis.lab_replay import major_branches
from neuristor.plotting.replay_plots import write_scope_gif
from neuristor.plotting.map_animations import resolved_rt_path
from neuristor.runs import RunBundle
from neuristor.provenance import snapshot_package

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'public_jobs/20260907_125315_physics-anchored-specimen-waveform-inference_717797'
AUDIT = ROOT/'public_jobs/20261025_082439_documented-alternating-amplitude-historical-simu_1072ef'


def render(job):
    path, branches_path, acquired_path, gif, png, current, dt, label, window, frames = job
    raw = np.load(path)
    t = raw['time_ns']
    mask = (t >= window[0]) & (t <= window[1])
    frame = pd.DataFrame(dict(time_ns=t[mask], current_step_uA=current,
        input_current_uA=raw['input_current_uA'][mask], simulated_voltage_mV=raw['voltage_mV'][mask],
        temperature_K=raw['temperature_K'][mask], resistance_ohm=raw['resistance_ohm'][mask]))
    acquired = pd.read_csv(acquired_path)
    caption = f'C=6.8355 pF, Cth=0.044214 pJ/K, dt={dt:g} ns (diagnostic)'
    write_scope_gif(frame, pd.read_csv(branches_path), gif, frames=frames, snapshot_path=png,
        stacked_amperes=True, time_window_ns=tuple(window), trajectory_breaks=True,
        parameter_caption=caption, acquisition_frame=acquired)
    return dict(source_label_mV=label, current_uA=current, dt_ns=dt, gif=str(gif), png=str(png))


def main():
    sys.path.insert(0, str(HERE))
    from cycle_tools import cycle_table, summarize
    cfg = load_toml(ROOT/'experiments/current/alternating_amplitude_gifs_20261025.toml')
    base, *_ = _current_params_from_config(cfg)
    tau_th = base.C_th_J_per_K/base.S_e_W_per_K*1e9
    saved = pd.read_csv(SOURCE/'fitted_traces.csv')
    saved = saved[saved.fit_mode=='relaxed'].copy()
    groups = [g.sort_values('time_ns') for _,g in saved.groupby('nominal_drive_mV',sort=True)]
    labels = [int(g.nominal_drive_mV.iloc[0]) for g in groups]
    t = groups[0].time_ns.to_numpy()
    currents = np.column_stack([g.measured_current_uA for g in groups])
    plateau = (t>=150)&(t<=250)
    medians = [float(np.median(g.measured_current_uA[plateau])) for g in groups]
    bundle = RunBundle.create(name='Fixed alternating amplitude classic GIFs',model='current',kind='analysis',
        config=resolved_copy(cfg),output_root=ROOT/'runs',command='Fixed-vector media study: build_media.py')
    print('RUN_DIRECTORY='+str(bundle.root),flush=True)
    (HERE/'latest_bundle.txt').write_text(str(bundle.root))
    bundle.write_json('source_hashes.json',snapshot_package(bundle),label='Frozen simulation/rendering source')
    bundle.write_json('physics_parameters.json',asdict(base),label='Exact preserved current-source physics')
    for source,target in [(ROOT/'presets/alternating_amplitude_relaxed_20260907.json','parameters.json'),
            (ROOT/'experiments/current/alternating_amplitude_gifs_20261025.toml','recipe.toml'),
            (Path(__file__),'analysis_source/build_media.py'),(HERE/'cycle_tools.py','analysis_source/cycle_tools.py')]:
        shutil.copy2(source,bundle.add_artifact(target,label='Frozen parameters/recipe or exact analysis source'))
    saved.to_csv(bundle.add_artifact('inputs/archived_relaxed_traces.csv',label='All 22 archived input/output records'),index=False)
    pd.read_csv(SOURCE/'parameter_comparison.csv').to_csv(bundle.add_artifact('parameter_comparison.csv',label='Original historical parameter comparison'),index=False)
    branches = major_branches(base.resist_params)
    branches_path = bundle.add_artifact('major_branches.csv',label='Same fitted-law major-branch guides')
    branches.to_csv(branches_path,index=False)
    jobs, summaries, primary_cycles, native_errors = [], [], [], []
    main_raw, main_cycles = None, None
    dt_values = cfg['animation_study']['time_steps_ns']
    for dt in dt_values:
        pars = replace(base,dt_s=dt*1e-9,t_pre_s=-t[0]*1e-9,t_end_s=t[-1]*1e-9)
        outputs = simulate_current_waveforms(currents,pars,waveform_time_s=t*1e-9,audit_proximity=True)
        for label,g,current,out in zip(labels,groups,medians,outputs):
            time_ns = out['t'].astype(float)*1e9
            voltage = out['V_vo2'].astype(float)*1e3
            error = float(np.max(np.abs(np.interp(t,time_ns,voltage)-g.predicted_voltage_mV)))
            if dt==.025 and error>1e-6:
                raise ValueError(f'Original replay differs: label {label}, {error:g} mV')
            native_errors.append(dict(source_label_mV=label,dt_ns=dt,saved_voltage_max_difference_mV=error))
            stem=f'dt_{dt:g}ns/current_{current:09.3f}_uA'
            path=bundle.add_artifact(stem+'.npz',label='Complete native simulated I,V,T,R samples')
            np.savez_compressed(path,time_ns=time_ns,voltage_mV=voltage,temperature_K=out['T'],
                                resistance_ohm=out['R'],input_current_uA=out['I_in'].astype(float)*1e6)
            part = cycle_table(time_ns,voltage,cfg['animation_study']['cycle_window_ns'])
            primary_cycles.append(part.assign(source_label_mV=label,dt_ns=dt))
            window=(time_ns>=50)&(time_ns<=280)
            tau_e=base.C_F*out['R'][window].astype(float)*1e9
            _,_,_,gaps=resolved_rt_path(time_ns[window],out['T'][window],out['R'][window])
            summaries.append(dict(source_label_mV=label,current_uA=current,dt_ns=dt,**summarize(part),
                tau_th_ns=tau_th,tau_e_min_ns=float(tau_e.min()),tau_e_max_ns=float(tau_e.max()),
                minimum_proximity_slope=float(out['minimum_proximity_slope']),
                minimum_temperature_K=float(out['T'].min()),maximum_temperature_K=float(out['T'].max()),
                temperature_valid=bool(out['T'].min()>=base.resist_params.T_min_K and out['T'].max()<=base.resist_params.T_max_K),
                rt_jump_count=gaps))
            if label==400 and dt==.025:
                main_raw=dict(time_ns=time_ns,voltage_mV=voltage,temperature_K=out['T'],resistance_ohm=out['R'])
                main_cycles=part
            if dt in cfg['animation_study']['render_steps_ns']:
                acquired=bundle.add_artifact(f'inputs/current_{label:04d}.csv',label='Original sampled input current')
                if not acquired.exists():
                    pd.DataFrame(dict(time_ns=t,input_current_uA=g.measured_current_uA.to_numpy())).to_csv(acquired,index=False)
                gif=bundle.add_artifact(stem+'.gif',label='Current above voltage above R(T) scope GIF')
                png=bundle.add_artifact(stem+'.png',label='Last frame of the synchronized scope')
                jobs.append((path,branches_path,acquired,gif,png,current,dt,label,cfg['animation_study']['window_ns'],cfg['animation_study']['frames']))
        print(f'Integrated {len(outputs)} measured inputs at dt={dt:g} ns',flush=True)
    summary=pd.DataFrame(summaries)
    summary.to_csv(bundle.add_artifact('cycle_summary.csv',label='All currents at three timesteps with time constants'),index=False)
    pd.concat(primary_cycles,ignore_index=True).to_csv(bundle.add_artifact('individual_cycles.csv',label='Primary-cycle amplitudes without adjacent-peak averaging'),index=False)
    pd.DataFrame(native_errors).to_csv(bundle.add_artifact('replay_checks.csv',label='Original-timestep equality and finer-step differences'),index=False)
    states=[]
    for c in main_cycles.itertuples():
        keep=(main_raw['time_ns']>=c.preceding_peak_time_ns)&(main_raw['time_ns']<=c.peak_time_ns)
        at=np.argmin(abs(main_raw['time_ns']-c.trough_time_ns))
        rmin=float(main_raw['resistance_ohm'][keep].min())
        states.append(dict(cycle_index=c.cycle_index,vpp_mV=c.vpp_mV,minimum_resistance_ohm=rmin,
            maximum_temperature_K=float(main_raw['temperature_K'][keep].max()),
            minimum_tau_e_ns=base.C_F*rmin*1e9,trough_voltage_mV=c.trough_mV,
            resistance_at_trough_ohm=float(main_raw['resistance_ohm'][at]),
            temperature_at_trough_K=float(main_raw['temperature_K'][at])))
    pd.DataFrame(states).to_csv(bundle.add_artifact('main_cycle_states.csv',label='Alternating shallow/deep reset thermal and resistance states'),index=False)
    catalog=[]
    with ProcessPoolExecutor(max_workers=cfg['animation_study']['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(render,jobs):
            row['gif']=Path(row['gif']).relative_to(bundle.root).as_posix()
            row['png']=Path(row['png']).relative_to(bundle.root).as_posix()
            catalog.append(row)
            print(f'Rendered {len(catalog)}/{len(jobs)} GIFs',flush=True)
    bundle.write_json('animations.json',catalog,label='Current/timestep GIF catalog')
    selected=saved[saved.nominal_drive_mV==400]
    # Compact scientific figure for the presentation, with raw samples only.
    fig,axes=plt.subplots(2,1,figsize=(7.6,5.2),gridspec_kw={'height_ratios':[1,1.05]})
    axes[0].plot(selected.time_ns,selected.predicted_voltage_mV,color='#7145d6',lw=1.5)
    axes[0].set(xlim=(60,235),ylabel='Voltage (mV)',xlabel='Time relative to pulse (ns)',title='Alternating shallow and deep resets')
    for k,(left,right,color,label) in enumerate([(main_cycles.iloc[2].preceding_peak_time_ns,main_cycles.iloc[2].peak_time_ns,'#d77b15','Shallow reset'),
                                              (main_cycles.iloc[3].preceding_peak_time_ns,main_cycles.iloc[3].peak_time_ns,'#008a9a','Deep reset')]):
        keep=(main_raw['time_ns']>=left)&(main_raw['time_ns']<=right)
        tt,tr,rr,_=resolved_rt_path(main_raw['time_ns'][keep],main_raw['temperature_K'][keep],main_raw['resistance_ohm'][keep])
        axes[1].semilogy(tr,rr,color=color,lw=2,label=label)
    axes[1].set(xlim=(325,350),xlabel='Simulated temperature (K)',ylabel='Resistance (Ω)')
    axes[1].legend(fontsize=9,loc='lower left')
    for a in axes:a.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(bundle.add_artifact('figures/alternating_resets_for_slide.png',label='Raw voltage and two native R(T) cycles for the slide'),dpi=190)
    plt.close(fig)
    core=summary[(summary.source_label_mV==400)&(summary.dt_ns==.025)].iloc[0]
    metrics=dict(gifs=len(catalog),source_currents=len(groups),integration_timesteps_ns=dt_values,
        native_integrations=len(groups)*len(dt_values),tau_th_ns=tau_th,
        tau_e_main_min_ns=float(core.tau_e_min_ns),tau_e_main_max_ns=float(core.tau_e_max_ns),
        tau_metal_ns=base.C_F*base.resist_params.Rm0*1e9,
        original_replay_max_error_mV=max(r['saved_voltage_max_difference_mV'] for r in native_errors if r['dt_ns']==.025),
        calibration_adopted=False,period_doubling_bifurcation_established=False)
    bundle.write_json('metrics.json',metrics,label='GIF coverage, preserved-vector checks and time constants')
    report=f'''# Fixed historical parameters and classic current-sweep GIFs

The exact 7 September relaxed diagnostic vector is frozen in parameters.json.
All {len(groups)} original baseline-corrected current inputs are replayed without optimization
at dt={dt_values} ns. The 44 GIFs show the original 0.025 ns and finer 0.00625 ns
replay, each with current at top, simulated voltage in the middle, and native
R(T) below. There is no measured-voltage overlay. Heating/cooling guides use the
same resistance law. The window is −30 to 280 ns. Playback frames do not alter
integration resolution. Unresolved R(T) jumps have breaks, never invented chords.
The complete native arrays at all three timesteps are preserved.

Original 0.025 ns sampled voltages reproduce the source within
{metrics['original_replay_max_error_mV']:.4g} mV maximum difference. Current labels
give the median input over 150–250 ns and retain the original source filename label
separately. The same vector drives all currents. Its C and gamma exceed the
scope of the historical constrained calibration. No calibration is adopted.

## Time constants and a working explanation

Passive thermal cooling scale: tau_th=Cth/Se={tau_th:.8f} ns.
Instantaneous frozen-resistance electrical scale: tau_e(t)=C R(T,H).
Along the 400 mV-source replay in 50–280 ns, tau_e spans
{core.tau_e_min_ns:.6f}–{core.tau_e_max_ns:.6f} ns; the fully metallic limit is
{metrics['tau_metal_ns']:.6f} ns. The thermal scale is not an independent fitted
parameter, and Joule heating/hysteresis mean it is not the full coupled-system
relaxation time or a predicted oscillation period.

Individual cycles alternate between shallow heating to about 338 K (minimum R
about 120 ohm) and deeper heating to about 348 K (minimum R about 20 ohm).
RC charging in the high-resistance state takes a time comparable to thermal
recovery. The much shorter electrical discharge during a deep low-R reset and
the different hysteresis reversal states can leave different charge/thermal
histories for the next cycle. This is a plausible electrothermal-memory
explanation, not a demonstrated causal bifurcation mechanism. Time-scale
comparison alone cannot establish that mechanism.

The earlier fixed-vector audit shows that short-window alternation persists
at two finer timesteps but a longer 2 us constant-current replay becomes irregular
at the finest step. Stable long-time period doubling and chaos remain unproven.
See original audit {AUDIT.name}. No physical solver or memory rule was changed.

The repository recipe is experiments/current/alternating_amplitude_gifs_20261025.toml.
The exact media-generation script and package source are preserved. parameters.json
contains all resistance and current-model settings in SI and user units.
'''
    bundle.write_text('report.md',report,label='Numerical method, time scales and qualified mechanism')
    rows=json.dumps(catalog)
    controls=f'''<!doctype html><html><head><meta charset="utf-8"><title>Alternating-amplitude current sweep</title>
<style>body{{font:17px system-ui;max-width:1150px;margin:25px auto;padding:0 20px;color:#20313b}}select,button{{font:inherit;padding:7px}}img{{max-width:850px;width:100%;display:block}}.note{{background:#fff1d6;padding:14px}}table{{border-collapse:collapse}}td{{padding:5px}}</style></head><body>
<h1>Alternating-amplitude diagnostic: the same parameters across 22 currents</h1>
<p>Top: measured input current. Middle: simulated device voltage. Bottom: simulated R(T).</p>
<p class="note">Fixed historical diagnostic, not an adopted calibration. The short-window alternating resets survive refinement, but longer constant-current behavior remains timestep-sensitive.</p>
<p>Current <select id="current"></select> Timestep <select id="dt"><option value="0.025">Original 0.025 ns</option><option value="0.00625">Finer 0.00625 ns</option></select></p>
<h2 id="title"></h2><p id="detail"></p><img id="gif" alt="Current above simulated voltage above resistance versus temperature">
<p><a id="download">Download this GIF</a> · <a id="still">Complete trace</a> · <a href="parameters.json">Saved parameters</a> · <a href="report.md">Method and interpretation</a></p>
<p>τth=Cth/Se={tau_th:.2f} ns. At the central replay, τe=CR spans {core.tau_e_min_ns:.2f}–{core.tau_e_max_ns:.2f} ns. Heating and hysteresis also control the coupled response.</p>
<script>const runs={rows};const current=document.getElementById('current'),dt=document.getElementById('dt');
const labels=[...new Set(runs.map(r=>r.source_label_mV))];for(const label of labels){{const r=runs.find(v=>v.source_label_mV===label);const o=document.createElement('option');o.value=String(label);o.textContent=r.current_uA.toFixed(3)+' µA (source '+label+' mV)';current.appendChild(o);}}
function show(){{const r=runs.find(r=>r.source_label_mV===Number(current.value)&&r.dt_ns===Number(dt.value));document.getElementById('gif').src=r.gif;document.getElementById('download').href=r.gif;document.getElementById('still').href=r.png;document.getElementById('title').textContent=r.current_uA.toFixed(3)+' µA';document.getElementById('detail').textContent='dt='+r.dt_ns+' ns; playback −30 to 280 ns; same frozen vector for every input.';}}
current.value='400';current.onchange=show;dt.onchange=show;show();</script></body></html>'''
    bundle.write_text('START_HERE.html',controls,label='Classic 44-GIF current/timestep gallery',media_type='text/html')
    for artifact in bundle._artifacts:assert (bundle.root/artifact['path']).is_file(),artifact
    bundle.complete(summary=metrics)
    print('COMPLETED='+str(bundle.root),flush=True)


if __name__=='__main__':
    main()
