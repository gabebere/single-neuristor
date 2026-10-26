"""Archive a fixed-parameter waveform finding; never modify the physical model."""
from pathlib import Path
from dataclasses import asdict
import base64
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from neuristor.runs import RunBundle
from neuristor.provenance import snapshot_package
from neuristor.config import load_toml
from neuristor.workflows import _current_params_from_config

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(__file__).resolve().parent
PARENT = ROOT / "public_jobs/20260907_125315_physics-anchored-specimen-waveform-inference_717797"
REFERENCE = ROOT / "runs/20261025_081228_archived-alternating-amplitude-constant-current-_0ef1a6"


def cycle_table(time_ns, voltage_mV, window, *, prominence_fraction=.10, smoothing_ns=.5):
    """Use individual peak-minus-preceding-trough amplitudes, without neighbor averaging.

    Smoothing locates primary peaks only. A local raw maximum refines each peak;
    raw minima between primary peaks supply the preceding trough. Prominence
    excludes small shoulders on the reset, not just quantization/noise. Edge
    fragments are omitted. These are descriptive signal features, not a new
    physical oscillation or statistical significance criterion.
    """
    t, v = np.asarray(time_ns, float), np.asarray(voltage_mV, float)
    if t.shape != v.shape or t.ndim != 1 or not np.isfinite(t).all() or not np.isfinite(v).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Matching finite, increasing waveform arrays required")
    mask = (t >= window[0]) & (t <= window[1])
    t, v = t[mask], v[mask]
    if len(t) < 9:
        return pd.DataFrame()
    dt = np.median(np.diff(t))
    width = max(5, int(round(smoothing_ns/dt)) | 1)
    width = min(width, len(v) if len(v) % 2 else len(v)-1)
    smooth = savgol_filter(v, width, 2) if smoothing_ns > 0 else v
    peaks, _ = find_peaks(smooth, prominence=max(2., prominence_fraction*np.ptp(smooth)),
                          distance=max(1, round(2./dt)))
    radius = max(0, round(.5/dt))
    peaks = np.unique([max(0, p-radius) + np.argmax(v[max(0,p-radius):min(len(v),p+radius+1)]) for p in peaks])
    rows = []
    for n, (left, right) in enumerate(zip(peaks[:-1], peaks[1:])):
        bottom = left + np.argmin(v[left:right+1])
        if bottom in (left, right) or v[right] <= v[bottom]:
            continue
        rows.append(dict(cycle_index=len(rows), peak_time_ns=t[right], preceding_peak_time_ns=t[left],
            trough_time_ns=t[bottom], peak_mV=v[right], trough_mV=v[bottom],
            vpp_mV=v[right]-v[bottom], period_ns=t[right]-t[left]))
    return pd.DataFrame(rows)


def summarize(cycles):
    """Separate two-cycle alternation from broad amplitude variation or drift."""
    if len(cycles) < 6:
        return dict(cycle_count=len(cycles), strong_alternation=False)
    a = cycles.vpp_mV.to_numpy()
    medians = np.array([np.median(a[::2]), np.median(a[1::2])])
    sep = float(np.ptp(medians)/np.mean(medians))
    mad = max(float(np.median(np.abs(a[k::2]-medians[k]))/medians[k]) for k in [0,1])
    lag1 = float(np.corrcoef(a[:-1], a[1:])[0,1]) if np.std(a) > 1e-6 else 0.
    lag2 = float(np.corrcoef(a[:-2], a[2:])[0,1]) if np.std(a) > 1e-6 else 0.
    strong = bool(len(a) >= 8 and sep >= .10 and mad <= .05 and lag1 <= -.8 and lag2 >= .8)
    times = cycles.peak_time_ns.to_numpy()
    return dict(cycle_count=len(a), phase0_median_vpp_mV=medians[0], phase1_median_vpp_mV=medians[1],
        low_phase_median_vpp_mV=float(min(medians)), high_phase_median_vpp_mV=float(max(medians)),
        phase_separation_fraction=sep, maximum_phase_relative_MAD=mad,
        amplitude_lag1=lag1, amplitude_lag2=lag2, strong_alternation=strong,
        median_adjacent_peak_period_ns=float(np.median(cycles.period_ns)),
        median_two_peak_period_ns=float(np.median(times[2:]-times[:-2])),
        adjacent_peak_rate_MHz=float(1000/np.median(cycles.period_ns)),
        two_peak_repeat_rate_MHz=float(1000/np.median(times[2:]-times[:-2])))


def analysis_self_checks():
    # Known alternating pulses and a monotone amplitude drift must differ.
    t = np.arange(0, 500, .05)
    fixed = np.zeros_like(t)
    alternating = np.zeros_like(t)
    drift = np.zeros_like(t)
    for k, center in enumerate(np.arange(10, 500, 20)):
        pulse = np.exp(-.5*((t-center)/2)**2)
        fixed += 100*pulse
        alternating += (100 if k % 2 else 200)*pulse
        drift += (100+k*3)*pulse
    assert summarize(cycle_table(t, alternating, [0,500]))["strong_alternation"]
    assert not summarize(cycle_table(t, fixed, [0,500]))["strong_alternation"]
    assert not summarize(cycle_table(t, drift, [0,500]))["strong_alternation"]
    return {"alternating_pulses": "pass", "uniform_pulses": "pass", "drifting_pulses": "pass"}


def main():
    self_checks = analysis_self_checks()
    assert "200 passed" in (SCRATCH/"pytest.log").read_text()
    bundle = RunBundle.create(name="Documented alternating-amplitude historical simulation", model="current", kind="analysis",
        config=dict(parent_bundle=PARENT.name, fit_mode="relaxed", primary_source_label_mV=400,
            neighbor_source_labels_mV=[350,450], dt_ns=[.025,.0125,.00625],
            replay_window_ns=[50,280], constant_current_window_ns=[1000,2000],
            prominence_fraction=.10, smoothing_for_detection_ns=.5, calibration_adopted=False),
        output_root=ROOT/"runs", command="Saved-data documentation: document_finding.py")
    bundle.write_json("source_hashes.json", snapshot_package(bundle), label="Frozen authoritative implementation")
    for p in [Path(__file__), SCRATCH/"scan_saved.py", SCRATCH/"scan_csv.py", SCRATCH/"replay_candidates.py"]:
        shutil.copy2(p, bundle.add_artifact("analysis_source/"+p.name, label="Exact analysis/replay source"))
    for name in ["parameter_comparison.csv", "resolved_config.json", "metrics.json", "run.json", "convergence.csv"]:
        shutil.copy2(PARENT/name, bundle.add_artifact("historical_parameters/"+name, label="Unchanged historical fit evidence"))
    for name in ["pytest.log", "validate.log", "reference_recipe.log"]:
        shutil.copy2(SCRATCH/name, bundle.add_artifact("checks/"+name, label="Software/workflow check log"))
    provenance = {}
    for label in [350,400,450]:
        p = ROOT/f"data/experimental/tia_current_sweep/{label}mv0_converted.csv"
        shutil.copy2(p, bundle.add_artifact("inputs/"+p.name, label="Original experimental numerical trace"))
        provenance[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    saved = pd.read_csv(PARENT/"fitted_traces.csv")
    selected = saved[(saved.fit_mode=="relaxed") & saved.nominal_drive_mV.isin([350,400,450])].copy()
    selected.to_csv(bundle.add_artifact("archived_selected_traces.csv", label="Exact archived rows for three source labels"), index=False)
    provenance[str((PARENT/"fitted_traces.csv").relative_to(ROOT))] = hashlib.sha256((PARENT/"fitted_traces.csv").read_bytes()).hexdigest()
    bundle.write_json("input_provenance.json", provenance, label="Original numerical input hashes")
    # Keep an exact standard CLI replay, including its own source/provenance.
    shutil.copytree(REFERENCE, bundle.root/"constant_current_recipe_run")
    for p in (bundle.root/"constant_current_recipe_run").rglob("*"):
        if p.is_file(): bundle.register_file(p.relative_to(bundle.root), label="Existing CLI workflow replay: "+p.name)
    params, *_ = _current_params_from_config(load_toml(ROOT/"experiments/current/alternating_amplitude_reference_20261025.toml"))
    bundle.write_json("physics_parameters.json", asdict(params), label="Exact fixed-parameter constant-current control")
    preset = ROOT/"presets/resistance_100425_chip1_gap3.json"
    shutil.copy2(preset, bundle.add_artifact("inputs/resistance_preset.json", label="Original static R-T preset"))
    recipe = (ROOT/"experiments/current/alternating_amplitude_reference_20261025.toml").read_text().replace(
        "../../presets/resistance_100425_chip1_gap3.json", "inputs/resistance_preset.json")
    bundle.write_text("reference_recipe.toml", recipe, label="Portable constant-current recipe", media_type="text/plain")
    summaries, all_cycles, waveforms = [], [], {}
    for label in [350,400,450]:
        g = selected[selected.nominal_drive_mV==label]
        for signal in ["predicted_voltage_mV", "measured_voltage_mV"]:
            c = cycle_table(g.time_ns, g[signal], [50,280])
            identity = dict(source="archived_1ns", input_mode="measured_current", source_label_mV=label,
                            dt_ns=1., signal=signal, window_start_ns=50, window_stop_ns=280)
            summaries.append(dict(**identity, **summarize(c)))
            all_cycles.append(c.assign(**identity))
        for mode in ["measured_current", "constant_current"]:
            for dt in [.025,.0125,.00625]:
                path = SCRATCH/f"waveform_{mode}_{label}_{dt:g}.npz"
                shutil.copy2(path, bundle.add_artifact("replays/"+path.name, label="Full native replay samples"))
                raw = dict(np.load(path))
                if label==400: waveforms[(mode,dt)] = raw
                window = [50,280] if mode=="measured_current" else [1000,2000]
                c = cycle_table(raw["time_ns"], raw["voltage_mV"], window)
                identity = dict(source="native_replay", input_mode=mode, source_label_mV=label, dt_ns=dt,
                    signal="predicted_voltage_mV", window_start_ns=window[0], window_stop_ns=window[1])
                summary = dict(**identity, **summarize(c), minimum_temperature_K=float(raw["temperature_K"].min()),
                    maximum_temperature_K=float(raw["temperature_K"].max()))
                if mode=="measured_current":
                    summary["saved_voltage_max_difference_mV"] = float(np.max(np.abs(
                        np.interp(g.time_ns, raw["time_ns"], raw["voltage_mV"])-g.predicted_voltage_mV)))
                summaries.append(summary)
                all_cycles.append(c.assign(**identity))
    summary = pd.DataFrame(summaries)
    summary.to_csv(bundle.add_artifact("cycle_summary.csv", label="Per-case cycle/amplitude and correlation summaries"), index=False)
    cycles = pd.concat(all_cycles, ignore_index=True)
    cycles.to_csv(bundle.add_artifact("individual_cycles.csv", label="Individual raw peak/trough amplitudes"), index=False)
    sensitivity = []
    for (mode,dt), raw in waveforms.items():
        for fraction in [.05,.10,.15,.20]:
            for smooth in [0.,.25,.5,1.]:
                window = [50,280] if mode=="measured_current" else [1000,2000]
                c = cycle_table(raw["time_ns"],raw["voltage_mV"],window,prominence_fraction=fraction,smoothing_ns=smooth)
                sensitivity.append(dict(input_mode=mode,dt_ns=dt,prominence_fraction=fraction,smoothing_ns=smooth,**summarize(c)))
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(bundle.add_artifact("peak_detection_sensitivity.csv", label="Primary-peak threshold and smoothing sensitivity"), index=False)
    bundle.write_json("analysis_self_checks.json", self_checks, label="Synthetic checks distinguish alternation, drift and uniform pulses")
    original = summary[(summary.source=="archived_1ns") & (summary.source_label_mV==400)]
    native = summary[(summary.source=="native_replay") & (summary.source_label_mV==400)]
    sim = original[original.signal=="predicted_voltage_mV"].iloc[0]
    meas = original[original.signal=="measured_voltage_mV"].iloc[0]
    assert sim.strong_alternation and not meas.strong_alternation
    matched = native[native.input_mode=="measured_current"]
    assert matched.strong_alternation.all()
    assert matched.loc[matched.dt_ns==.025,"saved_voltage_max_difference_mV"].iloc[0] < 1e-6
    case = selected[selected.nominal_drive_mV==400]
    c = cycles[(cycles.source=="archived_1ns") & (cycles.source_label_mV==400) & (cycles.signal=="predicted_voltage_mV")]
    fig, axes = plt.subplots(2,2,figsize=(13,8.5),gridspec_kw={"height_ratios":[1.2,1]})
    axes[0,0].plot(case.time_ns,case.predicted_voltage_mV,color="#5e47b5",lw=1.3)
    colors = np.where(c.cycle_index%2==0,"#d77b15","#008a9a")
    axes[0,0].scatter(c.peak_time_ns,c.peak_mV,c=colors,s=35,zorder=3)
    axes[0,0].scatter(c.trough_time_ns,c.trough_mV,c=colors,s=24,marker="v",zorder=3)
    axes[0,0].set(title="Archived simulation: alternating shallow and deep resets",ylabel="Simulated device voltage (mV)",xlim=(50,280),xlabel="Time relative to pulse (ns)")
    axes[0,1].plot(case.time_ns,case.measured_voltage_mV,color="#315d7c",lw=1.1)
    axes[0,1].set(title="Corresponding measurement: no comparable two-level alternation",ylabel="Measured output (mV)",xlabel="Time relative to pulse (ns)",xlim=(50,280))
    axes[1,0].plot(c.cycle_index,c.vpp_mV,color=".7",lw=1)
    axes[1,0].scatter(c.cycle_index,c.vpp_mV,c=colors,s=50)
    axes[1,0].set(title="Each amplitude uses its own preceding raw trough",xlabel="Primary-cycle index (arbitrary parity)",ylabel="Cycle Vpp (mV)")
    a=c.vpp_mV.to_numpy()
    axes[1,1].scatter(a[:-1],a[1:],c=colors[:-1],s=55)
    lo,hi=min(a)-20,max(a)+20;axes[1,1].plot([lo,hi],[lo,hi],"--",color=".6",lw=1)
    axes[1,1].set(title="Successive-amplitude return plot: two off-diagonal groups",xlabel="Vpp(n), mV",ylabel="Vpp(n+1), mV",xlim=(lo,hi),ylim=(lo,hi))
    for ax in axes.ravel():ax.grid(alpha=.2)
    fig.suptitle("Recovered historical example · 7 September relaxed fit · source label 400 mV (~303 µA)\nDiagnostic simulation; no adopted physical calibration",fontsize=13)
    fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(bundle.add_artifact("figures/archived_alternating_amplitudes.png",label="Documented archived waveform and amplitude sequence"),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharey="col")
    for row,dt in enumerate([.025,.0125,.00625]):
        raw=waveforms[("constant_current",dt)]
        m=(raw["time_ns"]>=1300)&(raw["time_ns"]<=1650)
        axes[row,0].plot(raw["time_ns"][m],raw["voltage_mV"][m],lw=.9,color="#5e47b5")
        axes[row,0].set(title=f"Constant current, dt={dt:g} ns",ylabel="Voltage (mV)",xlabel="Time (ns)")
        part=cycles[(cycles.source=="native_replay") & (cycles.source_label_mV==400) & (cycles.input_mode=="constant_current") & (cycles.dt_ns==dt)]
        axes[row,1].plot(part.peak_time_ns,part.vpp_mV,"o-",lw=.8,ms=3,color="#008a9a")
        axes[row,1].set(title="Individual primary-cycle amplitudes in 1–2 µs",ylabel="Vpp (mV)",xlabel="Peak time (ns)")
    for ax in axes.ravel():ax.grid(alpha=.2)
    fig.suptitle("Longer constant-current control: a stable period-two state is not established\nCoarser runs alternate regularly; the finest run develops irregular amplitudes",fontsize=13)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(bundle.add_artifact("figures/timestep_and_horizon_check.png",label="Numerical limitation in the long constant-current control"),dpi=150);plt.close(fig)
    fig=make_subplots(rows=2,cols=1,subplot_titles=["Native simulated voltage: choose input and timestep", "Each primary cycle's raw peak-minus-preceding-trough amplitude"],vertical_spacing=.14)
    choices=list(waveforms)
    for j,(mode,dt) in enumerate(choices):
        raw=waveforms[(mode,dt)]
        window=[0,300] if mode=="measured_current" else [900,2000]
        m=np.flatnonzero((raw["time_ns"]>=window[0])&(raw["time_ns"]<=window[1]))
        # Display thinning only. CSV/NPZ and quantitative analysis keep every sample.
        step=max(1,round(.05/np.median(np.diff(raw["time_ns"]))))
        m=m[::step]
        fig.add_trace(go.Scattergl(x=raw["time_ns"][m],y=raw["voltage_mV"][m],mode="lines",name=f"{mode}, {dt:g} ns",visible=j==0),row=1,col=1)
        part=cycles[(cycles.source=="native_replay")&(cycles.source_label_mV==400)&(cycles.input_mode==mode)&(cycles.dt_ns==dt)]
        fig.add_trace(go.Scatter(x=part.peak_time_ns,y=part.vpp_mV,mode="lines+markers",name="Individual cycle Vpp",visible=j==0),row=2,col=1)
    buttons=[dict(label=f"{mode.replace('_',' ')} · dt {dt:g} ns",method="update",args=[{"visible":[k//2==j for k in range(2*len(choices))]}]) for j,(mode,dt) in enumerate(choices)]
    fig.update_layout(height=760,showlegend=False,updatemenus=[dict(buttons=buttons,x=0,y=1.14,xanchor="left")],margin=dict(t=100))
    fig.update_xaxes(title_text="Time relative to pulse (ns)",row=1,col=1);fig.update_xaxes(title_text="Primary peak time (ns)",row=2,col=1)
    fig.update_yaxes(title_text="Simulated voltage (mV)",row=1,col=1);fig.update_yaxes(title_text="Cycle Vpp (mV)",row=2,col=1)
    metrics=dict(parent_bundle=PARENT.name, source_label_mV=400, constant_control_current_uA=float(case.measured_current_uA[(case.time_ns>=150)&(case.time_ns<=250)].median()),
        archived_primary_cycles=int(sim.cycle_count), archived_low_cycle_vpp_mV=float(sim.low_phase_median_vpp_mV),
        archived_high_cycle_vpp_mV=float(sim.high_phase_median_vpp_mV), archived_lag1=float(sim.amplitude_lag1),
        archived_lag2=float(sim.amplitude_lag2), measured_alternation_flag=bool(meas.strong_alternation),
        short_window_alternation_at_all_three_timesteps=bool(matched.strong_alternation.all()),
        stable_long_time_period_two_established=False, bifurcation_established=False, calibration_adopted=False,
        repository_tests_passed=200, native_replay_integrations=18,
        exact_archived_replay_max_difference_mV=float(matched.loc[matched.dt_ns==.025,"saved_voltage_max_difference_mV"].iloc[0]))
    bundle.write_json("metrics.json",metrics,label="Evidence and limits of the recovered phenomenon")
    report=f"""# Alternating amplitudes recovered in a historical simulation

Reviewed 25 October 2026. The clearest recovered example is the **relaxed**
simulation in `{PARENT.name}`, source file **400mv0_converted.csv**. It is in
`fitted_traces.csv`, `predicted_voltage_mV`, not the measured-voltage column.
Source labels 350 and 450 mV provide neighboring examples. The source label is
not imposed current: the constant-current control uses {metrics['constant_control_current_uA']:.9f} µA,
the median archived baseline-corrected current over 150–250 ns.

## What was observed

In the archived 50–280 ns window, {int(sim.cycle_count)} complete primary cycles
alternate between approximately **{sim.low_phase_median_vpp_mV:.2f} and
{sim.high_phase_median_vpp_mV:.2f} mV Vpp**. Adjacent amplitudes have correlation
{sim.amplitude_lag1:.5f}; amplitudes two cycles apart have correlation
{sim.amplitude_lag2:.5f}. The main visual difference is alternating deep/shallow
troughs, not a factor-of-two peak height. Median adjacent-primary-peak spacing is
{sim.median_adjacent_peak_period_ns:.2f} ns, and two-peak spacing is
{sim.median_two_peak_period_ns:.2f} ns. A short alternating pattern therefore has
an individual-peak rate near {sim.adjacent_peak_rate_MHz:.2f} MHz and a two-cycle
repeat rate near {sim.two_peak_repeat_rate_MHz:.2f} MHz. These are descriptive
rates, not evidence for two independent oscillators.

The corresponding measured output does not show a comparable two-level pattern
under the same primary-cycle analysis. Its parity separation is only
{100*meas.phase_separation_fraction:.2f}% versus {100*sim.phase_separation_fraction:.2f}%
in this archived simulated record. This documents a simulation phenomenon;
it does not establish experimental period doubling or device-terminal calibration.

## Replay and numerical limitations

No optimization was repeated. The exact eight-parameter historical vector was
replayed using the authoritative float32-faithful model with the original
measured input. At 0.025 ns, sampled voltages reproduce the archived data to
{metrics['exact_archived_replay_max_difference_mV']:.3g} mV maximum difference.
Primary-cycle alternation remains in the 50–280 ns window at 0.0125 and
0.00625 ns. Raw native samples are preserved; only display plotting is thinned.

A constant-current control removes the measured input's fluctuations and
extends to 2 µs. Its final 1–2 µs alternates regularly at 0.025 and 0.0125 ns,
but develops irregular primary-cycle amplitudes at 0.00625 ns. Consequently,
**a stable long-time period-two attractor and a period-doubling bifurcation are
not established**. Qualitative recovery of the archived short-window pattern
does not imply pointwise waveform convergence, a converged steady state, or chaos.
No physical solver, memory rule or tolerance was altered to preserve the effect.

{native[['input_mode','dt_ns','cycle_count','low_phase_median_vpp_mV','high_phase_median_vpp_mV','amplitude_lag1','amplitude_lag2','strong_alternation']].to_markdown(index=False)}

Even/odd amplitude medians in irregular runs are arbitrary parity summaries;
they should not be interpreted as two stable amplitude levels.

## Method

Locate primary voltage maxima using Savitzky–Golay smoothing over about 0.5 ns
(at least five samples; therefore 5 ns for the archived 1 ns output), prominence
max(2 mV, 10% of the window's smoothed range), and at least 2 ns separation.
Refine maxima on raw samples within ±0.5 ns. For each successive pair, take the
raw minimum between them and subtract it from the later raw maximum. Exclude
partial edge cycles. Never average neighboring peak heights: that can hide the
two-amplitude feature. Small reset shoulders are not primary cycles.

The descriptive `strong_alternation` flag requires at least eight primary
cycles, ≥10% parity-median separation, ≤5% relative median absolute deviation
within each parity, lag-1 correlation ≤−0.8 and lag-2 correlation ≥0.8.
These are explicit engineering criteria, not a significance test or a
bifurcation detector. `peak_detection_sensitivity.csv` varies prominence
5–20% and smoothing 0–1 ns. Synthetic uniform and drifting amplitudes are
negative controls; known alternating pulses are a positive control.
The exploratory 3%-prominence search is preserved only as discovery provenance;
its small-shoulder counts are not the documented primary-cycle measurements.

## Fixed parameters and scientific scope

C=6.835494476 pF, Cth=0.04421419302 pJ/K, Se=0.003712762043 mW/K,
T0=Tinit=314.363069806 K, Tc=333.338417242 K, w=7.041456088 K,
beta=0.3144377018 /K and gamma=0.1569637099. The remaining resistance scale and
activation parameters come from the preserved static preset. Initial V=0 and
the initial branch is insulating. The source uses the original 100425 specimen,
not the later M2 specimen. C and gamma lay outside that historical run's
configured constrained intervals; those intervals are not universal physical
laws. The old 22/22 classification claim is not adopted as quantitative
waveform agreement. No new specimen calibration is adopted here.

## Reproduction and checks

`neuristor simulate current --config reference_recipe.toml` reproduces the
finest constant-current control using the preserved local static preset.
The repository recipe is `experiments/current/alternating_amplitude_reference_20261025.toml`.
The exact CLI run, 18 full-resolution NPZ replays, original input files,
individual-cycle tables, parameter source, analysis/replay scripts and package
source snapshot are included. The replay script records the exact historical
bundle location and uses its archived numerical current samples, without refitting.
200 repository tests and repository validation pass. Synthetic signal checks,
artifact paths, input hashes and figures are separately checked. Browser
interaction is not claimed. Local archival publication does not upload to GitHub.

Next step, if pursued: test stronger timestep refinement and time horizon at
this fixed vector, then a controlled parameter sweep that distinguishes a
persistent two-cycle solution from transient/intermittent or numerical behavior.
"""
    bundle.write_text("report.md",report,label="Finding, parameters, evidence and numerical limitations")
    images=[]
    for name in ["archived_alternating_amplitudes.png","timestep_and_horizon_check.png"]:
        uri="data:image/png;base64,"+base64.b64encode((bundle.root/"figures"/name).read_bytes()).decode()
        images.append(f'<img src="{uri}" alt="{name}" style="width:100%">')
    introduction=f"""<!doctype html><html><head><meta charset="utf-8"><title>Recovered alternating-amplitude oscillations</title>
<style>body{{font:17px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#20313b;line-height:1.5}}.note{{background:#fff0d6;padding:18px;border-left:5px solid #b76b0e}}h1{{line-height:1.2}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px;border-bottom:1px solid #ddd}}</style></head><body>
<h1>Recovered: alternating deep and shallow voltage resets</h1><p>25 October 2026 · historical 7 September relaxed simulation · source label 400 mV (~303 µA).</p>
<p>The archived run shows primary-cycle amplitudes near <strong>{sim.low_phase_median_vpp_mV:.1f} and {sim.high_phase_median_vpp_mV:.1f} mV Vpp</strong> in the same trace. Neighboring source labels show related behavior.</p>
<p class="note"><strong>Simulation finding, not an accepted experimental result.</strong> The short pattern survives two finer timesteps, but a longer constant-current control becomes irregular at the finest step. A stable period-two state or a period-doubling bifurcation has not been established. The historical parameters are diagnostic, not an adopted calibration.</p>
<p><a href="report.md">Full documentation</a> · <a href="individual_cycles.csv">Individual cycle measurements</a> · <a href="cycle_summary.csv">Replay summary</a> · <a href="reference_recipe.toml">Reproduction recipe</a></p>
"""
    bundle.write_text("index.html",introduction+images[0]+fig.to_html(full_html=False,include_plotlyjs=True)+images[1]+"</body></html>",label="Standalone illustrated finding and interactive replays",media_type="text/html")
    for a in bundle._artifacts:
        assert (bundle.root/a['path']).is_file(), a
    bundle.complete(summary=metrics)
    print(bundle.root,flush=True)


if __name__ == "__main__":
    main()
