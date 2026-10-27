"""Regenerate the fixed-vector measured-current experiment from raw inputs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import multiprocessing

import numpy as np
import pandas as pd

from neuristor.analysis.alternating_amplitudes import cycle_table, summarize_cycles
from neuristor.analysis.experimental_waveforms import load_converted_sweep
from neuristor.analysis.lab_replay import major_branches
from neuristor.config import resolved_copy
from neuristor.current_drive_sim import simulate_current_step, simulate_current_waveforms
from neuristor.fitting.parameter_inference import prepare_inference_dataset
from neuristor.plotting.replay_plots import comparison_figure, scope_gallery, write_scope_gif
from neuristor.provenance import snapshot_package
from neuristor.runs import RunBundle


def _render_scope(job):
    """Render one saved native trajectory in a spawned worker."""
    native, acquired, branches, gif, png, current, dt, frames, window = job
    with np.load(native) as raw:
        keep = (raw["time_ns"] >= window[0]) & (raw["time_ns"] <= window[1])
        frame = pd.DataFrame(
            {
                key: raw[key][keep]
                for key in ("time_ns", "input_current_uA", "simulated_voltage_mV", "temperature_K", "resistance_ohm")
            }
        )
    frame["current_step_uA"] = current
    write_scope_gif(
        frame,
        pd.read_csv(branches),
        gif,
        frames=frames,
        snapshot_path=png,
        stacked_amperes=True,
        trajectory_breaks=True,
        time_window_ns=tuple(window),
        acquisition_frame=pd.read_csv(acquired),
        parameter_caption=f"Fixed alternating-reset diagnostic · dt={dt:g} ns",
    )


def _write_comparison(bundle, dataset, currents, dt):
    """Export bounded display samples; keep every native endpoint in the NPZs."""
    frames = []
    for index, label in enumerate(dataset.nominal_drives_mV):
        with np.load(bundle.root / f"dt_{dt:g}ns/current_{label:g}.npz") as raw:
            keep = np.unique(np.linspace(0, len(raw["time_ns"]) - 1, min(3000, len(raw["time_ns"]))).astype(int))
            time = raw["time_ns"][keep]
            frames.append(
                pd.DataFrame(
                    dict(
                        source_mV=float(label),
                        current_step_uA=float(currents[index]),
                        time_ns=time,
                        measured_voltage_mV=np.interp(time, dataset.time_ns, dataset.voltage_mV[:, index]),
                        simulated_voltage_mV=raw["simulated_voltage_mV"][keep],
                        resistance_ohm=raw["resistance_ohm"][keep],
                    )
                )
            )
    traces = pd.concat(frames, ignore_index=True)
    traces.to_csv(
        bundle.add_artifact("traces.csv", label="Display samples; full endpoints remain in native NPZs"), index=False
    )
    comparison_figure(traces, all_currents=True).write_html(
        bundle.add_artifact("comparison.html", label="Interactive all-current voltage and resistance comparison"),
        include_plotlyjs=True,
    )


def run_alternating_amplitudes(config, *, data_directory, output_root, command):
    """Integrate all measured current inputs at three steps; render two steps.

    Inputs come directly from the versioned laboratory CSVs. No archived fitted
    trace, rendered figure, or prior run is read. The historical parameter vector
    is an explicit declared model setting, not a new calibrated specimen fit.
    """
    from neuristor.workflows import _current_params_from_config

    base, *_ = _current_params_from_config(config)
    raw, _ = load_converted_sweep(data_directory)
    labels = sorted(raw.nominal_drive_mV.unique())
    dataset = prepare_inference_dataset(raw, holdout_drives_mV=[float(labels[0])])
    settings = config["animation_study"]
    bundle = RunBundle.create(
        name=config["name"],
        model="current",
        kind="analysis",
        config=resolved_copy(config),
        output_root=output_root,
        command=command,
    )
    print(f"RUN_DIRECTORY={bundle.root}", flush=True)
    try:
        bundle.write_json("source_hashes.json", snapshot_package(bundle), label="Complete numerical/rendering source")
        bundle.write_json("physics_parameters.json", asdict(base), label="Declared SI model parameters")
        branches = bundle.add_artifact("major_branches.csv", label="Model heating/cooling guides")
        major_branches(base.resist_params).to_csv(branches, index=False)
        plateau = (dataset.time_ns >= 150) & (dataset.time_ns <= 250)
        currents = np.median(dataset.current_uA[plateau], axis=0)
        rows, cycles, gallery, jobs = [], [], [], []
        acquired = []
        for i, label in enumerate(dataset.nominal_drives_mV):
            path = bundle.add_artifact(f"inputs/current_{label:g}.csv", label="Acquired input channels")
            pd.DataFrame(
                dict(
                    time_ns=dataset.time_ns,
                    input_current_uA=dataset.current_uA[:, i],
                    measured_voltage_mV=dataset.voltage_mV[:, i],
                )
            ).to_csv(path, index=False)
            acquired.append(path)
        for dt in settings["time_steps_ns"]:
            params = replace(
                base,
                dt_s=float(dt) * 1e-9,
                t_pre_s=max(0.0, -float(dataset.time_ns[0]) * 1e-9),
                t_end_s=float(dataset.time_ns[-1]) * 1e-9,
            )
            outputs = simulate_current_waveforms(
                dataset.current_uA, params, waveform_time_s=dataset.time_ns * 1e-9, audit_proximity=True
            )
            for i, out in enumerate(outputs):
                label = float(dataset.nominal_drives_mV[i])
                time_ns = out["t"].astype(float) * 1e9
                voltage_mV = out["V_vo2"].astype(float) * 1e3
                measured_cycles = cycle_table(time_ns, voltage_mV, settings["cycle_window_ns"])
                cycles.append(measured_cycles.assign(source_label_mV=label, dt_ns=dt))
                keep = (time_ns >= settings["cycle_window_ns"][0]) & (time_ns <= settings["cycle_window_ns"][1])
                tau = base.C_F * out["R"][keep].astype(float) * 1e9
                rows.append(
                    dict(
                        source_label_mV=label,
                        input_mode="measured_current",
                        current_uA=float(currents[i]),
                        dt_ns=dt,
                        **summarize_cycles(measured_cycles),
                        tau_th_ns=base.C_th_J_per_K / base.S_e_W_per_K * 1e9,
                        tau_e_min_ns=float(tau.min()),
                        tau_e_max_ns=float(tau.max()),
                        temperature_max_K=float(out["T"].max()),
                    )
                )
                stem = f"dt_{dt:g}ns/current_{label:g}"
                native = bundle.add_artifact(stem + ".npz", label="Full native simulation endpoints")
                np.savez_compressed(
                    native,
                    time_ns=time_ns,
                    simulated_voltage_mV=voltage_mV,
                    input_current_uA=out["I_in"].astype(float) * 1e6,
                    temperature_K=out["T"],
                    resistance_ohm=out["R"],
                )
                if dt in settings["render_steps_ns"]:
                    gif = bundle.add_artifact(stem + ".gif", label="Current/voltage/R(T) animation")
                    png = bundle.add_artifact(stem + ".png", label="Complete-trajectory image")
                    jobs.append(
                        (
                            native,
                            acquired[i],
                            branches,
                            gif,
                            png,
                            float(currents[i]),
                            dt,
                            int(settings["frames"]),
                            settings["window_ns"],
                        )
                    )
                    gallery.append(
                        dict(
                            label=f"{currents[i]:.3f} µA · source {label:g} mV · dt {dt:g} ns",
                            gif=gif.relative_to(bundle.root).as_posix(),
                            png=png.relative_to(bundle.root).as_posix(),
                        )
                    )
            print(f"Alternating replay: {len(outputs)} inputs at {dt:g} ns", flush=True)
        # The last slide also qualifies the longer constant-current result.
        # Recreate that central control at the same three integration steps.
        middle = int(np.argmin(abs(dataset.nominal_drives_mV - 400.0)))
        for dt in settings["time_steps_ns"]:
            control = replace(base, dt_s=float(dt) * 1e-9, t_pre_s=0.0, t_end_s=2e-6)
            out = simulate_current_step(float(currents[middle]), control)
            t = out["t"].astype(float) * 1e9
            v = out["V_vo2"].astype(float) * 1e3
            part = cycle_table(t, v, [1000.0, 2000.0])
            path = bundle.add_artifact(f"constant_current/dt_{dt:g}ns.npz", label="Longer constant-current control")
            np.savez_compressed(
                path, time_ns=t, simulated_voltage_mV=v, temperature_K=out["T"], resistance_ohm=out["R"]
            )
            rows.append(
                dict(
                    source_label_mV=float(dataset.nominal_drives_mV[middle]),
                    input_mode="constant_current",
                    current_uA=float(currents[middle]),
                    dt_ns=dt,
                    **summarize_cycles(part),
                    tau_th_ns=base.C_th_J_per_K / base.S_e_W_per_K * 1e9,
                    tau_e_min_ns=base.C_F * float(out["R"].min()) * 1e9,
                    tau_e_max_ns=base.C_F * float(out["R"].max()) * 1e9,
                    temperature_max_K=float(out["T"].max()),
                )
            )
        workers = int(settings["workers"])
        if workers == 1:
            for job in jobs:
                _render_scope(job)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for count, _ in enumerate(pool.map(_render_scope, jobs), 1):
                    print(f"Alternating animations: {count}/{len(jobs)}", flush=True)
        summary = pd.DataFrame(rows)
        summary.to_csv(
            bundle.add_artifact("cycle_summary.csv", label="Individual-cycle features and time constants"), index=False
        )
        pd.concat(cycles, ignore_index=True).to_csv(
            bundle.add_artifact("individual_cycles.csv", label="Raw individual peak/trough amplitudes"), index=False
        )
        bundle.write_json("animations.json", gallery, label="Current/timestep media index")
        _write_comparison(bundle, dataset, currents, max(settings["render_steps_ns"]))
        gallery_html = scope_gallery(gallery).replace(
            "All numerical data are in traces.csv.",
            "Display samples are in traces.csv; full native endpoints are in the dt_*ns folders.",
        )
        bundle.write_text(
            "START_HERE.html", gallery_html, label="All-current interactive gallery", media_type="text/html"
        )
        metrics = dict(
            source_currents=len(currents),
            native_integrations=len(rows),
            gifs=len(gallery),
            integration_timesteps_ns=settings["time_steps_ns"],
            tau_th_ns=base.C_th_J_per_K / base.S_e_W_per_K * 1e9,
            calibration_adopted=False,
            period_doubling_bifurcation_established=False,
        )
        bundle.write_json("metrics.json", metrics, label="Coverage and interpretation")
        bundle.write_text(
            "report.md",
            "# Recreated fixed-vector alternating resets\n\n"
            "All input histories were loaded from original measurements and integrated anew. "
            "No archived output was used. The saved diagnostic parameter vector is a declared "
            "setting, not a new specimen calibration. R and T are simulated. Individual-cycle "
            "amplitudes are raw peak minus preceding trough; smoothing only locates peaks. "
            "Short-window alternation does not establish stable period doubling: the historical "
            "longer-horizon control remains timestep-sensitive.\n",
            label="Methods and limitations",
        )
        bundle.complete(summary=metrics)
    except BaseException as error:
        bundle.fail(error)
        raise
    return bundle
