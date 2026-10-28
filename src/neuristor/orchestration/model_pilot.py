"""Bounded, auditable comparison of three optional material-model mechanisms."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
import copy
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from neuristor.analysis.experimental_waveforms import load_converted_sweep
from neuristor.config import ConfigError, resolved_copy, source_directory, validate_config
from neuristor.current_drive_sim import simulate_experimental_current_waveforms
from neuristor.fitting.filament_profile import acceptance
from neuristor.fitting.joint_inference import joint_model
from neuristor.fitting.model_pilot import (
    initial_cases,
    refinement_cases,
    pilot_metrics,
    evaluate_pilot_case,
    realized_units,
    realize_model,
)
from neuristor.fitting.parameter_inference import prepare_inference_dataset
from neuristor.fitting.steady_inference import measured_current_targets, settled_features
from neuristor.orchestration.filament_profile import _write, _read, _hash, _environment
from neuristor.orchestration.filament_study import _scalar_rows, convergence_table
from neuristor.provenance import snapshot_package
from neuristor.runs import RunBundle, _json_safe


def _context(config):
    from neuristor.workflows import _current_params_from_config

    base, *_ = _current_params_from_config(config)
    path = Path(config["steady"]["data_directory"])
    if not path.is_absolute():
        path = source_directory(config) / path
    raw, _ = load_converted_sweep(path)
    labels = sorted(raw.nominal_drive_mV.unique())
    dataset = prepare_inference_dataset(raw, holdout_drives_mV=[labels[0]])
    mask = (dataset.time_ns >= 50) & (dataset.time_ns <= 250)
    currents = dataset.current_uA[mask].mean(axis=0)
    targets = measured_current_targets(dataset, config["steady"]["target_current_range_uA"])
    settings = config["model_pilot"]

    def nearest(requested, eligible):
        result = [int(eligible[np.argmin(abs(currents[eligible] - x))]) for x in requested]
        if len(set(result)) != len(result):
            raise ConfigError("Requested pilot currents must select distinct records")
        return sorted(result, key=lambda i: currents[i])

    pilot = nearest(settings["target_currents_uA"], targets)
    controls = nearest(settings["control_currents_uA"], np.array([i for i in range(len(currents)) if i not in targets]))
    return dict(
        config=config,
        base=base,
        dataset=dataset,
        currents=currents,
        targets=targets,
        pilot=pilot,
        controls=controls,
        indices=sorted(pilot + controls),
        target_labels=dataset.nominal_drives_mV[pilot].tolist(),
        control_labels=dataset.nominal_drives_mV[controls].tolist(),
    )


def plan_model_pilot(config):
    """Validate a small fixed grid and disclose the measured-current selection."""
    validate_config(config)
    p = config["model_pilot"]
    if p["workers"] not in [1, 2]:
        raise ConfigError("Pilot uses one or two numerical workers")
    steps = p["verification_dt_ns"]
    if (
        len(steps) < 3
        or steps[0] != p["screen_dt_ns"]
        or any(not np.isfinite(x) or x <= 0 for x in steps)
        or any(b >= a for a, b in zip(steps[:-1], steps[1:]))
    ):
        raise ConfigError("Verification must include screening and at least two smaller positive steps")
    for name in ["secondary_capacity_ratios", "refinement_factors"]:
        if not len(p[name]) or any(not np.isfinite(x) or x <= 0 for x in p[name]):
            raise ConfigError("Positive ratios required: " + name)
    if not p["weights"] or any(not 0 < x < 1 for x in p["weights"]):
        raise ConfigError("Weights must be in (0,1)")
    if not p["phase_tau_ns"] or any(not np.isfinite(x) or x < 0 for x in p["phase_tau_ns"]):
        raise ConfigError("Nonnegative phase times required")
    if not p["coupling_ratios"] or any(not np.isfinite(x) or x < 0 for x in p["coupling_ratios"]):
        raise ConfigError("Nonnegative coupling required")
    for name in [
        "secondary_cooling_ratio",
        "voltage_scale_mV",
        "constant_duration_ns",
        "control_amplitude_floor_mV",
        "control_amplitude_ratio",
    ]:
        if not np.isfinite(p[name]) or p[name] <= 0:
            raise ConfigError("Positive finite value required: " + name)
    if len(p["refinement_factors"]) != 2:
        raise ConfigError("Declare two shared refinement factors")
    if config["steady"]["feature_method"] != "extrema" or not config["steady"].get("require_monotone_proximity"):
        raise ConfigError("Pilot uses extrema and audits proximity")
    static = [3, 4, 5, 6, 8, 9, 10]
    vectors = np.array([a["values"] for a in p["anchors"]], float)
    if (
        vectors.ndim != 2
        or vectors.shape[1] != 11
        or not np.isfinite(vectors).all()
        or not np.all(vectors[:, static] == vectors[0, static])
    ):
        raise ConfigError("Anchors must share static R(T) and ambient temperature")
    cases = initial_cases(p)
    context = _context(config)
    return dict(
        declared_cases=len(cases),
        shared_refinements_up_to=32,
        verification_cases_up_to=7,
        mechanisms=["baseline", "warm_parallel", "series_bottleneck", "phase_lag"],
        workers=p["workers"],
        pilot_currents_uA=context["currents"][context["pilot"]].tolist(),
        control_currents_uA=context["currents"][context["controls"]].tolist(),
        full_verification_records=len(context["currents"]),
        verification_dt_ns=steps,
        caveat="Historically seen records; fixed-grid mechanism pilot, not a blind test or calibration",
    )


_WORKER_CONTEXT = None


def _initialize_worker(root):
    global _WORKER_CONTEXT
    import torch

    torch.set_num_threads(1)
    _WORKER_CONTEXT = _context(_read(Path(root) / "execution_config.json"))


def _screen(root, case):
    root = Path(root)
    context = _WORKER_CONTEXT
    folder = root / "cases" / case["id"]
    folder.mkdir(parents=True, exist_ok=False)
    _write(folder / "case.json", case)
    started = time.monotonic()
    try:
        p = context["config"]["model_pilot"]
        rows, _, _ = evaluate_pilot_case(context, case, p["screen_dt_ns"], context["indices"])
        pd.DataFrame(_scalar_rows(rows)).to_csv(folder / "features.csv", index=False)
        metrics = pilot_metrics(rows, context["target_labels"], context["control_labels"], p)
        result = dict(case, metrics=metrics, simulated=True, elapsed_s=time.monotonic() - started)
    except (ValueError, FloatingPointError, OverflowError) as error:
        result = dict(
            case,
            simulated=False,
            error=str(error),
            metrics=dict(rank=[1, len(context["pilot"]), len(context["controls"]), 1e99]),
            elapsed_s=time.monotonic() - started,
        )
    _write(folder / "result.json", result)
    return result


def _verification(root, case):
    context = _WORKER_CONTEXT
    root = Path(root)
    p = context["config"]["model_pilot"]
    settings = context["config"]["steady"]
    folder = root / "verification" / case["id"]
    folder.mkdir(parents=True, exist_ok=False)
    tables = []
    samples = {}
    traces = []
    native_audit = []
    time_ns = context["dataset"].time_ns[context["dataset"].time_ns <= settings["window_ns"][1]]
    all_indices = np.arange(len(context["currents"]))
    for dt in p["verification_dt_ns"]:

        def native(index, out, params):
            if dt != min(p["verification_dt_ns"]) or case["role"] != "candidate" or index not in context["indices"]:
                return
            target = folder / "native" / f"current_{context['dataset'].nominal_drives_mV[index]:g}.npz"
            target.parent.mkdir(exist_ok=True)
            np.savez_compressed(target, **out)
            v = out["V_vo2"].astype(float)
            r = out["R"].astype(float)
            total = v * v / r
            allocated = out.get("P_primary", out["P"]).astype(float) + out.get("P_secondary", np.zeros_like(v)).astype(
                float
            )
            relative = float(np.max(abs(total - allocated) / np.maximum(abs(total), 1e-30)))
            if relative > 1e-6:
                raise ValueError("Native branch-power accounting failed")
            native_audit.append(dict(index=index, relative_power_error=relative))

        rows, voltage, _ = evaluate_pilot_case(context, case, dt, all_indices, native_callback=native)
        table = pd.DataFrame(_scalar_rows(rows)).assign(candidate=case["id"], dt_ns=dt)
        table["target"] = table.source_label_mV.isin(context["dataset"].nominal_drives_mV[context["targets"]])
        table["pilot_target"] = table.source_label_mV.isin(context["target_labels"])
        tables.append(table)
        samples[(case["id"], dt)] = voltage
        if dt == min(p["verification_dt_ns"]):
            for i, label in enumerate(context["dataset"].nominal_drives_mV):
                traces.append(
                    pd.DataFrame(
                        dict(
                            source_label_mV=label,
                            current_uA=context["currents"][i],
                            time_ns=time_ns,
                            input_current_uA=context["dataset"].current_uA[: len(time_ns), i],
                            measured_voltage_mV=context["dataset"].voltage_mV[: len(time_ns), i],
                            simulated_voltage_mV=voltage[:, i],
                        )
                    )
                )
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(folder / "verification.csv", index=False)
    conv = convergence_table(table, samples, time_ns, dict(settings, **p["success"]))
    conv.to_csv(folder / "convergence.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(folder / "traces.csv", index=False)
    finest = table[table.dt_ns == min(p["verification_dt_ns"])]
    accepted = acceptance(finest, conv, p["success"])
    targets = finest[finest.target]
    deltas = np.concatenate(
        [
            (pd.to_numeric(targets["predicted_" + field], errors="coerce") - targets["measured_" + field]).to_numpy(
                float
            )
            for field in ["peak_mV", "trough_mV"]
        ]
    )
    shape_valid = bool(np.isfinite(deltas).all())
    shape_rms = float(np.sqrt(np.mean(deltas**2))) if shape_valid else None
    accepted["gates"]["shape_accuracy"] = bool(shape_valid and shape_rms <= p["success"]["shape_RMSE_mV"])
    accepted["success"] = all(accepted["gates"].values())
    accepted["failed_gates"] = [key for key, value in accepted["gates"].items() if not value]
    pilot = pilot_metrics(finest.to_dict("records"), context["target_labels"], context["control_labels"], p)
    selected = finest[finest.pilot_target].sort_values("measured_current_A")
    trend = bool(selected.frequency_valid.all() and np.all(np.diff(selected.predicted_robust_vpp_mV) <= 0))
    params = joint_model(case["values"], context["base"], min(p["verification_dt_ns"]))
    result = dict(
        case,
        acceptance=accepted,
        pilot=pilot,
        shape_RMSE_mV=shape_rms,
        amplitude_decreases_across_pilot=trend,
        maximum_temperature_K=float(finest.temperature_max_K.max()),
        realized_SI=realized_units(params, case["specification"]),
        native_power_audit=native_audit,
    )
    _write(folder / "result.json", result)
    return result


def _long_control(root, case):
    root = Path(root)
    context = _WORKER_CONTEXT
    p = context["config"]["model_pilot"]
    duration = p["constant_duration_ns"]
    currents = context["currents"][context["pilot"]]
    rows = []
    for dt in p["verification_dt_ns"][-3:]:
        params = joint_model(case["values"], context["base"], dt)
        params = replace(params, t_pre_s=0.0, t_end_s=duration * 1e-9)
        primary, experiment = realize_model(params, case["specification"])
        outputs = simulate_experimental_current_waveforms(
            np.tile(currents, (2, 1)),
            primary,
            experiment,
            waveform_time_s=np.array([0.0, duration]) * 1e-9,
            audit_proximity=True,
        )
        for current, out in zip(currents, outputs):
            features = settled_features(
                out["t"].astype(float) * 1e9,
                out["V_vo2"].astype(float) * 1e3,
                [duration / 2, duration],
                method="extrema",
            )
            temperatures = [out["T"], out.get("T_secondary", out["T"])]
            rows.append(
                dict(
                    current_uA=current,
                    dt_ns=dt,
                    **_scalar_rows([features])[0],
                    temperature_valid=bool(
                        all(
                            t.min() >= params.resist_params.T_min_K and t.max() <= params.resist_params.T_max_K
                            for t in temperatures
                        )
                    ),
                    maximum_temperature_K=float(max(t.max() for t in temperatures)),
                    proximity_admissible=bool(
                        np.isfinite(out["minimum_proximity_slope"]) and out["minimum_proximity_slope"] >= 0
                    ),
                )
            )
    frame = pd.DataFrame(rows)
    good = True
    for _, group in frame.groupby("current_uA"):
        group = group.sort_values("dt_ns", ascending=False)
        good &= bool(group.sustained.all() and group.temperature_valid.all() and group.proximity_admissible.all())
        for field in ["frequency_MHz", "robust_vpp_mV"]:
            values = pd.to_numeric(group[field], errors="coerce").to_numpy(float)
            good &= bool(
                np.isfinite(values).all()
                and np.all(values > 0)
                and np.all(abs(np.diff(values)) / values[1:] <= p["success"]["convergence_fraction"])
            )
    folder = root / "long_controls" / case["id"]
    folder.mkdir(parents=True, exist_ok=False)
    frame.to_csv(folder / "features.csv", index=False)
    result = dict(id=case["id"], kind=case["specification"]["kind"], persistent_physical_and_converged=good)
    _write(folder / "result.json", result)
    return result


def _tasks(executor, function, root, cases, phase, progress):
    queue = list(cases)
    active = {}
    results = []
    while queue or active:
        while queue and len(active) < progress["workers"]:
            case = queue.pop(0)
            active[executor.submit(function, str(root), case)] = case
        done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
        for future in done:
            case = active.pop(future)
            result = future.result()
            results.append(result)
            progress["completed"] += 1
            _write(
                root / "status.json",
                dict(
                    state="running",
                    phase=phase,
                    completed_tasks=progress["completed"],
                    active=[c["id"] for c in active.values()],
                    elapsed_s=time.monotonic() - progress["started"],
                ),
            )
            print(f"{phase}: {len(results)}/{len(cases)} — {case['id']}", flush=True)
            if function is _screen:
                with (root / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(_json_safe(result), allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
    return results


def run_model_pilot(config, *, output_root=None, command="neuristor analyze model-pilot"):
    """Freeze inputs, screen mechanisms, then replay winners and matched ablations."""
    plan = plan_model_pilot(config)
    bundle = RunBundle.create(
        name=config["name"],
        model="experimental-material-model-pilot",
        kind="analysis",
        config=resolved_copy(config),
        output_root=output_root or config["output"]["root"],
        command=command,
    )
    print("PILOT_DIRECTORY=" + str(bundle.root), flush=True)
    root = bundle.root
    execution = resolved_copy(config)
    for section, key, relative in [
        ("resistance", "preset", "inputs/resistance_preset.json"),
        ("steady", "resistance_data", "inputs/resistance_data.tsv"),
    ]:
        source = Path(config[section][key])
        source = source if source.is_absolute() else source_directory(config) / source
        shutil.copy2(source, bundle.add_artifact(relative, label="Original measured/static input"))
        execution[section][key] = relative
    directory = Path(config["steady"]["data_directory"])
    if not directory.is_absolute():
        directory = source_directory(config) / directory
    for source in sorted(directory.glob("*_converted.csv")):
        shutil.copy2(
            source, bundle.add_artifact("inputs/current/" + source.name, label="Original current/voltage measurement")
        )
    execution["steady"]["data_directory"] = "inputs/current"
    execution["_source"] = str(root / "execution_config.json")
    bundle.write_json("execution_config.json", execution, label="Execution configuration rebound to copied inputs")
    shutil.copy2(config["_source"], bundle.add_artifact("recipe.toml", label="Original recipe"))
    bundle.write_json("source_hashes.json", snapshot_package(bundle), label="Executable source snapshot")
    bundle.write_json("environment.json", _environment(), label="Numerical package versions")
    bundle.write_json("plan.json", plan, label="Prespecified mechanism grid and selection")
    context = _context(execution)
    pd.DataFrame(
        [
            dict(
                index=i,
                source_label_mV=label,
                current_uA=context["currents"][i],
                pilot_target=i in context["pilot"],
                pilot_control=i in context["controls"],
                full_target=i in context["targets"],
            )
            for i, label in enumerate(context["dataset"].nominal_drives_mV)
        ]
    ).to_csv(bundle.add_artifact("current_split.csv", label="Actual measured currents and pilot roles"), index=False)
    progress = dict(workers=plan["workers"], completed=0, started=time.monotonic())
    for name in ["status.json", "report.md", "metrics.json", "START_HERE.html", "history.jsonl"]:
        bundle.add_artifact(name, label="Pilot " + name)
    _write(root / "status.json", dict(state="running", phase="declared grid"))
    (root / "report.md").write_text(
        "# Material-model pilot running\n\nNo result is available yet. See status.json and history.jsonl.\n"
    )
    (root / "START_HERE.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><h1>Material-model pilot running</h1><p>See status.json for numerical progress. No result is available yet.</p>'
    )
    _write(root / "metrics.json", dict(state="running", calibration_adopted=False))
    (root / "history.jsonl").touch()
    bundle._save_manifest()
    for name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[name] = "1"
    try:
        with ProcessPoolExecutor(
            max_workers=plan["workers"],
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(str(root),),
        ) as executor:
            results = _tasks(executor, _screen, root, initial_cases(config["model_pilot"]), "declared grid", progress)

            def winners():
                chosen = []
                for kind in plan["mechanisms"]:
                    eligible = [r for r in results if r["specification"]["kind"] == kind and r["simulated"]]
                    if not eligible:
                        raise RuntimeError("No finite numerical trajectory in " + kind)
                    chosen.append(min(eligible, key=lambda r: r["metrics"]["rank"]))
                return chosen

            refinements = refinement_cases(winners(), config["model_pilot"], config["steady"]["bounds"])
            bundle.write_json(
                "refinement_plan.json", refinements, label="Prespecified shared one-coordinate follow-ups"
            )
            results += _tasks(executor, _screen, root, refinements, "shared dynamic refinements", progress)
            chosen = winners()
            verification = []
            for case in chosen:
                verification.append(dict(case, role="candidate"))
                if case["specification"]["kind"] != "baseline":
                    ablation = copy.deepcopy(case)
                    ablation.update(
                        id="matched_baseline_" + case["id"],
                        specification=dict(kind="baseline"),
                        role="matched_baseline",
                        paired_with=case["id"],
                    )
                    verification.append(ablation)
            bundle.write_json(
                "verification_plan.json", verification, label="Winners and exact-vector mechanism-off comparisons"
            )
            verified = _tasks(
                executor, _verification, root, verification, "all-current and timestep verification", progress
            )
            controls = _tasks(executor, _long_control, root, chosen, "long constant-current controls", progress)
        for folder in ["cases", "verification", "long_controls"]:
            for path in (root / folder).rglob("*"):
                if path.is_file():
                    bundle.register_file(path.relative_to(root), label=folder + " evidence")
        control_by_id = {c["id"]: c for c in controls}
        successes = [
            r["id"]
            for r in verified
            if r["role"] == "candidate"
            and r["acceptance"]["success"]
            and control_by_id[r["id"]]["persistent_physical_and_converged"]
        ]
        metrics = dict(
            state="completed",
            declared_cases=plan["declared_cases"],
            refinement_cases=len(refinements),
            total_screened_cases=len(results),
            finite_screened_cases=sum(r["simulated"] for r in results),
            verified_cases=len(verified),
            long_control_cases=len(controls),
            successful_candidates=successes,
            calibration_adopted=False,
            elapsed_s=time.monotonic() - progress["started"],
            scope="Fixed-grid mechanism pilot; records and historical anchors seen previously",
        )
        bundle.write_json("metrics.json", metrics, label="Actual pilot coverage and joint success")
        from neuristor.plotting.model_pilot import render_model_pilot

        render_model_pilot(bundle, results, verified, controls, context)
        hashes = {
            path.relative_to(root).as_posix(): _hash(path)
            for path in root.rglob("*")
            if path.is_file()
            and path.name not in ["run.json", "artifact_hashes.json", "status.json"]
            and "__pycache__" not in path.parts
        }
        bundle.write_json("artifact_hashes.json", hashes, label="Immutable input/source/numerical artifact inventory")
        _write(root / "status.json", metrics)
        bundle.complete(summary=metrics)
    except BaseException as error:
        _write(root / "status.json", dict(state="failed", error=str(error)))
        for path in root.rglob("*"):
            if path.is_file() and path.name != "run.json":
                bundle.register_file(path.relative_to(root), label="Preserved pilot evidence")
        bundle.fail(error)
        raise
    return bundle
