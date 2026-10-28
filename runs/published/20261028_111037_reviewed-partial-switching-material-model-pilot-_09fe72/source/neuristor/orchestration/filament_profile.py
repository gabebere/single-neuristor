"""Checkpointed, bounded overnight profile of fixed filament conductance weights.

Searches call the existing fitter and authoritative solver. The supervisor only
owns scheduling, immutable inputs, progress, verification and provenance.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import copy
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

from neuristor.analysis.experimental_waveforms import load_converted_sweep
from neuristor.config import ConfigError, resolved_copy, source_directory, validate_config
from neuristor.current_drive_sim import simulate_current_waveforms
from neuristor.fitting.filament_profile import profile_bounds, profile_seeds, acceptance, diagnostic_rank
from neuristor.fitting.joint_inference import NAMES, natural_vector, joint_model
from neuristor.fitting.parameter_inference import prepare_inference_dataset
from neuristor.fitting.resistance_custom_analysis import load_experimental_rt
from neuristor.fitting.steady_inference import (
    search_settled,
    evaluate_settled,
    measured_current_targets,
    settled_features,
)
from neuristor.orchestration.filament_study import summarize, convergence_table, _scalar_rows
from neuristor.provenance import snapshot_package
from neuristor.runs import RunBundle, _json_safe


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(_json_safe(payload), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text())


def _environment():
    return dict(
        python=sys.version,
        packages={
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "pandas", "torch", "matplotlib", "plotly"]
        },
    )


def _context(config):
    from neuristor.workflows import _current_params_from_config

    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (source_directory(config) / path).resolve()

    base, *_ = _current_params_from_config(config)
    raw, _ = load_converted_sweep(resolve(config["steady"]["data_directory"]))
    labels = sorted(raw.nominal_drive_mV.unique())
    dataset = prepare_inference_dataset(raw, holdout_drives_mV=[labels[0]])
    targets = measured_current_targets(dataset, config["steady"]["target_current_range_uA"])
    currents = np.mean(dataset.current_uA[(dataset.time_ns >= 50) & (dataset.time_ns <= 250)], axis=0)
    targets = np.array(sorted(targets, key=lambda i: currents[i]), int)
    ranks = config["profile_search"]["validation_target_ranks"]
    if len(set(ranks)) != len(ranks) or any(not isinstance(r, int) or r < 0 or r >= len(targets) for r in ranks):
        raise ConfigError("Validation ranks must be distinct target indices")
    validation = targets[ranks]
    training = np.array([i for i in targets if i not in validation], int)
    if not len(training) or not len(validation):
        raise ConfigError("Both fitting and optimizer-excluded targets are required")
    robust = []
    measured = []
    for i in range(len(labels)):
        features = settled_features(
            dataset.time_ns, dataset.voltage_mV[:, i], config["steady"]["window_ns"], method="extrema"
        )
        measured.append(features)
        if i in targets and features["sustained"]:
            robust.append(i)
    if not robust:
        raise ConfigError("Long controls require at least one measured sustained target")
    controls = list(dict.fromkeys(robust[i] for i in [0, len(robust) // 2, len(robust) - 1]))
    table = pd.read_csv(resolve(config["steady"]["seed_table"])).set_index("candidate")
    seeds = [
        dict(
            values=config["profile_search"]["previous_baseline"]["values"],
            source="previous baseline",
            filament_fraction=1.0,
        ),
        *[
            dict(values=config["filament_study"][key]["values"], source=key, filament_fraction=1.0)
            for key in ["presentation_frequency", "presentation_combined"]
        ],
        *[
            dict(values=table.loc[key, list(NAMES)].to_numpy(float).tolist(), source=key, filament_fraction=1.0)
            for key in config["steady"]["seed_candidates"]
        ],
        dict(values=natural_vector(base).tolist(), source="static preset", filament_fraction=1.0),
    ]
    return dict(
        config=config,
        base=base,
        dataset=dataset,
        targets=targets,
        training=training,
        validation=validation,
        controls=controls,
        currents=currents,
        measured=measured,
        seeds=seeds,
        rt=load_experimental_rt(resolve(config["steady"]["resistance_data"])),
    )


def plan_filament_profile(config):
    """Validate the experimental schedule and disclose target/control selection."""
    validate_config(config)
    profile = config["profile_search"]
    fractions = list(map(float, profile["fractions"]))
    if (
        len(fractions) < 2
        or fractions[0] != 1
        or len(set(fractions)) != len(fractions)
        or any(not 0 < f <= 1 for f in fractions)
        or fractions != sorted(fractions, reverse=True)
    ):
        raise ConfigError("Fractions must be distinct, decreasing and start at 1")
    if profile["families"] != ["inherited", "volume_scaled"]:
        raise ConfigError("Both declared thermal families are required")
    if not 0 < float(profile["profile_hours"]) < float(profile["wall_hours"]) <= 12:
        raise ConfigError("Require 0 < profile hours < total hours <= 12")
    if int(profile["workers"]) not in [1, 2]:
        raise ConfigError("Use one or two independent numerical workers")
    if int(profile["population"]) < 5 or int(profile["restarts"]) < 1 or int(profile["maximum_evaluations"]) < 1:
        raise ConfigError("Require population >=5, positive restarts and evaluation cap")
    if int(profile["maximum_generations"]) < 0 or int(profile["local_evaluations"]) < 1:
        raise ConfigError("Require nonnegative generations and positive local evaluations")
    if any(not np.isfinite(v) or v <= 0 for v in profile["success"].values()):
        raise ConfigError("Success tolerances must be finite and positive")
    for key in [
        "baseline_seconds",
        "forward_seconds",
        "reverse_seconds",
        "refinement_seconds",
        "constant_current_duration_ns",
    ]:
        if not np.isfinite(profile[key]) or profile[key] <= 0:
            raise ConfigError("Positive finite profile budget required: " + key)
    for key in ["verification_dt_ns", "final_dt_ns"]:
        steps = profile[key]
        if (
            len(steps) < 3
            or any(not np.isfinite(x) or x <= 0 for x in steps)
            or any(b >= a for a, b in zip(steps[:-1], steps[1:]))
        ):
            raise ConfigError("Verification needs three decreasing positive timesteps")
    if (
        profile["verification_dt_ns"][0] != profile["search_dt_ns"]
        or profile["final_dt_ns"][0] != profile["refinement_dt_ns"]
    ):
        raise ConfigError("Each verification must start at its actual search timestep")
    if profile["mean_voltage_scale_mV"] <= 0 or not np.isfinite(profile["mean_voltage_scale_mV"]):
        raise ConfigError("A positive finite mean-voltage scale is required")
    if config["steady"]["feature_method"] != "extrema" or config["steady"]["loss_function"] != "relative_squared":
        raise ConfigError("Use the documented frequency/Vpp feature objective")
    if config["steady"]["fixed_parameters"] != ["T0_K"]:
        raise ConfigError("The static law and T0 remain fixed")
    context = _context(config)
    split = []
    for i, current in enumerate(context["currents"]):
        group = (
            "training"
            if i in context["training"]
            else "optimizer_excluded"
            if i in context["validation"]
            else "control"
        )
        split.append(
            dict(
                index=i,
                source_label_mV=float(context["dataset"].nominal_drives_mV[i]),
                current_uA=float(current),
                group=group,
            )
        )
    return dict(
        fractions=fractions,
        workers=profile["workers"],
        wall_hours=profile["wall_hours"],
        profile_hours=profile["profile_hours"],
        primary_profiles=1 + 2 * (len(fractions) - 1),
        reverse_continuations=2 * (len(fractions) - 1),
        adaptive_midpoints_up_to=4,
        final_refinements_up_to=6,
        split=split,
        constant_control_indices=context["controls"],
        success=profile["success"],
        caveat="Optimizer-excluded currents were seen previously and informed historical seeds; they are not blind validation.",
    )


def prepare_filament_profile(config, *, output_root=None, command="neuristor analyze filament-profile"):
    """Freeze all original inputs and executable source before detached execution."""
    plan = plan_filament_profile(config)
    bundle = RunBundle.create(
        name=config["name"],
        model="overnight-filament-profile",
        kind="analysis",
        config=resolved_copy(config),
        output_root=output_root or config["output"]["root"],
        command=command,
    )
    execution = resolved_copy(config)

    def original(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (source_directory(config) / path).resolve()

    bindings = [
        ("resistance", "preset", "inputs/resistance_preset.json"),
        ("steady", "resistance_data", "inputs/resistance_data.tsv"),
        ("steady", "seed_table", "inputs/seed_vectors.csv"),
    ]
    for section, key, relative in bindings:
        shutil.copy2(
            original(config[section][key]), bundle.add_artifact(relative, label="Original declared numerical input")
        )
        execution[section][key] = relative
    for source in sorted(original(config["steady"]["data_directory"]).glob("*_converted.csv")):
        shutil.copy2(source, bundle.add_artifact("inputs/current/" + source.name, label="Original measured channels"))
    execution["steady"]["data_directory"] = "inputs/current"
    execution["_source"] = str(bundle.root / "execution_config.json")
    bundle.write_json("execution_config.json", execution, label="Config rebound only to frozen input copies")
    shutil.copy2(
        config["_source"], bundle.add_artifact("inputs/original_recipe.toml", label="Versioned original recipe")
    )
    bundle.write_json("source_hashes.json", snapshot_package(bundle), label="Complete executable source snapshot")
    bundle.write_json("environment.json", _environment(), label="Python and numerical package versions")
    bundle.write_json("plan.json", plan, label="Budgets, success criteria and prespecified current split")
    pd.DataFrame(plan["split"]).to_csv(
        bundle.add_artifact("current_split.csv", label="Physical current values and optimization groups"), index=False
    )
    frozen = [p for lane in ["source", "inputs"] for p in (bundle.root / lane).rglob("*") if p.is_file()]
    frozen += [
        bundle.root / name for name in ["execution_config.json", "environment.json", "plan.json", "current_split.csv"]
    ]
    hashes = {p.relative_to(bundle.root).as_posix(): _hash(p) for p in frozen}
    bundle.write_json("frozen_hashes.json", hashes, label="Immutable numerical input/source closure")
    for name in ["status.json", "state.json", "metrics.json", "report.md", "START_HERE.html"]:
        bundle.add_artifact(name, label="Live profile " + name)
    _write(bundle.root / "state.json", dict(started_epoch=None, completed={}, attempts=0))
    _write(bundle.root / "status.json", dict(state="prepared", phase="ready", planned=plan))
    _write(bundle.root / "metrics.json", dict(state="prepared", calibration_adopted=False, successful_candidates=[]))
    (bundle.root / "report.md").write_text(
        "# Overnight fixed-f profile (prepared)\n\nSearch has not started. Plan and frozen inputs are ready.\n"
    )
    (bundle.root / "START_HERE.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>Overnight filament profile: prepared</h1><p>See status.json. No scientific result is available yet.</p>'
    )
    bundle._save_manifest()
    return bundle


@contextmanager
def _exclusive(root):
    lock = (root / "supervisor.lock").open("a+")
    try:
        if os.name == "nt":
            import msvcrt

            lock.seek(0)
            lock.write("0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        lock.close()


_CONTEXTS = {}


def _worker_context(root):
    import torch

    torch.set_num_threads(1)
    key = str(root)
    if key not in _CONTEXTS:
        _CONTEXTS[key] = _context(_read(root / "execution_config.json"))
    return _CONTEXTS[key]


def _compact(record):
    result = dict(record)
    if "per_current" in result:
        result["per_current"] = _scalar_rows(result["per_current"])
    return result


def _history(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    records = []
    valid = 0
    lines = raw.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i != len(lines) - 1:
                raise ValueError("Malformed completed history line")
            break
        if not line.endswith(b"\n"):
            break
        records.append(record)
        valid += len(line)
    if valid != len(raw):
        with path.open("r+b") as stream:
            stream.truncate(valid)
    return records


def _search_task(root, task, folder):
    context = _worker_context(root)
    profile = context["config"]["profile_search"]
    local = copy.deepcopy(context["config"]["steady"])
    local.update(
        bounds=profile_bounds(local, task["family"], task["fraction"]),
        fit_filament_fraction=False,
        restarts=int(task.get("restarts", profile["restarts"])),
        fixed_resistance_restarts=int(task.get("restarts", profile["restarts"])),
        population=int(task.get("population", profile["population"])),
        maximum_generations=int(task.get("maximum_generations", profile["maximum_generations"])),
        maximum_evaluations=int(task.get("maximum_evaluations", profile["maximum_evaluations"])),
        local_evaluations=int(profile["local_evaluations"]),
        search_dt_ns=task["dt_ns"],
        seed=task["seed"],
        mean_voltage_weight=task.get("mean_weight", 0.0),
        mean_voltage_scale_mV=profile["mean_voltage_scale_mV"],
    )
    base = replace(context["base"], filament_fraction=task["fraction"])
    seeds = profile_seeds(base, task["seed_records"], local["bounds"], task["fraction"])
    _write(folder / "seed_vectors.json", seeds)
    local["initial_seed_count"] = min(len(seeds), max(1, local["population"] // 2))
    if not (folder / "timing.json").exists():
        _write(
            folder / "timing.json",
            dict(started_epoch=time.time(), deadline=min(task["deadline"], time.time() + task["seconds"])),
        )
    timing = _read(folder / "timing.json")
    records = _history(folder / "history.jsonl")
    remaining = min(timing["deadline"], task["deadline"]) - time.time()
    dataset = replace(
        context["dataset"],
        train_indices=np.array(task["training"], int),
        test_indices=np.array([i for i in range(len(context["currents"])) if i not in task["training"]], int),
    )
    local["search_seconds"] = max(0.001, remaining)
    with (folder / "history.jsonl").open("a") as stream:

        def checkpoint(record, snapshot):
            record["filament_fraction"] = task["fraction"]
            record["elapsed_s"] = time.time() - timing["started_epoch"]
            slim = _compact(record)
            stream.write(json.dumps(_json_safe(slim), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            slim_snapshot = dict(
                snapshot,
                best=_compact(snapshot["best"]) if snapshot["best"] else None,
                elites=[_compact(r) for r in snapshot["elites"]],
            )
            _write(folder / "checkpoint.json", slim_snapshot)
            _write(
                folder / "status.json",
                dict(
                    state="searching",
                    unique_candidates=snapshot["unique_candidates"],
                    best_training_eligible=snapshot["best"] is not None,
                    elapsed_s=time.time() - timing["started_epoch"],
                    fraction=task["fraction"],
                    family=task["family"],
                ),
            )

        result = search_settled(
            dataset,
            context["rt"],
            base,
            local,
            [s["values"] for s in seeds],
            pd.DataFrame(),
            checkpoint,
            stop_requested=lambda: time.time() >= min(timing["deadline"], task["deadline"])
            or (root / "STOP").exists()
            or (root / "ABORT").exists(),
            initial_records=records,
        )
    history = _history(folder / "history.jsonl")
    usable = [r for r in history if r.get("simulated")]
    if not usable:
        raise RuntimeError("No numerical candidate was evaluated in " + task["id"])
    selected = result["best"] or result["diagnostic_best"] or min(usable, key=lambda row: row["score"])
    _write(folder / "search_result.json", result)
    return dict(
        kind="search",
        stage=task["id"],
        family=task["family"],
        fraction=task["fraction"],
        values=selected["values"],
        filament_fraction=task["fraction"],
        winner=_compact(selected),
        training=task["training"],
        mean_weight=task.get("mean_weight", 0.0),
        unique_candidates=len(history),
        search_elapsed_s=time.time() - timing["started_epoch"],
        eligible_training=bool(result["best"]),
        objective_label="mean-aware" if task.get("mean_weight", 0) else "frequency-plus-Vpp",
    )


def _verify_task(root, task, folder):
    context = _worker_context(root)
    config = context["config"]
    settings = dict(config["steady"], mean_voltage_weight=0.0)
    candidate = task["candidate"]
    base = replace(context["base"], filament_fraction=candidate["fraction"])
    data = context["dataset"]
    keep = data.time_ns <= settings["window_ns"][1]
    times = data.time_ns[keep]
    tables = []
    sampled = {}
    traces = []
    for dt in task["steps"]:
        if time.time() >= task["deadline"] or (root / "STOP").exists() or (root / "ABORT").exists():
            raise InterruptedError("Verification interrupted or wall-clock budget exhausted")

        def native(index, out, params):
            if not task.get("save_native") or dt != min(task["steps"]):
                return
            path = folder / "native" / f"current_{data.nominal_drives_mV[index]:g}.npz"
            path.parent.mkdir(exist_ok=True)
            voltage = out["V_vo2"].astype(float)
            hot = out.get("R_reference_hot", out["R"]).astype(float)
            np.savez_compressed(
                path,
                time_ns=out["t"].astype(float) * 1e9,
                voltage_V=voltage,
                input_current_A=out["I_in"],
                temperature_K=out["T"],
                effective_resistance_ohm=out["R"],
                reference_hot_resistance_ohm=hot,
                filament_current_A=out.get("I_filament", voltage / hot),
                surrounding_current_A=out.get("I_surrounding", np.zeros_like(voltage)),
                filament_power_W=out.get("P_filament", out["P"]),
                surrounding_power_W=out.get("P_surrounding", np.zeros_like(voltage)),
                filament_fraction=params.filament_fraction,
            )

        rows, voltage, _ = evaluate_settled(
            data, candidate["values"], base, np.arange(len(context["currents"])), settings, dt, native_callback=native
        )
        table = pd.DataFrame(_scalar_rows(rows)).assign(candidate=task["id"], dt_ns=dt)
        table["target"] = table.source_label_mV.isin(data.nominal_drives_mV[context["targets"]])
        table["optimizer_excluded"] = table.source_label_mV.isin(data.nominal_drives_mV[context["validation"]])
        table["used_in_this_fit"] = table.source_label_mV.isin(data.nominal_drives_mV[candidate["training"]])
        tables.append(table)
        sampled[(task["id"], dt)] = voltage
        if dt == min(task["steps"]):
            for i, label in enumerate(data.nominal_drives_mV):
                traces.append(
                    pd.DataFrame(
                        dict(
                            candidate=task["id"],
                            source_label_mV=label,
                            current_uA=context["currents"][i],
                            time_ns=times,
                            input_current_uA=data.current_uA[keep, i],
                            measured_voltage_mV=data.voltage_mV[keep, i],
                            simulated_voltage_mV=voltage[:, i],
                        )
                    )
                )
    table = pd.DataFrame.from_records([row for frame in tables for row in frame.to_dict("records")])
    table.to_csv(folder / "verification.csv", index=False)
    convergence = convergence_table(table, sampled, times, dict(settings, **config["profile_search"]["success"]))
    convergence.to_csv(folder / "convergence.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(folder / "traces.csv", index=False)
    fine = table[table.dt_ns == min(task["steps"])]
    results = {}
    for label, indices in [
        ("training", context["training"]),
        ("optimizer_excluded", context["validation"]),
        ("all_target", context["targets"]),
    ]:
        group = fine[fine.source_label_mV.isin(data.nominal_drives_mV[indices])]
        results[label] = summarize(group.to_dict("records"))
    return dict(
        kind="verification",
        stage=task["id"],
        candidate=candidate,
        **results,
        acceptance=acceptance(fine, convergence, config["profile_search"]["success"]),
        steps=task["steps"],
    )


def _control_task(root, task, folder):
    context = _worker_context(root)
    candidate = task["candidate"]
    profile = context["config"]["profile_search"]
    duration = profile["constant_current_duration_ns"]
    currents = context["currents"][context["controls"]]
    rows = []
    for dt in profile["final_dt_ns"]:
        if time.time() >= task["deadline"] or (root / "STOP").exists() or (root / "ABORT").exists():
            raise InterruptedError("Constant-current control budget exhausted")
        params = joint_model(candidate["values"], replace(context["base"], filament_fraction=candidate["fraction"]), dt)
        params = replace(params, t_pre_s=0.0, t_end_s=duration * 1e-9)
        outputs = simulate_current_waveforms(
            np.tile(currents, (2, 1)), params, waveform_time_s=np.array([0.0, duration]) * 1e-9, audit_proximity=True
        )
        for current, out in zip(currents, outputs):
            features = settled_features(
                out["t"].astype(float) * 1e9,
                out["V_vo2"].astype(float) * 1e3,
                [duration / 2, duration],
                method="extrema",
            )
            rows.append(
                dict(
                    current_uA=current,
                    dt_ns=dt,
                    **{k: v for k, v in features.items() if not isinstance(v, list)},
                    temperature_valid=bool(
                        out["T"].min() >= params.resist_params.T_min_K
                        and out["T"].max() <= params.resist_params.T_max_K
                    ),
                    proximity_admissible=bool(
                        np.isfinite(out["minimum_proximity_slope"]) and out["minimum_proximity_slope"] >= 0
                    ),
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(folder / "constant_current.csv", index=False)
    converged = True
    for _, group in table.groupby("current_uA"):
        group = group.sort_values("dt_ns", ascending=False)
        if not bool(group.sustained.all() and group.temperature_valid.all() and group.proximity_admissible.all()):
            converged = False
        for field in ["frequency_MHz", "robust_vpp_mV"]:
            values = pd.to_numeric(group[field], errors="coerce").to_numpy(float)
            if (
                not np.isfinite(values).all()
                or np.any(values <= 0)
                or np.any(abs(np.diff(values)) / values[1:] > profile["success"]["convergence_fraction"])
            ):
                converged = False
    return dict(
        kind="constant-control",
        stage=task["id"],
        candidate=candidate,
        persistent_and_feature_converged=converged,
        records=len(rows),
    )


def run_profile_task(root, task):
    """Spawn-safe task entry point with append-only evaluation history."""
    root = Path(root)
    folder = root / "stages" / task["id"]
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "task.json").exists():
        before = _read(folder / "task.json")
        if before != task:
            raise ValueError("A stage specification changed during resume")
    else:
        _write(folder / "task.json", task)
    try:
        result = (
            _search_task if task["kind"] == "search" else _verify_task if task["kind"] == "verify" else _control_task
        )(root, task, folder)
        _write(folder / "result.json", result)
        _write(folder / "status.json", dict(state="completed", kind=task["kind"]))
        return result
    except BaseException as error:
        _write(folder / "status.json", dict(state="failed", error=str(error), traceback=traceback.format_exc()))
        raise


class _Supervisor:
    def __init__(self, bundle, state, executor):
        self.bundle = bundle
        self.root = bundle.root
        self.state = state
        self.executor = executor
        self.config = _read(self.root / "execution_config.json")
        self.profile = self.config["profile_search"]
        self.context = _context(self.config)
        self.wall_deadline = state["started_epoch"] + 3600 * self.profile["wall_hours"]
        self.profile_deadline = state["started_epoch"] + 3600 * self.profile["profile_hours"]
        self.active = {}
        self.last_live = 0.0

    def stopped(self):
        return (self.root / "STOP").exists() or (self.root / "ABORT").exists()

    def get(self, stage):
        entry = self.state["completed"].get(stage)
        return entry["result"] if entry else None

    def existing(self, task):
        path = self.root / "stages" / task["id"] / "task.json"
        if not path.exists():
            return task
        before = _read(path)
        for key in ["id", "kind", "family", "fraction"]:
            if key in task and before.get(key) != task[key]:
                raise ValueError("Stored task identity differs from planned task")
        return before

    def search(self, stage, fraction, family, seconds, seeds, *, fine=False, mean_weight=0.0):
        profile = self.profile
        task = dict(
            id=stage,
            kind="search",
            fraction=float(fraction),
            family=family,
            seconds=float(seconds),
            seed_records=seeds,
            seed=int(self.config["seed"]) + int(hashlib.sha256(stage.encode()).hexdigest()[:7], 16),
            training=self.context["targets"].tolist() if fine else self.context["training"].tolist(),
            dt_ns=profile["refinement_dt_ns"] if fine else profile["search_dt_ns"],
            deadline=self.wall_deadline if fine else self.profile_deadline,
            mean_weight=mean_weight,
        )
        if stage == "baseline":
            task.update(maximum_evaluations=min(512, int(profile["maximum_evaluations"])))
        if fine:
            task.update(
                restarts=min(2, int(profile["restarts"])),
                population=min(16, int(profile["population"])),
                maximum_generations=min(8, int(profile["maximum_generations"])),
                maximum_evaluations=min(256, int(profile["maximum_evaluations"])),
            )
        return self.existing(task)

    def verification(self, candidate, *, final=False):
        return self.existing(
            dict(
                id=("final_" if final else "profile_check_") + candidate["stage"],
                kind="verify",
                candidate=candidate,
                steps=self.profile["final_dt_ns"] if final else self.profile["verification_dt_ns"],
                save_native=final,
                deadline=self.wall_deadline,
            )
        )

    def collect(self, task, future):
        result = future.result()
        folder = self.root / "stages" / task["id"]
        hashes = {p.relative_to(folder).as_posix(): _hash(p) for p in folder.rglob("*") if p.is_file()}
        for path in folder.rglob("*"):
            if path.is_file():
                self.bundle.register_file(path.relative_to(self.root), label="Stage evidence: " + task["id"])
        self.state["completed"][task["id"]] = dict(result=result, sha256=hashes)
        _write(self.root / "state.json", self.state)
        self.bundle._save_manifest()
        count = result.get("unique_candidates")
        print(f"Completed {task['id']}" + (f": {count} unique evaluations" if count is not None else ""), flush=True)
        self.live(force=True)
        return result

    def live(self, *, force=False, phase=None):
        if phase is not None:
            self.state["phase"] = phase
        if not force and time.time() - self.last_live < 10:
            return
        self.last_live = time.time()
        active = []
        for task in self.active.values():
            path = self.root / "stages" / task["id"] / "status.json"
            active.append(dict(stage=task["id"], progress=_read(path) if path.exists() else {"state": "starting"}))
        status = dict(
            state="running",
            phase=self.state.get("phase", "starting"),
            pid=os.getpid(),
            updated_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_s=time.time() - self.state["started_epoch"],
            deadline_utc=datetime.fromtimestamp(self.wall_deadline, timezone.utc).isoformat(),
            completed_stages=len(self.state["completed"]),
            active=active,
        )
        _write(self.root / "status.json", status)
        from neuristor.plotting.filament_profile import render_profile_progress

        render_profile_progress(self.bundle, self.state, status)

    def run_tasks(self, tasks, phase):
        self.state["phase"] = phase
        queue = list(tasks)
        outputs = []
        workers = int(self.profile["workers"])
        while queue or self.active:
            while queue and len(self.active) < workers:
                task = self.existing(queue.pop(0))
                if self.get(task["id"]):
                    outputs.append(self.get(task["id"]))
                    continue
                if self.stopped():
                    raise InterruptedError("STOP or ABORT requested")
                if time.time() >= task["deadline"]:
                    self.state.setdefault("skipped_budget", []).append(task["id"])
                    continue
                future = self.executor.submit(run_profile_task, str(self.root), task)
                self.active[future] = task
                print("Started " + task["id"], flush=True)
            self.live()
            if not self.active:
                continue
            done, _ = wait(self.active, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                task = self.active.pop(future)
                outputs.append(self.collect(task, future))
        _write(self.root / "state.json", self.state)
        return outputs

    def chains(self, reverse=False):
        phase = "reverse continuations" if reverse else "forward fixed-f profiles"
        self.state["phase"] = phase
        fractions = self.profile["fractions"][1:]
        if reverse:
            fractions = list(reversed(fractions))
        if int(self.profile["workers"]) == 1:
            baseline = self.get("baseline")
            for family in self.profile["families"]:
                previous = baseline
                for fraction in fractions:
                    suffix = _fraction_key(fraction)
                    stage = ("reverse_" if reverse else "forward_") + family + "_" + suffix
                    seeds = [dict(baseline, source="new f=1 baseline"), dict(previous, source="neighbor continuation")]
                    same = self.get("forward_" + family + "_" + suffix)
                    if same:
                        seeds.insert(0, dict(same, source="same-f forward winner"))
                    task = self.search(
                        stage,
                        fraction,
                        family,
                        self.profile["reverse_seconds" if reverse else "forward_seconds"],
                        seeds + self.context["seeds"],
                    )
                    outputs = self.run_tasks([task], phase)
                    if outputs:
                        previous = outputs[0]
            return
        queues = {family: list(fractions) for family in self.profile["families"]}
        baseline = self.get("baseline")
        previous = {family: baseline for family in queues}

        def next_task(family):
            while queues[family]:
                fraction = queues[family].pop(0)
                suffix = _fraction_key(fraction)
                stage = ("reverse_" if reverse else "forward_") + family + "_" + suffix
                seeds = [dict(baseline, source="new f=1 baseline")]
                same = self.get("forward_" + family + "_" + suffix)
                if same:
                    seeds.append(dict(same, source="same-f forward winner"))
                if previous[family]:
                    seeds.append(dict(previous[family], source="neighbor continuation"))
                seeds += self.context["seeds"]
                task = self.search(
                    stage, fraction, family, self.profile["reverse_seconds" if reverse else "forward_seconds"], seeds
                )
                if self.get(stage):
                    previous[family] = self.get(stage)
                    continue
                if time.time() >= self.profile_deadline:
                    self.state.setdefault("skipped_budget", []).append(stage)
                    continue
                return task
            return None

        for family in queues:
            task = next_task(family)
            if task:
                self.active[self.executor.submit(run_profile_task, str(self.root), task)] = task
        while self.active:
            if self.stopped():
                raise InterruptedError("STOP or ABORT requested")
            self.live()
            done, _ = wait(self.active, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                task = self.active.pop(future)
                family = task["family"]
                previous[family] = self.collect(task, future)
                follow = next_task(family)
                if follow:
                    self.active[self.executor.submit(run_profile_task, str(self.root), follow)] = follow
        _write(self.root / "state.json", self.state)

    def profile_winners(self):
        groups = {}
        for stage, entry in self.state["completed"].items():
            result = entry["result"]
            if result["kind"] != "search" or stage.startswith("refine_"):
                continue
            key = (result["family"], result["fraction"])
            if key not in groups or result["winner"]["score"] < groups[key]["winner"]["score"]:
                groups[key] = result
        return list(groups.values())

    def adaptive(self):
        path = self.root / "adaptive_plan.json"
        if path.exists():
            tasks = _read(path)
        else:
            tasks = []
            baseline = self.get("baseline")
            for family in self.profile["families"]:
                options = [r for r in self.profile_winners() if r["family"] == family and r["fraction"] < 1]
                if not options:
                    continue
                best = min(options, key=lambda r: r["winner"]["score"])
                values = list(map(float, self.profile["fractions"]))
                index = values.index(best["fraction"])
                for neighbor in [index - 1, index + 1]:
                    if not 0 <= neighbor < len(values):
                        continue
                    fraction = float(np.sqrt(values[neighbor] * best["fraction"]))
                    tasks.append(
                        self.search(
                            "adaptive_" + family + "_" + _fraction_key(fraction),
                            fraction,
                            family,
                            self.profile["reverse_seconds"],
                            [
                                dict(best, source="adjacent fixed-f winner"),
                                dict(baseline, source="f=1 baseline"),
                                *self.context["seeds"],
                            ],
                        )
                    )
            _write(path, tasks)
            self.bundle.register_file("adaptive_plan.json", label="Training-only adaptive midpoint plan")
        self.run_tasks(tasks, "adaptive fraction midpoints")

    def refinements(self, verifications):
        path = self.root / "refinement_plan.json"
        if path.exists():
            return _read(path)
        baseline = self.get("baseline")
        choices = [("baseline", baseline)]
        for family in self.profile["families"]:
            options = [
                r for r in verifications if r["candidate"]["family"] == family and r["candidate"]["fraction"] < 1
            ]
            if options:
                best = min(options, key=diagnostic_rank)["candidate"]
                choices.append((family, best))
        tasks = []
        for family, candidate in choices:
            for mean_weight in [0.0, 1.0]:
                stage = "refine_" + family + ("_mean" if mean_weight else "_primary")
                tasks.append(
                    self.search(
                        stage,
                        candidate["fraction"],
                        candidate["family"],
                        self.profile["refinement_seconds"],
                        [
                            dict(candidate, source="profile winner"),
                            dict(baseline, source="new f=1 baseline"),
                            *self.context["seeds"],
                        ],
                        fine=True,
                        mean_weight=mean_weight,
                    )
                )
        _write(path, tasks)
        self.bundle.register_file("refinement_plan.json", label="Refinement selection and paired objectives")
        return tasks


def _fraction_key(value):
    return f"{float(value):.9g}".replace(".", "p")


def _verify_frozen(root):
    hashes = _read(root / "frozen_hashes.json")
    for name, digest in hashes.items():
        path = root / name
        if not path.resolve().is_relative_to(root) or not path.is_file() or _hash(path) != digest:
            raise ValueError("Frozen source/input changed: " + name)
    package = Path(__file__).resolve().parents[1]
    for name, digest in _read(root / "source_hashes.json").items():
        relative = Path(name).relative_to("source/neuristor")
        if _hash(package / relative) != digest:
            raise ValueError("Runtime code differs from frozen source; resume with this bundle/source on PYTHONPATH")
    if _environment() != _read(root / "environment.json"):
        raise ValueError("Numerical environment changed; refusing to mix optimizer evidence")


def _verify_completed(root, state):
    for stage, entry in state["completed"].items():
        folder = root / "stages" / stage
        for name, digest in entry["sha256"].items():
            if not (folder / name).is_file() or _hash(folder / name) != digest:
                raise ValueError("Completed stage changed: " + stage + "/" + name)


def run_filament_profile(
    config=None, *, resume=None, output_root=None, prepare=False, command="neuristor analyze filament-profile"
):
    """Prepare or execute an overnight job under one exclusive supervisor lock."""
    if resume is None:
        bundle = prepare_filament_profile(config, output_root=output_root, command=command)
    else:
        root = Path(resume).expanduser().resolve()
        manifest = _read(root / "run.json")
        if manifest.get("model") != "overnight-filament-profile" or manifest.get("status") == "completed":
            raise ValueError("Resume requires an unfinished overnight filament profile")
        bundle = RunBundle(root, manifest, list(manifest["artifacts"]))
    print("PROFILE_DIRECTORY=" + str(bundle.root), flush=True)
    if prepare:
        return bundle
    root = bundle.root.resolve()
    with _exclusive(root):
        _verify_frozen(root)
        state = _read(root / "state.json")
        _verify_completed(root, state)
        if (root / "STOP").exists():
            raise InterruptedError("STOP exists; remove it only when explicitly resuming the stopped experiment")
        if (root / "ABORT").exists():
            (root / "ABORT").unlink()
        if state["started_epoch"] is None:
            state["started_epoch"] = time.time()
        state["attempts"] += 1
        _write(root / "state.json", state)
        _write(
            root / "process.json",
            dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(), python=sys.executable),
        )
        bundle.add_artifact("process.json", label="Supervisor process identity")
        bundle.manifest["status"] = "running"
        bundle.manifest.pop("error", None)
        bundle._save_manifest()
        profile = _read(root / "execution_config.json")["profile_search"]
        executor = ProcessPoolExecutor(
            max_workers=int(profile["workers"]), mp_context=multiprocessing.get_context("spawn")
        )
        supervisor = _Supervisor(bundle, state, executor)
        try:
            baseline = supervisor.search(
                "baseline", 1.0, "inherited", profile["baseline_seconds"], supervisor.context["seeds"]
            )
            supervisor.run_tasks([baseline], "f=1 reference search")
            if not supervisor.get("baseline"):
                raise RuntimeError("No baseline completed inside the search budget")
            supervisor.chains()
            supervisor.chains(reverse=True)
            supervisor.adaptive()
            verified = supervisor.run_tasks(
                [supervisor.verification(r) for r in supervisor.profile_winners()],
                "three-timestep profile verification",
            )
            refinements = supervisor.run_tasks(
                supervisor.refinements(verified), "paired primary/mean-aware fine-step refinement"
            )
            final = supervisor.run_tasks(
                [supervisor.verification(r, final=True) for r in refinements],
                "final two smaller timesteps and native histories",
            )
            controls = supervisor.run_tasks(
                [
                    supervisor.existing(
                        dict(
                            id="control_" + r["candidate"]["stage"],
                            kind="control",
                            candidate=r["candidate"],
                            deadline=supervisor.wall_deadline,
                        )
                    )
                    for r in final
                ],
                "long constant-current controls",
            )
            control_results = {r["candidate"]["stage"]: r for r in controls}
            successes = [
                r["candidate"]["stage"]
                for r in final
                if r["acceptance"]["success"]
                and control_results.get(r["candidate"]["stage"], {}).get("persistent_and_feature_converged", False)
            ]
            _verify_completed(root, state)
            total_evals = sum(e["result"].get("unique_candidates", 0) for e in state["completed"].values())
            metrics = dict(
                state="completed",
                successful_candidates=successes,
                calibration_adopted=False,
                elapsed_s=time.time() - state["started_epoch"],
                wall_hours=profile["wall_hours"],
                search_stages=sum(e["result"]["kind"] == "search" for e in state["completed"].values()),
                total_unique_candidate_evaluations=total_evals,
                completed_profile_points=len(supervisor.profile_winners()),
                final_candidates_verified=len(final),
                constant_control_cases=len(controls),
                coverage_complete=not bool(state.get("skipped_budget")),
                skipped_budget=sorted(set(state.get("skipped_budget", []))),
                interrupted_optimizer_trajectory_can_differ=state["attempts"] > 1,
                held_out_scope="Optimizer-excluded initially, historically seen; final refinements use all 11 targets",
            )
            bundle.write_json("metrics.json", metrics, label="Coverage, budget and prespecified success criteria")
            from neuristor.plotting.filament_profile import render_profile_progress, profile_report

            status = dict(metrics, phase="finished", updated_utc=datetime.now(timezone.utc).isoformat())
            _write(root / "status.json", status)
            render_profile_progress(bundle, state, status)
            bundle.write_text(
                "report.md", profile_report(bundle, state, metrics), label="Methods, outcomes and scientific limits"
            )
            bundle.complete(summary=metrics)
        except BaseException as error:
            (root / "ABORT").touch()
            _write(
                root / "status.json",
                dict(state="interrupted", error=str(error), updated_utc=datetime.now(timezone.utc).isoformat()),
            )
            bundle.write_text(
                "report.md",
                "# Overnight profile interrupted\n\n"
                + str(error)
                + "\n\nCompleted stages and evaluation histories remain intact. Resume only with the frozen source and unchanged inputs.",
                label="Failure and recovery",
            )
            bundle.fail(error)
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    return bundle
