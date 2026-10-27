"""Controlled presentation-data test of a normalized filament conductance split.

The static R(T) parameters are fixed. Only the optional forward-model extension
and its one conductance weight distinguish the two bounded fitting arms.
"""

from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from neuristor.analysis.experimental_waveforms import load_converted_sweep
from neuristor.config import ConfigError, resolved_copy, source_directory, validate_config
from neuristor.fitting.joint_inference import NAMES, natural_vector, static_rmse
from neuristor.fitting.parameter_inference import prepare_inference_dataset
from neuristor.fitting.resistance_custom_analysis import load_experimental_rt
from neuristor.fitting.steady_inference import (
    evaluate_settled,
    measured_current_targets,
    relative_candidate_score,
    search_settled,
)
from neuristor.provenance import package_source_files, snapshot_package
from neuristor.runs import RunBundle


def summarize(rows):
    """Keep acceptance separate from incomplete, differently denominated errors."""
    frame = pd.DataFrame(rows)
    outside = not bool(frame.temperature_valid.all())
    score = relative_candidate_score(rows, outside)
    frequency_error = pd.to_numeric(frame.frequency_relative_error, errors="coerce")
    return dict(
        score,
        records=len(rows),
        screened_targets=int(frame.frequency_valid.sum()),
        defined_frequencies=int(frame.frequency_relative_error.notna().sum()),
        frequency_MAPE_defined=float(frequency_error.abs().mean()) if frequency_error.notna().any() else None,
        amplitude_MAPE_all=float(frame.robust_amplitude_relative_error.abs().mean()),
        mean_voltage_RMSE_mV=float(np.sqrt(np.mean((frame.predicted_mean_mV - frame.measured_mean_mV) ** 2))),
        temperature_valid=not outside,
        maximum_temperature_K=float(frame.temperature_max_K.max()),
    )


def _scalar_rows(rows):
    return [{k: v for k, v in row.items() if not isinstance(v, (dict, list, tuple))} for row in rows]


def _atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    # RunBundle handles NumPy values/nulls for final evidence; live snapshots
    # contain ordinary finite optimizer values only.
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def convergence_table(table, sampled, time_ns, settings):
    """Compare both timestep halvings without replacing missing cycles by zero."""
    result = []
    window = (time_ns >= settings["window_ns"][0]) & (time_ns <= settings["window_ns"][1])
    for candidate, group in table.groupby("candidate", sort=False):
        steps = sorted(group.dt_ns.unique(), reverse=True)
        for coarse, fine in zip(steps[:-1], steps[1:]):
            a = group[group.dt_ns == coarse].set_index("source_label_mV")
            b = group[group.dt_ns == fine].set_index("source_label_mV")
            for index, label in enumerate(a.index):
                old, new = a.loc[label], b.loc[label]
                changes = {}
                for name, field in [("frequency", "predicted_frequency_MHz"), ("amplitude", "predicted_robust_vpp_mV")]:
                    x, y = old[field], new[field]
                    changes[name + "_relative_change"] = (
                        abs(x - y) / abs(y) if pd.notna(x) and pd.notna(y) and x > 0 and y > 0 else None
                    )
                rms = float(
                    np.sqrt(
                        np.mean(
                            (sampled[(candidate, coarse)][window, index] - sampled[(candidate, fine)][window, index])
                            ** 2
                        )
                    )
                )
                stable = all(
                    old[k] == new[k]
                    for k in ["frequency_valid", "sustained", "temperature_valid", "proximity_admissible"]
                )
                feature_ok = bool(
                    stable
                    and new.frequency_valid
                    and new.temperature_valid
                    and new.proximity_admissible
                    and all(v is not None and v <= settings["convergence_fraction"] for v in changes.values())
                )
                result.append(
                    dict(
                        candidate=candidate,
                        source_label_mV=label,
                        target=bool(new.target),
                        coarse_dt_ns=coarse,
                        fine_dt_ns=fine,
                        **changes,
                        waveform_RMS_difference_mV=rms,
                        classification_stable=stable,
                        features_converged=feature_ok,
                        waveform_converged=rms <= settings["waveform_rms_tolerance_mV"],
                    )
                )
    return pd.DataFrame(result)


def run_filament_study(config, *, output_root=None, command="neuristor analyze filament-study"):
    """Run sensitivity scans, matched bounded fits, and three-step all-current checks."""
    from neuristor.workflows import _current_params_from_config
    from neuristor.plotting.filament_study import render_filament_study

    validate_config(config)
    settings = copy.deepcopy(config["steady"])
    study = config["filament_study"]
    if settings["restarts"] != settings["fixed_resistance_restarts"] or settings.get("fixed_parameters") != ["T0_K"]:
        raise ConfigError("The filament comparison fixes all six R(T) constants and T0 in every restart")
    if (
        settings.get("loss_function") != "relative_squared"
        or settings.get("amplitude_weight") != 1
        or settings.get("frequency_weight") != 1
    ):
        raise ConfigError("Use the presentation frequency-plus-amplitude objective with unit weights")
    steps = list(map(float, settings["verification_dt_ns"]))
    if len(steps) < 3 or any(b >= a or b <= 0 for a, b in zip(steps[:-1], steps[1:])):
        raise ConfigError("At least three strictly decreasing positive verification timesteps are required")
    fractions = list(map(float, study["fractions"]))
    if (
        not fractions
        or not np.isfinite(fractions).all()
        or min(fractions) <= 0
        or max(fractions) > 1
        or 1.0 not in fractions
    ):
        raise ConfigError("The fraction scan must include 1 and lie in (0,1]")

    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (source_directory(config) / path).resolve()

    base, *_ = _current_params_from_config(config)
    data = resolve(settings["data_directory"])
    rt_path = resolve(settings["resistance_data"])
    preset_path = resolve(config["resistance"]["preset"])
    seed_path = resolve(settings["seed_table"])
    raw, _ = load_converted_sweep(data)
    labels = sorted(raw.nominal_drive_mV.unique())
    provisional = prepare_inference_dataset(raw, holdout_drives_mV=[labels[0]])
    selected = measured_current_targets(provisional, settings["target_current_range_uA"])
    targets = provisional.nominal_drives_mV[selected].tolist()
    dataset = prepare_inference_dataset(raw, holdout_drives_mV=[label for label in labels if label not in targets])
    rt = load_experimental_rt(rt_path)
    frozen = {
        key: dict(values=study[key]["values"], filament_fraction=1.0)
        for key in ["presentation_frequency", "presentation_combined"]
    }
    if any(static_rmse(item["values"], rt) > settings["maximum_static_rmse_log10"] for item in frozen.values()):
        raise ConfigError("Declared presentation reference fails the static-fit screen")
    seed_table = pd.read_csv(seed_path).set_index("candidate")
    seeds = [np.array(frozen[key]["values"]) for key in ["presentation_combined", "presentation_frequency"]]
    seeds += [seed_table.loc[name, list(NAMES)].to_numpy(float) for name in settings["seed_candidates"]]
    bundle = RunBundle.create(
        name=config["name"],
        model="experimental-filament-surroundings",
        kind="analysis",
        config=resolved_copy(config),
        output_root=output_root or config["output"]["root"],
        command=command,
    )
    print(f"RUN_DIRECTORY={bundle.root}", flush=True)
    started = time.monotonic()
    bundle.write_text(
        "report.md",
        "# Experimental filament study (running)\n\nNo new calibration has been adopted.",
        label="Scope and live status",
    )
    status_path = bundle.add_artifact("status.json", label="Durable stage progress")
    try:
        sources = [
            Path(config["_source"]),
            rt_path,
            preset_path,
            seed_path,
            *sorted(data.glob("*_converted.csv")),
            *package_source_files(),
        ]
        bundle.write_json(
            "source_input_hashes.json",
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            label="Source and original-input hashes",
        )
        bundle.write_json("source_hashes.json", snapshot_package(bundle), label="Complete frozen implementation")
        for source, target in [
            (rt_path, "resistance_data.tsv"),
            (preset_path, "resistance_preset.json"),
            (seed_path, "seed_vectors.csv"),
            (Path(config["_source"]), "recipe.toml"),
        ]:
            shutil.copy2(source, bundle.add_artifact("inputs/" + target, label="Original declared input"))
        for source in sorted(data.glob("*_converted.csv")):
            shutil.copy2(
                source, bundle.add_artifact("inputs/current/" + source.name, label="Original measured channels")
            )
        bundle.write_json(
            "baseline_parameters.json",
            dict(zip(NAMES, natural_vector(base).tolist())),
            label="Fixed static law and initial dynamic values",
        )
        candidates = dict(frozen)
        profile_rows = []
        for name, record in frozen.items():
            choices = []
            for fraction in fractions:
                _atomic(
                    status_path, dict(stage="frozen-reference sensitivity", reference=name, filament_fraction=fraction)
                )
                rows, _, _ = evaluate_settled(
                    dataset,
                    record["values"],
                    replace(base, filament_fraction=fraction),
                    dataset.train_indices,
                    settings,
                    settings["search_dt_ns"],
                )
                metrics = summarize(rows)
                profile_rows.append(dict(reference=name, filament_fraction=fraction, **metrics))
                choices.append((metrics["score"], fraction))
                pd.DataFrame(profile_rows).to_csv(
                    bundle.add_artifact(
                        "fraction_sensitivity.csv", label="Conditional scans with presentation dynamics fixed"
                    ),
                    index=False,
                )
                print(
                    f"Sensitivity {name}: f={fraction:g}, passing {metrics['screened_targets']}/{metrics['records']}, Vpp MAPE {metrics['amplitude_MAPE_all']:.3g}",
                    flush=True,
                )
            best_fraction = min(choices)[1]
            candidates[name + "_fraction_scan"] = dict(values=record["values"], filament_fraction=best_fraction)
        results = {}
        for model in ["baseline", "filament"]:
            local = copy.deepcopy(settings)
            local.update(
                fit_filament_fraction=model == "filament",
                filament_fraction_bounds=study["fraction_bounds"],
                initial_filament_fractions=study["initial_fractions"],
            )
            history_path = bundle.add_artifact(
                "search_" + model + ".jsonl", label="All unique " + model + " optimizer evaluations"
            )
            checkpoint_path = bundle.add_artifact(
                "checkpoint_" + model + ".json", label="Recoverable search candidates"
            )
            bundle._save_manifest()
            records = []
            last_print = [0.0]
            with history_path.open("w") as stream:

                def checkpoint(record, snapshot):
                    records.append(record)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    _atomic(checkpoint_path, snapshot)
                    _atomic(
                        status_path,
                        dict(
                            stage="search",
                            model=model,
                            unique_candidates=snapshot["unique_candidates"],
                            elapsed_s=time.monotonic() - started,
                        ),
                    )
                    if time.monotonic() - last_print[0] >= 20:
                        print(
                            f"{model}: {snapshot['unique_candidates']} candidates; best accepted={snapshot['best'] is not None}",
                            flush=True,
                        )
                        last_print[0] = time.monotonic()

                result = search_settled(
                    dataset,
                    rt,
                    base,
                    local,
                    seeds,
                    pd.DataFrame(),
                    checkpoint,
                    stop_requested=lambda: (bundle.root / "STOP").exists(),
                )
            results[model] = result
            bundle.write_json("search_result_" + model + ".json", result, label="Bounded " + model + " search result")
            usable = [r for r in records if r.get("simulated")]
            if not usable:
                raise ValueError("No simulated candidate available in " + model + " arm")
            winner = result["best"] or result["diagnostic_best"] or min(usable, key=lambda r: r["score"])
            candidates[model + "_fitted"] = dict(
                values=winner["values"], filament_fraction=winner.get("filament_fraction", 1.0)
            )
            if (bundle.root / "STOP").exists():
                raise InterruptedError("STOP requested; completed searches and checkpoints are preserved")
        # A conditional sensitivity scan is not a profile likelihood or a confidence interval.
        record = candidates["filament_fitted"]
        for fraction in sorted(set(fractions + [record["filament_fraction"]])):
            rows, _, _ = evaluate_settled(
                dataset,
                record["values"],
                replace(base, filament_fraction=fraction),
                dataset.train_indices,
                settings,
                settings["search_dt_ns"],
            )
            profile_rows.append(
                dict(reference="filament_fitted_dynamics_fixed", filament_fraction=fraction, **summarize(rows))
            )
        pd.DataFrame(profile_rows).to_csv(
            bundle.add_artifact(
                "fraction_sensitivity.csv", label="Conditional fraction sensitivity; other parameters fixed"
            ),
            index=False,
        )
        bundle.write_json("candidates.json", candidates, label="Declared references and selected diagnostic candidates")
        verification, sampled, traces = [], {}, []
        keep = dataset.time_ns <= settings["window_ns"][1]
        time_ns = dataset.time_ns[keep]
        all_indices = np.arange(len(labels))
        summaries = []
        replay_cache = {}
        for name, record in candidates.items():
            for dt in steps:
                print(f"Verifying {name}: {dt:g} ns, all {len(labels)} currents", flush=True)
                _atomic(status_path, dict(stage="verification", candidate=name, dt_ns=dt))

                def native_callback(index, out, params):
                    if dt != min(steps) or name not in ["baseline_fitted", "filament_fitted"]:
                        return
                    label = dataset.nominal_drives_mV[index]
                    voltage = out["V_vo2"].astype(float)
                    hot = out.get("R_reference_hot", out["R"]).astype(float)
                    fraction = params.filament_fraction
                    np.savez_compressed(
                        bundle.add_artifact(
                            f"native/{name}/current_{label:g}.npz", label="Full native SI branch histories"
                        ),
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
                        filament_fraction=fraction,
                    )

                key = (tuple(record["values"]), record["filament_fraction"], dt)
                # Repeated reference/scan winners are numerically identical;
                # fitted candidates still execute to retain their native histories.
                if key in replay_cache and name not in ["baseline_fitted", "filament_fitted"]:
                    rows, volts = replay_cache[key]
                else:
                    rows, volts, _ = evaluate_settled(
                        dataset,
                        record["values"],
                        replace(base, filament_fraction=record["filament_fraction"]),
                        all_indices,
                        settings,
                        dt,
                        native_callback=native_callback,
                    )
                    replay_cache[key] = rows, volts
                frame = pd.DataFrame(_scalar_rows(rows))
                frame["target"] = frame.source_label_mV.isin(targets)
                frame = frame.assign(candidate=name, dt_ns=dt, filament_fraction=record["filament_fraction"])
                verification.extend(frame.to_dict("records"))
                sampled[(name, dt)] = volts
                target_rows = [row for row in rows if row["source_label_mV"] in targets]
                summaries.append(
                    dict(
                        candidate=name,
                        dt_ns=dt,
                        filament_fraction=record["filament_fraction"],
                        **summarize(target_rows),
                    )
                )
                if dt == min(steps):
                    for index, label in enumerate(dataset.nominal_drives_mV):
                        traces.append(
                            pd.DataFrame(
                                dict(
                                    candidate=name,
                                    source_label_mV=label,
                                    current_uA=frame.measured_current_A.iloc[index] * 1e6,
                                    time_ns=time_ns,
                                    measured_voltage_mV=dataset.voltage_mV[keep, index],
                                    simulated_voltage_mV=volts[:, index],
                                    input_current_uA=dataset.current_uA[keep, index],
                                )
                            )
                        )
                pd.DataFrame.from_records(verification).to_csv(
                    bundle.add_artifact("verification.csv", label="All currents, both classifications, all timesteps"),
                    index=False,
                )
        verification = pd.DataFrame.from_records(verification)
        pd.DataFrame(summaries).to_csv(
            bundle.add_artifact("summary.csv", label="Target-only acceptance and errors with denominators"), index=False
        )
        pd.concat(traces, ignore_index=True).to_csv(
            bundle.add_artifact("traces.csv", label="Acquisition-grid comparison; full endpoints in native/"),
            index=False,
        )
        convergence = convergence_table(verification, sampled, time_ns, dict(settings, **study))
        convergence.to_csv(
            bundle.add_artifact("convergence.csv", label="Two timestep halvings, classification and waveform checks"),
            index=False,
        )
        render_filament_study(bundle)
        fine = pd.DataFrame(summaries)
        fine = fine[fine.dt_ns == min(steps)].set_index("candidate")
        metrics = dict(
            elapsed_s=time.monotonic() - started,
            original_measurements=len(labels),
            target_records=len(targets),
            control_records=len(labels) - len(targets),
            reference_contacts="same electrodes; uniform stage heating (user confirmed)",
            calibration_adopted=False,
            static_fit_unchanged=True,
            fit_counts={name: r["unique_candidates"] for name, r in results.items()},
            fitted_filament_fraction=candidates["filament_fitted"]["filament_fraction"],
            finest_target_results={name: fine.loc[name].to_dict() for name in ["baseline_fitted", "filament_fitted"]},
            target_feature_convergence={
                name: bool(part.features_converged.all())
                for name, part in convergence[convergence.target].groupby("candidate")
            },
            target_waveform_convergence={
                name: bool(part.waveform_converged.all())
                for name, part in convergence[convergence.target].groupby("candidate")
            },
        )
        bundle.write_json("metrics.json", metrics, label="Comparison, acceptance and numerical limits")
        bundle.write_text(
            "report.md",
            _report(json.loads((bundle.root / "metrics.json").read_text())),
            label="Physical hypothesis, matched protocol, results and limits",
        )
        _atomic(status_path, dict(stage="completed", elapsed_s=time.monotonic() - started))
        bundle.complete(summary=metrics)
    except BaseException as error:
        _atomic(status_path, dict(stage="interrupted", error=str(error)))
        bundle.fail(error)
        raise
    return bundle


def _report(metrics):
    return (
        """# Experimental filament and cold surroundings

The user confirmed that the fitted R(T) and oscillation recordings use the same
electrodes; the R(T) loop used uniform stage heating. The proposed hypothesis is
localized filament heating rather than a new intrinsic material resistivity.

G_eff = f/R_fit(T_f,H_f) + (1-f)/R_fit(T0,H0).
C dV/dt = I_in - V G_eff.
Cth_f dT_f/dt = f V^2/R_fit(T_f,H_f) - Se (T_f-T0).

The cold branch uses an independent initial hysteresis state held at ambient.
Its Joule heat goes to an ideal external reservoir, not to the hot node. This is
a deliberately limited thermal assumption, not an energy-free shunt. P is total
electrical dissipation; native files separately record both currents and powers.
Equal temperatures and histories recover the measured uniform R(T); f=1 recovers
the original equations and original floating-point arithmetic. f is a normalized
conductance share, not a measured filament width. Cth and Se describe the heated
region, whose geometry is not independently identified here.

Both optimization arms use the same measured 11 target currents, full prehistory,
fixed six static-fit constants and T0, dynamic bounds, seed vectors, DE/Powell
procedure, wall-clock allowance and evaluation cap. The filament arm adds f in
the declared logarithmic interval. Actual counts can differ because of time
limits and caching. The unchanged presentation objective is mean squared
relative frequency plus Vpp error; mean voltage is an independent diagnostic.
Ineligible candidates have no accepted scientific objective. Missing frequencies
remain null; amplitude MAPE on all targets retains the original zero-feature
convention for missing cycles. Frequency MAPE always states its denominator.

The frozen-reference scans are additional sensitivity calculations, outside the
matched optimizer caps. Their selected fractions are reported separately. The
post-fit fraction scan fixes all other parameters: it is NOT a profile likelihood
or confidence interval. All 22 currents are checked at three timesteps. The other
11 currents are previously inspected controls, not blind validation. Neither
lower scalar error nor a completed bundle establishes an accepted specimen fit.
The measured channel-to-device voltage/current mapping remains a separate issue.

Inspect summary.csv, verification.csv, and both timestep halvings in convergence.csv.
Waveform RMS differences use the same acquisition grid and late loss window;
feature convergence and waveform convergence are reported separately. The shared
temperature and proximity screens are not relaxed. No published bundle or
presentation route was modified.

Physical motivation: https://doi.org/10.1002/aelm.70604 (uniform heating versus
filamentary switching); this experiment is our proposed reduced model.

## Completed comparison

```json
"""
        + json.dumps(metrics, indent=2, default=str)
        + "\n```\n"
    )
