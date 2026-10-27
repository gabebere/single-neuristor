"""Checkpointed reconstruction of presentation dependencies from measured inputs.

All scientific work delegates to public workflows and the authoritative solvers.
Existing bundles and Release assets are never inputs. Completed stages remain
immutable; an interrupted run can resume only with the same source and inputs.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import html
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import quote

import numpy as np
import pandas as pd

from neuristor.config import ConfigError, load_toml, resolved_copy, source_directory, validate_config
from neuristor.provenance import package_source_files, snapshot_package
from neuristor.runs import RunBundle, find_project_root

STAGES = (
    "specimen_fit",
    "frequency_search",
    "amplitude_search",
    "comparison",
    "loss_gifs",
    "map_grid",
    "map_gifs",
    "scaling",
    "mica_fit",
    "mica_broad",
    "mica_lower",
    "mica_transition",
    "mica_onset",
    "mica_expanded",
    "mica_dense",
    "mica_checks",
    "mica_review",
    "alternating",
)
MICA_NAMES = dict(
    mica_broad="Broad",
    mica_lower="Lower heating",
    mica_transition="Transition",
    mica_onset="Onset",
    mica_expanded="Expanded",
    mica_dense="Dense onset",
)
ROUTES = {
    "/loss-gifs": ("loss_gifs", "START_HERE.html"),
    "/parameter-map": ("map_gifs", "parameter_maps.html"),
    "/mica-map": ("mica_review", "parameter_maps.html"),
    "/alternating-gifs": ("alternating", "START_HERE.html"),
}


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _resolve(config, value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else source_directory(config) / p).resolve()


def plan_presentation_reproduction(config):
    """Validate the complete source/input closure without reading saved results."""
    validate_config(config)
    if config["kind"] != "reproduction":
        raise ConfigError("Use a presentation reproduction recipe")
    root = find_project_root(source_directory(config))
    settings = config["reproduction"]
    if int(settings["static_fit"].get("bootstrap_samples", 0)) < 1:
        raise ConfigError("Presentation fitting requires bootstrap_samples >= 1")
    required = {"specimen_data", "mica_data", "waveforms", "search_seeds", "alternating_parameters", "device_photo"}
    if set(settings["inputs"]) != required:
        raise ConfigError(f"Presentation inputs must be {sorted(required)}")
    recipes = settings["recipes"]
    expected = {"frequency_search", "amplitude_search", "map_grid", "scaling", "alternating", *MICA_NAMES}
    if set(recipes) != expected:
        raise ConfigError(f"Presentation recipes must be {sorted(expected)}")
    files = [Path(config["_source"]).resolve()]
    inputs = {}
    for name, value in settings["inputs"].items():
        path = _resolve(config, value)
        if (
            not path.exists()
            or not path.is_relative_to(root)
            or "runs" in path.relative_to(root).parts
            or "private" in path.relative_to(root).parts
        ):
            raise ConfigError(f"{name}: expected a versioned source input inside the repository, outside runs/private")
        inputs[name] = str(path)
        if name == "waveforms":
            measured = sorted(path.glob("*_converted.csv"))
            if not measured:
                raise ConfigError("No measured *_converted.csv files found")
            files.extend(measured)
        else:
            if not path.is_file():
                raise ConfigError(f"{name} must be a file")
            files.append(path)
    plans = {}
    for key, value in recipes.items():
        path = _resolve(config, value)
        child = load_toml(path)
        files.append(path)
        if "maps" in child:
            count = 1
            for axis in child["maps"]["axes"].values():
                if "values" in axis:
                    values = axis["values"]
                else:
                    base = np.geomspace(axis["low"], axis["high"], axis["count"])
                    values = np.unique(np.r_[base, axis.get("extra_values", [])])
                count *= len(values)
            plans[key] = dict(grid_points=count, recipe=str(path.relative_to(root)))
    # The declaration is complete: additional stage inputs are generated upstream.
    files.extend(package_source_files())
    files.extend(sorted((root / "docs/presentation").glob("*.tex")))
    files.append(root / "pyproject.toml")
    fingerprints = {p.relative_to(root).as_posix(): _digest(p) for p in sorted(set(files))}
    environment = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "pandas", "torch", "matplotlib", "plotly", "imageio", "pillow")
    }
    environment["python"] = sys.version
    signature = hashlib.sha256(
        json.dumps(
            dict(config=resolved_copy(config), inputs=fingerprints, environment=environment), sort_keys=True
        ).encode()
    ).hexdigest()
    return dict(
        profile=settings.get("profile", "full"),
        stages=list(STAGES),
        inputs=inputs,
        recipes=plans,
        source_input_hashes=fingerprints,
        environment=environment,
        fingerprint=signature,
        time_allowance=settings["time_allowance"],
        caveat="Original laboratory measurements and declared parameter/seed settings are inputs; all derived results are recomputed.",
    )


def _child_config(key, config, parent, stages):
    """Bind every former saved-result reference to a newly computed dependency."""
    settings = config["reproduction"]
    child = load_toml(_resolve(config, settings["recipes"][key]))
    inputs = parent.root / "inputs"
    if key == "alternating":
        child["resistance"]["preset"] = str(inputs / "alternating_parameters.json")
        # No source bundle is needed: original measured channels supply inputs.
        child["animation_study"].pop("source_bundle", None)
        child["animation_study"]["workers"] = int(settings["workers"])
    elif key.startswith("mica_"):
        if "study" in child:
            child["study"]["resistance_data"] = str(inputs / "mica_data.tsv")
        else:
            child["resistance"]["preset"] = str(stages["mica_fit"] / "resistance_preset.json")
    else:
        child["resistance"]["preset"] = str(stages["specimen_fit"] / "resistance_preset.json")
    if "steady" in child:
        child["steady"].update(
            data_directory=str(inputs / "waveforms"),
            resistance_data=str(inputs / "specimen_data.tsv"),
            static_bootstrap=str(stages["specimen_fit"] / "parameter_bootstrap.csv"),
            seed_table=str(inputs / "search_seeds.csv"),
        )
    if "animations" in child:
        child["animations"].pop("source", None)
        child["animations"]["workers"] = int(settings["workers"])
    if settings.get("profile") == "smoke":
        # A separate developer profile, never activated for the actual slides.
        if "steady" in child:
            child["steady"].update(
                maximum_evaluations=8,
                restarts=1,
                fixed_resistance_restarts=1,
                population=5,
                maximum_generations=1,
                local_evaluations=2,
                search_seconds=120.0,
                total_seconds=1800.0,
            )
        if "maps" in child:
            for name, value in child["maps"]["fixed_values"].items():
                child["maps"]["axes"][name] = {"values": [value]}
            if key == "map_grid":
                point = dict(C_pF=3.0, C_th_pJ_per_K=0.072, S_e_mW_per_K=0.006)
                child["maps"]["fixed_values"] = point
                child["maps"]["axes"] = {k: {"values": [v]} for k, v in point.items()}
        if "animations" in child:
            child["animations"].update(frames=2)
        if "animation_study" in child:
            child["animation_study"].update(frames=2)
        if "trend_validation" in child:
            trend = child["trend_validation"]
            trend.update(
                anchors=trend["anchors"][:1],
                scenarios=trend["scenarios"][:1],
                gamma_values=[trend["gamma"]],
                dt_ns=[0.05, 0.025, 0.0125],
                maximum_timestep_refinements=0,
                initial_duration_ns=600.0,
                maximum_duration_ns=1200.0,
            )
    return child


def _mica_checks(config, parent, stages, output):
    """Repeat the declared 36-point pilot and 24 startup integrations."""
    from neuristor.workflows import _current_params_from_config
    from neuristor.current_drive_sim import simulate_current_parameter_grid
    from neuristor.analysis.physical_maps import screen_trajectory

    settings = dict(config["reproduction"]["mica_checks"])
    cfg = _child_config("mica_dense", config, parent, stages)
    base, *_ = _current_params_from_config(cfg)
    axes = [settings["pilot_C_th_pJ_per_K"], settings["pilot_S_e_mW_per_K"], settings["pilot_C_pF"]]
    pilot = list(itertools.product(*axes))
    points = settings["startup_points"]
    if config["reproduction"].get("profile") == "smoke":
        pilot, points = pilot[:1], points[:1]
        settings.update(pilot_dt_ns=0.01, pilot_duration_ns=150.0, startup_dt_ns=[0.01, 0.005])
    bundle = RunBundle.create(
        name="Mica independent pilot and startup checks",
        model="current",
        kind="analysis",
        config=settings,
        output_root=output,
        command="neuristor reproduce-presentation",
    )
    try:
        rows = []
        for label, coordinates, steps, duration in [
            ("pilot", pilot, [settings["pilot_dt_ns"]], settings["pilot_duration_ns"]),
            ("startup", points, settings["startup_dt_ns"], settings["startup_duration_ns"]),
        ]:
            a = np.asarray(coordinates)
            for dt in steps:
                params = replace(base, dt_s=dt * 1e-9, t_end_s=duration * 1e-9)
                outputs = simulate_current_parameter_grid(
                    500.0,
                    params,
                    C_th_J_per_K=a[:, 0] * 1e-12,
                    S_e_W_per_K=a[:, 1] * 1e-3,
                    C_F=a[:, 2] * 1e-12,
                )
                for point, out in zip(coordinates, outputs):
                    row = dict(
                        check=label,
                        C_th_pJ_per_K=point[0],
                        S_e_mW_per_K=point[1],
                        C_pF=point[2],
                        dt_ns=dt,
                        duration_ns=duration,
                        temperature_max_K=float(out["T"].max()),
                        temperature_min_K=float(out["T"].min()),
                        temperature_valid=bool(
                            out["T"].max() <= base.resist_params.T_max_K
                            and out["T"].min() >= base.resist_params.T_min_K
                        ),
                    )
                    if label == "pilot":
                        row.update(
                            screen_trajectory(
                                out, params, dict(window_ns=[duration / 2, duration], maximum_frequency_MHz=110.0)
                            )
                        )
                    rows.append(row)
        pd.DataFrame(rows).to_csv(
            bundle.add_artifact("checks.csv", label="Independent native temperature checks"), index=False
        )
        bundle.write_json("metrics.json", dict(integrations=len(rows)), label="Pilot/startup coverage")
        bundle.write_text(
            "report.md",
            "# Mica independent checks\n\nThe pilot and startup coordinates were declared before execution. "
            "All integrations use the authoritative solver; startup exclusions are not steady-state convergence or physical boundaries.\n",
            label="Scope",
        )
        bundle.complete(summary=dict(integrations=len(rows)))
    except BaseException as error:
        bundle.fail(error)
        raise
    return bundle


def _execute_stage(key, config, parent, stages, output):
    from neuristor import workflows

    settings = config["reproduction"]
    command = "neuristor reproduce-presentation"
    if key in {"specimen_fit", "mica_fit"}:
        mica = key == "mica_fit"
        return workflows.run_resistance_fit(
            parent.root / "inputs" / ("mica_data.tsv" if mica else "specimen_data.tsv"),
            name="Recreated Mica static fit" if mica else "Recreated specimen major-loop fit",
            method="major-loop",
            seed=42 if mica else int(settings["static_fit"]["seed"]),
            bootstrap_samples=0
            if mica
            else (2 if settings.get("profile") == "smoke" else int(settings["static_fit"]["bootstrap_samples"])),
            output_root=output,
            command=command,
        )
    if key in {"frequency_search", "amplitude_search"}:
        return workflows.run_steady_search(
            _child_config(key, config, parent, stages), output_root=output, command=command
        )
    if key == "comparison":
        return workflows.run_steady_comparison(
            stages["frequency_search"], stages["amplitude_search"], output_root=output, command=command
        )
    if key == "loss_gifs":
        return workflows.run_loss_scopes(
            _child_config("frequency_search", config, parent, stages),
            stages["frequency_search"],
            stages["amplitude_search"],
            output_root=output,
            frames=2 if settings.get("profile") == "smoke" else 96,
            command=command,
        )
    if key == "map_grid":
        return workflows.run_physical_parameter_maps(
            _child_config(key, config, parent, stages), output_root=output, command=command
        )
    if key == "map_gifs":
        cfg = _child_config("map_grid", config, parent, stages)
        cfg["animations"]["source"] = str(stages["map_grid"])
        return workflows.run_map_animations(cfg, output_root=output, command=command, package=False)
    if key == "scaling":
        return workflows.run_trend_validation(
            _child_config(key, config, parent, stages), output_root=output, command=command
        )
    if key in MICA_NAMES:
        cfg = _child_config(key, config, parent, stages)
        if "study" in cfg:
            return workflows.run_sample_study(cfg, output_root=output, command=command)
        return workflows.run_physical_parameter_maps(cfg, output_root=output, command=command)
    if key == "mica_checks":
        return _mica_checks(config, parent, stages, output)
    if key == "mica_review":
        from neuristor.plotting.presentation import render_mica_review

        review = RunBundle.create(
            name="Recreated Mica parameter exploration",
            model="current",
            kind="analysis",
            config=dict(parents={k: str(v) for k, v in stages.items() if k in MICA_NAMES}),
            output_root=output,
            command=command,
        )
        try:
            tables, links = {}, {}
            for stage, name in MICA_NAMES.items():
                source = stages[stage]
                if (source / "stages.json").is_file():
                    source = source / json.loads((source / "stages.json").read_text())["grid"]
                tables[name] = pd.read_csv(source / "grid.csv")
                links[name] = source / "parameter_maps.html"
            table = render_mica_review(tables, links, review)
            metrics = dict(grid_entries=len(table), retained=int(table.included.sum()), diagnostic_only=True)
            review.write_json("metrics.json", metrics, label="Fresh grid coverage and acceptance")
            review.write_text(
                "report.md",
                "# Recreated Mica review\n\nAll six grids were computed anew from the measured R(T) loop. "
                "The original exploratory 0.10 static-fit exception remains explicit; no specimen calibration is adopted.\n",
                label="Scope",
            )
            review.complete(summary=metrics)
        except BaseException as error:
            review.fail(error)
            raise
        return review
    if key == "alternating":
        from neuristor.orchestration.alternating_amplitudes import run_alternating_amplitudes

        return run_alternating_amplitudes(
            _child_config(key, config, parent, stages),
            data_directory=parent.root / "inputs/waveforms",
            output_root=output,
            command=command,
        )
    raise ValueError(f"Unknown reproduction stage {key}")


def _stage_hashes(stage):
    return {p.relative_to(stage).as_posix(): _digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}


def _register_tree(bundle, folder):
    for path in sorted(Path(folder).rglob("*")):
        if path.is_file():
            bundle.register_file(path.relative_to(bundle.root), label=path.name)


def _assemble(parent, config, stages):
    """Generate slide images, link manifest and a complete local result entrypoint."""
    from neuristor.plotting.presentation import plot_reproduction_assets, update_slide_tables

    plot_reproduction_assets(parent, stages)
    root = find_project_root(source_directory(config))
    photo = parent.add_artifact("presentation/vo2_device.png", label="Original experimental device photo (input)")
    shutil.copy2(parent.root / "inputs/device_photo.png", photo)
    # Keep editable sources beside the fresh figures. Figure references that
    # formerly pointed to immutable old run IDs are rebound to this run.
    sources = {source.name: source.read_text() for source in sorted((root / "docs/presentation").glob("*.tex"))}
    sources = update_slide_tables(sources, stages)
    for name, text in sources.items():
        for original, replacement in [
            (
                "../../runs/published/20260816_125905_sample-r-t-major-loop-hysteresis-fit_0849a9",
                stages["specimen_fit"],
            ),
            (
                "../../runs/published/20261020_170231_controlled-frequency-and-amplitude-loss-comparis_26574a",
                stages["comparison"],
            ),
        ]:
            text = text.replace(original, Path(os.path.relpath(replacement, photo.parent)).as_posix())
        parent.write_text("presentation/" + name, text, label="Presentation source with local figure links")
    routes = {route: (stages[key] / name).relative_to(parent.root).as_posix() for route, (key, name) in ROUTES.items()}
    for relative in routes.values():
        if not (parent.root / relative).is_file():
            raise ValueError(f"Missing recreated presentation dependency: {relative}")
    # HTML is the regenerated presentation-data companion. Original slides
    # retain their historical numbers; all fresh optimizer results are explicit.
    links = "".join(
        f'<li><a href="{quote(path, safe="/._-")}">{html.escape(route[1:])}</a></li>' for route, path in routes.items()
    )
    figures = "".join(
        f'<figure><img style="max-width:100%" src="presentation/assets/{p.name}"><figcaption>{html.escape(p.stem)}</figcaption></figure>'
        for p in sorted((parent.root / "presentation/assets").glob("*.png"))
    )
    summary = pd.read_csv(stages["comparison"] / "summary.csv")
    summary = summary[summary.dt_ns == summary.dt_ns.min()]
    parent.write_text(
        "START_HERE.html",
        '<!doctype html><meta charset="utf-8"><title>Recreated presentation data</title>'
        "<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}td,th{padding:6px}table{display:block;overflow:auto}</style>"
        "<h1>Presentation results recreated from measured inputs</h1><p>All derived numerical stages were rerun. "
        "Original laboratory measurements, optimizer seeds and the historical alternating-reset parameter vector are declared inputs. "
        "Search results can differ between environments because the original searches have time budgets. "
        "These remain diagnostic model results, not a newly accepted specimen calibration. "
        "The supplied PDF quotes the historical results; this companion contains the new results.</p>"
        "<ul>"
        + links
        + "</ul><h2>Fresh fitting outcomes</h2>"
        + summary.to_html(index=False, border=0)
        + figures
        + '<p><a href="report.md">Report</a> · <a href="presentation_manifest.json">Dependency manifest</a></p>',
        label="Fresh presentation figures, results and viewer links",
        media_type="text/html",
    )
    parent.write_json(
        "presentation_manifest.json",
        dict(
            schema_version=1,
            profile=config["reproduction"].get("profile", "full"),
            routes=routes,
            entry="START_HERE.html",
        ),
        label="Portable new-run presentation links",
    )


def run_presentation_reproduction(
    config, *, output_root=None, resume=None, project_root=None, command="neuristor reproduce-presentation"
):
    """Recompute all declared presentation stages, optionally resuming intact stages."""
    plan = plan_presentation_reproduction(config)
    root = find_project_root(project_root or source_directory(config))
    began = time.monotonic()
    if resume is not None:
        directory = Path(resume).expanduser().resolve()
        manifest = json.loads((directory / "run.json").read_text())
        if manifest.get("kind") != "presentation-reproduction" or manifest.get("status") == "completed":
            raise ValueError("Resume requires an unfinished presentation reproduction bundle")
        if json.loads((directory / "plan.json").read_text())["fingerprint"] != plan["fingerprint"]:
            raise ValueError(
                "Source, inputs, recipe or environment changed; start a new reproduction instead of mixing results"
            )
        parent = RunBundle(directory, manifest, list(manifest["artifacts"]))
        state = json.loads((directory / "stages.json").read_text())
        checks = json.loads((directory / "stage_checksums.json").read_text())
        if _stage_hashes(directory / "inputs") != json.loads((directory / "input_checksums.json").read_text()):
            raise ValueError("Copied inputs changed; start a new reproduction instead of mixing results")
        parent.manifest["status"] = "running"
        parent.manifest.pop("error", None)
        parent._save_manifest()
    else:
        parent = RunBundle.create(
            name=config["name"],
            model="current",
            kind="presentation-reproduction",
            config=resolved_copy(config),
            output_root=output_root or "runs/local",
            command=command,
            project_root=root,
        )
        parent.write_json("plan.json", plan, label="Declared input closure, source fingerprints and work plan")
        parent.write_json("source_hashes.json", snapshot_package(parent), label="Complete reproduction source")
        inputs = config["reproduction"]["inputs"]
        extensions = dict(
            specimen_data=".tsv",
            mica_data=".tsv",
            search_seeds=".csv",
            alternating_parameters=".json",
            device_photo=".png",
        )
        for name, value in inputs.items():
            original = _resolve(config, value)
            if name == "waveforms":
                for source in sorted(original.glob("*_converted.csv")):
                    shutil.copy2(
                        source,
                        parent.add_artifact("inputs/waveforms/" + source.name, label="Original measured channels"),
                    )
            else:
                shutil.copy2(
                    original,
                    parent.add_artifact("inputs/" + name + extensions[name], label="Original measured/declared input"),
                )
        state, checks = {}, {}
        parent.write_json("input_checksums.json", _stage_hashes(parent.root / "inputs"), label="Copied input integrity")
        parent.write_json("stages.json", state, label="Relative completed-stage locations")
        parent.write_json("stage_checksums.json", checks, label="Completed-stage immutable content checks")
        parent.write_json(
            "environment.json",
            plan.get("environment", {}),
            label="Actual execution package versions",
        )
    parent.add_artifact("status.json", label="Current stage progress")
    print(
        f"REPRODUCTION_DIRECTORY={parent.root}\n{plan['time_allowance']}\nResume interrupted work: neuristor reproduce-presentation --resume {parent.root}",
        flush=True,
    )
    stages = {}
    try:
        for key in STAGES:
            if key in state:
                path = (parent.root / state[key]).resolve()
                if not path.is_relative_to(parent.root.resolve()) or _stage_hashes(path) != checks[key]:
                    raise ValueError(f"Completed stage {key} changed; refusing to mix evidence")
                stages[key] = path
                print(f"Reusing intact completed stage: {key}", flush=True)
                continue
            print(f"[{len(stages) + 1}/{len(STAGES)}] Recomputing {key}", flush=True)
            _atomic(
                parent.path("status.json"), dict(state="running", stage=key, completed=list(stages), total=len(STAGES))
            )
            child = _execute_stage(key, config, parent, stages, parent.root / "stages" / key)
            if child.manifest["status"] != "completed":
                raise ValueError(f"{key} did not complete")
            stages[key] = child.root
            state[key] = child.root.relative_to(parent.root).as_posix()
            checks[key] = _stage_hashes(child.root)
            _register_tree(parent, child.root)
            parent.write_json("stages.json", state, label="Relative completed-stage locations")
            parent.write_json("stage_checksums.json", checks, label="Completed-stage immutable content checks")
        _assemble(parent, config, stages)
        try:
            relative_output = parent.root.resolve().relative_to(root)
            activate = relative_output.parts[:2] == ("runs", "local") and len(relative_output.parts) == 3
        except ValueError:
            activate = False
        activate = activate and plan["profile"] == "full"
        metrics = dict(
            presentation_activated=activate,
            completed_stages=len(stages),
            profile=plan["profile"],
            elapsed_this_session_s=time.monotonic() - began,
            regenerated=True,
            original_measurements_are_inputs=True,
        )
        parent.write_json("metrics.json", metrics, label="Reproduction completeness and timing")
        parent.write_text(
            "report.md",
            "# Presentation reproduction\n\n"
            f"Profile: {plan['profile']}. Completed {len(stages)} stages.\n\n"
            "Every derived numerical result, figure, GIF and HTML viewer was recreated using the existing physics/workflows. "
            "No saved run, GitHub Release or private PDF was an input. Original measured channels, resistance curves, optimizer "
            "seeds and the declared historical alternating-reset vector are necessary source inputs. "
            "The original search wall-clock budgets can produce different candidates on different hardware; compare the new "
            "tables and plots rather than assuming byte-identical optimization results. The supplied historical PDF retains "
            "its recorded numerical values. START_HERE.html contains the fresh values and all recreated images. "
            "The copied TeX sources have fresh figure links and regenerated result tables. Inspect them before compiling a revised PDF; the original PDF remains a historical record.\n\n"
            "Completed stages have content checksums. Resume reuses only intact completed stages and requires identical "
            "source, inputs and recipe. No physical equation, temperature/frequency screen or convergence tolerance was changed.\n",
            label="Method, scope and environment limitations",
        )
        _atomic(parent.path("status.json"), dict(state="completed", completed=list(stages), total=len(STAGES)))
        parent.complete(summary=metrics)
        if activate:
            relative = relative_output
            _atomic(
                root / "runs/local/presentation.json",
                dict(
                    schema_version=1,
                    bundle=relative.as_posix(),
                    manifest_sha256=_digest(parent.root / "presentation_manifest.json"),
                ),
            )
        return parent
    except BaseException as error:
        _atomic(parent.path("status.json"), dict(state="interrupted", completed=list(stages), total=len(STAGES)))
        parent.write_text(
            "report.md",
            "# Interrupted presentation reproduction\n\n"
            f"{type(error).__name__}: {error}\n\nCompleted stages are preserved. Rerun the same command with "
            f"`--resume {parent.root}`. A partially completed run never replaces the active presentation links.\n",
            label="Failure and recovery instructions",
        )
        parent.fail(error)
        raise
