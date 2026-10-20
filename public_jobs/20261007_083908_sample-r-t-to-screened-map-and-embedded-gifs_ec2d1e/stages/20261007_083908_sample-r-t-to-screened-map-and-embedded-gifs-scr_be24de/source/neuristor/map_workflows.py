"""Screened map and point-media orchestration behind the public workflows API.

The physics remains in model/current_drive_sim. Unit conversion is shared with
workflows; completed input bundles are never edited. Keeping these two related
workflows together makes the sampling, verification and packaging path readable.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolved_copy, source_directory, validate_config
from .runs import RunBundle


def run_map_animations(
    config, *, output_root=None, limit=None, reuse_media=None, command="neuristor analyze map-animations", package=True
):
    """Create a new portable map/media bundle without mutating the original map."""
    from time import monotonic
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import zipfile
    from .current_drive_sim import CurrentDriveParams
    from .model import YuanhangResistParams
    from .physical_maps import grid_values, render_physical_maps
    from .map_animations import replay_animation_batch, render_point_animation
    from .lab_replay import major_branches

    validate_config(config)
    settings = copy.deepcopy(config["animations"])
    source = Path(settings["source"])
    if not source.is_absolute():
        source = (source_directory(config) / source).resolve()
    if json.loads((source / "run.json").read_text())["status"] != "completed":
        raise ValueError("Animation source must be a completed map")
    original = json.loads((source / "resolved_config.json").read_text())
    physics = json.loads((source / "physics_parameters.json").read_text())
    physics["resist_params"] = YuanhangResistParams(**physics["resist_params"])
    params = CurrentDriveParams(**physics)
    map_settings = copy.deepcopy(original["maps"])
    current_uA = float(original["input"]["amplitude_uA"])
    map_settings.update(current_uA=current_uA, ambient_K=params.T0_K)
    finest = min(map_settings["verification_dt_ns"])
    if settings["dt_ns"] > finest / 2 + 1e-12:
        raise ValueError("Animation dt must be at most half the finest map timestep")
    if not 0 <= settings["maximum_refinements"] <= 3 or not 1 <= settings["workers"] <= 4:
        raise ValueError("Require 0–3 refinements and 1–4 render workers")
    if settings["batch_size"] < 1 or settings["frames"] < 2:
        raise ValueError("Positive batch size and at least two frames required")
    settings["window_ns"] = map_settings["window_ns"]
    grid = pd.read_csv(source / "grid.csv")
    selected = grid[grid.included].copy()
    total = len(selected)
    if limit is not None and total:
        if limit < 1:
            raise ValueError("Limit must be positive")
        selected = selected.iloc[np.unique(np.linspace(0, total - 1, min(limit, total)).astype(int))]
    bundle = RunBundle.create(
        name=config["name"] + (" benchmark" if limit else ""),
        model="current",
        kind="analysis",
        config=resolved_copy(config),
        output_root=output_root or config.get("output", {}).get("root", "runs"),
        command=command,
    )
    print(f"RUN_DIRECTORY={bundle.root}", flush=True)
    begin = monotonic()
    rows = []
    catalog = {}
    bundle.add_artifact("status.json", label="Media generation progress")

    def progress(state):
        elapsed = monotonic() - begin
        bundle.write_json(
            "status.json",
            dict(
                state=state,
                completed=len(rows),
                requested=len(selected),
                retained_points=total,
                elapsed_s=elapsed,
                projected_total_s=elapsed * len(selected) / len(rows) if rows else None,
            ),
            label="Media generation progress",
        )

    try:
        for folder in ("GIFs", "Figures"):
            bundle.path(folder).mkdir(exist_ok=True)
        for name in ("grid.csv", "verification.csv", "physics_parameters.json", "axis_values.json"):
            shutil.copy2(source / name, bundle.add_artifact(name, label=f"Original map {name}"))
        bundle.write_json(
            "source_provenance.json",
            dict(
                source=str(source),
                source_config=original,
                hashes={
                    name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                    for name in ("grid.csv", "verification.csv", "physics_parameters.json")
                },
            ),
            label="Original numerical map provenance",
        )
        for file in Path(__file__).parent.glob("*.py"):
            shutil.copy2(file, bundle.add_artifact(f"source/neuristor/{file.name}", label=f"Frozen source {file.name}"))
        branches = major_branches(params.resist_params)
        branches.to_csv(bundle.add_artifact("major_branches.csv", label="Model heating/cooling guides"), index=False)
        # An empty screened domain is a valid scientific outcome, not a failed
        # job. Keep readable headers and an empty catalog for the offline viewer.
        pd.DataFrame(columns=["point_index", "gif", "png", "eligible", "converged", "visually_resolved"]).to_csv(
            bundle.add_artifact("animation_checks.csv", label="Refinement and resolution checks"), index=False
        )
        bundle.write_json("gif_catalog.json", {}, label="Stable point IDs and local media links")
        progress("running")
        if reuse_media is not None:
            media_source = Path(reuse_media).resolve()
            if json.loads((media_source / "run.json").read_text())["status"] != "completed":
                raise ValueError("Only completed media bundles can be reused")
            if (media_source / "grid.csv").read_bytes() != (source / "grid.csv").read_bytes():
                raise ValueError("Reused media must correspond to the exact original grid")
            if (media_source / "physics_parameters.json").read_bytes() != (
                source / "physics_parameters.json"
            ).read_bytes():
                raise ValueError("Reused media physics must match exactly")
            old_config = json.loads((media_source / "resolved_config.json").read_text())
            if old_config.get("animations") != config.get("animations"):
                raise ValueError("Reused media integration and rendering recipe must match exactly")
            old_catalog = json.loads((media_source / "gif_catalog.json").read_text())
            hashes = {}
            for key in selected.point_index.astype(int):
                record = old_catalog[str(key)]
                if record["point_index"] != key:
                    raise ValueError("Catalog point ID mismatch")
                for field, folder, extension in [("gif", "GIFs", "gif"), ("png", "Figures", "png")]:
                    relative = f"{folder}/point_{key:05d}.{extension}"
                    if record[field] != relative:
                        raise ValueError("Unexpected media path")
                    shutil.copy2(media_source / relative, bundle.add_artifact(relative, label=record["caption"]))
                    hashes[relative] = hashlib.sha256((media_source / relative).read_bytes()).hexdigest()
                catalog[str(key)] = record
                rows.append(record)
            pd.DataFrame(rows).to_csv(
                bundle.add_artifact("animation_checks.csv", label="Preserved refinement and resolution checks"),
                index=False,
            )
            bundle.write_json("gif_catalog.json", catalog, label="Stable point IDs and local media links")
            bundle.write_json(
                "media_provenance.json",
                dict(source=str(media_source), resimulated=False, hashes=hashes),
                label="Unchanged media hashes and source",
            )
            shutil.copytree(media_source / "source", bundle.path("original_simulation_source"))
            print(f"Reused {len(rows)} GIFs without resimulation", flush=True)
        with ProcessPoolExecutor(
            max_workers=int(settings["workers"]), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            if reuse_media is not None:
                selected_for_simulation = selected.iloc[:0]
            else:
                selected_for_simulation = selected
            for start in range(0, len(selected_for_simulation), int(settings["batch_size"])):
                chunk = selected_for_simulation.iloc[start : start + int(settings["batch_size"])]
                replays = replay_animation_batch(chunk, params, current_uA, map_settings, settings)
                futures = []
                for point, out, record in replays:
                    futures.append(
                        pool.submit(
                            render_point_animation,
                            point,
                            out,
                            record,
                            branches,
                            str(bundle.root),
                            settings,
                            current_uA,
                            params.T0_K,
                        )
                    )
                for future in futures:
                    record = future.result()
                    for field in ("gif", "png"):
                        bundle.register_file(record[field], label=record["caption"])
                    key = str(record["point_index"])
                    catalog[key] = record
                    rows.append(record)
                    progress("running")
                pd.DataFrame(rows).to_csv(
                    bundle.add_artifact("animation_checks.csv", label="Refinement and resolution checks"), index=False
                )
                bundle.write_json("gif_catalog.json", catalog, label="Stable point IDs and local media links")
                print(f"Animated {len(rows)}/{len(selected)} points; elapsed {monotonic() - begin:.1f} s", flush=True)
        # Keep every old retained point in the map. A benchmark clearly reports
        # that only selected points have GIFs; rejected new replays are warnings.
        render_physical_maps(grid, grid_values(map_settings), map_settings, bundle.root, gif_catalog=catalog)
        bundle.register_file("parameter_maps.html", label="Right-click map and synchronized GIFs")
        bundle.register_file("heatmaps.png", label="Six parameter heatmaps")
        elapsed = monotonic() - begin
        metrics = dict(
            gifs=len(rows),
            retained_points=total,
            coverage_complete=len(rows) == total,
            resimulated=reuse_media is None,
            elapsed_s=elapsed,
            projected_all_points_s=elapsed * total / len(rows) if rows else 0.0,
            refinement_passes=sum(r["converged"] and r["eligible"] for r in rows),
            visually_resolved=sum(r["visually_resolved"] for r in rows),
        )
        bundle.write_json("metrics.json", metrics, label="Coverage, runtime and refinement checks")
        report = f"""# Portable map animations

Original map: {source.name}. {len(rows)}/{total} retained points have animations.
{"BENCHMARK ONLY: other points have no GIF yet." if limit else "Complete requested generation."}
Right-click a hovered point and choose View simulation GIF, or select it with a
left click and use the View button. Heatmap point IDs follow the active slider.
Keep this folder viewer parameter_maps.html beside GIFs/. No server or internet
is needed. This stage HTML alone does NOT include the GIF bytes.
{"A portable ZIP is included; extract it before opening." if package else "No ZIP is generated. The parent sample study embeds the GIFs in its standalone HTML."}

Input is constant {current_uA:g} µA; ambient/initial temperature {params.T0_K:g} K.
No matched experiment is available for these conditions, so no laboratory voltage
is overlaid. Three panels show input current, simulated voltage and simulated R(T).
GIFs reveal a few cycles from the steady-state window, not the startup transient.
All displayed R(T) samples come from the supported solver: no interpolation or
projection onto major branches. Simulation dt starts at {settings["dt_ns"]} ns,
half the old finest map step, and halves up to {settings["maximum_refinements"]}
times if frequency/Vpp change exceeds {100 * map_settings["convergence_fraction"]:g}%
or adjacent steady samples change by >0.25 K or >0.08 in ln(R). These last two
are visual resolution criteria, not physical laws. Residual jumps are broken in
the plot instead of connected by invented straight chords. Genuine minor loops
need not coincide with either major branch. Checks are in animation_checks.csv.
Old grid and verification files are byte-preserved; new replay failures remain
diagnostic warnings and do not silently alter the historical map membership.
Frames={settings["frames"]} controls playback only, not integration resolution.
All source code and resolved original physics are frozen in this bundle.

Elapsed {elapsed:.1f} seconds; projected full coverage {metrics["projected_all_points_s"] / 3600:.2f} hours
(small batches can substantially overestimate simulation cost because of batching).
"""
        bundle.write_text("report.md", report, label="Portable sharing and numerical methods")
        if reuse_media is not None:
            bundle.write_text(
                "report.md",
                report + "\nViewer-only rebuild: ordinary-list point metadata and defensive event lookup. "
                "All GIF/PNG bytes reused unchanged; original generation source is in original_simulation_source/. "
                "Runtime above is viewer packaging time, not simulation-generation time.\n",
                label="Portable sharing and numerical methods",
            )
        if package:
            archive = bundle.add_artifact("VO2_parameter_map_with_GIFs.zip", label="Portable offline map and GIFs")
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
                for name in (
                    "parameter_maps.html",
                    "report.md",
                    "animation_checks.csv",
                    "gif_catalog.json",
                    "grid.csv",
                ):
                    z.write(bundle.path(name), name)
                for name in catalog.values():
                    z.write(bundle.path(name["gif"]), name["gif"])
        progress("completed")
        bundle.complete(summary=metrics)
    except Exception as exc:
        progress("failed")
        bundle.fail(exc)
        raise
    return bundle


def run_physical_parameter_maps(
    config, *, output_root=None, command="neuristor analyze physical-maps", reuse_numerics=None
):
    """Create a dense, screened three-parameter map and exact fixed-value slices."""
    from .physical_maps import evaluate_map, render_physical_maps, grid_values

    validate_config(config)
    from .workflows import _current_params_from_config

    params, current_uA, *_ = _current_params_from_config(config)
    settings = copy.deepcopy(config["maps"])
    if settings["window_ns"][1] > params.t_end_s * 1e9 + 1e-6:
        raise ValueError("Map measurement window exceeds the simulation duration")
    settings.update(current_uA=current_uA, ambient_K=params.T0_K)
    bundle = RunBundle.create(
        name=config["name"],
        model="current",
        kind="analysis",
        config=resolved_copy(config),
        output_root=output_root or config.get("output", {}).get("root", "runs"),
        command=command,
    )
    print(f"RUN_DIRECTORY={bundle.root}", flush=True)
    try:
        checkpoint_path = bundle.add_artifact("screening_checkpoint.csv", label="Coarse screening checkpoint")

        def checkpoint(rows):
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
            print(f"Screened {len(rows)} points", flush=True)

        if reuse_numerics is None:
            grid, verification, axes = evaluate_map(params, current_uA, settings, checkpoint=checkpoint)
        else:
            source = Path(reuse_numerics).resolve()
            original = json.loads((source / "resolved_config.json").read_text())
            sections = ("maps", "time", "input", "initial", "electrical", "thermal", "resistance")
            if any(original.get(section) != config.get(section) for section in sections):
                raise ValueError("Reused map settings and timesteps must match exactly")
            if json.loads((source / "physics_parameters.json").read_text()) != dataclasses.asdict(params):
                raise ValueError("Reused physical parameters must match exactly")
            if json.loads((source / "run.json").read_text())["status"] != "completed":
                raise ValueError("Only a completed numerical map can be reused")
            grid = pd.read_csv(source / "grid.csv")
            verification = pd.read_csv(source / "verification.csv")
            axes = grid_values(settings)
            shutil.copy2(source / "screening_checkpoint.csv", checkpoint_path)
            hashes = {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ["grid.csv", "verification.csv", "screening_checkpoint.csv"]
            }
            bundle.write_json(
                "numerical_provenance.json",
                dict(source_bundle=str(source), hashes=hashes),
                label="Reused numerical provenance and hashes",
            )
        grid_path = bundle.add_artifact("grid.csv", label="All grid points and screening flags")
        verification_path = bundle.add_artifact("verification.csv", label="Two smaller timestep checks")
        if reuse_numerics is None:
            grid.to_csv(grid_path, index=False)
            verification.to_csv(verification_path, index=False)
        else:
            shutil.copy2(source / "grid.csv", grid_path)
            shutil.copy2(source / "verification.csv", verification_path)
        grid[grid.included].to_csv(
            bundle.add_artifact("included.csv", label="Retained physically screened oscillators"), index=False
        )
        bundle.write_json(
            "axis_values.json",
            dict(zip(("C_th_pJ_per_K", "S_e_mW_per_K", "C_pF"), [a.tolist() for a in axes])),
            label="Exact sampled axes",
        )
        render_physical_maps(grid, axes, settings, bundle.root)
        bundle.register_file("parameter_maps.html", label="Interactive 3D and six heatmaps", media_type="text/html")
        bundle.register_file("heatmaps.png", label="Six fixed-parameter heatmaps", media_type="image/png")
        metrics = dict(
            points=len(grid),
            retained_points=int(grid.included.sum()),
            coarse_oscillators=int(grid.sustained.sum()),
            coarse_eligible=int(grid.coarse_eligible.sum()),
            coarse_temperature_failures=int((~pd.read_csv(checkpoint_path).temperature_valid).sum()),
            maximum_displayed_frequency_MHz=float(grid.loc[grid.included, "frequency_MHz"].max()),
            maximum_displayed_temperature_K=float(grid.loc[grid.included, "temperature_max_K"].max()),
            current_uA=current_uA,
            ambient_K=params.T0_K,
            verification_dt_ns=settings["verification_dt_ns"],
        )
        bundle.write_json("metrics.json", metrics, label="Screened map summary")
        report = f"""# Steady-state extrema parameter maps

Input: {current_uA:g} µA ideal constant current; ambient and initial temperature
{params.T0_K:g} K. The same fitted static sample resistance is used everywhere.
There are {len(grid):,} actual simulations in the coarse grid, with
{int(grid.included.sum()):,} retained points after screening and verification.

Method: use only {settings["window_ns"][0]:g}–{settings["window_ns"][1]:g} ns.
Voltage amplitude is the median of complete-cycle maxima minus minima, in mV.
Frequency is 1000 divided by the median adjacent-maximum period, in ns.
Extrema are located using 0.5 ns smoothing; their voltages use original samples.
At least two complete cycles, cycles in both half-windows, amplitude retention
0.75–1.333 and period coefficient of variation <=0.2 are required.

The display additionally requires frequency <={settings["maximum_frequency_MHz"]:g} MHz,
the entire temperature trajectory inside [{params.resist_params.T_min_K:g},
{params.resist_params.T_max_K:g}] K, and nonnegative sampled proximity slope.
The frequency cutoff is the meeting's operating filter, not a proven material limit.
Each coarse survivor is rerun at {settings["verification_dt_ns"]} ns; frequency
and amplitude must change by <={100 * settings["convergence_fraction"]:g}% at both refinements.
Only these verified survivors are plotted. Coarse rejected points have not all
been verified at finer steps, so a blank cell does not prove absence of oscillations.

Each heatmap fixes exactly one grid value: {settings["fixed_values"]}.
The other two parameters vary independently; frequency and peak-to-peak voltage
are shown separately. Blank cells are screened out; grid.csv preserves all reasons.
No measured parameter uncertainty is inferred from these exploratory ranges.
The static R(T) preset does not independently identify gamma.
"""
        bundle.write_text("report.md", report, label="Method, assumptions and results")
        source_files = [
            "current_drive_sim.py",
            "model.py",
            "extrema.py",
            "physical_maps.py",
            "workflows.py",
            "map_workflows.py",
        ]
        for name in source_files:
            source_path = Path(__file__).parent / name
            if reuse_numerics is not None and name in ["current_drive_sim.py", "model.py", "extrema.py"]:
                source_path = source / "source" / "neuristor" / name
            shutil.copy2(source_path, bundle.add_artifact(f"source/neuristor/{name}", label=f"Source: {name}"))
        bundle.write_json(
            "physics_parameters.json", dataclasses.asdict(params), label="Resolved physical parameters in SI"
        )
        bundle.complete(summary=metrics)
    except BaseException as exc:
        bundle.fail(exc)
        raise
    return bundle
