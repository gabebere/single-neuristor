"""One sample, one recipe, one portable evidence folder.

This is orchestration, not a new physical model or dynamic optimizer. It calls
the established R(T) fitter, screened grid workflow and refined GIF workflow.
The major-loop fit cannot identify gamma, capacitance or thermal parameters;
those remain explicit user assumptions or exploration axes.
"""

from __future__ import annotations

import copy
import base64
import hashlib
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import ConfigError, resolved_copy, source_directory, validate_config
from .physical_maps import AXES, grid_values
from .resistance_custom_analysis import is_major_loop_temperature_trace
from .runs import RunBundle


def load_study_rt(path: str | Path, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Read CSV/TSV and convert declared units without sorting by temperature.

    Hysteresis depends on acquisition order. An optional time column sorts rows
    chronologically; otherwise file order is preserved. A recognized units row
    is permitted, but missing/non-numeric measurement rows are rejected rather
    than silently removing a branch. No data are extracted from images.
    """
    temperature_unit = settings.get("temperature_unit", "K")
    resistance_unit = settings.get("resistance_unit", "ohm")
    if temperature_unit not in {"K", "C"} or resistance_unit not in {"ohm", "kohm"}:
        raise ConfigError("Use temperature_unit='K' or 'C' and resistance_unit='ohm' or 'kohm'")
    frame = pd.read_csv(Path(path), sep=None, engine="python")
    frame.columns = [str(column).strip() for column in frame.columns]
    names = {
        "Temperature": str(settings.get("temperature_column", "Temperature")),
        "Resistance": str(settings.get("resistance_column", "Resistance")),
    }
    time_column = str(settings.get("time_column", ""))
    if time_column:
        names["Time"] = time_column
    missing = set(names.values()) - set(frame.columns)
    if missing:
        raise ConfigError(f"R(T) input is missing declared columns: {sorted(missing)}")
    frame = frame[list(names.values())].rename(columns={v: k for k, v in names.items()})
    if len(frame):
        first = str(frame.iloc[0]["Temperature"]).strip().lower()
        if first in {"k", "c", "°c", "kelvin", "celsius"}:
            expected = "K" if first in {"k", "kelvin"} else "C"
            resistance_label = str(frame.iloc[0]["Resistance"]).strip().lower()
            labels = {"ohm": {"ohm", "ohms", "ω"}, "kohm": {"kohm", "kohms", "kω"}}
            if expected != temperature_unit or resistance_label not in labels[resistance_unit]:
                raise ConfigError("Input units row conflicts with the declared study units")
            frame = frame.iloc[1:]
    frame = frame.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    if len(frame) < 10 or not np.isfinite(frame.to_numpy(float)).all():
        raise ConfigError("R(T) requires at least ten finite numeric rows; check missing values and units")
    if time_column:
        if frame.Time.duplicated().any():
            raise ConfigError("Acquisition times must be unique")
        frame = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    if temperature_unit == "C":
        frame["Temperature"] += 273.15
    if resistance_unit == "kohm":
        frame["Resistance"] *= 1000.0
    if (frame.Temperature <= 0).any() or (frame.Resistance <= 0).any():
        raise ConfigError("Temperatures in K and resistances in ohm must be positive")
    if frame.Temperature.median() < 200:
        raise ConfigError("This VO2 major-loop fitter requires a measured temperature median >=200 K")
    if not is_major_loop_temperature_trace(frame.Temperature.to_numpy()):
        raise ConfigError("Supply one complete heating/cooling major loop, with exactly one reversal")
    return frame


def _finite(value: Any, name: str, *, positive: bool = True) -> float:
    number = float(value)
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise ConfigError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return number


def plan_sample_study(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate inputs and report workload before fitting or creating a run.

    Integration counts are conservative upper bounds, not runtime predictions:
    only coarse survivors are refined or animated. This inexpensive plan does
    not run an optimizer and does not claim that any oscillator exists.
    """
    validate_config(config)
    if config["model"] != "current" or "study" not in config:
        raise ConfigError("A sample study requires model='current' and a [study] table")
    settings = config["study"]
    source = Path(str(settings["resistance_data"])).expanduser()
    if not source.is_absolute():
        source = source_directory(config) / source
    source = source.resolve()
    data = load_study_rt(source, settings)
    maps, animations = config["maps"], config["animations"]
    axes = grid_values(maps)
    count = math.prod(len(axis) for axis in axes)
    if count > int(maps.get("max_points", 30000)) or count < 1:
        raise ConfigError("The grid is empty or exceeds maps.max_points")
    dt = _finite(config["time"]["dt_ns"], "time.dt_ns")
    duration_ns = _finite(config["time"]["duration_us"], "time.duration_us") * 1000
    steps = [dt, *[_finite(x, "maps.verification_dt_ns") for x in maps["verification_dt_ns"]]]
    if len(steps) != 3 or any(b >= a for a, b in zip(steps[:-1], steps[1:])):
        raise ConfigError("Provide exactly two successively smaller verification timesteps")
    start, stop = map(float, maps["window_ns"])
    if not (math.isfinite(start) and math.isfinite(stop) and 0 <= start < stop <= duration_ns + 1e-6):
        raise ConfigError("maps.window_ns must lie inside the simulated time range")
    if (stop - start) / dt < 9:
        raise ConfigError("The measurement window needs at least nine coarse samples")
    ceiling = _finite(maps["maximum_frequency_MHz"], "maps.maximum_frequency_MHz")
    tolerance = _finite(maps["convergence_fraction"], "maps.convergence_fraction")
    if tolerance >= 1:
        raise ConfigError("maps.convergence_fraction must be smaller than one")
    for name, values in zip(AXES, axes):
        if not np.isclose(values, float(maps["fixed_values"][name]), rtol=1e-9, atol=0).any():
            raise ConfigError(f"maps.fixed_values.{name} must be one of its sampled axis values")
    gif_dt = _finite(animations["dt_ns"], "animations.dt_ns")
    if gif_dt > steps[-1] / 2 + 1e-12:
        raise ConfigError("animations.dt_ns must be at most half the finest map timestep")
    for name, lower, upper in [("maximum_refinements", 0, 3), ("workers", 1, 4), ("frames", 2, 1000)]:
        value = animations[name]
        if isinstance(value, bool) or int(value) != value or not lower <= int(value) <= upper:
            raise ConfigError(f"animations.{name} must be an integer in [{lower}, {upper}]")
    for name, value in [
        ("maps.batch_size", maps.get("batch_size", 384)),
        ("animations.batch_size", animations["batch_size"]),
    ]:
        if int(value) != value or int(value) < 1:
            raise ConfigError(f"{name} must be a positive integer")
    bootstrap = settings.get("bootstrap_samples", 0)
    if isinstance(bootstrap, bool) or int(bootstrap) != bootstrap or bootstrap < 0:
        raise ConfigError("study.bootstrap_samples must be a nonnegative integer")
    _finite(settings["maximum_static_rmse_log10"], "study.maximum_static_rmse_log10")
    if settings.get("temperature_domain", "measured") not in {"measured", "fit-padded"}:
        raise ConfigError("study.temperature_domain must be measured or fit-padded")
    # The map solver is a constant deterministic current experiment. Refuse
    # pulsed/noisy recipe fields instead of silently ignoring their meaning.
    if any(config["input"].get(key) not in (None, 0, 0.0) for key in ("on_us", "off_us")):
        raise ConfigError("Sample maps use constant current; omit input.on_us and input.off_us")
    if config["time"].get("pre_us", 0) != 0 or config["thermal"].get("noise_W_sqrt_s", 0) != 0:
        raise ConfigError("Sample maps require pre_us=0 and thermal noise=0")
    _finite(config["input"]["amplitude_uA"], "input.amplitude_uA")
    for key in ("C_th_pJ_per_K", "S_e_mW_per_K", "T0_K"):
        _finite(config["thermal"][key], f"thermal.{key}")
    _finite(config["electrical"]["C_pF"], "electrical.C_pF", positive=False)
    _finite(config.get("initial", {}).get("temperature_K", config["thermal"]["T0_K"]), "initial.temperature_K")
    _finite(config.get("initial", {}).get("voltage_V", 0), "initial.voltage_V", positive=False)
    overrides = config.get("resistance", {}).get("parameters", {})
    if set(overrides) - {"gamma", "proximity_function"}:
        raise ConfigError(
            "Automatic R(T) fitting permits only gamma/proximity_function overrides, not fitted constants"
        )
    if config.get("resistance", {}).get("start_branch", "insulator") not in {"insulator", "metal"}:
        raise ConfigError("resistance.start_branch must be insulator or metal")
    if overrides.get("proximity_function", "yuanhang") not in {"yuanhang", "tanh"}:
        raise ConfigError("Supported proximity functions are yuanhang and tanh")
    if "gamma" in overrides:
        _finite(overrides["gamma"], "resistance.parameters.gamma")
    finest_gif = gif_dt / 2 ** int(animations["maximum_refinements"])
    if duration_ns / finest_gif > 10_000_000:
        raise ConfigError("More than ten million samples per GIF trajectory; revise the timing/budget")
    return dict(
        resistance_data=str(source),
        input_rows=len(data),
        grid_points=count,
        current_uA=float(config["input"]["amplitude_uA"]),
        maximum_frequency_MHz=ceiling,
        measured_temperature_range_K=[float(data.Temperature.min()), float(data.Temperature.max())],
        bootstrap_samples=int(bootstrap),
        map_dt_ns=steps,
        finest_possible_gif_dt_ns=finest_gif,
        map_integration_steps_upper_bound=count * sum(math.ceil(duration_ns / x) for x in steps),
        gif_integration_steps_upper_bound=count
        * sum(math.ceil(duration_ns / (gif_dt / 2**i)) for i in range(int(animations["maximum_refinements"]) + 1)),
        interpretation="Conditional constant-current exploration, not a fitted laboratory circuit",
    )


def run_sample_study(config, *, output_root=None, command="neuristor study run") -> RunBundle:
    """Execute a validated fit → screened grid → full GIF pipeline.

    Completed child bundles stay immutable inside stages/. The root HTML is the
    entry point. Its point GIFs and fit image are embedded for standalone sharing;
    the full folder additionally retains fit data, checks and frozen source.
    """
    from .workflows import run_resistance_fit, run_physical_parameter_maps, run_map_animations

    plan = plan_sample_study(config)
    bundle = RunBundle.create(
        name=config["name"],
        model="current",
        kind="study",
        config=resolved_copy(config),
        output_root=output_root or config.get("output", {}).get("root", "runs"),
        command=command,
    )
    print(f"STUDY_DIRECTORY={bundle.root}", flush=True)
    try:
        bundle.write_json("plan.json", plan, label="Input validation and workload bounds")
        source = Path(plan["resistance_data"])
        shutil.copy2(
            source, bundle.add_artifact(f"inputs/resistance_original{source.suffix}", label="Original measured R(T)")
        )
        data = load_study_rt(source, config["study"])
        normalized = bundle.add_artifact("inputs/resistance_K_ohm.tsv", label="Measured R(T) in declared SI units")
        data.to_csv(normalized, sep="\t", index=False)
        bundle.write_json(
            "input_provenance.json",
            dict(
                original_path=str(source),
                original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                normalized_sha256=hashlib.sha256(normalized.read_bytes()).hexdigest(),
                declarations=config["study"],
            ),
            label="Input hashes and unit declarations",
        )
        for file in Path(__file__).parent.glob("*.py"):
            shutil.copy2(file, bundle.add_artifact(f"source/neuristor/{file.name}", label=f"Frozen source {file.name}"))
        stages = bundle.path("stages")
        fit = run_resistance_fit(
            normalized,
            name=f"{config['name']} - R(T) fit",
            seed=int(config.get("seed", 42)),
            method="major-loop",
            bootstrap_samples=plan["bootstrap_samples"],
            output_root=stages,
            command=command,
        )
        preset = json.loads((fit.root / "resistance_preset.json").read_text())
        rmse = float(preset["fit_metrics"]["rmse_log10"])
        if rmse > float(config["study"]["maximum_static_rmse_log10"]):
            raise ValueError(
                f"Static R(T) RMSE {rmse:.6g} exceeds the configured acceptance threshold; inspect the fit stage"
            )
        mapped = copy.deepcopy(dict(config))
        mapped["name"] += " - screened grid"
        mapped["resistance"] = dict(
            preset=str(fit.root / "resistance_preset.json"),
            start_branch=config.get("resistance", {}).get("start_branch", "insulator"),
            parameters=copy.deepcopy(config.get("resistance", {}).get("parameters", {})),
        )
        if config["study"].get("temperature_domain", "measured") == "measured":
            mapped["resistance"]["parameters"].update(
                T_min_K=plan["measured_temperature_range_K"][0], T_max_K=plan["measured_temperature_range_K"][1]
            )
        resistance = {**preset["resist_params"], **mapped["resistance"]["parameters"]}
        for temperature in (
            mapped["thermal"]["T0_K"],
            mapped.get("initial", {}).get("temperature_K", mapped["thermal"]["T0_K"]),
        ):
            if not resistance["T_min_K"] <= temperature <= resistance["T_max_K"]:
                raise ValueError("Ambient/initial temperature is outside the selected R(T) domain")
        grid = run_physical_parameter_maps(mapped, output_root=stages, command=command)
        animated = copy.deepcopy(mapped)
        animated["name"] = config["name"] + " - point GIFs"
        animated["animations"]["source"] = str(grid.root)
        media = run_map_animations(animated, output_root=stages, command=command, package=False)
        media_metrics = json.loads((media.root / "metrics.json").read_text())
        grid_metrics = json.loads((grid.root / "metrics.json").read_text())
        summary = dict(
            grid_metrics,
            static_rmse_log10=rmse,
            gifs=media_metrics["gifs"],
            coverage_complete=media_metrics["coverage_complete"],
            replay_refinement_passes=media_metrics["refinement_passes"],
            visually_resolved=media_metrics["visually_resolved"],
            dynamic_calibration_adopted=False,
        )
        bundle.write_json("metrics.json", summary, label="Static fit, map and animation results")
        relative = {
            name: child.root.relative_to(bundle.root).as_posix()
            for name, child in [("fit", fit), ("grid", grid), ("media", media)]
        }
        bundle.write_json("stages.json", relative, label="Immutable study stage locations")
        fit_link = relative["fit"] + "/figures/resistance_fit.png"
        map_link = relative["media"] + "/parameter_maps.html"
        description = (
            f"{plan['input_rows']} measured R(T) rows; log10 RMSE {rmse:.5g}. "
            f"{summary['retained_points']} / {plan['grid_points']} grid points retained; {summary['gifs']} GIFs. "
            f"Constant {plan['current_uA']:g} µA, frequency ceiling {plan['maximum_frequency_MHz']:g} MHz."
        )
        from .map_animations import write_standalone_map

        image_uri = "data:image/png;base64," + base64.b64encode((bundle.root / fit_link).read_bytes()).decode("ascii")
        introduction = (
            f'<section style="max-width:1050px;margin:25px auto;padding:20px;font:17px system-ui">'
            f"<h1>{html.escape(config['name'])}</h1><p>{html.escape(description)}</p>"
            "<p>Conditional model exploration, not an experimentally validated parameter set. "
            "Gamma and thermal/electrical constants are assumptions; only the major R(T) loop was fitted. "
            "Every point GIF is embedded: this HTML can be shared alone, without a ZIP.</p>"
            f'<details><summary>Sample R(T) fit</summary><img style="max-width:100%" '
            f'src="{image_uri}" alt="Measured and fitted resistance"></details>'
            "<details><summary>Exact recipe, resistance parameters and input provenance</summary><pre "
            'style="white-space:pre-wrap">'
            + html.escape(
                json.dumps(
                    dict(
                        config=resolved_copy(config),
                        simulation_resistance=resistance,
                        summary=summary,
                        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    ),
                    indent=2,
                )
            )
            + "</pre></details></section>"
        )
        embedded = write_standalone_map(
            media.root,
            bundle.path("parameter_maps.html"),
            introduction=introduction,
        )
        bundle.register_file("parameter_maps.html", label="Standalone map with embedded GIFs", media_type="text/html")
        summary.update(embedded)
        bundle.write_json("metrics.json", summary, label="Static fit, map, animation and sharing results")
        map_link = "parameter_maps.html"
        report = f"""# {config["name"]}

{description}

## What was fitted and what was assumed

The measured major loop was fitted for R0, Ea/kB, metallic resistance, Tc,
width and beta in log10 resistance. Gamma is not inferred from a major loop.
Its simulation value is {resistance["gamma"]:g}; proximity function is
{resistance.get("proximity_function", "yuanhang")}. Electrical/thermal capacitance,
environmental conductance, ambient temperature and initial conditions are recipe
assumptions or sampled axes, not estimates extracted from this R(T) fit.

The selected resistance domain is {config["study"].get("temperature_domain", "measured")}:
{resistance["T_min_K"]:g}–{resistance["T_max_K"]:g} K.
The grid uses late complete-cycle voltage maxima/minima for frequency and Vpp,
and separately screens persistence, temperature, sampled proximity slope and the
configured frequency ceiling. Survivors must pass two finer timestep checks.
Coarse rejected points are not all refined: blank cells are not a proof of no cycles.
Heatmap colors rescale independently to each displayed slice, never the whole grid.

GIFs use finer integration and refinement checks. {summary["replay_refinement_passes"]}
of {summary["gifs"]} replays passed refinement/eligibility; {summary["visually_resolved"]}
were visually resolved. Unresolved R(T) jumps are broken instead of inventing
smooth paths. Replay failures are warned in the viewer; they do not silently
change membership of the already completed numerical grid.

This is NOT a validation against experimental voltage/current traces or a fitted
TIA circuit. The cutoff is an operational choice, not a universal material law.
An empty screened domain is a valid outcome. No automatic laboratory overlay is
fabricated for an unmatched constant-current experiment.

## Review in order

1. [Resistance fit]({relative["fit"]}/report.md), its graph and parameter CSV.
2. [Grid methodology and exclusions]({relative["grid"]}/report.md), grid.csv and verification.csv.
3. [Clickable map and all GIFs]({map_link}), animation_checks.csv and gif_catalog.json.

Open parameter_maps.html; share that one file. All {embedded["embedded_gifs"]} GIFs
and the fitted R(T) image are embedded. No ZIP is generated. The standalone HTML
is {embedded["html_bytes"] / 1e6:.2f} MB; large grids may exceed email limits.
The standalone file includes the exact recipe, resolved resistance and input hash,
but the complete raw/numerical/source archive remains in the study folder.
Original measurements, hashes, resolved recipes and frozen source are included.
"""
        bundle.write_text("report.md", report, label="Study methodology, results and limitations")
        page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(config["name"])}</title>
<style>body{{font:17px system-ui;max-width:1000px;margin:50px auto;padding:0 24px;background:#f7f9fc;color:#18283a}}
a{{color:#185bb8}}.card{{padding:25px;background:white;border-radius:14px;margin:20px 0;box-shadow:0 3px 15px #15243812}}
img{{max-width:100%}}.note{{border-left:4px solid #eeb64e;padding-left:18px}}</style>
<h1>{html.escape(config["name"])}</h1><p>{html.escape(description)}</p>
<p class="note">Conditional model exploration, not an experimentally validated parameter set.
Gamma and dynamic constants are assumptions. Empty cells were screened out, not proved nonoscillatory.</p>
<div class="card"><h2>1. Sample resistance fit</h2><a href="{fit_link}"><img src="{fit_link}" alt="Measured and fitted hysteresis"></a>
<p><a href="{relative["fit"]}/parameter_summary.csv">Parameters</a> · <a href="{relative["fit"]}/report.md">Fit report</a></p></div>
<div class="card"><h2>2. Explore the map and simulations</h2><p><a href="{map_link}">Open clickable 3D map and adjustable heatmaps →</a></p>
<p>Select a point to view its current/voltage/R(T) GIF. Each slice has its own color scale.</p></div>
<div class="card"><h2>3. Evidence and sharing</h2><p><a href="report.md">Full methods and limitations</a> ·
<a href="{relative["grid"]}/grid.csv">All sampled points and screening flags</a> ·
<a href="{relative["media"]}/animation_checks.csv">GIF checks</a></p>
<p>Share parameter_maps.html alone: all GIFs and the R(T) fit image are embedded. No ZIP is needed.</p></div></html>"""
        bundle.write_text("START_HERE.html", page, label="One study entry point", media_type="text/html")
        # Register nested evidence, not duplicate it. Every declared artifact
        # remains relative to the parent bundle and survives publish/extraction.
        for child in (fit, grid, media):
            for file in child.root.rglob("*"):
                if file.is_file():
                    bundle.register_file(file.relative_to(bundle.root), label=f"{child.manifest['name']}: {file.name}")
        bundle.complete(summary=summary)
    except BaseException as exc:
        bundle.write_text(
            "report.md",
            f"# Study stopped\n\n{type(exc).__name__}: {exc}\n\n"
            "No dynamic calibration was adopted. Completed stages remain available under stages/.",
            label="Failure explanation",
        )
        bundle.fail(exc)
        raise
    return bundle
