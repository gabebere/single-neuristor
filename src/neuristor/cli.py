"""Unified terminal interface for simulations, sweeps, analysis, and archives."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
import pandas as pd

from .config import ConfigError, apply_overrides, load_toml
from .runs import RunRegistry, find_project_root
from .validation import validate_repository
from .plotting.visualization import animate_current_resistance_temperature, plot_resistance_temperature_trajectory
from .workflows import (
    run_environmental_conductance,
    run_lab_analysis,
    run_model_validation,
    run_oscillation_audit,
    run_hysteresis_reconstruction,
    run_waveform_parameter_inference,
    run_joint_inference,
    run_lab_replay,
    run_steady_search,
    run_steady_comparison,
    run_optimizer_audit,
    run_optimizer_review,
    run_physical_parameter_maps,
    run_boundary_maps,
    run_trend_validation,
    run_map_animations,
    export_saved_scope,
    run_loss_scopes,
    run_resistance_fit,
    run_simulation,
    run_sweep,
    run_thermal_capacitance,
    run_sample_study,
    refresh_parameter_maps,
)


app = typer.Typer(
    name="neuristor",
    help="Reproducible VO2 neuristor experiments and run-archive tools.",
    no_args_is_help=True,
    add_completion=False,
)
simulate_app = typer.Typer(help="Run a configured current- or voltage-source simulation.")
sweep_app = typer.Typer(help="Run parameter sweeps from TOML recipes.")
fit_app = typer.Typer(help="Fit measured specimen data.")
analyze_app = typer.Typer(help="Analyze laboratory evidence.")
runs_app = typer.Typer(help="Browse and publish the run archive.")
study_app = typer.Typer(help="Fit a sample R(T) curve and build a complete screened map with GIFs.")
app.add_typer(simulate_app, name="simulate")
app.add_typer(sweep_app, name="sweep")
app.add_typer(fit_app, name="fit")
app.add_typer(analyze_app, name="analyze")
app.add_typer(runs_app, name="runs")
app.add_typer(study_app, name="study")


def _command() -> str:
    return shlex.join(["neuristor", *sys.argv[1:]])


def _configured(path: Path, overrides: list[str], expected_model: str | None = None) -> dict:
    try:
        config = apply_overrides(load_toml(path), overrides)
    except (ConfigError, OSError, ValueError) as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(2) from exc
    if expected_model is not None and str(config["model"]) != expected_model:
        typer.echo(f"Configuration model is {config['model']!r}; expected {expected_model!r}.", err=True)
        raise typer.Exit(2)
    return config


def _announce_bundle(path: Path) -> None:
    typer.echo(f"Completed: {path}")
    typer.echo(f"Manifest: {path / 'run.json'}")


@app.command("reproduce-presentation")
def reproduce_presentation(
    config: Optional[Path] = typer.Option(None, "--config", exists=True, dir_okay=False),
    plan: bool = typer.Option(False, "--plan", help="Show inputs, stages and compute allowance without running experiments."),
    resume: Optional[Path] = typer.Option(None, "--resume", exists=True, file_okay=False,
        help="Resume an interrupted reproduction, keeping intact completed stages."),
    workers: Optional[int] = typer.Option(None, "--workers", min=1, help="Animation rendering processes."),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
    smoke: bool = typer.Option(False, "--smoke", hidden=True),
) -> None:
    """Optionally recreate all presentation results and viewers (several hours)."""
    from .orchestration.presentation_reproduction import plan_presentation_reproduction
    from .workflows import run_presentation_reproduction

    recipe = config or find_project_root()/"experiments/reproduce_presentation.toml"
    try:
        cfg = load_toml(recipe)
        if resume is not None:
            cfg = json.loads((resume/"resolved_config.json").read_text())
            cfg["_source"] = str(recipe.resolve())
        if workers is not None:
            cfg["reproduction"]["workers"] = workers
        if smoke:
            cfg["reproduction"]["profile"] = "smoke"
        if plan:
            details = plan_presentation_reproduction(cfg)
            details.pop("source_input_hashes")
            typer.echo(json.dumps(details, indent=2))
            return
        bundle = run_presentation_reproduction(cfg, output_root=output_root, resume=resume, command=_command())
        _announce_bundle(bundle.root)
        typer.echo(f"Recreated results and figures: {bundle.root / 'START_HERE.html'}")
        if bundle.manifest["summary"].get("presentation_activated"):
            typer.echo("The slide viewer links now select this completed reproduction. Open with: neuristor present")
        elif cfg["reproduction"].get("profile") == "smoke":
            typer.echo("Developer smoke check only; the presentation still uses the full results.")
        else:
            typer.echo("Custom output location: open its START_HERE.html; the default slide links were not changed.")
    except (ConfigError, OSError, ValueError, KeyError) as exc:
        typer.echo(f"Reproduction stopped: {exc}", err=True)
        raise typer.Exit(1) from exc


@study_app.command("plan")
def study_plan(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    set_values: list[str] = typer.Option([], "--set", help="Override a dotted TOML path."),
) -> None:
    """Check units, input columns, bounds and workload without running a fit."""
    from .orchestration.sample_study import plan_sample_study

    try:
        plan = plan_sample_study(_configured(config, set_values, "current"))
    except (ConfigError, ValueError, OSError, KeyError) as exc:
        typer.echo(f"Study configuration error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(plan, indent=2))


@study_app.command("run")
def study_run(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    set_values: list[str] = typer.Option([], "--set", help="Override a dotted TOML path."),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
) -> None:
    """Fit the sample loop, screen the grid and embed every GIF in one HTML."""
    try:
        bundle = run_sample_study(
            _configured(config, set_values, "current"), output_root=output_root, command=_command()
        )
    except (ConfigError, ValueError, OSError, KeyError) as exc:
        typer.echo(f"Study stopped: {exc}", err=True)
        raise typer.Exit(2) from exc
    _announce_bundle(bundle.root)
    typer.echo(f"Share this standalone HTML: {bundle.root / 'parameter_maps.html'}")


@simulate_app.command("current")
def simulate_current(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, readable=True),
    set_values: list[str] = typer.Option(
        [], "--set", help="Override a dotted TOML path, for example input.amplitude_uA=600."
    ),
    output_root: Optional[Path] = typer.Option(None, "--output-root", help="Override [output].root."),
) -> None:
    """Run an ideal-current-source experiment."""

    bundle = run_simulation(_configured(config, set_values, "current"), output_root=output_root, command=_command())
    _announce_bundle(bundle.root)


@simulate_app.command("voltage")
def simulate_voltage(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, readable=True),
    set_values: list[str] = typer.Option(
        [], "--set", help="Override a dotted TOML path, for example input.amplitude_V=14.5."
    ),
    output_root: Optional[Path] = typer.Option(None, "--output-root", help="Override [output].root."),
) -> None:
    """Run the Yuanhang voltage-source circuit experiment."""

    bundle = run_simulation(_configured(config, set_values, "voltage"), output_root=output_root, command=_command())
    _announce_bundle(bundle.root)


@sweep_app.command("run")
def sweep_run(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, readable=True),
    set_values: list[str] = typer.Option([], "--set", help="Override a dotted TOML path."),
    output_root: Optional[Path] = typer.Option(None, "--output-root", help="Override [output].root."),
) -> None:
    """Run a Cartesian parameter sweep."""

    configured = _configured(config, set_values)
    if str(configured.get("kind")) != "sweep":
        typer.echo("Configuration must set kind='sweep'.", err=True)
        raise typer.Exit(2)
    bundle = run_sweep(configured, output_root=output_root, command=_command())
    _announce_bundle(bundle.root)


@fit_app.command("resistance")
def fit_resistance(
    data: Path = typer.Option(..., "--data", exists=True, dir_okay=False, readable=True),
    name: str = typer.Option("Resistance fit", "--name"),
    seed: int = typer.Option(42, "--seed"),
    random_iters: int = typer.Option(12_000, "--random-iters", min=0),
    local_passes: int = typer.Option(180, "--local-passes", min=0),
    method: str = typer.Option("auto", "--method", help="auto, major-loop, or stateful"),
    bootstrap_samples: int = typer.Option(500, "--bootstrap-samples", min=0),
    output_root: Path = typer.Option(Path("runs/local"), "--output-root"),
) -> None:
    """Fit Yuanhang resistance/hysteresis parameters to measured R(T)."""

    bundle = run_resistance_fit(
        data,
        name=name,
        seed=seed,
        random_iters=random_iters,
        local_passes=local_passes,
        method=method,
        bootstrap_samples=bootstrap_samples,
        output_root=output_root,
        command=_command(),
    )
    _announce_bundle(bundle.root)


@analyze_app.command("lab")
def analyze_lab(
    data_directory: Path = typer.Option(..., "--data", exists=True, file_okay=False, readable=True),
    name: str = typer.Option("Measured laboratory current sweep", "--name"),
    output_root: Path = typer.Option(Path("runs/local"), "--output-root"),
) -> None:
    """Analyze professor-supplied numerical current/voltage waveforms."""

    bundle = run_lab_analysis(data_directory, name=name, output_root=output_root, command=_command())
    _announce_bundle(bundle.root)


@analyze_app.command("model-validation")
def analyze_model_validation(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, readable=True),
    set_values: list[str] = typer.Option([], "--set", help="Override a dotted TOML path."),
    output_root: Optional[Path] = typer.Option(None, "--output-root", help="Override [output].root."),
) -> None:
    """Compare one frozen specimen model with every measured current trace."""

    configured = _configured(config, set_values, "current")
    bundle = run_model_validation(configured, output_root=output_root, command=_command())
    _announce_bundle(bundle.root)


@analyze_app.command("reconstruct-hysteresis")
def analyze_hysteresis_reconstruction(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, readable=True),
    set_values: list[str] = typer.Option([], "--set", help="Override a dotted TOML path."),
    output_root: Optional[Path] = typer.Option(None, "--output-root", help="Override [output].root."),
) -> None:
    """Reconstruct conditional R(T) trajectories from measured power without a global fit."""

    bundle = run_hysteresis_reconstruction(
        _configured(config, set_values, "current"), output_root=output_root, command=_command()
    )
    _announce_bundle(bundle.root)


@analyze_app.command("oscillation-audit")
def analyze_oscillation_audit(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, readable=True),
    set_values: list[str] = typer.Option([], "--set", help="Override a dotted TOML path."),
    output_root: Optional[Path] = typer.Option(None, "--output-root", help="Override [output].root."),
    reuse_numerics: Optional[Path] = typer.Option(
        None,
        "--reuse-numerics",
        exists=True,
        file_okay=False,
        help="Render a new bundle from a completed audit with the identical resolved recipe.",
    ),
) -> None:
    """Audit sustained cycles and map C versus thermal time at selected currents."""

    bundle = run_oscillation_audit(
        _configured(config, set_values, "current"),
        output_root=output_root,
        command=_command(),
        reuse_numerics=reuse_numerics,
    )
    _announce_bundle(bundle.root)


@analyze_app.command("replay-lab")
def analyze_replay_lab(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    set_values: list[str] = typer.Option([], "--set"),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
) -> None:
    """Compare one editable shared model with all measured currents and render GIFs."""
    bundle = run_lab_replay(_configured(config, set_values, "current"), output_root=output_root, command=_command())
    _announce_bundle(bundle.root)


@analyze_app.command("boundary-maps")
def analyze_boundary_maps(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    set_values: list[str] = typer.Option([], "--set"),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
    reuse_exploration: Optional[Path] = typer.Option(None, "--reuse-exploration", exists=True, file_okay=False),
) -> None:
    """Expand a coarse domain, then verify fine transition brackets and open limits."""
    bundle = run_boundary_maps(_configured(config, set_values, "current"),
                              output_root=output_root, command=_command(), reuse_exploration=reuse_exploration)
    _announce_bundle(bundle.root)


@analyze_app.command("validate-trends")
def analyze_validate_trends(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    set_values: list[str] = typer.Option([], "--set"),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
    reuse_integrations: Optional[Path] = typer.Option(None, "--reuse-integrations", exists=True, file_okay=False),
) -> None:
    """Test scaling and one-parameter tendencies with new converged integrations."""
    bundle = run_trend_validation(_configured(config, set_values, "current"),
                                  output_root=output_root, command=_command(),reuse_integrations=reuse_integrations)
    _announce_bundle(bundle.root)


@analyze_app.command("physical-maps")
def analyze_physical_maps(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    set_values: list[str] = typer.Option([], "--set"),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
    reuse_numerics: Optional[Path] = typer.Option(None, "--reuse-numerics", exists=True, file_okay=False),
) -> None:
    """Screen a dense C/Cth/Se domain and render 3D plus fixed-parameter heatmaps."""
    bundle = run_physical_parameter_maps(
        _configured(config, set_values, "current"),
        output_root=output_root,
        command=_command(),
        reuse_numerics=reuse_numerics,
    )
    _announce_bundle(bundle.root)


@analyze_app.command("map-animations")
def analyze_map_animations(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    set_values: list[str] = typer.Option([], "--set"),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Benchmark evenly spaced retained points only"),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
    reuse_media: Optional[Path] = typer.Option(
        None,
        "--reuse-media",
        exists=True,
        file_okay=False,
        help="Rebuild the viewer using unchanged GIFs from a completed matching bundle",
    ),
) -> None:
    """Attach synchronized point GIFs to a preserved map and package an offline ZIP."""
    bundle = run_map_animations(
        _configured(config, set_values, "current"),
        output_root=output_root,
        limit=limit,
        reuse_media=reuse_media,
        command=_command(),
    )
    _announce_bundle(bundle.root)


@analyze_app.command("refresh-maps")
def analyze_refresh_maps(
    source: Optional[Path] = typer.Option(None, "--source", exists=True, file_okay=False),
    all_existing: bool = typer.Option(False, "--all-existing", help="Revise all completed maps once, without resimulation."),
    embed_gifs: Optional[bool] = typer.Option(None, "--embed-gifs/--link-gifs", help="Study defaults embed; dense maps link unchanged media."),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
) -> None:
    """Rebuild historical map viewers with fixed heatmap axes and color scales."""
    bundles = refresh_parameter_maps(source, all_existing=all_existing, output_root=output_root,
                                     embed_gifs=embed_gifs, command=_command())
    for bundle in bundles:
        _announce_bundle(bundle.root)
    if not bundles:
        typer.echo("All completed maps already have current fixed-scale viewers.")


@analyze_app.command("fit-steady")
def analyze_fit_steady(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    set_values: list[str] = typer.Option([], "--set"),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
) -> None:
    """Time-budgeted multi-start fitting of settled oscillation frequency/amplitude."""
    bundle = run_steady_search(_configured(config, set_values, "current"), output_root=output_root, command=_command())
    _announce_bundle(bundle.root)


@analyze_app.command("filament-study")
def analyze_filament_study(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    set_values: list[str] = typer.Option([], "--set"),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
    reuse_searches: Optional[Path] = typer.Option(None, "--reuse-searches", exists=True, file_okay=False,
        help="Reuse completed searches after checking identical numerical code and inputs; rerun verification."),
) -> None:
    """Experimental filament versus original-model fits and timestep checks."""
    from .workflows import run_filament_study

    bundle = run_filament_study(_configured(config, set_values, "current"), output_root=output_root, reuse_searches=reuse_searches, command=_command())
    _announce_bundle(bundle.root)
    typer.echo(f"Results: {bundle.root / 'START_HERE.html'}")


@analyze_app.command("filament-profile")
def analyze_filament_profile(
    config: Optional[Path] = typer.Option(None, "--config", "-c", exists=True, dir_okay=False),
    set_values: list[str] = typer.Option([], "--set"),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
    resume: Optional[Path] = typer.Option(None, "--resume", exists=True, file_okay=False),
    plan: bool = typer.Option(False, "--plan", help="Show the bounded schedule and current split without fitting."),
    prepare: bool = typer.Option(False, "--prepare", help="Freeze executable source and inputs without starting fits."),
) -> None:
    """Run a checkpointed overnight filament-fraction profile and fine-step controls."""
    from .orchestration.filament_profile import plan_filament_profile
    from .workflows import run_filament_profile

    try:
        if resume is not None and (config is not None or set_values or output_root is not None or prepare or plan):
            raise ConfigError("Resume uses the frozen recipe; do not combine it with configuration options")
        cfg = None if resume else _configured(
            config or find_project_root()/"experiments/current/specimen_filament_overnight_20261027.toml",
            set_values, "current")
        if plan:
            typer.echo(json.dumps(plan_filament_profile(cfg), indent=2))
            return
        bundle = run_filament_profile(cfg, resume=resume, output_root=output_root, prepare=prepare, command=_command())
        typer.echo(f"{'Prepared' if prepare else 'Completed'} profile: {bundle.root}")
        typer.echo(f"Progress and results: {bundle.root / 'START_HERE.html'}")
    except (ConfigError, OSError, ValueError, KeyError, InterruptedError) as exc:
        typer.echo(f"Filament profile stopped: {exc}", err=True)
        raise typer.Exit(1) from exc


@analyze_app.command("review-optimizer")
def analyze_review_optimizer(
    audit_run: Path = typer.Option(...,"--audit-run",exists=True,file_okay=False),
    frequency_run: Path = typer.Option(...,"--frequency-run",exists=True,file_okay=False),
    combined_run: Path = typer.Option(...,"--combined-run",exists=True,file_okay=False),
    output_root: Optional[Path] = typer.Option(None,"--output-root"),
) -> None:
    """Review completed finite-difference, gradient and expanded-search experiments."""
    bundle=run_optimizer_review(audit_run,frequency_run,combined_run,output_root=output_root,command=_command())
    _announce_bundle(bundle.root)


@analyze_app.command("audit-optimizer")
def analyze_audit_optimizer(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    set_values: list[str] = typer.Option([], "--set"),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
) -> None:
    """Probe finite differences and timestep stability, then test L-BFGS-B."""
    bundle=run_optimizer_audit(_configured(config,set_values,"current"),output_root=output_root,command=_command())
    _announce_bundle(bundle.root)


@analyze_app.command("compare-steady")
def analyze_compare_steady(
    frequency_run: Path = typer.Option(..., "--frequency-run", exists=True, file_okay=False),
    combined_run: Path = typer.Option(..., "--combined-run", exists=True, file_okay=False),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
) -> None:
    """Compare saved frequency-only and frequency/amplitude fits without rerunning."""
    bundle = run_steady_comparison(frequency_run, combined_run,
                                  output_root=output_root, command=_command())
    _announce_bundle(bundle.root)


@app.command("present")
def present_slides() -> None:
    """Start all linked local viewers and open the presentation PDF."""
    from .interfaces.presentation_server import present
    pdf = present()
    typer.echo(f'Opened {pdf}; all slide viewers are running on this computer.')


@analyze_app.command("serve-presentation")
def analyze_serve_presentation(background: bool = typer.Option(False, '--background')) -> None:
    """Serve the presentation's HTML/GIF viewers on this Mac at port 8767."""
    from .interfaces.presentation_server import create_presentation_server, ensure_running
    if background:
        ensure_running()
        typer.echo('Presentation viewers ready at http://127.0.0.1:8767')
        return
    with create_presentation_server() as server:
        typer.echo(f'Presentation viewers ready at http://127.0.0.1:{server.server_port}')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


@analyze_app.command("export-scope")
def analyze_export_scope(
    source: Path = typer.Option(..., "--source", exists=True, file_okay=False),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
    frames: int = typer.Option(96, "--frames", min=2),
    original_labels_uA: bool = typer.Option(False, "--original-labels-uA"),
) -> None:
    """Export exact saved traces as stacked GIFs named by input current in amperes."""
    bundle = export_saved_scope(
        source, output_root=output_root, frames=frames, original_labels_uA=original_labels_uA, command=_command()
    )
    _announce_bundle(bundle.root)


@analyze_app.command("loss-scopes")
def analyze_loss_scopes(
    config: Path = typer.Option(..., '--config', exists=True, dir_okay=False),
    frequency_run: Path = typer.Option(..., '--frequency-run', exists=True, file_okay=False),
    combined_run: Path = typer.Option(..., '--combined-run', exists=True, file_okay=False),
    frames: int = typer.Option(96, '--frames', min=2),
    output_root: Optional[Path] = typer.Option(None, '--output-root'),
) -> None:
    """Render all inputs for both retained loss candidates with stacked I/V/R(T)."""
    bundle = run_loss_scopes(load_toml(config), frequency_run, combined_run,
        frames=frames, output_root=output_root, command=_command())
    _announce_bundle(bundle.root)


@analyze_app.command("fit-joint")
def analyze_fit_joint(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    set_values: list[str] = typer.Option([], "--set"),
    output_root: Optional[Path] = typer.Option(None, "--output-root"),
) -> None:
    """Jointly fit measured static resistance and persistent waveform features."""
    bundle = run_joint_inference(
        _configured(config, set_values, "current"), output_root=output_root, command=_command()
    )
    _announce_bundle(bundle.root)


@analyze_app.command("fit-waveforms")
def analyze_fit_waveforms(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, readable=True),
    set_values: list[str] = typer.Option([], "--set", help="Override a dotted TOML path."),
    output_root: Optional[Path] = typer.Option(None, "--output-root", help="Override [output].root."),
) -> None:
    """Infer one shared parameter set from all measured current waveforms."""

    configured = _configured(config, set_values, "current")
    bundle = run_waveform_parameter_inference(configured, output_root=output_root, command=_command())
    _announce_bundle(bundle.root)


def _csv_numbers(text: str, option: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"{option} must be a comma-separated numeric list") from exc
    if not values:
        raise typer.BadParameter(f"{option} cannot be empty")
    return values


@analyze_app.command("conductance")
def analyze_conductance(
    data_directory: Path = typer.Option(..., "--data", exists=True, file_okay=False, readable=True),
    resistance_preset: Path = typer.Option(..., "--resistance-preset", exists=True, dir_okay=False, readable=True),
    resistance_bootstrap: Optional[Path] = typer.Option(
        None, "--resistance-bootstrap", exists=True, dir_okay=False, readable=True
    ),
    ambient_K: float = typer.Option(314.4, "--ambient-K"),
    ambient_interval_K: str = typer.Option("314.25,314.55", "--ambient-interval-K"),
    baseline_window_ns: str = typer.Option("-200,-50", "--baseline-window-ns"),
    steady_window_ns: str = typer.Option("100,250", "--steady-window-ns"),
    bootstrap_samples: int = typer.Option(1000, "--bootstrap-samples", min=1),
    block_size: int = typer.Option(10, "--block-size", min=1),
    seed: int = typer.Option(20260817, "--seed"),
    name: str = typer.Option("Environmental thermal-conductance estimate", "--name"),
    output_root: Path = typer.Option(Path("runs/local"), "--output-root"),
) -> None:
    """Estimate environmental thermal conductance from a settled pre-onset trace."""

    ambient_bounds = _csv_numbers(ambient_interval_K, "--ambient-interval-K")
    baseline_bounds = _csv_numbers(baseline_window_ns, "--baseline-window-ns")
    steady_bounds = _csv_numbers(steady_window_ns, "--steady-window-ns")
    if len(ambient_bounds) != 2 or len(baseline_bounds) != 2 or len(steady_bounds) != 2:
        typer.echo("Ambient, baseline, and steady intervals must each contain two values.", err=True)
        raise typer.Exit(2)
    bundle = run_environmental_conductance(
        data_directory,
        name=name,
        resistance_preset=resistance_preset,
        resistance_bootstrap=resistance_bootstrap,
        ambient_temperature_K=ambient_K,
        ambient_interval_K=(ambient_bounds[0], ambient_bounds[1]),
        baseline_window_ns=(baseline_bounds[0], baseline_bounds[1]),
        steady_window_ns=(steady_bounds[0], steady_bounds[1]),
        bootstrap_samples=bootstrap_samples,
        block_size=block_size,
        seed=seed,
        output_root=output_root,
        command=_command(),
    )
    _announce_bundle(bundle.root)


@analyze_app.command("thermal-capacitance")
def analyze_thermal_capacitance(
    data_directory: Path = typer.Option(..., "--data", exists=True, file_okay=False, readable=True),
    resistance_preset: Path = typer.Option(..., "--resistance-preset", exists=True, dir_okay=False, readable=True),
    conductance_mW_per_K: float = typer.Option(..., "--conductance-mW-per-K", min=1e-12),
    resistance_bootstrap: Optional[Path] = typer.Option(
        None, "--resistance-bootstrap", exists=True, dir_okay=False, readable=True
    ),
    conductance_bootstrap: Optional[Path] = typer.Option(
        None, "--conductance-bootstrap", exists=True, dir_okay=False, readable=True
    ),
    ambient_K: float = typer.Option(314.4, "--ambient-K"),
    electrical_capacitance_pF: float = typer.Option(0.0, "--electrical-capacitance-pF", min=0.0),
    selected_drives_mV: str = typer.Option("100,150,200", "--selected-drives-mV"),
    near_transition_check_mV: Optional[float] = typer.Option(250.0, "--near-transition-check-mV"),
    baseline_window_ns: str = typer.Option("-200,-50", "--baseline-window-ns"),
    integration_window_ns: str = typer.Option("-50,80", "--integration-window-ns"),
    fit_window_ns: str = typer.Option("15,35", "--fit-window-ns"),
    smoothing_window: int = typer.Option(9, "--smoothing-window", min=5),
    bootstrap_samples: int = typer.Option(1000, "--bootstrap-samples", min=1),
    fit_window_jitter_ns: int = typer.Option(2, "--fit-window-jitter-ns", min=0),
    seed: int = typer.Option(20260817, "--seed"),
    name: str = typer.Option("Thermal time constant and capacitance estimate", "--name"),
    output_root: Path = typer.Option(Path("runs/local"), "--output-root"),
) -> None:
    """Fit the specimen thermal time constant from nonswitching heating edges."""

    drives = _csv_numbers(selected_drives_mV, "--selected-drives-mV")
    baseline_bounds = _csv_numbers(baseline_window_ns, "--baseline-window-ns")
    integration_bounds = _csv_numbers(integration_window_ns, "--integration-window-ns")
    fit_bounds = _csv_numbers(fit_window_ns, "--fit-window-ns")
    if len(baseline_bounds) != 2 or len(integration_bounds) != 2 or len(fit_bounds) != 2:
        typer.echo("Baseline, integration, and fit windows must each contain two values.", err=True)
        raise typer.Exit(2)
    if smoothing_window % 2 == 0:
        typer.echo("--smoothing-window must be odd.", err=True)
        raise typer.Exit(2)
    bundle = run_thermal_capacitance(
        data_directory,
        name=name,
        resistance_preset=resistance_preset,
        S_e_mW_per_K=conductance_mW_per_K,
        ambient_temperature_K=ambient_K,
        electrical_capacitance_pF=electrical_capacitance_pF,
        selected_drives_mV=tuple(drives),
        near_transition_check_mV=near_transition_check_mV,
        resistance_bootstrap=resistance_bootstrap,
        conductance_bootstrap=conductance_bootstrap,
        baseline_window_ns=(baseline_bounds[0], baseline_bounds[1]),
        integration_window_ns=(integration_bounds[0], integration_bounds[1]),
        fit_window_ns=(fit_bounds[0], fit_bounds[1]),
        smoothing_window=smoothing_window,
        bootstrap_samples=bootstrap_samples,
        fit_window_jitter_ns=fit_window_jitter_ns,
        seed=seed,
        output_root=output_root,
        command=_command(),
    )
    _announce_bundle(bundle.root)


@runs_app.command("list")
def runs_list(
    storage: Optional[str] = typer.Option(None, "--storage", help="Filter local, public, or legacy-local."),
    model: Optional[str] = typer.Option(None, "--model"),
    limit: int = typer.Option(50, "--limit", min=1),
) -> None:
    """List newest archived runs."""

    records = RunRegistry().discover()
    if storage:
        records = [record for record in records if record.storage == storage]
    if model:
        records = [record for record in records if record.model == model]
    typer.echo("ID\tSTATUS\tSTORAGE\tMODEL\tNAME")
    for record in records[:limit]:
        typer.echo(f"{record.id}\t{record.status}\t{record.storage}\t{record.model}\t{record.name}")


@runs_app.command("show")
def runs_show(run_id: str = typer.Argument(..., help="Run identifier from `neuristor runs list`.")) -> None:
    """Print a normalized run manifest."""

    try:
        record = RunRegistry().get(run_id)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    payload = dict(record.manifest)
    payload["resolved_root"] = str(record.root)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@runs_app.command("index")
def runs_index(
    output_directory: Optional[Path] = typer.Option(None, "--output-directory"),
) -> None:
    """Refresh a lightweight HTML/CSV catalog; never copy or delete evidence."""
    from .orchestration.artifact_catalog import build_artifact_catalog

    typer.echo(json.dumps(build_artifact_catalog(output_directory=output_directory), indent=2))


@runs_app.command("publish")
def runs_publish(run_id: str = typer.Argument(...)) -> None:
    """Copy one local bundle into the Git-trackable public archive."""

    try:
        record = RunRegistry().publish(run_id)
    except (KeyError, FileExistsError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Published: {record.root}")


@runs_app.command("manage")
def runs_manage(port: int = typer.Option(0, "--port", min=0, max=65535)) -> None:
    """Open the local artifact manager with recoverable Delete/Restore controls."""
    from .interfaces.run_manager import create_manager

    with create_manager(port=port) as server:
        typer.echo(f"Run manager: http://127.0.0.1:{server.server_port}/runs/catalog/index.html")
        typer.echo("Delete moves local runs to recoverable trash. Published/running runs are protected.")
        typer.echo("Keep this terminal running; press Ctrl+C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


@runs_app.command("delete")
def runs_delete(run_id: str = typer.Argument(...), yes: bool = typer.Option(False, "--yes")) -> None:
    """Move a completed/failed local run to recoverable project trash."""
    from .orchestration.artifact_catalog import build_artifact_catalog

    registry = RunRegistry()
    try:
        record = registry.get(run_id)
        reason = registry.trash_reason(record)
        if reason:
            raise ValueError(reason)
        if not yes:
            typer.confirm(f"Move '{record.name}' ({record.root}) to recoverable trash?", abort=True)
        result = registry.trash(run_id)
    except (KeyError, ValueError, OSError) as exc:
        typer.echo(f"Not deleted: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Moved to recoverable trash. Restore with: neuristor runs restore {result['trash_id']}")
    build_artifact_catalog()


@runs_app.command("restore")
def runs_restore(trash_id: str = typer.Argument(...)) -> None:
    """Restore a run from trash without overwriting any existing folder."""
    from .orchestration.artifact_catalog import build_artifact_catalog

    try:
        result = RunRegistry().restore(trash_id)
    except (KeyError, ValueError, OSError) as exc:
        typer.echo(f"Not restored: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Restored: {result['original_path']}")
    build_artifact_catalog()


@runs_app.command("trash")
def runs_trash() -> None:
    """List recoverable deleted runs and their restore IDs."""
    typer.echo(json.dumps(RunRegistry().trashed(), indent=2))


@runs_app.command("visualize")
def runs_visualize(
    run_id: str = typer.Argument(..., help="Current-drive run identifier."),
    output_directory: Optional[Path] = typer.Option(None, "--output-directory", "-o"),
    frames: int = typer.Option(96, "--frames", min=2, help="Number of GIF frames."),
    duration_s: float = typer.Option(10.0, "--duration-s", min=0.1, help="GIF duration in seconds."),
) -> None:
    """Create an R(T) trajectory figure and synchronized current/voltage GIF."""

    try:
        record = RunRegistry().get(run_id)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    trace_path = record.root / "traces.csv"
    if record.model != "current" or not trace_path.is_file():
        typer.echo("This command requires a current-drive run with traces.csv.", err=True)
        raise typer.Exit(2)
    output = output_directory or (find_project_root() / "runs/scratch" / "run_visuals" / run_id)
    output = output.expanduser().resolve()
    frame = pd.read_csv(trace_path)
    peak_current_uA = float(frame["current_uA"].max()) if "current_uA" in frame else 0.0
    try:
        static_path = plot_resistance_temperature_trajectory(
            frame,
            output / "resistance_temperature_trajectory.png",
            title=f"Equivalent resistance-temperature trajectory at {peak_current_uA:g} uA",
        )
        gif_path = animate_current_resistance_temperature(
            frame,
            output / "current_voltage_rt_evolution.gif",
            title=f"{peak_current_uA:g} uA current-drive electrothermal evolution",
            frame_count=frames,
            duration_s=duration_s,
        )
    except ValueError as exc:
        typer.echo(f"Trace error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Static figure: {static_path}")
    typer.echo(f"Animation: {gif_path}")


@app.command("validate")
def validate(
    publication: bool = typer.Option(False, "--publication", help="Also check exact staged Git blobs for private PDFs and oversized files."),
    distribution: Optional[Path] = typer.Option(None, "--distribution", exists=True, dir_okay=False, help="Inspect one proposed release file without uploading it."),
) -> None:
    """Validate experiment recipes and current/historical run manifests."""

    report = validate_repository()
    if publication or distribution is not None:
        from .analysis.distribution import approved_pdf_hashes, validate_distribution_file, validate_publication_index

        root = find_project_root()
        try:
            if publication:
                report.errors.extend(validate_publication_index(root))
            if distribution is not None:
                report.errors.extend(validate_distribution_file(distribution, approved_pdf_hashes(root)))
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            report.errors.append(f"Publication inspection failed: {exc}")
    typer.echo(
        f"Checked {report.checked_configs} experiment configs, {report.checked_runs} archived runs, "
        f"and {report.checked_evidence} paper evidence entries."
    )
    for warning in report.warnings:
        typer.echo(f"WARNING: {warning}")
    for error in report.errors:
        typer.echo(f"ERROR: {error}", err=True)
    if not report.ok:
        raise typer.Exit(1)
    typer.echo("Validation passed.")


def main() -> None:
    """Console-script entry point."""

    app()


if __name__ == "__main__":
    main()
