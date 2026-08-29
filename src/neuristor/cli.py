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
from .visualization import animate_current_resistance_temperature, plot_resistance_temperature_trajectory
from .workflows import (
    run_environmental_conductance,
    run_lab_analysis,
    run_model_validation,
    run_waveform_parameter_inference,
    run_resistance_fit,
    run_simulation,
    run_sweep,
    run_thermal_capacitance,
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
app.add_typer(simulate_app, name="simulate")
app.add_typer(sweep_app, name="sweep")
app.add_typer(fit_app, name="fit")
app.add_typer(analyze_app, name="analyze")
app.add_typer(runs_app, name="runs")


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
    output_root: Path = typer.Option(Path("runs"), "--output-root"),
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
    output_root: Path = typer.Option(Path("runs"), "--output-root"),
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
    output_root: Path = typer.Option(Path("runs"), "--output-root"),
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
    electrical_capacitance_pF: float = typer.Option(
        0.0, "--electrical-capacitance-pF", min=0.0
    ),
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
    output_root: Path = typer.Option(Path("runs"), "--output-root"),
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


@runs_app.command("publish")
def runs_publish(run_id: str = typer.Argument(...)) -> None:
    """Copy one local bundle into the Git-trackable public archive."""

    try:
        record = RunRegistry().publish(run_id)
    except (KeyError, FileExistsError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Published: {record.root}")


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
    output = output_directory or (find_project_root() / "outputs" / "run_visuals" / run_id)
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
def validate() -> None:
    """Validate experiment recipes and current/historical run manifests."""

    report = validate_repository()
    typer.echo(f"Checked {report.checked_configs} experiment configs and {report.checked_runs} archived runs.")
    for warning in report.warnings:
        typer.echo(f"WARNING: {warning}")
    for error in report.errors:
        typer.echo(f"ERROR: {error}", err=True)
    if not report.ok:
        raise typer.Exit(1)
    typer.echo("Validation passed.")


@app.command("dashboard")
def dashboard(
    port: int = typer.Option(8501, "--port", min=1, max=65_535),
) -> None:
    """Open the read-only Streamlit archive and comparison dashboard."""

    root = find_project_root()
    command = [sys.executable, "-m", "streamlit", "run", str(root / "app.py"), "--server.port", str(port)]
    raise typer.Exit(subprocess.run(command, cwd=root, check=False).returncode)


def main() -> None:
    """Console-script entry point."""

    app()


if __name__ == "__main__":
    main()
