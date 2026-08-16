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
from .workflows import run_lab_analysis, run_lab_estimates, run_resistance_fit, run_simulation, run_sweep


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
    output_root: Path = typer.Option(Path("runs"), "--output-root"),
) -> None:
    """Fit Yuanhang resistance/hysteresis parameters to measured R(T)."""

    bundle = run_resistance_fit(
        data,
        name=name,
        seed=seed,
        random_iters=random_iters,
        local_passes=local_passes,
        output_root=output_root,
        command=_command(),
    )
    _announce_bundle(bundle.root)


@analyze_app.command("lab")
def analyze_lab(
    image_directory: Path = typer.Option(..., "--images", exists=True, file_okay=False, readable=True),
    name: str = typer.Option("Digitized laboratory current sweep", "--name"),
    output_root: Path = typer.Option(Path("runs"), "--output-root"),
) -> None:
    """Digitize current/voltage traces from the laboratory PNG sequence."""

    bundle = run_lab_analysis(image_directory, name=name, output_root=output_root, command=_command())
    _announce_bundle(bundle.root)


def _csv_numbers(text: str, option: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"{option} must be a comma-separated numeric list") from exc
    if not values:
        raise typer.BadParameter(f"{option} cannot be empty")
    return values


@analyze_app.command("estimates")
def analyze_estimates(
    resistance_preset: Path = typer.Option(..., "--resistance-preset", exists=True, dir_okay=False, readable=True),
    summary: Optional[Path] = typer.Option(None, "--summary", exists=True, dir_okay=False, readable=True),
    image_directory: Optional[Path] = typer.Option(None, "--images", exists=True, file_okay=False, readable=True),
    ambient_K: str = typer.Option("298,325,330,333", "--ambient-K"),
    thermal_times_ns: str = typer.Option("10,20,50,100", "--thermal-times-ns"),
    ripple_threshold_mV: float = typer.Option(20.0, "--ripple-threshold-mV", min=0.0),
    name: str = typer.Option("Lab current-trace parameter estimates", "--name"),
    output_root: Path = typer.Option(Path("runs"), "--output-root"),
) -> None:
    """Estimate electrical C and scenario-dependent thermal parameters."""

    if (summary is None) == (image_directory is None):
        typer.echo("Provide exactly one of --summary or --images.", err=True)
        raise typer.Exit(2)
    bundle = run_lab_estimates(
        name=name,
        resistance_preset=resistance_preset,
        summary_path=summary,
        image_directory=image_directory,
        ambient_temperatures_K=_csv_numbers(ambient_K, "--ambient-K"),
        thermal_times_ns=_csv_numbers(thermal_times_ns, "--thermal-times-ns"),
        ripple_threshold_mV=ripple_threshold_mV,
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
