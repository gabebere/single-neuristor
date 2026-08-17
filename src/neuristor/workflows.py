"""High-level experiment workflows shared by the CLI and archive dashboard.

The numerical modules remain the source of truth for the physics.  This module
only translates documented user units, invokes those solvers, and writes a
portable run bundle.  Keeping orchestration here prevents command-line and UI
code from quietly developing different scientific behavior.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import ConfigError, apply_overrides, deep_set, load_toml, resolved_copy, source_directory, validate_config
from .current_drive_sim import CurrentDriveParams, current_drive_operating_estimates, simulate_current_step
from .experimental_waveforms import load_converted_sweep
from .metrics import current_run_metrics, voltage_run_metrics
from .lab_estimates import estimate_lab_parameters
from .model import YuanhangCircuitParams, YuanhangResistParams, series_first, simulate_yuanhang
from .resistance_custom_analysis import (
    fit_major_loop_resistance_params,
    fit_resistance_params,
    is_major_loop_temperature_trace,
    load_experimental_rt,
)
from .runs import RunBundle
from .visualization import (
    plot_current_run,
    plot_lab_summary,
    plot_lab_parameter_estimates,
    plot_resistance_fit,
    plot_sweep_summary,
    plot_voltage_run,
    plot_voltage_floor_comparison,
)


@dataclass(frozen=True)
class SimulationResult:
    """In-memory result used by single-run and parameter-sweep workflows."""

    frame: pd.DataFrame
    metrics: dict[str, Any]
    diagnostics: dict[str, Any]


def _table(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"Expected [{name}] to be a TOML table")
    return value


def _optional_us(table: Mapping[str, Any], key: str) -> float | None:
    value = table.get(key)
    return None if value is None else float(value) * 1e-6


def resistance_from_config(config: Mapping[str, Any]) -> tuple[YuanhangResistParams, str, str]:
    """Resolve a built-in, JSON-preset, or inline resistance specification."""

    resistance = _table(config, "resistance")
    preset = str(resistance.get("preset", "yuanhang")).strip()
    start_branch = str(resistance.get("start_branch", "insulator")).lower()
    if start_branch not in {"insulator", "metal"}:
        raise ConfigError("resistance.start_branch must be 'insulator' or 'metal'")

    raw: dict[str, Any] = {}
    provenance = "built-in Yuanhang Zhang parameters"
    if preset.lower() not in {"", "yuanhang", "reference", "yuanhang-reference"}:
        path = Path(preset).expanduser()
        if not path.is_absolute():
            path = source_directory(config) / path
        path = path.resolve()
        if not path.is_file():
            raise ConfigError(f"Resistance preset does not exist: {path}")
        payload = json.loads(path.read_text())
        candidate = payload.get("resist_params", payload.get("parameters", payload))
        if not isinstance(candidate, Mapping):
            raise ConfigError(f"Resistance preset has no parameter mapping: {path}")
        raw.update(candidate)
        start_branch = str(resistance.get("start_branch", payload.get("start_branch", start_branch))).lower()
        provenance = str(path)

    inline = resistance.get("parameters", {})
    if inline:
        if not isinstance(inline, Mapping):
            raise ConfigError("resistance.parameters must be a TOML table")
        raw.update(inline)
        provenance += " with inline overrides"
    known = {field.name for field in dataclasses.fields(YuanhangResistParams)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"Unknown resistance parameter(s): {', '.join(unknown)}")
    params = YuanhangResistParams(**{key: float(value) for key, value in raw.items()})
    return params, start_branch, provenance


def evaluate_simulation(config: Mapping[str, Any]) -> SimulationResult:
    """Run one already-validated simulation without creating files."""

    validate_config(config)
    model = str(config["model"]).lower()
    return _evaluate_current(config) if model == "current" else _evaluate_voltage(config)


def _analysis_window(config: Mapping[str, Any]) -> tuple[float | None, float | None]:
    analysis = _table(config, "analysis")
    return _optional_us(analysis, "start_us"), _optional_us(analysis, "stop_us")


def _evaluate_current(config: Mapping[str, Any]) -> SimulationResult:
    time = _table(config, "time")
    input_table = _table(config, "input")
    electrical = _table(config, "electrical")
    thermal = _table(config, "thermal")
    initial = _table(config, "initial")
    resistance, start_branch, resistance_source = resistance_from_config(config)
    amplitude_uA = float(input_table["amplitude_uA"])
    params = CurrentDriveParams(
        dt_s=float(time["dt_ns"]) * 1e-9,
        t_end_s=float(time["duration_us"]) * 1e-6,
        t_pre_s=float(time.get("pre_us", 0.0)) * 1e-6,
        pulse_on_s=float(input_table.get("on_us", 0.0)) * 1e-6,
        pulse_off_s=_optional_us(input_table, "off_us"),
        V_init_V=float(initial.get("voltage_V", 0.0)),
        T0_K=float(thermal.get("T0_K", 325.0)),
        T_init_K=float(initial.get("temperature_K", thermal.get("T0_K", 325.0) - 0.1)),
        C_F=float(electrical["C_pF"]) * 1e-12,
        C_th_J_per_K=float(thermal["C_th_pJ_per_K"]) * 1e-12,
        S_e_W_per_K=float(thermal["S_e_mW_per_K"]) * 1e-3,
        sigma_W_sqrt_s=float(thermal.get("noise_W_sqrt_s", 0.0)),
        resist_params=resistance,
        start_branch=start_branch,
    )
    raw = simulate_current_step(amplitude_uA, params=params, seed=int(config.get("seed", 0)))
    start_s, stop_s = _analysis_window(config)
    metrics = current_run_metrics(raw, analysis_start_s=start_s, analysis_stop_s=stop_s)
    operating = current_drive_operating_estimates(params, I_uA=amplitude_uA)
    frame = pd.DataFrame(
        {
            "time_us": raw["t"] * 1e6,
            "current_uA": raw["I_in"] * 1e6,
            # This is the instantaneous fixed point of the electrical equation
            # if VO2 is fully metallic: V = I(t) R_m.  Archiving it beside the
            # simulated voltage keeps the plotted reference trace reproducible.
            "metallic_voltage_floor_V": raw["I_in"] * resistance.Rm,
            "voltage_V": raw["V_vo2"],
            "temperature_K": raw["T"],
            "resistance_ohm": raw["R"],
            "power_uW": raw["P"] * 1e6,
            "semiconducting_fraction": raw["g_eq"],
        }
    )
    diagnostics: dict[str, Any] = {
        "resistance_source": resistance_source,
        "start_branch": start_branch,
        "metallic_resistance_ohm": resistance.Rm,
        "metallic_voltage_floor_V": operating["V_metal_floor_V"],
        "electrical_metal_time_constant_ns": operating["tau_metal_s"] * 1e9,
        "thermal_time_constant_us": operating["tau_thermal_s"] * 1e6,
        "electrical_to_thermal_time_constant_ratio": operating["tau_metal_over_tau_thermal"],
        "thermal_only_lower_current_uA": operating["thermal_only_lower_current_uA"],
        "thermal_only_upper_current_uA": operating["thermal_only_upper_current_uA"],
        "thermal_only_window_exists": bool(operating["thermal_only_window_exists"]),
        "resistance_calibration_min_K": float(resistance.T_min_K),
        "resistance_calibration_max_K": float(resistance.T_max_K),
        "temperature_outside_resistance_calibration": bool(
            float(np.min(raw["T"])) < resistance.T_min_K or float(np.max(raw["T"])) > resistance.T_max_K
        ),
    }
    metrics.update(diagnostics)
    return SimulationResult(frame=frame, metrics=metrics, diagnostics=diagnostics)


def _evaluate_voltage(config: Mapping[str, Any]) -> SimulationResult:
    time = _table(config, "time")
    input_table = _table(config, "input")
    electrical = _table(config, "electrical")
    thermal = _table(config, "thermal")
    initial = _table(config, "initial")
    resistance, start_branch, resistance_source = resistance_from_config(config)
    source_voltage = float(input_table["amplitude_V"])
    circuit = YuanhangCircuitParams(
        R_series_kohm=float(electrical["R_series_kohm"]),
        C_par_pF=float(electrical["C_pF"]),
        Cth_mW_ns_per_K=float(thermal["C_th_pJ_per_K"]),
        Sth_mW_per_K=float(thermal["S_e_mW_per_K"]),
        couple_factor=float(thermal.get("couple_factor", 0.0)),
        Cth_factor=float(thermal.get("C_th_factor", 1.0)),
        noise_strength=float(thermal.get("noise_K_per_ns", 0.0)),
        T_base_K=float(thermal.get("T0_K", 325.0)),
    )
    raw = simulate_yuanhang(
        Vin=source_voltage,
        t_end=float(time["duration_us"]) * 1e-6,
        dt=float(time["dt_ns"]) * 1e-9,
        resist_params=resistance,
        circuit_params=circuit,
        init={
            "Vn": float(initial.get("voltage_V", 0.0)),
            "T_K": float(initial.get("temperature_K", thermal.get("T0_K", 325.0))),
        },
        start_branch=start_branch,
        noise_seed=int(config.get("seed", 0)),
    )
    time_s = np.asarray(raw["time_s"], dtype=float)
    voltage = np.asarray(series_first(raw["V_node"]), dtype=float)
    current = np.asarray(series_first(raw["I_vo2"]), dtype=float)
    load_current = np.asarray(series_first(raw["I_load"]), dtype=float)
    temperature = np.asarray(series_first(raw["T_K"]), dtype=float)
    resistance_ohm = np.asarray(series_first(raw["R_vo2"]), dtype=float)
    fraction = np.asarray(series_first(raw["g"]), dtype=float)
    start_s, stop_s = _analysis_window(config)
    metrics = voltage_run_metrics(
        time_s,
        voltage,
        current,
        temperature,
        resistance_ohm,
        analysis_start_s=start_s,
        analysis_stop_s=stop_s,
    )
    diagnostics = {
        "resistance_source": resistance_source,
        "start_branch": start_branch,
        "electrical_series_time_constant_us": circuit.R_series_ohm * circuit.C_par_F * 1e6,
        "thermal_time_constant_us": (
            circuit.Cth_J_per_K / circuit.S_env_W_per_K * 1e6 if circuit.S_env_W_per_K > 0.0 else float("inf")
        ),
        "resistance_calibration_min_K": float(resistance.T_min_K),
        "resistance_calibration_max_K": float(resistance.T_max_K),
        "temperature_outside_resistance_calibration": bool(
            float(np.min(temperature)) < resistance.T_min_K or float(np.max(temperature)) > resistance.T_max_K
        ),
    }
    metrics.update(diagnostics)
    frame = pd.DataFrame(
        {
            "time_us": time_s * 1e6,
            "source_voltage_V": np.full(time_s.size, source_voltage),
            "voltage_V": voltage,
            "current_mA": current * 1e3,
            "load_current_mA": load_current * 1e3,
            "temperature_K": temperature,
            "resistance_ohm": resistance_ohm,
            "semiconducting_fraction": fraction,
        }
    )
    return SimulationResult(frame=frame, metrics=metrics, diagnostics=diagnostics)


def run_simulation(
    config: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    command: str = "neuristor simulate",
) -> RunBundle:
    """Execute one simulation and persist its complete portable run bundle."""

    validate_config(config)
    output = output_root or _table(config, "output").get("root", "runs")
    bundle = RunBundle.create(
        name=str(config["name"]),
        model=str(config["model"]),
        kind="simulation",
        config=resolved_copy(config),
        output_root=output,
        command=command,
    )
    try:
        result = evaluate_simulation(config)
        traces_path = bundle.add_artifact("traces.csv", label="Simulation traces", media_type="text/csv")
        result.frame.to_csv(traces_path, index=False)
        bundle.write_json("metrics.json", result.metrics, label="Simulation metrics")
        figure_path = bundle.add_artifact("figures/overview.png", label="Overview figure", media_type="image/png")
        if str(config["model"]) == "current":
            plot_current_run(result.frame, figure_path, title=str(config["name"]))
        else:
            plot_voltage_run(result.frame, figure_path, title=str(config["name"]))
        bundle.write_text("report.md", _simulation_report(config, result), label="Scientific report")
        bundle.complete(summary=result.metrics)
    except BaseException as exc:
        bundle.fail(exc)
        raise
    return bundle


def _simulation_report(config: Mapping[str, Any], result: SimulationResult) -> str:
    metrics = result.metrics
    lines = [
        f"# {config['name']}",
        "",
        f"- Model: `{config['model']}`",
        f"- Oscillatory: **{bool(metrics.get('oscillatory', False))}**",
        f"- Estimated frequency: **{float(metrics.get('frequency_MHz', 0.0)):.6g} MHz**",
        "",
        "## Interpretation",
        "",
    ]
    if str(config["model"]) == "current":
        lines.extend(
            [
                "The ideal-current model obeys `C dV/dt = I - V/R(T)`. In its fast or zero-capacitance limit, "
                "the voltage is `V = I R(T)`.",
                "",
                f"For this run, the metallic resistance is **{result.diagnostics['metallic_resistance_ohm']:.6g} ohm**, "
                f"so the predicted metallic voltage floor is **{result.diagnostics['metallic_voltage_floor_V']:.6g} V**. "
                "A low valley is therefore a physical consequence of the chosen metallic resistance; capacitance "
                "changes how rapidly that valley is approached, not its steady-state value.",
                "",
                f"- Electrical metallic time constant: **{result.diagnostics['electrical_metal_time_constant_ns']:.6g} ns**",
                f"- Thermal time constant: **{result.diagnostics['thermal_time_constant_us']:.6g} us**",
            ]
        )
    else:
        lines.extend(
            [
                "The voltage-source model includes the external series resistor and parasitic capacitance. "
                "Its electrical charging time competes with VO2 heating and cooling.",
                "",
                f"- Series RC time constant: **{result.diagnostics['electrical_series_time_constant_us']:.6g} us**",
                f"- Thermal time constant: **{result.diagnostics['thermal_time_constant_us']:.6g} us**",
            ]
        )
    if bool(result.diagnostics.get("temperature_outside_resistance_calibration", False)):
        lines.extend(
            [
                "",
                "> **Validity warning:** Part of the temperature trajectory lies outside the configured R(T) "
                f"calibration range ({result.diagnostics['resistance_calibration_min_K']:.3g}–"
                f"{result.diagnostics['resistance_calibration_max_K']:.3g} K). The resistance law is clamped there.",
            ]
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "Use the command stored in `run.json`; all resolved inputs are in `resolved_config.json`.",
        ]
    )
    return "\n".join(lines)


def _axis_values(axis: Mapping[str, Any]) -> list[Any]:
    values = axis.get("values")
    if isinstance(values, list) and values:
        return values
    start, stop, step = float(axis["start"]), float(axis["stop"]), float(axis["step"])
    if step == 0.0 or (stop - start) * step < 0.0:
        raise ConfigError(f"Invalid sweep range for {axis['path']}: start={start}, stop={stop}, step={step}")
    count = int(np.floor((stop - start) / step + 1e-12)) + 1
    return [start + index * step for index in range(count)]


def run_sweep(
    config: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    command: str = "neuristor sweep run",
) -> RunBundle:
    """Evaluate a Cartesian parameter sweep using a simulation TOML as its base."""

    sweep = _table(config, "sweep")
    base_path = Path(str(sweep["base_config"])).expanduser()
    if not base_path.is_absolute():
        base_path = source_directory(config) / base_path
    base = load_toml(base_path)
    if str(base.get("kind", "simulation")) != "simulation":
        raise ConfigError("sweep.base_config must describe kind='simulation'")
    if str(base["model"]) != str(config["model"]):
        raise ConfigError("Sweep model must match the base simulation model")
    base_overrides = sweep.get("base_overrides", {})
    if base_overrides:
        if not isinstance(base_overrides, Mapping):
            raise ConfigError("sweep.base_overrides must be a mapping of dotted paths to values")
        base = apply_overrides(base, [f"{path}={json.dumps(value)}" for path, value in base_overrides.items()])

    axes = list(sweep["axes"])
    axis_paths = [str(axis["path"]) for axis in axes]
    value_sets = [_axis_values(axis) for axis in axes]
    points = list(itertools.product(*value_sets))
    max_points = int(sweep.get("max_points", 10_000))
    if len(points) > max_points:
        raise ConfigError(f"Sweep has {len(points)} points, exceeding sweep.max_points={max_points}")

    output = output_root or _table(config, "output").get("root", "runs")
    bundle = RunBundle.create(
        name=str(config["name"]),
        model=str(config["model"]),
        kind="sweep",
        config=resolved_copy(config),
        output_root=output,
        command=command,
    )
    rows: list[dict[str, Any]] = []
    try:
        for index, values in enumerate(points):
            point_config = copy.deepcopy(base)
            for path, value in zip(axis_paths, values):
                deep_set(point_config, path, value)
            validate_config(point_config)
            result = evaluate_simulation(point_config)
            row: dict[str, Any] = {path: value for path, value in zip(axis_paths, values)}
            row.update(
                {key: value for key, value in result.metrics.items() if isinstance(value, (str, int, float, bool))}
            )
            row["point_index"] = index
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary_path = bundle.add_artifact("sweep.csv", label="Sweep metrics", media_type="text/csv")
        summary.to_csv(summary_path, index=False)
        bundle.write_json(
            "metrics.json",
            {
                "points": len(summary),
                "oscillatory_points": int(
                    summary.get("oscillatory", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
                ),
                "maximum_frequency_MHz": float(summary["frequency_MHz"].max()) if "frequency_MHz" in summary else 0.0,
                "axes": axis_paths,
            },
            label="Sweep summary metrics",
        )
        if len(axis_paths) <= 3:
            figure = bundle.add_artifact("figures/sweep.png", label="Sweep figure", media_type="image/png")
            plot_sweep_summary(summary, axis_paths, figure, title=str(config["name"]))
        report = (
            f"# {config['name']}\n\n"
            f"Evaluated **{len(summary)}** Cartesian parameter combinations using `{base_path}`.\n\n"
            "Each row in `sweep.csv` contains the exact axis values and canonical oscillation metrics.\n"
        )
        bundle.write_text("report.md", report, label="Scientific report")
        bundle.complete(
            summary={
                "points": len(summary),
                "oscillatory_points": int(
                    summary.get("oscillatory", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
                ),
                "maximum_frequency_MHz": float(summary["frequency_MHz"].max()) if "frequency_MHz" in summary else 0.0,
            }
        )
    except BaseException as exc:
        bundle.fail(exc)
        raise
    return bundle


def run_resistance_fit(
    data_path: str | Path,
    *,
    name: str,
    seed: int = 42,
    random_iters: int = 12_000,
    local_passes: int = 180,
    method: str = "auto",
    bootstrap_samples: int = 500,
    output_root: str | Path = "runs",
    command: str = "neuristor fit resistance",
) -> RunBundle:
    """Fit R(T) data and archive the preset, metrics, source data, and overlay."""

    source = Path(data_path).expanduser().resolve()
    config = {
        "schema_version": 1,
        "name": name,
        "kind": "fit",
        "model": "resistance",
        "data": str(source),
        "seed": seed,
        "random_iters": random_iters,
        "local_passes": local_passes,
        "method": method,
        "bootstrap_samples": bootstrap_samples,
    }
    bundle = RunBundle.create(
        name=name,
        model="resistance",
        kind="fit",
        config=config,
        output_root=output_root,
        command=command,
    )
    try:
        data = load_experimental_rt(source)
        requested_method = str(method).strip().lower()
        if requested_method not in {"auto", "major-loop", "stateful"}:
            raise ValueError("Resistance fit method must be auto, major-loop, or stateful")
        use_major_loop = requested_method == "major-loop" or (
            requested_method == "auto"
            and is_major_loop_temperature_trace(data["Temperature"].to_numpy(dtype=float))
        )
        bootstrap = pd.DataFrame()
        if use_major_loop:
            result, prediction, bootstrap = fit_major_loop_resistance_params(
                data,
                seed=seed,
                bootstrap_samples=bootstrap_samples,
            )
        else:
            result, prediction = fit_resistance_params(
                data,
                seed=seed,
                random_iters=random_iters,
                local_passes=local_passes,
            )
        result.source_data = str(source)
        measured_path = bundle.add_artifact("measured.csv", label="Normalized measured R(T)", media_type="text/csv")
        measured = data.copy()
        measured["model_resistance_ohm"] = prediction
        measured["log10_residual"] = np.log10(np.maximum(prediction, 1e-12)) - np.log10(
            np.maximum(measured["Resistance"].to_numpy(dtype=float), 1e-12)
        )
        measured.to_csv(measured_path, index=False)
        payload = result.to_jsonable()
        bundle.write_json("resistance_preset.json", payload, label="Fitted resistance preset")
        bundle.write_json("metrics.json", payload["fit_metrics"], label="Fit metrics")
        parameter_rows = []
        for parameter, estimate in payload["resist_params"].items():
            ci = payload.get("parameter_ci95", {}).get(parameter, {})
            alias = {
                "R0": "R0_ohm",
                "Ea_over_k": "Ea_over_k_K",
                "Rm0": "Rm_ohm",
                "w": "w_K",
                "Tc_K": "Tc_K",
                "beta": "beta_per_K",
            }.get(parameter)
            if alias:
                ci = payload.get("parameter_ci95", {}).get(alias, ci)
            parameter_rows.append(
                {
                    "parameter": parameter,
                    "estimate": estimate,
                    "ci95_lower": ci.get("lower"),
                    "ci95_upper": ci.get("upper"),
                    "status": "fitted" if alias else "fixed or conventional",
                }
            )
        pd.DataFrame(parameter_rows).to_csv(
            bundle.add_artifact("parameter_summary.csv", label="Parameter estimates", media_type="text/csv"),
            index=False,
        )
        if not bootstrap.empty:
            bootstrap.to_csv(
                bundle.add_artifact(
                    "parameter_bootstrap.csv",
                    label="Block-bootstrap parameter samples",
                    media_type="text/csv",
                ),
                index=False,
            )
        figure = bundle.add_artifact("figures/resistance_fit.png", label="Resistance fit", media_type="image/png")
        plot_resistance_fit(data, prediction, figure)
        bundle.write_text(
            "report.md",
            f"# {name}\n\nFitted {len(data)} R(T) samples from `{source}`.\n\n"
            f"Method: **{result.fit_method}**. Overall log10 RMSE: **{result.rmse_log10:.6g}**. "
            f"Start branch: **{result.start_branch}**.\n\n"
            "Gamma is fixed to the Yuanhang value for major-loop data because it is a minor-loop parameter.\n",
            label="Scientific report",
        )
        bundle.complete(summary={**payload["fit_metrics"], "fit_method": result.fit_method})
    except BaseException as exc:
        bundle.fail(exc)
        raise
    return bundle


def run_lab_analysis(
    data_directory: str | Path,
    *,
    name: str,
    output_root: str | Path = "runs",
    command: str = "neuristor analyze lab",
) -> RunBundle:
    """Archive and summarize the professor-supplied numerical waveforms."""

    data_dir = Path(data_directory).expanduser().resolve()
    config = {
        "schema_version": 1,
        "name": name,
        "kind": "analysis",
        "model": "lab-current-waveforms",
        "data_directory": str(data_dir),
        "analysis_windows_ns": {
            "baseline": [-200.0, -50.0],
            "edge": [0.0, 30.0],
            "plateau": [50.0, 250.0],
        },
    }
    bundle = RunBundle.create(
        name=name,
        model="lab-current-waveforms",
        kind="analysis",
        config=config,
        output_root=output_root,
        command=command,
    )
    try:
        trace_frame, summary = load_converted_sweep(data_dir)
        summary.to_csv(
            bundle.add_artifact("summary.csv", label="Measured sweep summary", media_type="text/csv"), index=False
        )
        trace_frame.to_csv(
            bundle.add_artifact("traces.csv", label="Numerical oscilloscope traces", media_type="text/csv"), index=False
        )
        figure = bundle.add_artifact("figures/lab_summary.png", label="Lab sweep summary", media_type="image/png")
        plot_lab_summary(summary, figure)
        oscillatory = summary[summary["oscillation_detected"].astype(bool)]
        metrics = {
            "waveforms": len(summary),
            "samples": len(trace_frame),
            "current_min_uA": float(summary["current_plateau_uA"].min()),
            "current_max_uA": float(summary["current_plateau_uA"].max()),
            "oscillation_current_min_uA": float(oscillatory["current_plateau_uA"].min()),
            "oscillation_current_max_uA": float(oscillatory["current_plateau_uA"].max()),
            "oscillation_frequency_min_MHz": float(oscillatory["oscillation_frequency_MHz"].min()),
            "oscillation_frequency_max_MHz": float(oscillatory["oscillation_frequency_MHz"].max()),
        }
        bundle.write_json("metrics.json", metrics, label="Waveform metrics")
        bundle.write_text(
            "report.md",
            f"# {name}\n\nLoaded **{len(summary)}** numerical traces from `{data_dir}`. "
            "No values were recovered from images.\n\n"
            f"Coherent oscillations are detected from **{metrics['oscillation_current_min_uA']:.3g}** to "
            f"**{metrics['oscillation_current_max_uA']:.3g} uA**, with measured frequencies from "
            f"**{metrics['oscillation_frequency_min_MHz']:.3g}** to "
            f"**{metrics['oscillation_frequency_max_MHz']:.3g} MHz**.\n\n"
            "These electrical traces can constrain gamma only through a dynamic model. Given independently "
            "calibrated C, C_th, S_e, and T0, the resistive current is I_R=I_in-C dV/dt, power is P=V I_R, "
            "and the thermal equation reconstructs T(t). Gamma can then be fitted to the repeated minor-loop "
            "reversals. Without those thermal constraints, gamma is correlated with the latent temperature "
            "trajectory and is not independently identified.\n",
            label="Scientific report",
        )
        bundle.complete(summary=metrics)
    except BaseException as exc:
        bundle.fail(exc)
        raise
    return bundle


def run_lab_estimates(
    *,
    name: str,
    resistance_preset: str | Path,
    summary_path: str | Path | None = None,
    data_directory: str | Path | None = None,
    ambient_temperatures_K: Sequence[float] = (298.0, 325.0, 330.0),
    thermal_times_ns: Sequence[float] = (10.0, 20.0, 50.0, 100.0),
    ripple_threshold_mV: float = 20.0,
    output_root: str | Path = "runs",
    command: str = "neuristor analyze estimates",
) -> RunBundle:
    """Archive direct and assumption-dependent estimates from lab traces."""

    if (summary_path is None) == (data_directory is None):
        raise ValueError("Provide exactly one of summary_path or data_directory")
    summary_source = (
        Path(summary_path).expanduser().resolve()
        if summary_path is not None
        else Path(str(data_directory)).expanduser().resolve()
    )
    preset_path = Path(resistance_preset).expanduser().resolve()
    config = {
        "schema_version": 1,
        "name": name,
        "kind": "analysis",
        "model": "lab-parameter-estimates",
        "summary_source": str(summary_source),
        "resistance_preset": str(preset_path),
        "ambient_temperatures_K": list(ambient_temperatures_K),
        "thermal_times_ns": list(thermal_times_ns),
        "ripple_threshold_mV": ripple_threshold_mV,
    }
    bundle = RunBundle.create(
        name=name,
        model="lab-parameter-estimates",
        kind="analysis",
        config=config,
        output_root=output_root,
        command=command,
    )
    try:
        if summary_path is not None:
            summary = pd.read_csv(summary_source)
        else:
            _, summary = load_converted_sweep(summary_source)
        payload = json.loads(preset_path.read_text())
        raw_parameters = payload.get("resist_params", payload)
        resistance = YuanhangResistParams(**raw_parameters)
        estimates = estimate_lab_parameters(
            summary,
            transition_temperature_K=float(resistance.Tc_K + 0.5 * resistance.w_eff),
            ambient_temperatures_K=ambient_temperatures_K,
            thermal_times_ns=thermal_times_ns,
            ripple_threshold_mV=ripple_threshold_mV,
        )
        tables = {
            "electrical_capacitance.csv": (estimates.electrical_capacitance, "Electrical capacitance estimates"),
            "thermal_conductance.csv": (estimates.thermal_conductance, "Thermal conductance scenarios"),
            "thermal_capacitance.csv": (estimates.thermal_capacitance, "Thermal capacitance scenarios"),
            "effective_resistance.csv": (estimates.effective_resistance, "Effective plateau resistance"),
        }
        for filename, (frame, label) in tables.items():
            frame.to_csv(bundle.add_artifact(filename, label=label, media_type="text/csv"), index=False)
        overview = bundle.add_artifact(
            "figures/parameter_estimates.png", label="Parameter estimates", media_type="image/png"
        )
        plot_lab_parameter_estimates(
            estimates.electrical_capacitance,
            estimates.thermal_conductance,
            estimates.effective_resistance,
            overview,
        )
        floor = bundle.add_artifact(
            "figures/voltage_floor.png", label="Metallic voltage-floor comparison", media_type="image/png"
        )
        plot_voltage_floor_comparison(estimates.effective_resistance, resistance.Rm, YuanhangResistParams().Rm, floor)
        high_current = estimates.effective_resistance[estimates.effective_resistance["current_plateau_uA"] >= 350.0]
        metrics = {
            "switching_lower_current_uA": float(estimates.pre_switch["current_plateau_uA"]),
            "switching_upper_current_uA": float(estimates.post_switch["current_plateau_uA"]),
            "electrical_capacitance_median_pF": float(estimates.electrical_capacitance["C_slope_pF"].median()),
            "electrical_capacitance_min_pF": float(estimates.electrical_capacitance["C_slope_pF"].min()),
            "electrical_capacitance_max_pF": float(estimates.electrical_capacitance["C_slope_pF"].max()),
            "specimen_metallic_resistance_ohm": float(resistance.Rm),
            "high_current_effective_resistance_min_ohm": float(high_current["R_effective_ohm"].min()),
            "high_current_effective_resistance_max_ohm": float(high_current["R_effective_ohm"].max()),
        }
        bundle.write_json("metrics.json", metrics, label="Estimate summary")
        report = f"""# {name}

The switching onset is bracketed between **{metrics["switching_lower_current_uA"]:.3g}** and
**{metrics["switching_upper_current_uA"]:.3g} uA** from the plateau-ripple increase.

The cold-edge estimate `C = I/(dV/dt)` gives a median electrical capacitance of
**{metrics["electrical_capacitance_median_pF"]:.3g} pF**.

`S_e = P_switch/(T_switch-T0)` is tabulated for each candidate ambient temperature.
The electrical waveform alone does not identify thermal capacitance: once a recovery
time is measured, use `C_th = S_e tau_th`; the scenario table makes that dependence
explicit.

The fitted specimen metallic resistance is **{resistance.Rm:.3g} ohm**. In an ideal
current source, it fixes the steady switched-state floor through `V = I Rm`.
Electrical capacitance changes the approach time but cannot change that steady floor.
"""
        bundle.write_text("report.md", report, label="Scientific report")
        bundle.complete(summary=metrics)
    except BaseException as exc:
        bundle.fail(exc)
        raise
    return bundle
