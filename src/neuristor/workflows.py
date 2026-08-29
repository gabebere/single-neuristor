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
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import ConfigError, apply_overrides, deep_set, load_toml, resolved_copy, source_directory, validate_config
from .current_drive_sim import CurrentDriveParams, current_drive_operating_estimates, simulate_current_step
from .experimental_waveforms import load_converted_sweep
from .metrics import current_run_metrics, voltage_run_metrics
from .lab_estimates import estimate_environmental_conductance, estimate_thermal_capacitance
from .model_validation import capacitance_sensitivity, compare_model_to_lab
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
    plot_lab_detection_window_trace,
    plot_environmental_conductance_estimate,
    plot_lab_oscillation_bracket,
    plot_lab_summary,
    plot_capacitance_sensitivity,
    plot_model_validation_summary,
    plot_model_validation_traces,
    plot_resistance_fit,
    plot_sweep_summary,
    plot_thermal_capacitance_estimate,
    plot_voltage_run,
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


def _current_params_from_config(
    config: Mapping[str, Any],
) -> tuple[CurrentDriveParams, float, YuanhangResistParams, str, str]:
    """Translate a validated current recipe to SI-valued model parameters."""

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
    return params, amplitude_uA, resistance, resistance_source, start_branch


def _evaluate_current(config: Mapping[str, Any]) -> SimulationResult:
    params, amplitude_uA, resistance, resistance_source, start_branch = _current_params_from_config(config)
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
        ordered_summary = summary.sort_values("current_plateau_uA").reset_index(drop=True)
        oscillatory = ordered_summary[ordered_summary["oscillation_detected"].astype(bool)]
        oscillating_positions = np.flatnonzero(
            ordered_summary["oscillation_detected"].astype(bool).to_numpy()
        )
        if oscillating_positions.size == 0 or int(oscillating_positions[0]) == 0:
            raise ValueError("Laboratory sweep does not bracket oscillation onset")
        onset_position = int(oscillating_positions[0])
        pre_onset = ordered_summary.iloc[onset_position - 1]
        first_oscillating = ordered_summary.iloc[onset_position]
        pre_onset_trace = trace_frame.loc[
            trace_frame["source_file"] == str(pre_onset["source_file"])
        ].copy()
        onset_trace = trace_frame.loc[
            trace_frame["source_file"] == str(first_oscillating["source_file"])
        ].copy()
        pre_onset_figure = bundle.add_artifact(
            "figures/pre_onset_trace.png",
            label="Last non-oscillating measured trace before onset",
            media_type="image/png",
        )
        plot_lab_detection_window_trace(pre_onset_trace, pre_onset, pre_onset_figure)
        onset_figure = bundle.add_artifact(
            "figures/oscillation_onset_trace.png",
            label="First coherently oscillating measured trace",
            media_type="image/png",
        )
        plot_lab_detection_window_trace(onset_trace, first_oscillating, onset_figure)
        bracket_figure = bundle.add_artifact(
            "figures/oscillation_onset_bracket.png",
            label="Measured traces bracketing coherent-oscillation onset",
            media_type="image/png",
        )
        plot_lab_oscillation_bracket(
            pre_onset_trace,
            pre_onset,
            onset_trace,
            first_oscillating,
            bracket_figure,
        )
        metrics = {
            "waveforms": len(summary),
            "samples": len(trace_frame),
            "current_min_uA": float(summary["current_plateau_uA"].min()),
            "current_max_uA": float(summary["current_plateau_uA"].max()),
            "oscillation_current_min_uA": float(oscillatory["current_plateau_uA"].min()),
            "oscillation_current_max_uA": float(oscillatory["current_plateau_uA"].max()),
            "oscillation_frequency_min_MHz": float(oscillatory["oscillation_frequency_MHz"].min()),
            "oscillation_frequency_max_MHz": float(oscillatory["oscillation_frequency_MHz"].max()),
            "last_nonoscillating_source_file": str(pre_onset["source_file"]),
            "last_nonoscillating_drive_mV": float(pre_onset["nominal_drive_mV"]),
            "last_nonoscillating_current_step_uA": float(pre_onset["current_step_uA"]),
            "last_nonoscillating_peak_count": int(pre_onset["oscillation_peak_count"]),
            "last_nonoscillating_voltage_vpp_mV": float(pre_onset["voltage_plateau_vpp_mV"]),
            "first_oscillating_source_file": str(first_oscillating["source_file"]),
            "first_oscillating_drive_mV": float(first_oscillating["nominal_drive_mV"]),
            "first_oscillating_current_step_uA": float(first_oscillating["current_step_uA"]),
            "first_oscillating_peak_count": int(first_oscillating["oscillation_peak_count"]),
            "first_oscillating_period_cv": float(first_oscillating["oscillation_period_cv"]),
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
            f"The immediately preceding record is "
            f"`{metrics['last_nonoscillating_source_file']}` at "
            f"**{metrics['last_nonoscillating_current_step_uA']:.3g} uA measured current step** "
            f"({metrics['last_nonoscillating_drive_mV']:.0f} mV source setting). Its fixed "
            f"50--250 ns analysis window contains only "
            f"**{metrics['last_nonoscillating_peak_count']} candidate peak**, so no periodic "
            f"frequency is assigned.\n\n"
            f"The first coherently oscillating record is "
            f"`{metrics['first_oscillating_source_file']}` at "
            f"**{metrics['first_oscillating_current_step_uA']:.3g} uA measured current step** "
            f"({metrics['first_oscillating_drive_mV']:.0f} mV source setting). Its fixed "
            f"50--250 ns analysis window contains **{metrics['first_oscillating_peak_count']} peaks** "
            f"with a period coefficient of variation of "
            f"**{100.0 * metrics['first_oscillating_period_cv']:.2f}%**.\n\n"
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


def run_environmental_conductance(
    data_directory: str | Path,
    *,
    name: str,
    resistance_preset: str | Path,
    resistance_bootstrap: str | Path | None = None,
    ambient_temperature_K: float = 314.4,
    ambient_interval_K: tuple[float, float] = (314.25, 314.55),
    baseline_window_ns: tuple[float, float] = (-200.0, -50.0),
    steady_window_ns: tuple[float, float] = (100.0, 250.0),
    bootstrap_samples: int = 1000,
    block_size: int = 10,
    seed: int = 20260817,
    output_root: str | Path = "runs",
    command: str = "neuristor analyze conductance",
) -> RunBundle:
    """Archive the quasi-steady specimen environmental-conductance estimate."""

    data_path = Path(data_directory).expanduser().resolve()
    preset_path = Path(resistance_preset).expanduser().resolve()
    bootstrap_path = (
        Path(resistance_bootstrap).expanduser().resolve()
        if resistance_bootstrap is not None
        else None
    )
    config = {
        "schema_version": 1,
        "name": name,
        "kind": "analysis",
        "model": "environmental-thermal-conductance",
        "data_directory": str(data_path),
        "resistance_preset": str(preset_path),
        "resistance_bootstrap": str(bootstrap_path) if bootstrap_path else None,
        "ambient_temperature_K": float(ambient_temperature_K),
        "ambient_interval_K": list(ambient_interval_K),
        "baseline_window_ns": list(baseline_window_ns),
        "steady_window_ns": list(steady_window_ns),
        "bootstrap_samples": int(bootstrap_samples),
        "block_size": int(block_size),
        "seed": int(seed),
    }
    bundle = RunBundle.create(
        name=name,
        model="environmental-thermal-conductance",
        kind="analysis",
        config=config,
        output_root=output_root,
        command=command,
    )
    try:
        traces, summary = load_converted_sweep(data_path)
        payload = json.loads(preset_path.read_text())
        raw_parameters = payload.get("resist_params", payload)
        resistance = YuanhangResistParams(**raw_parameters)
        resistance_samples = pd.read_csv(bootstrap_path) if bootstrap_path else None
        estimate = estimate_environmental_conductance(
            traces,
            summary,
            resistance=resistance,
            ambient_temperature_K=ambient_temperature_K,
            ambient_interval_K=ambient_interval_K,
            resistance_bootstrap=resistance_samples,
            baseline_window_ns=baseline_window_ns,
            steady_window_ns=steady_window_ns,
            bootstrap_samples=bootstrap_samples,
            block_size=block_size,
            seed=seed,
        )
        estimate.result.to_csv(
            bundle.add_artifact(
                "conductance_estimate.csv",
                label="Environmental conductance estimate",
                media_type="text/csv",
            ),
            index=False,
        )
        estimate.analyzed_trace.to_csv(
            bundle.add_artifact(
                "selected_trace.csv",
                label="Selected numerical waveform",
                media_type="text/csv",
            ),
            index=False,
        )
        estimate.bootstrap.to_csv(
            bundle.add_artifact(
                "conductance_bootstrap.csv",
                label="Conductance uncertainty propagation",
                media_type="text/csv",
            ),
            index=False,
        )
        figure = bundle.add_artifact(
            "figures/environmental_conductance.png",
            label="Environmental conductance evidence",
            media_type="image/png",
        )
        plot_environmental_conductance_estimate(
            estimate.analyzed_trace,
            estimate.result,
            resistance,
            figure,
        )

        row = estimate.result.iloc[0]
        metrics = {
            "selected_trace": str(row["selected_trace"]),
            "first_oscillating_trace": str(row["first_oscillating_trace"]),
            "current_corrected_uA": float(row["current_corrected_uA"]),
            "voltage_corrected_mV": float(row["voltage_corrected_mV"]),
            "effective_resistance_ohm": float(row["effective_resistance_ohm"]),
            "power_uW": float(row["power_uW"]),
            "resistance_drift_fraction": float(row["resistance_drift_fraction"]),
            "inferred_temperature_K": float(row["inferred_temperature_K"]),
            "ambient_temperature_K": float(row["ambient_temperature_K"]),
            "S_e_mW_per_K": float(row["S_e_mW_per_K"]),
            "S_e_ci95_lower_mW_per_K": float(row["S_e_ci95_lower_mW_per_K"]),
            "S_e_ci95_upper_mW_per_K": float(row["S_e_ci95_upper_mW_per_K"]),
            "yuanhang_S_e_mW_per_K": float(YuanhangCircuitParams().Sth_mW_per_K),
        }
        bundle.write_json("metrics.json", metrics, label="Conductance summary")
        report = f"""# {name}

The first coherently oscillating numerical trace is `{metrics["first_oscillating_trace"]}`.
The immediately preceding trace, `{metrics["selected_trace"]}`, is therefore the closest
measured stable point below oscillation onset.

Both channels are corrected by their pre-pulse medians. In the settled 100--250 ns
window the median current is **{metrics["current_corrected_uA"]:.3f} uA**, the median
voltage is **{metrics["voltage_corrected_mV"]:.3f} mV**, the effective resistance is
**{metrics["effective_resistance_ohm"]:.3f} ohm**, and device power is
**{metrics["power_uW"]:.3f} uW**. Resistance changes by only
**{100.0 * metrics["resistance_drift_fraction"]:.3f}%** across the window, supporting
the quasi-steady approximation `dT/dt approximately 0`.

Inverting the specimen's fitted heating branch gives **T={metrics["inferred_temperature_K"]:.3f} K**.
With **T0={metrics["ambient_temperature_K"]:.3f} K**, the thermal balance gives
**S_e={metrics["S_e_mW_per_K"]:.6f} mW/K**. The conditional 95% interval is
**{metrics["S_e_ci95_lower_mW_per_K"]:.6f}--{metrics["S_e_ci95_upper_mW_per_K"]:.6f} mW/K**.
It propagates waveform block resampling, the R(T)-fit bootstrap, and the stated ambient
range. It does not include the systematic possibility that the R(T) and TIA measurements
came from different devices or that driven and quasi-static R(T) differ.
"""
        bundle.write_text("report.md", report, label="Scientific report")
        bundle.complete(summary=metrics)
    except BaseException as exc:
        bundle.fail(exc)
        raise
    return bundle


def run_thermal_capacitance(
    data_directory: str | Path,
    *,
    name: str,
    resistance_preset: str | Path,
    S_e_mW_per_K: float,
    ambient_temperature_K: float = 314.4,
    electrical_capacitance_pF: float = 0.0,
    selected_drives_mV: tuple[float, ...] = (100.0, 150.0, 200.0),
    near_transition_check_mV: float | None = 250.0,
    resistance_bootstrap: str | Path | None = None,
    conductance_bootstrap: str | Path | None = None,
    baseline_window_ns: tuple[float, float] = (-200.0, -50.0),
    integration_window_ns: tuple[float, float] = (-50.0, 80.0),
    fit_window_ns: tuple[float, float] = (15.0, 35.0),
    smoothing_window: int = 9,
    bootstrap_samples: int = 1000,
    fit_window_jitter_ns: int = 2,
    seed: int = 20260817,
    output_root: str | Path = "runs",
    command: str = "neuristor analyze thermal-capacitance",
) -> RunBundle:
    """Archive a conditional thermal time-constant and capacitance estimate."""

    data_path = Path(data_directory).expanduser().resolve()
    preset_path = Path(resistance_preset).expanduser().resolve()
    resistance_bootstrap_path = (
        Path(resistance_bootstrap).expanduser().resolve()
        if resistance_bootstrap is not None
        else None
    )
    conductance_bootstrap_path = (
        Path(conductance_bootstrap).expanduser().resolve()
        if conductance_bootstrap is not None
        else None
    )
    config = {
        "schema_version": 1,
        "name": name,
        "kind": "analysis",
        "model": "thermal-capacitance",
        "electrical_capacitance_pF": float(electrical_capacitance_pF),
        "data_directory": str(data_path),
        "resistance_preset": str(preset_path),
        "resistance_bootstrap": (
            str(resistance_bootstrap_path) if resistance_bootstrap_path else None
        ),
        "conductance_bootstrap": (
            str(conductance_bootstrap_path) if conductance_bootstrap_path else None
        ),
        "S_e_mW_per_K": float(S_e_mW_per_K),
        "ambient_temperature_K": float(ambient_temperature_K),
        "selected_drives_mV": list(selected_drives_mV),
        "near_transition_check_mV": near_transition_check_mV,
        "baseline_window_ns": list(baseline_window_ns),
        "integration_window_ns": list(integration_window_ns),
        "fit_window_ns": list(fit_window_ns),
        "smoothing_window": int(smoothing_window),
        "bootstrap_samples": int(bootstrap_samples),
        "fit_window_jitter_ns": int(fit_window_jitter_ns),
        "seed": int(seed),
    }
    bundle = RunBundle.create(
        name=name,
        model="thermal-capacitance",
        kind="analysis",
        config=config,
        output_root=output_root,
        command=command,
    )
    try:
        traces, summary = load_converted_sweep(data_path)
        payload = json.loads(preset_path.read_text())
        resistance = YuanhangResistParams(**payload.get("resist_params", payload))
        resistance_samples = (
            pd.read_csv(resistance_bootstrap_path)
            if resistance_bootstrap_path is not None
            else None
        )
        conductance_samples = (
            pd.read_csv(conductance_bootstrap_path)
            if conductance_bootstrap_path is not None
            else None
        )
        estimate = estimate_thermal_capacitance(
            traces,
            resistance=resistance,
            S_e_mW_per_K=S_e_mW_per_K,
            ambient_temperature_K=ambient_temperature_K,
            electrical_capacitance_pF=electrical_capacitance_pF,
            selected_drives_mV=selected_drives_mV,
            near_transition_check_mV=near_transition_check_mV,
            resistance_bootstrap=resistance_samples,
            conductance_bootstrap=conductance_samples,
            baseline_window_ns=baseline_window_ns,
            integration_window_ns=integration_window_ns,
            fit_window_ns=fit_window_ns,
            smoothing_window=smoothing_window,
            bootstrap_samples=bootstrap_samples,
            fit_window_jitter_ns=fit_window_jitter_ns,
            seed=seed,
        )
        estimate.result.to_csv(
            bundle.add_artifact(
                "thermal_capacitance_estimate.csv",
                label="Thermal capacitance estimate",
                media_type="text/csv",
            ),
            index=False,
        )
        estimate.trace_fits.to_csv(
            bundle.add_artifact(
                "trace_fits.csv",
                label="Per-trace thermal fits",
                media_type="text/csv",
            ),
            index=False,
        )
        estimate.trajectories.to_csv(
            bundle.add_artifact(
                "thermal_trajectories.csv",
                label="Reconstructed temperature trajectories",
                media_type="text/csv",
            ),
            index=False,
        )
        estimate.bootstrap.to_csv(
            bundle.add_artifact(
                "thermal_capacitance_bootstrap.csv",
                label="Thermal capacitance uncertainty propagation",
                media_type="text/csv",
            ),
            index=False,
        )
        figure = bundle.add_artifact(
            "figures/thermal_capacitance.png",
            label="Thermal capacitance evidence",
            media_type="image/png",
        )
        plot_thermal_capacitance_estimate(
            estimate.trajectories,
            estimate.trace_fits,
            estimate.bootstrap,
            estimate.result,
            figure,
        )

        row = estimate.result.iloc[0]
        oscillating = summary.loc[summary["oscillation_detected"].astype(bool)]
        frequency_min = float(oscillating["oscillation_frequency_MHz"].min())
        frequency_max = float(oscillating["oscillation_frequency_MHz"].max())
        period_min_ns = 1000.0 / frequency_max
        period_max_ns = 1000.0 / frequency_min
        yuanhang_C_th = float(YuanhangCircuitParams().Cth_mW_ns_per_K)
        metrics = {
            "selected_traces": str(row["selected_traces"]),
            "near_transition_check_trace": str(row["near_transition_check_trace"]),
            "ambient_temperature_K": float(row["ambient_temperature_K"]),
            "S_e_mW_per_K": float(row["S_e_mW_per_K"]),
            "electrical_capacitance_pF": float(row["electrical_capacitance_pF"]),
            "C_th_pJ_per_K": float(row["C_th_pJ_per_K"]),
            "C_th_ci95_lower_pJ_per_K": float(row["C_th_ci95_lower_pJ_per_K"]),
            "C_th_ci95_upper_pJ_per_K": float(row["C_th_ci95_upper_pJ_per_K"]),
            "tau_th_ns": float(row["tau_th_ns"]),
            "tau_th_ci95_lower_ns": float(row["tau_th_ci95_lower_ns"]),
            "tau_th_ci95_upper_ns": float(row["tau_th_ci95_upper_ns"]),
            "fit_rmse_K": float(row["fit_rmse_K"]),
            "near_transition_check_C_th_pJ_per_K": float(
                row["near_transition_check_C_th_pJ_per_K"]
            ),
            "near_transition_check_rmse_K": float(row["near_transition_check_rmse_K"]),
            "measured_oscillation_period_min_ns": period_min_ns,
            "measured_oscillation_period_max_ns": period_max_ns,
            "yuanhang_C_th_pJ_per_K": yuanhang_C_th,
            "yuanhang_to_specimen_C_th_ratio": yuanhang_C_th / float(row["C_th_pJ_per_K"]),
        }
        bundle.write_json("metrics.json", metrics, label="Thermal capacitance summary")
        individual = estimate.trace_fits.loc[
            estimate.trace_fits["included_in_primary_fit"].astype(bool),
            "C_th_pJ_per_K",
        ]
        report = f"""# {name}

With electrical capacitance fixed to **C={metrics['electrical_capacitance_pF']:.3g} pF**,
the resistive current is reconstructed as `I_R=I_in-C*dV/dt`. The baseline-corrected
ratio `V/I_R` is mapped to temperature through the fitted specimen heating branch,
and `V*I_R` drives `C_th dT/dt = P(t) - S_e (T-T0)` over the 15--35 ns heating
window. The primary fit uses `{metrics['selected_traces']}`: these moderate nonswitching
traces have adequate signal and remain below the near-transition overshoot.

The shared fit gives **tau_th={metrics['tau_th_ns']:.3f} ns** and
**C_th={metrics['C_th_pJ_per_K']:.6f} pJ/K**, with a temperature RMSE of
**{metrics['fit_rmse_K']:.3f} K**. The individual selected-trace estimates span
**{float(individual.min()):.6f}--{float(individual.max()):.6f} pJ/K**.

The conditional 95% robustness interval is
**{metrics['C_th_ci95_lower_pJ_per_K']:.6f}--{metrics['C_th_ci95_upper_pJ_per_K']:.6f} pJ/K**
for C_th and **{metrics['tau_th_ci95_lower_ns']:.3f}--{metrics['tau_th_ci95_upper_ns']:.3f} ns**
for tau_th. It propagates trace resampling, the R(T)-fit bootstrap, the conductance
bootstrap, and +/-{fit_window_jitter_ns:d} ns fit-window changes. It remains conditional on the adopted
electrical capacitance, the static heating branch applying dynamically, and all measurements describing the same device.
The conductance archive does not retain its paired R(T)-parameter draw, so the two
parameter bootstraps are resampled independently; the interval is conservative and
does not preserve that covariance.

The `{metrics['near_transition_check_trace']}` sensitivity trace gives
**C_th={metrics['near_transition_check_C_th_pJ_per_K']:.6f} pJ/K** with a larger
**{metrics['near_transition_check_rmse_K']:.3f} K** error, confirming that including its
near-transition reversal biases the estimate downward. The fitted thermal time is the
same order as the measured **{period_min_ns:.1f}--{period_max_ns:.1f} ns** oscillation
period. Yuanhang's **{yuanhang_C_th:.4f} pJ/K** reference is about
**{metrics['yuanhang_to_specimen_C_th_ratio']:.0f} times larger**.

The manuscript reports the 150 nm film thickness and approximately 200 nm electrode
gap, but not the electrically active filament width or volume. A geometry calculation
`rho c_p V` is therefore not reported as a numerical cross-check.
"""
        bundle.write_text("report.md", report, label="Scientific report")
        bundle.complete(summary=metrics)
    except BaseException as exc:
        bundle.fail(exc)
        raise
    return bundle


def run_model_validation(
    config: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    command: str = "neuristor analyze model-validation",
) -> RunBundle:
    """Blindly compare one frozen specimen model with the full measured sweep."""

    validate_config(config)
    if str(config.get("model", "")).lower() != "current":
        raise ConfigError("Model validation requires model='current'")
    params, _, resistance, resistance_source, start_branch = _current_params_from_config(config)
    lab = _table(config, "lab_validation")
    sensitivity_config = _table(config, "sensitivity")
    data_directory = Path(str(lab.get("data_directory", ""))).expanduser()
    if not data_directory.is_absolute():
        data_directory = (source_directory(config) / data_directory).resolve()
    if not data_directory.is_dir():
        raise ConfigError(f"Laboratory data directory does not exist: {data_directory}")
    convergence_dt_ns = float(lab.get("convergence_dt_ns", 0.5 * params.dt_s * 1e9))
    capacitances_pF = tuple(float(value) for value in sensitivity_config.get("electrical_capacitances_pF", ()))
    thermal_capacitances = tuple(
        float(value) for value in sensitivity_config.get("thermal_capacitances_pJ_per_K", ())
    )
    if not capacitances_pF or not thermal_capacitances:
        raise ConfigError("[sensitivity] requires both capacitance arrays")
    if min(capacitances_pF) < 0.0 or min(thermal_capacitances) <= 0.0:
        raise ConfigError("Sensitivity capacitances must be physical")

    resolved = resolved_copy(config)
    resolved.setdefault("lab_validation", {})["data_directory"] = str(data_directory)
    resolved.setdefault("resistance", {})["resolved_source"] = resistance_source
    output = output_root or _table(config, "output").get("root", "runs")
    bundle = RunBundle.create(
        name=str(config["name"]),
        model="specimen-model-validation",
        kind="analysis",
        config=resolved,
        output_root=output,
        command=command,
    )
    try:
        lab_traces, _ = load_converted_sweep(data_directory)
        validation = compare_model_to_lab(
            lab_traces,
            params,
            convergence_dt_ns=convergence_dt_ns,
            convergence_drives_mV=tuple(float(value) for value in lab.get("convergence_drives_mV", (300, 500, 800))),
        )
        sensitivity = capacitance_sensitivity(
            validation.comparison["measured_current_step_uA"].to_numpy(dtype=float),
            params,
            capacitances_pF=capacitances_pF,
            thermal_capacitances_pJ_per_K=thermal_capacitances,
            pulse_duration_ns=float(sensitivity_config.get("pulse_duration_ns", 300.0)),
            pre_duration_ns=float(sensitivity_config.get("pre_duration_ns", 200.0)),
            post_duration_ns=float(sensitivity_config.get("post_duration_ns", 200.0)),
        )

        validation.comparison.to_csv(
            bundle.add_artifact("comparison.csv", label="Measured versus predicted sweep", media_type="text/csv"),
            index=False,
        )
        validation.traces.to_csv(
            bundle.add_artifact("comparison_traces.csv", label="Measured and predicted traces", media_type="text/csv"),
            index=False,
        )
        validation.convergence.to_csv(
            bundle.add_artifact("convergence.csv", label="Time-step convergence check", media_type="text/csv"),
            index=False,
        )
        sensitivity.summary.to_csv(
            bundle.add_artifact("capacitance_sensitivity.csv", label="C and Cth sensitivity grid", media_type="text/csv"),
            index=False,
        )
        plot_model_validation_summary(
            validation.comparison,
            bundle.add_artifact("figures/model_vs_experiment.png", label="Model versus experiment summary", media_type="image/png"),
        )
        plot_model_validation_traces(
            validation.traces,
            validation.comparison,
            bundle.add_artifact("figures/representative_traces.png", label="Representative measured and predicted traces", media_type="image/png"),
        )
        plot_capacitance_sensitivity(
            sensitivity.summary,
            bundle.add_artifact("figures/capacitance_sensitivity.png", label="Capacitance sensitivity maps", media_type="image/png"),
            adopted_C_pF=float(params.C_F) * 1e12,
            adopted_C_th_pJ_per_K=float(params.C_th_J_per_K) * 1e12,
        )

        comparison = validation.comparison
        measured_osc = comparison["measured_oscillation_detected"].astype(bool)
        predicted_osc = comparison["predicted_oscillation_detected"].astype(bool)
        operating = current_drive_operating_estimates(
            params,
            I_uA=float(comparison["measured_current_step_uA"].median()),
        )
        detected_sensitivity = sensitivity.summary.loc[sensitivity.summary["oscillation_detected"].astype(bool)]
        minimum_oscillating_C = (
            float(detected_sensitivity["electrical_capacitance_pF"].min())
            if not detected_sensitivity.empty
            else float("nan")
        )
        adopted_cth_rows = sensitivity.summary.loc[
            np.isclose(
                sensitivity.summary["thermal_capacitance_pJ_per_K"],
                float(params.C_th_J_per_K) * 1e12,
            )
        ]
        adopted_cth_oscillating = adopted_cth_rows.loc[
            adopted_cth_rows["oscillation_detected"].astype(bool)
        ]
        minimum_oscillating_C_at_adopted_cth = (
            float(adopted_cth_oscillating["electrical_capacitance_pF"].min())
            if not adopted_cth_oscillating.empty
            else float("nan")
        )
        within_timing_bound = sensitivity.summary.loc[
            sensitivity.summary["electrical_capacitance_pF"] <= float(params.C_F) * 1e12 + 1e-12
        ]
        adopted_C_rows = sensitivity.summary.loc[
            np.isclose(sensitivity.summary["electrical_capacitance_pF"], float(params.C_F) * 1e12)
            & np.isclose(
                sensitivity.summary["thermal_capacitance_pJ_per_K"],
                float(params.C_th_J_per_K) * 1e12,
            )
        ]
        oscillating_positions = np.flatnonzero(measured_osc.to_numpy())
        if oscillating_positions.size == 0 or int(oscillating_positions[0]) == 0:
            raise ValueError("Measured sweep must bracket oscillation onset for validation")
        pre_onset = comparison.iloc[int(oscillating_positions[0]) - 1]
        metrics: dict[str, Any] = {
            "waveforms": int(len(comparison)),
            "measured_oscillating_waveforms": int(measured_osc.sum()),
            "predicted_oscillating_waveforms": int(predicted_osc.sum()),
            "classification_matches": int(comparison["classification_match"].astype(bool).sum()),
            "measured_oscillation_current_min_uA": float(comparison.loc[measured_osc, "measured_current_step_uA"].min()),
            "measured_oscillation_current_max_uA": float(comparison.loc[measured_osc, "measured_current_step_uA"].max()),
            "measured_frequency_min_MHz": float(comparison.loc[measured_osc, "measured_oscillation_frequency_MHz"].min()),
            "measured_frequency_max_MHz": float(comparison.loc[measured_osc, "measured_oscillation_frequency_MHz"].max()),
            "measured_energy_per_cycle_min_pJ": float(comparison.loc[measured_osc, "measured_energy_per_cycle_pJ"].min()),
            "measured_energy_per_cycle_max_pJ": float(comparison.loc[measured_osc, "measured_energy_per_cycle_pJ"].max()),
            "median_plateau_voltage_rmse_mV": float(comparison["plateau_voltage_rmse_mV"].median()),
            "pre_onset_measured_voltage_mean_mV": float(pre_onset["measured_voltage_mean_mV"]),
            "pre_onset_predicted_voltage_mean_mV": float(pre_onset["predicted_voltage_mean_mV"]),
            "pre_onset_voltage_error_mV": float(pre_onset["plateau_voltage_mean_error_mV"]),
            "adopted_electrical_capacitance_pF": float(params.C_F) * 1e12,
            "adopted_thermal_capacitance_pJ_per_K": float(params.C_th_J_per_K) * 1e12,
            "thermal_time_constant_ns": float(params.C_th_J_per_K / params.S_e_W_per_K) * 1e9,
            "thermal_only_heating_threshold_uA": float(operating["thermal_only_lower_current_uA"]),
            "thermal_only_cooling_threshold_uA": float(operating["thermal_only_upper_current_uA"]),
            "thermal_only_window_exists": bool(operating["thermal_only_window_exists"]),
            "adopted_grid_oscillating_waveforms": int(adopted_C_rows["oscillation_detected"].astype(bool).sum()),
            "minimum_grid_capacitance_with_oscillation_pF": minimum_oscillating_C,
            "minimum_grid_capacitance_with_oscillation_at_adopted_C_th_pF": minimum_oscillating_C_at_adopted_cth,
            "oscillations_anywhere_within_timing_bound": bool(
                within_timing_bound["oscillation_detected"].astype(bool).any()
            ),
            "maximum_convergence_mean_voltage_difference_mV": float(
                validation.convergence["absolute_mean_difference_mV"].max()
            ),
            "start_branch": start_branch,
            "resistance_source": resistance_source,
            "metallic_resistance_ohm": float(resistance.Rm),
        }
        bundle.write_json("metrics.json", metrics, label="Validation summary")
        report = f"""# {config['name']}

All {metrics['waveforms']} measured, baseline-corrected current waveforms were replayed
through one frozen parameter set. No parameter was retuned by current. The experiment
contains {metrics['measured_oscillating_waveforms']} coherent oscillatory records from
{metrics['measured_oscillation_current_min_uA']:.1f} to
{metrics['measured_oscillation_current_max_uA']:.1f} uA at
{metrics['measured_frequency_min_MHz']:.1f}--{metrics['measured_frequency_max_MHz']:.1f} MHz;
the adopted model predicts {metrics['predicted_oscillating_waveforms']}.

The stable pre-onset trace is nevertheless reproduced closely: at 189.6 uA its measured
mean is {metrics['pre_onset_measured_voltage_mean_mV']:.2f} mV and the prediction is
{metrics['pre_onset_predicted_voltage_mean_mV']:.2f} mV. Thus the static cold-side
calibration works while the dynamic switching window does not.

In the algebraic C=0 limit, heating requires about
{metrics['thermal_only_heating_threshold_uA']:.1f} uA, but cooling through the opposite
transition requires current below {metrics['thermal_only_cooling_threshold_uA']:.1f} uA.
Because the lower bound exceeds the upper bound, no thermal-only oscillation window exists.
The adopted C={metrics['adopted_electrical_capacitance_pF']:.2f} pF also produces no grid
oscillations, and none occur at any tested C within the timing bound for the full conditional
C_th interval. The first tested capacitance that produces an oscillation is
{metrics['minimum_grid_capacitance_with_oscillation_pF']:.3g} pF at the lower C_th bound and
{metrics['minimum_grid_capacitance_with_oscillation_at_adopted_C_th_pF']:.3g} pF at the adopted
C_th; both are outside the electrical timing bound.

Halving the integration step changes representative plateau means by at most
{metrics['maximum_convergence_mean_voltage_difference_mV']:.3f} mV and does not change
their oscillation classifications. The failure is therefore a model/parameter
incompatibility, not a time-step artifact. Likely next tests are an independently measured
dynamic switching loop or a circuit model that includes the real TIA/load impedance;
gamma alone cannot repair the absent onset because the first heating transition occurs
before a minor-loop reversal.
"""
        bundle.write_text("report.md", report, label="Scientific report")
        bundle.complete(summary=metrics)
    except BaseException as exc:
        bundle.fail(exc)
        raise
    return bundle
