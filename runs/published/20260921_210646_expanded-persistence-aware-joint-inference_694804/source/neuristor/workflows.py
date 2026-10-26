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
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import ConfigError, apply_overrides, deep_set, load_toml, resolved_copy, source_directory, validate_config
from .current_drive_sim import CurrentDriveParams, current_drive_operating_estimates, simulate_current_step
from .experimental_waveforms import load_converted_sweep
from .metrics import current_run_metrics, voltage_run_metrics
from .lab_estimates import estimate_environmental_conductance, estimate_thermal_capacitance, reconstruct_hysteresis
from .model_validation import capacitance_sensitivity, compare_model_to_lab
from .oscillation_audit import AuditSettings, audit_voltage, evaluate_candidate, parameter_map
from .parameter_inference import (
    PARAMETER_NAMES,
    FitParameter,
    ObjectiveWeights,
    evaluate_fitted_model,
    evaluate_parameter_vector,
    fit_parameter_set,
    parameter_vector_from_model,
    model_from_parameter_vector,
    prepare_inference_dataset,
)
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
    plot_inference_operating_summary,
    plot_inference_optimization,
    plot_inference_representative_traces,
    plot_resistance_fit,
    plot_sweep_summary,
    plot_thermal_capacitance_estimate,
    plot_voltage_run,
    plot_oscillation_audit,
    plot_oscillation_parameter_maps,
    plot_audit_candidate_traces,
    plot_reconstructed_hysteresis,
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


def run_hysteresis_reconstruction(
    config: Mapping[str, Any], *, output_root: str | Path | None = None,
    command: str = "neuristor analyze reconstruct-hysteresis",
) -> RunBundle:
    """Low-cost inverse consistency test using measured power, with no optimization."""

    validate_config(config)
    if config["model"] != "current":
        raise ConfigError("Hysteresis reconstruction requires the current model")
    params, _, rp, resistance_source, _ = _current_params_from_config(config)
    options = _table(config, "reconstruction")
    directory = Path(str(options["data_directory"]))
    if not directory.is_absolute():
        directory = (source_directory(config) / directory).resolve()
    default_case = {"C_pF": params.C_F * 1e12, "C_th_pJ_per_K": params.C_th_J_per_K * 1e12,
                    "S_e_mW_per_K": params.S_e_W_per_K * 1e3}
    cases = [{**default_case, **dict(case)} for case in options["cases"]]
    if len({case["name"] for case in cases}) != len(cases):
        raise ConfigError("Reconstruction case names must be unique")
    resolved = resolved_copy(config)
    resolved["reconstruction"]["resolved_cases"] = cases
    resolved["resistance"]["resolved_source"] = resistance_source
    bundle = RunBundle.create(name=str(config["name"]), model="hysteresis-reconstruction", kind="analysis",
                              config=resolved, output_root=output_root or config["output"]["root"], command=command)
    try:
        raw, _ = load_converted_sweep(directory)
        summary, trajectories = reconstruct_hysteresis(
            raw, rp, cases=cases, gammas=list(map(float, options["gamma_values"])),
            replay_steps_ns=list(map(float, options["replay_steps_ns"])),
            ambient_temperature_K=params.T0_K,
        )
        # Rank gamma on measured oscillatory records, weighting currents equally.
        measured = prepare_inference_dataset(raw, holdout_drives_mV=[50, 1000]).measured_summary
        oscillator_files = measured.loc[measured.oscillation_detected, "source_file"]
        gamma_scores = summary[summary.source_file.isin(oscillator_files)].groupby(
            ["name", "gamma", "replay_step_ns"], as_index=False,
        ).agg(mean_log_resistance_rmse=("resistance_log_rmse", "mean"))
        for name, frame in (("reconstruction", summary), ("trajectories", trajectories), ("gamma_scores", gamma_scores)):
            frame.to_csv(bundle.path(f"{name}.csv"), index=False)
            bundle.register_file(f"{name}.csv", label=name.replace("_", " ").title(), media_type="text/csv")
        figure = "figures/reconstructed_hysteresis.png"
        plot_reconstructed_hysteresis(trajectories, rp, bundle.path(figure),
                                      case=str(cases[0]["name"]), drives=options["figure_source_settings_mV"])
        bundle.register_file(figure, label="Measured resistance versus conditional temperature", media_type="image/png")
        finest = float(summary.replay_step_ns.min())
        central = summary[(summary.name == cases[0]["name"]) & (summary.gamma == options["gamma_values"][0])
                          & (summary.replay_step_ns == finest)]
        selected = central[central.source_setting_mV.isin(options["figure_source_settings_mV"])].sort_values("current_uA")
        metrics = {"cases": len(cases), "currents": int(raw.source_file.nunique()), "replay_steps_ns": options["replay_steps_ns"],
                   "gamma_values": options["gamma_values"], "selected_central_results": selected.to_dict("records")}
        bundle.write_json("metrics.json", metrics, label="Reconstruction findings")
        report = ["# Conditional reconstruction of the driven resistance law", "",
                  "The measured channels are baseline corrected and smoothed with a five-point "
                  "quadratic Savitzky–Golay filter (one sensitivity case uses nine points). "
                  "The existing thermal-analysis implementation computes I_R = I - C dV/dt, "
                  "R_eff = V/I_R and P = V I_R. Its exact piecewise-linear-power integrator "
                  "then calculates T(t) from Cth dT/dt = P - Se(T-T0), starting at T0 at -200 ns.", "",
                  "The authoritative Yuanhang hysteresis law is replayed on this prescribed "
                  "temperature history. This tests its resistance response without asking an "
                  "optimizer to repair a forward waveform. Temperature is conditional on the "
                  "assumed lumped thermal balance, channel interpretation and parameter values; "
                  "it is not an independent thermometer. Static R(T) is not used to reconstruct T.", "",
                  "Sensitivity cases change C, Se, Cth or smoothing one at a time. They are "
                  "not a joint confidence region. Gamma is scanned with the major-loop "
                  "parameters fixed. Replays at 1, 0.5 and 0.25 ns test hysteresis sampling; "
                  "the measured channels remain sampled at 1 ns. All 22 currents are included, "
                  "and gamma is ranked by mean log-resistance RMSE on the 11 original oscillators.", "",
                  "The table uses 150–250 ns, the central C=0.39 pF case and Yuanhang gamma. "
                  "Invalid nonpositive resistive current and excursions outside the resistance "
                  "temperature range are explicitly reported in reconstruction.csv.", "",
                  "| Current (uA) | Conditional mean T (K) | Measured mean R (ohm) | Replayed mean R (ohm) |",
                  "|---:|---:|---:|---:|"]
        for row in selected.itertuples():
            report.append(f"| {row.current_uA:.1f} | {row.temperature_mean_K:.2f} "
                          f"| {row.measured_resistance_mean_ohm:.1f} | {row.replayed_resistance_mean_ohm:.1f} |")
        report += ["", "A mismatch identifies a conflict among the constitutive law, thermal "
                   "model, parameters and measured-channel interpretation; it does not uniquely "
                   "identify which assumption is wrong. A good conditional gamma score would "
                   "still require independent forward validation.", "",
                   "![Reconstructed trajectories](figures/reconstructed_hysteresis.png)"]
        bundle.write_text("report.md", "\n".join(report), label="Method and conditional results")
        bundle.complete(summary=metrics)
    except BaseException as exc:
        bundle.fail(exc)
        raise
    return bundle


def run_oscillation_audit(
    config: Mapping[str, Any], *, output_root: str | Path | None = None,
    command: str = "neuristor analyze oscillation-audit",
    reuse_numerics: str | Path | None = None,
) -> RunBundle:
    """Audit archived persistence, map three currents, and verify shared candidates.

    Archived bundles are read only. The old detector remains available for exact
    historical reproduction; this workflow adds independent evidence and flags its
    finite-window false successes. Ranking uses equal continuous feature terms.
    """

    validate_config(config)
    if config["model"] != "current":
        raise ConfigError("Oscillation audit requires the current-source model")
    audit = _table(config, "audit")

    def local_path(key: str) -> Path:
        value = Path(str(audit[key])).expanduser()
        return value.resolve() if value.is_absolute() else (source_directory(config) / value).resolve()

    reference = local_path("reference_bundle")
    data_directory = local_path("data_directory")
    mode = str(audit.get("reference_mode", "relaxed"))
    parameters = pd.read_csv(reference / "parameter_comparison.csv").set_index("parameter")
    values = parameters.loc[list(PARAMETER_NAMES), f"{mode}_fit"].to_numpy(float)
    base, _, _, resistance_source, _ = _current_params_from_config(config)
    base = model_from_parameter_vector(values, base, dt_ns=float(config["time"]["dt_ns"]))
    settings = AuditSettings(**{key: tuple(value) if isinstance(value, list) else value
                                for key, value in _table(audit, "metrics").items()})
    grid = _table(audit, "grid")
    axes = {}
    for key, reference_value in (("C_pF", base.C_F * 1e12),
                                 ("tau_th_ns", base.C_th_J_per_K / base.S_e_W_per_K * 1e9),
                                 ("gamma", base.resist_params.gamma)):
        values_axis = list(map(float, grid[key]))
        if bool(grid.get("include_reference", True)):
            values_axis.append(reference_value)
        axis = np.unique(values_axis)
        if (not len(axis) or not np.all(np.isfinite(axis)) or np.any(axis < 0)
                or (key != "C_pF" and np.any(axis == 0))):
            raise ConfigError(f"Invalid audit.grid.{key}")
        axes[key] = axis.tolist()
    steps = [float(value) for value in audit["verification_dt_ns"]]
    if any(step <= 0 for step in steps) or base.S_e_W_per_K <= 0:
        raise ConfigError("Audit requires positive time steps and environmental conductance")
    resolved = resolved_copy(config)
    resolved["audit"]["resolved_grid"] = axes
    resolved["audit"]["reference_values"] = dict(zip(PARAMETER_NAMES, values.tolist()))
    resolved["audit"]["resolved_reference_bundle"] = str(reference)
    resolved["resistance"]["resolved_source"] = resistance_source
    bundle = RunBundle.create(name=str(config["name"]), model="oscillation-audit", kind="analysis",
                              config=resolved, output_root=output_root or config["output"]["root"], command=command)
    try:
        if reuse_numerics is not None:
            return _render_existing_audit(bundle, Path(reuse_numerics), resolved, reference, mode, settings)
        lab, _ = load_converted_sweep(data_directory)
        dataset = prepare_inference_dataset(lab, holdout_drives_mV=[50.0, 1000.0])
        # Full recorded prehistory is retained; data after the audit ends is not needed.
        keep = dataset.time_ns <= settings.window_edges_ns[-1]
        dataset = dataclasses.replace(dataset, time_ns=dataset.time_ns[keep],
                                      current_uA=dataset.current_uA[keep], voltage_mV=dataset.voltage_mV[keep])
        indices = []
        for drive in audit["source_settings_mV"]:
            found = np.flatnonzero(np.isclose(dataset.nominal_drives_mV, float(drive)))
            if len(found) != 1:
                raise ConfigError(f"Expected one measured record for source setting {drive}")
            indices.append(int(found[0]))
        indices = np.asarray(indices)
        measured = [audit_voltage(dataset.time_ns, dataset.voltage_mV[:, i], settings)[0]
                    for i in range(len(dataset.nominal_drives_mV))]
        measured_table = pd.DataFrame(measured).assign(source_setting_mV=dataset.nominal_drives_mV,
                                                     current_uA=dataset.measured_summary.current_step_uA.to_numpy())
        history = pd.read_csv(reference / "fitted_traces.csv")
        old_rows, old_windows = [], []
        for (fit_mode, drive), frame in history.groupby(["fit_mode", "nominal_drive_mV"]):
            frame = frame.sort_values("time_ns")
            summary, window = audit_voltage(frame.time_ns.to_numpy(), frame.predicted_voltage_mV.to_numpy(), settings)
            old_rows.append({"fit_mode": fit_mode, "source_setting_mV": drive, **summary})
            old_windows.append(window.assign(fit_mode=fit_mode, source_setting_mV=drive))
        archived = pd.DataFrame(old_rows)
        archived_windows = pd.concat(old_windows, ignore_index=True)

        def progress(done: int, total: int) -> None:
            if done == 1 or done % 12 == 0 or done == total:
                print(f"Mapped {done}/{total} shared parameter combinations", flush=True)

        mapped, windows = parameter_map(dataset, base, indices, capacitances_pF=axes["C_pF"],
                                        thermal_times_ns=axes["tau_th_ns"], gammas=axes["gamma"],
                                        settings=settings, progress=progress)
        ranking = mapped.groupby(["candidate_id", "C_pF", "tau_th_ns", "C_th_pJ_per_K", "gamma"], as_index=False).agg(
            mean_feature_score=("feature_score", "mean"), sustained_currents=("sustained", "sum"),
            worst_feature_score=("feature_score", "max"),
        ).sort_values(["mean_feature_score", "candidate_id"])
        choices = {"reference": {"C_pF": base.C_F * 1e12,
                                  "tau_th_ns": base.C_th_J_per_K / base.S_e_W_per_K * 1e9,
                                  "gamma": base.resist_params.gamma}}
        choices["best_features"] = ranking.iloc[0].to_dict()
        persistent = ranking[ranking.sustained_currents == len(indices)]
        if len(persistent):
            choices["best_sustained"] = persistent.iloc[0].to_dict()
        bounded = ranking[ranking.C_pF <= .39 + 1e-12]
        if len(bounded):
            choices["best_timing_bound"] = bounded.iloc[0].to_dict()
        # Each candidate is shared across all currents. Selection used three records;
        # the rest are cross-current checks, not a pristine blind validation set.
        verification, verification_windows, traces = [], [], []
        all_indices = np.arange(len(measured))
        for label, choice in choices.items():
            for dt in steps:
                params = dataclasses.replace(
                    base, C_F=choice["C_pF"] * 1e-12, dt_s=dt * 1e-9,
                    C_th_J_per_K=choice["tau_th_ns"] * 1e-9 * base.S_e_W_per_K,
                    resist_params=dataclasses.replace(base.resist_params, gamma=choice["gamma"]),
                )
                print(f"Checking {label} at {dt:g} ns on all {len(measured)} currents", flush=True)
                table, window, voltage = evaluate_candidate(dataset, params, all_indices, settings, measured)
                verification.append(table.assign(candidate=label, dt_ns=dt))
                verification_windows.append(window.assign(candidate=label, dt_ns=dt))
                for i in range(len(measured)):
                    mask = dataset.time_ns >= 0
                    traces.append(pd.DataFrame({"candidate": label, "dt_ns": dt,
                                                "source_setting_mV": dataset.nominal_drives_mV[i],
                                                "current_uA": measured_table.current_uA.iloc[i],
                                                "time_ns": dataset.time_ns[mask],
                                                "measured_voltage_mV": dataset.voltage_mV[mask, i],
                                                "predicted_voltage_mV": voltage[mask, i]}))
        verification = pd.concat(verification, ignore_index=True)
        verification_windows = pd.concat(verification_windows, ignore_index=True)
        traces = pd.concat(traces, ignore_index=True)
        threshold_rows = []
        for floor in (4.0, 6.0, 8.0):
            for retention in (.25, .5, .75):
                for label, table in [("measured", measured_table), *list(archived.groupby("fit_mode"))]:
                    count = ((table.minimum_window_periodic_vpp_mV >= floor)
                             & (table.amplitude_retention >= retention)
                             & (table.late_coherence >= settings.minimum_coherence)).sum()
                    threshold_rows.append({"series": label, "minimum_periodic_vpp_mV": floor,
                                           "minimum_retention": retention, "sustained_records": int(count)})
        threshold = pd.DataFrame(threshold_rows)
        for name, frame in (("measured_metrics", measured_table), ("archived_metrics", archived),
                            ("archived_windows", archived_windows), ("parameter_map", mapped),
                            ("map_windows", windows), ("candidate_ranking", ranking),
                            ("verification", verification), ("verification_windows", verification_windows),
                            ("candidate_traces", traces), ("threshold_sensitivity", threshold)):
            path = bundle.path(f"{name}.csv")
            frame.to_csv(path, index=False)
            bundle.register_file(path.relative_to(bundle.root), label=name.replace("_", " ").title(), media_type="text/csv")
        plot_oscillation_audit(history, settings, list(map(float, audit["source_settings_mV"])),
                               bundle.path("figures/persistence_audit.png"), mode=mode)
        plot_oscillation_parameter_maps(mapped, measured_table, choices,
                                       bundle.path("figures/parameter_maps.png"))
        plot_audit_candidate_traces(traces, list(map(float, audit["source_settings_mV"])),
                                    bundle.path("figures/candidate_comparison.png"), dt_ns=min(steps))
        for name in ("persistence_audit", "parameter_maps", "candidate_comparison"):
            bundle.register_file(f"figures/{name}.png", label=name.replace("_", " ").title(), media_type="image/png")
        verification_summary = []
        for (label, dt), frame in verification.groupby(["candidate", "dt_ns"]):
            truth = measured_table.set_index("source_setting_mV").loc[frame.source_setting_mV, "sustained"].to_numpy(bool)
            predicted = frame.sustained.to_numpy(bool)
            verification_summary.append({"candidate": label, "dt_ns": dt,
                                         "matches": int(np.sum(truth == predicted)),
                                         "misses": int(np.sum(truth & ~predicted)),
                                         "false_positives": int(np.sum(~truth & predicted)),
                                         "sustained_records": int(predicted.sum()),
                                         "oscillatory_feature_score": float(frame.loc[truth, "feature_score"].mean())})
        metrics = {"grid_combinations": len(ranking), "selected_source_settings_mV": audit["source_settings_mV"],
                   "measured_sustained_records": int(measured_table.sustained.sum()),
                   "reference_legacy_records": int(archived.loc[archived.fit_mode == mode, "legacy_detected"].sum()),
                   "reference_sustained_records": int(archived.loc[archived.fit_mode == mode, "sustained"].sum()),
                   "grid_combinations_sustained_at_all_three": len(persistent),
                   "choices": choices, "verification": verification_summary,
                   "settings": dataclasses.asdict(settings)}
        bundle.write_json("metrics.json", metrics, label="Audit findings and verification")
        bundle.write_text("report.md", _oscillation_audit_report(metrics), label="Method and results")
        bundle.complete(summary={key: value for key, value in metrics.items() if key not in {"choices", "settings"}})
    except BaseException as exc:
        bundle.fail(exc)
        raise
    return bundle


def _render_existing_audit(bundle, source, resolved, reference, mode, settings) -> RunBundle:
    """Re-render numerical evidence into a new bundle without mutating the source.

    Exact recipe equality prevents a different grid or physics configuration from
    being presented as a cached simulation. File hashes and the original numerical
    code commit are retained alongside the new rendering provenance.
    """

    manifest = json.loads((source / "run.json").read_text())
    old_config = json.loads((source / "resolved_config.json").read_text())
    if manifest.get("status") != "completed" or manifest.get("model") != "oscillation-audit":
        raise ConfigError("Numerical reuse requires a completed oscillation audit")
    if old_config != resolved:
        raise ConfigError("Numerical reuse requires the identical resolved audit recipe")
    checksums = {}
    for path in sorted(source.glob("*.csv")):
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, bundle.path(path.name))
        bundle.register_file(path.name, label=path.stem.replace("_", " ").title(), media_type="text/csv")
    metrics = json.loads((source / "metrics.json").read_text())
    bundle.manifest["numerical_source"] = {
        "id": manifest["id"], "provenance": manifest["provenance"], "sha256": checksums,
    }
    history = pd.read_csv(reference / "fitted_traces.csv")
    mapped = pd.read_csv(bundle.path("parameter_map.csv"))
    measured = pd.read_csv(bundle.path("measured_metrics.csv"))
    traces = pd.read_csv(bundle.path("candidate_traces.csv"))
    drives = metrics["selected_source_settings_mV"]
    plot_oscillation_audit(history, settings, drives, bundle.path("figures/persistence_audit.png"), mode=mode)
    plot_oscillation_parameter_maps(mapped, measured, metrics["choices"], bundle.path("figures/parameter_maps.png"))
    plot_audit_candidate_traces(traces, drives, bundle.path("figures/candidate_comparison.png"),
                               dt_ns=float(traces.dt_ns.min()))
    for name in ("persistence_audit", "parameter_maps", "candidate_comparison"):
        bundle.register_file(f"figures/{name}.png", label=name.replace("_", " ").title(), media_type="image/png")
    bundle.write_json("metrics.json", metrics, label="Audit findings and verification")
    bundle.write_text("report.md", _oscillation_audit_report(metrics), label="Method and results")
    bundle.complete(summary={key: value for key, value in metrics.items() if key not in {"choices", "settings"}})
    return bundle


def _oscillation_audit_report(metrics: Mapping[str, Any]) -> str:
    """Summarize explicit diagnostic definitions without claiming a new calibration."""

    lines = ["# Sustained-oscillation audit and controlled parameter maps", "",
             "The prior peak-count classification is not evidence of sustained oscillation. "
             f"It accepts {metrics['reference_legacy_records']} reference traces; the window audit accepts "
             f"{metrics['reference_sustained_records']}. The same window audit accepts "
             f"{metrics['measured_sustained_records']} experimental traces.", "",
             "## Method", "",
             "Four windows cover 50–100, 100–150, 150–200 and 200–250 ns. Each reports raw "
             "and 5–95% voltage spans, mean voltage and sinusoidal amplitude. A Hann spectrum "
             "on 150–250 ns locates the dominant component in 10–200 MHz, followed by local "
             "sinusoidal regression with a fitted baseline slope. The 100 ns record has about "
             "10 MHz Fourier-bin spacing: interpolation does not create additional experimental "
             "resolution. A separate peak-period estimate has no 8 ns minimum spacing. Frequencies "
             "of flat/noisy signals are descriptive only and are excluded from the error term.", "",
             "A trace passes the operational persistence check when every window has at least "
             "6 mV fitted fundamental Vpp, the last/first robust span is at least 0.5, and the "
             "late fundamental explains at least 40% of detrended variance. These thresholds "
             "are not physical constants; threshold_sensitivity.csv repeats the audit at "
             "4/6/8 mV and retention 0.25/0.5/0.75. A finite record cannot prove a limit cycle.", "",
             "The maps replay the measured current waveforms at three settings (300, 500, "
             "800 mV source labels; approximately 228, 381, 606 uA measured currents). "
             "They vary electrical C and thermal time Cth/Se at several gamma values. "
             "All other quantities are fixed to the archived reference fit. Cth is calculated "
             "from Se times thermal time, with explicit pJ/K units in parameter_map.csv. "
             "This is a conditional slice, not an exhaustive search over eight parameters. "
             "C above 0.39 pF remains an exploratory violation of the prior timing estimate.", "",
             "Each candidate is ranked by equal-weight squared errors scaled to 20% amplitude "
             "ratio (with a 3 mV floor), 10% frequency, 20 mV mean voltage, and factor-two "
             "retention. There is no large binary classification reward. These are engineering "
             "scales, not measurement uncertainties. The best score, best score passing all "
             "three persistence checks (if any), and best score within the C timing bound are "
             "verified, along with the reference. No fit is selected independently per current.", "",
             "## Results", "",
             f"Mapped {metrics['grid_combinations']} shared combinations; "
             f"{metrics['grid_combinations_sustained_at_all_three']} pass persistence at all three currents.", "",
             "| Candidate | Step (ns) | Sustained | Misses | False positives | Feature score on measured oscillators |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in metrics["verification"]:
        lines.append(f"| {row['candidate']} | {row['dt_ns']:g} | {row['sustained_records']} | {row['misses']} "
                     f"| {row['false_positives']} | {row['oscillatory_feature_score']:.3f} |")
    lines += ["", "All 22 currents are checked at each listed step. The other currents are "
              "cross-current checks, not pristine blind validation: earlier development already "
              "inspected these data. Parameter choices and per-current convergence metrics are "
              "archived. Identical labels alone do not establish waveform convergence. "
              "Historical bundles and their objective are unchanged; this audit supplies the "
              "corrected interpretation.", "", "![Persistence audit](figures/persistence_audit.png)", "",
              "![Parameter maps](figures/parameter_maps.png)", "",
              "![Candidate traces](figures/candidate_comparison.png)"]
    return "\n".join(lines)


def run_joint_inference(config, *, output_root=None, command="neuristor analyze fit-joint") -> RunBundle:
    """Fit static R(T) and dynamic features jointly with a finite simulation budget."""
    from .joint_inference import (NAMES, fit_joint, natural_vector, joint_model,
                                  static_rmse, rt_prediction, predict_features,
                                  voltage_features, feature_loss, combined_feature_loss)
    import matplotlib.pyplot as plt

    base, *_ = _current_params_from_config(config)
    settings = copy.deepcopy(config["joint"])
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else (source_directory(config)/path).resolve()
    data_path = resolve(settings["data_directory"])
    rt_path = resolve(settings["resistance_data"])
    frames, _ = load_converted_sweep(data_path)
    train = settings["training_drives_mV"]
    holdout = [x for x in sorted(frames.nominal_drive_mV.unique()) if x not in train]
    dataset = prepare_inference_dataset(frames, holdout_drives_mV=holdout)
    rt = load_experimental_rt(rt_path)
    seeds = [base]
    seed_labels = ["frozen"]
    seed_paths = [resolve(p) for p in settings["seed_parameter_tables"]]
    for label, path in zip(("prior_anchored", "prior_amplitude"), seed_paths):
        table = pd.read_csv(path).set_index("parameter")
        seed = model_from_parameter_vector(table.loc[list(PARAMETER_NAMES), "relaxed_fit"].to_numpy(), base,
                                           dt_ns=settings["search_dt_ns"])
        seeds.append(seed)
        seed_labels.append(label)
    for spec in settings.get("joint_seeds", []):
        path = resolve(spec["table"])
        row = pd.read_csv(path).set_index("candidate").loc[spec["candidate"]]
        seeds.append(joint_model(row[list(NAMES)].to_numpy(float), base, settings["search_dt_ns"]))
        seed_labels.append(spec["label"])
        seed_paths.append(path)
    if len(seeds) > int(settings["population"]):
        raise ValueError("Population must accommodate the named seeds")
    bundle = RunBundle.create(name=config["name"], model="joint-inference", kind="analysis",
                              config=resolved_copy(config), output_root=output_root or config.get("output", {}).get("root", "runs"),
                              command=command)
    try:
        files = [rt_path, Path(config["_source"]), *seed_paths, *data_path.glob("*_converted.csv")]
        files += list(Path(__file__).parent.glob("*.py"))
        bundle.write_json("source_hashes.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, label="Input and source hashes")
        # Preserve executable sources because the parent worktree may be dirty.
        for path in Path(__file__).parent.glob("*.py"):
            target = bundle.path(f"source/neuristor/{path.name}")
            shutil.copy2(path, target)
            bundle.register_file(f"source/neuristor/{path.name}", label=f"Source: {path.name}")
        fits, history, budget = fit_joint(dataset, rt, base, settings, seeds,
                    progress=lambda mode, count, best, unique: print(f"Joint {mode}: {count} objective calls, {unique} unique candidates; best={best:.4g}", flush=True))
        history.to_csv(bundle.add_artifact("optimization_history.csv", label="Every objective evaluation"), index=False)
        candidates = dict(zip(seed_labels, map(natural_vector, seeds)))
        candidates.update({name: fit["values"] for name, fit in fits.items()})
        params_table = pd.DataFrame([dict(candidate=name, **dict(zip(NAMES, v)),
            C_th_pJ_per_K=v[1]*v[2], R0_ohm=v[8]*np.exp(-v[9]/315),
            static_rmse_log10=static_rmse(v, rt)) for name, v in candidates.items()])
        params_table.to_csv(bundle.add_artifact("parameters.csv", label="Shared fitted and reference parameters"), index=False)
        measured = [voltage_features(dataset.time_ns, v) for v in dataset.voltage_mV.T]
        rows, window_rows, trace_rows = [], [], []
        indices = np.arange(len(dataset.nominal_drives_mV))
        for name, values in candidates.items():
            for dt in settings["verification_dt_ns"]:
                features, predictions, outside = predict_features(dataset, joint_model(values, base, dt), indices)
                for i, (summary, windows) in enumerate(features):
                    ms = measured[i][0]
                    errors = feature_loss(measured[i], features[i], missing_frequency_penalty=float(settings.get("missing_frequency_penalty", 0)))
                    meta = dict(candidate=name, dt_ns=dt, source_mV=dataset.nominal_drives_mV[i],
                                split="train" if i in dataset.train_indices else "validation")
                    rows.append(dict(**meta, outside_domain=outside,
                                     **{f"measured_{k}": v for k, v in ms.items()},
                                     **{f"predicted_{k}": v for k, v in summary.items()},
                                     feature_loss=combined_feature_loss(errors, settings)))
                    window_rows.append(windows.assign(**meta))
                    if dt == min(settings["verification_dt_ns"]):
                        keep = dataset.time_ns <= 250
                        trace_rows.append(pd.DataFrame(dict(time_ns=dataset.time_ns[keep],
                            measured_mV=dataset.voltage_mV[keep, i], predicted_mV=predictions[:, i])).assign(**meta))
                print(f"Verified {name}, dt={dt} ns on all {len(indices)} records", flush=True)
        results = pd.DataFrame(rows)
        results.to_csv(bundle.add_artifact("verification.csv", label="All-current metrics at each timestep"), index=False)
        pd.concat(window_rows).to_csv(bundle.add_artifact("windows.csv", label="Four-window persistence evidence"), index=False)
        traces = pd.concat(trace_rows)
        traces.to_csv(bundle.add_artifact("traces.csv", label="Finest measured and predicted traces"), index=False)
        summaries = []
        for (name, dt, split), part in results.groupby(["candidate", "dt_ns", "split"]):
            osc = part.measured_sustained.astype(bool)
            pred = part.predicted_sustained.astype(bool)
            comparable = osc & pred
            summaries.append(dict(candidate=name, dt_ns=dt, split=split, records=len(part),
                mean_feature_loss=part.feature_loss.mean(),
                mean_voltage_RMSE_mV=np.sqrt(np.mean((part.predicted_late_mean_mV-part.measured_late_mean_mV)**2)),
                periodic_amplitude_MAE_mV=np.mean(np.abs(part.predicted_late_periodic_vpp_mV-part.measured_late_periodic_vpp_mV)),
                oscillators_recovered=int((osc & pred).sum()), measured_oscillators=int(osc.sum()),
                false_positives=int((~osc & pred).sum()), frequency_comparable_count=int(comparable.sum()),
                frequency_MAE_MHz=float(np.mean(np.abs(part.loc[comparable, "predicted_late_frequency_MHz"]-part.loc[comparable, "measured_late_frequency_MHz"]))) if comparable.any() else None))
        summary = pd.DataFrame(summaries)
        summary.to_csv(bundle.add_artifact("summary.csv", label="Training and validation performance"), index=False)
        rt_table = rt[["Temperature", "Resistance"]].copy()
        for name, values in candidates.items():
            rt_table[name] = rt_prediction(values, rt)
        rt_table.to_csv(bundle.add_artifact("resistance_fit.csv", label="Static-data tradeoff"), index=False)
        # Comparison figures use identical metrics, samples and subsets.
        finest = results[results.dt_ns == min(settings["verification_dt_ns"])]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        reference = finest[finest.candidate == "frozen"]
        axes[0].plot(rt.Temperature, rt.Resistance, ".", color="black", ms=2, label="Measured")
        axes[1].plot(reference.source_mV, reference.measured_late_mean_mV, "k.-", label="Measured")
        axes[2].plot(reference.source_mV, reference.measured_late_periodic_vpp_mV, "k.-", label="Measured")
        for name in ("frozen", *fits):
            part = finest[finest.candidate == name]
            axes[0].plot(rt.Temperature, rt_table[name], lw=1, label=name.replace("_", " "))
            axes[1].plot(part.source_mV, part.predicted_late_mean_mV, ".-", label=name.replace("_", " "))
            axes[2].plot(part.source_mV, part.predicted_late_periodic_vpp_mV, ".-", label=name.replace("_", " "))
        axes[0].set(yscale="log", xlabel="Temperature (K)", ylabel="Resistance (Ω)", title="Same-device static R(T)")
        axes[1].set(xlabel="Source setting (mV; record identifier)", ylabel="Late mean (mV)", title="Mean voltage across all records")
        axes[2].set(xlabel="Source setting (mV; record identifier)", ylabel="Late periodic Vpp (mV)", title="Oscillation amplitude")
        axes[0].legend(fontsize=7)
        fig.savefig(bundle.add_artifact("figures/joint_summary.png", label="Joint-fit tradeoff figure"), dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
        for ax, drive in zip(axes.flat, (250, 300, 500, 800)):
            part = traces[(traces.source_mV == drive) & (traces.time_ns >= 50)]
            m = part[part.candidate == "frozen"]
            ax.plot(m.time_ns, m.measured_mV, "k", lw=1.2, label="Measured")
            for name in fits:
                x = part[part.candidate == name]
                ax.plot(x.time_ns, x.predicted_mV, lw=.8, label=name.replace("_", " "))
            ax.set(title=f"Source setting {drive} mV", xlabel="Time (ns)", ylabel="Voltage (mV)")
        axes[0, 0].legend(fontsize=8)
        fig.savefig(bundle.add_artifact("figures/joint_traces.png", label="Representative finest-step waveforms"), dpi=180)
        plt.close(fig)
        metrics = dict(**budget, objective_calls=len(history), fitted_parameters=list(NAMES),
                       training_drives_mV=train, validation_drives_mV=holdout,
                       fits={name: {**fit, "values": fit["values"].tolist()} for name, fit in fits.items()},
                       verification=summary.to_dict("records"))
        bundle.write_json("metrics.json", metrics, label="Budget, termination and validation")
        bundle.write_text("report.md", "# Budgeted joint R(T)/waveform inference\n\n"
            "One shared vector fits all static measurements and nine configured representative current records. "
            "Other currents are excluded from optimization, but were seen in earlier research and are not pristine blind data. "
            "The eleven free quantities include all six major-loop parameters, gamma, ambient temperature, electrical capacitance and two thermal quantities.\n\n"
            "Mixed log/linear unit-cube coordinates reduce scale disparities. Rs at 315 K replaces the correlated R0 coordinate; "
            "thermal time replaces Cth. Both searches share a feasible Latin-hypercube/perturbed historical-seed population and cached evaluations. "
            "Differential evolution is followed by a strictly capped Powell refinement; no new optimizer dependency is required.\n\n"
            "Waveform loss = mean-voltage term + 3 × windowed periodic-amplitude term + robust-amplitude term + 0.5 × gated-frequency term. "
            "Mean scale is 20 mV, frequency scale 10 MHz, amplitudes use log ratios with a 3 mV floor. "
            "Static loss = weight × (log10 R RMSE / 0.05)^2. Weights, bounds, static rejection ceiling and budgets are in resolved_config.json. "
            "These are engineering preferences, not a noise likelihood or confidence intervals.\n\n"
            f"Budget: {len(history)} objective calls; {budget['unique_candidates']} unique candidates; "
            f"{budget['simulations']} search simulations; {budget['elapsed_s']:.1f} search seconds.\n\n"
            "Inspect summary.csv by split and timestep; parameters.csv records static-fit degradation. "
            f"Additional persistence weight: {settings.get('persistence_weight', 0)}; missing-frequency penalty: {settings.get('missing_frequency_penalty', 0)}. "
            "The persistence deficit penalizes missing window amplitude, retention below 0.5 and coherence below 0.4 on measured oscillators. "
            "Frequency error is reported only where both measured and predicted persistence pass; the denominator is explicit. "
            "All final candidates are checked on all currents at every configured timestep. "
            "Neither early termination nor a lower objective proves global optimality or a physical calibration. "
            "Source snapshots and hashes preserve the dirty-worktree implementation.\n", label="Methods and limitations")
        bundle.complete(summary={"objective_calls": len(history), **budget})
    except Exception as exc:
        bundle.fail(exc)
        raise
    return bundle


def run_waveform_parameter_inference(
    config: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    command: str = "neuristor analyze fit-waveforms",
) -> RunBundle:
    """Fit one shared parameter vector to the laboratory current sweep."""

    validate_config(config)
    if str(config.get("model", "")).lower() != "current":
        raise ConfigError("Waveform inference requires model='current'")
    base, _, _, resistance_source, start_branch = _current_params_from_config(config)
    inference = _table(config, "inference")
    data_directory = Path(str(inference.get("data_directory", ""))).expanduser()
    if not data_directory.is_absolute():
        data_directory = (source_directory(config) / data_directory).resolve()
    if not data_directory.is_dir():
        raise ConfigError(f"Laboratory data directory does not exist: {data_directory}")

    def parse_parameters(key: str) -> tuple[FitParameter, ...]:
        rows = inference.get(key)
        if not isinstance(rows, list) or not rows:
            raise ConfigError(f"inference.{key} must be an array of parameter tables")
        parameters = tuple(
            FitParameter(
                name=str(row["name"]),
                lower=float(row["lower"]),
                upper=float(row["upper"]),
                prior_center=(float(row["prior_center"]) if "prior_center" in row else None),
                prior_scale=(float(row["prior_scale"]) if "prior_scale" in row else None),
                initial_value=(float(row["initial_value"]) if "initial_value" in row else None),
            )
            for row in rows
        )
        if tuple(parameter.name for parameter in parameters) != PARAMETER_NAMES:
            raise ConfigError(f"inference.{key} must be ordered as {PARAMETER_NAMES}")
        if any(parameter.lower >= parameter.upper for parameter in parameters):
            raise ConfigError(f"inference.{key} contains an invalid bound")
        return parameters

    constrained_parameters = parse_parameters("constrained_parameters")
    relaxed_parameters = parse_parameters("relaxed_parameters")
    weight_config = _table(inference, "weights")
    weights = ObjectiveWeights(
        **{
            field: float(weight_config.get(field, getattr(ObjectiveWeights(), field)))
            for field in ObjectiveWeights.__dataclass_fields__
        }
    )
    optimizer = _table(inference, "optimizer")
    search_dt_ns = float(inference.get("search_dt_ns", 0.2))
    final_dt_ns = float(inference.get("final_dt_ns", 0.05))
    if search_dt_ns <= 0.0 or final_dt_ns <= 0.0:
        raise ConfigError("Inference time steps must be positive")
    seed = int(config.get("seed", 0))
    resolved = resolved_copy(config)
    resolved.setdefault("inference", {})["data_directory"] = str(data_directory)
    resolved.setdefault("resistance", {})["resolved_source"] = resistance_source
    output = output_root or _table(config, "output").get("root", "runs")
    bundle = RunBundle.create(
        name=str(config["name"]),
        model="global-waveform-parameter-inference",
        kind="analysis",
        config=resolved,
        output_root=output,
        command=command,
    )
    try:
        lab_traces, _ = load_converted_sweep(data_directory)
        dataset = prepare_inference_dataset(
            lab_traces,
            holdout_drives_mV=tuple(float(value) for value in inference.get("holdout_drives_mV", ())),
        )
        baseline_values = parameter_vector_from_model(base)
        constrained_fit = fit_parameter_set(
            "constrained",
            constrained_parameters,
            base,
            dataset,
            search_dt_ns=search_dt_ns,
            weights=weights,
            include_prior=True,
            seed=seed,
            maxiter=int(optimizer.get("maxiter_constrained", 5)),
            popsize=int(optimizer.get("popsize", 4)),
            local_max_evaluations=int(optimizer.get("local_max_evaluations", 100)),
        )
        relaxed_fit = fit_parameter_set(
            "relaxed",
            relaxed_parameters,
            base,
            dataset,
            search_dt_ns=search_dt_ns,
            weights=weights,
            include_prior=False,
            seed=seed + 1,
            maxiter=int(optimizer.get("maxiter_relaxed", 8)),
            popsize=int(optimizer.get("popsize", 4)),
            local_max_evaluations=int(optimizer.get("local_max_evaluations", 100)),
        )

        fit_definitions = (
            ("baseline", baseline_values, constrained_parameters, True),
            ("constrained", constrained_fit.values, constrained_parameters, True),
            ("relaxed", relaxed_fit.values, relaxed_parameters, False),
        )
        evaluations = {}
        fit_summary_rows: list[dict[str, Any]] = []
        trace_metrics: list[pd.DataFrame] = []
        prediction_traces: list[pd.DataFrame] = []
        for mode, values, parameters, include_prior in fit_definitions:
            evaluation = evaluate_fitted_model(
                mode,
                values,
                parameters,
                base,
                dataset,
                dt_ns=final_dt_ns,
                weights=weights,
                include_prior=include_prior,
            )
            evaluations[mode] = evaluation
            metrics = evaluation.trace_metrics.copy()
            metrics.insert(0, "fit_mode", mode)
            trace_metrics.append(metrics)
            prediction_traces.append(evaluation.traces)
            for split_name, indices in (
                ("train", dataset.train_indices),
                ("test", dataset.test_indices),
                ("all", np.arange(len(dataset.source_files), dtype=int)),
            ):
                objective, split_metrics, _, _ = evaluate_parameter_vector(
                    values,
                    parameters,
                    base,
                    dataset,
                    indices,
                    dt_ns=final_dt_ns,
                    weights=weights,
                    include_prior=include_prior,
                )
                fit_summary_rows.append(
                    {
                        "fit_mode": mode,
                        "split": split_name,
                        "traces": int(len(indices)),
                        "classification_matches": int(
                            np.sum(split_metrics["measured_oscillation"] == split_metrics["predicted_oscillation"])
                        ),
                        "predicted_oscillating_traces": int(split_metrics["predicted_oscillation"].sum()),
                        **{f"objective_{name}": float(value) for name, value in objective.items()},
                    }
                )
        fit_summary = pd.DataFrame(fit_summary_rows)
        trace_metrics_frame = pd.concat(trace_metrics, ignore_index=True)
        prediction_trace_frame = pd.concat(prediction_traces, ignore_index=True)

        parameter_rows: list[dict[str, Any]] = []
        for index, name in enumerate(PARAMETER_NAMES):
            physical = constrained_parameters[index]
            relaxed_value = float(relaxed_fit.values[index])
            parameter_rows.append(
                {
                    "parameter": name,
                    "original_estimate": float(baseline_values[index]),
                    "constrained_fit": float(constrained_fit.values[index]),
                    "physical_lower": float(physical.lower),
                    "physical_upper": float(physical.upper),
                    "relaxed_fit": relaxed_value,
                    "relaxed_outside_physical_bounds": bool(
                        relaxed_value < physical.lower or relaxed_value > physical.upper
                    ),
                }
            )
        parameter_table = pd.DataFrame(parameter_rows)
        history = pd.concat(
            [
                constrained_fit.history.assign(fit_mode="constrained"),
                relaxed_fit.history.assign(fit_mode="relaxed"),
            ],
            ignore_index=True,
        )

        convergence_rows: list[dict[str, Any]] = []
        for mode, values, parameters, include_prior in fit_definitions:
            for dt_ns in sorted(set((search_dt_ns, 0.1, final_dt_ns, final_dt_ns / 2.0)), reverse=True):
                objective, metrics, _, _ = evaluate_parameter_vector(
                    values,
                    parameters,
                    base,
                    dataset,
                    np.arange(len(dataset.source_files), dtype=int),
                    dt_ns=dt_ns,
                    weights=weights,
                    include_prior=include_prior,
                )
                convergence_rows.append(
                    {
                        "fit_mode": mode,
                        "dt_ns": float(dt_ns),
                        "objective_total": float(objective["total"]),
                        "classification_matches": int(
                            np.sum(metrics["measured_oscillation"] == metrics["predicted_oscillation"])
                        ),
                        "predicted_oscillating_traces": int(metrics["predicted_oscillation"].sum()),
                    }
                )
        convergence = pd.DataFrame(convergence_rows)

        parameter_table.to_csv(
            bundle.add_artifact("parameter_comparison.csv", label="Original and fitted shared parameters", media_type="text/csv"),
            index=False,
        )
        fit_summary.to_csv(
            bundle.add_artifact("fit_summary.csv", label="Train, test, and full objective summary", media_type="text/csv"),
            index=False,
        )
        trace_metrics_frame.to_csv(
            bundle.add_artifact("trace_metrics.csv", label="Per-trace fit evidence", media_type="text/csv"),
            index=False,
        )
        prediction_trace_frame.to_csv(
            bundle.add_artifact("fitted_traces.csv", label="Measured and globally fitted waveforms", media_type="text/csv"),
            index=False,
        )
        history.to_csv(
            bundle.add_artifact("optimization_history.csv", label="Optimizer evaluation history", media_type="text/csv"),
            index=False,
        )
        convergence.to_csv(
            bundle.add_artifact("convergence.csv", label="Fine-step fit verification", media_type="text/csv"),
            index=False,
        )
        plot_inference_optimization(
            history,
            bundle.add_artifact("figures/optimization_history.png", label="Optimization convergence", media_type="image/png"),
        )
        plot_inference_operating_summary(
            trace_metrics_frame,
            bundle.add_artifact("figures/operating_summary.png", label="Measured and fitted operating features", media_type="image/png"),
        )
        plot_inference_representative_traces(
            prediction_trace_frame,
            bundle.add_artifact("figures/representative_fits.png", label="Representative globally fitted traces", media_type="image/png"),
        )

        all_summary = fit_summary.loc[fit_summary["split"] == "all"].set_index("fit_mode")
        relaxed_conflicts = parameter_table.loc[parameter_table["relaxed_outside_physical_bounds"], "parameter"].tolist()
        metrics = {
            "training_traces": int(len(dataset.train_indices)),
            "held_out_traces": int(len(dataset.test_indices)),
            "holdout_drives_mV": dataset.nominal_drives_mV[dataset.test_indices].tolist(),
            "search_dt_ns": search_dt_ns,
            "final_dt_ns": final_dt_ns,
            "objective_weights": {
                field: float(getattr(weights, field)) for field in ObjectiveWeights.__dataclass_fields__
            },
            "baseline_objective_all": float(all_summary.loc["baseline", "objective_total"]),
            "constrained_objective_all": float(all_summary.loc["constrained", "objective_total"]),
            "relaxed_objective_all": float(all_summary.loc["relaxed", "objective_total"]),
            "baseline_classification_matches": int(all_summary.loc["baseline", "classification_matches"]),
            "constrained_classification_matches": int(all_summary.loc["constrained", "classification_matches"]),
            "relaxed_classification_matches": int(all_summary.loc["relaxed", "classification_matches"]),
            "constrained_predicted_oscillating_traces": int(all_summary.loc["constrained", "predicted_oscillating_traces"]),
            "relaxed_predicted_oscillating_traces": int(all_summary.loc["relaxed", "predicted_oscillating_traces"]),
            "constrained_evaluations": int(constrained_fit.evaluations),
            "relaxed_evaluations": int(relaxed_fit.evaluations),
            "constrained_optimizer_messages": {
                "global": constrained_fit.differential_evolution_message,
                "local": constrained_fit.local_message,
            },
            "relaxed_optimizer_messages": {
                "global": relaxed_fit.differential_evolution_message,
                "local": relaxed_fit.local_message,
            },
            "relaxed_parameters_outside_physical_bounds": relaxed_conflicts,
            "resistance_source": resistance_source,
            "start_branch": start_branch,
        }
        bundle.write_json("metrics.json", metrics, label="Global inference summary")
        conflicts = ", ".join(relaxed_conflicts) if relaxed_conflicts else "none"
        report = f"""# {config['name']}

One shared eight-parameter vector was fitted to {metrics['training_traces']} traces;
the source settings {metrics['holdout_drives_mV']} mV were excluded from optimization
and used only for validation. The objective combines normalized waveform RMSE,
phase-tolerant oscillatory shape, plateau mean and amplitude, spectrum, frequency,
oscillation classification, sustained periodic amplitude across four plateau
segments, the 0--50 ns edge, and (for the constrained fit) weak independent-
measurement priors. The exact weights are archived in `resolved_config.json` and
`metrics.json`.

The original estimates give a full-data objective of
**{metrics['baseline_objective_all']:.5g}** and classify
**{metrics['baseline_classification_matches']}/22** traces correctly. The physically
constrained fit gives **{metrics['constrained_objective_all']:.5g}**,
**{metrics['constrained_classification_matches']}/22** correct classifications, and
predicts **{metrics['constrained_predicted_oscillating_traces']}** oscillating traces.
The relaxed diagnostic fit gives **{metrics['relaxed_objective_all']:.5g}**,
**{metrics['relaxed_classification_matches']}/22** correct classifications, and
predicts **{metrics['relaxed_predicted_oscillating_traces']}** oscillating traces.

Relaxed values outside the independently allowed ranges: **{conflicts}**. These are
effective values required by the present equations, not measurements. The held-out
traces and fine-step reruns distinguish generalization from memorization and numerical
step artifacts. No confidence intervals are assigned: this deterministic global fit
is an identifiability diagnostic, and several parameters remain correlated.
"""
        bundle.write_text("report.md", report, label="Scientific report")
        bundle.complete(summary=metrics)
    except BaseException as exc:
        bundle.fail(exc)
        raise
    return bundle
