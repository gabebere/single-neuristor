"""Small, consistent plotting surface for run bundles and the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from .model import HysteresisArray, YuanhangResistParams


COLORS = {
    "ink": "#172033",
    "blue": "#2563eb",
    "green": "#059669",
    "orange": "#ea580c",
    "purple": "#7c3aed",
    "red": "#dc2626",
    "grid": "#dbe3ef",
}


def _finish(fig: plt.Figure, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_current_run(frame: pd.DataFrame, out_path: str | Path, *, title: str) -> Path:
    """Plot imposed current, voltage, temperature, and resistance for one run."""

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.4), sharex=True, gridspec_kw={"height_ratios": [0.8, 1.35, 1.2]})
    axes[0].plot(frame["time_us"], frame["current_uA"], color=COLORS["ink"], linewidth=1.5)
    axes[0].set_ylabel("Current (uA)")
    axes[1].plot(frame["time_us"], frame["voltage_V"], color=COLORS["blue"], linewidth=1.2)
    axes[1].axhline(0.0, color=COLORS["ink"], linewidth=0.8)
    axes[1].set_ylabel("Voltage (V)")
    axes[2].plot(frame["time_us"], frame["temperature_K"], color=COLORS["orange"], linewidth=1.1, label="Temperature")
    axes[2].set_ylabel("Temperature (K)", color=COLORS["orange"])
    resistance_ax = axes[2].twinx()
    resistance_ax.plot(frame["time_us"], frame["resistance_ohm"], color=COLORS["purple"], linewidth=1.0, alpha=0.8)
    resistance_ax.set_ylabel("Resistance (Ohm)", color=COLORS["purple"])
    resistance_ax.set_yscale("log")
    axes[2].set_xlabel("Time (us)")
    for axis in axes:
        axis.grid(True, color=COLORS["grid"], alpha=0.7)
    fig.suptitle(title, fontsize=15, color=COLORS["ink"])
    return _finish(fig, out_path)


def _current_trace_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Return validated arrays required by current-drive trajectory plots."""

    columns = ["time_us", "current_uA", "voltage_V", "temperature_K", "resistance_ohm"]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Current-drive trace is missing columns: {', '.join(missing)}")
    arrays = tuple(frame[column].to_numpy(dtype=float) for column in columns)
    finite = np.logical_and.reduce([np.isfinite(values) for values in arrays])
    positive_resistance = arrays[-1] > 0.0
    keep = finite & positive_resistance
    if np.count_nonzero(keep) < 2:
        raise ValueError("Current-drive trace needs at least two finite samples with positive resistance")
    return tuple(values[keep] for values in arrays)


def _yuanhang_major_branches() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the published Yuanhang major heating and cooling branches."""

    params = YuanhangResistParams()
    temperature_K = np.linspace(params.T_min_K, params.T_max_K, 600, dtype=np.float32)
    branch_resistance: list[np.ndarray] = []
    for branch in ("insulator", "metal"):
        hysteresis = HysteresisArray(params, len(temperature_K), start_branch=branch)
        hysteresis.initialize(temperature_K)
        resistance_ohm, _ = hysteresis.evaluate(temperature_K)
        branch_resistance.append(np.asarray(resistance_ohm, dtype=float))
    return np.asarray(temperature_K, dtype=float), branch_resistance[0], branch_resistance[1]


def _plot_yuanhang_hysteresis_background(ax: plt.Axes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw the Yuanhang major loop as a quiet reference behind a trajectory."""

    reference_temperature_K, heating_resistance_ohm, cooling_resistance_ohm = _yuanhang_major_branches()
    ax.fill_between(
        reference_temperature_K,
        cooling_resistance_ohm,
        heating_resistance_ohm,
        color=COLORS["grid"],
        alpha=0.38,
        zorder=0,
    )
    ax.plot(
        reference_temperature_K,
        heating_resistance_ohm,
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1.35,
        alpha=0.72,
        label="Yuanhang heating branch",
        zorder=1,
    )
    ax.plot(
        reference_temperature_K,
        cooling_resistance_ohm,
        color=COLORS["blue"],
        linestyle="--",
        linewidth=1.35,
        alpha=0.72,
        label="Yuanhang cooling branch",
        zorder=1,
    )
    return reference_temperature_K, heating_resistance_ohm, cooling_resistance_ohm


def plot_resistance_temperature_trajectory(
    frame: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str = "Resistance-temperature trajectory",
) -> Path:
    """Plot the simulated R(T) path with color indicating elapsed time."""

    time_us, _, _, temperature_K, resistance_ohm = _current_trace_arrays(frame)
    points = np.column_stack([temperature_K, resistance_ohm]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_time = 0.5 * (time_us[:-1] + time_us[1:])
    norm = Normalize(vmin=float(time_us.min()), vmax=float(time_us.max()))

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    _plot_yuanhang_hysteresis_background(ax)
    trajectory = LineCollection(segments, cmap="viridis", norm=norm, linewidth=1.8)
    trajectory.set_array(segment_time)
    trajectory.set_zorder(3)
    ax.add_collection(trajectory)
    ax.scatter(
        [temperature_K[0], temperature_K[-1]],
        [resistance_ohm[0], resistance_ohm[-1]],
        c=[COLORS["ink"], COLORS["red"]],
        s=42,
        zorder=4,
    )
    ax.annotate("start", (temperature_K[0], resistance_ohm[0]), xytext=(6, 6), textcoords="offset points")
    ax.annotate("end", (temperature_K[-1], resistance_ohm[-1]), xytext=(6, -13), textcoords="offset points")
    ax.autoscale()
    ax.margins(x=0.04, y=0.08)
    ax.set_yscale("log")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Equivalent resistance (Ohm)")
    ax.set_title(title)
    ax.grid(True, which="both", color=COLORS["grid"], alpha=0.75)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.colorbar(trajectory, ax=ax, label="Time (us)")
    return _finish(fig, out_path)


def animate_current_resistance_temperature(
    frame: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str = "Current-drive electrothermal evolution",
    frame_count: int = 96,
    duration_s: float = 10.0,
) -> Path:
    """Animate current/voltage time traces beside the simultaneous R(T) path."""

    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    time_us, current_uA, voltage_V, temperature_K, resistance_ohm = _current_trace_arrays(frame)

    # A few thousand points preserve the waveform while keeping GIF rendering and
    # repository size manageable for long, sub-nanosecond simulations.
    display_indices = np.unique(np.linspace(0, len(time_us) - 1, min(len(time_us), 5000), dtype=int))
    time_us = time_us[display_indices]
    current_uA = current_uA[display_indices]
    voltage_V = voltage_V[display_indices]
    temperature_K = temperature_K[display_indices]
    resistance_ohm = resistance_ohm[display_indices]
    animation_indices = np.unique(np.linspace(0, len(time_us) - 1, min(frame_count, len(time_us)), dtype=int))

    fig, (wave_ax, rt_ax) = plt.subplots(1, 2, figsize=(12.2, 5.2))
    voltage_ax = wave_ax.twinx()
    wave_ax.plot(time_us, current_uA, color=COLORS["ink"], linewidth=0.9, alpha=0.18)
    voltage_ax.plot(time_us, voltage_V, color=COLORS["blue"], linewidth=0.9, alpha=0.18)
    (current_line,) = wave_ax.plot([], [], color=COLORS["ink"], linewidth=1.7, label="Current")
    (voltage_line,) = voltage_ax.plot([], [], color=COLORS["blue"], linewidth=1.6, label="Voltage")
    cursor = wave_ax.axvline(time_us[0], color=COLORS["red"], linewidth=1.0, alpha=0.75)
    wave_ax.set_xlim(float(time_us.min()), float(time_us.max()))
    wave_ax.set_ylim(min(0.0, float(current_uA.min())), max(1.0, float(current_uA.max()) * 1.08))
    voltage_ax.set_ylim(min(0.0, float(voltage_V.min())), max(1e-6, float(voltage_V.max()) * 1.08))
    wave_ax.set_xlabel("Time (us)")
    wave_ax.set_ylabel("Current (uA)", color=COLORS["ink"])
    voltage_ax.set_ylabel("Voltage (V)", color=COLORS["blue"])
    wave_ax.grid(True, color=COLORS["grid"], alpha=0.75)
    wave_ax.legend([current_line, voltage_line], ["Current", "Voltage"], loc="upper right")

    reference_temperature_K, heating_resistance_ohm, cooling_resistance_ohm = (
        _plot_yuanhang_hysteresis_background(rt_ax)
    )
    rt_ax.plot(temperature_K, resistance_ohm, color=COLORS["purple"], linewidth=1.0, alpha=0.18, zorder=2)
    (rt_line,) = rt_ax.plot([], [], color=COLORS["purple"], linewidth=1.8)
    (rt_point,) = rt_ax.plot([], [], "o", color=COLORS["red"], markersize=6)
    rt_ax.set_xlim(float(reference_temperature_K.min()) - 1.0, float(reference_temperature_K.max()) + 1.0)
    reference_resistance = np.concatenate([heating_resistance_ohm, cooling_resistance_ohm, resistance_ohm])
    rt_ax.set_ylim(float(reference_resistance.min()) * 0.85, float(reference_resistance.max()) * 1.18)
    rt_ax.set_yscale("log")
    rt_ax.set_xlabel("Temperature (K)")
    rt_ax.set_ylabel("Equivalent resistance (Ohm)")
    rt_ax.grid(True, which="both", color=COLORS["grid"], alpha=0.75)
    rt_ax.legend(loc="upper right", fontsize=7.5)
    time_label = rt_ax.text(0.02, 0.98, "", transform=rt_ax.transAxes, ha="left", va="top")
    fig.suptitle(title, fontsize=14, color=COLORS["ink"])
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    def update(sample_index: int) -> tuple[plt.Artist, ...]:
        stop = int(sample_index) + 1
        current_line.set_data(time_us[:stop], current_uA[:stop])
        voltage_line.set_data(time_us[:stop], voltage_V[:stop])
        cursor.set_xdata([time_us[sample_index], time_us[sample_index]])
        rt_line.set_data(temperature_K[:stop], resistance_ohm[:stop])
        rt_point.set_data([temperature_K[sample_index]], [resistance_ohm[sample_index]])
        time_label.set_text(f"t = {time_us[sample_index]:.2f} us")
        return current_line, voltage_line, cursor, rt_line, rt_point, time_label

    animation = FuncAnimation(fig, update, frames=animation_indices, interval=1000 * duration_s / len(animation_indices))
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=len(animation_indices) / duration_s), dpi=105)
    plt.close(fig)
    return path


def plot_voltage_run(frame: pd.DataFrame, out_path: str | Path, *, title: str) -> Path:
    """Plot source/node voltage, VO2 current, temperature, and resistance."""

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.4), sharex=True)
    axes[0].plot(frame["time_us"], frame["source_voltage_V"], color=COLORS["ink"], label="Source")
    axes[0].plot(frame["time_us"], frame["voltage_V"], color=COLORS["blue"], label="VO2 node")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].legend(loc="upper right")
    axes[1].plot(frame["time_us"], frame["current_mA"], color=COLORS["green"])
    axes[1].set_ylabel("VO2 current (mA)")
    axes[2].plot(frame["time_us"], frame["temperature_K"], color=COLORS["orange"], label="Temperature")
    axes[2].set_ylabel("Temperature (K)", color=COLORS["orange"])
    resistance_ax = axes[2].twinx()
    resistance_ax.plot(frame["time_us"], frame["resistance_ohm"], color=COLORS["purple"], alpha=0.8)
    resistance_ax.set_yscale("log")
    resistance_ax.set_ylabel("Resistance (Ohm)", color=COLORS["purple"])
    axes[2].set_xlabel("Time (us)")
    for axis in axes:
        axis.grid(True, color=COLORS["grid"], alpha=0.7)
    fig.suptitle(title, fontsize=15, color=COLORS["ink"])
    return _finish(fig, out_path)


def plot_sweep_summary(frame: pd.DataFrame, axes: Iterable[str], out_path: str | Path, *, title: str) -> Path:
    """Plot a line, heatmap, or faceted heatmaps for one to three axes."""

    axis_names = list(axes)
    if len(axis_names) == 1:
        fig, ax = plt.subplots(figsize=(9.4, 5.8))
        x = axis_names[0]
        ordered = frame.sort_values(x)
        ax.plot(ordered[x], ordered["frequency_MHz"], "o-", color=COLORS["blue"])
        ax.set_xlabel(x)
        ax.set_ylabel("Frequency (MHz)")
        ax.grid(True, color=COLORS["grid"])
        ax.set_title(title)
        return _finish(fig, out_path)
    if len(axis_names) == 2:
        x, y = axis_names
        pivot = frame.pivot_table(index=y, columns=x, values="frequency_MHz", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(9.4, 6.8))
        image = ax.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{value:g}" for value in pivot.columns], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{value:g}" for value in pivot.index])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label="Frequency (MHz)")
        return _finish(fig, out_path)
    if len(axis_names) == 3:
        facet, x, y = axis_names
        facet_values = sorted(frame[facet].dropna().unique())
        columns = min(3, len(facet_values))
        rows = int(math.ceil(len(facet_values) / columns))
        fig, panel = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.4 * rows), squeeze=False)
        finite = frame["frequency_MHz"].to_numpy(dtype=float)
        vmax = float(np.nanmax(finite)) if np.any(np.isfinite(finite)) else 1.0
        image = None
        for axis, value in zip(panel.flat, facet_values):
            subset = frame[frame[facet] == value]
            pivot = subset.pivot_table(index=y, columns=x, values="frequency_MHz", aggfunc="mean")
            image = axis.imshow(
                pivot.to_numpy(dtype=float),
                origin="lower",
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=max(vmax, 1e-12),
            )
            axis.set_xticks(np.arange(len(pivot.columns)))
            axis.set_xticklabels([f"{item:g}" for item in pivot.columns], rotation=35, ha="right")
            axis.set_yticks(np.arange(len(pivot.index)))
            axis.set_yticklabels([f"{item:g}" for item in pivot.index])
            axis.set_xlabel(x)
            axis.set_ylabel(y)
            axis.set_title(f"{facet} = {value:g}")
        for axis in panel.flat[len(facet_values) :]:
            axis.set_visible(False)
        if image is not None:
            fig.colorbar(image, ax=panel.ravel().tolist(), label="Frequency (MHz)", shrink=0.84)
        fig.suptitle(title, fontsize=15)
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path
    raise ValueError("Automatic sweep plotting supports up to three axes")


def plot_lab_summary(summary: pd.DataFrame, out_path: str | Path) -> Path:
    """Visual summary of digitized current, voltage plateau, and ripple."""

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    axes[0].plot(summary["frame_index"], summary["current_inferred_uA"], "o-", color=COLORS["ink"])
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Inferred current (uA)")
    axes[1].plot(summary["current_inferred_uA"], summary["v_plateau_mean_mV"], "o-", color=COLORS["blue"])
    axes[1].set_xlabel("Current (uA)")
    axes[1].set_ylabel("Plateau voltage (mV)")
    axes[2].plot(summary["current_inferred_uA"], summary["v_plateau_vpp_mV"], "o-", color=COLORS["purple"])
    axes[2].set_xlabel("Current (uA)")
    axes[2].set_ylabel("Plateau ripple (mV pp)")
    for axis in axes:
        axis.grid(True, color=COLORS["grid"])
    fig.suptitle("Digitized laboratory current sweep", fontsize=15)
    return _finish(fig, out_path)


def plot_lab_parameter_estimates(
    capacitance: pd.DataFrame,
    conductance: pd.DataFrame,
    resistance: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Show the separately identifiable electrical, cooling, and voltage-floor quantities."""

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    axes[0].scatter(capacitance["current_inferred_uA"], capacitance["C_slope_pF"], s=48, color=COLORS["purple"])
    median_c = float(capacitance["C_slope_pF"].median())
    axes[0].axhline(median_c, color=COLORS["ink"], linestyle="--", label=f"median {median_c:.1f} pF")
    axes[0].set_xlabel("Current (uA)")
    axes[0].set_ylabel("C from I/(dV/dt) (pF)")
    axes[0].set_title("Cold electrical transient")
    axes[0].legend()
    axes[1].plot(conductance["T0_K"], conductance["S_e_mW_per_K"], "o-", color=COLORS["orange"])
    axes[1].set_xlabel("Assumed ambient T0 (K)")
    axes[1].set_ylabel("S_e (mW/K)")
    axes[1].set_title("Ambient/cooling degeneracy")
    axes[2].plot(resistance["current_inferred_uA"], resistance["R_effective_ohm"], "o-", color=COLORS["blue"])
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Current (uA)")
    axes[2].set_ylabel("Plateau V/I (Ohm)")
    axes[2].set_title("Measured voltage floor")
    for axis in axes:
        axis.grid(True, color=COLORS["grid"])
    fig.suptitle("Parameter estimates separated from assumptions", fontsize=15)
    return _finish(fig, out_path)


def plot_voltage_floor_comparison(
    resistance: pd.DataFrame,
    specimen_Rm_ohm: float,
    yuanhang_Rm_ohm: float,
    out_path: str | Path,
) -> Path:
    """Compare measured plateaus with the ideal-current prediction V=I*Rm."""

    current = resistance["current_inferred_uA"].to_numpy(dtype=float)
    measured = resistance["v_plateau_mean_mV"].to_numpy(dtype=float)
    grid = np.linspace(0.0, max(1400.0, float(np.max(current)) * 1.03), 400)
    fig, ax = plt.subplots(figsize=(9.8, 6.3))
    ax.scatter(current, measured, s=32, color=COLORS["ink"], label="Digitized lab plateau", zorder=5)
    ax.plot(
        grid,
        grid * specimen_Rm_ohm / 1000.0,
        color=COLORS["red"],
        linewidth=2.1,
        label=f"Fitted Rm={specimen_Rm_ohm:.1f} Ohm",
    )
    ax.plot(
        grid,
        grid * yuanhang_Rm_ohm / 1000.0,
        color=COLORS["blue"],
        linewidth=2.1,
        label=f"Yuanhang Rm={yuanhang_Rm_ohm:.0f} Ohm",
    )
    ax.set_xlim(0.0, float(np.max(grid)))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Applied current (uA)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title("Metallic resistance sets the ideal-current voltage floor")
    ax.grid(True, color=COLORS["grid"])
    ax.legend(loc="upper left")
    ax.text(
        0.99,
        0.02,
        "Capacitance changes the approach time; these steady-state lines do not depend on C.",
        transform=ax.transAxes,
        ha="right",
        color="#4b5563",
        fontsize=9,
    )
    return _finish(fig, out_path)


def plot_resistance_fit(data: pd.DataFrame, prediction: np.ndarray, out_path: str | Path) -> Path:
    """Plot measured and fitted heating/cooling resistance branches."""

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    temperature_column = "temperature_K" if "temperature_K" in data.columns else "Temperature"
    resistance_column = "resistance_ohm" if "resistance_ohm" in data.columns else "Resistance"
    temperature = data[temperature_column].to_numpy(dtype=float)
    if np.nanmedian(temperature) < 200.0:
        temperature = temperature + 273.15
    resistance = data[resistance_column].to_numpy(dtype=float)
    if "branch" in data.columns:
        branch = data["branch"].astype(str).str.lower().to_numpy()
    else:
        direction = np.diff(temperature, prepend=temperature[0])
        branch = np.where(direction < 0.0, "cooling", "heating")
    for label, color in (("heating", COLORS["red"]), ("cooling", COLORS["blue"]), ("data", COLORS["ink"])):
        mask = branch == label
        if np.any(mask):
            ax.scatter(temperature[mask], resistance[mask], s=18, color=color, alpha=0.72, label=f"Measured {label}")
    order = np.argsort(temperature)
    ax.plot(
        temperature[order], np.asarray(prediction)[order], color=COLORS["green"], linewidth=1.5, label="Fitted model"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Resistance (Ohm)")
    ax.grid(True, which="both", color=COLORS["grid"])
    ax.legend()
    ax.set_title("Measured and fitted hysteretic resistance")
    return _finish(fig, out_path)
