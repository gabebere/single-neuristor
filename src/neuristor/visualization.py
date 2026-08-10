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
