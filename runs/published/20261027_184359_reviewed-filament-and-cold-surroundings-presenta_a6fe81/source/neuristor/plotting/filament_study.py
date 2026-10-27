"""Local, offline figures for the optional two-region presentation experiment."""

from __future__ import annotations

import html
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def render_filament_study(bundle):
    """Show complete errors and gaps, plus raw-current-driven voltage comparisons."""
    verification = pd.read_csv(bundle.root / "verification.csv")
    finest = verification[verification.dt_ns == verification.dt_ns.min()]
    names = ["presentation_frequency", "presentation_combined", "baseline_fitted", "filament_fitted"]
    labels = {
        "presentation_frequency": "Presentation frequency-only",
        "presentation_combined": "Presentation combined",
        "baseline_fitted": "Refitted baseline",
        "filament_fitted": "Refitted filament",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for name in names:
        group = finest[finest.candidate == name].sort_values("measured_current_A")
        current = group.measured_current_A * 1e6
        detected = group.predicted_frequency_MHz.notna() & (group.predicted_cycle_count >= 2)
        for ax, field in zip(axes, ["frequency_MHz", "robust_vpp_mV", "mean_mV"]):
            values = group["predicted_" + field]
            if field != "mean_mV":
                values = np.where(detected, values, np.nan)
            ax.plot(current, values, ".-", label=labels[name])
    measured = finest[finest.candidate == names[0]].sort_values("measured_current_A")
    for ax, field, ylabel in zip(
        axes, ["frequency_MHz", "robust_vpp_mV", "mean_mV"], ["Frequency (MHz)", "Vpp (mV)", "Mean voltage (mV)"]
    ):
        ax.plot(measured.measured_current_A * 1e6, measured["measured_" + field], "k.--", label="Measured")
        ax.set(xlabel="Measured current (µA)", ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.savefig(
        bundle.add_artifact("figures/feature_comparison.png", label="All-current finest-step model comparison"), dpi=170
    )
    plt.close(fig)
    profiles = pd.read_csv(bundle.root / "fraction_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for name, group in profiles.groupby("reference", sort=False):
        group = group.sort_values("filament_fraction")
        axes[0].semilogx(group.filament_fraction, 100 * group.amplitude_MAPE_all, ".-", label=name)
        axes[1].semilogx(group.filament_fraction, group.screened_targets, ".-", label=name)
    axes[0].set(xlabel="Filament conductance weight f", ylabel="Diagnostic Vpp MAPE (%)")
    axes[1].set(xlabel="Filament conductance weight f", ylabel="Passing cycle/frequency targets")
    axes[0].legend(fontsize=6)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(
        bundle.add_artifact(
            "figures/fraction_sensitivity.png", label="Conditional sensitivity, not confidence intervals"
        ),
        dpi=170,
    )
    plt.close(fig)
    # Use the exact comparison-grid values from verification. No phase shift or
    # extra filtering is introduced for the visual waveform comparison.
    traces = pd.read_csv(bundle.root / "traces.csv")
    traces = traces[traces.candidate.isin(["baseline_fitted", "filament_fitted"])]
    voltages = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Measured versus simulated device-output hypothesis", "Original measured input current"],
    )
    records = sorted(traces.source_label_mV.unique())
    titles = []
    for index, source in enumerate(records):
        group = traces[traces.source_label_mV == source]
        baseline = group[group.candidate == "baseline_fitted"]
        extended = group[group.candidate == "filament_fitted"]
        titles.append(f"{baseline.current_uA.iloc[0]:.2f} µA · source ID {source:g} mV")
        for frame, column, title, color in [
            (baseline, "measured_voltage_mV", "Measured", "#202c39"),
            (baseline, "simulated_voltage_mV", "Baseline", "#1469b0"),
            (extended, "simulated_voltage_mV", "Filament", "#a335ba"),
            (baseline, "input_current_uA", "Measured source current", "#27844b"),
        ]:
            voltages.add_trace(
                go.Scatter(
                    x=frame.time_ns.tolist(),
                    y=frame[column].tolist(),
                    name=title,
                    visible=index == 0,
                    line=dict(color=color),
                ),
                row=2 if column == "input_current_uA" else 1,
                col=1,
            )
    voltages.update_layout(
        title=titles[0],
        height=650,
        template="plotly_white",
        sliders=[
            dict(
                currentvalue=dict(prefix="Current record: "),
                steps=[
                    dict(
                        label=str(i + 1),
                        method="update",
                        args=[{"visible": [j // 4 == i for j in range(4 * len(records))]}, {"title": title}],
                    )
                    for i, title in enumerate(titles)
                ],
            )
        ],
    )
    voltages.update_xaxes(range=[0, 250], title_text="Time (ns)", row=2, col=1)
    voltages.update_yaxes(title_text="Voltage (mV)", row=1, col=1)
    voltages.update_yaxes(title_text="Current (µA)", row=2, col=1)
    waveform_html = pio.to_html(voltages, full_html=False, include_plotlyjs=True)
    summary = pd.read_csv(bundle.root / "summary.csv")
    summary = summary[summary.dt_ns == summary.dt_ns.min()]
    columns = [
        "candidate",
        "filament_fraction",
        "screened_targets",
        "records",
        "defined_frequencies",
        "frequency_MAPE_defined",
        "amplitude_MAPE_all",
        "mean_voltage_RMSE_mV",
        "temperature_valid",
        "eligible",
    ]
    candidates = json.loads((bundle.root / "candidates.json").read_text())
    bundle.write_text(
        "START_HERE.html",
        '<!doctype html><meta charset="utf-8"><title>Experimental filament study</title>'
        "<style>body{font:16px system-ui;max-width:1300px;margin:25px auto;padding:20px}img{max-width:100%}table{display:block;overflow:auto}td,th{padding:7px}</style>"
        "<h1>Experimental filament and cold surroundings</h1><p>One conductance weight shared across all measured currents. "
        "The static R(T) curve is fixed. Results remain diagnostic unless all acceptance and timestep checks pass. "
        "The voltage comparison retains the existing channel-to-device mapping hypothesis. Missing frequencies are gaps.</p>"
        '<p><a href="report.md">Methods and interpretation</a> · <a href="verification.csv">Every current and timestep</a> · '
        '<a href="convergence.csv">Timestep checks</a> · <a href="candidates.json">Exact parameters</a></p>'
        + "<p>Fitted filament weight: "
        + html.escape(str(candidates["filament_fitted"]["filament_fraction"]))
        + "</p>"
        + summary[columns].to_html(index=False, border=0, na_rep="undefined")
        + "<p>MAPE columns are fractions (multiply by 100 for percent). Frequency uses defined records only; "
        "diagnostic Vpp MAPE includes the original zero-feature convention for missing cycles.</p>"
        + '<img src="figures/feature_comparison.png"><img src="figures/fraction_sensitivity.png">'
        + waveform_html,
        label="Offline interactive measured-current comparison and numerical evidence",
        media_type="text/html",
    )
