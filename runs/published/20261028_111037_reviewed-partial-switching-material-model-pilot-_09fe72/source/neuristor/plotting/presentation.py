"""Presentation figures and diagnostic viewers rendered from fresh numerical tables."""

from __future__ import annotations

import html
import os
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from neuristor.analysis.alternating_amplitudes import cycle_table
from neuristor.plotting.map_animations import resolved_rt_path

AXES = ("C_th_pJ_per_K", "S_e_mW_per_K", "C_pF")
METRICS = (("frequency_MHz", "Frequency (MHz)"), ("extrema_vpp_mV", "Vpp (mV)"))


def _save(fig, bundle, name):
    path = bundle.add_artifact(name, label="Recreated presentation figure")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_parameter_slice(table, bundle, name, *, diagnostic=False, fixed=1.5):
    """Show one sampled C/Se slice, leaving unmeasured and rejected cells blank."""
    values = np.sort(table.C_th_pJ_per_K.unique())
    chosen = values[np.argmin(abs(np.log(values / fixed)))]
    part = table[np.isclose(table.C_th_pJ_per_K, chosen)].copy()
    valid = part.sustained if diagnostic else part.included
    xs, ys = np.sort(part.C_pF.unique()), np.sort(part.S_e_mW_per_K.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (metric, label) in zip(axes, METRICS):
        z = (
            part.assign(value=np.where(valid, part[metric], np.nan))
            .pivot(index="S_e_mW_per_K", columns="C_pF", values="value")
            .reindex(index=ys, columns=xs)
        )
        # Log-space cell coordinates display actual sampled values only.
        image = ax.pcolormesh(xs, ys, z.to_numpy(), shading="nearest", cmap="viridis")
        ax.set(xscale="log", yscale="log", xlabel="C (pF)", ylabel="Se (mW/K)", title=label)
        fig.colorbar(image, ax=ax)
    suffix = "computed diagnostic values; exclusions remain in CSV" if diagnostic else "retained values only"
    fig.suptitle(f"Cth = {chosen:g} pJ/K · {suffix}", fontsize=10)
    fig.tight_layout()
    return _save(fig, bundle, name)


def render_mica_review(tables, links, bundle):
    """Render retained/excluded sampled oscillations without relabeling exclusions."""
    parts = []
    combined = []
    for number, (name, grid) in enumerate(tables.items()):
        combined.append(grid.assign(study=name))
        active = grid[grid.sustained & grid.frequency_MHz.notna()]
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=[label for _, label in METRICS],
        )
        for col, (metric, label) in enumerate(METRICS, 1):
            for accepted, symbol, title in (
                (False, "x", "Excluded computed oscillation"),
                (True, "circle", "Verified retained oscillator"),
            ):
                d = active[active.included == accepted]
                if d.empty:
                    continue
                fig.add_trace(
                    go.Scatter3d(
                        x=np.log10(d[AXES[0]]),
                        y=np.log10(d[AXES[1]]),
                        z=np.log10(d[AXES[2]]),
                        mode="markers",
                        name=title,
                        customdata=d[[*AXES, metric, "temperature_max_K"]].to_numpy().tolist(),
                        marker=dict(
                            size=4,
                            symbol=symbol,
                            color=d[metric],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title=label, x=0.45 if col == 1 else 1.02),
                        ),
                        hovertemplate="Cth=%{customdata[0]} pJ/K<br>Se=%{customdata[1]} mW/K<br>C=%{customdata[2]} pF"
                        "<br>Value=%{customdata[3]}<br>Tmax=%{customdata[4]} K<extra>%{fullData.name}</extra>",
                    ),
                    row=1,
                    col=col,
                )
        fig.update_layout(
            title=name + " — diagnostic computed outputs",
            height=520,
            scene=dict(xaxis_title="log10 Cth", yaxis_title="log10 Se", zaxis_title="log10 C"),
            scene2=dict(xaxis_title="log10 Cth", yaxis_title="log10 Se", zaxis_title="log10 C"),
        )
        panels = [pio.to_html(fig, full_html=False, include_plotlyjs=True if number == 0 else False)]
        for fixed in AXES:
            other = [a for a in AXES if a != fixed]
            x, y = [np.sort(grid[a].unique()) for a in other]
            values = np.sort(grid[fixed].unique())
            slices = make_subplots(rows=1, cols=2, subplot_titles=[label for _, label in METRICS])
            for i, value in enumerate(values):
                part = active[np.isclose(active[fixed], value)]
                for col, (metric, label) in enumerate(METRICS, 1):
                    z = part.pivot(index=other[1], columns=other[0], values=metric).reindex(index=y, columns=x)
                    slices.add_trace(
                        go.Heatmap(
                            x=np.log10(x).tolist(),
                            y=np.log10(y).tolist(),
                            z=z.to_numpy().tolist(),
                            visible=i == len(values) // 2,
                            colorscale="Viridis",
                            zmin=float(active[metric].min()) if len(active) else 0.0,
                            zmax=float(active[metric].max()) if len(active) else 1.0,
                            colorbar=dict(title=label, x=0.45 if col == 1 else 1.02),
                        ),
                        row=1,
                        col=col,
                    )
            slices.update_layout(
                title=f"{name}: fix {fixed} (all computed oscillations; inspect exclusion flags)",
                height=430,
                sliders=[
                    dict(
                        active=len(values) // 2,
                        currentvalue=dict(prefix=f"{fixed} = "),
                        steps=[
                            dict(
                                label=f"{v:g}",
                                method="update",
                                args=[{"visible": [j // 2 == i for j in range(2 * len(values))]}],
                            )
                            for i, v in enumerate(values)
                        ],
                    )
                ],
            )
            panels.append(pio.to_html(slices, full_html=False, include_plotlyjs=False))
        href = quote(os.path.relpath(links[name], bundle.root), safe="/._-")
        parts.append(
            f"<h2>{html.escape(name)}</h2><p>{len(grid)} sampled entries; {int(grid.included.sum())} retained. "
            f'<a href="{href}">Screened map and provenance</a></p>' + "".join(panels)
        )
    table = pd.concat(combined, ignore_index=True)
    table.to_csv(
        bundle.add_artifact("all_grid_entries.csv", label="All fresh Mica grid entries and original flags"), index=False
    )
    bundle.write_text(
        "parameter_maps.html",
        '<!doctype html><meta charset="utf-8"><title>Mica reproduction</title>'
        "<style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:20px}</style>"
        "<h1>Mica parameter exploration — freshly computed</h1><p>The approximate static fit failed the usual 0.05 gate; "
        "the declared 0.10 exception is diagnostic only. Crosses and colored slices include excluded trajectories. "
        "Computed frequencies/amplitudes are not accepted specimen predictions or physical boundaries.</p>"
        + "".join(parts),
        label="Interactive Mica grids and fixed-scale slices",
        media_type="text/html",
    )
    return table


def plot_reproduction_assets(bundle, stages):
    """Generate every numerical image in the current slides from the new stages."""
    grid = pd.read_csv(stages["map_grid"] / "grid.csv")
    plot_parameter_slice(grid, bundle, "presentation/assets/parameter_exploration_example.png", fixed=0.047873236)
    table = pd.read_csv(stages["mica_review"] / "all_grid_entries.csv")
    latest = table[table.study == "Dense onset"]
    plot_parameter_slice(latest, bundle, "presentation/assets/mica_parameter_slice.png", diagnostic=True)
    fit = pd.read_csv(stages["mica_fit"] / "measured.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(fit.Temperature, fit.Resistance, ".", label="Measured")
    ax.semilogy(fit.Temperature, fit.model_resistance_ohm, "-", label="Fresh major-loop fit")
    ax.set(xlabel="Temperature (K)", ylabel="Resistance (ohm)", title="Mica: approximate R(T) fit")
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, bundle, "presentation/assets/mica_rt_curve.png")
    scaling = pd.read_csv(stages["scaling"] / "case_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for anchor, part in scaling[(scaling.family == "scale") & (scaling.protocol == "scaled_dt")].groupby("anchor"):
        part = part.sort_values("scale")
        reference = part[np.isclose(part.scale, 1.0)].iloc[0]
        valid = (part.classification == "oscillatory") & part.timestep_verified
        for ax, field in zip(axes, ("frequency_MHz", "extrema_vpp_mV")):
            ax.plot(part.scale, np.where(valid, part[field] / reference[field], np.nan), "o-", label=anchor)
    for ax, label in zip(axes, ("Frequency / reference", "Vpp / reference")):
        ax.set(xlabel="Common C and Cth multiplier", ylabel=label)
        ax.grid(alpha=0.2)
        ax.legend()
    _save(fig, bundle, "presentation/assets/controlled_capacitance_scaling.png")
    natives = sorted(stages["alternating"].glob("dt_0.025ns/current_400.npz"))
    if not natives:
        natives = sorted(stages["alternating"].glob("dt_*ns/current_*.npz"))[:1]
    with np.load(natives[0]) as raw:
        t, v, temperature, resistance = [
            raw[k] for k in ("time_ns", "simulated_voltage_mV", "temperature_K", "resistance_ohm")
        ]
        cycles = cycle_table(t, v, [50.0, 280.0])
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 5))
        axes[0].plot(t, v, color="#7145d6")
        axes[0].set(xlim=(60, 235), xlabel="Time (ns)", ylabel="Voltage (mV)")
        if len(cycles) >= 2:
            for row, label in zip(cycles.iloc[2:4].itertuples(), ("Reset 1", "Reset 2")):
                keep = (t >= row.preceding_peak_time_ns) & (t <= row.peak_time_ns)
                _, tt, rr, _ = resolved_rt_path(t[keep], temperature[keep], resistance[keep])
                axes[1].semilogy(tt, rr, label=label)
        axes[1].set(xlabel="Simulated temperature (K)", ylabel="Resistance (ohm)")
        for ax in axes:
            ax.grid(alpha=0.2)
        if len(cycles) >= 4:
            axes[1].legend()
        fig.tight_layout()
        _save(fig, bundle, "presentation/assets/alternating_resets_for_slide.png")


def update_slide_tables(sources, stages):
    """Replace numerical slide tables/captions with freshly computed results.

    The original reviewed PDF stays immutable. These editable sources accompany
    the new figures and HTML; compiling them produces a new result presentation.
    Missing cycles remain dashes and diagnostic candidates remain labeled.
    """
    import json
    import re

    def value(number, decimals=1):
        return "--" if pd.isna(number) else f"{float(number):.{decimals}f}"

    def table(spec, header, rows):
        return (
            r"\begin{tabular}{"
            + spec
            + "}\n"
            + r"\toprule"
            + "\n"
            + header
            + r"\\"
            + "\n"
            + r"\midrule"
            + "\n"
            + "\n".join(" & ".join(row) + r" \\" for row in rows)
            + "\n"
            + r"\bottomrule"
            + "\n"
            + r"\end{tabular}"
        )

    comparison = stages["comparison"]
    summary = pd.read_csv(comparison / "summary.csv")
    summary = summary[summary.dt_ns == summary.dt_ns.min()].set_index("mode")
    parameters = pd.read_csv(comparison / "parameters.csv").set_index("mode")
    rows = []
    for mode, label in [("frequency_only", "Frequency only"), ("frequency_amplitude", "Frequency + amplitude")]:
        row = summary.loc[mode]
        n = int(row.target_count)
        rows.append(
            [
                label,
                f"{int(row.screened_target_count)}/{n}",
                f"{value(100 * row.frequency_MAPE_defined)}" + r"\%" + f" ({int(row.defined_frequency_count)}/{n})",
                value(100 * row.amplitude_MAPE) + r"\%",
            ]
        )
    blocks = [
        table(
            "@{}lrrr@{}",
            r"\textbf{Search} & \textbf{Passing targets} & \textbf{Frequency error} & \textbf{Vpp error}",
            rows,
        )
    ]
    parameters_rows = []
    for label, column, factor, decimals in [
        (r"\(\gamma\)", "gamma", 1, 4),
        (r"\(S_e\) (\(\mu\)W/K)", "S_e_mW_per_K", 1000, 3),
        (r"\(C\) (pF)", "C_pF", 1, 3),
        (r"\(C_{th}\) (pJ/K)", "C_th_pJ_per_K", 1, 5),
    ]:
        parameters_rows.append(
            [
                label,
                *[
                    value(parameters.loc[mode, column] * factor, decimals)
                    for mode in ("frequency_only", "frequency_amplitude")
                ],
            ]
        )
    blocks.append(
        table(
            "@{}lrr@{}",
            r"\textbf{Parameter} & \textbf{Frequency only} & \textbf{Frequency + amplitude}",
            parameters_rows,
        )
    )
    representatives = pd.read_csv(comparison / "representative_metrics.csv")
    for mode in ("frequency_only", "frequency_amplitude"):
        chosen = representatives[representatives["mode"] == mode].sort_values("measured_current_A")
        data = []
        for row in chosen.itertuples():
            prediction = row.predicted_frequency_MHz
            data.append(
                [
                    value(row.measured_current_A * 1e6),
                    value(row.measured_frequency_MHz),
                    value(prediction),
                    value(row.measured_extrema_vpp_mV),
                    value(row.predicted_extrema_vpp_mV) if pd.notna(prediction) else "--",
                    "Pass" if row.frequency_valid else "Fail",
                ]
            )
        blocks.append(
            table(
                "@{}rrrrrl@{}",
                r"\textbf{Input (\(\mu\)A)} & \textbf{Measured f} & \textbf{Simulated f} & \textbf{Measured Vpp} & \textbf{Simulated Vpp} & \textbf{Check}",
                data,
            )
        )
    text = sources["loss_comparison_results.tex"]
    matches = list(re.finditer(r"\\begin\{tabular\}.*?\\end\{tabular\}", text, flags=re.S))
    if len(matches) != len(blocks):
        raise ValueError("Slide comparison template changed; cannot safely replace numerical tables")
    for match, block in reversed(list(zip(matches, blocks))):
        text = text[: match.start()] + block + text[match.end() :]
    text = text.replace(
        "Neither is an all-target fit; the combined candidate drops from 9 to 8 passing targets at the finest step.",
        "Finite-budget search candidates remain diagnostic; inspect all-current and timestep checks in the recreated report.",
    )
    text = text.replace(
        "250 and 250 unique evaluations",
        f"{int(summary.loc['frequency_only', 'unique_candidates'])} and {int(summary.loc['frequency_amplitude', 'unique_candidates'])} unique evaluations",
    )
    text = text.replace("New runs: 543408 and 5ad999.", "Fresh searches: see stages.json.")
    text = text.replace("Comparison bundle 26574a.", "Fresh comparison: see stages.json.")
    sources["loss_comparison_results.tex"] = text
    fit = json.loads((stages["mica_fit"] / "resistance_preset.json").read_text())
    p = fit["resist_params"]
    rows = [
        [label, value(p[key] * factor, decimals)]
        for label, key, factor, decimals in [
            (r"\(R_0\) (\(\Omega\))", "R0", 1, 3),
            (r"\(E_a/k_B\) (K)", "Ea_over_k", 1, 2),
            (r"\(R_m\) (\(\Omega\))", "Rm0", float(p["Rm_factor"]), 2),
            (r"\(T_c\) (K)", "Tc_K", 1, 3),
            (r"\(w\) (K)", "w", float(p["width_factor"]), 3),
            (r"\(\beta\) (K\(^{-1}\))", "beta", 1, 5),
        ]
    ]
    mica = sources["mica_specimen.tex"]
    mica = re.sub(
        r"\\begin\{tabular\}.*?\\end\{tabular\}",
        lambda m: table("@{}lr@{}", r"\textbf{Fitted parameter} & \textbf{Mica}", rows),
        mica,
        count=1,
        flags=re.S,
    )
    mica = mica.replace("0.08065", value(fit["fit_metrics"]["rmse_log10"], 5))
    all_grid = pd.read_csv(stages["mica_review"] / "all_grid_entries.csv")
    additions = int(all_grid.study.isin(["Expanded", "Dense onset"]).sum())
    mica = mica.replace("1,843", f"{additions:,}").replace("2,616", f"{len(all_grid):,}")
    mica = mica.replace(
        "No point passed every check. All detected sustained oscillations still exceeded the measured R(T) temperature range.",
        f"{int(all_grid.included.sum())} points passed every check in this reproduction. Crosses remain excluded diagnostic results; inspect the saved screen flags.",
    )
    sources["mica_specimen.tex"] = mica
    grid = pd.read_csv(stages["map_grid"] / "grid.csv")
    media = json.loads((stages["map_gifs"] / "metrics.json").read_text())
    text = (
        sources["parameter_exploration.tex"]
        .replace("12,167", f"{len(grid):,}")
        .replace("1,258", f"{int(media['gifs']):,}")
    )
    text = text.replace(
        "12/12 scale checks pass 2\\%; plotted points use co-scaled timesteps. Extracted Vpp varies by less than 0.9\\% because the feature measurement uses a fixed-time smoothing rule.",
        "fresh scale checks and their unresolved cases are recorded in the recreated scaling tables; plotted points use co-scaled timesteps.",
    )
    sources["parameter_exploration.tex"] = text
    specimen = json.loads((stages["specimen_fit"] / "resistance_preset.json").read_text())
    p = specimen["resist_params"]
    text = sources["main.tex"]
    for old, key, factor, decimals in [
        ("0.7989", "R0", 1, 4),
        ("2536.97", "Ea_over_k", 1, 2),
        ("18.215", "Rm0", float(p["Rm_factor"]), 3),
        ("333.494", "Tc_K", 1, 3),
        ("6.882", "w", float(p["width_factor"]), 3),
        ("0.2992", "beta", 1, 4),
    ]:
        text = text.replace(old, value(p[key] * factor, decimals))
    for old, key in [("0.99873", "r2_log10"), ("0.03663", "rmse_log10")]:
        if key in specimen["fit_metrics"]:
            text = text.replace(old, value(specimen["fit_metrics"][key], 5))
    sources["main.tex"] = text
    text = sources["loss_function_story.tex"]
    text = text.replace(
        "Both searches completed 250 unique candidates under a 256-evaluation cap",
        f"The fresh searches completed {int(summary.loc['frequency_only', 'unique_candidates'])} and "
        f"{int(summary.loc['frequency_amplitude', 'unique_candidates'])} unique candidates",
    )
    sources["loss_function_story.tex"] = text
    # The frozen vector is unchanged, but measured-input integration may differ
    # across environments. Do not carry historical reset amplitudes into a new run.
    replay = pd.read_csv(stages["alternating"] / "cycle_summary.csv")
    if "input_mode" in replay:
        replay = replay[replay.input_mode == "measured_current"]
    central = replay[np.isclose(replay.source_label_mV, 400.0)].sort_values("dt_ns", ascending=False).iloc[0]
    text = sources["alternating_amplitudes.tex"]
    text = text.replace("about 303", f"about {central.current_uA:.0f}")
    text = text.replace(
        "Successive voltage swings: about 240 and 350 mV Vpp.",
        "Fresh individual-cycle amplitudes and checks are saved in the replay tables.",
    )
    text = text.replace(
        "Hysteresis memory could alternate partial (about 338 K) and deep (about 348 K) resets.",
        "Hysteresis memory could produce different reset depths; this remains a hypothesis.",
    )
    text = text.replace(
        "short-window alternation survives refinement.", "inspect the fresh cycle checks at all three timesteps."
    )
    text = text.replace("11.91", value(central.tau_th_ns, 2)).replace(
        r"0.13\text{--}12", value(central.tau_e_min_ns, 2) + r"\text{--}" + value(central.tau_e_max_ns, 1)
    )
    text = re.sub(r"\\note\{.*?\}\s*", "", text, flags=re.S)
    sources["alternating_amplitudes.tex"] = text
    return sources
