"""Assemble this budgeted experiment's saved numerical evidence; no new physics."""
from pathlib import Path
import base64
import hashlib
import html
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from neuristor.runs import RunBundle

PROJECT = Path(__file__).resolve().parents[2]
RUNS = PROJECT / "runs"
AXES = ["C_th_pJ_per_K", "S_e_mW_per_K", "C_pF"]
LABELS = ["Cth (pJ/K)", "Se (mW/K)", "C (pF)"]
METRICS = [("frequency_MHz", "Frequency (MHz)"), ("extrema_vpp_mV", "Voltage amplitude Vpp (mV)")]
PRIOR = PROJECT / "public_jobs/20261021_062155_m2-1600nm-budgeted-r-t-exploration-review_c7387b"
STUDIES = {
    "Expanded": RUNS / "20261021_063603_m2-1600nm-approximate-r-t-expanded-20261021_6d29bf",
    "Dense onset": RUNS / "20261021_063603_m2-1600nm-approximate-r-t-dense-onset-20261021_57e853",
}
for source in STUDIES.values():
    assert json.loads((source / "run.json").read_text())["status"] == "completed", source
probe = pd.read_csv(RUNS / "m2_1600nm_diagnostics/near_boundary_checks.csv")
assert len(probe) == 24, "Wait for both smaller-step startup checks"
assert "196 passed" in (RUNS / "m2_1600nm_expansion_pytest_20261021.log").read_text()

bundle = RunBundle.create(name="M2 1600nm expanded R-T exploration review", model="current", kind="analysis",
    config={"parents": {k: str(v) for k, v in STUDIES.items()}, "diagnostic_only": True,
            "current_uA": 500, "ambient_K": 298, "static_gate_exception": "0.05 to 0.10, exploratory only"},
    output_root=RUNS, command="Saved-evidence assembly: build_review.py")
shutil.copy2(__file__, bundle.add_artifact("build_review.py", label="Exact saved-evidence assembly source"))
for filename in ["resistance_fit.png", "metrics.json"]:
    shutil.copy2(PRIOR / filename, bundle.add_artifact("prior_" + filename, label="Prior study static fit and metrics"))
prior_study = PRIOR / "studies/broad"
prior_stages = json.loads((prior_study / "stages.json").read_text())
shutil.copy2(prior_study / prior_stages["fit"] / "resistance_preset.json", bundle.add_artifact("resistance_preset.json", label="Unchanged inherited approximate static fit"))
shutil.copy2(PROJECT / "data/experimental/m2_8.9_1stL_1600nm_A_RT.tsv", bundle.add_artifact("inputs/resistance_original.tsv", label="Unmodified measured R-T loop"))
all_tables, summaries, parts = [], [], []
plotly_js = True
fit = json.loads((bundle.root / "resistance_preset.json").read_text())
fit_uri = "data:image/png;base64," + base64.b64encode((bundle.root / "prior_resistance_fit.png").read_bytes()).decode()
for order, (name, source) in enumerate(STUDIES.items()):
    slug = name.lower().replace(" ", "_")
    destination = bundle.root / "studies" / slug
    shutil.copytree(source, destination)
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            bundle.register_file(path.relative_to(bundle.root), label=f"{name}: {path.name}")
    grid = pd.read_csv(source / "grid.csv")
    grid["study"] = name
    all_tables.append(grid)
    config = json.loads((source / "resolved_config.json").read_text())
    summary = dict(study=name, points=len(grid), sustained=int(grid.sustained.sum()),
        temperature_valid=int(grid.temperature_valid.sum()), retained=int(grid.included.sum()),
        sustained_temperature_excluded=int((grid.sustained & ~grid.temperature_valid).sum()),
        dt_ns=config["time"]["dt_ns"], duration_us=config["time"]["duration_us"])
    summaries.append(summary)
    oscillators = grid[grid.sustained & grid.frequency_MHz.notna()].copy()
    figure = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                           subplot_titles=[x[1] for x in METRICS])
    for column, (metric, title) in enumerate(METRICS, 1):
        for included, symbol, label in [(False, "x", "Excluded computed oscillation"), (True, "circle", "Verified retained oscillator")]:
            d = oscillators[oscillators.included == included]
            if d.empty:
                continue
            custom = d[[*AXES, "frequency_MHz", "extrema_vpp_mV", "temperature_max_K", "temperature_valid", "converged", "frequency_valid"]].to_numpy()
            figure.add_trace(go.Scatter3d(x=np.log10(d[AXES[0]]), y=np.log10(d[AXES[1]]), z=np.log10(d[AXES[2]]),
                mode="markers", name=label, legendgroup=label, showlegend=column == 1,
                marker=dict(size=5, symbol=symbol, color=d[metric], colorscale="Viridis", showscale=True,
                            colorbar=dict(title=title, x=.45 if column == 1 else 1.02, len=.65)),
                customdata=custom,
                hovertemplate="Cth=%{customdata[0]:.5g} pJ/K<br>Se=%{customdata[1]:.5g} mW/K<br>C=%{customdata[2]:.5g} pF"
                "<br>f=%{customdata[3]:.4g} MHz<br>Vpp=%{customdata[4]:.5g} mV<br>Tmax=%{customdata[5]:.5g} K"
                "<br>Within measured T range=%{customdata[6]}<br>Converged=%{customdata[7]}<br>Frequency screen=%{customdata[8]}<extra>%{fullData.name}</extra>"), row=1, col=column)
        scene = {}
        for axis, label, coordinate in zip(AXES, LABELS, ["xaxis", "yaxis", "zaxis"]):
            values = np.sort(grid[axis].unique())
            scene[coordinate] = dict(title=label, tickvals=np.log10(values), ticktext=[f"{v:g}" for v in values])
        figure.update_layout(**{f"scene{column if column > 1 else ''}": scene})
    figure.update_layout(height=570, margin=dict(l=0, r=60, t=60, b=20), legend=dict(orientation="h", y=-.05),
                         title=f"{name}: computed oscillations, including exclusions — logarithmic parameter axes")
    panels = [pio.to_html(figure, full_html=False, include_plotlyjs=plotly_js)]
    plotly_js = False
    # Each slice uses fixed axes and global metric limits across its slider.
    for fixed_axis, fixed_label in zip(AXES, LABELS):
        other = [a for a in AXES if a != fixed_axis]
        x, y = [np.sort(grid[a].unique()) for a in other]
        fixed_values = np.sort(grid[fixed_axis].unique())
        figs = make_subplots(rows=1, cols=2, subplot_titles=[m[1] for m in METRICS])
        for i, value in enumerate(fixed_values):
            part = oscillators[np.isclose(oscillators[fixed_axis], value)]
            for column, (metric, label) in enumerate(METRICS, 1):
                z = part.pivot(index=other[1], columns=other[0], values=metric).reindex(index=y, columns=x).to_numpy()
                limits = oscillators[metric].agg(["min", "max"])
                figs.add_trace(go.Heatmap(x=np.log10(x), y=np.log10(y), z=z, visible=i == len(fixed_values)//2,
                    zmin=limits["min"] if len(oscillators) else 0, zmax=limits["max"] if len(oscillators) else 1,
                    colorscale="Viridis", colorbar=dict(title=label, x=.45 if column == 1 else 1.02, len=.6),
                    hovertemplate=f"{label}=%{{z:.5g}}<extra>Diagnostic: inspect grid.csv for screen flags</extra>"), row=1, col=column)
        for column in [1, 2]:
            figs.update_xaxes(title=LABELS[AXES.index(other[0])], tickvals=np.log10(x), ticktext=[f"{v:g}" for v in x], row=1, col=column)
            figs.update_yaxes(title=LABELS[AXES.index(other[1])], tickvals=np.log10(y), ticktext=[f"{v:g}" for v in y], row=1, col=column)
        steps = [dict(method="update", args=[{"visible": [j//2 == i for j in range(2*len(fixed_values))]}],
                      label=f"{value:g}") for i, value in enumerate(fixed_values)]
        figs.update_layout(title=f"Fix {fixed_label}: diagnostic slices include excluded oscillations",
            height=450, margin=dict(l=80, r=80, b=90, t=75),
            sliders=[dict(active=len(fixed_values)//2, currentvalue=dict(prefix=f"{fixed_label} = "), steps=steps)])
        panels.append(pio.to_html(figs, full_html=False, include_plotlyjs=False))
    visible = "" if name == "Dense onset" else ' hidden'
    parts.append(f'<section id="panel-{slug}"{visible}><h2>{html.escape(name)}</h2><p>{summary["points"]} grid entries; '
        f'{summary["sustained"]} sustained by the finite-window detector; <strong>{summary["retained"]} retained</strong>. '
        f'Base step {summary["dt_ns"]:g} ns; duration {summary["duration_us"]:g} µs. '
        f'<a href="studies/{slug}/parameter_maps.html">Original screened map and full provenance</a>.</p>' + "".join(panels) + '</section>')

table = pd.concat(all_tables, ignore_index=True)
table.to_csv(bundle.add_artifact("all_grid_entries.csv", label="All grid entries with original screen flags"), index=False)
for recipe in sorted((PROJECT / "experiments/current").glob("m2_1600nm_*20261021.toml")):
    text = recipe.read_text()
    import re
    text = re.sub(r'preset = "[^"]+"', 'preset = "../resistance_preset.json"', text)
    bundle.write_text(f"recipes/{recipe.name}", text, label="Portable expanded-search recipe", media_type="text/plain")
for filename in ["near_boundary_checks.csv", "near_boundary_config.json", "check_near_boundary.py"]:
    path = RUNS / "m2_1600nm_diagnostics" / filename
    if path.is_file():
        shutil.copy2(path, bundle.add_artifact(filename, label="Independent finer-step temperature exclusion check"))
for log in ["m2_1600nm_expansion_pytest_20261021.log", "m2_1600nm_expansion_validate_20261021.log"]:
    shutil.copy2(RUNS / log, bundle.add_artifact(f"checks/{log}", label="Software verification log"))

rmse = fit["fit_metrics"]["rmse_log10"]
metrics = dict(studies=summaries, grid_entries=len(table), retained=int(table.included.sum()),
    static_rmse_log10=rmse, static_default_gate_passed=False, exploratory_static_gate=0.10,
    input_sha256=hashlib.sha256((PROJECT / "data/experimental/m2_8.9_1stL_1600nm_A_RT.tsv").read_bytes()).hexdigest(),
    current_uA=500, ambient_K=298, measured_temperature_range_K=[289.98, 370.1795],
    prior_grid_entries=773, cumulative_grid_entries=len(table)+773, previous_fine_pilot_points=36,
    startup_check_settings=12, startup_check_integrations=len(probe), startup_check_exclusions=int(probe.excluded_in_startup.sum()),
    smallest_fine_startup_maximum_K=float(probe.maximum_temperature_K.min()),
    tests_passed=196, repository_validation="passed", dynamic_calibration_adopted=False)
bundle.write_json("metrics.json", metrics, label="Reviewed experiment counts and limitations")
summary_html = pd.DataFrame(summaries).to_html(index=False, border=0)
buttons = " ".join(f'<button onclick="showPanel(\'{n.lower().replace(" ", "_")}\')">{html.escape(n)}</button>' for n in STUDIES)
warning = (f"Exploratory model only. The R–T fit has log10 RMSE {rmse:.5f} (about {(10**rmse-1)*100:.1f}% typical multiplicative discrepancy), "
    "and failed the normal 0.05 gate. The first study explicitly allowed 0.10 for diagnostics; this expansion reuses that exact approximate fit. No calibration is accepted. "
    "The measured temperature domain remains 289.98–370.1795 K. Crosses and colored diagnostic cells include excluded trajectories; "
    "their frequencies and amplitudes are computed outputs, not validated device predictions. "
    "Only original screened maps count verified retained points. Blank cells do not prove absence of oscillations.")
document = '<!doctype html><html><head><meta charset="utf-8"><title>M2 1600 nm — expanded R–T exploration review</title>' + '''
<style>body{font:16px system-ui;margin:25px auto;max-width:1350px;color:#192a35;padding:0 20px}h1{font-size:30px}.warning{background:#fff1d6;border-left:5px solid #bc6b05;padding:18px;line-height:1.55}button{padding:10px 18px;margin:7px;border:1px solid #678;background:#eef4f8;cursor:pointer;border-radius:6px}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd}img{max-width:100%}section[hidden]{display:none}</style></head><body>''' + \
    '<h1>M2 1600 nm: expanded R–T exploration</h1><p>21 October 2026 · ideal constant current 500 µA · ambient and initial temperature 298 K · axes Cth, Se, C.</p>' + \
    f'<p class="warning">{warning}</p><p><strong>{len(table)} sampled grid entries across two expanded grids; {metrics["retained"]} verified retained points.</strong> '
document += '<br>The earlier study had 773 grid entries; both stages together contain 2,616 sampled entries, including overlaps. No equations were changed.</p>' + summary_html + \
    '<p>Choose a grid, rotate either 3D plot, or move a slice slider. Every colored cell is an actual sampled result; no interpolation is used. ' + \
    'Color scales stay fixed across slider positions within each grid.</p>' + buttons + "".join(parts) + \
    f'<details><summary>Measured R–T fit and residuals</summary><img src="{fit_uri}" alt="R-T fit and residuals"></details>' + \
    '<p><a href="report.md">Method and limitations</a> · <a href="all_grid_entries.csv">All numerical entries and screen flags</a> · ' + \
    f'<a href="../{PRIOR.name}/parameter_maps.html">Previous 773-entry review</a></p>' + '''
<script>function showPanel(name){document.querySelectorAll('section[id^="panel-"]').forEach(p=>p.hidden=p.id!=='panel-'+name);document.querySelectorAll('#panel-'+name+' .js-plotly-plot').forEach(p=>Plotly.Plots.resize(p));}showPanel('dense_onset');</script></body></html>'''
bundle.write_text("parameter_maps.html", document, label="Standalone diagnostic 3D and six slice maps", media_type="text/html")
report = f"""# M2 1600 nm: expanded R–T exploration

Additional search requested 21 October 2026 with a 15-minute budget. Two new
coarse grids add {len(table)} sampled entries to the earlier 773 (2,616 total,
including overlaps). Fixed ideal current 500 µA, ambient and initial temperature
298 K, zero initial voltage, insulating branch and gamma=0.956269682. No physics
or static-fit parameter was changed.

## Results

{pd.DataFrame(summaries).to_markdown(index=False)}

{metrics['retained']} new verified retained points. Diagnostic plots include
excluded computed oscillations, marked as such; they are not validated device
predictions. The full trajectory must remain inside the measured R–T interval
289.98–370.1795 K. This finite search does not prove absence of an in-domain
oscillator. Only coarse-eligible points receive the workflow's two-step check;
other rejected cells are not individually certified at the finest timestep.

An additional startup-only check replays 12 nearby sustained settings at
0.00078125 and 0.000390625 ns for the first 150 ns. Selection uses the lowest
coarse Tmax among the earlier onset grid and an interim dense-grid checkpoint;
the exact selection is saved. {int(probe.excluded_in_startup.sum())}/24 integrations
already exceed the measured range within that startup window; the lowest
observed maximum is {probe.maximum_temperature_K.min():.6f} K. This verifies
those early temperature exclusions without asserting steady-state feature or
waveform convergence. It is not a full fine replay of all 1,843 points.

## Search ranges and numerical settings

The 8×8×8 broad grid expands Cth to 0.001–50 pJ/K, Se to
0.0001–3 mW/K, and C to 0.0001–500 pF. Base dt=0.0125 ns, duration
2.4 µs, late window 1.2–2.4 µs, verification steps 0.00625 and 0.003125 ns.
The 11×11×11 onset grid samples Cth=0.5–4 pJ/K, Se=0.10–0.22 mW/K,
C=0.1–1.2 pF. Base dt=0.00625 ns, duration 1.2 µs, late window
0.6–1.2 µs, verification steps 0.003125 and 0.0015625 ns.
All retain the operational 110 MHz screen, existing persistence detector,
sampled proximity check and 5% feature-convergence threshold. Search bounds
and calibration exclusions are not physical oscillation boundaries.

## Inherited static-fit limitation

The same 374-row R–T fit has log10 RMSE {rmse:.8f} (typical multiplicative
factor {10**rmse:.5f}); it failed the normal 0.05 gate. The first study explicitly
allowed 0.10 for diagnostic maps. This expansion reuses its exact fit without
claiming improved static agreement. Gamma, electrical/thermal capacitance and
thermal conductance remain assumptions or axes, not measured estimates.
Input SHA-256: {metrics['input_sha256']}.

## Reproduction and checks

Run `neuristor analyze physical-maps --config` on the two portable TOML recipes
in `recipes/`; the static preset is preserved in this bundle. The complete new
map bundles, all flags, two-step verification tables (where applicable), frozen
source, physics parameters, raw measurement and inherited fit are included.
Animations are a separate optional stage, not an unbounded cost in this search.
The exact saved-evidence rendering script is preserved as `build_review.py`.
The standalone HTML embeds Plotly and its R–T image. Original screened maps
remain unchanged and are linked beside each diagnostic grid.
Browser interaction was not verified; numerical artifacts, links and the static
PNG were checked. Software validation logs are in `checks/`. The previous
773-entry study remains immutable at {PRIOR.name}. No new calibration is adopted.
"""
bundle.write_text("report.md", report, label="Experiment method, results and limitations")

# A static overview supports visual review without requiring a browser.
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), subplot_kw={"projection": "3d"})
last = all_tables[-1]
last = last[last.sustained & last.frequency_MHz.notna()]
for ax, (metric, label) in zip(axs, METRICS):
    if len(last):
        artist = ax.scatter(*[np.log10(last[a]) for a in AXES], c=last[metric], marker="x", cmap="viridis", s=35)
        fig.colorbar(artist, ax=ax, shrink=.55, pad=.1, label=label)
    else:
        ax.text2D(.2, .5, "No sustained oscillations in onset grid", transform=ax.transAxes)
    ax.set(xlabel="log10 Cth (pJ/K)", ylabel="log10 Se (mW/K)", zlabel="log10 C (pF)", title=label)
fig.suptitle("M2 expanded onset grid — diagnostic computed oscillations", fontsize=14)
fig.tight_layout()
fig.savefig(bundle.add_artifact("diagnostic_3d.png", label="Static diagnostic 3D overview"), dpi=160)
plt.close(fig)
for artifact in bundle._artifacts:
    path = bundle.root / artifact["path"]
    assert path.is_file(), path
bundle.complete(summary=metrics)
print(bundle.root, flush=True)
