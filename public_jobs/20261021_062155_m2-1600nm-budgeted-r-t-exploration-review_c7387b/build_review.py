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
STUDIES = {
    "Broad": RUNS / "20261021_061156_m2-1600nm-approximate-r-t-exploratory-3d-map-202_557171",
    "Lower heating": RUNS / "20261021_061258_m2-1600nm-approximate-r-t-focused-3d-map-2026102_ca4707",
    "Transition": RUNS / "20261021_061537_m2-1600nm-approximate-r-t-transition-3d-map-2026_a7691f",
}
onset_log = (RUNS / "m2_1600nm_onset_20261021.log").read_text()
STUDIES["Onset"] = Path(next(x.split("=", 1)[1] for x in onset_log.splitlines() if x.startswith("STUDY_DIRECTORY=")))
for source in STUDIES.values():
    assert json.loads((source / "run.json").read_text())["status"] == "completed", source

bundle = RunBundle.create(name="M2 1600nm budgeted R-T exploration review", model="current", kind="analysis",
    config={"parents": {k: str(v) for k, v in STUDIES.items()}, "diagnostic_only": True,
            "current_uA": 500, "ambient_K": 298, "static_gate_exception": "0.05 to 0.10, exploratory only"},
    output_root=RUNS, command="Saved-evidence assembly: build_review.py")
shutil.copy2(__file__, bundle.add_artifact("build_review.py", label="Exact saved-evidence assembly source"))
strict = RUNS / "20261021_061107_m2-1600nm-r-t-coarse-3d-exploration-20261021_3ad05c"
shutil.copytree(strict, bundle.root / "strict_static_check")
for path in sorted((bundle.root / "strict_static_check").rglob("*")):
    if path.is_file():
        bundle.register_file(path.relative_to(bundle.root), label=f"Preserved strict static failure: {path.name}")
all_tables, summaries, parts = [], [], []
plotly_js = True
fit_uri = None
for order, (name, source) in enumerate(STUDIES.items()):
    slug = name.lower().replace(" ", "_")
    destination = bundle.root / "studies" / slug
    shutil.copytree(source, destination)
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            bundle.register_file(path.relative_to(bundle.root), label=f"{name}: {path.name}")
    stages = json.loads((source / "stages.json").read_text())
    grid = pd.read_csv(source / stages["grid"] / "grid.csv")
    grid["study"] = name
    all_tables.append(grid)
    config = json.loads((source / "resolved_config.json").read_text())
    summary = dict(study=name, points=len(grid), sustained=int(grid.sustained.sum()),
        temperature_valid=int(grid.temperature_valid.sum()), retained=int(grid.included.sum()),
        sustained_temperature_excluded=int((grid.sustained & ~grid.temperature_valid).sum()),
        dt_ns=config["time"]["dt_ns"], duration_us=config["time"]["duration_us"])
    summaries.append(summary)
    if fit_uri is None:
        fit_path = source / stages["fit"] / "figures/resistance_fit.png"
        shutil.copy2(fit_path, bundle.add_artifact("resistance_fit.png", label="Measured and fitted R(T) with residuals"))
        fit_uri = "data:image/png;base64," + base64.b64encode(fit_path.read_bytes()).decode()
        fit = json.loads((source / stages["fit"] / "resistance_preset.json").read_text())
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
    visible = "" if name == "Onset" else ' hidden'
    parts.append(f'<section id="panel-{slug}"{visible}><h2>{html.escape(name)}</h2><p>{summary["points"]} grid entries; '
        f'{summary["sustained"]} sustained by the finite-window detector; <strong>{summary["retained"]} retained</strong>. '
        f'Base step {summary["dt_ns"]:g} ns; duration {summary["duration_us"]:g} µs. '
        f'<a href="studies/{slug}/parameter_maps.html">Original screened map and full provenance</a>.</p>' + "".join(panels) + '</section>')

table = pd.concat(all_tables, ignore_index=True)
table.to_csv(bundle.add_artifact("all_grid_entries.csv", label="All grid entries with original screen flags"), index=False)
shutil.copy2(PROJECT / "runs/m2_1600nm_diagnostics/fine_pilot.csv", bundle.add_artifact("fine_pilot.csv", label="36-point 32x-finer exclusion pilot"))
for recipe in sorted((PROJECT / "experiments/studies").glob("m2_*20261021.toml")):
    text = recipe.read_text().replace("../../data/experimental/m2_8.9_1stL_1600nm_A_RT.tsv",
                                     "../studies/broad/inputs/resistance_original.tsv")
    bundle.write_text(f"recipes/{recipe.name}", text, label="Portable experiment recipe", media_type="text/plain")
pilot_physics = json.loads(next((STUDIES["Lower heating"] / "stages").glob("*/physics_parameters.json")).read_text())
pilot_physics.update(dt_s=0.0015625e-9, t_end_s=0.6e-6)
bundle.write_json("fine_pilot_config.json", dict(physics=pilot_physics, current_uA=500,
    axes=dict(C_th_pJ_per_K=[0.1, 0.3, 1.0], S_e_mW_per_K=[0.03, 0.1, 0.3], C_pF=[0.03, 0.1, 0.3, 1.0]),
    window_ns=[300, 600], maximum_frequency_MHz=110), label="Exact 36-point finer pilot configuration")
for log in ["m2_1600nm_pytest_20261021.log", "m2_1600nm_validate_20261021.log"]:
    shutil.copy2(RUNS / log, bundle.add_artifact(f"checks/{log}", label="Software verification log"))

rmse = fit["fit_metrics"]["rmse_log10"]
metrics = dict(studies=summaries, grid_entries=len(table), retained=int(table.included.sum()),
    static_rmse_log10=rmse, static_default_gate_passed=False, exploratory_static_gate=0.10,
    input_sha256=hashlib.sha256((PROJECT / "data/experimental/m2_8.9_1stL_1600nm_A_RT.tsv").read_bytes()).hexdigest(),
    current_uA=500, ambient_K=298, measured_temperature_range_K=[289.98, 370.1795],
    fine_pilot_points=36, tests_passed=194, repository_validation="passed", dynamic_calibration_adopted=False)
bundle.write_json("metrics.json", metrics, label="Reviewed experiment counts and limitations")
summary_html = pd.DataFrame(summaries).to_html(index=False, border=0)
buttons = " ".join(f'<button onclick="showPanel(\'{n.lower().replace(" ", "_")}\')">{html.escape(n)}</button>' for n in STUDIES)
warning = (f"Exploratory model only. The R–T fit has log10 RMSE {rmse:.5f} (about {(10**rmse-1)*100:.1f}% typical multiplicative discrepancy), "
    "and failed the normal 0.05 gate. The recipe explicitly permits 0.10 for this diagnostic; no static or dynamic calibration is accepted. "
    "The measured temperature domain remains 289.98–370.1795 K. Crosses and colored diagnostic cells include excluded trajectories; "
    "their frequencies and amplitudes are computed outputs, not validated device predictions. "
    "Only original screened maps count verified retained points. Blank cells do not prove absence of oscillations.")
document = '<!doctype html><html><head><meta charset="utf-8"><title>M2 1600 nm — R–T exploration review</title>' + '''
<style>body{font:16px system-ui;margin:25px auto;max-width:1350px;color:#192a35;padding:0 20px}h1{font-size:30px}.warning{background:#fff1d6;border-left:5px solid #bc6b05;padding:18px;line-height:1.55}button{padding:10px 18px;margin:7px;border:1px solid #678;background:#eef4f8;cursor:pointer;border-radius:6px}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd}img{max-width:100%}section[hidden]{display:none}</style></head><body>''' + \
    '<h1>M2 1600 nm: budgeted R–T exploration</h1><p>21 October 2026 · ideal constant current 500 µA · ambient and initial temperature 298 K · axes Cth, Se, C.</p>' + \
    f'<p class="warning">{warning}</p><p><strong>{len(table)} sampled grid entries across four coarse grids; {metrics["retained"]} verified retained points.</strong> '
document += 'The 36-point finer-step pilot is separate. No equations were changed.</p>' + summary_html + \
    '<p>Choose a grid, rotate either 3D plot, or move a slice slider. Every colored cell is an actual sampled result; no interpolation is used. ' + \
    'Color scales stay fixed across slider positions within each grid.</p>' + buttons + "".join(parts) + \
    f'<details><summary>Measured R–T fit and residuals</summary><img src="{fit_uri}" alt="R-T fit and residuals"></details>' + \
    '<p><a href="report.md">Method and limitations</a> · <a href="all_grid_entries.csv">All numerical entries and screen flags</a></p>' + '''
<script>function showPanel(name){document.querySelectorAll('section[id^="panel-"]').forEach(p=>p.hidden=p.id!=='panel-'+name);document.querySelectorAll('#panel-'+name+' .js-plotly-plot').forEach(p=>Plotly.Plots.resize(p));}showPanel('onset');</script></body></html>'''
bundle.write_text("parameter_maps.html", document, label="Standalone diagnostic 3D and six slice maps", media_type="text/html")
report = f"""# M2 1600 nm: budgeted R–T exploration

Completed 21 October 2026. Same ideal-current physics, 500 µA, 298 K, initial V=0,
insulating branch and assumed gamma=0.956269682 as the earlier map. The supplied
file contains 374 acquisition-ordered measurements in K and ohm, 289.98–370.1795 K.
Its exact raw bytes are preserved in every study and their hash is in metrics.json.

## Result

{len(table)} sampled grid entries in four deliberately coarse grids;
{metrics['retained']} verified retained points. These are sampled entries, not a claim of
that many distinct physical settings. The map viewer shows computed sustained
oscillators with their exclusions visible; original screened viewers remain
unchanged. All sustained results in the first three grids exceeded the measured
temperature domain. This is not proof that no in-domain oscillator exists.
Coarse rejected points do not automatically receive the two-step recheck.

{pd.DataFrame(summaries).to_markdown(index=False)}

## Static model limitation

The six-parameter inherited law gives log10 RMSE {rmse:.8f}, R²
{fit['fit_metrics']['r2_log10']:.8f}, typical multiplicative discrepancy
{10**rmse:.5f} and worst multiplicative discrepancy
{fit['diagnostics']['maximum_multiplicative_error']:.5f}. Structured transition
residuals remain. The strict initial run stopped at its 0.05 gate; exploratory
recipes explicitly permit 0.10. This is a recorded exception, not an improved fit.
No equations, physics tolerances or convergence criteria were modified.
Gamma, C, Cth and Se are not identified from the measured major loop.

## Checks and interpretation

All displayed data are saved numerical outputs. Frequency uses late voltage peak
periods; Vpp uses complete-cycle extrema and the existing persistence detector.
Original screens retain the measured temperature domain, nonnegative sampled
proximity slope, operational 110 MHz ceiling and 5% feature changes at two smaller
steps. A separate 36-point pilot at 0.0015625 ns (32× finer than the first grid)
still found no eligible oscillator; this does not finely verify every excluded cell.
Finite-window stable/no-cycle cells and search edges are not physical boundaries.
The HTML includes all diagnostic maps and the fit image with no network dependencies.
Browser interaction was not verified because the available browser blocks local
file URLs. Numerical tables, generated figures, HTML structure and artifact paths
were checked independently. Software checks: 194 tests passed; repository validation
passed. No adopted specimen calibration or experimental waveform agreement.

## Reproduce

Run `neuristor study run --config` with each of the four portable recipes in
`recipes/`; their data path points to the preserved raw input in this bundle.
The source repository also contains recipes under `experiments/studies/` and
the identical raw input under `data/experimental/`. The strict initial failure
is preserved separately in `strict_static_check/`. The finer 36-point pilot's
complete physics, axes, timestep and feature window are in `fine_pilot_config.json`.
Each `studies/` directory is an immutable completed bundle with frozen source,
resolved settings, raw data, fits, grids, and manifests. This review's exact
assembly script is `build_review.py`; it renders saved data without changing physics.
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
fig.suptitle("M2 onset grid — all shown oscillations exceed the measured temperature range", fontsize=14)
fig.tight_layout()
fig.savefig(bundle.add_artifact("diagnostic_3d.png", label="Static diagnostic 3D overview"), dpi=160)
plt.close(fig)
for artifact in bundle._artifacts:
    path = bundle.root / artifact["path"]
    assert path.is_file(), path
bundle.complete(summary=metrics)
print(bundle.root, flush=True)
