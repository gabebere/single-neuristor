# Final project working archive

This directory is the single entry point for the report and the evidence used to
build it. It is a working collection: the manuscript will continue to change as the
remaining simulations are completed.

## Start here

- [`Simulations_for_VO2_AGC.pdf`](Simulations_for_VO2_AGC.pdf): latest compiled report.
- [`main.tex`](main.tex): editable LaTeX manuscript.
- [`data/`](data/): raw measurements and reviewed numerical results.
- [`supplementary/`](supplementary/): selected figures and animations for inspection
  and presentation.
- [`references/`](references/): the experimental manuscript, Yuanhang paper, and
  original Yuanhang implementation used by this project.

The report PDF and report-specific media are stored directly in this directory. The
data and reviewed-result directories deliberately point to the repository's canonical
files. This avoids silently creating two copies of the same measurement or analysis.
Relative links remain valid after cloning the complete Git repository.

## What is current

The report currently contains:

1. Yuanhang-based current-drive validation.
2. The sample-specific resistance-temperature fit.
3. Experimental timing and ambient temperature.
4. Environmental thermal conductance.
5. Electrical capacitance, using the conservative timing-resolution upper bound
   `C = 0.39 pF` while retaining `C = 0` as the constrained best fit.
6. Thermal time constant and thermal capacitance.
7. Blind model predictions against all 22 measured current traces.
8. Electrical/thermal capacitance dependence and the thermal-only limit.

## Data map

| Folder | Contents | Status |
|---|---|---|
| `data/resistance/raw_measurement.tsv` | Professor-supplied cooling/heating R(T) sweep | Raw source |
| `data/resistance/reviewed_fit/` | Fit, uncertainty samples, preset, report, and figure | Current reviewed result |
| `data/experiment_runs/raw_oscilloscope_exports/` | Original CSV/XLSX TIA sweep files and checksums | Raw source |
| `data/experiment_runs/normalized_current_sweep/` | Normalized traces, summary, and operating-window figure | Current reviewed result |
| `data/parameter_estimates/environmental_conductance/` | Conductance estimate and bootstrap evidence | Current reviewed result |
| `data/parameter_estimates/thermal_capacitance/` | Thermal fit, trajectories, uncertainty, and figure | Current reviewed result |
| `data/reviewed_runs/20260829_100718_specimen-model-prediction-versus-measured-curren_eefab7/` | Blind prediction, common-window metrics, convergence, and C--Cth map | Current reviewed result |
| `data/reviewed_runs/` | Complete tracked run archive | Mixed current and historical evidence |

Every reviewed run is immutable. If an analysis changes, a new bundle is published
instead of editing the existing one.

## Supplementary media

`supplementary/figures/` collects the current report figures in one place for quick
inspection. `supplementary/animations/` contains the synchronized current/voltage and
R(T) animation made for this report, plus selected earlier simulation animations.
The older presentation animations are labeled as historical and should not be treated
as sample-specific parameter estimates.

The complete July presentation package is available through
`supplementary/historical_presentation_package/`; it is retained for context rather
than used as current quantitative evidence.

## Rebuilding the report

From the repository root, compile `docs/final_project/main.tex`, write the build output
as `output/pdf/Simulations_for_VO2_AGC.pdf`, and refresh the identically named PDF in
this directory. The checked-in copy is the current readable snapshot; `main.tex`
remains the editable source.

For the exact commands that reproduce each numerical analysis, see the manuscript
appendix and [`../ARCHIVE_INDEX.md`](../ARCHIVE_INDEX.md).
