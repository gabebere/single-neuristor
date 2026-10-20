# New sample → fitted R(T) → screened map → embedded GIFs

This is the reusable continuation of the earlier separate fitting, mapping and
animation jobs. One TOML file replaces manually passing presets between jobs.
It is not a new dynamic fitting method or a claim that the TIA mismatch is solved.

## Run it

```bash
neuristor study plan --config experiments/studies/sample_parameter_map.toml
neuristor study run --config experiments/studies/sample_parameter_map.toml
neuristor runs index
```

The plan is read-only: it checks input rows/units, major-loop structure, grid
budget, windows, frequency ceiling and integration settings. It reports upper
bounds on numerical steps, not a runtime prediction. The run prints the study
folder immediately; nested stage manifests track fit/grid/media progress.

Start with the 27-point example. Increasing each axis to 23 values means 12,167
coarse combinations, not 69. Only retained oscillators receive GIFs, but dense
explorations can take much longer. Do not assume that a model tested for one
current/sample will work with unchanged timing for another.

## What to edit

Copy `experiments/studies/sample_parameter_map.toml` to your own recipe. Relative
data paths resolve from the recipe's directory, not your terminal directory.
`--output-root DIRECTORY` overrides output placement. Dotted `--set path=value`
overrides must refer to existing keys and are saved in the resolved configuration.

| Field | Meaning |
|---|---|
| `study.resistance_data` | Numerical CSV or tab-separated R(T) measurements |
| `temperature_column`, `resistance_column` | Exact input column headings |
| `time_column` | Acquisition time column; `""` preserves file row order |
| `temperature_unit` | Explicit `K` or `C` (Celsius) |
| `resistance_unit` | Explicit `ohm` or `kohm` |
| `bootstrap_samples` | 0 for fit only; positive number for conditional fit intervals |
| `maximum_static_rmse_log10` | Refuse maps if the static fit exceeds this error |
| `temperature_domain` | `measured` by default; optional `fit-padded` uses the fitter's padded domain |
| `input.amplitude_uA` | Actual constant imposed current in µA, not nominal generator labels |
| `thermal.T0_K`, `initial.temperature_K` | Ambient and starting temperatures in K |
| `resistance.parameters.gamma` | Declared minor-loop assumption, not a major-loop estimate |
| `proximity_function` | `yuanhang` by default; `tanh` is a separate research variant |
| `maps.axes.*` | Independent sampled Cth / Se / C values, with explicit units |
| `maps.fixed_values.*` | Starting slider positions; must be sampled axis values |
| `maps.maximum_frequency_MHz` | Maximum accepted frequency, in **MHz** |
| `maps.window_ns` | Late observation window, inside `time.duration_us` |
| `verification_dt_ns` | Exactly two successively smaller steps after the coarse step |
| `convergence_fraction` | Maximum relative feature change at both map refinements |
| `animations.dt_ns` | At most half the finest map step |
| `maximum_refinements` | 0–3 further halvings for feature and visual checks |
| `workers`, `batch_size` | Local rendering/simulation budgets, not remote services |
| `frames` | GIF playback frames, not numerical integration samples |

For each axis, use `values = [...]`, or `low`, `high`, `count` for logarithmic
sampling plus optional `extra_values = [...]`. A one-value axis locks that
parameter. All logarithmic map coordinates must be positive; the separate solver
still supports the exact C=0 limit. Values of Cth/Se/C in the base recipe are
the reference values; grid candidates replace them with their own axis values.

## Input requirements

Supply one complete measured heating/cooling major loop with exactly one reversal.
It is read in acquisition order, never sorted by temperature. An explicitly
declared time column orders rows chronologically. Duplicate times, nonfinite
measurements, nonpositive resistance, conflicting units rows and insufficient
data are rejected rather than silently ignored. A conventional first units row
(e.g. K/Ohm) is supported. Only the declared measurement columns are used.

The current fitter is intended for VO₂ transition loops near room temperature,
not arbitrary materials or arbitrary hysteresis protocols. Static fitting bounds
remain those of the established fitter. Multiple minor loops need the separate
stateful analysis; they are not silently treated as a major loop by this command.
No points are extracted from graph images. No column units are guessed.

## Scientific procedure and limits

1. Normalize declared units to K and ohm, preserve the raw file and hash both.
2. Fit R0, Ea/kB, metallic resistance, Tc, transition width and beta in log10 R.
   Gamma is not determined by the major loop and remains a declared assumption.
3. Reject a poor static fit before generating a grid. Restrict resistance-domain
   validity to the measured temperature interval by default. Ambient and initial
   temperatures must lie inside the selected interval.
4. Simulate the Cartesian grid using the supported **constant ideal current**
   solver and the fitted static resistance. Electrical/thermal constants remain
   supplied assumptions or search coordinates, not newly measured estimates.
5. Measure frequency from adjacent late voltage maxima; measure amplitude from
   complete-cycle maxima/minima. Separately screen persistence, full-trajectory
   temperature, sampled proximity slope and frequency ceiling. Two smaller steps
   must preserve eligibility and frequency/Vpp within the configured tolerance.
6. Replay every retained point more finely for GIFs. Preserve all integrated R(T)
   endpoints; unresolved jumps break the curve instead of creating a smooth chord.
   Per-point replay warnings do not retroactively rewrite the completed map grid.
7. Export the map with heatmap axes and retained-point color ranges fixed across sliders and embed
   all GIFs, the fit image, recipe, resolved resistance and input hash in one HTML.

The 110 MHz ceiling is operational, not a universal physical cutoff. Persistence
is a finite-window test, not infinite-time stability. Coarse rejected points are
not all finely verified, so blank cells do not prove absence of oscillations.
Sampled proximity monotonicity is not proof of every hysteresis memory property.
`fit-padded` explicitly permits extrapolation outside measured R(T) temperatures;
the default `measured` does not. An empty retained grid is an honest valid result.

This workflow does not optimize frequency/amplitude against experimental traces
and does not identify a real TIA's source/readout impedance. It never fabricates
a matched voltage overlay for unmatched constant-current conditions. Use the
separate laboratory replay/fitting commands for that research question.

## Output folder

```text
runs/<study-id>/
├── parameter_maps.html       # Share THIS file alone: all GIFs and fit image embedded
├── START_HERE.html           # Local study navigation and evidence links
├── report.md, metrics.json   # Method, assumptions, exclusions and results
├── run.json                 # One parent manifest, status and Git provenance
├── resolved_config.json     # Recipe after overrides
├── plan.json, stages.json    # Workload and immutable child-stage locations
├── input_provenance.json    # Original/normalized input hashes and declared units
├── inputs/                  # Raw measurement copy and normalized K/ohm TSV
├── source/neuristor/         # Frozen implementation for reproduction
└── stages/
    ├── <fit-id>/             # Preset, residuals, parameter CSV, fit graph/intervals
    ├── <grid-id>/            # Every grid point, exclusions, two-step verification
    └── <media-id>/           # GIFs/, Figures/, catalog and refinement checks
```

One parent bundle contains all evidence; no external old preset is needed to
understand its resolved physics. Completed stages are never edited in place.
The registry indexes the parent, not each nested stage as a separate top-level
study. After publication the root remains navigable through relative links.

No ZIP is generated by the sample-study command. Base64 embeds GIF bytes at
roughly 4/3 their original size, so thousands of GIFs may produce an HTML too
large for email. Check `metrics.json: html_bytes` before attaching it. The folder
viewer in the media stage is lighter but requires its adjacent GIFs/ directory.
The standalone HTML is for sharing/review; the entire study folder is the
reproducibility archive containing raw/numerical data and source snapshots.

## History, preservation and checks

`neuristor runs index` rebuilds `artifacts/index.html`, `runs.csv` and `files.csv`.
It links canonical outputs rather than copying gigabytes or adding a database.
Old public runs remain untouched. The former Desktop archive was separately
consolidated on 16 October 2026: [canonical results and unique notes](../archive/desktop_consolidation_20261016/index.html).
No scientific results were changed by that filing. See
[the archive policy](../artifacts/README.md) and [the scientific chronology](ARCHIVE_INDEX.md).

```bash
pytest -q
neuristor validate
neuristor runs publish STUDY_ID
```

Publishing is a local archive operation, not a Git commit/push. Static parameters
are sample-dependent; dynamically plausible map points are not automatically
accepted laboratory parameter sets.
