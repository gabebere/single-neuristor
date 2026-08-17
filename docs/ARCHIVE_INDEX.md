# Research archive index

This index distinguishes current quantitative evidence from historical simulation and
presentation artifacts. That distinction matters because Git history preserves earlier
experiments, while only the current solver and audit should be used for new numerical
claims.

## Active final-project manuscript

- `docs/final_project/main.tex`: working LaTeX manuscript for the VO2 automatic-gain-
  control simulation project. The Yuanhang validation and sample-specific R(T) fit are
  complete; later thermal/electrical parameter sections remain explicitly marked as planned.
- The Case I figure and metrics point to the immutable public run bundle
  `20260817_100102_current-step-with-a-nonzero-metallic-voltage-val_6765e0`, whose voltage
  panel includes the instantaneous metallic fixed point `I(t) R_m`, rather than duplicating
  scientific output inside the manuscript directory.
- `docs/final_project/figures/` contains the manuscript's derived R(T) trajectory and
  synchronized current/voltage--R(T) animation; both can be regenerated from that bundle.
- `20260816_125905_sample-r-t-major-loop-hysteresis-fit_0849a9` is the reviewed
  sample-specific major-loop fit, including the normalized data, fitted preset, residual
  figure, parameter table, and 1000 block-bootstrap parameter samples.

## Current model authority

- `src/neuristor/model.py`: Yuanhang hysteretic resistance and voltage-driven model.
- `src/neuristor/current_drive_sim.py`: supported ideal-current single-device model.
- `docs/HYSTERESIS_IMPLEMENTATION_AUDIT.md`: branch-memory implementation, numerical
  ordering, provenance, and fidelity checks.
- `docs/CURRENT_DRIVE_CALIBRATION.md`: voltage-floor diagnosis, parameter
  identifiability, capacitance study, and recommended laboratory calibration sequence.
- `data/experimental/tia_current_sweep/`: untouched professor-supplied numerical
  sources for paper Figures 6 and 7, with units, provenance, and ZIP checksum.
- `docs/manuscript/theory_behind_simulations.pdf`: compiled model theory manuscript.

## Current corroborating evidence

- `docs/figures/current_drive/capacitance_study/frequency_heatmaps_all_currents.png`:
  shared-scale `C_th`-versus-`C` frequency maps for six currents.
- `docs/figures/current_drive/capacitance_study/capacitance_trace_comparison_600uA.png`:
  direct demonstration that `C` changes timing while the metallic bound is `I*Rm`.
- The numerical laboratory workflow archives measured current/voltage traces, the
  Figure 7 operating window, and a baseline-corrected environmental-conductance
  estimate in immutable `public_jobs/` bundles.
- `20260817_134254_measured-laboratory-current-sweep_a45254`: all 22 normalized raw
  traces, their summary table, and the measured 41.7--62.5 MHz operating-window figure.
- `20260817_153807_environmental-thermal-conductance-estimate_761640`: canonical
  clean-provenance selected settled
  pre-onset waveform, fitted R(T) temperature inversion, conditional conductance
  interval, numerical tables, and evidence figure.
- `20260817_152216_environmental-thermal-conductance-estimate_c4be6c`: numerically
  identical pre-commit run retained for immutable provenance and superseded by `761640`.
- `20260817_134254_lab-current-trace-parameter-estimates_4223e0`: superseded historical
  bundle retained for provenance; its 19.8 pF cold-edge result is not valid.
- `docs/figures/current_drive/nonzero_valley_examples/current_input_voltage_output.png`:
  direct imposed-current/voltage-output examples, including an unchanged specimen
  control and an explicit nonzero-valley effective-resistance candidate.
- The adjacent CSV, JSON, and README files preserve the plotted numerical values,
  simulation settings, assumptions, and validity warnings.

## Historical artifacts

- `public_jobs/`: tracked reviewed evidence. The registry reads both historical July
  2026 `job.json` records and current portable `run.json` bundles.
- `Simulations_on_VO2/`: July 2026 Beamer presentation source, compiled deck, figures,
  and animations.
- `docs/presentation/project_year_presentation_outline.md`: longer historical talk
  outline, explicitly marked with its current-status note.
- Git history is the recovery path for removed experimental current-source variants and
  obsolete presets. They are not active physical models.

## Reproduction and verification

From the repository root:

```bash
neuristor simulate current \
  --config experiments/current/nonzero_voltage_valley.toml
neuristor sweep run \
  --config experiments/sweeps/current_capacitance_map.toml
neuristor analyze conductance \
  --data data/experimental/tia_current_sweep \
  --resistance-preset presets/resistance_100425_chip1_gap3.json \
  --resistance-bootstrap public_jobs/20260816_125905_sample-r-t-major-loop-hysteresis-fit_0849a9/parameter_bootstrap.csv \
  --ambient-K 314.4 --ambient-interval-K 314.25,314.55
pytest -q
neuristor validate
```

The laboratory workflows read the checked-in numerical exports directly. The former
screenshot digitizer, image-fit preset, screenshot sequence, and image-derived bundle
were removed; the corresponding history remains recoverable through Git. Other
pre-refactor reproduction sources remain in `legacy_scripts/` and in the
`v0.1.0-working-baseline` tag.
