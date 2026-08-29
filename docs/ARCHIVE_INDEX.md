# Research archive index

This index distinguishes current quantitative evidence from historical simulation and
presentation artifacts. That distinction matters because Git history preserves earlier
experiments, while only the current solver and audit should be used for new numerical
claims.

## Active final-project manuscript

- `docs/final_project/`: human-facing final-project hub. Its README organizes the
  editable manuscript, current PDF, raw resistance and oscilloscope data, reviewed
  run bundles, and supplementary figures and animations without duplicating their
  canonical sources.
- `docs/final_project/main.tex`: working LaTeX manuscript for the VO2 automatic-gain-
  control simulation project. The Yuanhang validation and sample-specific R(T) fit are
  complete, as are the electrical-capacitance, environmental-conductance, thermal-
  capacitance, blind current-comparison, capacitance-sensitivity, and global inverse-
  fitting studies.
- The Case I figure and metrics point to the immutable public run bundle
  `20260817_100102_current-step-with-a-nonzero-metallic-voltage-val_6765e0`, whose voltage
  panel includes the instantaneous metallic fixed point `I(t) R_m`, rather than duplicating
  scientific output inside the manuscript directory.
- `docs/final_project/supplementary/` presents the manuscript's derived R(T)
  trajectory, synchronized current/voltage--R(T) animation, current evidence figures,
  and selected historical animations. Generated evidence remains authoritative in its
  immutable `public_jobs/` bundle.
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
- `20260827_142103_measured-laboratory-current-sweep-with-current-l_ec6ec4`: canonical
  archive of all 22 normalized traces, the measured 41.7--62.5 MHz operating window,
  separate onset waveforms, and the shared-axis bracket used in the manuscript. Its
  labels lead with the measured current step and retain the source-voltage setting
  only for provenance.
- `20260827_123959_measured-laboratory-current-sweep-with-onset-bra_82870f`: earlier
  immutable onset-bracket archive superseded by `ec6ec4`, which clarifies the
  measured-current versus source-setting distinction.
- `20260827_120011_measured-laboratory-current-sweep-with-onset-tra_ecc00b`: earlier
  archive retained immutably and superseded by `82870f`, which adds the 250 mV
  non-oscillating control and direct shared-scale comparison.
- `20260817_134254_measured-laboratory-current-sweep_a45254`: earlier numerically
  identical sweep archive retained immutably and superseded by the later onset-evidence
  bundles.
- `20260817_153807_environmental-thermal-conductance-estimate_761640`: canonical
  clean-provenance selected settled
  pre-onset waveform, fitted R(T) temperature inversion, conditional conductance
  interval, numerical tables, and evidence figure.
- `20260828_112314_thermal-capacitance-estimate-with-conservative-0_aa2469`: canonical
  heating-edge reconstruction using the adopted electrical upper bound `C=0.39 pF`.
  It subtracts `C dV/dt` before forming resistance and power, then fits the shared
  `tau_th` and `C_th` with the conditional robustness interval and near-transition
  sensitivity check.
- `20260829_100718_specimen-model-prediction-versus-measured-curren_eefab7`: canonical
  blind prediction test. It replays all 22 measured current waveforms through one
  frozen specimen parameter set, archives common-window measured/predicted metrics and
  convergence evidence, and maps C--Cth sensitivity. The stable pre-onset voltage is
  reproduced, but the adopted model predicts none of the 11 measured oscillatory runs;
  oscillations require electrical capacitance outside the 0.39 pF timing bound.
- `20260829_105704_global-specimen-parameter-inference-from-all-cur_8f12d6`: canonical
  global inverse fit using 17 training and five held-out current settings. It archives
  the complete objective, constrained and relaxed parameter searches, per-trace
  predictions, optimization history, and time-step convergence. The constrained fit
  cannot restore oscillation; the relaxed fit improves waveform statistics only by
  leaving seven of eight physical intervals and producing turn-on transients.
- `20260829_153853_oscillation-priority-global-specimen-parameter-i_fe76d2`: current
  oscillation-priority diagnostic. Its four-segment persistence score and asymmetric
  classification penalty recover ten consecutive oscillatory currents with 21/22
  classifications and no false positives. The identical window at 0.025 ns is
  numerically stable, but all eight fitted values violate independent physical
  intervals and the predicted amplitude remains too large.
- `20260828_112026_thermal-capacitance-estimate-with-conservative-0_ce51fa`: earlier
  numerically identical upper-bound run retained immutably and superseded by
  `aa2469`, whose figure labels the corrected ratio as `V/I_R`.
- `20260817_161851_thermal-time-constant-and-capacitance-estimate_b3833c`: earlier
  immutable `C=0` thermal fit, retained as the lower-endpoint sensitivity analysis
  and superseded for the manuscript by `ce51fa`.
- `20260817_161539_thermal-time-constant-and-capacitance-estimate_c285f9`: numerically
  identical pre-provenance run retained immutably and superseded by `b3833c`, whose
  resolved configuration also records `C=0` and the nine-sample smoothing window.
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
neuristor analyze thermal-capacitance \
  --data data/experimental/tia_current_sweep \
  --resistance-preset presets/resistance_100425_chip1_gap3.json \
  --resistance-bootstrap public_jobs/20260816_125905_sample-r-t-major-loop-hysteresis-fit_0849a9/parameter_bootstrap.csv \
  --conductance-mW-per-K 0.003675126546984294 \
  --conductance-bootstrap public_jobs/20260817_153807_environmental-thermal-conductance-estimate_761640/conductance_bootstrap.csv \
  --ambient-K 314.4 --electrical-capacitance-pF 0.39 \
  --selected-drives-mV 100,150,200 --fit-window-ns 15,35
neuristor analyze model-validation \
  --config experiments/current/specimen_model_validation.toml
pytest -q
neuristor validate
```

The laboratory workflows read the checked-in numerical exports directly. The former
screenshot digitizer, image-fit preset, screenshot sequence, and image-derived bundle
were removed; the corresponding history remains recoverable through Git. Other
pre-refactor reproduction sources remain in `legacy_scripts/` and in the
`v0.1.0-working-baseline` tag.
