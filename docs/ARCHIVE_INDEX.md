# Research archive index

This index distinguishes current quantitative evidence from historical simulation and
presentation artifacts. That distinction matters because Git history preserves earlier
experiments, while only the current solver and audit should be used for new numerical
claims.

## Current model authority

- `src/neuristor/model.py`: Yuanhang hysteretic resistance and voltage-driven model.
- `src/neuristor/current_drive_sim.py`: supported ideal-current single-device model.
- `docs/HYSTERESIS_IMPLEMENTATION_AUDIT.md`: branch-memory implementation, numerical
  ordering, provenance, and fidelity checks.
- `docs/CURRENT_DRIVE_CALIBRATION.md`: voltage-floor diagnosis, parameter
  identifiability, capacitance study, and recommended laboratory calibration sequence.
- `docs/manuscript/theory_behind_simulations.pdf`: compiled model theory manuscript.

## Current corroborating evidence

- `docs/figures/current_drive/capacitance_study/frequency_heatmaps_all_currents.png`:
  shared-scale `C_th`-versus-`C` frequency maps for six currents.
- `docs/figures/current_drive/capacitance_study/capacitance_trace_comparison_600uA.png`:
  direct demonstration that `C` changes timing while the metallic bound is `I*Rm`.
- `docs/figures/current_drive/lab_estimates/voltage_floor_comparison.png`: digitized
  laboratory plateau compared with fitted-specimen and Yuanhang ideal-current floors.
- `docs/figures/current_drive/lab_estimates/lab_parameter_estimates.png`: independently
  supported electrical capacitance, ambient/cooling degeneracy, and effective plateau
  resistance.
- `docs/figures/current_drive/nonzero_valley_examples/current_input_voltage_output.png`:
  direct imposed-current/voltage-output examples, including an unchanged specimen
  control and an explicit nonzero-valley effective-resistance candidate.
- The adjacent CSV, JSON, and README files preserve the plotted numerical values,
  simulation settings, assumptions, and validity warnings.

## Historical artifacts

- `public_jobs/`: immutable July 2026 Streamlit run records. See its README for an index
  and the correction-status caveat.
- `Simulations_on_VO2/`: July 2026 Beamer presentation source, compiled deck, figures,
  and animations.
- `docs/presentation/project_year_presentation_outline.md`: longer historical talk
  outline, explicitly marked with its current-status note.
- Git history is the recovery path for removed experimental current-source variants and
  obsolete presets. They are not active physical models.

## Reproduction and verification

From the repository root:

```bash
python scripts/sweep_current_capacitances.py \
  --output-dir docs/figures/current_drive/capacitance_study
python scripts/estimate_lab_current_parameters.py \
  --output-dir docs/figures/current_drive/lab_estimates
python scripts/generate_nonzero_valley_examples.py
python scripts/audit_model_fidelity.py
python scripts/check_current_drive_model.py
python -m unittest discover -s tests -v
```

The lab estimator digitizes the tracked `data/Current Results/` frames by default; it
does not require an ignored local output file. The stored March image-fit JSON is
retained only as a correlated historical starting point, not an independent parameter
measurement.
