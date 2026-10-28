# Overnight fixed-f filament profile

No candidate passed every prespecified observable, physical, timestep and long-control gate. **No calibration is adopted.** An unsuccessful bounded search does not exclude a filament or identify a unique fraction.

## Method

The authoritative ideal-current solver is unchanged. Its optional model uses
`G_eff = f/R_fit(T_filament,H_filament) + (1-f)/R_fit(T0,H0)`, with only
`f V²/R_fit(T_filament,H_filament)` warming the hot node. The cold branch is an ambient reservoir.
The six fitted R(T) constants and T0 are fixed. Fits vary C, Se, tau=Cth/Se and gamma.

Declared fractions: [1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01]. Inherited bounds preserve the previous presentation intervals.
The second family multiplies the Se interval by f and keeps the tau interval, allowing a proportionally
smaller heated volume. That scaling is an explicit diagnostic hypothesis, not a measured heat-flow law.
Starts include the prior baseline, both presentation vectors, direct and coupled thermal starts,
Latin-hypercube populations, neighboring fractions, reverse continuations and training-selected geometric midpoints.
All clamps are recorded. Circuit capacitance is never geometrically scaled.

Initial fits use 8 targets; 3 are optimizer-excluded by current rank. All 22 measured inputs are replayed.
These records and historical seeds have been seen before: this is not blind validation.
Final paired frequency/Vpp and mean-aware refinements use all 11 targets. Their selection uses full-current checks,
so initial exclusions cannot be presented as independent test evidence after refinement.
Primary loss is equal-weight squared relative frequency and robust-Vpp error. Mean-aware loss adds
`((mean_sim - mean_meas)/20.0 mV)^2` with unit weight.
Finite rejection scores rank invalid candidates; they are not accepted scientific losses.

Search step: 0.025 ns; profile checks: [0.025, 0.0125, 0.00625] ns.
Refinement step: 0.0125 ns; final checks: [0.0125, 0.00625, 0.003125] ns.
The two smaller steps must each pass convergence. Long constant-current controls last
2000.0 ns, at three measured sustained-target currents selected before fitting.
Native finest-step final histories preserve the electrical split and heating channels.

## Prespecified acceptance

All 11 target cycles/frequencies must be valid; all 22 records must remain in the configured temperature
domain with admissible proximity. Frequency MAPE ≤ 10% and worst error ≤
15%; Vpp MAPE ≤ 20% and worst error ≤
30%. All-record mean-voltage RMSE ≤
20 mV. Quiet controls must not acquire false large oscillations.
Both timestep halvings require feature changes ≤ 2% and
waveform RMS difference ≤ 2 mV on all targets.
Long controls must remain sustained, physical and feature-converged at all three steps.

## Execution and evidence

Elapsed: 7.064 hours; wall allowance: 8.0 hours;
21232 unique evaluations across 43 search stages.
Completed profile points: 21; final candidates verified:
6; long-control candidate cases: 6.
Coverage complete: True. Budget-skipped stages: [].
Resume attempts: 1. Resume preserves cached evaluations and the original wall deadline;
optimizer trajectories may differ after interruption. Completed stage and frozen-input hashes are checked.

Open [START_HERE.html](START_HERE.html), [verified_candidates.csv](verified_candidates.csv),
[search_parameters.csv](search_parameters.csv), [current_split.csv](current_split.csv) and the stage subdirectories.
`frozen_hashes.json`, `source_hashes.json`, `environment.json`, copied inputs and executable source establish provenance.

## Interpretation limits

The circuit channel/current conversion, readout transfer and geometric thermal mapping remain unresolved.
A successful conditional fit would warrant follow-up measurements, not automatic physical identification.
Frequency MAPE on defined records must always be read beside its denominator; missing cycles are never scored as zero error.
Failures, temperature excursions and coarse-only oscillations remain visible in the tables.
