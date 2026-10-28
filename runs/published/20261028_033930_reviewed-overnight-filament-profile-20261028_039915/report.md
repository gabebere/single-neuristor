# Reviewed overnight filament-fraction experiment — 28 October 2026

**The broader search did not recover a quantitative, converged match to the presentation data.**
No candidate passes every prespecified gate and no calibration is adopted.
This bounded result neither identifies f nor rules out current localization.

## Coverage and provenance

The frozen job ran for **7.064 hours**, from 21:35 on
27 October to approximately 04:39 on 28 October, Israel time. It completed all
76 stages, including 43 searches, 21 profile verifications, six final replays and
six long-control cases, before the original eight-hour deadline. Nothing was skipped.
There were **21,232 stage-unique candidate evaluations**; repeated starts across
stages are included in that total. The saved vectors contain 20,698 distinct
physical parameter/f combinations. No restart was required.

Nine declared fractions (1 to 0.01), two thermal-bound families, forward/reverse
continuations and four training-selected geometric midpoints were tested.
The six static R(T) constants and T0 stayed fixed; C, Se, tau=Cth/Se and gamma
were refitted. The scaled family multiplies the Se interval by f while leaving
tau bounds unchanged. That is a separate diagnostic thermal assumption.
The authoritative solver and its normalized parallel-conductance model were unchanged.

Initial optimization used eight currents with three excluded by rank; all 22
recordings were replayed. Those records were seen historically and informed old
seeds, so these are not blind validation data. Final fits and selection use all
11 targets. Paired primary and mean-aware attempts have the same bounds/budgets
and declared warm starts, with independent deterministic random populations;
differences between their outcomes are not a controlled causal estimate of the
mean term's effect. All fits stopped under finite search/evaluation allowances,
not a proof of optimizer convergence.

Numerical source: **8ae0081ba73b90474545ca8032843bba057a1c06**.
Original full local bundle: `20261027_193458_overnight-fixed-f-filament-profile-and-cross-cur_5e8844`.
All 94 frozen input/source files and all 76 completed-stage inventories pass
hash verification. Independently reconstructed native voltage features match
the saved final tables for all **132 finest-step histories**. Branch-current
and Joule-power identities agree to maximum relative error
**2.59e-07**. This checks branch
accounting, not device-terminal interpretation or experimental validation.

## Matched final comparisons

All rows use the finest **0.003125 ns** replay. Frequency MAPE uses the displayed
number of defined frequencies, which can exceed the number passing sustained-cycle
checks. Vpp errors use all 11 targets; mean-voltage RMSE uses all 22 recordings.
Outside-T counts refer to the full configured 284.8545–375.1625 K domain.
Feature/waveform pair counts cover two timestep halvings on 11 targets (22 pairs).

| Family | f | Objective | Valid | Defined | Frequency_MAPE | Vpp_MAPE | Mean_RMSE | Outside_T | Feature_pairs | Waveform_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Inherited | 1 | Frequency + Vpp | 10/11 | 10/11 | 16.59% | 1437.3% | 35.52 mV | 0/22 | 17/22 | 0/22 |
| Inherited | 1 | + mean voltage | 9/11 | 10/11 | 35.17% | 1582.7% | 46.24 mV | 0/22 | 17/22 | 2/22 |
| Inherited | 0.75 | Frequency + Vpp | 10/11 | 10/11 | 19.80% | 1898.5% | 69.64 mV | 4/22 | 12/22 | 2/22 |
| Inherited | 0.75 | + mean voltage | 10/11 | 10/11 | 19.80% | 1898.5% | 69.64 mV | 4/22 | 12/22 | 2/22 |
| Scaled thermal volume | 0.866025 | Frequency + Vpp | 10/11 | 10/11 | 22.89% | 1677.2% | 60.48 mV | 0/22 | 15/22 | 2/22 |
| Scaled thermal volume | 0.866025 | + mean voltage | 10/11 | 10/11 | 24.42% | 1655.8% | 60.75 mV | 0/22 | 14/22 | 2/22 |

The primary f=1 baseline has the lowest frequency/Vpp/mean errors among these
six retained final candidates, yet still misses one target, has Vpp MAPE about
1,437%, violates one quiet-control criterion and passes none of the 22 target
waveform comparisons. Smaller fractions do not solve that mismatch in this
bounded search. At f=0.75 the retained primary and mean-aware vectors coincide;
four of 22 measured-input paths exceed the calibrated temperature domain,
reaching 379.82 K. Scaled-family f≈0.866025 avoids those short-replay excursions
and false quiet-control oscillations, but its amplitude and mean errors remain
larger than the f=1 baseline. None has an accepted all-target scientific loss.

Some very small-f profile candidates have Vpp error near 100% because their
oscillations disappear. That is not amplitude agreement. Failed/undefined cycles,
temperature/proximity exclusions and control errors remain visible in
`audited_verification.csv`; the unchanged original summary remains in `verified_candidates.csv`.

## Timestep and long-time checks

Final replays use 0.0125, 0.00625 and 0.003125 ns. No final candidate passes the
full feature-and-waveform convergence gate. Feature-only comparisons pass 12–17
of 22 pairs; waveform comparisons pass 0–2 of 22. Phase-sensitive waveform
failure is a failed operational numerical criterion, not proof of a physical
instability or a unique diagnosis of the integrator.

All six 2 µs constant-current cases fail the joint persistence/physical/convergence
gate. The f=1 primary case oscillates at all three selected currents at all
three steps, with convergent features, but the high-current case (~569.55 µA)
leaves the temperature domain at every step. It passes two of the three
current-specific joint checks. The other five retain no sustained low-current
oscillation (~228.28 µA), pass the middle-current check and also leave the
temperature domain at the high current. Thus they pass one of three joint checks.
All long controls keep admissible proximity. The full current-by-current record
is in `long_control_audit.csv` and the unmodified constant-current tables.

## Reporting correction discovered during review

DataFrame conversion changed missing-cycle `None` losses to NaN. The original
post-processing helper consequently serialized partial diagnostic losses as null.
The optimizer's direct-row objectives and original acceptance gates were not
changed by that reporting path. The audit explicitly restores missing status
before averaging defined losses. **Both selected refinement seeds are identical
under the corrected ranking**, and every recomputed acceptance gate matches the
original. The maintained helper is fixed with a regression test (post-run commit
1322225); frozen code, optimizer records and completed evidence remain unchanged.
`audited_verification.csv` records corrected partial values and denominators;
accepted losses remain undefined. It also distinguishes all-record temperature
maxima from the original summary's target-only maximum.

## Recommended next step

Do not infer a specimen f from this negative fit or add an arbitrary f(t) simply
to reduce the loss. First reconcile the exported channels with device-terminal
voltage and imposed current: document probe nodes, conversion factors, gain,
offset and channel delay, ideally using a known-resistor control. In parallel,
perform a dedicated upstream-fidelity/timestep audit on a few fixed representative
vectors before interpreting another optimizer as calibrated physics. Preserve
the inherited float32 hysteresis unless that dedicated audit justifies a change.

If channel mapping is confirmed and independent evidence indicates surrounding
film warming, a two-temperature model with measured/constrained thermal coupling
is a more interpretable next hypothesis than a freely varying f. That extension
has not been implemented or tested here.

## Archive contents and reproducibility

This separate compact review preserves the original `metrics.json`,
`resolved_config.json`, inputs, executable source, plans, selected stage records,
all verification/convergence/constant-control tables, all six final sampled
waveform tables and final figures. `preserved_artifact_hashes.json` proves byte
identity. The active review manifest lists only files actually included.
The original manifest/state/report are retained under `original_*` names.

The approximately 964 MB full numerical bundle stays local and immutable.
Full optimizer JSONL histories, large native NPZ arrays, intermediate checkpoint
details and profile-only sampled waveforms are deliberately omitted here.
`full_local_inventory.json` gives sizes, hashes and an explicit inclusion flag.
Those omitted bytes are **not backed up on GitHub** by this compact review.
The versioned recipe and original measured numerical inputs support rerunning
the workflow; wall-clock allowances and hardware can select different optima,
so a new search is not promised to reproduce optimizer choices bit for bit.

Run `neuristor analyze filament-profile --config experiments/current/specimen_filament_overnight_20261027.toml`
from an installed checkout. Use the preserved source commit for the exact original
implementation. Copied execution configurations are original provenance, not
instructions to resume this completed review. No third-party PDFs are included.
