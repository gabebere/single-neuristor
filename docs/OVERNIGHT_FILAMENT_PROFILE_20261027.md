# Overnight filament-fraction profile — 27 October 2026

**Completed 28 October: no quantitative, converged fit found.** This is a
bounded follow-up to the [249-evaluation trial](FILAMENT_SURROUNDINGS_20261027.md),
not an adopted specimen calibration. Gabriel authorized an overnight search.

The full job **started at 21:35 Israel time on 27 October** and completed at
approximately **04:39 on 28 October**, before the original 05:35 deadline.
It ran 7.06 hours and completed all 76 stages without a restart or skipped stage.
Reviewed entry point:
[`START_HERE.html`](../runs/published/20261028_033930_reviewed-overnight-filament-profile-20261028_039915/START_HERE.html).
Frozen source commit: `8ae0081ba73b90474545ca8032843bba057a1c06`, pushed on
`codex/overnight-filament-profile`. The original full local numerical bundle is
`20261027_193458_overnight-fixed-f-filament-profile-and-cross-cur_5e8844`.
The compact review preserves original metrics/configuration and input/source hashes;
large arrays and optimizer histories remain local and are not backed up on GitHub.

## Completed results

The 43 searches made **21,232 stage-unique evaluations**, including repeated starts
across stages (20,698 distinct saved physical vectors). Twenty-one profile points,
six final candidates and six 2 µs control cases were verified. The finest final
step is 0.003125 ns. Frequency MAPE below uses 10 defined frequencies in every
row; valid sustained-target counts can be smaller. Vpp errors use all 11 targets;
mean-voltage RMSE uses all 22 records.

| Final candidate | Valid targets | Frequency MAPE | Vpp MAPE | Mean RMSE (mV) |
|---|---:|---:|---:|---:|
| f=1, primary | 10/11 | 16.59% | 1,437.3% | 35.52 |
| f=1, mean-aware | 9/11 | 35.17% | 1,582.7% | 46.24 |
| f=0.75, inherited, primary | 10/11 | 19.80% | 1,898.5% | 69.64 |
| f=0.75, inherited, mean-aware | 10/11 | 19.80% | 1,898.5% | 69.64 |
| f≈0.866025, scaled, primary | 10/11 | 22.89% | 1,677.2% | 60.48 |
| f≈0.866025, scaled, mean-aware | 10/11 | 24.42% | 1,655.8% | 60.75 |

No final candidate passes both timestep halvings: feature comparisons pass 12–17
of 22 target/halving pairs and waveform comparisons only 0–2 of 22. At f=0.75,
four measured-input paths exceed the 284.8545–375.1625 K temperature domain,
reaching 379.82 K. Short-replay proximity is admissible in all six final cases.
The scaled alternatives avoid false quiet-control oscillations; the primary f=1
case has one and inherited f=0.75 has two. No candidate passes every gate.

All final 2 µs controls leave the temperature domain at ~569.55 µA at every
step. Only the primary f=1 case remains sustained at all three selected currents;
the other five lose low-current oscillations at ~228.28 µA. Thus the primary f=1
case passes two of three current-specific joint long-control checks and the others
one of three, rather than establishing a fully physical persistent fit.

All 94 frozen input/source files and 76 completed-stage inventories pass hash
verification. All 132 native final histories reconstruct the saved voltage features
and preserve branch-current/power identities within 2.60e-7 relative error.
A DataFrame missing-value issue erased partial diagnostic losses in the original
summary. Audited finite-loss summaries restore them and verify that both selected
refinement seeds and all acceptance decisions are unchanged. Maintained reporting
was corrected in commit `1322225` with a regression test; the frozen search and
original evidence were not modified. The full software suite passes 224 tests.

This bounded negative result does not identify f, establish a global optimum or
rule out current localization. Before adding arbitrary f(t), confirm channel
conversion/gain/offset/delay and probe nodes with a known-resistor control, and
audit timestep/hysteresis fidelity on fixed representative vectors. If surrounding
warming is independently supported, a constrained two-temperature model is a
possible next hypothesis, not a result of this search. No calibration is adopted.

```bash
neuristor analyze filament-profile --plan
neuristor analyze filament-profile
```

The default versioned input is
[`specimen_filament_overnight_20261027.toml`](../experiments/current/specimen_filament_overnight_20261027.toml).
Allow **up to eight hours**, two numerical worker processes and several GB of
local output. Timing is a ceiling checked between evaluations/integrations;
an integration already running can finish slightly beyond a deadline.
Keep the computer powered, awake and connected to power. The command writes a
new ignored `runs/local/` bundle; `START_HERE.html` updates during the run.

## Question and model

Does a constant conductance weight f improve the presentation-data fit after
jointly refitting the electrical and thermal dynamics? The previous small search
returning f=1 did not identify f or rule out a filament. A change in f alters
both the effective resistance and the power deposited in the hot region, so
varying f while freezing all thermal parameters can be misleading.

The existing optional model is unchanged:

```
G_eff = f/R_fit(T_filament, H_filament) + (1-f)/R_fit(T0, H0)
P_filament = f V²/R_fit(T_filament, H_filament)
C dV/dt = I - V G_eff
```

The cold branch stays at the ambient reservoir. Same electrodes and uniform
stage heating were confirmed for the measured R(T). Thus the weights sum to one
and recover the measured law at uniform temperature. f is a conductance weight,
not a measured geometrical area. No time-dependent f or new physics is added.

## Search design

The six fitted static R(T) constants and T0=314.4 K remain fixed. C, Se,
tau=Cth/Se and gamma are refitted independently at each fraction.

| Component | Declared design |
|---|---|
| Fixed fractions | 1, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01 |
| Inherited thermal family | Original Se and tau intervals |
| Volume-scaled thermal family | Se interval multiplied by f; tau interval unchanged, allowing Cth to shrink with f |
| Reference f=1 | One shared search, 20 minutes / 512 unique evaluations maximum |
| Forward continuation | Eight fractions per family; 40 minutes / 1,600 evaluations each |
| Reverse continuation | Eight per family, smallest to largest f; 5 minutes / 1,600 each |
| Adaptive midpoints | At most two geometric neighbors around the best training-score fraction per family; 5 minutes each |
| Search population | Four restarts, population 24, 14 DE generations plus up to 48 Powell calls per restart |
| Primary objective | Equal-weight squared relative frequency and robust Vpp errors |
| Initial timestep | 0.025 ns |
| Profile verification | All 22 records at 0.025, 0.0125 and 0.00625 ns |
| Fine refinements | f=1 and the strongest fractional point from each family; paired primary and mean-aware objectives |
| Refinement allowance | Six cases, 10 minutes / 256 evaluations each, two restarts, population 16, 0.0125 ns |
| Mean-aware term | Add squared mean-voltage error normalized by 20 mV, unit weight |
| Final verification | All 22 records at 0.0125, 0.00625 and 0.003125 ns; finest-step native channels saved |
| Long controls | 2 µs at three prespecified sustained-record mean currents, all three final timesteps |

Se inherited bounds are 0.0003–0.03 mW/K, tau 0.5–150 ns, C 0.05–80 pF,
gamma 0.6–0.98. Multiplying the Se interval by f is an explicitly separate
diagnostic heat-flow hypothesis, not measured geometry. Circuit C is never
scaled with f. The f=1 model belongs to both families and is searched once.

Starts include the preceding best baseline, both presentation vectors and the
declared seed table. The new f=1 winner is included in subsequent fits, along
with direct and coupled starts (Se multiplied by the fraction ratio, tau kept
fixed), previous-neighbor and same-f forward winners. Every bound clamp is
recorded. Independent Latin-hypercube starts retain basin diversity. Each family
continues separately; profile selection compares the same primary objective.

Eight of the 11 target currents are used for initial optimization. Zero-based
current ranks 1, 5 and 9 are excluded: approximately **266.63, 418.86 and
569.55 µA**. All 22 measured waveforms, including prehistory, are replayed in
verification. These records were seen historically and informed old seeds, so
they are **optimizer-excluded cross-current checks, not blind validation**.
Final refinements use all 11 targets. Their selection also uses full-current
diagnostics; do not report later results as independent held-out performance.
The 2 µs controls use measured sustained targets selected before fitting:
approximately 228.28, 418.86 and 569.55 µA.

The profile phase has a 6.5-hour ceiling within the overall eight-hour allowance.
Two workers run independent cases with numerical-library threads limited to one.
Unstarted stages past a budget are explicitly recorded as skipped. Counts and
coverage must be checked in the final metrics; exhausting a budget is not proof
of absence and an optimizer stopping at its cap is not convergence.

## Prespecified evidence required for success

- All 11 target cycles/frequencies valid; every one of the 22 records inside
  the configured temperature domain and with admissible proximity.
- Target frequency MAPE ≤10%, worst individual error ≤15%; target Vpp MAPE
  ≤20%, worst individual error ≤30%.
- Mean-voltage RMSE across all 22 records ≤20 mV.
- No simulated sustained oscillation above 10 mV Vpp on a nonsustained measured
  control with measured Vpp ≤5 mV.
- Both timestep halvings retain target classifications and change frequency/Vpp
  by ≤2%; target waveform RMS differences ≤2 mV at each halving.
- Long controls remain sustained and physical, with ≤2% adjacent-step feature
  changes at all three selected currents.

These are operational diagnostic thresholds, not measurement-derived confidence
intervals. Missing cycles remain undefined; the finite optimizer rejection score
is always separate from an accepted scientific loss. Frequency MAPE on defined
records must be reported with its denominator. No calibration is automatically
adopted, even if all these conditional gates pass. Readout transfer, current
conversion, probe nodes and the physical heated volume still need experimental
confirmation.

## Frozen execution and restart

```bash
neuristor analyze filament-profile --prepare
PYTHONPATH="PATH_TO_BUNDLE/source" python -c 'from neuristor.cli import main; main()' analyze filament-profile --resume "PATH_TO_BUNDLE"
```

Preparation copies the exact original numerical inputs, recipe, package sources
and numerical package versions. The execution recipe changes only input paths
to those copies. Run detached jobs from that frozen source; later checkout
changes must not silently affect worker imports. The normal command also
prepares these copies before executing, and checks its runtime source matches.

An exclusive lock prevents two supervisors from writing the same bundle.
Per-stage specifications and seed vectors are saved, every unique evaluation is
appended and flushed, and completed stage hashes are checked before resume.
The original total deadline and unique-evaluation cap persist across restarts;
the optimizer restarts using cached results, so its later trajectory can differ.
Only an incomplete last history line can be discarded after an interruption.
Place a file named `STOP` in the bundle to request a graceful stop; remove it
only when explicitly resuming. Failures leave their trace and completed evidence.

Final outputs include named parameter tables, all-target errors and failed gates,
all-current traces, native final histories, timestep comparisons, long-control
features, profile figures and low/mid/high-current overlays for every final
candidate. Completed public bundles are preserved. Review new evidence before
publishing a separate compact bundle and updating the research handoff.
