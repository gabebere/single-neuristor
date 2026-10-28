# Partial-switching material-model pilot — 28 October 2026

**Completed: none of the tested mechanisms yields a validated fit.** This follows the negative
[overnight fixed-f study](OVERNIGHT_FILAMENT_PROFILE_20261027.md). Gabriel
authorized a small model-comparison pilot. It introduces optional hypotheses;
the supported original solver, its float32 hysteresis and completed evidence
remain unchanged. No calibration is adopted.

The numerical run completed in **14.12 minutes**: 87 declared cases plus 32
shared local probes, seven all-current replays/ablations and four 2 µs control
cases. The [reviewed offline comparison](../runs/published/20261028_111037_reviewed-partial-switching-material-model-pilot-_09fe72/START_HERE.html)
preserves all cases, source/input hashes and the independent native-output audit.

## Reviewed outcomes

All values below use the finest 0.003125 ns replay. Frequency MAPE uses the
displayed defined denominator; Vpp uses all 11 targets and mean RMSE all 22
records. The supported temperature domain is 284.8545–375.1625 K.

| Pilot selection | Valid targets / defined frequencies | Frequency MAPE | Vpp MAPE | Mean RMSE (mV) | All-record max T (K) |
|---|---:|---:|---:|---:|---:|
| Original model | 7/11 / 7/11 | 32.83% | 559.1% | 37.97 | 363.52 |
| Warm parallel | 10/11 / 10/11 | 15.06% | 1,639.0% | 64.02 | 415.79 |
| Series bottleneck | 4/11 / 4/11 | 21.96% | 522.0% | 45.80 | 367.55 |
| Finite switching time | 9/11 / 9/11 | 36.67% | 1,087.9% | 43.47 | 394.65 |

The series candidate's smaller aggregate amplitude error partly reflects lost
oscillations: predicted Vpp at the three pilot currents is 0, 312.6 and 0 mV,
versus measured 101.8, 34.8 and 9.6 mV. Warm parallel regions retain more cycles
but remain far too large and exceed calibration. Finite switching time produces
rounder waves and reduces some swings, but slows the oscillations and still
overheats. The selected lag is the largest tested value (6 ns), not an identified
switching time. Both selected spatial variants use zero mutual coupling among
the two values tried; this is not evidence that real coupling is zero.

None captures the decreasing-amplitude trend while retaining all three pilot
cycles. None passes the full waveform-convergence, shape or long-control gate.
The series long controls retain only 3/9 sustained current/timestep records;
warm and phase-lag controls retain 9/9 but have respectively 0/9 and 6/9 physical
records. The pilot baseline has 6/9 sustained and 8/9 physical long-control rows.

The selected pilot baseline differs from the earlier overnight reference,
which retained 10/11 targets with frequency MAPE 16.59%, Vpp MAPE 1,437.28% and
mean RMSE 35.52 mV. Coarse five-record physical ranking and the small shared
refinement selected a different trade-off here. The new seven-target baseline
does not supersede that earlier reference. Coarse exclusions can change with
refinement, and these sparse grids do not establish complete fine-step feasibility
or rule out either physical mechanism.

All 520 original artifact hashes and all frozen-source hashes pass. All 20
saved native histories reconstruct the reported voltage features; heat allocation
matches V²/R within 2.38e-7 relative error, with independent topology-specific
current/voltage checks. The full suite passes **246 tests** and validation covers
**51 recipes**. The compact review omits the large native arrays, with explicit
hash/inclusion inventory; those bytes remain local, not backed up on GitHub.
Source numerical commit: `0e39b5e`.

This supplies no verified quantitative improvement warranting another large fit
without better circuit/channel constraints. Obtain the actual TIA/probe/current
boundary information; a fixed gain/offset correction was already insufficient.
Further material work needs constrained geometry/minor-loop/phase assumptions,
with the same full-current and finer-timestep checks. No calibration is adopted.

```bash
neuristor analyze model-pilot --plan
neuristor analyze model-pilot
```

Versioned recipe:
[`specimen_model_pilot_20261028.toml`](../experiments/current/specimen_model_pilot_20261028.toml).
The new local bundle preserves original inputs, source, all cases, numerical
tables and an offline HTML viewer. This is a finite grid and local follow-up,
not another overnight global optimization. Runtime depends on the machine.

## Question

From about 267 to 606 µA, measured Vpp decreases from roughly 102 to 10 mV while
mean output remains near 180–190 mV. The retained overnight baseline instead
has Vpp about 397–297 mV and repeatedly drops nearly to zero. Can an additional
material state or spatial redistribution produce small oscillations about a
finite mean, with one shared parameter vector across currents?

The exported circuit boundary is still unresolved. This pilot does not model
the unknown active TIA, assign its recorded current to a newly assumed node,
or fit per-record gains, offsets, delays or arbitrary f(t).

## Three separate hypotheses

**Warm parallel regions.**

`G = w/Rfit(T1,H1) + (1-w)/Rfit(T2,H2)`.
Each branch receives its own `V²Gi` heat. Both temperatures evolve:

```
C1 dT1/dt = P1 - S1(T1-T0) - K(T1-T2)
C2 dT2/dt = P2 - S2(T2-T0) + K(T1-T2)
```

Mutual exchange cancels in the total energy balance. The fraction w stays
constant, while the branch-current fraction changes with both conductances.

**Series VO₂ bottleneck and access region.**

`Rdevice = w Rfit(T1,H1) + (1-w) Rfit(T2,H2)`.
Both regions carry `Idevice = V/Rdevice` and receive `Idevice²Rregion` heat.
Their voltage drops add to the device voltage. They use the same two thermal
equations. These are regions inside the measured VO₂ device, not an external
resistor changing an ideal imposed current. Neither topology double-counts a
new contact contribution on top of the measured R(T).

Both spatial hypotheses recover `Rdevice=Rfit(T,H)` when their temperatures and
hysteresis states coincide. The reference Cth/Se budgets are split by w and
1-w, then secondary ratios introduce asymmetry. Those relations are explicit
diagnostic assumptions; electrical weights are not measured thermal volumes.

**Finite switching time.**

```
tau_phi dx/dt = g_hyst(T,H) - x
R(T,x) = Rm + R0 exp(Ea_over_k/T) x
```

x is the semiconducting weight, not a second geometric fraction. Hysteresis
anchors continue to follow the original quasi-static target; actual resistance
uses the relaxing state. This phenomenological lag adds no separate latent-heat
model. Slow/constant-temperature equilibration recovers the existing R(T).
Tau=0 delegates exactly to the original solver. Phase lag and spatial changes
are not combined in this pilot.

All models retain `C dV/dt = Iin - V/Rdevice`, including exact algebraic
`V=Iin Rdevice` for C=0. The experimental spatial all-metal uniform limit still
has resistance Rm. Each frozen electrical/thermal/phase subproblem is advanced
with its exact exponential; mutual thermal evolution uses a matrix exponential.
The coupled nonlinear trajectory still requires timestep refinement.

## Declared comparison

- Three shared starts: the retained overnight primary vector and both historical
  presentation vectors. All six static R(T) constants and T0 remain fixed.
- Initial grid: three baselines; 36 warm-parallel and 36 series cases;
  12 phase-lag cases, **87 total**.
- Spatial weights: 0.25, 0.5, 0.75; secondary capacity ratios: 10, 100;
  secondary cooling ratio: 1; mutual-coupling ratios relative to reference Se:
  0, 1. Phase times: 0.2, 0.6, 2, 6 ns. These are exploratory values.
- One best case per arm receives shared one-coordinate probes: C, Se and tau_th
  multiplied separately by 0.7/1.4, gamma changed by ±0.05, with declared bounds
  and all clamps recorded. At most **32** additional cases.
- Screening step: 0.025 ns, full measured prehistory. Three pilot target records
  are chosen before fitting by nearest measured currents to 267, 419, 606 µA
  (actual 266.63, 418.86, 606.28 µA). Controls are 189.78 and 907.04 µA.
- Ranking prioritizes temperature/proximity, target cycles and excessive control
  oscillations, then relative frequency/Vpp plus mean and peak/trough errors
  normalized by 20 mV. Missing cycles never receive an accepted scientific loss.
  Control excess is sustained predicted Vpp above max(10 mV, twice measured Vpp).
- One winner per arm plus three mechanism-off comparisons replay **all 22**
  inputs at 0.025, 0.0125, 0.00625 and 0.003125 ns: up to seven cases. Each
  mechanism-off case keeps the exact selected reference vector and restores
  the original single-region solver. Removing the spatial model also removes
  its reference-budget partition; this is not an independent geometry measurement.
- All four mechanism winners receive 2 µs constant-current controls at the three
  pilot currents and three finest timesteps. Native finest measured-input
  channels are saved for the five pilot/control records.

The original full-target frequency, Vpp, mean, temperature, proximity, quiet-control
and waveform-convergence gates apply, with an additional ≤20 mV peak/trough RMSE
gate on all targets. Both spatial temperatures and hysteresis slopes are checked.
History/seed-informed records are not blind validation. Pilot improvement is
distinct from success across all 11 targets and all 22 controls/records.

## Verification and evidence

Independent controls check uniform-temperature normalization, topology-specific
current/voltage identities, positive branch powers, thermal exchange conservation,
the phase exponential and the exact C=0 limit. Coupled temperatures are compared
with an independent RC-driven thermal ODE at three timesteps. Baseline/zero-phase
paths preserve the original solver; 116 pre-change scalar/batch output arrays
are bitwise identical after implementation. The 116-array capture covers batched
waveform outputs. Independent scalar/default checks remain in the existing suite.
Analytic RC charging checks retain their tight tolerances; the late float32
plateau is additionally checked by exact original-solver identity.

Every finite or failed screen remains in the output. Exact specifications,
feature tables, actual SI parameters, mechanism-off controls, convergence tables,
long controls, original numerical inputs, package source and hashes stay with the
bundle. The offline viewer compares all four winners at every measured current.
Any reviewed publication must preserve completed original bundles and explicitly
identify omitted large native arrays. No third-party reference PDFs are inputs.
