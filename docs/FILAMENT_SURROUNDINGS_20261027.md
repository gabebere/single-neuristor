# Experimental hot filament and cold surroundings

Status: implemented and tested on the presentation measurements. The first
bounded trial did not establish an improved quantitative fit. This is an optional
forward-model extension, not a replacement specimen calibration.

Gabriel confirmed that the measured R(T) loop and oscillation recordings use the
same electrode pair, with uniform stage heating for R(T). The proposed mechanism
is a spatially localized hot filament during electrical switching, whereas many
connected paths can become metallic under uniform heating. Unconnected film is
not automatically part of the measured conductance.

Let `R_fit(T,H)` be the existing fitted, hysteretic whole-device law. The reduced
model is:

```text
G_eff = f/R_fit(T_f,H_f) + (1-f)/R_fit(T0,H0)
C dV/dt = I_in - V G_eff
Cth_f dT_f/dt = f V^2/R_fit(T_f,H_f) - Se (T_f-T0)
```

`0 < f <= 1` is one normalized conductance weight. The surrounding weight is
`1-f`, not a second independently fitted coefficient. Its temperature is fixed
at ambient and its independent initial hysteresis state is frozen. Surrounding
Joule heat goes to an ideal external thermal reservoir. We do not allocate that
heat to the filament. A dynamically warming background would require another
thermal state and additional measured constraints.

Equal temperatures **and hysteresis states** give the original uniform R(T)
curve. The `f=1` limit preserves the original integration arithmetic exactly.
At `C=0`, voltage is the exact algebraic `V = I_in/G_eff`. The conditional hot
metallic voltage floor is `I_in/[f/Rm + (1-f)/R_cold]`. Capacitance changes the
approach to that floor. No series resistor or non-ideal source is introduced.
The float32 hysteresis law remains unchanged. Both branch currents, electrical
powers and the filament temperature are retained in native outputs; `P` means
total electrical dissipation while `P_filament` drives the thermal update.

The weight is not an independently measured filament area fraction. Cth and Se
are effective properties of the heated region. Their correlation with f, possible
contact resistance, and the unresolved channel-to-device mapping limit physical
interpretation. The original global R(T) data alone cannot identify f.

## Run the declared presentation-data comparison

```bash
neuristor analyze filament-study --config experiments/current/specimen_filament_surroundings_20261027.toml
```

Allow tens of minutes on a typical computer; this is a planning allowance, not a
benchmark. The two searches each retain the presentation's 900-second allowance
and 256-evaluation cap. Progress and candidates are saved during execution.
An empty `STOP` file inside the printed bundle folder stops at the next candidate
boundary; completed searches and durable JSON checkpoints remain inspectable.

The recipe declares the two presentation reference vectors as inputs. It reads
the original measured currents/voltages, measured R(T), static preset and seed
table, not historical simulated trajectories. The protocol is:

1. Scan f from 1 to 0.01 with each presentation reference's dynamic values fixed.
2. Refit the original four dynamic parameters and then those four plus f, using
   matched bounds, seeds, DE/Powell procedure, elapsed-time allowance and call cap.
   Actual counts can differ through caching and model evaluation cost.
3. Keep all six static R(T) constants and ambient 314.4 K fixed. Fit one parameter
   vector across the same 11 measured 225–610 µA target inputs.
4. Keep the presentation objective: mean squared relative frequency plus Vpp
   error. Report mean voltage separately. Do not add an unmeasured resistance
   target derived from `V/I`, which includes unresolved circuit/current issues.
5. Scan f again with the selected filament fit's other coordinates fixed. This
   conditional scan is not a profile likelihood, a confidence interval or proof
   that the parameters are identifiable.
6. Replay the references, selected scan candidates and both fitted candidates
   across all 22 measured inputs at 0.025, 0.0125 and 0.00625 ns. The other 11
   records are previously seen controls, not independent blind validation.
   Explicit f=0.75 and f=0.1 reference cases receive the same finer checks even
   if the selected fit returns to f=1. Three representative native histories
   per fractional case expose the actual branch currents and powers.
7. Report both timestep halvings, class changes, frequency/Vpp relative changes
   (2% criterion) and waveform RMS differences (2 mV criterion). Missing cycles
   remain undefined. No tolerance is weakened to accept a changed trajectory.

Cycle persistence, the 110 MHz operating ceiling, the configured static-fit
temperature range and monotone proximity screen retain their existing meaning.
An accepted scientific objective is reported only for a candidate passing every
target screen. Diagnostic amplitude MAPE retains the presentation's zero-feature
convention for missing cycles, so a smaller amplitude error alone can reflect
quenching rather than a better oscillatory fit. Frequency errors always state
the number of records with a defined frequency.

Open the generated `START_HERE.html`, then inspect `summary.csv`,
`verification.csv`, `convergence.csv`, `fraction_sensitivity.csv`, `candidates.json`
and `report.md`. `native/` contains full SI branch histories for both new fitted
models at the finest timestep. All manifests, original inputs and recursive code
snapshots stay in a new ignored local RunBundle. Existing presentation files,
viewer routes and completed published bundles are preserved.

To extend verification after a completed run without repeating the searches:

```bash
neuristor analyze filament-study --config experiments/current/specimen_filament_surroundings_20261027.toml --reuse-searches PATH_TO_COMPLETED_BUNDLE
```

This checks unchanged measured inputs, numerical source and search settings.
Only additional verification fractions may differ. It copies the searches with
their original provenance and content hashes, then reruns scans and verification
into a new bundle. No completed source bundle is edited.

## Evidence and physical motivation

- [Presentation comparison](LOSS_COMPARISON_20261020.md).
- [Gildor et al., DOI 10.1002/aelm.70604](https://doi.org/10.1002/aelm.70604)
  distinguishes uniform heating and filamentary switching, and discusses cooler
  surrounding domains and filament widening. The normalized two-region model
  above is our proposed diagnostic approximation, not a model validated by that paper.

## Trial results

Both arms completed **249 unique evaluations**, taking 557.16 s for the baseline
and 551.34 s for the added-parameter arm. The returned filament candidate is
**f=1**, exactly the existing combined-loss presentation vector. It is ineligible,
so this is not a measurement of the filament fraction or evidence against the
localization hypothesis. A finite search in five coordinates can also return a
weaker diagnostic than a search in four: the full f=1 model family contains every
baseline vector, irrespective of which seed a bounded search returns.

At the finest **0.00625 ns** step:

| Candidate | Passing cycle/frequency targets | Defined frequencies | Frequency MAPE on defined records | Diagnostic Vpp MAPE, all 11 | Mean voltage RMSE |
|---|---:|---:|---:|---:|---:|
| Frozen presentation frequency-only | 10/11 | 10/11 | 14.48% | 1965.70% | 83.67 mV |
| Frozen presentation combined | 8/11 | 9/11 | 18.18% | 902.98% | 40.10 mV |
| New baseline refit | 10/11 | 10/11 | 33.78% | 1592.15% | 56.03 mV |
| Added-parameter search, returned f=1 | 8/11 | 9/11 | 18.18% | 902.98% | 40.10 mV |

These different frequency denominators are not interchangeable rankings. None
passes every target screen; all accepted all-target objective values remain null.
Neither fitted candidate passes the feature and waveform convergence criteria
on every target across both timestep halvings. No calibration is adopted.

| Dynamic parameter | Baseline refit | Returned filament candidate |
|---|---:|---:|
| f | 1 | 1 |
| C (pF) | 4.612496 | 6.253977 |
| Cth (pJ/K) | 0.024892 | 0.030069 |
| Se (µW/K) | 4.772955 | 4.096025 |
| Cth/Se (ns) | 5.215323 | 7.340934 |
| gamma | 0.721599 | 0.630809 |

Holding either presentation vector's other parameters fixed, f=0.75 retains
9/11 target cycle/frequency classifications at the finest step, but three
targets exceed the configured temperature domain, reaching 378.25–378.46 K.
Vpp errors rise to about 2266% and 1618%. At f=0.1, neither reference passes
any target cycle/frequency check; hottest targets reach 483.34–485.43 K. Some
smaller-f cases simply quench the oscillations, which is not amplitude agreement.
These exclusions are calibration/operating screens, not physical impossibility
proofs. The fixed cold background and inherited thermal bounds are also conditional.

The original fits/replays are preserved in local bundle
`20261027_175518_experimental-filament-and-cold-surroundings-pres_20c97c`
(numerical code `93ffaa7`). Additional f<1 cases use
`20261027_183751_experimental-filament-and-cold-surroundings-pres_1d1196`
(verification code `2dc3e8d`) with the completed searches copied only after
checking identical numerical modules, measured inputs and search configuration.

The [reviewed evidence and offline comparison](../runs/published/20261027_184359_reviewed-filament-and-cold-surroundings-presenta_a6fe81/START_HERE.html)
preserve exact numerical tables, acquisition-grid traces, inputs, parameters,
figures, source and provenance. The compact review omits the large native arrays
and full optimizer JSONL histories; they remain in the local numerical bundles
and can be regenerated with the recipe. Its named `parameter_comparison.csv`
also defines the order of the eleven coordinates in `candidates.json`.

Verification: **216 tests pass**, repository validation passes, the baseline
matches six pre-change scalar/batched finite-C/C=0 reference traces bit for bit,
and an independent analytic RC control verifies convergent local heat allocation.
All 56 native SI histories pass branch-current and branch-power conservation
checks. The original presentation frequency/Vpp numbers are reproduced at f=1.
Frequency estimates on non-target controls are raw feature diagnostics, not
independent claims that those noisy records are sustained oscillators.

A useful follow-up would constrain geometry, contact/readout contributions and
heated-volume thermal parameters, and then test coupled thermal-volume seeds.
This first search does not establish an identifiable f, an exhaustive absence
of a matching parameter set, or that the professor's geometric explanation is wrong.
