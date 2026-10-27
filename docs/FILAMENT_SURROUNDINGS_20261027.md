# Experimental hot filament and cold surroundings

Status: implemented protocol; scientific trial pending. This is an optional
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

## Evidence and physical motivation

- [Presentation comparison](LOSS_COMPARISON_20261020.md).
- [Gildor et al., DOI 10.1002/aelm.70604](https://doi.org/10.1002/aelm.70604)
  distinguishes uniform heating and filamentary switching, and discusses cooler
  surrounding domains and filament widening. The normalized two-region model
  above is our proposed diagnostic approximation, not a model validated by that paper.

## Trial results

Pending the full declared recipe; implementation checks are not scientific fit results.
