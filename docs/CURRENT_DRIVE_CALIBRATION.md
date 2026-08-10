# Current-drive model audit and calibration notes

## What Yuanhang's original implementation actually simulates

The vendored upstream model is a voltage-source circuit, not an ideal-current circuit. A source voltage drives a VO2 device through a load resistor, with a capacitance from the VO2 node to ground:

\[
C\frac{dV}{dt}=\frac{V_{in}-V}{R_{load}}-\frac{V}{R_{VO2}(T,\mathcal H)}.
\]

Its single-device thermal equation is

\[
C_{th}\frac{dT}{dt}=\frac{V^2}{R_{VO2}}-S_e(T-T_0).
\]

The upstream numerical units are ns, kOhm, pF, mW, and mW ns/K. The corresponding baseline values in `references/yuanhangzhang98-collective_dynamics_neuristor-217d4f0/model.py` are:

| Parameter | Yuanhang value | SI value |
|---|---:|---:|
| Load resistance | 12 kOhm in `main.py` | 12 kOhm |
| Electrical capacitance | 145.346 pF | 145.346 pF |
| Thermal capacitance | 49.6278 mW ns/K | 49.6278 pJ/K |
| Total thermal conductance | 0.205587 mW/K | 0.205587 mW/K |
| Ambient temperature | 325 K | 325 K |
| Metallic resistance | 1286.317 Ohm | 1286.317 Ohm |

For an array, the reference code partitions thermal conductance into an environmental term and a nearest-neighbor term. For a single device there is no measurable neighbor coupling. Use `couple_factor=0` and interpret `S_e` as the device-to-substrate/environment conductance.

## The derived ideal-current model in this repository

The lab-oriented current-source extension uses

\[
C\frac{dV}{dt}=I_{in}(t)-\frac{V}{R_{VO2}(T,\mathcal H)},
\]

with the same thermal equation. A resistor in series with a mathematically ideal current source does not change the injected node current. If the real source has finite compliance, finite output resistance, cables, contacts, or a measured voltage that includes other elements, those must be modeled explicitly; they are not represented by the ideal-current equation.

## Why the simulated voltage falls close to zero

During a current pulse, the electrical fixed point at a given resistance is

\[
V_{ss}=I R(T).
\]

After switching, its lower bound is approximately `I*Rm`. Electrical capacitance controls how quickly voltage approaches this value, not the value itself.

At 400 uA:

| Resistance choice | Rm | Ideal-current floor `I*Rm` |
|---|---:|---:|
| Yuanhang reference | 1286.3 Ohm | 514.5 mV |
| Fitted specimen preset | 18.3 Ohm | 7.3 mV |
| Prior nonzero-valley trial | 250 Ohm | 100 mV lower bound; about 159 mV after settling |

Therefore, decreasing `C` cannot prevent the voltage valley. It makes the drop faster. The main causes of the mismatch are the fitted specimen's very low metallic resistance and uncertainty about which physical voltage the oscilloscope includes.

The digitized lab family supports this diagnosis. Above 350 uA, plateau `V/I` ranges from about 135 to 472 Ohm, while the fitted `Rm` is only 18.3 Ohm. Because the plateau stays near 190 mV over a broad current range, one fixed added series resistance is also insufficient to reproduce the whole family; the temperature-dependent state and/or nonideal source/measurement circuit matters.

### Evidence: measured plateau versus the ideal-current floor

![Measured plateau and ideal-current resistance floors](figures/current_drive/lab_estimates/voltage_floor_comparison.png)

The red line is the fitted specimen prediction `V=I*18.3 Ohm`; it stays near zero
compared with the digitized laboratory plateau. Yuanhang's much larger metallic
resistance produces a much larger floor, while a fixed 250 Ohm addition crosses the
data only locally and cannot reproduce the nearly current-independent 190 mV plateau.
None of these steady-state lines depends on electrical capacitance.

## Model and numerical corrections made in this audit

1. The current solver now advances the frozen-resistance RC equation with its exact exponential solution. This remains bounded and nonnegative when `C` is small even if `dt` is longer than `R*C`.
2. `C=0` is now a real algebraic limit, `V=I_in*R(T)`. It is no longer silently replaced by 1 pF.
3. The deterministic cooling term is also advanced by its frozen-power exponential solution. Timestep convergence is still required because resistance and hysteresis change during a step.
4. The active hysteresis implementation previously clipped every reversal fraction to `[1e-6, 1-1e-6]`. This changed a valid upstream minor-loop resistance by as much as 56.8%. It now regularizes exact endpoints only and matches the upstream float32 trace to about 0.001 Ohm. The old 1e-6 convention remains isolated in `SampleFitHysteresisArray` so saved fit metrics still replay exactly.

The low voltage predicted with the 18.3 Ohm specimen `Rm` is not itself a numerical error. The former explicit Euler update could add artifacts when `C` was reduced, but a stable integrator cannot raise the physical `I*Rm` floor.

## Cth-versus-C frequency sweep with Yuanhang values

`scripts/sweep_current_capacitances.py` evaluates a 7 by 7 grid centered on Yuanhang's `C` and `C_th`, at 300, 400, 500, 600, 700, and 800 uA. It excludes startup behavior and only labels regular, persistent late-time cycles as oscillations.

At the nominal Yuanhang capacitances:

| Current | Frequency |
|---:|---:|
| 300 uA | 0.147 MHz |
| 400 uA | 0.218 MHz |
| 500 uA | 0.274 MHz |
| 600 uA | 0.321 MHz |
| 700 uA | 0.362 MHz |
| 800 uA | 0.397 MHz |

The nominal 600 uA result changes from 0.32101 MHz at `dt=1 ns` to 0.32138 MHz at `dt=0.25 ns`. A fast corner (`C=18.17 pF`, `C_th=24.81 pJ/K`, 600 uA) changes from 1.7673 to 1.7725 MHz between `dt=0.5 ns` and `0.125 ns`.

The sweep shows three useful trends:

- Increasing `C_th` generally lowers the frequency.
- Reducing finite `C` can make cycles faster, but removing the electrical state entirely stops the Yuanhang-parameter oscillation.
- When the metallic electrical time `Rm*C` is much longer than `C_th/S_e`, most tested cells settle, but this ratio alone is not a universal stability criterion; current and the full hysteretic trajectory also matter.

![Six-current capacitance-frequency study](figures/current_drive/capacitance_study/frequency_heatmaps_all_currents.png)

The white square marks Yuanhang's nominal electrical and thermal capacitances. The
gray `C=0` column corroborates that the algebraic thermal-only limit does not sustain a
regular cycle here. Moving to smaller finite `C` can raise frequency, but many such
cells leave the calibrated `R(T)` temperature range, marked by black stars.

![Electrical-capacitance trace comparison at 600 uA](figures/current_drive/capacitance_study/capacitance_trace_comparison_600uA.png)

The trace comparison makes the distinction explicit: finite `C` changes charging and
discharging timing and therefore cycle frequency; it does not change the metallic
steady-state bound `I*Rm`. The `C=0` case removes electrical memory and settles.

Cells that drive temperature outside the 305--370 K calibrated resistance range are marked with `*`. In those cells the model clamps `R(T)` and the result is an extrapolation.

## Can the system oscillate with only thermal dynamics?

For `C=0`, voltage is algebraic and power becomes `P=I^2 R(T)`. A simple thermal-cycle estimate requires both:

\[
I > \sqrt{\frac{S_e(T_{heat}-T_0)}{R_{heat}(T_{heat})}}
\]

to heat through the upper transition, and

\[
I < \sqrt{\frac{S_e(T_{cool}-T_0)}{R_{cool}(T_{cool})}}
\]

to cool through the lower transition.

For Yuanhang's values these estimates are about 382.8 uA and 198.6 uA, respectively. Since the required lower current is greater than the allowed upper current, there is no overlap. The simulated `C=0` column likewise settles for every tested current and `C_th`. Finite electrical memory is essential for the oscillations found in this parameter set.

## Estimates from the digitized lab screenshots

These are image-based estimates, not substitutes for raw oscilloscope data.

![Lab-derived parameter constraints](figures/current_drive/lab_estimates/lab_parameter_estimates.png)

The left panel supports the approximately 22.7 pF electrical-capacitance estimate.
The center panel shows why ambient temperature and environmental conductance cannot be
fit independently from switching power alone. The right panel shows that the measured
high-current plateau corresponds to a current-dependent effective `V/I`, rather than
one constant series resistance.

### Electrical capacitance

Before switching and near `V=0`, `dV/dt` is approximately `I/C`, so

\[
C\approx\frac{I}{dV/dt}.
\]

Four clean pre-switch frames give 20.9--25.0 pF, with a median of 22.7 pF. This is consistent with the earlier joint image fit (`C=20.05 pF`) and is much smaller than Yuanhang's 145.35 pF.

### Switching current and environmental conductance

The digitized family brackets switching onset between 194.6 and 233.4 uA. Using the last pre-switch point (`V=315.2 mV`, `P=61.35 uW`) and the fitted specimen heating midpoint `T_switch=336.96 K`:

| Assumed T0 | Estimated Se |
|---:|---:|
| 298 K | 0.00157 mW/K |
| 325 K | 0.00513 mW/K |
| 330 K | 0.00881 mW/K |
| 333 K | 0.01549 mW/K |

This strong dependence is why ambient temperature must be measured rather than absorbed freely into `S_e`.

### Thermal capacitance

Voltage screenshots do not independently identify `C_th`. First measure the thermal recovery time from a post-pulse or pump-probe trace, then calculate

\[
C_{th}=S_e\tau_{th}.
\]

For example, if `T0=325 K`, the onset estimate gives `S_e=0.00513 mW/K`; thermal recovery times of 20, 50, and 100 ns imply `C_th=0.103`, `0.256`, and `0.513 pJ/K`. The earlier correlated image fit used `C_th=6.55 pJ/K` together with the much larger `S_e=0.129 mW/K`, giving a similar order-50-ns thermal time. This illustrates that waveform fitting often identifies the ratio `C_th/S_e` more strongly than either parameter alone.

## Recommended calibration order

1. Record the exact stage/substrate ambient temperature and clarify whether measured voltage is across VO2 only or includes contacts, leads, sense resistance, or source compliance.
2. From a low-current, nonswitching edge, fit `tau_el` and use `C=tau_el/R` (or the early-slope approximation above).
3. From a post-pulse or pump-probe recovery, fit `tau_th`.
4. Use power immediately before switching and the measured `T0` to estimate `S_e=P_switch/(T_switch-T0)`.
5. Calculate `C_th=S_e*tau_th`.
6. Fit `Rm` and any explicit contact/source model to the measured switched-state voltage floor. Do not use `C` for this purpose.
7. Only after these steps, refine current amplitude, transition temperature, and hysteresis width against the full waveform and repeat at smaller `dt`.

## Reproduction commands

```bash
python scripts/sweep_current_capacitances.py
python scripts/estimate_lab_current_parameters.py
python scripts/audit_model_fidelity.py
python -m unittest discover -s tests -v
```

Generated outputs are written to `outputs/current_capacitance_study/` and `outputs/lab_parameter_estimates/`.
Pass `--output-dir docs/figures/current_drive/...` as shown in the root README to
regenerate the committed GitHub evidence package.
