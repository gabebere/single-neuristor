# Alternating amplitudes recovered in a historical simulation

Reviewed 25 October 2026. The clearest recovered example is the **relaxed**
simulation in `20260907_125315_physics-anchored-specimen-waveform-inference_717797`, source file **400mv0_converted.csv**. It is in
`fitted_traces.csv`, `predicted_voltage_mV`, not the measured-voltage column.
Source labels 350 and 450 mV provide neighboring examples. The source label is
not imposed current: the constant-current control uses 303.391147209 µA,
the median archived baseline-corrected current over 150–250 ns.

## What was observed

In the archived 50–280 ns window, 8 complete primary cycles
alternate between approximately **238.92 and
352.98 mV Vpp**. Adjacent amplitudes have correlation
-0.99939; amplitudes two cycles apart have correlation
0.99924. The main visual difference is alternating deep/shallow
troughs, not a factor-of-two peak height. Median adjacent-primary-peak spacing is
24.50 ns, and two-peak spacing is
49.00 ns. A short alternating pattern therefore has
an individual-peak rate near 40.82 MHz and a two-cycle
repeat rate near 20.41 MHz. These are descriptive
rates, not evidence for two independent oscillators.

The corresponding measured output does not show a comparable two-level pattern
under the same primary-cycle analysis. Its parity separation is only
4.64% versus 38.54%
in this archived simulated record. This documents a simulation phenomenon;
it does not establish experimental period doubling or device-terminal calibration.

## Replay and numerical limitations

No optimization was repeated. The exact eight-parameter historical vector was
replayed using the authoritative float32-faithful model with the original
measured input. At 0.025 ns, sampled voltages reproduce the archived data to
1.52e-11 mV maximum difference.
Primary-cycle alternation remains in the 50–280 ns window at 0.0125 and
0.00625 ns. Raw native samples are preserved; only display plotting is thinned.

A constant-current control removes the measured input's fluctuations and
extends to 2 µs. Its final 1–2 µs alternates regularly at 0.025 and 0.0125 ns,
but develops irregular primary-cycle amplitudes at 0.00625 ns. Consequently,
**a stable long-time period-two attractor and a period-doubling bifurcation are
not established**. Qualitative recovery of the archived short-window pattern
does not imply pointwise waveform convergence, a converged steady state, or chaos.
No physical solver, memory rule or tolerance was altered to preserve the effect.

| input_mode       |   dt_ns |   cycle_count |   low_phase_median_vpp_mV |   high_phase_median_vpp_mV |   amplitude_lag1 |   amplitude_lag2 | strong_alternation   |
|:-----------------|--------:|--------------:|--------------------------:|---------------------------:|-----------------:|-----------------:|:---------------------|
| measured_current | 0.025   |             8 |                   244.695 |                    353.36  |        -0.999642 |         0.999739 | True                 |
| measured_current | 0.0125  |             8 |                   246.706 |                    351.063 |        -0.999034 |         0.99921  | True                 |
| measured_current | 0.00625 |             8 |                   247.511 |                    348.778 |        -0.998832 |         0.999365 | True                 |
| constant_current | 0.025   |            40 |                   244.73  |                    353.301 |        -0.999992 |         0.999992 | True                 |
| constant_current | 0.0125  |            40 |                   246.787 |                    350.906 |        -0.999999 |         0.999999 | True                 |
| constant_current | 0.00625 |            42 |                   250.081 |                    283.129 |        -0.51389  |         0.706797 | False                |

Even/odd amplitude medians in irregular runs are arbitrary parity summaries;
they should not be interpreted as two stable amplitude levels.

## Method

Locate primary voltage maxima using Savitzky–Golay smoothing over about 0.5 ns
(at least five samples; therefore 5 ns for the archived 1 ns output), prominence
max(2 mV, 10% of the window's smoothed range), and at least 2 ns separation.
Refine maxima on raw samples within ±0.5 ns. For each successive pair, take the
raw minimum between them and subtract it from the later raw maximum. Exclude
partial edge cycles. Never average neighboring peak heights: that can hide the
two-amplitude feature. Small reset shoulders are not primary cycles.

The descriptive `strong_alternation` flag requires at least eight primary
cycles, ≥10% parity-median separation, ≤5% relative median absolute deviation
within each parity, lag-1 correlation ≤−0.8 and lag-2 correlation ≥0.8.
These are explicit engineering criteria, not a significance test or a
bifurcation detector. `peak_detection_sensitivity.csv` varies prominence
5–20% and smoothing 0–1 ns. Synthetic uniform and drifting amplitudes are
negative controls; known alternating pulses are a positive control.
The exploratory 3%-prominence search is preserved only as discovery provenance;
its small-shoulder counts are not the documented primary-cycle measurements.

## Fixed parameters and scientific scope

C=6.835494476 pF, Cth=0.04421419302 pJ/K, Se=0.003712762043 mW/K,
T0=Tinit=314.363069806 K, Tc=333.338417242 K, w=7.041456088 K,
beta=0.3144377018 /K and gamma=0.1569637099. The remaining resistance scale and
activation parameters come from the preserved static preset. Initial V=0 and
the initial branch is insulating. The source uses the original 100425 specimen,
not the later M2 specimen. C and gamma lay outside that historical run's
configured constrained intervals; those intervals are not universal physical
laws. The old 22/22 classification claim is not adopted as quantitative
waveform agreement. No new specimen calibration is adopted here.

## Reproduction and checks

`neuristor simulate current --config reference_recipe.toml` reproduces the
finest constant-current control using the preserved local static preset.
The repository recipe is `experiments/current/alternating_amplitude_reference_20261025.toml`.
The exact CLI run, 18 full-resolution NPZ replays, original input files,
individual-cycle tables, parameter source, analysis/replay scripts and package
source snapshot are included. The replay script records the exact historical
bundle location and uses its archived numerical current samples, without refitting.
200 repository tests and repository validation pass. Synthetic signal checks,
artifact paths, input hashes and figures are separately checked. Browser
interaction is not claimed. Local archival publication does not upload to GitHub.

Next step, if pursued: test stronger timestep refinement and time horizon at
this fixed vector, then a controlled parameter sweep that distinguishes a
persistent two-cycle solution from transient/intermittent or numerical behavior.
