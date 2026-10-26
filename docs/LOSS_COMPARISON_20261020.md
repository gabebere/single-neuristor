# Controlled frequency versus frequency-and-amplitude comparison

## Question and objectives

Can changing only the optimized voltage features improve one shared model across
11 measured oscillatory current inputs? The model is unchanged. Let
`V_i(t; theta) = S[I_i_measured(t); theta]`, with the four unknowns
`theta = (gamma, Se, C, Cth)`. The six fitted major-loop constants and adopted
`T0 = 314.4 K` stay fixed. Noise strength is zero.

The two objectives are:

- Frequency only: `Lf = mean[(f_sim/f_measured - 1)^2]`.
- Frequency and amplitude: `LfA = Lf + mean[(Vpp_sim/Vpp_measured - 1)^2]`.

Vpp is peak-to-peak voltage, not half amplitude. Frequency is 1000 divided by
the median complete-cycle period in ns. Vpp is the median complete-cycle height
(the mean of two bounding maxima minus the intervening minimum). These features
come from 150–250 ns. Smoothing targets 0.5 ns but has a minimum of five samples; the 1 ns
record grid therefore uses five samples. It locates extrema only; reported
voltage values use the original samples. Finite acquisition resolution limits
frequency precision. No DC, phase, waveform, logarithmic or static-data penalty
is added to either scientific objective.

## Matched protocol

- Targets are selected by baseline-corrected measured current over 50–250 ns,
  between 225 and 610 µA, giving 11 records. Filename mV numbers are identifiers,
  not physical currents. The complete measured current history drives each replay.
- The remaining 11 records are controls, not independent blind validation.
- Both searches use the same seed, initial vectors, mixed log/linear coordinates,
  differential-evolution/Powell procedure, two fixed-static restarts, population
  16, six DE generations, at most 16 Powell calls per restart, and cap of 256
  unique evaluations. Repeated vectors are cached, so actual unique counts can
  be lower. Search time is capped at 900 s, with 1800 s including verification.
- Gamma: 0.60–0.98; C: 0.05–80 pF; Se: 0.3–30 µW/K; Cth/Se: 0.5–150 ns.
  These broad diagnostic search bounds are not measured specimen intervals.
  Cth is derived from Se times the internal thermal-time coordinate; there is no
  additional independent thermal-time parameter. The bounds imply correlated
  limits on Cth, not an independently measured Cth interval.
- Eligibility is separate: complete sustained cycles on every target, frequency
  <=110 MHz, temperatures in the configured resistance domain, and nonnegative
  sampled proximity response throughout the simulated path. Rejected candidates
  receive a finite feasibility sentinel, not a scientific loss. Missing
  frequencies and their relative errors remain undefined.
- All 22 inputs are replayed at 0.025, 0.0125 and 0.00625 ns. The temperature
  domain flag in the existing all-current export aggregates all 22 records;
  a high-current control can therefore mark the full replay outside the domain.
- The highest target, about 606 µA, has measurable frequency/Vpp but fails the
  selected retention gate in the experiment itself. Control-oscillation counts
  refer to the simulated gate and explicit control group, not an experimental
  reclassification by the noisier late-window peak detector.
- Low, middle and high examples are selected by current rank before inspecting
  errors: about 228.3, 418.9 and 606.3 µA. No phase alignment is applied.

## Reproduction

```bash
neuristor analyze fit-steady --config experiments/current/specimen_frequency_only_20261020.toml
neuristor analyze fit-steady --config experiments/current/specimen_frequency_amplitude_20261020.toml
neuristor analyze compare-steady --frequency-run FREQUENCY_RUN --combined-run COMBINED_RUN
```

The comparison bundles exact saved samples and hashes of the two parents. It
re-evaluates both scientific objectives on both candidates; accepted objective
values remain null if eligibility fails. Partial frequency diagnostics always
state their denominator. No historical bundle is edited in place.


## Completed results

Both searches completed 250 unique evaluations. Frequency-only search time was
535.63 s; combined search time was 524.91 s. All nine planned verification batches
completed for each parent (three vectors × three timesteps × 22 records).

| Objective | Passing target cycle/frequency checks | Defined frequencies | Frequency MAPE on defined records | Vpp MAPE on all 11 | Oscillating controls / 11 |
|---|---:|---:|---:|---:|---:|
| Frequency only | 10/11 | 10/11 | 14.48% | 1965.70% | 5 |
| Frequency + amplitude | 8/11 | 9/11 | 18.18% | 902.98% | 0 |

These are failed diagnostic attempts, not accepted fits or established physical
parameters. Both accepted all-target objective values remain undefined. Frequency
MAPE has different denominators, so it is not an all-target ranking. Missing-cycle
Vpp uses the existing zero-feature convention in diagnostic amplitude error; plots
and example tables show a gap/dash when no complete-cycle Vpp is available.

| Objective | C (pF) | Se (µW/K) | Cth (pJ/K) | Gamma |
|---|---:|---:|---:|---:|
| Frequency only | 5.360973 | 5.794097 | 0.0280007 | 0.707477 |
| Frequency + amplitude | 6.253977 | 4.096025 | 0.0300687 | 0.630809 |

### Three examples per objective, finest step

| Objective | Measured current (µA) | Measured / simulated f (MHz) | Measured / simulated Vpp (mV) | Pass cycle/frequency checks? |
|---|---:|---|---|---|
| Frequency only | 228.28 | 41.67 / undefined | 92.40 / undefined | No |
| Frequency only | 418.86 | 54.05 / 55.56 | 34.78 / 507.80 | Yes |
| Frequency only | 606.28 | 62.50 / 76.92 | 9.60 / 485.13 | Yes |
| Combined | 228.28 | 41.67 / undefined | 92.40 / undefined | No |
| Combined | 418.86 | 54.05 / 55.56 | 34.78 / 410.02 | Yes |
| Combined | 606.28 | 62.50 / undefined | 9.60 / undefined | No |

### Timestep and scope limits

The frequency-only candidate keeps ten target cycle classifications but its
maximum target frequency change from 0.0125 to 0.00625 ns is 5.88%; maximum Vpp
change is 8.39 mV. The combined candidate drops from nine passing targets to
eight at the finest step, with maximum defined-frequency change 16.67% and Vpp
change 218.62 mV. Neither is a fully converged quantitative fit. Both full
22-record replays include temperatures outside the configured resistance domain;
the target-only search screen did not guarantee in-domain high-current controls.

Adding amplitude error lowered the diagnostic Vpp error and removed simulated
control oscillations, but did not achieve the requested frequency/amplitude match.
This limited search does not prove the absence of a parameter vector. No new
calibration is adopted.

### Reviewed evidence

All 22 inputs under each retained candidate are available as
[44 stacked current/voltage/R(T) GIFs](LOSS_ANIMATIONS_20261021.md), with native
R/T histories and saved-voltage fidelity checks. No optimization was repeated.

- Frequency-only: `runs/published/20261020_155947_controlled-frequency-only-specimen-fit_543408`.
- Combined: `runs/published/20261020_165107_controlled-frequency-amplitude-specimen-fit_5ad999`.
- Comparison: `runs/published/20261020_170231_controlled-frequency-and-amplitude-loss-comparis_26574a`.

The earlier comparison `2feaaa` is superseded for display only: the new version
shows missing complete-cycle Vpp as gaps rather than connecting its zero sentinel.
Parent simulations, metrics and saved voltage samples are unchanged.

Checks: 180 tests passed; both exact recipes completed; real comparison command
and three-timestep replays completed. Repository validation and rendered-slide
review accompany the final presentation. No Git commit, push, or new automation
was performed.

