# What can the full exploration tell us?

The original viewer covers 23³ = 12,167 combinations of C, Cth and Se. Gamma is
fixed at 0.956269682, current at 500 µA, T0 and initial temperature at 298 K,
and the specimen major-loop resistance fit is fixed. This is a three-parameter
conditional exploration, not a four-parameter specimen calibration. Source
screens used 284.8545–375.1625 K (a padded range) and a 110 MHz operating ceiling.
The stricter later boundary study must be consulted for boundary interpretation.

1,258 points passed the old map screens. All have GIFs. Finer GIF replays retain
1,256 eligible, converged points; 1,174 pass the visual-resolution checks.
The original map membership remains unchanged. Statistics use finer GIF features
on the 1,256 retained replays. Illustrations additionally require visual resolution.

## Tendencies with the other two parameters held fixed

Increasing C slows oscillations and usually raises Vpp or approaches an amplitude
plateau. Increasing Cth also slows oscillations but usually reduces Vpp or leaves
it nearly unchanged. Increasing Se mostly lowers frequency and raises Vpp in
this retained region, with exceptions. These are conditional local tendencies,
not universal monotonic laws or inferred sample parameter values.

matched_neighbors.csv compares adjacent positions of the original axis only,
holding both other parameters exactly fixed. Both endpoints must pass the finer
replay checks; missing cells are never bridged. The summary gives 1% and 5%
change bins. The 1% bins describe saved values and are not precision claims:
the original convergence threshold is 5%. Counts overlap and are correlated;
they are not independent statistical trials or confidence intervals.

The displayed one-dimensional examples maximize the number of visually resolved
points at fixed other parameters (ascending-coordinate tie break). Low, middle
and high examples use their current rank. They were not selected by error size.

## What helps separate the parameters?

C and Cth both tune frequency, so frequency alone cannot distinguish them well.
Their different amplitude tendencies offer additional information. More strongly,
at deterministic constant current multiplying BOTH capacitances by s preserves
the trajectory path and Vpp while replacing f by f/s. The saved ratio-40 ray
illustrates this identity with independent pre-existing integrations. Small
deviations reflect finite-step/window extraction; no new integration was run.
Thus amplitude/loop shape constrain the ratio and thermal conductance, while
frequency constrains the common time scale, conditional on fixed gamma and T0.
This does not establish unique identifiability from the measured data.

## Limits and next useful comparison

Blank cells mean excluded or unresolved under the source screens, not proven
nonoscillation. Finite map edges are not physical bounds. The later adaptive
boundary evidence (bundle 504198) retains unresolved/calibration-censored regions
and establishes the open common-capacitance direction analytically.

The retained map has hundreds of mV of Vpp at these conditions. It cannot be
used directly as agreement with the measured sweep at adopted T0=314.4 K.
The useful next experiment is a matched-condition ratio/Se exploration at that
ambient, with gamma explicitly fixed or swept, followed by a common-scale
frequency adjustment. That is a proposal, not a completed new simulation.

The original HTML and GIF collection are preserved. This bundle reads saved
measurements only, checks byte identity with the fixed-scale viewer, and records
input hashes and its exact analysis source. It contains no reference PDFs.

## Metrics

```json
{
  "grid_points": 12167,
  "original_retained": 1258,
  "finer_replays_retained": 1256,
  "visually_resolved": 1174,
  "common_scale_points": 16,
  "common_scale_max_frequency_discrepancy_fraction": 0.001973466444115113,
  "common_scale_amplitude_span_fraction": 0.007419035615312734,
  "resimulated": false
}
```
