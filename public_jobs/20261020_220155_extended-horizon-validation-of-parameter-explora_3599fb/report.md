# Fresh tests of the parameter-map trends

These are new independent integrations, not saved-grid reanalysis or optimization.
The versioned recipe and planned_cases.csv specify anchors, scenarios and all
factor changes before simulation. No physical equation or hysteresis rule changed.

Common-capacitance factors are tested with fixed physical timestep and with a
timestep multiplied by the same factor. Observation horizons are multiplied by
that factor in both protocols, preserving equal dimensionless startup/settling
time. Original-step comparisons and two halvings are retained. Unresolved
timestep cases may receive the declared additional halving. Every case starts
with V=0, T=T0 and insulating hysteresis memory. Both late windows and successive
observation horizons must agree, using the existing boundary classifier.

The ratio scaling law follows by replacing t with t/s in BOTH deterministic
equations. Its assumptions are constant ideal current, fixed Se/gamma/R(T)/ambient,
identical initial state, no independent memory clock and zero noise. It does not
apply unchanged to a fixed-duration measured pulse. Scaled dt tests discrete
solver covariance; fixed dt tests numerical convergence independently. The
feature extractor has fixed physical 0.5 ns smoothing, so extracted Vpp/f can
deviate even if state paths scale. Sampled path errors use the entire history,
without phase alignment, on the declared common normalized-time grid. They are
finite-resolution comparisons, not maximum-error bounds between sample times.

All local C/Cth/Se changes keep the other parameters fixed. Separate gamma
values, a different current and the adopted 314.4 K ambient probe transfer of
the old map tendencies. Censored, nonoscillatory and unresolved cases remain
in all tables and do not become zero-frequency points. No 110 MHz ceiling enters
classification. Temperature must stay inside the measured resistance interval;
this is stricter than the historical map's padded interval.

Direction labels require two verified oscillatory endpoints. A change must exceed
both the declared minimum direction threshold and twice the larger endpoint's
last-three-step change. That envelope is a numerical-resolution diagnostic,
not a statistical confidence interval. Repeated comparisons and centers are
correlated. Aggregate results must retain scenario and endpoint denominators.
This bounded design does not prove universal monotonicity, global stability,
complete boundaries, unique identifiability or specimen calibration.

See the numerical tables and figures for completed results. The research note
documents the reviewed conclusions and exceptions after all checks finish.

## Completed counts

108 logical cases; 610 distinct integrations: 466 reused and 144 new. Repeated centers are not independent trials.

| Classification | Logical cases |
|---|---:|
| oscillatory | 59 |
| time_unresolved | 36 |
| nonoscillatory | 11 |
| calibration_censored | 2 |

## Final verified common-scale comparisons

12/12 pairs have verified oscillatory endpoints; 12/12 are within the declared feature tolerance.

| Protocol | Verified pairs | Maximum frequency error | Maximum Vpp error |
|---|---:|---:|---:|
| fixed_dt | 6/6 | 0.3443% | 0.8608% |
| scaled_dt | 6/6 | 0% | 0.8706% |

## Independent frequency measurement

independent_frequency_checks.csv uses rising mid-level crossings in the last half of the saved sampled path, with no smoothing or peak detector. Three crossings are required. This diagnostic does not replace persistence or physical screens; path sampling and linear crossing interpolation limit its precision.


## Controlled direction counts

Counts include all prescribed adjacent pairs; unavailable pairs stay unresolved.

| Scenario | Axis | Feature | Decrease | Increase | Small/unresolved change | Unusable endpoints |
|---|---|---|---:|---:|---:|---:|
| lower_current | C_pF | frequency_MHz | 0 | 0 | 0 | 6 |
| lower_current | C_pF | extrema_vpp_mV | 0 | 0 | 0 | 6 |
| lower_current | C_th_pJ_per_K | frequency_MHz | 0 | 0 | 0 | 6 |
| lower_current | C_th_pJ_per_K | extrema_vpp_mV | 0 | 0 | 0 | 6 |
| lower_current | S_e_mW_per_K | frequency_MHz | 0 | 0 | 0 | 6 |
| lower_current | S_e_mW_per_K | extrema_vpp_mV | 0 | 0 | 0 | 6 |
| original_map | C_pF | frequency_MHz | 5 | 0 | 0 | 1 |
| original_map | C_pF | extrema_vpp_mV | 0 | 2 | 3 | 1 |
| original_map | C_th_pJ_per_K | frequency_MHz | 5 | 0 | 0 | 1 |
| original_map | C_th_pJ_per_K | extrema_vpp_mV | 2 | 0 | 3 | 1 |
| original_map | S_e_mW_per_K | frequency_MHz | 5 | 0 | 0 | 1 |
| original_map | S_e_mW_per_K | extrema_vpp_mV | 0 | 5 | 0 | 1 |
| original_map | gamma | frequency_MHz | 1 | 0 | 5 | 0 |
| original_map | gamma | extrema_vpp_mV | 0 | 0 | 6 | 0 |
| paper_ambient | C_pF | frequency_MHz | 0 | 0 | 0 | 6 |
| paper_ambient | C_pF | extrema_vpp_mV | 0 | 0 | 0 | 6 |
| paper_ambient | C_th_pJ_per_K | frequency_MHz | 0 | 0 | 0 | 6 |
| paper_ambient | C_th_pJ_per_K | extrema_vpp_mV | 0 | 0 | 0 | 6 |
| paper_ambient | S_e_mW_per_K | frequency_MHz | 0 | 0 | 0 | 6 |
| paper_ambient | S_e_mW_per_K | extrema_vpp_mV | 0 | 0 | 0 | 6 |
