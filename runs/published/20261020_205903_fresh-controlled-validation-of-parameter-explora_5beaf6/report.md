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
