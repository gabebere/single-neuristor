# Adaptive oscillation-domain search

Input: 500 µA constant ideal current. Ambient 298 K,
initial temperature 298 K, branch insulator.
Resistance domain: [289.8545, 370.1625] K. Static R(T) and gamma are fixed.

## Why the three-parameter domain need not close

The deterministic equations are C dV/dt = I - V/R(T,H) and
Cth dT/dt = V²/R(T,H) - Se(T-T0). Hysteresis depends on the temperature path.
Replacing both C and Cth with s times their values gives the same state path at
time t/s. Thus Vpp and temperature extrema stay fixed and frequency divides by s.
We search C/Cth (pF K/pJ) and Se (mW/K) at reference Cth
0.072 pJ/K. The common scale direction is open
by model symmetry, not a missing closed upper surface. This reduction applies
only to deterministic constant current and the declared initial state/resistance law.
The scaled 3D samples are predictions from this identity, not extra simulations.
Their display range and the 110 MHz operational
selection never enter the oscillation-boundary classification.

## Two stages

The coarse log grid expands by its original log spacing where oscillators or
unresolved dynamics touch an outer face. Old points are cached. Hard axis limits,
expansion limits and point budgets remain explicit, never physical boundaries.
The fine grid covers the active coarse envelope plus neighboring cells. Every
fine edge is inspected again, and extends outward if finer integration reveals
oscillatory or unresolved points there. The envelope is not frozen to the coarse
pass's potentially incomplete classification. Every
fine point, including the nonoscillating side, is rerun at
[0.025, 0.0125, 0.00625] ns. Only matching
classifications and <= 5% frequency/Vpp
changes qualify. Disagreements get up to 2
additional step halvings, comparing the last three steps. Each fine row records
its actual verification_steps_ns. Neighboring verified oscillators/equilibria define log-space
brackets, then midpoint refinement narrows them. Censored midpoints stop a bracket.

Each normalized replay starts from the same state. The horizon starts at
300 ns, extends up to 1200 ns,
and requires agreement at two horizons and adjacent late windows. Slow or decaying
transients remain unresolved. Nonoscillation requires small voltage/temperature
spans and small current/heat-balance residuals, not merely absent peaks.
Thresholds: voltage span <= 0.5 mV,
temperature span <= 0.02 K,
relative balance residuals <= 0.001.
These are approximate equilibrium criteria, not rigorous asymptotic stability.

## Results and scope

Coarse points: 42. Fine points: 211.
Fine classifications: {'nonoscillatory': 100, 'oscillatory': 57, 'time_unresolved': 35, 'calibration_censored': 13, 'numerical_unresolved': 6}.
Reused point evaluations: 253. New integration/horizon checks:
0. Reused records have their original tables and frozen
source retained separately, with byte hashes and numerical compatibility checks.
Transition brackets: 18, with 15
meeting the requested log-width tolerance. Budget exhausted: False.
Inspect metrics.json for every outer-face status, and boundaries.csv for both sides.

No complete global physical boundary is claimed. Finite grids can miss narrow
or disconnected islands, calibration/proximity exclusions censor a domain rather
than prove nonoscillation, and a finite horizon can remain inconclusive.
All limits and rejected points remain in the numerical tables and viewer.
No experimental calibration or parameter uncertainty is inferred.
