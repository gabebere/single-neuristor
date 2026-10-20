# Adaptive oscillation-boundary exploration

Run the reusable recipe:

```bash
neuristor analyze boundary-maps --config experiments/current/adaptive_boundary_maps.toml
```

The command first explores a coarse grid, expanding where oscillators or
unresolved dynamics touch an outer face. It then samples a finer grid around the
active coarse envelope, including neighboring cells, and narrows transition
brackets. The physics and float32 hysteresis remain unchanged. Historical
fixed-box maps are preserved, not reinterpreted as complete physical domains.

## A scale direction that cannot have a finite boundary

For deterministic constant current,

\[
C\dot V=I-V/R(T,H),\qquad
C_{\rm th}\dot T=V^2/R(T,H)-S_e(T-T_0).
\]

At the same initial state, replacing \((C,C_{\rm th})\) by
\((sC,sC_{\rm th})\) gives the same path at time \(t/s\). The path-dependent
hysteresis has no independent clock. Thus temperature extrema and voltage Vpp
stay fixed, while period multiplies by \(s\) and frequency divides by \(s\).
An oscillator remains an oscillator along this common scale ray in this model.
This excludes thermal noise, finite pulses and rate-dependent model extensions.

Consequently, finding a closed domain in all three positive parameters is not
generally possible. The search explores the shape in \(C/C_{\rm th}\) and \(S_e\)
at a declared reference Cth. The ratio is reported in pF K/pJ, numerically
`C_pF / C_th_pJ_per_K`. The 3D reconstruction samples arbitrary display Cth
values using the scale identity. Those points are explicitly labeled scaled
predictions, not independent integrations or measured parameters.

To extend or densify a completed exploration without repeating unchanged point
integrations, use `--reuse-exploration COMPLETED_BOUNDARY_BUNDLE`. The command
checks resolved SI parameters, input current, numerical classification settings,
solver/extrema bytes and the frozen evaluator/classifier syntax tree. Geometry,
display ranges and point budgets may change. Numerical controls may not. Original
tables and frozen source are copied separately, with hashes and provenance.
The new run reports reused point evaluations and new integration attempts.

## Coarse exploration and fine verification

- Initial axes, hard safety limits, seeds, expansion count and point budgets
  are declared in `[boundary]`. Seed coordinates keep a known narrow band in
  the search without asserting that all possible islands have been discovered.
- Expansion adds outer shells at the original log spacing, preserving/caching
  evaluated coordinates. It inspects both ends of both normalized axes.
- Each replay starts from the same full initial state. Observation time doubles
  up to its cap. Two horizons must agree, with sustained cycles in two adjacent
  late windows and stable frequency/Vpp. Both positive and negative classifications
  require this time-extension check.
  Cycle detection reuses the existing extrema method: 0.5 ns smoothing for
  locating extrema at the reference scale, a minimum 2 mV prominence,
  period coefficient of variation <=0.2 and half-window amplitude retention
  0.75–1.333. Near-onset oscillations below the detection/residual thresholds
  can remain unresolved or qualify as approximate equilibria. Brackets therefore
  include declared numerical resolution, not an exact mathematical onset.
- Absence of peaks is insufficient for nonoscillation. Both late windows must
  have small voltage/temperature spans and small relative current/thermal balance
  residuals. These define an approximate equilibrium, not a stability theorem.
- Every fine point is simulated at the declared fine timestep and **two smaller
  steps**, including nonoscillating neighbors. Classification changes or frequency/
  Vpp changes above the declared tolerance remain numerical uncertainty.
  Disagreements can receive additional halvings using
  `maximum_timestep_refinements`. Only agreement across the last three steps
  can resolve them. Actual steps are stored for each fine point.
- Fine outer faces are checked again. If finer integration finds oscillatory or
  unresolved behavior on an edge that the coarse pass missed, the fine grid also
  expands in shells. It does not freeze an incorrect coarse envelope.
- Neighboring verified oscillatory/equilibrium points form transition brackets.
  Logarithmic midpoint sampling narrows them to a configured width or iteration
  budget. A censored/unresolved midpoint stops certification of that bracket.

Coarse spatial sampling is cheap, but its integration step must still resolve the
physics. Increasing the timestep too much can erase narrow oscillatory regions.
The default uses 0.05 ns for coarse screening, 0.025 ns for fine sampling, and
0.0125/0.00625 ns for verification. The `[boundary]` time settings control this
workflow's integration and adaptive horizons; the general `[time]` table is the
base-model configuration recorded for provenance.

## Different kinds of limits

| Label | Meaning |
|---|---|
| `oscillatory` | Persistent finite-window cycles, with fine points checked at three steps |
| `nonoscillatory` | Approximate settled equilibrium under the declared criteria |
| `calibration_censored` | Full temperature history leaves the resistance calibration domain |
| `proximity_censored` | Sampled effective-temperature slope becomes negative |
| `time_unresolved` | Observation cap reached without a stable classification |
| `numerical_unresolved` | Nonfinite data or disagreement under timestep refinement |
| `step_budget` | Required integration exceeds the explicit sample-count budget |

Outer faces report `oscillatory_open`, `unresolved_open`, `censored_open` or
`sampled_nonoscillatory`. The last means sampled absence under these conditions,
not proof that a disconnected island cannot exist farther away. The common
capacitance scale always reports `open_by_model_symmetry`.

The frequency ceiling is a **separate operating selection** on reconstructed
points. It never removes an oscillator from the boundary classification.
The example uses the actual measured static temperature interval,
289.8545–370.1625 K, rather than the old padded interval. Its excluded edge is
a calibration limit, not evidence of a physical disappearance of oscillation.

## Output bundle

`parameter_maps.html` is an offline viewer containing both sampled classification
maps and the reconstructed three-parameter cloud. It shows excluded/unknown
points rather than painting them as empty physical space. No interpolation
creates an apparent enclosing surface. `boundary_maps.png` is a static overview.

Numerical evidence includes:

- `coarse_grid.csv`, `fine_grid.csv`: every point and final classification.
- `verification_checkpoint.csv`: every integration at every step and horizon.
- `boundaries.csv`: both sides, log-width, refinement status and tolerance result.
- `scaled_oscillators.csv`: 3D scale predictions, equivalent horizons and the
  optional operating-frequency selection.
- `metrics.json`: all outer-face statuses, counts, budgets and open-scale scope.
- `physics_parameters.json`, `resolved_config.json`, frozen package source and
  hashes, `report.md`, and the standard `run.json` manifest.

Finite grids can miss small/disconnected domains. Brackets apply to the fixed
current, ambient, initial state, gamma and resistance calibration. A different
initial branch can have a different basin of attraction. This workflow provides
inspectable numerical evidence and honest open limits, not a complete global
bifurcation proof or an experimentally calibrated specimen parameter set.
