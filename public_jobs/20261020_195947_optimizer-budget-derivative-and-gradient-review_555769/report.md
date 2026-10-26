# Do a larger search or gradient methods improve the fit?

This review introduces no new simulations or altered objectives. It joins saved
all-current checks for the earlier centers, expanded DE/Powell searches, and
L-BFGS-B score trials. The sources retain complete histories and termination
reasons. Original data, fixed static fit/T0, bounds and relative losses match.

frequency_derivatives.csv retains per-input df_i/du_j wherever three neighboring
frequencies are defined. That can hold even if another input makes the full
scientific loss undefined. Missing frequencies are not zero. A zero finite slope
can be a sampling plateau; forward/backward disagreement can include curvature.
The figure shows the middle input by measured-current rank and capacitance
perturbations chosen before seeing residuals. No smoother frequency estimator
was introduced.

method_summary.csv uses the same complete-cycle definition for every method.
Full accepted losses remain null on failed eligibility; partial frequency
errors carry their denominators. A lower partial error is not an all-target fit.
The existing outside-domain flag aggregates each complete 22-record replay.
Gradient methods were warm-started using the profile experiment, while expanded
searches received larger populations and more starts. These are practical tests,
not a compute-matched benchmark proving one algorithm universally superior.

Parameter and numerical convergence checks remain essential. Optimizer success
or loss reduction alone does not establish specimen calibration or prove that
no better model parameter set exists.

## Review metrics
{
  "method_candidates": 6,
  "frequency_stencils": 792,
  "frequency_stencils_defined": 644,
  "frequency_stencil_plateaus": 327,
  "defined_disagreement_over_half": 205,
  "eligible_finest_candidates": 0,
  "dt_ns": [
    0.00625,
    0.0125,
    0.025
  ]
}
