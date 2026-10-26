# Parameter perturbations, timestep sensitivity and gradient-based trials

Physical equations, measured input waveforms, feature extraction, loss weights,
bounds and eligibility screens are unchanged. Four normalized log coordinates
are C, Se, Cth/Se and gamma. A normalized increment is a fraction of that
coordinate's log-bound range; it is not an increment in SI units. In particular,
changing Se at fixed Cth/Se changes Cth too. The six R(T) values and T0 stay fixed.

Each center is tested at positive/negative coordinate perturbations, multiple
step sizes, and three timesteps. Forward/backward/centered slopes of the finite
optimizer score are separate from derivatives of an eligible all-target loss.
A missing frequency makes the latter undefined. A score plateau can mean peak
quantization or the rejection tie-breaker cap; it is not proof of a physical
stationary point. Slope disagreement alone is not a proof of nondifferentiability.

L-BFGS-B uses the identical finite optimizer score at two declared difference
steps. If its starting point is infeasible, it is an exploratory test of score
refinement, not a gradient calculation on a defined all-target scientific loss.
The score ranks failed-target count before its bounded partial error term.
Optimizer convergence messages are recorded and do not constitute a fit.
The best sampled candidate per objective is replayed on all currents at three
steps, alongside the original center. Neither gradient trial is a global search.

Frequency features are extracted on the original acquisition sample grid after
interpolating simulated voltage. The smoothing rule targets 0.5 ns but has a
minimum of five samples; the actual 1 ns data grid therefore uses five samples.
Peak times, median periods and rejection decisions can change discretely.
No smoother proxy loss or changed physical model is introduced in this audit.

## Completion
{
  "unique_evaluations": 230,
  "profile_points": 150,
  "stencils": 72,
  "scientific_derivatives_defined": 0,
  "gradient_trials": 4,
  "verified_batches": 12,
  "elapsed_s": 1194.014824999962,
  "normalized_coordinates": [
    "C_pF",
    "S_e_mW_per_K",
    "tau_th_ns",
    "gamma"
  ],
  "dt_ns": [
    0.025,
    0.0125,
    0.00625
  ]
}

See gradient_summary.csv, finite_differences.csv and summary.csv for outcomes.
