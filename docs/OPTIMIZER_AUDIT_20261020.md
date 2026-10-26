# Is the optimizer limiting the fit? — 20 October 2026

This follow-up keeps the four unknowns, fixed major-loop fit, adopted T0, measured
current waveforms, extrema extraction, loss weights, bounds and eligibility rules
of [the previous controlled comparison](LOSS_COMPARISON_20261020.md).
No physical equation, hysteresis convention or tolerance was changed.

## Experiments

1. Around each previous best attempt, perturb each normalized coordinate in both
   directions by 0.001, 0.005 and 0.02. Repeat at 0.025, 0.0125 and 0.00625 ns.
   Record forward, backward and centered finite differences and gate changes.
2. Increase each derivative-free search from two starts to four, with a cap of
   768 unique evaluations rather than 256. Both new searches receive both previous
   best attempts and the same older seeds. Objectives and bounds are unchanged.
3. Try bounded finite-difference L-BFGS-B at steps 0.005 and 0.02, at most 60
   objective requests and six iterations per trial. Start from an eligible profile
   point when one exists; otherwise explicitly test the finite rejection score.
   Verify the best sampled gradient result per objective on all 22 inputs at the
   same three timesteps, together with its previous baseline.

The search coordinates are normalized logarithms of C, Se, Cth/Se and gamma.
`e_j` selects one of these coordinates. An h-step is a fraction of its logarithmic
bound range, not a fixed increment in SI. Varying Se at fixed Cth/Se varies Cth too.
This is the existing invertible four-coordinate parameterization; it does not
introduce another physical unknown.

## What a derivative means here

Both previous candidates lack at least one target frequency. The full scientific
objective is undefined at those points. Its derivative cannot be recovered by
calling an optimizer. We therefore preserve two different quantities:

- Scientific loss and finite differences only where every required target is
  eligible, with undefined values retained elsewhere.
- Finite optimizer score, which first ranks the number of failed targets and then
  a bounded partial-error tie-breaker. Slopes of this score are labeled as such.

A flat score can arise from discrete peak detection or a capped tie-breaker;
it does not establish a stationary point of the physical objective. Different
forward/backward slopes can also reflect curvature, so a slope-disagreement
statistic alone is not proof of nondifferentiability. L-BFGS-B termination is
recorded separately from fit quality and eligibility.

Simulated voltages are interpolated onto the original acquisition times before
feature extraction. The records have a 1 ns grid. Smoothing has a nominal 0.5 ns
target but a minimum of five samples, so these records use five samples. This
clarifies the existing implementation; no feature or numerical result was changed.
Frequency extraction through discrete peak locations can have plateaus and jumps.

## Reproduction

```bash
neuristor analyze audit-optimizer --config experiments/current/specimen_optimizer_audit_20261020.toml
neuristor analyze fit-steady --config experiments/current/specimen_frequency_only_expanded_20261020.toml
neuristor analyze fit-steady --config experiments/current/specimen_frequency_amplitude_expanded_20261020.toml
```

All jobs write new bundles with resolved inputs, source hashes, status, numerical
tables and reports. Existing completed evidence remains immutable.

The saved-data review additionally computes per-input `df_i/du_j` whenever the
three frequencies in a stencil exist, even when a different input invalidates
the full loss. It plots direct frequency profiles for the middle measured input.
This avoids attributing every difficulty to the rejection score alone. The review
is reproducible with `neuristor analyze review-optimizer --audit-run RUN
--frequency-run RUN --combined-run RUN` after all three jobs finish.

## Completed local derivative and gradient checks

The audit produced 150 coordinate-profile points and 72 full-loss derivative
stencils. None of the 72 had a defined full scientific loss at all required
points: at least one target frequency was missing. Looking at each input
separately, 644 of 792 frequency stencils were defined; 327 of these were flat
and 205 had forward/backward slope disagreement above 0.5. These overlapping
statistics diagnose step sensitivity, not mathematical proofs of discontinuity.

Four short L-BFGS-B trials used two finite-difference steps per objective. All
started from the infeasible parent centers because no eligible profile point was
found. Each requested the finite rejection score 25 times, with zero requests
having a defined full scientific loss. All stopped abnormally at the configured
four-step line-search limit. The best sampled point is retained and replayed;
it must not be described as a converged gradient solution.

At the finest timestep, the frequency-only sampled result kept 10/11 passing
targets and reduced partial frequency MAPE from 14.5% to 12.0%, while amplitude
error increased. The combined sampled result increased passing targets from
8/11 to 9/11 and reduced partial frequency error, also with worse amplitude
error. Thus the trials found some local improvements but did not recover a full
fit. Their small line-search budget prevents a general verdict against gradient
methods.

The expanded frequency-only search evaluated 765 unique candidates in 1751 s
and retained the previous in-domain diagnostic candidate. No eligible all-target
solution was found. Some restart winners had more detectable cycles but violated
the configured temperature domain; they are not accepted fits.

## Expanded searches and common finest-step results

Both expanded searches completed 765 unique evaluations (four starts); search
elapsed times were 1751 s for frequency-only and 1879 s for combined. Both
selected the old frequency-only vector as their in-domain diagnostic. The combined
score prioritizes failed-target count before partial error; it therefore prefers
this vector despite its much larger amplitude error. That is a property of the
finite rejection score, not a reduction of a defined all-target loss.

All entries below are replays at 0.00625 ns. Frequency MAPE uses only records
with defined frequencies, whose count is shown. Vpp MAPE uses all 11 targets,
including the existing zero-cycle-amplitude sentinel when no cycle was found.
Passing targets refers to cycle/frequency checks, not small residuals.

| Objective | Method | Passing targets | Frequency MAPE (defined) | Vpp MAPE | Oscillating controls |
|---|---|---:|---:|---:|---:|
| frequency_amplitude | baseline | 8/11 | 18.2% (9/11) | 903.0% | 0/11 |
| frequency_only | baseline | 10/11 | 14.5% (10/11) | 1965.7% | 5/11 |
| frequency_amplitude | expanded | 10/11 | 14.5% (10/11) | 1965.7% | 5/11 |
| frequency_only | expanded | 10/11 | 14.5% (10/11) | 1965.7% | 5/11 |
| frequency_amplitude | gradient | 9/11 | 14.2% (9/11) | 1195.5% | 0/11 |
| frequency_only | gradient | 10/11 | 12.0% (10/11) | 1979.8% | 5/11 |

No candidate has an accepted complete scientific loss. All six complete
22-record replays leave the configured temperature domain somewhere; that global
flag does not imply every target is outside it. The last refinement changes
frequency by up to 3.85% for the sampled frequency-gradient result and 11.67% for
the sampled combined-gradient result, so neither is established as numerically
converged. Expanded searches retain the baseline frequency-only convergence
limitations. These experiments show optimizer and feature-extraction sensitivity;
they do not establish that optimizer choice alone explains the model mismatch.

## Reviewed evidence

- Audit: `runs/published/20261020_185046_finite-difference-and-gradient-optimization-audi_21879a`.
- Expanded frequency: `runs/published/20261020_184721_expanded-optimizer-test-frequency-only-specimen-_aae50f`.
- Expanded combined: `runs/published/20261020_184721_expanded-optimizer-test-frequency-amplitude-spec_d5e6bb`.
- Common saved-data review: `runs/published/20261020_195947_optimizer-budget-derivative-and-gradient-review_555769`.

Software checks: 186 tests passed; repository validation passed. The exact three
new numerical recipes and the saved-data review command completed. No physical
equation or extraction algorithm was changed.
