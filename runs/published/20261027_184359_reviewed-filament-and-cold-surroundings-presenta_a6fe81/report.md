# Reviewed filament/cold-surroundings experiment — 27 October 2026

The optional model is implemented, but this bounded trial does **not** establish
an improved quantitative fit. Both searches completed 249 unique evaluations.
The extra-parameter arm returned f=1, reproducing the presentation combined-loss
candidate. It did not discover a better fractional model within this budget.
This is not evidence that the physical filament fraction is one, or a rejection
of localization: the candidate is ineligible and the parameters remain unidentified.
The nested f=1 family also contains every baseline candidate, including the new
baseline refit; a finite five-dimensional search can return a weaker diagnostic
than a four-dimensional search without making the model family intrinsically worse.

At the finest 0.00625 ns step, the baseline refit passes cycle/frequency checks
on 10/11 targets; the returned extra-parameter candidate passes on 8/11, with
9 defined frequencies. Diagnostic all-target Vpp MAPE is approximately 1592%
and 903%, respectively. These are large errors with different frequency coverage,
not an accepted improvement. All-target accepted objective values remain null.

Explicit f=0.75 cases pass cycle/frequency checks on 9/11 targets but push three
outside the calibrated temperature range (target Tmax about 378.3–378.5 K).
Explicit f=0.1 cases pass none of the 11 target cycle/frequency checks and reach
about 483–485 K on the hottest targets. Background Joule power is accounted for
separately rather than heating the filament. Smaller Vpp in a quenched case does
not demonstrate agreement. No tested candidate passes every target's feature
and waveform checks across both timestep halvings. These are conditional
numerical diagnostics, not finely converged physical predictions.

The two original presentation vectors reproduce the published finest-step
frequency/Vpp diagnostics. Baseline f=1 scalar/batched, finite-C and C=0 traces
also matched the pre-change 13e5188 outputs bit for bit in six reference cases.
The new independent constant-R control shows convergent partitioned heating
against an analytic RC solution across two timestep halvings. All 216 tests pass.

The 11 other currents are previously seen controls. Static R(T) and T0 are fixed.
A useful next experiment would constrain geometry/contact/readout quantities and
explore physically coupled filament thermal-volume seeds before enlarging the
search; the current finite-budget scan does not rule out other f, bounds, starts,
a warming surrounding region, or a different circuit-boundary description.

## Scope of this compact review

This is a new, self-contained evidence bundle. Numerical tables, sample-grid
waveforms, figures, input copies and code are preserved exactly with hashes.
The original completed numerical bundles were not edited. Full native arrays
(56 histories) and both full search histories stay in the two local source
bundles. Recreate them with the versioned recipe; the review omits those large
arrays and JSONL histories rather than duplicating them in Git. Their original
provenance and hashes are recorded. Search code provenance is 93ffaa7; the added
fractional verification uses 2dc3e8d with identical numerical modules and inputs.
The native-current/power conservation audit passed for all 56 histories.

Open START_HERE.html, then summary.csv, verification.csv, convergence.csv,
parameter_comparison.csv and candidates.json. The candidate vectors use the
named order documented by parameter_comparison.csv and the frozen joint fitter.
Existing slides and presentation routes are unchanged. No calibration is adopted.

## Full workflow method

# Experimental filament and cold surroundings

The user confirmed that the fitted R(T) and oscillation recordings use the same
electrodes; the R(T) loop used uniform stage heating. The proposed hypothesis is
localized filament heating rather than a new intrinsic material resistivity.

G_eff = f/R_fit(T_f,H_f) + (1-f)/R_fit(T0,H0).
C dV/dt = I_in - V G_eff.
Cth_f dT_f/dt = f V^2/R_fit(T_f,H_f) - Se (T_f-T0).

The cold branch uses an independent initial hysteresis state held at ambient.
Its Joule heat goes to an ideal external reservoir, not to the hot node. This is
a deliberately limited thermal assumption, not an energy-free shunt. P is total
electrical dissipation; native files separately record both currents and powers.
Equal temperatures and histories recover the measured uniform R(T); f=1 recovers
the original equations and original floating-point arithmetic. f is a normalized
conductance share, not a measured filament width. Cth and Se describe the heated
region, whose geometry is not independently identified here.

Both optimization arms use the same measured 11 target currents, full prehistory,
fixed six static-fit constants and T0, dynamic bounds, seed vectors, DE/Powell
procedure, wall-clock allowance and evaluation cap. The filament arm adds f in
the declared logarithmic interval. Actual counts can differ because of time
limits and caching. The unchanged presentation objective is mean squared
relative frequency plus Vpp error; mean voltage is an independent diagnostic.
Ineligible candidates have no accepted scientific objective. Missing frequencies
remain null; amplitude MAPE on all targets retains the original zero-feature
convention for missing cycles. Frequency MAPE always states its denominator.

The frozen-reference scans are additional sensitivity calculations, outside the
matched optimizer caps. Their selected fractions are reported separately. The
post-fit fraction scan fixes all other parameters: it is NOT a profile likelihood
or confidence interval. All 22 currents are checked at three timesteps. The other
11 currents are previously inspected controls, not blind validation. Neither
lower scalar error nor a completed bundle establishes an accepted specimen fit.
The measured channel-to-device voltage/current mapping remains a separate issue.

Explicit f<1 reference cases receive both finer timestep checks even if the
selected candidate returns to f=1. Search reuse is recorded in search_reuse.json
and metrics.json; it requires unchanged measured inputs, numerical model,
analysis and fitting modules, and the same search configuration. Only additional
verification fractions may differ. Reused searches are copied with hashes and
their original code provenance. Sensitivity calculations and verification are new.

Inspect summary.csv, verification.csv, and both timestep halvings in convergence.csv.
Waveform RMS differences use the same acquisition grid and late loss window;
feature convergence and waveform convergence are reported separately. The shared
temperature and proximity screens are not relaxed. No published bundle or
presentation route was modified.

Physical motivation: https://doi.org/10.1002/aelm.70604 (uniform heating versus
filamentary switching); this experiment is our proposed reduced model.

## Completed comparison

```json
{
  "calibration_adopted": false,
  "control_records": 11,
  "elapsed_s": 192.55692287499915,
  "finest_target_results": {
    "baseline_fitted": {
      "amplitude_MAPE_all": 15.921505221163983,
      "defined_frequencies": 10,
      "dt_ns": 0.00625,
      "dynamic_loss": 449.6068192218938,
      "eligible": false,
      "filament_fraction": 1.0,
      "frequency_MAPE_defined": 0.33783369697956817,
      "invalid_count": 1,
      "maximum_temperature_K": 365.0939636230469,
      "mean_voltage_RMSE_mV": 56.02790619434691,
      "objective_loss": null,
      "proximity_admissible": true,
      "records": 11,
      "score": 1001449.6068192219,
      "screened_targets": 10,
      "temperature_valid": true
    },
    "filament_fitted": {
      "amplitude_MAPE_all": 9.029837271636108,
      "defined_frequencies": 9,
      "dt_ns": 0.00625,
      "dynamic_loss": 162.01259893946337,
      "eligible": false,
      "filament_fraction": 1.0,
      "frequency_MAPE_defined": 0.18184545127073865,
      "invalid_count": 3,
      "maximum_temperature_K": 369.4241638183594,
      "mean_voltage_RMSE_mV": 40.09982357474286,
      "objective_loss": null,
      "proximity_admissible": true,
      "records": 11,
      "score": 1003162.0125989395,
      "screened_targets": 8,
      "temperature_valid": true
    }
  },
  "fit_counts": {
    "baseline": 249,
    "filament": 249
  },
  "fitted_filament_fraction": 1.0,
  "original_measurements": 22,
  "reference_contacts": "same electrodes; uniform stage heating (user confirmed)",
  "search_source_run_id": "20261027_175518_experimental-filament-and-cold-surroundings-pres_20c97c",
  "searches_reused": true,
  "static_fit_unchanged": true,
  "target_feature_convergence": {
    "baseline_fitted": false,
    "filament_fitted": false,
    "presentation_combined": false,
    "presentation_combined_f_0.1": false,
    "presentation_combined_f_0.75": false,
    "presentation_combined_fraction_scan": false,
    "presentation_frequency": false,
    "presentation_frequency_f_0.1": false,
    "presentation_frequency_f_0.75": false,
    "presentation_frequency_fraction_scan": false
  },
  "target_records": 11,
  "target_waveform_convergence": {
    "baseline_fitted": false,
    "filament_fitted": false,
    "presentation_combined": false,
    "presentation_combined_f_0.1": false,
    "presentation_combined_f_0.75": false,
    "presentation_combined_fraction_scan": false,
    "presentation_frequency": false,
    "presentation_frequency_f_0.1": false,
    "presentation_frequency_f_0.75": false,
    "presentation_frequency_fraction_scan": false
  }
}
```
