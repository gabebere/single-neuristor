# Relative-error oscillation fitting (finished)

Targets are selected by measured baseline-corrected current, averaged over
50–250 ns. Their physical magnitudes and archival record IDs are in targets.csv.
Selection range (µA): [225.0, 610.0].
The full measured current waveform and prehistory drive the original model.
Features use complete voltage cycles in [150.0, 250.0] ns.
Frequency is 1000/median cycle period (ns); amplitude is median cycle Vpp (mV).

L = mean[wf*(f_sim/f_measured - 1)^2 + wA*(Vpp_sim/Vpp_measured - 1)^2].
wf = 1.0; wA = 1.0.
Zero amplitude weight removes amplitude error from optimization, but amplitude
and mean voltage are still reported. No phase or DC error enters either loss.

Eligibility is checked separately: sustained complete cycles, frequency <=
110.0 MHz, the configured temperature domain,
and sampled nonnegative proximity slope: True.
The frequency ceiling and temperature bounds are operational screens.
Rejected candidates receive a finite optimizer sentinel, not a scientific loss.
Missing frequencies/errors remain null. A failed attempt is labeled diagnostic.
An objective_loss is reported only for a candidate eligible on every target.

All six static parameters are fixed during the first 2
restarts. Additional fixed coordinates: ['T0_K'].
Cth is represented internally by tau_th = Cth/Se, not an extra physical parameter.
Search bounds: {"C_pF": [0.05, 80.0], "S_e_mW_per_K": [0.0003, 0.03], "tau_th_ns": [0.5, 150.0], "T0_K": [300.0, 325.0], "Tc_K": [326.0, 342.0], "w_K": [1.0, 16.0], "beta_per_K": [0.08, 1.2], "gamma": [0.6, 0.98], "Rs_315K_ohm": [1200.0, 5000.0], "Ea_over_k_K": [1800.0, 3300.0], "Rm_ohm": [5.0, 80.0]}.
Maximum unique evaluations: 256.
Two candidates from different losses must be cross-evaluated before attributing
an improvement to the objective. No global optimum or physical calibration is implied.

All 22 records are replayed at [0.025, 0.0125, 0.00625] ns.
Other currents are controls, not blind validation. Create STOP to stop gracefully.


Accepted search candidate found: False.


## Completion
{
  "unique_candidates": 250,
  "search_elapsed_s": 524.9117288750131,
  "elapsed_s": 570.0429971250123,
  "stopped_by_user": false,
  "verified_batches": 9,
  "expected_verification_batches": 9,
  "target_labels": [
    300.0,
    350.0,
    400.0,
    450.0,
    500.0,
    550.0,
    600.0,
    650.0,
    700.0,
    750.0,
    800.0
  ],
  "verification_complete": true
}

Inspect summary.csv at the finest timestep before selecting a result. All target records were used in optimization; other currents are diagnostics, not blind validation.
