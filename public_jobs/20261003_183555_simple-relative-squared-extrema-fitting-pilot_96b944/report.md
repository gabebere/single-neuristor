# Background settled-oscillation search (finished)

User-selected target: original file labels [300, 350, 400, 450, 500, 550, 600, 650, 700], NOT physical current values.
Actual plateau currents are in targets.csv in amperes. Dynamic objective uses only
150.0–250.0 ns, excluding DC offset and startup.
The full measured current prehistory still drives the unmodified ideal-current model.
Proximity variant: yuanhang. For tanh, the gamma
coordinate means k in P(x)=1-tanh(k*x); it is not the sine-kernel gamma.
Sustained-cycle gate enabled: True. When enabled,
require at least three prominent peaks, peaks in both halves, retention 0.75–1.333,
period CV <=0.2 and the frequency signal gate. Failed traces are rejected separately;
it cannot qualify as near-target. Best available does not imply sustained success.
Feature method: extrema.
For extrema, frequency = 1000/median adjacent-maximum period (ns) and amplitude
= median complete-cycle maximum-minus-minimum voltage (mV). Only these two
features enter the dynamic loss: mean[(f_sim/f_exp-1)^2 + (Vpp_sim/Vpp_exp-1)^2].
No scaling, logarithm or amplitude offset is used. Static R(T) is a hard
constraint only. A finite optimizer rejection sentinel guides feasibility and
is not the scientific loss. Missing frequencies have null errors, not 100% error.
A diagnostic candidate is NOT an accepted fit. The legacy
robust/fundamental table columns both mean extrema Vpp in this mode.
For spectral mode, robust/fundamental amplitudes and half-window decay enter
the historical objective. Flat traces cannot evade frequency error.
Static log10 R(T) RMSE must remain <= 0.05.
Require nonnegative proximity dTeff/dT at every integration sample, including
prehistory and transients: True.
When enabled, verification also reports this check on historical references;
an accurate waveform does not override a failed proximity check. This is a
sampled path constraint, not a proof of all possible minor-loop properties.
The first 1 restarts fix all six static parameters.
Other restarts allow them to vary, using correlated static-bootstrap starting points.
No likelihood, confidence interval, global optimum or perfect fit is implied.

This local Python process does not use ChatGPT or API calls. Check status.json and
checkpoint.json; each unique evaluation is appended to optimization_history.jsonl.
Create an empty STOP file in this folder to end the search gracefully.
Nominal total budget: 0.17 hours including verification.
The deadline is checked between simulations, so one in-flight batch may overrun it.


Accepted search candidate found: False.


## Completion
{
  "unique_candidates": 85,
  "search_elapsed_s": 181.54696558300202,
  "elapsed_s": 228.17314808299852,
  "stopped_by_user": false,
  "verified_batches": 9,
  "expected_verification_batches": 9,
  "target_labels": [
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700
  ],
  "verification_complete": true
}

Inspect summary.csv at the finest timestep before selecting a result. All target records were used in optimization; other currents are diagnostics, not blind validation.
