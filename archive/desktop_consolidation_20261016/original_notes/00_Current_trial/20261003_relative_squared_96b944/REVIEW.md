# Simple frequency/amplitude trial — diagnostic, not an accepted fit

Start with `GIF Results/START_HERE.html` to select a measured current.
Individual animations: `GIF Results/GIFs/`. Final stills: `GIF Results/Figures/`.
ZIP of all 22 animations: `GIF Results/All_current_GIFs.zip`.
Numerical evidence and frozen fitting source: `Search Results/`.

## What changed

Loss = mean[(f_sim/f_exp − 1)² + (Vpp_sim/Vpp_exp − 1)²].
Both features come from complete-cycle maxima/minima in 150–250 ns.
Equal percentage errors have equal weight; no phase, DC or pointwise voltage
error. No log, epsilon or tolerance scaling. Static R(T) is only a hard constraint.
Physical eligibility is separate. Optimizer rejection sentinels are not the loss.

## What happened

85 candidates, two restarts, approximately 3 minutes searching and 48 seconds
of verification. All three candidate/reference vectors were checked on all 22
records at 0.025, 0.0125 and 0.00625 ns. No searched candidate passes all nine
target eligibility checks. GIFs show diagnostic_1 at 0.00625 ns.

- Sustained target cycles: 8/9; sustained and <=110 MHz: 5/9.
- Combined 2% frequency / 5% amplitude goal: 0/9.
- Static R(T) RMSE: 0.036628 in log10 R. Proximity minimum slope: +0.24467.
- Mean absolute amplitude relative error: 973.4% over nine targets.
- Mean absolute frequency relative error: 78.9% over eight defined frequencies.
  The ninth is missing, not zero error. This is not a valid nine-target objective.
- Temperature range in the full all-current GIF replay: 317.46–364.07 K.

At ~381 µA, measured frequency/Vpp are 51.28 MHz / 45.76 mV;
simulation gives 90.91 MHz / 401.98 mV. Oscillations exist, but they are too fast
and too large. The lowest target does not oscillate; three highest target
frequencies exceed the selected ceiling. No new parameter calibration is adopted.

The generic replay summary uses a historical persistence definition, not the
new complete-cycle gate. Its 10/11 recovery count is NOT a new fit-success claim.
Use Search Results/verification.csv for this trial's actual extrema checks.
Likewise, mean_dynamic_loss for rejected candidates averages defined diagnostic
cycle losses only; it is not an accepted all-target objective. Null cells are
intentional where no frequency can be measured.

GIFs display measured current, measured/simulated voltage and simulated R(T)
over model major branches. R and T are not measured. Large title numbers are
the historical original-numeric-label aliases; use the subtitle, amperes axis,
and current filenames for actual measured current. No mV→µA conversion is assumed.
Playback is slowed and visually sampled (48 frames); simulation step is 0.00625 ns.
Replay voltage agrees with the search's same-candidate finest-step samples to
within 0.0093 mV across all 22 traces; the replay also includes the pulse fall.

## Interpretation / next decision

The simpler loss works as intended, but does not fix the model discrepancy.
Treat this as a failed bounded pilot, not a global optimum. Review the middle
currents first, then onset and high-current controls. Before a longer search,
decide whether to continue the original proximity kernel or test another
mechanism under the same new objective. Do not change kernel and objective
together if the aim is to identify which change helps.

Canonical reviewed run IDs: search `96b944`, GIF replay `4e79ac`.
Recipe files are in the repository experiments/current folder.
This folder is a local copy. It does not confirm a GitHub commit or upload.
