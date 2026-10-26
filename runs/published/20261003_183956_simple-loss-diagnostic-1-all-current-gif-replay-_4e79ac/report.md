# Manual laboratory replay

One shared vector drives every measured current record, retaining its full prehistory. Both experimental channels have their pre-pulse median removed. Output voltage is compared as exported; its equivalence to device terminal voltage remains conditional. Resistance is simulated, not inferred V/I.

The HTML includes a current slider. Every replay includes synchronized three-panel GIFs showing imposed current, measured/simulated voltage and the evolving simulated R(T) path over the heating/cooling major branches. START_HERE.html selects the current and All_current_GIFs.zip contains all animations. Editing parameters does not alter completed runs. This is a single-step exploratory replay, not a convergence certificate.

- currents: 22
- measured_oscillators: 11
- recovered_oscillators: 10
- false_positives: 1
- late_mean_RMSE_mV: 38.25784012571174
- outside_calibration: False

Reproduce with `neuristor analyze replay-lab --config PATH_TO_BUNDLE/recipe.toml`.
