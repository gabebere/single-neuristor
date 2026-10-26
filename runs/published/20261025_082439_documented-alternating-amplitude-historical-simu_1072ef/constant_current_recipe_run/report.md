# Archived alternating-amplitude constant-current control 20261025

- Model: `current`
- Oscillatory: **False**
- Estimated frequency: **43.8757 MHz**

## Interpretation

The ideal-current model obeys `C dV/dt = I - V/R(T)`. In its fast or zero-capacitance limit, the voltage is `V = I R(T)`.

For this run, the metallic resistance is **18.2153 ohm**, so the predicted metallic voltage floor is **0.00552635 V**. A low valley is therefore a physical consequence of the chosen metallic resistance; capacitance changes how rapidly that valley is approached, not its steady-state value.

- Electrical metallic time constant: **0.12451 ns**
- Thermal time constant: **0.0119087 us**

## Reproduction

Use the command stored in `run.json`; all resolved inputs are in `resolved_config.json`.
