# Current step with a nonzero metallic voltage valley

- Model: `current`
- Oscillatory: **True**
- Estimated frequency: **0.221685 MHz**

## Interpretation

The ideal-current model obeys `C dV/dt = I - V/R(T)`. In its fast or zero-capacitance limit, the voltage is `V = I R(T)`.

For this run, the metallic resistance is **1286.32 ohm**, so the predicted metallic voltage floor is **0.77179 V**. A low valley is therefore a physical consequence of the chosen metallic resistance; capacitance changes how rapidly that valley is approached, not its steady-state value.

- Electrical metallic time constant: **186.961 ns**
- Thermal time constant: **0.965581 us**

## Reproduction

Use the command stored in `run.json`; all resolved inputs are in `resolved_config.json`.
