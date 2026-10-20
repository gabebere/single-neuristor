# Yuanhang voltage-driven relaxation oscillator

- Model: `voltage`
- Oscillatory: **True**
- Estimated frequency: **0.437749 MHz**

## Interpretation

The voltage-source model includes the external series resistor and parasitic capacitance. Its electrical charging time competes with VO2 heating and cooling.

- Series RC time constant: **1.74415 us**
- Thermal time constant: **0.241395 us**

> **Validity warning:** Part of the temperature trajectory lies outside the configured R(T) calibration range (305–370 K). The resistance law is clamped there.

## Reproduction

Use the command stored in `run.json`; all resolved inputs are in `resolved_config.json`.
