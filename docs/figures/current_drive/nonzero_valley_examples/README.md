# Nonzero-valley current-input examples

These examples answer a narrow question: can the supported ideal-current model sustain oscillations without the voltage reaching zero? Yes, but the interpretation depends on the resistance model.

## A. Yuanhang-centered, in-range demonstration

- Input pulse: 600 uA
- C=145.346 pF, C_th=198.511 pJ/K, S_e=0.205587 mW/K
- Frequency=0.2217 MHz; late voltage range=0.905--6.337 V
- Full-pulse temperature range=325.00--368.79 K, inside the published 305--370 K R(T) calibration range.
- This uses Yuanhang's R(T), C, S_e, and T0, with C_th set to four times the nominal value to keep the entire pulse trajectory inside the calibrated temperature range.

## B. Lab-scale diagnostic comparison

Both traces use I=1000 uA, the fitted specimen R(T) shape, measured C estimate of 22.7 pF, C_th=0.5 pJ/K, and S_e=0.08 mW/K.

- Unchanged fitted Rm=18.3 Ohm: minimum=0.018 V.
- Effective candidate Rm=150 Ohm: minimum=0.194 V, frequency=16.32 MHz.
- Candidate full-pulse temperature range=325.00--355.36 K, inside the fitted specimen R(T) range.

The 150 Ohm value is not claimed as the intrinsic metallic resistance measured by the R(T) fit. It is an effective switched-state candidate that could represent unresolved contact resistance, source/measurement impedance, or a different device state. The comparison isolates why C alone cannot raise the valley.

## Reproduce

```bash
neuristor simulate current \
  --config experiments/current/nonzero_voltage_valley.toml
```

The exact multi-panel historical generator is retained as
`legacy_scripts/generate_nonzero_valley_examples.py` and in the
`v0.1.0-working-baseline` Git tag.
