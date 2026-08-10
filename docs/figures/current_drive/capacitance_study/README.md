# Current-drive capacitance study

This sweep uses Yuanhang's resistance, thermal-conductance, and ambient-temperature values in the derived ideal-current circuit.
The upstream reference circuit itself is voltage-driven through a 12 kOhm load; it is not an ideal-current circuit.

## Baseline parameters

- Electrical capacitance: 145.34619 pF
- Thermal capacitance: 49.627768 pJ/K
- Environmental thermal conductance: 0.20558726 mW/K
- Ambient temperature: 325 K
- Thermal time constant C_th/S_e: 241.395 ns
- Metallic electrical time constant Rm*C: 186.961 ns
- Integration step: 0.5 ns; simulated to 30 us; analysis starts at 8 us

## Main findings

- Yuanhang metallic resistance is 1286.317 Ohm, so the ideal-current voltage floor is I*Rm, not zero.
- Lower C shortens the approach to that floor; it does not raise the floor.
- The approximate RC=0 thermal-cycle bounds are 382.8 to 198.6 uA. Because the lower bound exceeds the upper bound, the Yuanhang parameters do not support a thermal-only limit cycle in this approximation.
- Sustained cells found in the finite-C grid: 104 of 294.
- 109 cells leave the 305-370 K calibrated resistance range; they are marked with * and should be treated as extrapolations with saturated R(T).
- Cells marked zero/gray either settle to a fixed point or fail the late-time regularity criteria; startup ringing is excluded.

## Calibration equations

- From a fixed-resistance electrical transient: C = tau_el/R.
- From thermal recovery: C_th = S_e*tau_th.
- At switching onset: S_e approximately equals P_switch/(T_switch-T0), with P=V^2/R (or I^2 R at current steady state).
- In a single-device run, neighbor coupling is zero. Fit the environment/substrate path as S_e; reserve S_c for multi-device thermal coupling.
