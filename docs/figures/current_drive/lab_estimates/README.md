# Lab current-trace parameter estimates

Digitized source: `data/Current Results`
Resistance preset: `presets/resistance_100425_chip1_gap3.json`

## Directly supported by the digitized traces

- Switching onset is bracketed between 194.6 and 233.4 uA, where plateau ripple rises sharply.
- Cold-slope electrical capacitance estimates span 20.9-25.0 pF; median C=22.7 pF.
- Above 350 uA, measured plateau V/I spans 135.0-472.0 Ohm. The specimen R(T) fit has Rm=18.3 Ohm, so using that bare Rm as the switched-state floor forces voltage far below the observed plateau.
- The high-current plateau remains near 190 mV while current changes substantially, so one constant series resistance alone cannot reproduce the entire family.

## Thermal conductance and ambient temperature

Using the last pre-switch point and S_e=P_switch/(T_switch-T0), the estimate depends strongly on the actual stage/substrate temperature:

|   T0_K |   T_switch_K |   Delta_T_K |   I_pre_switch_uA |   V_pre_switch_mV |   P_pre_switch_mW |   S_e_mW_per_K |
|-------:|-------------:|------------:|------------------:|------------------:|------------------:|---------------:|
|    298 |       336.96 |     38.962  |            194.65 |            315.18 |          0.061349 |      0.0015746 |
|    325 |       336.96 |     11.962  |            194.65 |            315.18 |          0.061349 |      0.0051288 |
|    330 |       336.96 |      6.9615 |            194.65 |            315.18 |          0.061349 |      0.0088126 |
|    333 |       336.96 |      3.9615 |            194.65 |            315.18 |          0.061349 |      0.015486  |

T0 must therefore be measured or fixed from the experiment before fitting S_e. In a single-device simulation, this S_e is the environment/substrate coupling. Neighbor coupling S_c is zero and cannot be inferred from one device.

## Thermal capacitance

The screenshots do not provide an independent temperature-recovery time. Once tau_th is measured from a pump-probe or cooling transient, use C_th=S_e*tau_th. The generated scenario CSV tabulates this relation for 10, 20, 50, and 100 ns.

The prior image-based joint fit used C=20.05 pF, C_th=6.55 pJ/K, S_e=0.1294 mW/K, and T0=337.93 K. Treat these as correlated starting values, not independent measurements.

## Recommended next measurement

Export raw time, source current, and voltage CSVs together with stage temperature and the exact measured voltage nodes. Fit the cold electrical edge first, then the post-pulse thermal recovery, and only then fit the coupled switching waveform.
