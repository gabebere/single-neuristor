# Environmental thermal-conductance estimate

The first coherently oscillating numerical trace is `300mv0_converted.csv`.
The immediately preceding trace, `250mv0_converted.csv`, is therefore the closest
measured stable point below oscillation onset.

Both channels are corrected by their pre-pulse medians. In the settled 100--250 ns
window the median current is **190.162 uA**, the median
voltage is **319.213 mV**, the effective resistance is
**1679.759 ohm**, and device power is
**60.658 uW**. Resistance changes by only
**0.145%** across the window, supporting
the quasi-steady approximation `dT/dt approximately 0`.

Inverting the specimen's fitted heating branch gives **T=330.905 K**.
With **T0=314.400 K**, the thermal balance gives
**S_e=0.003675 mW/K**. The conditional 95% interval is
**0.003434--0.004085 mW/K**.
It propagates waveform block resampling, the R(T)-fit bootstrap, and the stated ambient
range. It does not include the systematic possibility that the R(T) and TIA measurements
came from different devices or that driven and quasi-static R(T) differ.
