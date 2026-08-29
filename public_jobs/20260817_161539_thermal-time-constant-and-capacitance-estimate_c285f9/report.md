# Thermal time constant and capacitance estimate

With the electrical state fixed to the unresolved limit `C=0`, the baseline-corrected
ratio `V/I` is mapped to temperature through the fitted specimen heating branch. The
measured power drives `C_th dT/dt = P(t) - S_e (T-T0)` over the 15--35 ns heating
window. The primary fit uses `100mv0_converted.csv,150mv0_converted.csv,200mv0_converted.csv`: these moderate nonswitching
traces have adequate signal and remain below the near-transition overshoot.

The shared fit gives **tau_th=12.833 ns** and
**C_th=0.047163 pJ/K**, with a temperature RMSE of
**0.806 K**. The individual selected-trace estimates span
**0.041627--0.049521 pJ/K**.

The conditional 95% robustness interval is
**0.020122--0.092353 pJ/K**
for C_th and **5.412--24.769 ns**
for tau_th. It propagates trace resampling, the R(T)-fit bootstrap, the conductance
bootstrap, and +/-2 ns fit-window changes. It remains conditional on C=0, the static
heating branch applying dynamically, and all measurements describing the same device.

The `250mv0_converted.csv` sensitivity trace gives
**C_th=0.022962 pJ/K** with a larger
**2.089 K** error, confirming that including its
near-transition reversal biases the estimate downward. The fitted thermal time is the
same order as the measured **16.0--24.0 ns** oscillation
period. Yuanhang's **49.6278 pJ/K** reference is about
**1052 times larger**.

The manuscript reports the 150 nm film thickness and approximately 200 nm electrode
gap, but not the electrically active filament width or volume. A geometry calculation
`rho c_p V` is therefore not reported as a numerical cross-check.
