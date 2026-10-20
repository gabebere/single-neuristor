# Thermal capacitance estimate with conservative 0.39 pF electrical bound

With electrical capacitance fixed to **C=0.39 pF**,
the resistive current is reconstructed as `I_R=I_in-C*dV/dt`. The baseline-corrected
ratio `V/I_R` is mapped to temperature through the fitted specimen heating branch,
and `V*I_R` drives `C_th dT/dt = P(t) - S_e (T-T0)` over the 15--35 ns heating
window. The primary fit uses `100mv0_converted.csv,150mv0_converted.csv,200mv0_converted.csv`: these moderate nonswitching
traces have adequate signal and remain below the near-transition overshoot.

The shared fit gives **tau_th=13.026 ns** and
**C_th=0.047873 pJ/K**, with a temperature RMSE of
**1.145 K**. The individual selected-trace estimates span
**0.043261--0.049830 pJ/K**.

The conditional 95% robustness interval is
**0.021918--0.092624 pJ/K**
for C_th and **5.865--25.174 ns**
for tau_th. It propagates trace resampling, the R(T)-fit bootstrap, the conductance
bootstrap, and +/-2 ns fit-window changes. It remains conditional on the adopted
electrical capacitance, the static heating branch applying dynamically, and all measurements describing the same device.
The conductance archive does not retain its paired R(T)-parameter draw, so the two
parameter bootstraps are resampled independently; the interval is conservative and
does not preserve that covariance.

The `250mv0_converted.csv` sensitivity trace gives
**C_th=0.022858 pJ/K** with a larger
**2.380 K** error, confirming that including its
near-transition reversal biases the estimate downward. The fitted thermal time is the
same order as the measured **16.0--24.0 ns** oscillation
period. Yuanhang's **49.6278 pJ/K** reference is about
**1037 times larger**.

The manuscript reports the 150 nm film thickness and approximately 200 nm electrode
gap, but not the electrically active filament width or volume. A geometry calculation
`rho c_p V` is therefore not reported as a numerical cross-check.
