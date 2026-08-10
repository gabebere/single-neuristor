# Lab current-trace parameter estimates

The switching onset is bracketed between **195** and
**233 uA** from the plateau-ripple increase.

The cold-edge estimate `C = I/(dV/dt)` gives a median electrical capacitance of
**22.7 pF**.

`S_e = P_switch/(T_switch-T0)` is tabulated for each candidate ambient temperature.
The screenshot alone does not identify thermal capacitance: once a recovery time is
measured, use `C_th = S_e tau_th`; the scenario table makes that dependence explicit.

The fitted specimen metallic resistance is **18.3 ohm**. In an ideal
current source, it fixes the steady switched-state floor through `V = I Rm`.
Electrical capacitance changes the approach time but cannot change that steady floor.
