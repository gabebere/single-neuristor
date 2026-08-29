# Measured laboratory current sweep with onset trace

Loaded **22** numerical traces from `/Users/gabrielberezovsky/Documents/Projects/single-neuristor/data/experimental/tia_current_sweep`. No values were recovered from images.

Coherent oscillations are detected from **234** to **622 uA**, with measured frequencies from **41.7** to **62.5 MHz**.

The first coherently oscillating record is `300mv0_converted.csv` at **300 mV drive**. Its fixed 50--250 ns analysis window contains **8 peaks** with a period coefficient of variation of **2.03%**.

These electrical traces can constrain gamma only through a dynamic model. Given independently calibrated C, C_th, S_e, and T0, the resistive current is I_R=I_in-C dV/dt, power is P=V I_R, and the thermal equation reconstructs T(t). Gamma can then be fitted to the repeated minor-loop reversals. Without those thermal constraints, gamma is correlated with the latent temperature trajectory and is not independently identified.
