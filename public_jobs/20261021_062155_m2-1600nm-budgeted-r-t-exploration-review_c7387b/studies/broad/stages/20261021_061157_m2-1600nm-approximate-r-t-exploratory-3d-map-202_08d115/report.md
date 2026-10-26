# Steady-state extrema parameter maps

Input: 500 µA ideal constant current; ambient and initial temperature
298 K. The same fitted static sample resistance is used everywhere.
There are 216 actual simulations in the coarse grid, with
0 retained points after screening and verification.

Method: use only 300–600 ns.
Voltage amplitude is the median of complete-cycle maxima minus minima, in mV.
Frequency is 1000 divided by the median adjacent-maximum period, in ns.
Extrema are located using 0.5 ns smoothing; their voltages use original samples.
At least two complete cycles, cycles in both half-windows, amplitude retention
0.75–1.333 and period coefficient of variation <=0.2 are required.

The display additionally requires frequency <=110 MHz,
the entire temperature trajectory inside [289.98,
370.18] K, and nonnegative sampled proximity slope.
The frequency cutoff is the meeting's operating filter, not a proven material limit.
Each coarse survivor is rerun at [0.025, 0.0125] ns; frequency
and amplitude must change by <=5% at both refinements.
Only these verified survivors are plotted. Coarse rejected points have not all
been verified at finer steps, so a blank cell does not prove absence of oscillations.

Each heatmap fixes exactly one grid value: {'C_th_pJ_per_K': 0.1, 'S_e_mW_per_K': 0.003, 'C_pF': 6.4}.
The other two parameters vary independently; frequency and peak-to-peak voltage
are shown separately. Blank cells are screened out; grid.csv preserves all reasons.
No measured parameter uncertainty is inferred from these exploratory ranges.
The static R(T) preset does not independently identify gamma.
