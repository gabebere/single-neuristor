# Specimen model prediction versus measured current sweep

All 22 measured, baseline-corrected current waveforms were replayed
through one frozen parameter set. No parameter was retuned by current. The experiment
contains 11 coherent oscillatory records from
228.2 to
606.3 uA at
41.7--62.5 MHz;
the adopted model predicts 0.

The stable pre-onset trace is nevertheless reproduced closely: at 189.6 uA its measured
mean is 318.86 mV and the prediction is
319.53 mV. Thus the static cold-side
calibration works while the dynamic switching window does not.

In the algebraic C=0 limit, heating requires about
329.6 uA, but cooling through the opposite
transition requires current below 254.4 uA.
Because the lower bound exceeds the upper bound, no thermal-only oscillation window exists.
The adopted C=0.39 pF also produces no grid
oscillations, and none occur at any tested C within the timing bound for the full conditional
C_th interval. The first tested capacitance that produces an oscillation is
3 pF at the lower C_th bound and
7 pF at the adopted
C_th; both are outside the electrical timing bound.

Halving the integration step changes representative plateau means by at most
0.228 mV and does not change
their oscillation classifications. The failure is therefore a model/parameter
incompatibility, not a time-step artifact. Likely next tests are an independently measured
dynamic switching loop or a circuit model that includes the real TIA/load impedance;
gamma alone cannot repair the absent onset because the first heating transition occurs
before a minor-loop reversal.
