# Global specimen parameter inference from all current waveforms

One shared eight-parameter vector was fitted to 17 traces;
the source settings [200.0, 400.0, 600.0, 800.0, 1000.0] mV were excluded from optimization
and used only for validation. The objective combines normalized waveform RMSE,
phase-tolerant oscillatory shape, plateau mean and amplitude, spectrum, frequency,
oscillation classification, the 0--50 ns edge, and (for the constrained fit) weak
independent-measurement priors.

The original estimates give a full-data objective of
**0.88131** and classify
**11/22** traces correctly. The physically
constrained fit gives **0.87496**,
**11/22** correct classifications, and
predicts **0** oscillating traces.
The relaxed diagnostic fit gives **0.68273**,
**11/22** correct classifications, and
predicts **0** oscillating traces.

Relaxed values outside the independently allowed ranges: **C_pF, C_th_pJ_per_K, S_e_mW_per_K, T0_K, Tc_K, w_K, beta_per_K**. These are
effective values required by the present equations, not measurements. The held-out
traces and fine-step reruns distinguish generalization from memorization and numerical
step artifacts. No confidence intervals are assigned: this deterministic global fit
is an identifiability diagnostic, and several parameters remain correlated.
