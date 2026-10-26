# Physics-anchored specimen waveform inference

One shared eight-parameter vector was fitted to 20 traces;
the source settings [50.0, 1000.0] mV were excluded from optimization
and used only for validation. The objective combines normalized waveform RMSE,
phase-tolerant oscillatory shape, plateau mean and amplitude, spectrum, frequency,
oscillation classification, sustained periodic amplitude across four plateau
segments, the 0--50 ns edge, and (for the constrained fit) weak independent-
measurement priors. The exact weights are archived in `resolved_config.json` and
`metrics.json`.

The original estimates give a full-data objective of
**1280** and classify
**11/22** traces correctly. The physically
constrained fit gives **1271.8**,
**11/22** correct classifications, and
predicts **0** oscillating traces.
The relaxed diagnostic fit gives **35.455**,
**22/22** correct classifications, and
predicts **11** oscillating traces.

Relaxed values outside the independently allowed ranges: **C_pF, gamma**. These are
effective values required by the present equations, not measurements. The held-out
traces and fine-step reruns distinguish generalization from memorization and numerical
step artifacts. No confidence intervals are assigned: this deterministic global fit
is an identifiability diagnostic, and several parameters remain correlated.
