# Tanh-proximity pilot — 23 September 2026

Completed a bounded three-start search: 113 candidates, approximately 4 minutes
including all 12 verification batches. Separate P(x)=1-tanh(kx) variant;
original Yuanhang mode remains the default and old results are unchanged.
The gamma column means k in this variant. Best k=0.13072523.

At the finest timestep (0.00625 ns), search_1 passes the operational sustained-cycle
gate on 9/9 original labels 300–700, with frequency mean absolute relative error
16.93%, robust-amplitude error 646.98%, and static log10 RMSE 0.042914 (<0.05).
All nine pass at 0.025 and 0.0125 ns too. Minimum effective-temperature slope on
the target histories is +0.86927. Zero traces meet the near-target accuracy goal.
Thus the pilot demonstrates sustained oscillations without the diagnosed proximity
hook; it does not demonstrate a quantitative fit, infinite-time stability or a
validated physical calibration. Amplitude remains far too large.

The identical new persistence gate applied to archived finest-step data gives 0/9
for the constrained sine-kernel best and 7/9 for the unconstrained sine-kernel best.
The comparison changes BOTH the kernel and the fitting loss; it does not isolate
the effect of the kernel on optimized performance. Reference vectors reevaluated
under tanh are labelled accordingly in the new bundle. Raw errors remain separate.

Checks: 77 tests passed; 22 recipes / 82 archived runs validated. Three existing
NumPy arctanh warnings remain. Tanh nested reversal paths were checked at three
step sizes with decreasing differences; default upstream fidelity tests pass.

Artifacts: bundle in Results; comparable_metrics.csv reapplies the same gate to
old saved waveforms, without changing them. settled_waveforms.png shows the
amplitude discrepancy directly. No larger optimization is running.

Next useful search: prioritize reducing amplitude while retaining all nine
sustained traces and the resistance/monotonicity constraints. Verify against
all current controls and smaller timesteps; this pilot alone does not warrant
claiming the channel/circuit discrepancy is resolved.

Canonical bundle: `runs/20260923_061836_tanh-proximity-bounded-pilot-labels-300-to-700_11a3f7`.
