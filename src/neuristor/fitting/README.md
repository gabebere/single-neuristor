# Fitting: estimate parameters, not a new model

| Module | Purpose |
|---|---|
| `resistance_custom_analysis.py` | Fit static major-loop resistance data; retain historical stateful fitting. |
| `steady_inference.py` | Fit settled features with recipe-selected extrema/loss definitions and physical gates. |
| `optimizer_audit.py` | Cache coordinate perturbations and test bounded L-BFGS-B on the same finite optimizer score. |
| `joint_inference.py` | Earlier joint static/dynamic searches with recorded parameter constraints. |
| `parameter_inference.py` | Earlier waveform-level fitting and inverse/reconstruction calculations. |
| `current_domain_search.py` | Historical domain exploration; not the new sample-study path. |

Old objectives remain here because archived trials depend on them. Their
existence does not make them the recommended objective or an accepted fit.
The current method is selected explicitly in the TOML recipe. Major-loop R(T)
data do not independently identify gamma or thermal/electrical dynamics.

[Current extrema loss](../../../docs/SIMPLE_EXTREMA_LOSS_20261003.md) ·
[Research handoff](../../../docs/RESEARCH_HANDOFF.md)
