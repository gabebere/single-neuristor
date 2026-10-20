# Analysis: what the data and simulations tell us

These modules calculate evidence from numerical measurements or solver output.
They do not invent experimental data from screenshots.

| Module | Purpose |
|---|---|
| `experimental_waveforms.py` | Read measured current/voltage exports and retain acquisition labels. |
| `extrema.py` | Extract complete steady-state cycles, frequency, and peak-to-peak voltage. |
| `metrics.py` | Basic single-run summaries; not a replacement for the extrema objective. |
| `lab_estimates.py` | Conditional thermal conductance and timing-based capacitance estimates. |
| `model_validation.py` | Compare frozen model predictions with measurements. |
| `oscillation_audit.py` | Distinguish sustained oscillations from startup or decaying peaks. |
| `lab_replay.py` | Replay measured current through the same model and construct branch guides. |
| `physical_maps.py` | Sample and screen Cth/Se/C grids, check refinement, and construct map views. |
| `boundary_search.py` | Expand coarse ratio/Se exploration, extend horizons, verify fine transitions, and retain open/censored limits. |
| `distribution.py` | Check staged/release contents against the private-PDF policy; never upload files or infer scientific acceptance. |

Acquisition labels in mV are not measured currents. Frequency is reported in MHz
at map boundaries. Screening and timestep agreement are not proof of a correct
sample calibration. [Scientific handoff](../../../docs/RESEARCH_HANDOFF.md).
