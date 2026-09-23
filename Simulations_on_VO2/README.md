# VO2 mechanism and parameter-estimation presentation

This directory contains the editable Beamer presentation that began as the July 2026
mechanism-validation talk and was extended in September 2026 with the specimen
parameter-estimation evidence. `main.tex` is the source and `main.pdf` is the current
compiled deck. The original July version remains recoverable from Git history.

The first section demonstrates the Yuanhang mechanism under ideal current drive. The
continuation then gives one evidence-focused slide for each experimental input or
model parameter: measured current, ambient temperature, major-loop resistance
parameters, minor-loop memory, environmental conductance, electrical capacitance,
and thermal capacitance. It ends with the frozen forward comparison, which shows that
the physically anchored parameter set does not yet reproduce the measured oscillatory
region. The 21 September continuation adds a bounded joint fit of all six major-loop
parameters and five dynamic/ambient quantities, with three method slides and five
result slides. `joint_fit_results.tex` holds the run-specific editable results.

The joint-fit evidence is bundle
`20260921_130032_budgeted-joint-resistance-and-waveform-inference_6b5d8f`.
The static-preserving candidate reduces the validation feature score by 33.5%
relative to the replayed anchored seed, but recovers only 4/11 persistent
oscillators; the diagnostic recovers 5/11 while substantially worsening R(T).
Neither is adopted as a physical calibration. Full definitions and denominators
are in [`docs/JOINT_INFERENCE_20260921.md`](../docs/JOINT_INFERENCE_20260921.md).

The deck summarizes the evidence but is not the numerical source of truth. Current
values, limitations, commands, and immutable bundle links live in
`docs/RESEARCH_HANDOFF.md`; the full derivations are in
`docs/final_project/Simulations_for_VO2_AGC.pdf`.

## Continuation evidence map

| Topic | Slide evidence |
|---|---|
| Measured current and onset bracket | `public_jobs/...ec6ec4/figures/oscillation_onset_bracket.png` |
| Major-loop resistance fit | `public_jobs/...0849a9/figures/resistance_fit.png` |
| Minor-loop-memory limitation | `public_jobs/...fa4b66/figures/reconstructed_hysteresis.png` |
| Environmental conductance | `public_jobs/...761640/figures/environmental_conductance.png` |
| Electrical-capacitance bound | `public_jobs/...eefab7/figures/capacitance_sensitivity.png` |
| Thermal capacitance | `public_jobs/...aa2469/figures/thermal_capacitance.png` |
| Frozen forward comparison | `public_jobs/...eefab7/figures/representative_traces.png` |

The shortened bundle suffixes above are unique within `public_jobs/`; the TeX source
contains the complete paths used to compile the deck.

Expanded-search results and the editable laboratory are documented on slides 33–34
in `expanded_search_results.tex`. These report the 694804 bundle at the finest
step and retain the unresolved persistence/amplitude tradeoff. Full notes:
`../docs/EXPANDED_SEARCH_20260922.md`.
