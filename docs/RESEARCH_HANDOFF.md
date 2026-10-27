# Research handoff

Last scientific review: **25 October 2026**. Layout updated **26 October 2026**. Read this, the root README, and AGENTS.md before
continuing. The [detailed research history](RESEARCH_HISTORY.md) preserves previous
results, commands, parameter tables, and caveats without making them a new task list.

## Purpose and honest status

Gabriel Berezovsky is developing simulations to complement Amir Gildor's work on
VO₂ automatic gain control in transimpedance amplifiers, supervised by Amir and
Prof. Yoav Kalcheim. We seek one physically supported parameter set that predicts
the measured sweep, not a different parameter set for each current.

**Implemented and tested:** Zhang's voltage circuit and inherited hysteresis;
the ideal-current extension; static R(T) fitting; laboratory replay; feature-based
search; persistence and timestep checks; screened 3D maps and slider-controlled
2D slices; point GIFs; an end-to-end new-sample workflow; a searchable run library.

**Not achieved:** a quantitatively satisfactory, physically supported fit to the
experimental TIA oscillations. A good static resistance fit does not establish
dynamic agreement. Historical “22/22” claims used permissive detection and must
not be repeated as sustained-waveform agreement.

## Where to start

| Need | Entry point |
|---|---|
| Plain-language navigation | [Project guide](../README.md) |
| Run and inspect a new sample | [SAMPLE_STUDY](SAMPLE_STUDY.md) |
| Authoritative physics/code responsibilities | [CODEBASE_MAP](CODEBASE_MAP.md), [AGENTS](../AGENTS.md) |
| Experimental inputs and units | [Data guide](../data/README.md) |
| Reports, methodology, presentations | [Document guide](README.md) |
| Current versus superseded evidence | [ARCHIVE_INDEX](ARCHIVE_INDEX.md) |
| All local runs and files | [Artifact library](../runs/catalog/index.html) |
| Paper claim → inputs → command → outputs | [Reproduction guide](reproduction/README.md) |
| Public/private material and GitHub release rules | [DATA_POLICY](DATA_POLICY.md) |

Keep existing completed bundle IDs, contents, and relative artifact paths intact.
Local publication (`runs publish`) is archival copying, not GitHub upload or
scientific acceptance. Do not regenerate the entire research history to orient
yourself.

Optional presentation reconstruction is available with `neuristor reproduce-presentation`.
It rebuilds the declared presentation dependencies from original measurements and
settings, then connects local viewers to the new bundle. Allow several hours or
overnight; this demonstrates the workflow and does not adopt a new calibration.

## Current scientific conclusions

- The implemented ideal-current equation is `C dV/dt = I - V/R(T,H)`; `C=0`
  is the exact algebraic limit. Its metallic steady floor is approximately `I Rm`.
  A series resistor does not change an ideal imposed current.
- Major-loop R(T) data constrain the static parameters, **not gamma** or the
  electrical/thermal dynamics. The sample fit is good; dynamic agreement is not.
- Thermal estimates depend on channel interpretation, transferred static
  thermometry, and timing/alignment assumptions. The 19.8 pF shortcut was rejected;
  0.39 pF is a conditional resolution bound, not a positive measurement.
- Available exports do not establish that the measured output and converted input
  are simultaneous device-terminal voltage and resistive-branch current. That
  unresolved circuit boundary limits interpretation of inferred power and R=V/I.
- The 110 MHz ceiling is an operational display/eligibility screen, not a universal
  VO₂ physical law. Temperature calibration and timestep checks are separate gates.
- At constant current with deterministic dynamics, scaling C and Cth together
  preserves the trajectory path and amplitude while rescaling time. The absolute
  three-parameter oscillatory domain therefore need not close along that direction.
  The boundary search did **not** establish all physical boundaries; unresolved and
  calibration-censored regions must stay visible. See [BOUNDARY_SEARCH](BOUNDARY_SEARCH.md).

Numbers and conditional intervals are retained in [the parameter record](RESEARCH_HISTORY.md#4-parameters-distinguish-estimates-from-diagnostic-fits).
No new calibration was adopted during repository consolidation.

## Latest fitting method versus requested next work

On 27 October Gabriel confirmed that the fitted R(T) and oscillation data use
the same electrode pair, with uniform stage heating for R(T). An optional
[filament/cold-surroundings experiment](FILAMENT_SURROUNDINGS_20261027.md) is
implemented for a controlled comparison. Its scientific trial is pending;
implementation tests are not evidence of improved agreement or a new calibration.

An archived alternating-amplitude example was recovered in the 7 September
relaxed simulation at source label 400 mV (~303 µA): primary-cycle Vpp alternates
near 239/353 mV. The short measured-input window survives two finer timesteps;
a 2 µs constant-current control becomes irregular at the finest step. This is
a documented numerical phenomenon, not an established stable period-two state,
bifurcation or measured device result. No calibration is adopted. See
[waveform, exact settings and limits](ALTERNATING_AMPLITUDES_20261025.md).
The vector is frozen under `data/parameters/alternating_amplitude_relaxed_20260907.json`.
44 classic GIFs now replay all 22 measured inputs at two displayed timesteps.
Slide 22 gives tau_th=11.91 ns and central tau_e=0.13–12 ns, with the proposed
electrothermal/hysteresis-memory explanation explicitly qualified.

A separate new-sample M2 1600 nm exploration is complete: the 374-row static
loop fits with log10 RMSE 0.08065, failing the usual 0.05 gate. Four labeled
diagnostic grids (773 entries, explicit 0.10 static exception) and a 36-point
finer pilot retain no oscillators inside the measured 289.98–370.1795 K domain
at 500 µA and 298 K. Computed excluded f/Vpp maps are available for inspection;
they are not accepted predictions or proof of no oscillatory domain. No physics
or calibration was changed. See [M2 study and limitations](M2_RT_EXPLORATION_20261021.md).
The requested expansion adds 1,843 wider/denser grid entries (2,616 cumulative,
including overlaps), still with no retained oscillator. All 24 finer startup
checks of 12 nearby settings exceed the measured domain; this is not a finely
verified physical boundary or an exhaustive absence result.

The completed 20 October comparison fixes the six R(T) parameters and T0 and
varies only gamma, Se, C and Cth. It compares squared relative frequency error
against frequency plus Vpp error, using actual measured inputs (~228–606 µA).
Both searches completed 250 unique evaluations and three-timestep all-current
replay. At 0.00625 ns, frequency-only passes target cycle/frequency checks on
10/11 records, combined on 8/11; Vpp MAPE remains ~1966% and ~903%, respectively.
The combined count changes with timestep, and both full replays leave the
configured resistance temperature domain on controls. Neither is a converged
quantitative fit or adopted calibration. See [methods/results and bundles](LOSS_COMPARISON_20261020.md).
The [44 all-input animations](LOSS_ANIMATIONS_20261021.md) show measured current,
measured/simulated output and native simulated R(T) for these two retained vectors.
They reproduce the saved finest-step voltages; no optimization was repeated.

A follow-up checked local finite differences at three timesteps, expanded both
searches to four starts, and tried short L-BFGS-B refinements. Full-loss
finite differences were undefined at the original candidates because some
frequencies were missing; individual frequency profiles often showed plateaus.
Gradient trials used the finite rejection score and reached their short
line-search limit. Their sampled improvements are not converged gradient fits.
Both larger searches completed 765 candidates and retained the previous
frequency-only diagnostic. Combined selection follows failed-target priority,
even though amplitude error increases. No eligible complete fit was found.
See [optimizer methods and results](OPTIMIZER_AUDIT_20261020.md).

The earlier equal-weight extrema trial is preserved in
[SIMPLE_EXTREMA_LOSS_20261003](SIMPLE_EXTREMA_LOSS_20261003.md). It also found no
eligible all-target fit. Neither trial is a resistance-extrema fit.

Meeting direction 1 (frequency-only versus frequency-and-amplitude fitting) is
completed as a bounded diagnostic. Direction 3 (a coupled gamma sweep) is now
also completed at three prespecified vectors over gamma=0.70–0.95627; the effect
on f/Vpp is modest there, without identifying a specimen gamma. See the
[fresh controlled trend checks](FRESH_PARAMETER_TRENDS_20261020.md).
The following directions remain **planned, not completed**:

2. Compare apparent resistance maxima/minima only after documenting the current
   split/channel assumptions. `V/I` is not automatically the device resistance
   when capacitive current or readout transfer is significant.
4. Finish a source-backed Brown/Pollner comparison, keeping device geometry,
   circuit capacitance, thermal time and effective-volume assumptions distinct.
   Preliminary ranges discussed in chat are not adopted sample bounds.
5. Assemble one concise supervisor-facing PDF after the scientific scope is
   agreed. No new consolidated meeting PDF has been completed.

The full historical 1,258-GIF map was reviewed using saved finer replays:
[controlled tendencies and scaling evidence](PARAMETER_EXPLORATION_TRENDS_20261020.md).
Both capacitances slow oscillations, but their amplitude tendencies differ.
Sixteen saved points illustrate the common-capacitance inverse-frequency scale
law. This map fixes gamma and 298 K; it is not a fit to the 314.4 K measured
sweep. The presentation now has 21 slides, retaining only the frequency-only and
frequency-plus-amplitude loss explanations and results. Older loss formulations
and the optimizer follow-up remain archived rather than shown. A final slide
introduces the existing three-parameter heatmap/GIF explorer with its fixed
conditions stated; no new map simulation or fit was performed.

Fresh 20–21 October simulations verify the common-capacitance scale relation
on 12/12 final comparisons across three anchors, using timestep and horizon
checks plus an independent frequency measurement. In the original 500 µA,
298 K map region, all 15 usable C/Cth/Se comparisons lower frequency when the
parameter increases. Capacitance-induced amplitude changes are often small;
Se-induced increases are clearer. The same trends are not established at
350 µA or 314.4 K because too few adjacent endpoints are verified oscillators.
610 distinct integrations cover 75 physical settings; no calibration is adopted.
Short-window and coarse-overshoot exclusions must not be called physical
boundaries. Full pointwise waveform convergence at fixed dt is not established.
See [methods, results and exceptions](FRESH_PARAMETER_TRENDS_20261020.md).

The six frequency/amplitude slice maps and fixed-across-slider axes/color limits
already exist; do not rebuild their physics to improve their appearance.

For continuity only, the last chat floated working priors `Se = 2–10 µW/K`,
`tau_th = 5–30 ns`, hence jointly `Cth = Se*tau_th` spanning `0.01–0.3 pJ/K`,
and `C = 0–0.4 pF` under the conditional timing interpretation. These are
**unadopted proposals to audit**, not measured intervals or verified values
extracted from Brown/Pollner. A few-pF circuit diagnostic is a separate hypothesis.
Do not silently turn these into independent fitted parameter bounds.

## Recommended next research step

First confirm the Figure 6 schematic, probe nodes, current conversion, channel
delay, and a known-resistor control with Amir/Yoav. Until available, work on one
bounded, interpretable diagnostic at a time. The bounded coupled gamma and
parameter-trend checks are now complete. Match the operating conditions,
acquisition/feature extraction and circuit boundary
before interpreting another parameter search as a specimen fit.

An improvement must persist across low/mid/high-current records, retain stable
controls, and survive timestep refinement. Report frequency, amplitude, mean and
persistence separately; a lower scalar loss is not by itself physical validation.

## Repository consolidation and disclosure

The canonical repository is `gabebere/single-neuristor`; do not create another
project repository. On 20 October Gabriel approved public experimental numerical
data and our generated results, but explicitly prohibited redistributing
third-party paper/manuscript PDFs. Private copies must stay local and out of releases.
Historical public commits already contain some paper PDFs; deleting them from a
new checkout does not erase that history. No force-push/history purge is authorized.

Consult the [consolidation record](https://github.com/gabebere/single-neuristor/blob/12be9a9a729e03b164464fad0daab7c1b9d4f4c4/docs/CONSOLIDATION_20261020.md) for backup,
verification and actual GitHub status; do not infer remote backup from a local
commit or catalog. This cleanup does not introduce a new physical calibration.

The consolidation is merged through [PR #4](https://github.com/gabebere/single-neuristor/pull/4)
into `main`, with reviewed code checkpoint `b46f76f`. Both large-map assets are
[published](https://github.com/gabebere/single-neuristor/releases/tag/research-library-2026-10-20).
Final checks: **174 tests** locally/fresh checkout, independent Linux Python
3.10/3.13 checks, validation, real control/sample-study recipes and successful
large-bundle reconstruction. These are software/reproducibility results, not a
new experimental fit.

The [26 October layout migration](REPOSITORY_LAYOUT.md) consolidates all jobs
and outputs under `runs/`, with one root README for collaborators. The checkpoint
before relocation is `b2ee2ff`. Bundle IDs and scientific contents are preserved;
this is an organizational change, not a new research result.

Unused archives, import wrappers and older document assemblies were pruned on
26 October. Scientific records, measured data, supported commands and completed
bundles are preserved. Use [the recovery map](REPOSITORY_LAYOUT.md) for retired
files; the latest organization checkpoints remain local until explicitly pushed.

Streamlit interfaces were retired on 26 October; simulation and replay use the
existing CLI and TOML recipes. The local run manager, exported HTML and
presentation viewers remain available. Scientific conclusions are unchanged.
