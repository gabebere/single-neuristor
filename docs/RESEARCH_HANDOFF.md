# Research handoff

Last reviewed: **20 October 2026**. Read this, the root README, and AGENTS.md before
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
| Plain-language navigation | [START_HERE](../START_HERE.md) |
| Run and inspect a new sample | [SAMPLE_STUDY](SAMPLE_STUDY.md) |
| Authoritative physics/code responsibilities | [CODEBASE_MAP](CODEBASE_MAP.md), [AGENTS](../AGENTS.md) |
| Experimental inputs and units | [Data guide](../data/README.md) |
| Reports, methodology, presentations | [Document guide](README.md) |
| Current versus superseded evidence | [ARCHIVE_INDEX](ARCHIVE_INDEX.md) |
| All local runs and files | [Artifact library](../artifacts/index.html) |
| Paper claim → inputs → command → outputs | [Reproduction guide](../paper/README.md) |
| Public/private material and GitHub release rules | [DATA_POLICY](DATA_POLICY.md) |

Keep existing completed bundle IDs, contents, and relative artifact paths intact.
Local publication (`runs publish`) is archival copying, not GitHub upload or
scientific acceptance. Do not regenerate the entire research history to orient
yourself.

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

The implemented simple trial uses settled complete-cycle voltage extrema and
mean squared relative **frequency and Vpp** errors with equal weights, not
point-by-point phase matching. See [the exact objective and results](SIMPLE_EXTREMA_LOSS_20261003.md).
It found no eligible all-target fit. Do not call it a resistance-extrema fit.

The following meeting directions remain **planned, not completed**:

1. Try frequency-only optimization first, retaining amplitude and extrema as
   out-of-objective validation. Specify windows, missing-cycle handling, stable
   controls, and timestep checks before running it.
2. Compare apparent resistance maxima/minima only after documenting the current
   split/channel assumptions. `V/I` is not automatically the device resistance
   when capacitive current or readout transfer is significant.
3. Hold one genuinely oscillating parameter vector fixed and vary gamma in the
   freely evolving coupled model. Existing gamma GIFs prescribe temperature;
   they do not measure gamma's effect on oscillator frequency.
4. Finish a source-backed Brown/Pollner comparison, keeping device geometry,
   circuit capacitance, thermal time and effective-volume assumptions distinct.
   Preliminary ranges discussed in chat are not adopted sample bounds.
5. Assemble one concise supervisor-facing PDF after the scientific scope is
   agreed. No new consolidated meeting PDF has been completed.

The six frequency/amplitude slice maps and fixed-across-slider axes/color limits
already exist; do not rebuild their physics to improve their appearance.

## Recommended next research step

First confirm the Figure 6 schematic, probe nodes, current conversion, channel
delay, and a known-resistor control with Amir/Yoav. Until available, work on one
bounded, interpretable diagnostic at a time. A controlled coupled gamma sweep is
more informative than another unconstrained many-parameter search.

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

Consult the [consolidation record](CONSOLIDATION_20261020.md) for backup,
verification and actual GitHub status; do not infer remote backup from a local
commit or catalog. This cleanup does not introduce a new physical calibration.
