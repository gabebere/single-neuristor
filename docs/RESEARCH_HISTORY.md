# Research history: VO2 current-driven simulations (through 16 October 2026)

This is the preserved detailed research log, not the current task list. Start with
[the concise handoff](RESEARCH_HANDOFF.md). Dated statements below describe their
own checkpoints; later entries and the current handoff supersede older claims.

## All-input loss animations (21 October 2026)

[44 stacked GIFs and method](LOSS_ANIMATIONS_20261021.md) cover all 22 inputs for
each retained objective. The fixed saved vectors are replayed at 0.00625 ns to
recover native R/T histories; complete saved voltage records must agree within
0.000001 mV before rendering. Both measured channels keep their acquisition grid.
No new optimization or calibration is performed. Later optimizer runs are excluded
at Gabriel's request. Parent bundles stay immutable. Bundle: `15921f`.

## Controlled frequency/amplitude comparison (20 October 2026)

[Method, results and three examples per objective](LOSS_COMPARISON_20261020.md).
The six fitted static values and T0 are fixed; only gamma, Se, C and Cth vary.
Matched starts, seed, bounds and evaluation cap; 250 unique candidates each.
Both scientific objectives and all 22 inputs are checked at three timesteps.
Frequency-only: 10/11 passing targets and 1965.7% Vpp MAPE at 0.00625 ns.
Combined: 8/11 and 903.0%, with classification changing under refinement.
Neither is a converged all-current fit; full-replay temperature checks also fail.
Measured current magnitudes replace filename labels on all presentation plots.
Reviewed parents: `543408` and `5ad999`; comparison `26574a`. Earlier comparison
`2feaaa` is retained but superseded for display (missing-cycle Vpp now uses gaps).
The presentation now explains the loss-function progression and shows three
current-ranked examples per fresh search. No new physical calibration is adopted.
180 tests pass; recipes and saved-sample comparison command complete.

## Project files consolidated from Desktop (16 October 2026)

Use [the consolidated Desktop guide](../archive/desktop_consolidation_20261016/index.html)
instead of the former `Desktop/VO2 Research Archive` location. All 926 content
files were accounted for: 914 exact duplicates remain at existing project paths,
and 12 unique historical notes/index/recipe files are preserved under `archive/`.
The original Desktop directory is recoverable in macOS Trash; its six symlink
targets were already inside this project and were left unchanged. Per-file
SHA-256, byte-comparison destinations, original link targets, and recovery
instructions are retained with the guide. No data, physics, fit conclusions, or
completed public bundles changed. Older Desktop references in dated entries
below are historical; no new research result or GitHub push is implied.

## Adaptive coarse-to-fine boundary exploration (8 October 2026)

Presentation work is paused at Gabriel's request. The immediate question is which
map edges represent transitions and which reflect finite search/time/numerical
limits. New command: `neuristor analyze boundary-maps --config
experiments/current/adaptive_boundary_maps.toml`. Read [the method](BOUNDARY_SEARCH.md).

The deterministic constant-current model has a common-capacitance time-scaling
symmetry: scaling C and Cth together preserves the V/T/R path and Vpp, multiplies
period, and divides frequency. This was numerically checked at three timesteps.
Thus the full three-parameter oscillatory domain need not close along this scale
direction. The new search explores C/Cth and Se, and reconstructs 3D scale rays
with display ends explicitly labeled. It does not alter either physics module.

Coarse shells expand at active/unresolved faces. Fine sampling checks every point
at three timesteps, retries disagreements at up to two extra halvings, extends
observation horizons, and verifies both sides of approximate-equilibrium/cycle
brackets. Fine shells also expand when better integration discovers activity at
an edge the coarse pass missed. No frequency cutoff enters the domain classifier.
Temperature/proximity exclusions and unresolved points remain visible.

Reviewed portable bundle:
`public_jobs/20261008_105134_adaptive-coarse-to-fine-oscillation-boundary-sea_504198`.
At 500 µA, T0=Tinit=298 K and reference Cth=0.072 pJ/K: 42 coarse points,
211 fine/refinement points, 57 verified oscillators, 100 approximate equilibria,
13 temperature-calibration exclusions, 35 time-unresolved and six numerically
unresolved points. There are 18 transition brackets, 15 meeting the configured
0.025-decade width. No verified oscillator touches a final sampled outer face.
The high-Se face has sampled equilibria; the high-ratio face is calibration-
censored; lower ratio/Se faces remain unresolved. **All physical boundaries have
not been established.** Expansion stopped after its declared number of rounds,
not because the lower unresolved faces were proven nonoscillatory.

The constant-current search uses the measured 289.8545–370.1625 K resistance
interval, not the older 5 K padding. Nonoscillation is an approximate balance/
quiet-span classification, with finite peak-detection and residual thresholds;
these are numerical transition brackets, not exact bifurcation proofs. Initial
state, static R(T), gamma, current and ambient remain conditional assumptions.
This is not a laboratory waveform fit or a new specimen parameter calibration.

`--reuse-exploration` permits denser/extended geometry only after matching SI
parameters, current, numerical controls, solver/extrema bytes and the evaluator/
classifier AST. The reviewed portable bundle reproduces its parent tables exactly
with zero new integrations and retains the full verification/source ancestry.
Its numerical parents are local completed `4d15ed` and `f123b6` bundles, with 621
and 986 integration/horizon records respectively. Source checks and current-source
hashes agree. Earlier failed/interrupted software trials remain in ignored runs.
Checks: 163 tests passed with four existing arctanh warnings; repository validation
passes. No Git commit/push or automatic follow-up is implied.

Next research question: distinguish critical slowing/long transients from numerical
precision or resolution limits in the lower unresolved region, and obtain justified
R(T) support before claiming a high-ratio physical boundary outside calibration.
Do not draw a closed surface through the censored/unresolved cells or infer
absolute C/Cth bounds from the reconstructed display scale.

## Comparable parameter-map slices (7 October 2026)

The current renderer keeps parameter axes and frequency/Vpp color limits fixed
across locked-variable slider positions. Metric limits cover finite retained
points in the same map, not the 110 MHz screening ceiling or excluded points.
This replaces the former per-slice rescaling for new viewers; equations,
screening, trajectories, point IDs and scientific interpretation are unchanged.

`neuristor analyze refresh-maps --all-existing` produced 13 display-only revisions
of completed historical maps/studies. Their numerical tables are byte-identical
to the sources; original GIFs are linked or embedded unchanged, without replay.
The artifact hub links each source to its updated viewer. Old completed HTML/ZIPs
remain historical exports, deliberately not edited in place. Dense refreshed
HTMLs link to original media and must not be emailed alone; study revisions are
standalone. See [display/reuse instructions](MAP_DISPLAY.md).

The fresh sample-study recipe `9f299d` confirms the future-run pathway. This is
software/display verification only, not improved experimental fitting.

## Organized library and paper-evidence navigation (7 October 2026)

Start with [the project directory](../START_HERE.html) or
[plain-language guide](../START_HERE.md). Implementation is grouped into analysis,
fitting, plotting, orchestration, interfaces, and legacy helpers; authoritative
physics paths and public commands are unchanged. Old imports alias the same
module object. The replay UI has an explicit rerunnable main entry point.

[Migration/verification record](STRUCTURE_MIGRATION_20261007.md) documents what
moved and what did not. The actual sample-study rerun reproduces the preceding
ec2d1e metrics, numerical tables, GIF and standalone HTML exactly. Both current
and voltage control recipes ran successfully; 134 tests passed with four existing
arctanh warnings. Physics file hashes are unchanged. Recursive source snapshots
now include the nested package; reused numerical source stays distinct from the
current rendering source. Existing data, completed bundles and documents are intact.

[Paper reproduction index](../paper/README.md) links five supporting evidence
items to inputs, commands and output contracts; final figure numbering and a
quantitative TIA calibration are still unresolved. `neuristor runs index` now
also generates a source inventory, with no new metadata database or media copies.
The local environment is recorded, but is not a portable complete dependency
lock. Wheel build/local-target import checks are not fresh-machine reproduction.
CI/citation/attribution and [release guidance](RELEASE_CHECKLIST.md) are prepared
locally. No Git commit/push, external archive, or new accepted scientific fit is
claimed by the organization work.

## Unified sample-study program and artifact hub (7 October 2026)

New entry point: `neuristor study plan/run --config
experiments/studies/sample_parameter_map.toml`. Read [the user guide](SAMPLE_STUDY.md)
and [module map](CODEBASE_MAP.md). One declared-unit numerical major R(T) loop is
fitted, then Cth/Se/C are sampled at the chosen constant ideal current, screened
using late extrema, full-history temperature/proximity, frequency ceiling and
two smaller timesteps. Every retained point gets a refined GIF. Gamma and all
dynamic constants remain explicit assumptions, not quantities learned from the
major loop. No dynamic laboratory fit or new physical calibration is claimed.

The default temperature domain is now the actual measured interval for **new
sample studies**, not the old padded map interval. Older recipes/plots remain
unchanged. The example fits 264 same-sample measurements with log10 RMSE
0.0366276052, tests 27 combinations and retains one oscillator at about
23.2153 MHz (finest map step); its GIF replay is about 23.2491 MHz and passes
feature/visual checks. One point is not an exhaustive domain search. Coarse
rejected points are not all finely verified.

Reviewed software demonstration:
`public_jobs/20261007_083908_sample-r-t-to-screened-map-and-embedded-gifs_ec2d1e`.
Open its `parameter_maps.html`: all GIFs and the fit image are embedded in one
approximately 5.8 MB HTML. **No ZIP is generated by the sample-study command.**
Sharing that HTML alone works without companion media; the complete parent
folder retains raw data, numerical checks, resolved recipes and frozen source.
Embedding expands media by roughly one third and can exceed email limits for
dense grids. The exporter streams one GIF at a time to limit Python memory use.
This does not change the old eaf6ab viewer, which still requires its GIF folder.

`neuristor runs index` rebuilds `artifacts/index.html`, runs.csv and files.csv
from canonical manifests/files. It introduces no database, duplicate media or
historical deletion. Categories indicate workflow, not scientific credibility;
completed/public are generation/local-archive status, not accepted-fit/GitHub
status. Existing Desktop archives stay intact. The first failed embedding trial
is retained as failed evidence, with its completed numerical stages and explicit
failure page; later successful verification runs are local software trials.

Refactor: public workflow API retained; cohesive map/media functions moved to
`map_workflows.py`, complete study coordination to `sample_study.py`, catalog to
`artifact_catalog.py`. Physics was not changed. The concise root README links
the user guide and per-module map; the old full README is preserved in
WORKFLOW_REFERENCE.md. No Git commit or push was made. Browser hit-testing has
not been manually verified; automated point/slider DOM logic and embedded-byte
checks plus actual fit/grid/GIF recipe execution provide the current software evidence.
Verification: 106 tests pass (four existing arctanh warnings); repository
validation passes for 31 recipes and 109 distinct archived records. The actual
published example has all manifest artifacts, no ZIP, and one 32-frame embedded
GIF whose bytes match the original exactly. No external script/image dependency
is required by the standalone HTML. The generated catalog currently indexes
13,964 produced files; refresh it after subsequent runs/publication.

## Parameter-map GIF generation and viewer correction (7 October 2026)

Full generation completed: 1,258 GIFs, 1,256 refinement passes and 1,174 visually
resolved paths. 84 unresolved paths have explicit line breaks. Original 81ae34
viewer had a point-customdata representation mismatch reported by the user.
Corrected complete viewer: `runs/20261007_074702_portable-screened-parameter-map-with-point-gifs_eaf6ab`.
All media are reused byte-for-byte without resimulation; ordinary-list metadata
and typed/missing-event-data fallbacks fix point selection. All 1,258 point IDs
and all 69 slider slices resolve to existing media. Replace only the HTML in an
existing extracted folder, or use the corrected complete ZIP. Old bundles remain
immutable. See the linked methods review for tests and caveats.
Correction checks: 93 tests passed (four historical arctanh warnings), repository
validation and ZIP integrity passed. Corrected eaf6ab is copied to public_jobs/;
all point/slice identities were checked against media. No GitHub push was made.

The following launch checkpoints are historical, not current process status.

User approved portable ZIP: right-click 3D/heatmap points to play synchronized
constant-input / voltage / R(T) GIFs. Read [methods and live locator](PARAMETER_MAP_GIFS_20261007.md).
No matching experimental trace exists at constant 500 µA, 298 K; no lab voltage
is overlaid. Original map data remain immutable. Initial dt 0.00625 ns, halved
up to twice for feature/visual resolution checks. Full integrated R(T) samples
are rendered; unresolved jumps break lines rather than inventing smooth paths.
3-point benchmark f34881 passed checks; full job:
`runs/20261007_061243_portable-screened-parameter-map-with-point-gifs_81ae34`.
Check status.json / run.json before calling it complete. Recipe:
`experiments/current/physical_parameter_map_gifs.toml`; command `analyze map-animations`.
Delivery checkpoint: 204/1,258 generated; approximately 39-minute projected total.
93 tests / repository validation passed; four historical arctanh warnings remain.
The completed f34881 benchmark is published locally and copied into Desktop
`VO2 Research Archive/04_Parameter_maps/`. No GitHub push or automatic follow-up
is scheduled. Full-job state must be read from status.json, not this snapshot.

## Simple relative-squared extrema trial (3 October 2026)

Read [the method/results review](SIMPLE_EXTREMA_LOSS_20261003.md).
Recipe `specimen_steady_relative.toml` removes tolerance scaling, logarithms,
amplitude offsets and the static weighted objective. Scientific loss is mean
squared relative frequency and Vpp errors. Static R(T), temperature, proximity
and sustained <=110 MHz cycles are separate eligibility checks. Finite optimizer
rejection sentinels are not the scientific objective. Historical recipes unchanged.
85 candidates / all nine final verification batches; no eligible all-target fit.
Reviewed search: `public_jobs/20261003_183555_simple-relative-squared-extrema-fitting-pilot_96b944`.
Best failed attempt diagnostic_1 sustains 8/9 targets, passes frequency screen
5/9 at 0.00625 ns, and has 973.4% average amplitude error. Missing frequency
errors are null; averages of defined rejected-cycle losses are diagnostic only.
Static RMSE 0.036628, proximity slope +0.24467; domain checks pass.
Finest-step GIF recipe: `specimen_relative_diagnostic_replay.toml`.
Reviewed GIF bundle:
`public_jobs/20261003_183956_simple-loss-diagnostic-1-all-current-gif-replay-_4e79ac`;
22 GIFs and 22 PNGs, 48 frames each, verified readable. Same-vector replay agrees
with search voltage samples within 0.0093 mV. Full replay temperatures 317.46–364.07 K.
Desktop original results moved recoverably into `VO2 Research Archive`, organized
by fitting trials / parameter diagnostics / historical references with START_HERE.md.
Two previously linked historical results additionally have real frozen copies.
Historical inventory records hashes/counts and symlink targets; new Desktop
copies and the two frozen historical copies were verified against canonical files.
87 tests and repository validation passed (four existing arctanh warnings).
No new physical calibration is adopted. No GitHub commit or push was made.

## Meeting extrema protocol and screened parameter maps (2 October 2026)

Latest HTML viewer:
`public_jobs/20261002_152003_focused-dense-physically-screened-steady-state-e_c1772e/parameter_maps.html`.
It has independent Cth/Se/C sliders over all 23 sampled values per axis. Each
slice updates frequency and Vpp maps with independent visible-data color ranges
in MHz/mV. Empty slices are labeled; excluded cells stay blank. Numerical CSVs
are unchanged. This supersedes the common color range for the interactive viewer;
the previous immutable artifact is preserved below.
All 69 slices retain every eligible point after tolerant indexing of CSV float
coordinates. Initial slider rendering `2bffd0` is superseded by this correction.

Read [the meeting review](MEETING_PARAMETER_MAPS_20261002.md). New voltage fitting
uses complete-cycle maxima/minima in the late window; there is no phase or
point-by-point voltage error. Historical recipes retain their spectral method.
The independent static R(T) constraint is unchanged. Recipe:
`experiments/current/specimen_steady_extrema.toml`.

The old binary 15^3 map cannot be frequency/temperature filtered. Fresh broad and
focused maps were generated using the supported ideal-current model, at 500 µA
and T0=Tinit=298 K. The focused 12,167-point grid has 1,258 retained oscillators
after persistence, <=110 MHz, configured resistance-domain, sampled proximity
and two timestep refinement checks. Its HTML includes the 3D view and six fixed-
parameter frequency/amplitude heatmaps. Reviewed bundle:
`public_jobs/20261002_145342_focused-dense-physically-screened-steady-state-e_b9e298`.
Maximum displayed temperature is 372.188 K, inside the padded implementation
domain but slightly above the measured static R(T) maximum 370.1625 K. This is
conditional screening, not a calibrated physical parameter set. Broad grid:
`public_jobs/20261002_135437_dense-physically-screened-steady-state-extrema-p_9ac3ad`.
Recipes: `physical_parameter_maps.toml` and `physical_parameter_maps_focused.toml`
under `experiments/current/`. `--reuse-numerics` preserves source CSV hashes.

The reviewed bounded extrema pilot evaluated 79 candidates but found no acceptable
match on nine measured target records. Bundle:
`public_jobs/20261002_145243_meeting-extrema-only-physically-screened-fitting_32d91d`.
At the finest step no searched target has an accepted frequency; the 100% frequency
error column is the missing-frequency penalty, not a valid measured discrepancy.
No new fit has been adopted. The existing
measurement/circuit-boundary uncertainty remains unresolved; the filtered
constant-current map at 298 K is not a replay of the finite laboratory pulses.
Checks: 83 tests passed and repository validation passed. Bundles are local
tracked-directory copies; no Git commit or push was made in this task.

## Requested 500 uA, 298 K diagnostic (23 September 2026)

Recipe `experiments/current/500uA_ultralow_Cth_298K.toml` applies the user-provided
`C_th = 1.00e-4 pJ/K`, `S_e = 2.68e-3 mW/K`, `C = 16.4 pF`, `T0 = Tinit = 298 K`
at an ideal 500 uA input. Unspecified resistance and hysteresis values use eef073.
At `dt = 0.001 ns`, the result oscillates at 66.133 MHz with 0.37017 V peak-to-peak
amplitude; halving the step gives 66.162 MHz and 0.37005 V, so these displayed
features change by less than 0.05%. However, the simulated temperature reaches
about 2494 K, far beyond the resistance calibration ceiling of 375.16 K. The model
therefore evaluates its clamped out-of-domain resistance law for most of the hot
trajectory. Treat the close agreement with the supplied 67.24 MHz label as a
numerical reproduction, not a physically supported calibration.

## Standard replay animation output (23 September 2026)

Every `analyze replay-lab` run and every Simulation Lab run now automatically
writes synchronized three-panel GIFs for all 22 measured currents: imposed input
current in amperes, measured/simulated voltage, and the evolving simulated R(T)
trajectory over heating/cooling major-branch guides. Standard bundle entry points
are `START_HERE.html`, `All_current_GIFs.zip`, `GIFs/`, `Figures/`, and
`major_branches.csv`. `scope_frames` controls only animation sampling; numerical
integration and `traces.csv` are unchanged. The earlier two-panel cursor animation
is legacy opt-in. Pure current simulations can request a corresponding three-panel
GIF through `[output] scope_gif = true`; its middle panel contains simulated voltage
only. Replay tests verify 22 GIFs/PNGs and the three-panel aspect.

## Completed tanh pilot (23 September 2026)

See [pilot review](TANH_PROXIMITY_PILOT_20260923.md). Separate tanh kernel,
P=1-tanh(kx), k stored in gamma coordinate, was implemented and tested without
changing the default Yuanhang mode. A 113-candidate, three-start pilot using a
stricter persistence gate yields 9/9 sustained target traces at all three steps,
positive proximity slope, and static RMSE 0.042914. Frequency error is 16.93%,
but amplitude error is 646.98%; none is near-target. This is a mechanism lead,
not a fitted calibration. No background search remains running. 77 tests pass.
Recipe: `experiments/current/specimen_steady_tanh_pilot.toml`.
Run: `runs/20260923_061836_tanh-proximity-bounded-pilot-labels-300-to-700_11a3f7`.
Desktop: `VO2 Tanh Proximity Pilot`. Previous constrained sine best passes 0/9
under the same new gate; unconstrained sine best passes 7/9, but violates proximity.
Changing loss and kernel together prevents attributing fit differences to kernel alone.


## Completed constrained rerun (23 September 2026, Israel time)

User authorized a repeat with physically screened proximity behavior. PID 36174
started 22 September 21:07 UTC (23 September 00:07 Israel), nominal deadline
23:07 UTC (02:07 Israel). Eight restarts, same target/window/static constraint.
Gamma range 0.60–0.98 plus nonnegative finite dTeff/dT checked on the actual state
at every integration sample, including prehistory/transients. Failed candidates
cannot enter elites or final search winners. Historical references are retained
and their failed checks are reported rather than hidden. All-current verification
reports the condition again at three timesteps; inspect it before adopting a fit.
No trajectory equation changed. The audit preserves bitwise trajectories at
three steps. 74 tests and validation passed; constrained recipe smoke passed.
This does not guarantee all minor-loop memory/ordering properties.
Recipe: `experiments/current/specimen_steady_monotone.toml`.
Locator: `outputs/active_monotone_search.json`.
Frozen source: `outputs/monotone_background_20260922_210709/source`.
Run: `runs/20260922_210712_monotone-proximity-constrained-settled-oscillati_ffdabf`.
Desktop: `VO2 Monotone Proximity Search`. No automatic LLM follow-up is scheduled.
Read status.json; the local job needs no ChatGPT calls. STOP file ends it gracefully.


## Cooling-hook diagnosis (22 September 2026)

The completed steady search evaluated 5,301 candidates and all 18 verification batches.
At 0.00625 ns, `search_1` has mean frequency error 14.9%, robust amplitude error
109.5%, static log10 RMSE 0.04999 and zero near-target traces. See
[the cooling-hook diagnosis](HYSTERESIS_HOOK_DIAGNOSIS_20260922.md) before adopting it.
The fitted gamma 0.102 (and eef073 gamma 0.181) makes the LLP effective-temperature
mapping locally nonmonotone. A cooling resistance drop is reproduced by upstream
code and analytic/double-precision derivatives. This is a missing minor-loop
admissibility screen in the broad search, not a local branch-sign bug. Major-loop
R(T) constraints do not restrict gamma. No physics or archived fit was altered.
Next bounded work: screen minor-loop shape, then compare a constrained refit;
do not continue unconstrained gamma search or assume its lower loss is physical.


## Background search launch record (completed; 22 September 2026)

User explicitly selected **original record labels 300–700**, not measured currents.
Detached PID 31846 started at 17:51 UTC (20:51 Israel), nominal deadline
19:51 UTC (22:51 Israel). No automatic LLM follow-up is scheduled.
Recipe: `experiments/current/specimen_steady_multistart.toml`; immutable launch
snapshot: `outputs/steady_background_20260922_205149`.
Run: `runs/20260922_175152_background-settled-oscillations-for-original-lab_f426de`. Live locator: `outputs/active_steady_search.json`.
Read `status.json` before assuming completion. Objective excludes all startup,
DC and phase errors; it targets 150–250 ns frequency, robust/fundamental amplitude
and late retention. Eight independent restarts; hard static log10 RMSE <=0.05.
Two fixed-major-loop starts plus six joint-static starts; final all-current checks
at 0.025, 0.0125, 0.00625 ns. No new result has been adopted. This explicitly
supersedes the earlier recommendation to avoid another broad fit for this user-
authorized, targeted experiment. Review fine-step summary and persistence before
claiming success; all target traces were training data. Desktop folder:
`VO2 Background Search`. Full test suite: 72 passed; validation passed.

Updated **22 September 2026**. The [expanded search and interactive laboratory](EXPANDED_SEARCH_20260922.md)
are complete. T0 was already free (308–322 K). A larger cached DE/Powell search
at 0.025 ns with persistence penalties recovers 7/11 oscillators with the stronger
static constraint and 6/11 with relaxed static weight at the finest step.
Late-mean voltage RMSE improves to 44.6/39.8 mV, but oscillating-record amplitude
errors remain large (127.9/130.6 mV); the prior anchored vector still recovers 9/11.
One stronger-static classification changes on final timestep refinement. No new
calibration is adopted. Use `neuristor playground` for editable all-current V(t)/R(t)
comparisons, bottom current slider, per-current GIFs and reproducible saved recipes.

The same stronger-static parameter vector also has a synchronized scope export
(recipe `experiments/current/specimen_scope_export.toml`) at 0.00625 ns: progressively
drawn V(t), imposed measured I(t), and simulated R(T) with major-branch guides.
Its 22-current replay confirms 7/11 persistence and 44.59 mV late-mean RMSE.
An offline viewer and all-result copy were delivered to the Desktop on 22 September.

User-selected visual reference: **eef073**, the exact 0.025 ns replay, rather
than the finer-step export. `neuristor analyze export-scope --source PATH_TO_RUN`
renders the saved samples without resimulation into vertically stacked I(t),
V(t) and evolving R(T) GIFs. Current labels and filenames are in amperes, using
the measured 50–250 ns plateau mean. Fine-step scientific caveats remain unchanged.

Previous update: Updated **21 September 2026**. The new
joint fit is now complete: see [budgeted joint inference](JOINT_INFERENCE_20260921.md).
It freed all six major-loop parameters plus gamma and four circuit/thermal
quantities in two searches totaling 240 objective calls. The static-preserving candidate
improves the validation feature score but recovers only 4/11 persistent records;
the relaxed-static diagnostic recovers 5/11 and worsens static R(T). Neither is
a replacement calibration. The requested mechanism/parameter-estimation deck
now includes the method and verified results.

The
[independent discrepancy audit](DISCREPANCY_AUDIT_20260921.md) confirms the
current-source mismatch at three timesteps, fixes a separate voltage-source
small-capacitance floor bug, and adds a frozen-major-branch stability calculation
and literature-guided parameter/measurement plan. Gabriel confirmed that the
static R(T) was measured on the **same device**; the full TIA schematic remains
unavailable. Device identity is now confirmed; uniform versus localized driven
heating and the terminal-channel map remain unresolved.
The LT1228's documented 5–6 pF output capacitance is a circuit-topology clue,
not a measured VO₂ parallel capacitance. No new physical parameter set is adopted.

Prior numerical bundle evidence through 14 September;
the eight-slide presentation was completed on 15 September. The 18 September
[channel/circuit audit](CHANNEL_CIRCUIT_AUDIT.md) adds a source-backed desk audit,
a controlled one-parameter replay, and a timestep-diagnostic correction. It
adopts no new parameter estimate.
The [experiment/model reconciliation](EXPERIMENT_MODEL_RECONCILIATION.md)
packages a paper-panel comparison for supervisor review. The public Gildor
preprint reports experiments and a physical interpretation, without a
reproducible numerical simulation; the July "paper-frequency analog" was
our own illustrative run. A mislabeled local experimental-PDF link formerly
opened the unrelated Almeida static-hysteresis paper and has been corrected.

## 1. Start here

**The software runs and the mechanism control oscillates. We have not obtained a
quantitatively satisfactory, physically supported fit to the experimental sweep.**
Do not restart from the historical claim that all 22 records were reproduced: that
was a permissive peak-count classification, not agreement in sustained dynamics.

Gabriel Berezovsky is complementing Amir Gildor's experimental work on *Harnessing
the VO2 Phase Transition for Automatic Gain Control in Transimpedance Amplifiers*,
supervised by Amir Gildor and Prof. Yoav Kalcheim. Both supervisors are already
closely involved. The target is one shared parameter set that predicts voltage
from measured current across all records, including the stable records, with
realistic mean voltage, amplitude, frequency and persistence.

Recommended reading order:

1. This handoff, especially Sections 5–8.
2. [Experiment/model reconciliation](EXPERIMENT_MODEL_RECONCILIATION.md):
   colleague-facing paper-panel comparison and assumption ledger.
3. [Eight-slide presentation](final_project/presentation/VO2_Fitting_Journey.pdf)
   and its [source/evidence map](final_project/presentation/README.md).
4. [Research report](final_project/Simulations_for_VO2_AGC.pdf), especially Sections
   4–6 for parameter assumptions and 9–12 for fitting and the latest diagnostics.
5. [Archive index](ARCHIVE_INDEX.md) for current versus superseded bundles.
6. [Root README](../README.md), [AGENTS.md](../AGENTS.md), and the relevant recipe
   before changing code. Read [architecture](ARCHITECTURE.md) for implementation.

The next recommended task is **experimental confirmation of the measurement-channel
and circuit map**, followed by a same-device thermal/dynamic check. See Section 8.

## 2. Files and data: what is authoritative

All paths in commands below are relative to the repository root.

| Item | Canonical location and meaning |
|---|---|
| Raw specimen R(T) | `data/experimental/100425_chip1_gap3.tsv`; professor-supplied major loop |
| Fitted resistance preset | `presets/resistance_100425_chip1_gap3.json`; values, intervals, diagnostics and fitted temperature domain |
| Raw oscilloscope exports | `data/experimental/tia_current_sweep/`; preserve unchanged |
| Experimental data provenance | The above directory's `README.md` and `SHA256SUMS` |
| Original reference implementation | `references/yuanhangzhang98-collective_dynamics_neuristor-217d4f0/` |
| Papers | `docs/final_project/references/` links to canonical local copies; the experimental manuscript link was corrected on 18 September to Gildor et al. arXiv:2604.04594v1, and Almeida's distinct static-hysteresis paper retained separately |
| Editable report / tracked PDF | `docs/final_project/main.tex` / `Simulations_for_VO2_AGC.pdf` |
| Editable slides / tracked PDF | `docs/final_project/presentation/VO2_Fitting_Journey.tex` / `.pdf` |
| Mechanism and parameter-estimation deck | `Simulations_on_VO2/main.tex` / `main.pdf`; the July mechanism talk extended with one evidence slide per specimen parameter and the frozen forward check |
| GIF and supplementary figures | `docs/final_project/supplementary/`; some figures link into immutable bundles |
| Reviewed numerical evidence | `public_jobs/`; each new-style bundle has a manifest, resolved inputs, metrics, report, tables and figures |
| Scratch work | Ignored `runs/`; never the only copy of an important result |

The professor supplied `Data.zip` on 17 August. Its 22 `_converted.csv` files were
used for Figure 7; the three XLSX workbooks were used for Figure 6. Converted files
have **no header** and contain time in **ns**, current in **µA**, voltage in **mV**,
with 1000 samples each. The workbooks preserve conversion information and original
channel voltages; inspect these before assuming what a channel physically measures.
The second workbook time column is shifted by 150 ns for presentation, not by a
physical channel delay. Do not combine that display shift with alignment corrections.

Source labels such as **250 mV** and **300 mV** are record identifiers, not imposed
sample current and not demonstrated unit mistakes. The corresponding measured
current steps are about **189.6 µA** and **228.2 µA**. Model input is the actual
measured current waveform. Different averaging windows/preprocessing explain small
label differences such as 606.3 versus 606.7 µA; they do not identify different runs.

Current import code is `src/neuristor/experimental_waveforms.py`. Former screenshot
digitization and image-derived fitting artifacts were removed from the active
codebase. Do not recreate data from paper images when the numerical exports exist.
Original Downloads paths are historical provenance, not runtime dependencies.

## 3. Model and implementation boundaries

The Yuanhang voltage-source circuit is authoritative in `src/neuristor/model.py`.
Our supported ideal-current extension is `src/neuristor/current_drive_sim.py`:

\[
C\dot V=I(t)-V/R(T,\mathcal H),\qquad
C_{\rm th}\dot T=V^2/R(T,\mathcal H)-S_e(T-T_0).
\]

Here the resistance law carries hysteresis history, denoted by H. Important rules:

- `C_F = 0` is the exact algebraic limit `V = I R(T)`, not an arbitrarily small C.
- The metallic steady voltage floor is approximately `I Rm`. Reducing capacitance
  changes dynamics; it does not independently raise that floor.
- A series resistor cannot change an imposed ideal current. Real source compliance,
  contacts, readout impedance or circuit dynamics require an explicit circuit model.
- Preserve float32-faithful hysteresis, initialization and reversal ordering. A
  seemingly harmless precision change can change a trajectory. Read
  [the hysteresis audit](HYSTERESIS_IMPLEMENTATION_AUDIT.md) before modifying it.
- Keep temperature-domain checks visible in technical diagnostics; do not quietly
  move a preset's limits to make a trajectory pass. Excursions are not calibration.
- Physics modules use SI unless a field name explicitly says otherwise. Boundary
  conversions matter: `1 pF = 1e-12 F`, `1 pJ/K = 1e-12 J/K`,
  `1 mW/K = 1e-3 W/K`, and `1 ns = 1e-9 s`.

The presentation layer must not contain a second copy of the scientific equations.

| Code | Role |
|---|---|
| `model.py`, `current_drive_sim.py` | Authoritative physics and hysteresis |
| `resistance_custom_analysis.py` | Major-loop fitting and resistance analysis |
| `experimental_waveforms.py` | Numerical exports, normalization and experimental summaries |
| `lab_estimates.py` | Conductance/capacitance estimation; conditional hysteresis reconstruction |
| `parameter_inference.py` | Shared-parameter search and historical objective |
| `model_validation.py` | Frozen-model comparisons and sensitivity studies |
| `oscillation_audit.py` | Windowed persistence, frequency/features, controlled maps and candidate checks |
| `workflows.py` | Unit-aware orchestration and standard run bundles |
| `visualization.py`, `plots.py` | Figures, not independent scientific models |
| `cli.py`, `dashboard.py` | Commands and read-only archive browsing |
| `experiments/*.toml` | Reusable user-facing inputs; resolved bundle configuration is authoritative for a particular run |

## 4. Parameters: distinguish estimates from diagnostic fits

### Specimen baseline, not a successful oscillation fit

| Quantity | Adopted value | Evidence and limitation |
|---|---:|---|
| R0 | 0.798923 Ω | Arrhenius prefactor, **not** the insulating resistance itself |
| Ea/kB | 2536.965 K | Major-loop fit; Ea ≈ 0.21862 eV |
| Metallic Rm | 18.2153 Ω | Major-loop fit; interval 17.4943–18.9010 Ω |
| Tc | 333.4936 K | Major-loop center; interval 333.1369–333.8383 K |
| Width w | 6.88230 K | Major-loop fit; interval 6.62905–7.26815 K |
| Beta | 0.299156 K⁻¹ | Major-loop fit; interval 0.280917–0.320206 K⁻¹ |
| Gamma | 0.956269682 | Borrowed Yuanhang minor-loop value, not measured on this specimen |
| Ambient T0 | 314.4 K | Adopted related-measurement range 314.25–314.55 K; verify applicability to this sweep |
| Environmental Se | 3.67513 µW/K | Conditional interval 3.43444–4.08505 µW/K |
| Electrical C | 0.39 pF | Conservative timing-resolution upper bound, not a positive measurement |
| Thermal Cth | 0.0478732 pJ/K | Conditional interval 0.021918–0.092624 pJ/K |
| Thermal time Cth/Se | 13.026 ns | Fit to moderate nonswitching heating edges |

Use full precision from the preset and recipes for computation, not rounded table
entries. The insulating branch includes `R0 exp[(Ea/kB)/T]`; R0 smaller than Rm
does not mean the insulating resistance is smaller than the metallic resistance.
In code `Rm = Rm0 * Rm_factor`; the report intentionally gives actual Rm directly.
The major-loop fit has log10 RMSE 0.03663 and R² 0.99873. Its good static fit does
not establish dynamic minor-loop behavior. Gamma needs minor-loop/dynamic evidence.

The inference order was R(T) and ambient conditions, then Se, then electrical C,
then Cth, then forward comparison. Thermal quantities are correlated and conditional:

1. **Se:** select the last settled, non-oscillating record before onset; baseline
   correct; infer temperature from its resistance on the fitted heating branch;
   use `Se = P/(T-T0)` from approximately zero dT/dt. This assumes the channel
   ratio is device resistance and static thermometry transfers to driven operation.
2. **C:** the original **19.8 pF estimate was rejected**. The shortcut I/(dV/dt)
   misassigned predominantly resistive current to charging and used a source-limited
   edge. The full balance is `C dV/dt = I - V/R`. Present traces do not resolve a
   positive C; the constrained fit reaches zero. The 1 ns timing scale divided by
   about 2.570 kΩ gives 0.39 pF. Zero is not a measurement of physical zero C.
3. **Cth:** use `I_R = I-C dV/dt`, `R=V/I_R`, `P=V I_R`; fit thermal heating for
   source labels 100/150/200 mV, 15–35 ns, then `Cth=Se*tau`. The 250 mV record
   is a near-transition sensitivity check, not part of that shared fit.

Recorded pulse width is 300 ns, rise time 26–27 ns (10–90%), fall time 18–23 ns
(90–10%). Replay measured input edges; do not replace them with ideal steps when
comparing to the experiment.

The **Yuanhang mechanism control is different**: at 600 µA, Cth was increased
fourfold from 49.6278 to 198.5111 pJ/K to sustain oscillations; other Yuanhang
parameters were retained, including much larger Rm and C. It gives about 0.222 MHz
and a 0.906–6.333 V steady range. This is neither a specimen parameter estimate nor
a claim that the unmodified upstream voltage circuit reproduces this experiment.

## 5. Trial history and what it actually established

Short bundle suffixes below resolve to full directories in [Section 10](#10-evidence-register).

| Stage | Intervention / question | Result | Interpretation |
|---|---|---|---|
| Mechanism control, 6765e0 | Yuanhang-based ideal-current run, Cth ×4 | Stable nonzero-valley oscillation | Implementation/mechanism check only |
| Sample R(T), 0849a9 | Fit measured heating and cooling major loop in log resistance | Accurate static fit | Does not determine gamma or dynamic switching |
| Frozen prediction, eefab7 | Replay all 22 measured currents using baseline estimates | Stable 189.6 µA mean within 0.67 mV; 0 of 11 measured oscillators recovered | Cold-side agreement, dynamic mismatch |
| Constrained global fit, 8f12d6 | Fit eight shared parameters inside supported bounds | All-trace objective improved 0.72%; still no oscillators | No rescue within this search |
| Relaxed global fit, 8f12d6 | Expand bounds and remove prior | Objective improved 22.5%, initially held-out objective 14.9%; 7/8 parameters outside intervals; no coherent cycles | Better errors via transients, not a physical calibration |
| Oscillation-priority, ac1c5e | Strong detection penalty, broad bounds | 21/22 historical labels; all eight fitted quantities outside prior ranges | Oscillation reward can produce cycles at unrealistic values |
| Amplitude-priority, 85526c | Increase absolute/relative amplitude penalties | Smaller voltage excursions; 19/22 historical labels; seven quantities outside ranges | Amplitude/classification trade-off |
| Physics-anchored, 717797 | Anchor Se, T0, Cth, Tc, w, beta; broaden C and gamma | 22/22 historical labels; C=6.8355 pF, gamma=0.15696; excessive amplitude/frequency | Not a sustained 22/22 success; later corrected |
| Persistence audit and grid, 26ff13 | Four windows; 126 C/tau/gamma combinations at three currents | No satisfactory shared candidate; see below | Shows conditional trade-offs, not impossibility over all parameters |
| Inverse consistency test, fa4b66 | Infer T from measured power, replay hysteresis without optimizing voltage | At about 607 µA, measured R≈298 Ω versus replayed 26–34 Ω across four gamma values | Conflict among assumptions; no unique cause identified |

The original global fit used 17 training records and five initially withheld source
labels (200, 400, 600, 800, 1000 mV). Later work inspected these data. Subsequent
all-current checks are not pristine blind validation. The anchored fit withheld
two stable settings (50 and 1000 mV) as negative controls, not a new blind test.
Different objectives have different weights: do not compare their raw loss values
as though they were a single metric. Finite search budgets do not establish global
optimality or mathematically rule out every parameter vector.

### The amplitude comparison has two different denominators

For comparison on the **same 11 measured oscillators**, mean absolute Vpp errors
are 289.6 mV (oscillation-priority), 81.2 mV (amplitude-priority), and 212.5 mV
(anchored). These are the presentation's numbers. Compute them from each bundle's
`trace_metrics.csv`, selecting `fit_mode == relaxed` and `measured_oscillation`,
then averaging `abs(predicted_vpp_mV - measured_vpp_mV)`.

Older manuscript values 280.1 and 88.5 mV restrict the average to **jointly detected**
experimental/model oscillators, changing the subset between fits. They are not
arithmetic typos, but should not be used as an unqualified same-population comparison.
The presentation README explicitly records this difference. The manuscript wording
has not yet been harmonized; clarify the denominator when editing it next.

## 6. Latest audit: how to judge an improvement

The previous detector accepted regular early peaks. At 606 µA the anchored model's
raw Vpp falls from about 50.2 mV in 50–100 ns to 0.49 mV in 200–250 ns, while the
experiment remains near 10–12 mV. The legacy detector's 8 ns peak spacing also
limited high-frequency discrimination; the new spectral/peak checks avoid it.

`oscillation_audit.py` uses four windows: 50–100, 100–150, 150–200 and 200–250 ns.
Its nominal persistence criterion requires all of:

- fitted periodic Vpp of at least 6 mV in every window;
- final/first robust (5–95%) voltage-span ratio of at least 0.5;
- late dominant-component fit explaining at least 40% of detrended variance.

Late frequency uses 150–250 ns and a 10–200 MHz search band. A 100 ns record has
about 10 MHz Fourier-bin spacing; interpolation does not add measurement resolution.
Thresholds are operational definitions, not physical constants. Sensitivity tests
give 9–13 measured detections; nominal settings give 11. A finite record is not proof
of an asymptotic limit cycle. Raw Vpp, robust Vpp and sinusoidal Vpp are different
quantities: always state the definition, time window and current subset.

Retain the full pre-pulse input history when simulating and crop only for analysis.
The audit recipe includes 296 ns of prehistory; the inverse reconstruction starts
its thermal integration at -200 ns. Initializing the state at the start of a late
comparison window would be a different experiment.

The 126-point map varies C, tau=Cth/Se and gamma at measured currents near 228, 381,
606 µA, with every other parameter from the archived anchored vector. The recipe
adds the reference C/tau/gamma to its listed grid values. **Its top-level baseline
fields are not the values held fixed in this map**; inspect `reference_bundle`,
resolved configuration and the numerical parameter table. Reconstruction, by
contrast, uses the independent-estimate baseline rather than the anchored vector.

Selected shared candidates were verified on all 22 currents at 0.025, 0.0125 and
0.00625 ns. At the finest step:

| Candidate | C (pF) | tau (ns) | gamma | Recovered / 11 | False positives |
|---|---:|---:|---:|---:|---:|
| Prior anchored reference | 6.8355 | 11.9087 | 0.15696 | 9 | 0 |
| Best three-current feature score | 8 | 13.026 | 0.15696 | 10 | 0 |
| Best score satisfying three-current persistence | 8 | 9 | 0.95627 | 11 | 1 |
| Best tested within C timing bound | 0 | 11.9087 | 0.15696 | 0 | 0 |

The persistent candidate falsely oscillates at about 645 µA and gives late robust
Vpp 294.5 mV versus measured 9.25 mV near 606 µA (31.8×). No mapped C≤0.39 pF
point sustains cycles at any of the three mapped currents. This is a conditional
grid result, not a theorem about all possible parameters or thermal-only models.
The best three-current feature point does not beat the reference's aggregate
all-oscillator score at the finest step; it is not a new best global fit.

Identical classifications are not waveform convergence. From 0.0125 to 0.00625 ns,
the persistent candidate changes by at most about 1.2% in late amplitude and 0.9%
in frequency over persistent records, but the best-feature candidate changes by
about 17.3% and 13.7%. Preserve timestep checks before claiming an improvement.

## 7. Latest inverse test: what is and is not inferred

Compute `I_R=I-C*dV/dt`, effective `R=V/I_R` and `P=V*I_R` from baseline-corrected,
smoothed numerical channels. Integrate `Cth*dT/dt=P-Se*(T-T0)` from -200 ns at T0,
then replay the authoritative hysteresis law on that prescribed T history. **Static
R(T) is not used to infer T in this diagnostic**, but the thermal estimates were
themselves obtained using static thermometry earlier. This is not independent
thermometry, a forward fit, or a measurement of gamma.

Central late means (150–250 ns, C=0.39 pF, borrowed gamma):

| Current (µA) | Conditional T (K) | Effective measured R (Ω) | Replayed R (Ω) |
|---:|---:|---:|---:|
| 189.6 | 330.91 | 1680.6 | 1679.2 |
| 228.7 | 328.96 | 1017.9 | 1762.5 |
| 381.1 | 333.64 | 488.4 | 1215.1 |
| 606.7 | 344.25 | 298.2 | 29.7 |

Below-onset agreement is not independent validation because that region helped
estimate Se. Gamma values 0.15696, 0.5, 0.95627 and 2 do not close the high-current
gap (26–34 Ω under central assumptions). The upper Se bound gives 71–112 Ω there,
still well below 298 Ω. Eight cases vary C, Se, Cth or smoothing **one at a time**;
they do not form a joint confidence region or exhaust every combination.
Replays at 1/0.5/0.25 ns test numerical sampling, not added experimental bandwidth.
The maximum mean-R change for the final halving is about 4.85% across tested cases.

Conclusion: the present combination of constitutive law, thermal assumptions,
parameters and channel interpretation is inconsistent with these records. The
analysis does not distinguish a thermal-estimate error from contacts/readout,
dynamic/partial switching or nonuniform heating. These are hypotheses, not findings
that license adding arbitrary degrees of freedom. **No replacement calibration is adopted.**

## 8. Channel audit completed; circuit boundary remains unresolved

The [18 September desk audit](CHANNEL_CIRCUIT_AUDIT.md) checked all three original
XLSX workbooks against their CSV exports and the experimental manuscript. The
workbooks' numerical conversions are internally consistent: CH1 voltage maps to
output mV by ×1000; CH2 voltage maps to input µA by approximately ×794.33; the
second time column is exactly +150 ns for presentation. The exports match the
workbook-derived columns to rounding precision. No raw data or bundles changed.

The available files do **not** establish synchronized VO2 terminal voltages, the CH2
current-conversion circuit, feedback-path current, sign, probe loading or calibrated
channel skew. The manuscript identifies an active TIA feedback path, LT1228
transconductance stage and common-base protection, but its `Vout*Iin` power
calculation does not independently establish VO2 Joule power. A circuit correction
cannot be adopted or tested against persistence until those nodes and currents are
measured or documented.

The paper's approximately 1.2–1.9 pJ per-oscillation result is numerically
reproducible as **total output-derived energy per period**. On the 11 nominally
persistent records, direct integration of the baseline-corrected channel product
between adjacent late voltage peaks gives 1.159–1.754 pJ. The archived
`mean power / frequency` values agree to 1.30% median and 2.76% maximum relative
difference. Subtracting each cycle's minimum power before integration gives only
0.044–0.313 pJ (2.5–27.0% of the full-period energy; 9.3% median). Both are valid
definitions for different questions, but only the first reproduces Figure 7's
scale. Calling either an intrinsic VO2 switching energy still requires the missing
device-boundary voltage and current calibration.

An additional cross-current clue: after baseline correction, the late output is
approximately 173–188 mV from 303 to 908 µA, while apparent `V/I` falls from
619 to 190 Ω. The 607 µA conflict is part of this whole high-current branch.
Recorded current ripple at the 607 µA voltage frequency is about 0.08 µA
peak-to-peak, far below the roughly 21 µA needed to explain the measured periodic
voltage across a fixed 298 Ω. The frozen forward model at that current predicts
about 153 mV and 339.7 K; measured output is about 181 mV. Mapping that 28 mV
power difference through the adopted thermal balance moves the conditional
temperature about 4.7 K and sharply lowers replayed R(T). This explains how
the roughly tenfold *inverse* resistance gap can arise from a smaller forward
mean-voltage error, while leaving the sustained oscillations unresolved.
Across currents, the forward mean error changes sign, so a constant output
offset is inadequate.
Even fitting an affine `a+b V_model` map to all 22 means leaves 29.3 mV
RMSE and a 62.7 mV miss at the 190 µA control; adding a linear current
term leaves 27.0 mV RMSE. The predicted means change only 4.4 mV from
190 to 228 µA, whereas measured means drop 83.8 mV. A simple fixed
readout gain, offset or series term cannot repair the onset.

A targeted replay changing only `S_e` from 3.675 to 4.413 µW/K makes the
606 µA mean 181.7 mV versus 181.6 mV measured, but worsens the 228 µA
mean error from 80 to 117 mV; at 907 µA it still misses by 44 mV. Its
606 µA voltage span is only 1.8 mV over 50–250 ns. Conditional
major-branch, quasi-steady inversion of the late measured `V/I` would demand
apparent `S_e` values from roughly 2.47 to 6.09 µW/K across the sweep.
These are diagnostics of the joint assumptions, not a replacement thermal
calibration.
Even freeing both constant `S_e` and `T₀` does not make the static-branch,
quasi-steady late points consistent: `ΔP/ΔT` is 5.85 µW/K between 190 and
606 µA but 51.0 µW/K between 606 and 908 µA. The ratio stays 7.4–10.7
across paired static R(T) bootstrap draws. This tests the entire assumed
measurement/constitutive/thermal chain, not thermal conductance alone.
Crossing four pre-pulse baseline and four output averaging windows leaves
the 800 mV apparent resistance at 296.1–301.4 Ω and the slope ratio at
8.49–9.24. Window selection does not explain the cross-current gap within
this conditional static-branch analysis.
A roughly tenfold *relative* channel-resistance rescaling can equalize the
three selected slopes, but it implies `T₀ ≈ 335.1 K` (adopted 314.4 K) and
misses the 38 µA control's inferred temperature by about 4.4 K. This is a
stress test of a fixed gain explanation, not a calibrated transfer factor.
Across onset, the 250→300→350 mV labels also show both falling apparent
resistance (1680→1017→711 Ω) and falling apparent power
(60.73→53.23→50.76 µW). The sign persists across 16 window choices and
cannot arise from one monotone quasi-steady heating branch with constant
positive conductance and fixed positive channel gains. The latter two traces
oscillate, so this tests that *joint interpretation* of the mean data, not
the validity of the full time-dependent thermal equations.
Avoiding the strongest midrange oscillations gives the same qualitative
result: the 50→250 mV low-current and 900→1200 mV high-current pairs imply
4.64 versus 55.31 µW/K under the direct-channel, quasi-steady major-branch
reading. The slope ratio stays 11.52–12.78 across 16 time-window choices;
its 95% range across the archived paired R(T) bootstrap is 10.73–13.14.
The high-current records have small residual periodic output, and this
comparison remains conditional on the channel and static-state assumptions.

Pre-pulse medians vary from 3.87 to 38.23 µA and −10.32 to +19.52 mV;
some raw voltage baselines are negative despite positive current. Per-record
subtraction is justified as a channel normalization, but it does not tell us
whether the reported leakage flows through and preheats VO₂. The replay's
common cold initial state is therefore another conditional assumption.
Treating the full recorded pre-pulse current as real VO₂ leakage makes the
high-current incremental prediction worse: about 110 versus 181 mV at
606 µA, and 16 versus 173 mV at 908 µA. It also predicts about 89 mV
pre-pulse device voltage in the highest record versus 19.5 mV raw output.
This rejects that particular full-leakage assignment under the adopted
model, while leaving the actual current split unresolved.
Varying the frozen model's initial temperature from 285 to 375 K while
retaining the baseline-corrected 296 ns prehistory changes any 50–250 ns
predicted mean by at most 0.144 mV, leaves mean RMSE near 44.82 mV and
produces zero oscillators. Starting on the metallic rather than insulating
major branch has the same maximum effect. The prehistory is about 22.7 times
the adopted 13.03 ns thermal time, so ordinary modeled thermal/major-branch
initial memory is erased. This does not test an unmodeled long-lived filament
state; starting directly at the comparison window would be a different input.

The early-heating `C_th` fit is also alignment sensitive. With all other
assumptions fixed, synthetic shifts `I_shift(t)=I_recorded(t+shift)` of
−2, 0 and +2 ns give `C_th = 0.06117, 0.04787, 0.03792 pJ/K`
respectively at adopted `C=0.39 pF`; fit RMSE is 2.263, 1.145 and
0.635 K. The original conditional interval did not vary channel delay.
No actual delay has been measured, so these are sensitivity values rather
than corrected parameters.

The recorded turn-off edge is also inconsistent with a passive positive-R
parallel branch if CH1 and CH2 are its actual voltage and current: 9 of 22
records contain negative raw output voltage with more than 50 µA positive
raw input current during 270–400 ns. The recorded current remains nearly
linear in source setting up to about 908 µA, so the observed late voltage
plateau is not simply clipping of that recorded current channel. These checks
strengthen the need for a circuit-boundary measurement but do not identify
a unique circuit or intrinsic mechanism.
This edge flag is alignment sensitive: a synthetic 15 ns advance of the
recorded current removes the `V < −10 mV, I > 50 µA` overlap in all nine
records. At each first corrected voltage zero crossing, reaching under
10 µA instead takes about 18–25 ns of current advance. Neither is a measured
delay; the edge observation is a calibration target, not proof of a specific
active circuit effect. The late-output mean and cross-current slope checks
are comparatively insensitive to such shifts.

At the two highest source labels, late output retains about 5.7–6.0 mV
peak-to-peak periodic components near 69 MHz, far above the same-frequency
pre-pulse component but below the nominal four-window persistence gate.
Treat the high-current "stable" label as operational; a dummy feedback
resistor and longer pulses are needed to identify this small residual signal.

**Next:** request the Figure 6 board schematic, probe locations, gains/termination,
feedback-path current conversion and channel-delay calibration from Amir/Yoav; then
perform a simultaneous two-terminal device-voltage/current measurement with a
known feedback resistor control at about 190, 267, 607 and 908 µA. The
190/267 µA pair tests the onset power-ordering reversal. If those channels
validate the simple model boundary, test same-device driven R(T), minor-loop and
thermal assumptions one change at a time. Do not silently reinterpret source
labels or tune an unexplained voltage offset.
Gabriel currently does not have the schematic or scope setup; continue bounded
tests on existing data while that external information is unavailable.

A useful stopping criterion: a single shared vector must improve late amplitude,
frequency, mean and persistence across selected low/mid/high currents, preserve
stable controls, and remain stable under timestep refinement. Report trade-offs;
do not promote a fit solely because its scalar loss or label count improves.

## 9. Reproduce efficiently and keep evidence safe

### Setup and checks

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install pytest
pytest -q
neuristor validate
```

The project requires Python ≥3.10. Dependencies are in `pyproject.toml`; there is
no fully pinned environment lock, so inspect run provenance before expecting exact
cross-platform reproducibility. Verify supplied-file hashes from
`data/experimental/tia_current_sweep/` with `shasum -a 256 -c SHA256SUMS`.

Checks repeated on **17 September 2026** passed: **51 tests, 15 experiment recipes,
64 archived runs**. All 22 handoff links and all listed recipe paths were also checked.
These counts are a dated baseline, not requirements to preserve
by suppressing new tests or evidence. Test files cover resistance fitting, hysteresis
reversals, simulation convergence, inference, persistence, reconstruction and CLI bundles.
The 18 September channel audit and diagnostic correction passed **52 tests** and
the same **15 recipes / 64 archived runs** validation gate; no scientific
trajectory or public bundle was regenerated.
The corrected timestep report can change automatic step choice in the standalone
sweep-GIF helper on future runs; frozen model validation and inverse replay use
their explicit steps and are unaffected.

### Start with archived evidence; rerun only the relevant diagnostic

```bash
# Cheap orientation / mechanism control.
neuristor runs list
neuristor simulate current --config experiments/current/nonzero_voltage_valley.toml

# Inverse check: lower cost than a broad forward parameter search.
neuristor analyze reconstruct-hysteresis --config experiments/current/specimen_hysteresis_reconstruction.toml

# Controlled map plus fine-step all-current verification; more expensive.
neuristor analyze oscillation-audit --config experiments/current/specimen_oscillation_audit.toml

# Frozen specimen prediction and capacitance sensitivity.
neuristor analyze model-validation --config experiments/current/specimen_model_validation.toml
```

Do not rerun the entire history simply to familiarize yourself. The global searches
below are historical reproduction options, not a recommended first step:

```bash
neuristor analyze fit-waveforms --config experiments/current/specimen_waveform_inference.toml
neuristor analyze fit-waveforms --config experiments/current/specimen_oscillation_inference.toml
neuristor analyze fit-waveforms --config experiments/current/specimen_oscillation_amplitude_inference.toml
neuristor analyze fit-waveforms --config experiments/current/specimen_physics_anchored_inference.toml
```

Exact R(T), Se and Cth commands are in the root README and report appendix. Use
`--help` for current command options. The audit supports `--reuse-numerics` for
figure-only changes: use the identical resolved recipe; it makes a new bundle and
preserves original numerical provenance and CSV checksums instead of relabeling a
new render as a new simulation. The 26ff13 audit's numerical commit is `79c0692`;
its manifest separately identifies rendering provenance.

### Change and archive workflow

Follow AGENTS.md: pure analysis/physics function → workflow returning RunBundle →
CLI command → reusable recipe → tests → documentation. No new top-level one-off
scripts. Preserve unrelated edits. Physics changes require at least two smaller
timesteps and a convergence assertion; do not loosen tolerances to accept changes.

```bash
neuristor runs publish RUN_ID
git add public_jobs/RUN_ID
# Also stage the intentional source, recipe, test and documentation changes.
git commit -m "Describe the scientific change and reviewed evidence"
git push
```

Never overwrite a completed public bundle, even to fix a caption. Publish a new
one and update the archive index's supersession note. `runs publish` does not commit
or push automatically. An old `run.json` summary can contain a superseded claim;
read this handoff and the current archive index before interpreting it.

### Report and slide maintenance

Compile from the directory containing the TeX source so relative figure/data paths
resolve. With Tectonic installed:

```bash
# From docs/final_project:
tectonic -X compile --keep-logs --outdir ../../output/pdf main.tex
# Copy generated main.pdf to the named output and the tracked report snapshot.

# From docs/final_project/presentation:
tectonic -X compile --keep-logs --outdir ../../../../output/pdf VO2_Fitting_Journey.tex
# Refresh the identically named PDF snapshot in this presentation directory.
```

Render and visually inspect changed pages, not just compiler success. Slides must
remain exactly eight pages unless Gabriel requests otherwise. The slide README
records CSV selection and hashes. The report and presentation are separate artifacts;
updating one does not rebuild the other. Generated `output/` is ignored; keep the
reviewed readable snapshots with their TeX sources under `docs/final_project/`.

### Working preferences inherited from Gabriel

- Work on one interpretable research question at a time. Explain methodology and
  assumptions before showing parameter calculations or results.
- Keep the report concise; supervisors know the model. Tables and clear plots are
  preferred to lengthy introductory material. Avoid unnecessary subsections.
- Keep title typography consistent; supervisory attribution should use consistent
  sizing. Preserve the existing report/presentation style unless asked to redesign.
- Document formulas, provenance and limitations. Do not describe a diagnostic fit
  as a parameter measurement or erase old evidence when a better attempt appears.
- Prefer reusable CLI recipes, readable code and comments explaining scientific
  intent. The dashboard is an archive browser, not another simulation interface.
- Be economical with research/computation. The September 14 request to conserve
  usage for five hours was a time-limited constraint, not an ongoing scheduled job.

### Repository / host checkpoint

The work is on `codex/unified-cli-dashboard-refactor`, remote
`https://github.com/gabebere/single-neuristor.git`. The completed pre-handoff checkpoint
is `c668a82` (presentation), following `4c5c4c4` (report/reconstruction). The recovery
tag `v0.1.0-working-baseline` preserves pre-refactor work. This does not imply the
research branch has been merged into the default branch; check Git before assuming it.

On 17 September, the system Git command hit an unaccepted Xcode license prompt.
The already installed bundled Git worked at
`/Users/gabrielberezovsky/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git`.
This is a local environment issue, not a repository failure; no license was accepted
on the user's behalf. Previous successful LaTeX builds used `/opt/anaconda3/bin/tectonic`;
scientific Python was `/opt/anaconda3/bin/python`. These are convenience paths on
Gabriel's machine, not portable project requirements. A clean checkout/venv is the
portable starting point. Check current tool paths rather than assuming `/usr/bin/git`
or the shell's `python3` is the working scientific environment.

## 10. Evidence register

Each link is a full immutable run directory. Start with `report.md`, then inspect
`resolved_config.json`, `metrics.json` and the cited numerical tables.

| Short ID | Bundle |
|---|---|
| 6765e0 | [Mechanism control](../public_jobs/20260817_100102_current-step-with-a-nonzero-metallic-voltage-val_6765e0/) |
| 0849a9 | [Specimen major-loop fit](../public_jobs/20260816_125905_sample-r-t-major-loop-hysteresis-fit_0849a9/) |
| ec6ec4 | [Normalized measured sweep and onset bracket](../public_jobs/20260827_142103_measured-laboratory-current-sweep-with-current-l_ec6ec4/) |
| 761640 | [Environmental conductance](../public_jobs/20260817_153807_environmental-thermal-conductance-estimate_761640/) |
| aa2469 | [Thermal capacitance with C upper bound](../public_jobs/20260828_112314_thermal-capacitance-estimate-with-conservative-0_aa2469/) |
| 4223e0 | [Rejected capacitance shortcut, historical only](../public_jobs/20260817_134254_lab-current-trace-parameter-estimates_4223e0/) |
| eefab7 | [Frozen model and capacitance sweep](../public_jobs/20260829_100718_specimen-model-prediction-versus-measured-curren_eefab7/) |
| 8f12d6 | [Original constrained/relaxed inference](../public_jobs/20260829_105704_global-specimen-parameter-inference-from-all-cur_8f12d6/) |
| ac1c5e | [Oscillation-priority inference](../public_jobs/20260829_160147_oscillation-priority-global-specimen-parameter-i_ac1c5e/) |
| 85526c | [Amplitude-priority inference](../public_jobs/20260829_210212_amplitude-tuned-oscillation-priority-specimen-in_85526c/) |
| 717797 | [Physics-anchored diagnostic fit](../public_jobs/20260907_125315_physics-anchored-specimen-waveform-inference_717797/) |
| 26ff13 | [Current persistence audit and controlled maps](../public_jobs/20260914_073330_sustained-oscillation-audit-and-three-current-ma_26ff13/) |
| fa4b66 | [Current conditional inverse reconstruction](../public_jobs/20260914_074731_conditional-hysteresis-reconstruction-from-measu_fa4b66/) |

Particularly useful audit tables: `archived_windows.csv`, `measured_metrics.csv`,
`parameter_map.csv`, `candidate_ranking.csv`, `verification.csv`,
`threshold_sensitivity.csv` and `candidate_traces.csv`. Reconstruction tables:
`reconstruction.csv`, `gamma_scores.csv`, `trajectories.csv`. No latest fit should
be selected by lexicographic run date alone; use the status interpretation above.

## 11. Ready-to-use continuation brief

> Read README.md, AGENTS.md, docs/RESEARCH_HANDOFF.md and the 18 September
> CHANNEL_CIRCUIT_AUDIT.md. Continue from the persistence audit and conditional
> reconstruction, not the superseded 22/22 peak-count claim. The desk audit
> verified workbook conversions and exposed a nearly flat 173–188 mV output
> branch, but the actual VO2 terminal voltage and feedback current remain
> unverified. Obtain the schematic/probe/current-conversion details and calibrate
> the circuit boundary with a known resistor and simultaneous device-terminal
> measurement before adding parameters or another broad fit. Preserve raw data
> and completed bundles; keep thermal intervals conditional on channel meaning.

## 20 October 2026 — optimizer sensitivity follow-up

Completed three-timestep coordinate perturbations, two larger derivative-free
searches (765 unique candidates each, four starts), and four short L-BFGS-B
trials. No eligible all-target fit was found. Both expanded searches retain the
old frequency-only diagnostic; combined selection reflects failed-target priority,
not a smaller complete scientific loss. The finite-score gradient trials reached
the four-step line-search limit. Their best sampled points modestly improve some
frequency comparisons, with worse amplitude error and remaining timestep changes.
Of 792 individual-frequency stencils, 644 were defined and 327 flat; none of the
72 full-loss stencils was defined at the original centers. Discrete extraction
and missing oscillations limit derivative interpretation.

Detailed methods, tables, commands and immutable bundle IDs:
[OPTIMIZER_AUDIT_20261020](OPTIMIZER_AUDIT_20261020.md). Software checks: 186 tests
passed, validation passed, exact numerical recipes and saved-data review completed.
No physical equation or feature-extraction algorithm changed; no specimen
calibration was adopted.

## 20 October 2026 — return to full-map interpretation

Gabriel requested removal of the three optimizer follow-up slides. The prior
19-slide presentation is restored; optimizer evidence remains immutable. The
full 12,167-point/1,258-GIF exploration was analyzed from saved finer replays,
without new physics or integration. Controlled neighbor comparisons distinguish
C and Cth amplitude tendencies, while a 16-point constant-ratio ray supports
inverse frequency scaling with nearly unchanged Vpp. Conditions remain
500 µA, 298 K, fixed gamma=0.956269682. No calibration is adopted.
[Methods, figures, evidence and limitations](PARAMETER_EXPLORATION_TRENDS_20261020.md).

## 20–21 October 2026 — fresh controlled parameter-trend verification

Completed 610 distinct integrations over 75 physical settings (108 logical
comparison cases with repeated centers): 466 in initial pass `5beaf6`, then
144 new integrations in checked-cache, longer-window follow-up `3599fb`.
The original ideal-current physics, major-loop fit and float32 hysteresis remain
unchanged. Three decreasing steps, optional extra halving, successive late-window
and observation-horizon checks use the declared 2% feature rule. Temperatures
must remain in the measured R(T) interval; no frequency ceiling removes cycles.

All 12 final common-scale pairs pass. Co-scaled timestep gives identical sampled
V/T/R histories; fixed-dt final frequency errors are <=0.3443%, while extracted
Vpp changes are <=0.8706% across protocols because the measurement operator has
a fixed physical-time width. An independent crossing estimator agrees with
peak frequency within 0.0034%. Full pointwise fixed-dt waveform convergence is
not established; unaligned path differences remain appreciable.

At 500 µA/298 K, frequency decreases in all five usable adjacent comparisons
for each of C, Cth and Se. C amplitude increases and Cth amplitude decreases are
resolved in only two pairs each; three more changes per capacitance are too
small to resolve. Se raises amplitude in all five usable pairs. Reducing Se at
anchor A by 25% ends verified oscillation. Two anchor-C perturbations are
calibration-censored. Gamma=0.70–0.95627 changes features modestly at the three
fixed anchors. Lower-current and warmer-ambient sweeps supply no usable
adjacent pair, so these directions are not generalized to those conditions.

Longer windows resolve the old cycle-count limitation for slow centers; a coarse
half-scale overshoot disappears under refinement. Neither limitation proves a
physical boundary. Verified examples extend beyond old 20 pF/110 MHz display
limits. No new specimen calibration or experimental fit is adopted. The
19-slide presentation is unchanged.

[Detailed methods/results](FRESH_PARAMETER_TRENDS_20261020.md). Reviewed immutable
bundles: `20261020_205903_fresh-controlled-validation-of-parameter-explora_5beaf6`
and `20261020_220155_extended-horizon-validation-of-parameter-explora_3599fb`.
Full suite: 192 passed; exact recipes and published-parent reuse check completed.

## 21 October 2026 — budgeted new-sample M2 exploration

The supplied `m2_8.9_1stL_1600nm_A_RT` numerical file was preserved byte-for-byte
under `data/experimental/` with an explicit `.tsv` extension. The existing
end-to-end study workflow was reused at 500 µA, 298 K and inherited assumed
gamma. Static log10 RMSE 0.08065244 failed the normal 0.05 gate; that stopped
run remains preserved. Four recipes explicitly allow a 0.10 static error for
diagnostic use while leaving temperature, persistence and timestep criteria intact.
They cover 216+216+216+125 grid entries; none is retained. A separate 36-point
pilot at 0.0015625 ns also found no eligible oscillator. Every sustained grid
detection exceeded the measured temperature interval, so this result is not a
new calibration or a proof of absent physical oscillations. The locally archived
review `20261021_062155_m2-1600nm-budgeted-r-t-exploration-review_c7387b` preserves
all studies and provides explicit exclusion-aware frequency/amplitude maps.
See [full result and reproducibility notes](M2_RT_EXPLORATION_20261021.md).

## 21 October 2026 — additional M2 search expansion

With another 15-minute budget, the existing physical-maps command reused the
same approximate static preset for an 8³ wider grid and an 11³ onset grid.
These add 1,843 sampled entries (2,616 cumulative, including overlaps); 35 and
373 sustained detections, respectively, all exceed the measured temperature
domain. No point is retained, and no static/dynamic calibration is adopted.
A separate 150 ns startup check of 12 nearby sustained settings at 0.00078125
and 0.000390625 ns confirms early temperature exclusion in all 24 integrations
(minimum observed maximum 372.67096 K). It does not establish full-grid or
steady-state convergence. The new immutable local archive is
`20261021_064103_m2-1600nm-expanded-r-t-exploration-review_226099`.
196 tests and repository validation passed. [Expanded maps and recipes](M2_RT_EXPLORATION_20261021.md).

## 25 October 2026 — recovery of different-amplitude cycles in one run

A read-only saved-trace search recovered a clear alternating-amplitude example
in the 7 September relaxed waveform fit, source label 400 mV. Eight primary
cycles in 50–280 ns alternate near 238.92/352.98 mV Vpp, with lag-1/lag-2
amplitude correlations −0.99939/+0.99924. Neighboring labels show related
behavior. This is in the simulated, not measured, voltage column. The native
0.025 ns replay matches saved voltages within 1.53e-11 mV; 0.0125 and 0.00625 ns
retain the short-window pattern. A 303.391147 µA constant-current control
extended to 2 µs is regularly alternating at the two coarser steps but irregular
at 0.00625 ns. No stable period-two attractor, bifurcation or experimental
period doubling is established. No physical law, tolerance or calibration changed.
Evidence `20261025_082439_documented-alternating-amplitude-historical-simu_1072ef`
preserves 18 replays, individual-cycle measurements, detection sensitivity,
source/parameter provenance and the exact CLI control. 200 tests and validation
pass. [Documented finding and figures](ALTERNATING_AMPLITUDES_20261025.md).

## 25 October 2026 — frozen parameters, classic GIFs and final slide

The recovered vector is preserved in
`presets/alternating_amplitude_relaxed_20260907.json`. New immutable evidence
`20261025_091004_fixed-alternating-amplitude-classic-gifs_4b9e8c` contains
44 classic stacked GIFs for all 22 measured inputs at 0.025 and 0.00625 ns.
66 native integrations cover three timesteps without refitting; original-step
voltages agree with the archived fit within 1.53e-11 mV. Input current appears
above simulated voltage and native R(T), without a measured-voltage overlay.
The new final slide (22) gives tau_th=Cth/Se=11.9087 ns and the central
tau_e=CR range 0.1334–11.9846 ns, with a 0.1245 ns metallic limit. Native
cycles show shallow heating near 338 K and deep heating near 348 K, motivating
a qualified electrothermal/hysteresis-memory explanation. The longer-run
timestep limitation remains explicit. All original 21 slide texts are preserved.
The loopback gallery button, 44 GIF frame counts and paths, exact recipe and
compiled final slide are checked. 201 tests and repository validation pass.
