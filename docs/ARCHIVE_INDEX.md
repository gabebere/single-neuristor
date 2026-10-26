# Research archive index

Historical paths below describe the files at each dated checkpoint. Retired code
and document assemblies are now available through the
[layout and recovery record](REPOSITORY_LAYOUT.md); current results and measured
inputs remain in this checkout.

This index distinguishes current quantitative evidence from historical simulation and
presentation artifacts. That distinction matters because Git history preserves earlier
experiments, while only the current solver and audit should be used for new numerical
claims.

## Recovered alternating-amplitude simulation — 25 October 2026

- `20261025_091004_fixed-alternating-amplitude-classic-gifs_4b9e8c`:
  freezes the exact diagnostic vector and adds 44 classic stacked GIFs for all
  22 measured inputs at two timesteps, with 66 native integrations at three
  timesteps. Derived tau_th=11.9087 ns; central tau_e spans 0.1334–11.9846 ns.
  Slide 22 explains the qualified electrothermal-memory hypothesis and links
  to the GIF gallery. No refit, physical solver change or calibration adoption.
- `20261025_082439_documented-alternating-amplitude-historical-simu_1072ef`:
  recovered source-label 400 mV in the 7 September relaxed fit: eight archived
  primary cycles alternate near 239/353 mV Vpp. Exact replay and two finer
  timesteps retain the short measured-input pattern. A longer constant-current
  control becomes irregular at the finest step, so stable period doubling and
  experimental agreement are not established. Includes 18 native replays,
  individual-cycle and threshold-sensitivity tables, figures and a reusable
  control recipe. [Finding and limitations](ALTERNATING_AMPLITUDES_20261025.md).

## M2 1600 nm exploratory R–T maps — 21 October 2026

- `20261021_064103_m2-1600nm-expanded-r-t-exploration-review_226099`:
  adds 512 wide-range and 1,331 dense-onset grid entries with the unchanged
  approximate static fit and operating assumptions. No new retained oscillator;
  all 24 finer startup checks of 12 nearby settings exceed the measured range.
  Cumulative grid entries: 2,616, including overlaps. [Larger maps and limits](M2_RT_EXPLORATION_20261021.md).
- `20261021_062155_m2-1600nm-budgeted-r-t-exploration-review_c7387b`:
  new 374-row R–T loop; four coarse grids (773 entries) and a 36-point finer
  pilot. Static fit fails the usual 0.05 gate (RMSE 0.08065); labeled diagnostics
  explicitly permit 0.10. No oscillator survives the measured temperature-domain
  screen. The interactive companion shows excluded f/Vpp results without treating
  them as accepted predictions. No solver changes or calibration adoption.
  [Results, recipes and limits](M2_RT_EXPLORATION_20261021.md).

## All-input loss animations — 21 October 2026

- `20261021_054434_loss-function-comparison-all-44-stacked-gifs_15921f`:
  22 GIFs for frequency-only and 22 for frequency plus amplitude, with measured
  input current, measured/simulated output, and native simulated R(T) history.
  Fixed-parameter replays are checked against the saved finest-step voltages;
  no optimization or calibration changes. [Viewer and method](LOSS_ANIMATIONS_20261021.md).

## Fresh parameter-trend checks — 20–21 October 2026

- `20261020_205903_fresh-controlled-validation-of-parameter-explora_5beaf6`:
  initial 466 fresh integrations. Short late windows leave slower centers
  inconclusive; retained as the original verification pass, not a final boundary.
- `20261020_220155_extended-horizon-validation-of-parameter-explora_3599fb`:
  checked reuse of those 466 plus 144 new integrations. Final scale tests pass
  12/12; 15 usable original-condition parameter pairs support falling frequency,
  with qualified amplitude effects. Independent crossing frequency, gamma
  tests, operating-condition exclusions and all source/provenance are saved.
  75 distinct physical settings, no adopted calibration or experimental fit.
  [Results and limitations](FRESH_PARAMETER_TRENDS_20261020.md).

## Full-map tendencies — 20 October 2026

- `20261020_202706_saved-full-parameter-exploration-trends_cf3c76`:
  saved-data review of the complete GIF map, adjacent controlled comparisons,
  representative GIFs and a 16-point common-capacitance scaling check. No new
  integration, changed membership, or specimen calibration. Conditions fixed
  at 500 µA, 298 K and inherited gamma. See
  [interpretation](PARAMETER_EXPLORATION_TRENDS_20261020.md).

## Optimizer follow-up — 20 October 2026

- `20261020_185046_finite-difference-and-gradient-optimization-audi_21879a`:
  150 coordinate profiles, three solver timesteps, four limited L-BFGS-B trials,
  and all-current verification. No defined full-loss derivative at the parent
  candidates; gradient trials operate on the finite rejection score.
- `20261020_184721_expanded-optimizer-test-frequency-only-specimen-_aae50f`:
  four-start frequency-only search, 765 unique candidates, unchanged prior best
  in-domain diagnostic and no eligible all-target fit.

- `20261020_184721_expanded-optimizer-test-frequency-amplitude-spec_d5e6bb`:
  matched four-start combined search; 765 candidates, no eligible all-target fit.
  Its selected diagnostic is the old frequency-only vector because failure count
  takes priority over residual amplitude error.
- `20261020_195947_optimizer-budget-derivative-and-gradient-review_555769`:
  common six-candidate finest-step comparison, per-input frequency derivatives,
  direct profiles and complete saved-data provenance.

See [methods and interpretation](OPTIMIZER_AUDIT_20261020.md).

## Controlled frequency/amplitude fitting — 20 October 2026

[Methods, measured-current examples and limits](LOSS_COMPARISON_20261020.md).
Two bounded searches with fixed R(T)/T0 and four unknowns, 250 evaluations each:

- [Frequency-only search](../runs/published/20261020_155947_controlled-frequency-only-specimen-fit_543408/report.md).
- [Frequency and amplitude search](../runs/published/20261020_165107_controlled-frequency-amplitude-specimen-fit_5ad999/report.md).
- [Reviewed comparison and figures](../runs/published/20261020_170231_controlled-frequency-and-amplitude-loss-comparis_26574a/report.md).

These are unsuccessful diagnostic attempts, not accepted calibrations. Passing
target cycle/frequency counts at the finest step: 10/11 and 8/11; amplitudes remain
far too large. Combined classification changes with timestep, and full-replay
control temperatures exceed the configured resistance domain. Current-ranked
examples use measured ~228, 419 and 606 µA. The earlier comparison `2feaaa` remains
immutable but is superseded for display by `26574a`; numerical parents are identical.
Local publication does not imply a GitHub push.

## Canonical GitHub library — 20 October 2026

The definitive `single-neuristor` repository now has the consolidated code,
approved numerical inputs, documentation and compact reviewed evidence on `main`
through [PR #4](https://github.com/gabebere/single-neuristor/pull/4).
[The dated Release](https://github.com/gabebere/single-neuristor/releases/tag/research-library-2026-10-20)
holds the large GIF map and complementary source/numerical/PNG provenance.
The [selected-results page](../runs/index.html) labels controls, static fits,
conditional exploration and rejected fitting trials separately.

See [the full consolidation record](https://github.com/gabebere/single-neuristor/blob/12be9a9a729e03b164464fad0daab7c1b9d4f4c4/docs/CONSOLIDATION_20261020.md): 174 local/fresh-
checkout tests, successful independent Linux Python 3.10/3.13 checks, actual
control/sample-study runs and complete map reconstruction. A full private local
snapshot preserves ignored exploratory history too. Third-party PDFs are excluded
from the current public tree; historical exposure is not a history purge.
This is a reproducibility/software checkpoint, **not a new accepted calibration**.

## Desktop consolidation — 16 October 2026

[Consolidated Desktop guide](https://github.com/gabebere/single-neuristor/blob/12be9a9a729e03b164464fad0daab7c1b9d4f4c4/archive/desktop_consolidation_20261016/index.html)
maps 19 historical run-folder references back to existing project reports/viewers.
All 926 Desktop content files were checked by SHA-256 and full-byte comparison:
914 were already in the repository; 12 unique notes/index/recipe files were
preserved under `archive/desktop_consolidation_20261016/original_notes/`.
Six existing symbolic links pointed into the project. The original Desktop
archive was moved intact to recoverable macOS Trash, not permanently deleted.
The filing inventory records per-file destinations and the recovery path.
No scientific result, canonical data, completed public bundle, or code changed.
Earlier references to Desktop below describe historical deliveries; use the
consolidated guide now. Local filing does not constitute a GitHub backup.

## Active final-project manuscript

- [Adaptive boundary exploration](BOUNDARY_SEARCH.md), 8 October 2026:
  [reviewed coarse/fine and open-scale viewer](../runs/published/20261008_105134_adaptive-coarse-to-fine-oscillation-boundary-sea_504198/parameter_maps.html).
  42 coarse points and 211 fine/refinement points; 18 transition brackets, 15
  narrowed to the requested width. The common C/Cth scale is analytically open.
  No verified oscillator touches a searched outer face, but 35 time-unresolved,
  six numerically unresolved and 13 calibration-censored points prevent a claim
  of complete physical bounds. Both coarse and fine edges can expand. Solver
  equations are unchanged. The bundle exactly reproduces its numerical parent
  and retains the full original checks/source ancestry, with checked reuse.
  This supersedes treating old fixed-box map extents as physical boundaries,
  while preserving those historical bundles. No experimental fit is adopted.

- [Fixed-scale parameter-map displays](MAP_DISPLAY.md): 13 preserved historical
  maps/studies have new viewer revisions, discoverable through **Updated
  fixed-scale viewer** links in the artifact hub. [Complete 1,258-GIF map with
  fixed scales](../runs/local/20261007_123036_fixed-scale-viewer-portable-screened-parameter-m_4325d1/parameter_maps.html)
  reuses the `eaf6ab` numerical data and GIFs without resimulation. This local
  HTML depends on the original public GIF folder; it is not a standalone email
  attachment. Old HTML/ZIPs are unchanged. A fresh [sample-study run](../runs/local/20261007_123034_sample-r-t-to-screened-map-and-embedded-gifs_9f299d/parameter_maps.html)
  also uses stable scales and remains standalone. No scientific parameters changed.

- [Organized library and paper evidence](https://github.com/gabebere/single-neuristor/blob/12be9a9a729e03b164464fad0daab7c1b9d4f4c4/docs/STRUCTURE_MIGRATION_20261007.md):
  purpose-specific source packages, compatibility imports, plain-language project
  navigation, source inventory, and a five-item reproduction index. Physics and
  completed bundles are unchanged. Local sample-study verification `896056`
  exactly reproduces the previous `ec2d1e` numerical/media output; this is software
  verification, not a new experimental calibration. No GitHub push or external
  release occurred.

- [Unified sample-study program](SAMPLE_STUDY.md): reusable declared-unit R(T)
  fit → screened Cth/Se/C grid → refined point GIF workflow. [Standalone example](../runs/published/20261007_083908_sample-r-t-to-screened-map-and-embedded-gifs_ec2d1e/parameter_maps.html)
  embeds its fit image and all GIFs, without a ZIP. Same-sample static RMSE
  0.0366276; 27 grid combinations, one retained oscillator near 23.2 MHz.
  Default screening uses the **measured**, not padded, R(T) domain. Gamma and
  dynamic constants are assumptions; this is a software demonstration, not a
  new laboratory calibration. The [artifact hub](../runs/catalog/index.html) provides
  categorized chronology, presentations and a full CSV file inventory, linking
  existing evidence without deleting or duplicating historical media.

- [7 October map-to-GIF extension](PARAMETER_MAP_GIFS_20261007.md): portable
  right-click viewer, stable heatmap IDs, full-resolution R(T) samples with
  timestep refinement and explicit unresolved gaps. Completed **three-point
  benchmark only**: `20261007_061058_portable-screened-parameter-map-with-point-gifs-_f34881`.
  Full 1,258-point generation `81ae34` completed, but its original viewer selection was faulty.
  Corrected complete viewer `20261007_074702_portable-screened-parameter-map-with-point-gifs_eaf6ab`
  reuses all 1,258 GIFs unchanged and supersedes the original HTML/ZIP. Ordinary
  JSON point metadata and defensive event lookup correct the selection mapping.
  The corrected bundle is copied to `runs/published/` with all GIFs unchanged.

- [3 October simple extrema loss](SIMPLE_EXTREMA_LOSS_20261003.md): equal-weight
  squared relative frequency/Vpp objective, separate feasibility checks. Reviewed
  `20261003_183555_simple-relative-squared-extrema-fitting-pilot_96b944` has 85
  candidates and no acceptable all-target fit. Diagnostic GIFs use the finest
  verification timestep, not a new fitted calibration. Original Desktop results
  are preserved in `VO2 Research Archive` with a categorized index.
  GIF replay: `20261003_183956_simple-loss-diagnostic-1-all-current-gif-replay-_4e79ac`
  contains 22 synchronized three-panel GIFs and PNGs at 0.00625 ns. These are
  explicitly diagnostic; no physically supported quantitative fit is claimed.

- [2 October meeting implementation](MEETING_PARAMETER_MAPS_20261002.md):
  extrema-only late-window voltage fitting and screened C/Cth/Se maps.
  [Adjustable slice viewer](../runs/published/20261002_152003_focused-dense-physically-screened-steady-state-e_c1772e/parameter_maps.html)
  adds three independent sliders and visible-slice frequency/Vpp normalization,
  with unchanged numerical CSVs. This is the corrected historical slice viewer;
  the eaf6ab media viewer above is the latest complete historical GIF rendering.
  It supersedes initial slider rendering `2bffd0` by correcting floating-point
  indexing; all 69 slices preserve their exact eligible-point counts.
  [Focused interactive 3D and six heatmaps](../runs/published/20261002_145342_focused-dense-physically-screened-steady-state-e_b9e298/parameter_maps.html)
  retain 1,258 of 12,167 simulated combinations after physical/numerical screens.
  [Broad map report](../runs/published/20261002_135437_dense-physically-screened-steady-state-extrema-p_9ac3ad/report.md)
  preserves every original 15^3 coordinate plus added intermediate values.
  Neither map is a specimen calibration; the 110 MHz threshold is an operating
  filter. The [79-candidate extrema fitting pilot](../runs/published/20261002_145243_meeting-extrema-only-physically-screened-fitting_32d91d/report.md)
  found no acceptable laboratory match at three verification timesteps.

- [Expanded search and interactive laboratory](EXPANDED_SEARCH_20260922.md):
  reviewed bundle `20260921_210646_expanded-persistence-aware-joint-inference_694804`,
  399 search simulations and seven vectors verified at three steps. Stronger-static
  fit recovers 7/11 with 44.6 mV mean error; relaxed fit 6/11 with 39.8 mV.
  Amplitude mismatch and timestep-sensitive classification remain. No calibration.
  Editable replay is available through `neuristor playground`.


- [Budgeted joint inference](JOINT_INFERENCE_20260921.md) and
  `runs/published/20260921_130032_budgeted-joint-resistance-and-waveform-inference_6b5d8f/`:
  joint raw-static/feature fit with eleven shared parameters, two static-loss
  weights and 201 simulated search candidates. All five reference/fitted vectors
  were verified on 22 currents at three smaller timesteps. The static-preserving
  result improves the excluded-current feature score but recovers only 4/11
  persistent oscillators; the diagnostic recovers 5/11. No new calibration.
  Methods/results were added to `docs/presentation/main.tex` and `main.pdf`.

- [Research handoff](RESEARCH_HANDOFF.md): consolidated entry point for continuing
  the work, updated 21 September 2026. Read its corrected interpretation and
  metric-denominator notes before comparing historical fit claims.
- [Independent discrepancy audit](DISCREPANCY_AUDIT_20260921.md): code and unit
  review, separate voltage-source capacitance-floor fix, same-device confirmation,
  LT1228 capacitance/bandwidth evidence, conditional major-branch stability
  derivation, filament hypothesis and ordered calibration experiments.
- `runs/published/20260921_124525_independent-units-stability-and-specimen-replay-_ffaded/`:
  fresh replay of all 22 measured inputs at 0.05/0.025/0.0125 ns, with zero
  predicted versus 11 measured persistent records. Finest late-mean RMSE is
  45.33 mV; the last refinement changes any late mean by at most 0.266 mV.
  Frozen-memory major-branch calculations give a minimum trace-zero capacitance
  near 2.80 pF, not a full hysteretic bifurcation boundary. Exact reproduction
  code, input/source hashes, working-tree patch, tables and figure are included.
- [Experiment/model reconciliation](EXPERIMENT_MODEL_RECONCILIATION.md):
  colleague-facing, paper-anchored comparison of the original Figure 6 panels
  with the archived frozen ideal-current replay, assumptions, discrepancies,
  hypotheses and three discriminating measurements. The public experimental
  preprint contains no reproducible numerical neuristor simulation; this memo
  does not attribute our illustrative July runs to its authors.
- [Channel and circuit audit](CHANNEL_CIRCUIT_AUDIT.md): 18 September desk audit of
  workbook conversions, manuscript channel descriptions, the high-current output
  plateau, conditional thermal inconsistency, a one-parameter conductance replay,
  and three targeted next tests. It introduces no new parameter calibration or
  reviewed simulation bundle.
- [Experimental preprint](https://arxiv.org/abs/2604.04594v1):
  the corrected local link now opens Gildor et al., arXiv:2604.04594v1. The
  previous target was Almeida et al. (2002); that distinct static-hysteresis
  reference is retained as [Almeida](https://doi.org/10.1117/1.1501095).

- `docs/final_project/presentation/`: eight-slide LaTeX/Beamer research presentation
  dated 15 September 2026, with a reviewed PDF, editable source, lossless archived
  waveform selection and slide-to-evidence map. It summarizes existing trials;
  no new model fit is introduced. Amplitude comparisons use a common set of the
  11 measured oscillators, and historical peak counts are distinguished from
  persistence checks.

- `runs/published/20260914_074731_conditional-hysteresis-reconstruction-from-measu_fa4b66/`:
  conditional inverse hysteresis consistency test on all 22 measured records,
  eight one-at-a-time cases, four gamma values, and three replay steps. At about
  607 µA, measured resistance is 298 Ω versus 26–34 Ω replayed under central
  thermal assumptions. This is a diagnostic conflict, not a new calibration or
  independent temperature measurement. Reproduce with
  `neuristor analyze reconstruct-hysteresis --config experiments/current/specimen_hysteresis_reconstruction.toml`.

- `docs/final_project/`: human-facing final-project hub. Its README organizes the
  editable manuscript, current PDF, raw resistance and oscilloscope data, reviewed
  run bundles, and supplementary figures and animations without duplicating their
  canonical sources.
- `docs/final_project/main.tex`: working LaTeX manuscript for the VO2 automatic-gain-
  control simulation project. The Yuanhang validation and sample-specific R(T) fit are
  complete, as are the electrical-capacitance, environmental-conductance, thermal-
  capacitance, blind current-comparison, capacitance-sensitivity, and global inverse-
  fitting studies.
- The Case I figure and metrics point to the immutable public run bundle
  `20260817_100102_current-step-with-a-nonzero-metallic-voltage-val_6765e0`, whose voltage
  panel includes the instantaneous metallic fixed point `I(t) R_m`, rather than duplicating
  scientific output inside the manuscript directory.
- `docs/final_project/supplementary/` presents the manuscript's derived R(T)
  trajectory, synchronized current/voltage--R(T) animation, current evidence figures,
  and selected historical animations. Generated evidence remains authoritative in its
  immutable `runs/published/` bundle.
- `20260816_125905_sample-r-t-major-loop-hysteresis-fit_0849a9` is the reviewed
  sample-specific major-loop fit, including the normalized data, fitted preset, residual
  figure, parameter table, and 1000 block-bootstrap parameter samples.

## Current model authority

- `src/neuristor/model.py`: Yuanhang hysteretic resistance and voltage-driven model.
- `src/neuristor/current_drive_sim.py`: supported ideal-current single-device model.
- `docs/HYSTERESIS_IMPLEMENTATION_AUDIT.md`: branch-memory implementation, numerical
  ordering, provenance, and fidelity checks.
- `docs/CURRENT_DRIVE_CALIBRATION.md`: voltage-floor diagnosis, parameter
  identifiability, capacitance study, and recommended laboratory calibration sequence.
- `data/experimental/tia_current_sweep/`: untouched professor-supplied numerical
  sources for paper Figures 6 and 7, with units, provenance, and ZIP checksum.
- `docs/manuscript/theory_behind_simulations.pdf`: compiled model theory manuscript.

## Current corroborating evidence

- `docs/figures/current_drive/capacitance_study/frequency_heatmaps_all_currents.png`:
  shared-scale `C_th`-versus-`C` frequency maps for six currents.
- `docs/figures/current_drive/capacitance_study/capacitance_trace_comparison_600uA.png`:
  direct demonstration that `C` changes timing while the metallic bound is `I*Rm`.
- The numerical laboratory workflow archives measured current/voltage traces, the
  Figure 7 operating window, and a baseline-corrected environmental-conductance
  estimate in immutable `runs/published/` bundles.
- `20260827_142103_measured-laboratory-current-sweep-with-current-l_ec6ec4`: canonical
  archive of all 22 normalized traces, the measured 41.7--62.5 MHz operating window,
  separate onset waveforms, and the shared-axis bracket used in the manuscript. Its
  labels lead with the measured current step and retain the source-voltage setting
  only for provenance.
- `20260827_123959_measured-laboratory-current-sweep-with-onset-bra_82870f`: earlier
  immutable onset-bracket archive superseded by `ec6ec4`, which clarifies the
  measured-current versus source-setting distinction.
- `20260827_120011_measured-laboratory-current-sweep-with-onset-tra_ecc00b`: earlier
  archive retained immutably and superseded by `82870f`, which adds the 250 mV
  non-oscillating control and direct shared-scale comparison.
- `20260817_134254_measured-laboratory-current-sweep_a45254`: earlier numerically
  identical sweep archive retained immutably and superseded by the later onset-evidence
  bundles.
- `20260817_153807_environmental-thermal-conductance-estimate_761640`: canonical
  clean-provenance selected settled
  pre-onset waveform, fitted R(T) temperature inversion, conditional conductance
  interval, numerical tables, and evidence figure.
- `20260828_112314_thermal-capacitance-estimate-with-conservative-0_aa2469`: canonical
  heating-edge reconstruction using the adopted electrical upper bound `C=0.39 pF`.
  It subtracts `C dV/dt` before forming resistance and power, then fits the shared
  `tau_th` and `C_th` with the conditional robustness interval and near-transition
  sensitivity check.
- `20260829_100718_specimen-model-prediction-versus-measured-curren_eefab7`: canonical
  blind prediction test. It replays all 22 measured current waveforms through one
  frozen specimen parameter set, archives common-window measured/predicted metrics and
  convergence evidence, and maps C--Cth sensitivity. The stable pre-onset voltage is
  reproduced, but the adopted model predicts none of the 11 measured oscillatory runs;
  oscillations require electrical capacitance outside the 0.39 pF timing bound.
- `20260829_105704_global-specimen-parameter-inference-from-all-cur_8f12d6`: canonical
  global inverse fit using 17 training and five held-out current settings. It archives
  the complete objective, constrained and relaxed parameter searches, per-trace
  predictions, optimization history, and time-step convergence. The constrained fit
  cannot restore oscillation; the relaxed fit improves waveform statistics only by
  leaving seven of eight physical intervals and producing turn-on transients.
- `20260914_073330_sustained-oscillation-audit-and-three-current-ma_26ff13`: current
  persistence audit and 126-point controlled C/tau/gamma maps at three measured
  currents. It corrects the previous 22/22 claim, verifies four shared candidates
  across all 22 currents at 0.025/0.0125/0.00625 ns, and records the remaining
  amplitude, decay and false-positive tradeoffs. No replacement calibration is
  promoted. Numerical evidence was produced with commit `79c0692`; rendering
  provenance and hashes preserve the exact reused numerical tables.
- `20260907_125315_physics-anchored-specimen-waveform-inference_717797`: historical
  physics-anchored inference. It constrains `S_e`, `T0`, `C_th`, `Tc`, `w`, and `beta`
  to their conditional intervals and matches 22/22 historical peak-count labels.
  Interpretation corrected by the later persistence audit: the 606.3 uA waveform
  decays, so the full sustained oscillation window was not reproduced. Effective
  `C=6.8355 pF` and `gamma=0.15696`, amplitude and frequency remain unresolved;
  membership in the thermal intervals does not establish their physical correctness.
- `20260829_160147_oscillation-priority-global-specimen-parameter-i_ac1c5e`:
  superseded unrestricted classification-first search retained as development
  evidence. It obtained 21/22 classifications only by moving all eight quantities
  outside their independent intervals.
- `20260829_210212_amplitude-tuned-oscillation-priority-specimen-in_85526c`:
  superseded unrestricted amplitude-priority search retained as tradeoff evidence.
  It reduced Vpp MAE to 88.5 mV but fell to 19/22 classifications and moved seven
  quantities outside their independent intervals.
- `20260828_112026_thermal-capacitance-estimate-with-conservative-0_ce51fa`: earlier
  numerically identical upper-bound run retained immutably and superseded by
  `aa2469`, whose figure labels the corrected ratio as `V/I_R`.
- `20260817_161851_thermal-time-constant-and-capacitance-estimate_b3833c`: earlier
  immutable `C=0` thermal fit, retained as the lower-endpoint sensitivity analysis
  and superseded for the manuscript by `ce51fa`.
- `20260817_161539_thermal-time-constant-and-capacitance-estimate_c285f9`: numerically
  identical pre-provenance run retained immutably and superseded by `b3833c`, whose
  resolved configuration also records `C=0` and the nine-sample smoothing window.
- `20260817_152216_environmental-thermal-conductance-estimate_c4be6c`: numerically
  identical pre-commit run retained for immutable provenance and superseded by `761640`.
- `20260817_134254_lab-current-trace-parameter-estimates_4223e0`: superseded historical
  bundle retained for provenance; its 19.8 pF cold-edge result is not valid.
- `docs/figures/current_drive/nonzero_valley_examples/current_input_voltage_output.png`:
  direct imposed-current/voltage-output examples, including an unchanged specimen
  control and an explicit nonzero-valley effective-resistance candidate.
- The adjacent CSV, JSON, and README files preserve the plotted numerical values,
  simulation settings, assumptions, and validity warnings.

## Historical artifacts

- `runs/published/`: tracked reviewed evidence. The registry reads both historical July
  2026 `job.json` records and current portable `run.json` bundles.
- `docs/presentation/`: Beamer presentation source, compiled deck, figures, and
  animations. Its first section preserves the July 2026 mechanism study; the current
  continuation documents the specimen parameter-estimation chain and frozen forward
  comparison. The original July-only version remains recoverable from Git history.
- `docs/presentation/project_year_presentation_outline.md`: longer historical talk
  outline, explicitly marked with its current-status note.
- Git history is the recovery path for removed experimental current-source variants and
  obsolete presets. They are not active physical models.

## Reproduction and verification

From the repository root:

```bash
neuristor simulate current \
  --config experiments/current/nonzero_voltage_valley.toml
neuristor sweep run \
  --config experiments/sweeps/current_capacitance_map.toml
neuristor analyze conductance \
  --data data/experimental/tia_current_sweep \
  --resistance-preset data/parameters/resistance_100425_chip1_gap3.json \
  --resistance-bootstrap runs/published/20260816_125905_sample-r-t-major-loop-hysteresis-fit_0849a9/parameter_bootstrap.csv \
  --ambient-K 314.4 --ambient-interval-K 314.25,314.55
neuristor analyze thermal-capacitance \
  --data data/experimental/tia_current_sweep \
  --resistance-preset data/parameters/resistance_100425_chip1_gap3.json \
  --resistance-bootstrap runs/published/20260816_125905_sample-r-t-major-loop-hysteresis-fit_0849a9/parameter_bootstrap.csv \
  --conductance-mW-per-K 0.003675126546984294 \
  --conductance-bootstrap runs/published/20260817_153807_environmental-thermal-conductance-estimate_761640/conductance_bootstrap.csv \
  --ambient-K 314.4 --electrical-capacitance-pF 0.39 \
  --selected-drives-mV 100,150,200 --fit-window-ns 15,35
neuristor analyze model-validation \
  --config experiments/current/specimen_model_validation.toml
pytest -q
neuristor validate
```

The laboratory workflows read the checked-in numerical exports directly. The former
screenshot digitizer, image-fit preset, screenshot sequence, and image-derived bundle
were removed; the corresponding history remains recoverable through Git. Other
pre-refactor reproduction sources remain in Git history and in the
`v0.1.0-working-baseline` tag.
