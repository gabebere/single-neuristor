# Research handoff: VO2 current-driven simulations

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
