# Research handoff: VO2 current-driven simulations

Updated **17 September 2026**. Scientific evidence through 14 September; the
eight-slide presentation was completed on 15 September. This handoff introduces
no new simulations or parameter estimates.

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
2. [Eight-slide presentation](final_project/presentation/VO2_Fitting_Journey.pdf)
   and its [source/evidence map](final_project/presentation/README.md).
3. [Research report](final_project/Simulations_for_VO2_AGC.pdf), especially Sections
   4–6 for parameter assumptions and 9–12 for fitting and the latest diagnostics.
4. [Archive index](ARCHIVE_INDEX.md) for current versus superseded bundles.
5. [Root README](../README.md), [AGENTS.md](../AGENTS.md), and the relevant recipe
   before changing code. Read [architecture](ARCHITECTURE.md) for implementation.

The next recommended task is a **measurement-channel and circuit-interpretation
audit**, not another broad optimizer run. See Section 8 for its scope and deliverable.

## 2. Files and data: what is authoritative

All paths in commands below are relative to the repository root.

| Item | Canonical location and meaning |
|---|---|
| Raw specimen R(T) | `data/experimental/100425_chip1_gap3.tsv`; professor-supplied major loop |
| Fitted resistance preset | `presets/resistance_100425_chip1_gap3.json`; values, intervals, diagnostics and fitted temperature domain |
| Raw oscilloscope exports | `data/experimental/tia_current_sweep/`; preserve unchanged |
| Experimental data provenance | The above directory's `README.md` and `SHA256SUMS` |
| Original reference implementation | `references/yuanhangzhang98-collective_dynamics_neuristor-217d4f0/` |
| Papers | `docs/final_project/references/` links to canonical local copies |
| Editable report / tracked PDF | `docs/final_project/main.tex` / `Simulations_for_VO2_AGC.pdf` |
| Editable slides / tracked PDF | `docs/final_project/presentation/VO2_Fitting_Journey.tex` / `.pdf` |
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

## 8. Next task: a bounded channel/circuit audit

This is proposed work, **not completed**. First produce a short source-backed mapping
of each recorded column to the physical circuit and its conversion. Use the three
original XLSX workbooks and the experimental manuscript. Distinguish information
actually present there from questions requiring Amir/Yoav.

Required output:

1. A circuit/channel table: voltage probe nodes, current sensing/conversion equation,
   sign, scaling, units, channel time alignment, loading and bandwidth where known.
2. A check of the workbooks' conversions against the corresponding converted exports,
   without altering any raw data. Do not treat the 150 ns display shift as a delay.
3. An explicit answer, or a documented unknown: is recorded voltage directly across
   VO2, and is recorded current the total current entering the modeled parallel R–C
   branch? Does V/I include contact/load/readout contributions?
4. One minimal testable correction if supported, its expected effect on mean voltage,
   amplitude and frequency, and a comparison using the persistence audit.

If the channels cannot be resolved from existing sources, ask for the schematic,
probe locations, gains/termination, current conversion and channel-delay calibration.
Do not silently reinterpret 250/300 mV labels as µA or tune unexplained offsets.
If the channel map is confirmed, the next branch is a controlled test of static-to-
dynamic resistance transfer or thermal-model assumptions, one change at a time.
New dynamic minor loops or independent temperature/capacitance information would
help separate gamma, thermal parameters and circuit effects.

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

> Read README.md, AGENTS.md and docs/RESEARCH_HANDOFF.md. Continue the VO2 specimen
> research from the current persistence audit and conditional reconstruction, not
> the superseded 22/22 peak-count claim. Do not launch a broad fit or alter physical
> parameters yet. First inspect the supplied workbooks and experimental manuscript
> to map voltage/current channels, conversion formulas, probe nodes and timing to
> the modeled VO2 branch. Deliver a concise evidence-backed mapping, explicit
> unknowns, and one proposed minimal test. Preserve raw data and completed bundles;
> distinguish conditional thermal estimates from independent measurements.
