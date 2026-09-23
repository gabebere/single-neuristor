# Single VO₂ Neuristor

A reproducible research codebase for current- and voltage-driven VO₂ neuristor
simulations, laboratory-trace analysis, parameter sweeps, and evidence archival.

**Taking over the project? Start with the [research handoff](docs/RESEARCH_HANDOFF.md).**
It records the current scientific status, parameter assumptions, trial history,
corrected claims, reproducible commands and the next bounded investigation.

The [21 September discrepancy audit](docs/DISCREPANCY_AUDIT_20260921.md)
checks units and implementation, fixes a separate small-capacitance bug in the
voltage solver, and documents why the current-source mismatch remains, with
new timestep evidence and a literature-guided measurement plan.

The project now has one workflow:

```text
human-readable TOML  ->  neuristor CLI  ->  tested physics  ->  portable run bundle
                                                          ->  archive dashboard
```

The original Yuanhang Zhang implementation is preserved under
[`references/yuanhangzhang98-collective_dynamics_neuristor-217d4f0/`](references/yuanhangzhang98-collective_dynamics_neuristor-217d4f0/).
The last known-working pre-refactor repository is permanently tagged
[`v0.1.0-working-baseline`](https://github.com/gabebere/single-neuristor/tree/v0.1.0-working-baseline).

The current final-project report, measurements, reviewed analysis bundles, figures,
and animations are organized from one documented entry point:
[`docs/final_project/`](docs/final_project/). Its PDF and report-specific media are
stored there directly; relative data links point to canonical measurements and run
bundles so scientific evidence is not duplicated.

## Scientific result in one sentence

For an ideal current source, the switched-state voltage floor is set by

\[
V_{\mathrm{floor}} \approx I R_{\mathrm{metal}},
\]

not by electrical capacitance. Capacitance changes how quickly voltage approaches
the floor and can change oscillation timing, but it cannot raise the steady floor.
The full derivation, lab comparison, limitations, and figures are in
[`docs/CURRENT_DRIVE_CALIBRATION.md`](docs/CURRENT_DRIVE_CALIBRATION.md).

![Current input and nonzero voltage oscillations](docs/figures/current_drive/nonzero_valley_examples/current_input_voltage_output.png)

## Quick start

Python 3.10 or newer is required.

```bash
git clone https://github.com/gabebere/single-neuristor.git
cd single-neuristor
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

Run the checked-in nonzero-valley experiment:

```bash
neuristor simulate current \
  --config experiments/current/nonzero_voltage_valley.toml
```

Run the upstream-style voltage oscillator:

```bash
neuristor simulate voltage \
  --config experiments/voltage/yuanhang_oscillator.toml
```

Open the archive dashboard:

```bash
neuristor dashboard
```

The dashboard is intentionally read-only. Simulations are created in the terminal so
every result starts from a reviewable recipe and can be reproduced without clicking
through UI state.

## Command line

Use `neuristor --help` or `neuristor <group> --help` for the complete live reference.

| Goal | Command |
|---|---|
| Current-source run | `neuristor simulate current --config FILE.toml` |
| Voltage-source run | `neuristor simulate voltage --config FILE.toml` |
| Parameter sweep | `neuristor sweep run --config FILE.toml` |
| Fit measured R(T) | `neuristor fit resistance --data FILE.tsv` |
| Analyze numerical lab traces | `neuristor analyze lab --data DIRECTORY` |
| Estimate environmental conductance | `neuristor analyze conductance --data DIRECTORY --resistance-preset FILE.json` |
| Estimate thermal capacitance | `neuristor analyze thermal-capacitance --data DIRECTORY --resistance-preset FILE.json --conductance-mW-per-K VALUE` |
| Validate specimen model against lab sweep | `neuristor analyze model-validation --config FILE.toml` |
| Fit shared waveform parameters | `neuristor analyze fit-waveforms --config FILE.toml` |
| Jointly fit static R(T) and waveform features | `neuristor analyze fit-joint --config FILE.toml` |
| Audit persistence and map three currents | `neuristor analyze oscillation-audit --config FILE.toml` |
| Reconstruct conditional driven R(T) | `neuristor analyze reconstruct-hysteresis --config FILE.toml` |
| Browse runs | `neuristor runs list` / `neuristor runs show RUN_ID` |
| Visualize a current run | `neuristor runs visualize RUN_ID` |
| Copy a run to the Git archive | `neuristor runs publish RUN_ID` |
| Validate recipes and archive | `neuristor validate` |
| Open dashboard | `neuristor dashboard` |

Temporary parameter changes use repeatable dotted overrides:

```bash
neuristor simulate current \
  --config experiments/current/nonzero_voltage_valley.toml \
  --set input.amplitude_uA=700 \
  --set electrical.C_pF=25
```

Overrides are type-checked, reject misspelled paths, and are written into the run's
resolved configuration.

A complete cooling/heating R(T) sweep is fitted directly as a major hysteresis loop:

```bash
neuristor fit resistance \
  --data data/experimental/100425_chip1_gap3.tsv \
  --method major-loop \
  --bootstrap-samples 1000
```

This fits the six major-loop parameters in log-resistance space and archives block-
bootstrap confidence intervals. The minor-loop parameter `gamma` remains fixed unless
minor-loop measurements are available. `--method auto` selects this path when the input
contains exactly one cooling/heating reversal and otherwise uses the stateful fitter.

## Experiment recipes

Recipes live under [`experiments/`](experiments/) and state units in every physical
field name. A minimal current-source recipe looks like this:

```toml
schema_version = 1
name = "600 uA current step"
kind = "simulation"
model = "current"
seed = 0

[time]
dt_ns = 0.5
duration_us = 40.0

[input]
amplitude_uA = 600.0

[electrical]
C_pF = 145.34619293

[thermal]
C_th_pJ_per_K = 198.51107324
S_e_mW_per_K = 0.20558726
T0_K = 325.0

[resistance]
preset = "yuanhang"
start_branch = "insulator"
```

The complete schema and path-resolution rules are documented in
[`docs/EXPERIMENT_FORMAT.md`](docs/EXPERIMENT_FORMAT.md).

### Capacitance and current studies

The requested `C_th` versus `C` frequency study is a three-axis recipe. Its first
axis is current, so the output figure contains one heatmap for every current:

```bash
neuristor sweep run \
  --config experiments/sweeps/current_capacitance_map.toml
```

For a smaller single-current map:

```bash
neuristor sweep run \
  --config experiments/sweeps/capacitance_vs_thermal_600uA.toml
```

`C_pF = 0` is supported exactly. It removes the electrical state and enforces
`V(t) = I(t) R(T)`, which is the thermal-only limit discussed in the calibration
notes.

## Laboratory parameter estimation

Analyze the professor-supplied numerical oscilloscope exports and archive their
measured traces:

```bash
neuristor analyze lab \
  --data data/experimental/tia_current_sweep
```

Estimate environmental thermal conductance from the closest settled trace below
oscillation onset:

```bash
neuristor analyze conductance \
  --data data/experimental/tia_current_sweep \
  --resistance-preset presets/resistance_100425_chip1_gap3.json \
  --resistance-bootstrap public_jobs/20260816_125905_sample-r-t-major-loop-hysteresis-fit_0849a9/parameter_bootstrap.csv \
  --ambient-K 314.4 \
  --ambient-interval-K 314.25,314.55
```

The command subtracts each channel's pre-pulse median, selects the last
non-oscillating waveform before coherent oscillation begins, verifies that its
settled resistance is stable, maps that resistance to temperature through the fitted
heating branch, and evaluates `S_e = P/(T-T0)`. Its conditional interval propagates
waveform block resampling, the R(T)-fit bootstrap, and the measured ambient range.
Electrical capacitance is not estimated from the source-limited pulse edge; the
present traces do not resolve a positive value. The sample-specific analysis adopts
the conservative timing-resolution upper bound `C=0.39 pF`, while `C=0` remains the
constrained best fit.

With `C=0.39 pF` and `S_e` fixed, subtract `C*dV/dt` from the measured current and fit
the moderate nonswitching heating edges to obtain the thermal time constant and
`C_th=S_e*tau_th`:

```bash
neuristor analyze thermal-capacitance \
  --data data/experimental/tia_current_sweep \
  --resistance-preset presets/resistance_100425_chip1_gap3.json \
  --resistance-bootstrap public_jobs/20260816_125905_sample-r-t-major-loop-hysteresis-fit_0849a9/parameter_bootstrap.csv \
  --conductance-mW-per-K 0.003675126546984294 \
  --conductance-bootstrap public_jobs/20260817_153807_environmental-thermal-conductance-estimate_761640/conductance_bootstrap.csv \
  --ambient-K 314.4 --electrical-capacitance-pF 0.39 \
  --selected-drives-mV 100,150,200 --fit-window-ns 15,35
```

The shared fit gives `tau_th=13.026 ns` and `C_th=0.047873 pJ/K`. Its conditional
robustness interval propagates trace selection, R(T), conductance, ambient temperature,
and small fit-window changes. The 250 mV near-transition trace is reported separately
because its reversal biases the single-heating-branch estimate downward.

Run the resulting frozen parameter set against every measured current waveform, then
map the electrical/thermal capacitance sensitivity:

```bash
neuristor analyze model-validation \
  --config experiments/current/specimen_model_validation.toml
```

This blind test reproduces the 189.6 uA stable pre-onset mean voltage within 0.67 mV,
but predicts none of the 11 measured oscillatory traces. No oscillation occurs for any
tested `C <= 0.39 pF` across the conditional `C_th` interval. At the adopted `C_th`,
oscillations begin only at 7 pF, which contradicts the measured edge-timing bound.
Consequently, fitting `gamma` is deferred until the dynamic switching loop or the real
TIA/load impedance resolves this model incompatibility.

Fit one shared parameter vector to all measured current/voltage waveforms, while
reserving five source settings as a blind validation set:

```bash
neuristor analyze fit-waveforms \
  --config experiments/current/specimen_waveform_inference.toml
```

The physically constrained fit changes the total objective by only 0.72% and still
predicts zero oscillatory records. A relaxed diagnostic fit improves the all-trace
objective by 22.5% and the held-out objective by 14.9%, but seven of eight fitted
parameters leave their independently supported intervals and the model still produces
only turn-on transients. The result is therefore evidence of model-form mismatch, not
a replacement set of physical parameter estimates.

Run the confidence-ordered fit that anchors independently estimated parameters and
searches broadly only over electrical capacitance and minor-loop curvature with:

```bash
neuristor analyze fit-waveforms \
  --config experiments/current/specimen_physics_anchored_inference.toml
```

This diagnostic matches the historical peak-count labels for all 22 records at both
0.025 and 0.0125 ns. A later window audit found that the 606.3 uA prediction decays,
so these labels do **not** establish the complete sustained oscillation window. `S_e`, `T0`, `C_th`, `Tc`,
`w`, and `beta` remain inside their independent intervals. The remaining conflict is
isolated to effective `C=6.8355 pF` and `gamma=0.15696`; voltage amplitude and
high-current frequency are still too large. The result therefore narrows the next
modeling work to the electrical/readout dynamics and dynamic switching law.

Inspect sustained cycles and controlled parameter changes with:

```bash
neuristor analyze oscillation-audit \
  --config experiments/current/specimen_oscillation_audit.toml
```

This rechecks the archived fit in four 50 ns windows, separates late amplitude,
frequency, mean voltage and decay, and maps `C` versus `C_th/S_e` at several `gamma`
values using the measured inputs near 228, 381 and 606 uA. Every candidate uses one
shared parameter vector. Selected grid points are checked on all 22 currents at
0.025, 0.0125 and 0.00625 ns. The historical detector and bundles remain reproducible;
the new tables explicitly distinguish regular early peaks from persistent cycles.
The grids are conditional slices, not an exhaustive parameter search or a new
physical calibration. See the generated report for definitions and threshold checks.

The 126-point audit found no quantitatively satisfactory shared fit. Its best
persistent candidate retains all 11 experimental oscillators but also oscillates at
645 uA and overpredicts the late 606 uA amplitude by about 32 times. The candidate
with the smallest three-current feature score still decays at the upper boundary.
The old reference loses persistence at 570 uA as well at the finer steps. These
findings and the numerical evidence are in the current audit bundle listed in
[`docs/ARCHIVE_INDEX.md`](docs/ARCHIVE_INDEX.md).

Use a cheaper inverse consistency check before another broad search:

```bash
neuristor analyze reconstruct-hysteresis \
  --config experiments/current/specimen_hysteresis_reconstruction.toml
```

It reconstructs temperature from measured resistive power, then replays the existing
hysteresis law on that prescribed temperature path. It varies C, Se, Cth and smoothing
one at a time, checks four gamma values, and repeats at three sampling steps. This
tests the resistance law without a forward optimization. Inferred temperatures and
effective resistances remain conditional on the thermal model and channel definitions.

## Run bundles and GitHub archive

### Budgeted joint resistance/dynamics fit

```bash
neuristor analyze fit-joint --config experiments/current/specimen_joint_inference.toml
```

This estimates eleven shared quantities, including all six major-loop resistance
parameters, using the raw same-device R(T) curve and nine representative current
records. Two short differential-evolution searches share cached evaluations and
starting points, followed by capped Powell refinement. They differ only in the
static-data penalty. The default budget is at most 240 objective calls, with at
most 216 distinct candidate evaluations before screening/cache savings. No
parallel agent or external compute service is used.

Thirteen other currents are excluded from optimization; they are a validation
subset, not a new blind dataset after earlier research. The archived references
and both winners are checked on all 22 currents at three smaller timesteps.
The bundle records separate static error, voltage, amplitude, frequency and
persistence metrics, source snapshots, optimizer termination and the full loss
definition. The TOML declares bounds, engineering weights and every budget.

Every command writes the same portable directory under `runs/`:

```text
runs/<run-id>/
├── run.json                # index, status, command, Git provenance
├── resolved_config.json    # exact inputs after overrides
├── metrics.json            # normalized scalar results
├── report.md               # human interpretation and limitations
├── traces.csv or sweep.csv # numerical evidence
└── figures/                # generated visual evidence
```

`runs/` is ignored by Git for exploratory work. When a run is worth preserving:

```bash
neuristor runs publish RUN_ID
git add public_jobs/RUN_ID
git commit -m "Archive <description>"
git push
```

Publishing copies the immutable bundle into tracked [`public_jobs/`](public_jobs/);
it does not silently make a Git commit. Historical `job.json` records and new
`run.json` bundles appear together in the dashboard. The bundle contract is specified
in [`docs/RUN_BUNDLES.md`](docs/RUN_BUNDLES.md).

## Models

### Voltage-driven Yuanhang circuit

The upstream-style circuit is

\[
C\frac{dV}{dt}=\frac{V_{in}-V}{R_{series}}-\frac{V}{R_{VO_2}(T,\mathcal H)},
\qquad
C_{th}\frac{dT}{dt}=\frac{V^2}{R_{VO_2}}-S_e(T-T_0).
\]

Its implementation and hysteresis memory are authoritative in
[`src/neuristor/model.py`](src/neuristor/model.py).

### Ideal-current extension

The laboratory-oriented current-source model is

\[
C\frac{dV}{dt}=I_{in}(t)-\frac{V}{R_{VO_2}(T,\mathcal H)}.
\]

The electrical RC and deterministic cooling subproblems use stable exact
frozen-coefficient updates. The implementation is in
[`src/neuristor/current_drive_sim.py`](src/neuristor/current_drive_sim.py).

The current model is deliberately ideal: source compliance, contact resistance,
cabling, and measurement impedance are not hidden inside an unused “series
resistance” parameter. Add those circuit elements explicitly if the measured voltage
includes them.

## Repository map

| Path | Purpose |
|---|---|
| `src/neuristor/model.py` | Voltage model and Yuanhang-faithful hysteresis |
| `src/neuristor/current_drive_sim.py` | Ideal-current solver and limiting cases |
| `src/neuristor/workflows.py` | Unit conversion, orchestration, reports, bundles |
| `src/neuristor/cli.py` | Unified terminal interface |
| `src/neuristor/dashboard.py` | Read-only archive and run comparison UI |
| `src/neuristor/config.py` | TOML loading, validation, and overrides |
| `src/neuristor/runs.py` | Run-bundle writer and historical archive registry |
| `experiments/` | Versioned, reusable simulation and sweep recipes |
| `tests/` | Physics convergence, hysteresis, CLI, and archive tests |
| `docs/` | Scientific interpretation, evidence, and specifications |
| `presets/` | Resistance fits and sample parameter sets |
| `public_jobs/` | Git-tracked, dashboard-readable run evidence |
| `legacy_scripts/` | Frozen pre-CLI one-offs; not the active interface |
| `references/` | Upstream implementation and source papers |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for dependency rules and
[`AGENTS.md`](AGENTS.md) for the safe change protocol used by humans and AI agents.

## Verification

Run before trusting or publishing a change:

```bash
pytest -q
neuristor validate
```

The test suite covers hysteresis reversals, current-source limiting behavior,
serial/vectorized equality, timestep convergence, TOML overrides, CLI execution, and
run-bundle discovery. Physics changes require an explicit convergence test; a plot
that merely “looks right” is not sufficient.

## Research context

Created by Gabriel Berezovsky under the supervision of PhD candidate Amir Gildor in
the Quantum Materials for Neuromorphic Computation Lab at the Technion.

### Editable all-current simulation lab

Run `neuristor playground` and open http://127.0.0.1:8502. Resistance-law controls
are separate from electrical/thermal controls, including ambient T0. Load a
saved fit, edit values and press **Run all measured currents**. A slider below
the plots selects the measured/simulated voltage overlay and simulated resistance.
Each run saves optional GIFs for all currents, a ZIP, an offline interactive HTML
comparison, numerical traces and a reproducible parameter recipe. These controls
are separate from the read-only archive dashboard.

```bash
neuristor analyze replay-lab --config experiments/current/specimen_lab_replay.toml
neuristor analyze fit-joint --config experiments/current/specimen_joint_expanded.toml
```

See [expanded search results and interface notes](docs/EXPANDED_SEARCH_20260922.md).

To export synchronized progressively drawn V(t), measured imposed I(t), and the
simulated R(T) trajectory over heating/cooling major branches, run:

```bash
neuristor analyze replay-lab --config experiments/current/specimen_scope_export.toml
```

The recipe uses the expanded stronger-static-fit parameters at 0.00625 ns.
Open `scope_viewer.html` in the output bundle for an offline current slider,
96-frame GIFs and full-trace PNGs. Playback frames are visual subsamples;
`traces.csv` retains the results sampled on the experimental time grid.

To export the **exact selected saved replay** without rerunning the simulator:

```bash
neuristor analyze export-scope --source runs/20260921_215248_interactive-specimen-laboratory-replay_eef073
```

This writes three vertically stacked panels (input current, voltage comparison,
evolving R–T hysteresis), with GIF filenames and current labels in amperes.
The filename current is the measured 50–250 ns pulse mean. `START_HERE.html`
selects among currents; `All_current_GIFs.zip` contains all animations.

For user-requested titles that retain the original numeric source-setting labels
and display µA, add `--original-labels-uA`. This is a display alias rather than
a unit conversion; a subtitle retains the actual measured plateau current, and
all numerical axes, filenames and data retain their measured-current meaning.

### Background search of settled oscillations

```bash
neuristor analyze fit-steady --config experiments/current/specimen_steady_multistart.toml
```

This standalone process targets original record labels 300–700 (actual measured
currents approximately 228–533 µA), scoring only 150–250 ns. It uses eight
independent DE/Powell restarts, a hard static R(T) log10-RMSE ceiling of 0.05,
and a 110-minute search plus a 10-minute verification allowance. Startup, phase
alignment and mean-voltage error are excluded from the objective. Full measured
input prehistory is retained. Frequency, robust amplitude, fundamental amplitude
and late amplitude retention determine the dynamic loss.

`status.json` and `checkpoint.json` update during execution; every unique candidate
is retained in `optimization_history.jsonl`. Create `STOP` inside the run directory
to stop gracefully between simulations. Final `parameters.csv`, `summary.csv`,
`verification.csv` and figures distinguish targets from out-of-band diagnostics.
All nine target records train the search; they are not a blind validation set.
Final candidates are checked at 0.025, 0.0125 and 0.00625 ns. The run does not call
an LLM or API and does not promise global optimality or a near-perfect result.

### Settled fit with a monotone proximity response

```bash
neuristor analyze fit-steady --config experiments/current/specimen_steady_monotone.toml
```

This repeats the two-hour, eight-restart 300–700 original-label search, with gamma
restricted to 0.60–0.98 and a diagnostic check of dTeff/dT at every integration
sample, including prehistory. Negative or nonfinite slopes exclude a candidate
from the winners. The diagnostic leaves the physics trajectory unchanged. Final
verification reports admissibility for each current and timestep, including old
references that fail the new condition. This is a sampled monotonicity constraint,
not proof of every hysteresis memory property. See the
[cooling-hook diagnosis](docs/HYSTERESIS_HOOK_DIAGNOSIS_20260922.md).

### Bounded tanh-proximity pilot

`neuristor analyze fit-steady --config experiments/current/specimen_steady_tanh_pilot.toml`
uses a separate `resistance.parameters.proximity_function = "tanh"` variant,
P(x)=1−tanh(kx), with the existing gamma coordinate interpreted as k. The default
Yuanhang function and old recipes are unchanged. The pilot searches k=0.03–0.98
for three minutes across three starts, then checks three timesteps. It retains
the major-loop fit and actual-path proximity checks. A 10,000 loss penalty per
failed target requires at least three prominent peaks, peaks in both window
halves, amplitude retention 0.75–1.333 and period CV <=0.2, in addition to the
frequency signal gate. This operational persistence test is not infinite-time
stability. Reference vectors in this bundle are explicitly reevaluated under
tanh, not claimed to reproduce their original model. No wider search follows
automatically. Almeida (2002), text preceding Eq. (19), discussed this function
but preferred another for their specimen.
