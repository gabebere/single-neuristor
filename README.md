# Single VO₂ Neuristor

A reproducible research codebase for current- and voltage-driven VO₂ neuristor
simulations, laboratory-trace analysis, parameter sweeps, and evidence archival.

The project now has one workflow:

```text
human-readable TOML  ->  neuristor CLI  ->  tested physics  ->  portable run bundle
                                                          ->  archive dashboard
```

The original Yuanhang Zhang implementation is preserved under
[`references/yuanhangzhang98-collective_dynamics_neuristor-217d4f0/`](references/yuanhangzhang98-collective_dynamics_neuristor-217d4f0/).
The last known-working pre-refactor repository is permanently tagged
[`v0.1.0-working-baseline`](https://github.com/gabebere/single-neuristor/tree/v0.1.0-working-baseline).

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
present traces do not resolve a positive value. Thermal capacitance still requires
an independently measured thermal time constant.

The oscillatory traces can later constrain the minor-loop parameter `gamma`, but only
after `C`, `C_th`, `S_e`, and `T0` are fixed well enough to reconstruct resistive
current, power, and the latent temperature trajectory. The low- and high-current
traces constrain the other parameters; only repeated transition reversals carry
meaningful information about `gamma`.

## Run bundles and GitHub archive

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
