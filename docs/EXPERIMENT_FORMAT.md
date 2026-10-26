# Experiment TOML format

For the unified sample workflow, use the commented
`experiments/studies/sample_parameter_map.toml` recipe and
[sample-study specification](SAMPLE_STUDY.md). The core schema below is unchanged;
`[study]`, `[maps]` and `[animations]` add explicit unit/input/workload settings.
`neuristor study plan` validates those settings before any fit or simulation.

The stable schema version is `1`. Simulation recipes require the following top-level
fields:

| Field | Values | Meaning |
|---|---|---|
| `schema_version` | `1` | Configuration contract version |
| `name` | string | Human-readable run name |
| `kind` | `simulation` | Workflow type |
| `model` | `current` or `voltage` | Circuit boundary condition |
| `seed` | integer | Reproducible stochastic seed |

## Shared tables

For display-only loss comparisons, `neuristor analyze loss-scopes --config RECIPE
--frequency-run RUN --combined-run RUN` reuses the original frequency-only recipe
and validates that both parent runs share its fixed simulation settings. It uses
the selected vectors and finest saved verification timestep rather than the
recipe's initial guesses. `--frames` controls playback only. See
[all-input animation method](LOSS_ANIMATIONS_20261021.md).

| TOML path | Unit | Required | Meaning |
|---|---:|---:|---|
| `time.dt_ns` | ns | yes | Integration timestep |
| `time.duration_us` | us | yes | Simulated duration after pre-time |
| `initial.voltage_V` | V | no | Initial VO₂-node voltage |
| `initial.temperature_K` | K | no | Initial device temperature |
| `thermal.C_th_pJ_per_K` | pJ/K | yes | Thermal capacitance |
| `thermal.S_e_mW_per_K` | mW/K | yes | Environment thermal conductance |
| `thermal.T0_K` | K | no | Ambient/substrate temperature |
| `resistance.preset` | keyword/path | no | `yuanhang` or JSON preset path |
| `resistance.start_branch` | string | no | `insulator` or `metal` |
| `analysis.start_us` | us | no | Ignore earlier samples in metrics |
| `analysis.stop_us` | us | no | Ignore later samples in metrics |
| `output.root` | path | no | Run root, default `runs` |

Inline resistance overrides live under `[resistance.parameters]` and must match fields
of `YuanhangResistParams` exactly.

## Current model

| TOML path | Unit | Meaning |
|---|---:|---|
| `input.amplitude_uA` | uA | Ideal imposed current |
| `input.on_us` | us | Pulse start, default `0` |
| `input.off_us` | us | Optional pulse end |
| `time.pre_us` | us | Optional pre-pulse time shown as negative |
| `electrical.C_pF` | pF | Node capacitance; zero selects algebraic limit |
| `thermal.noise_W_sqrt_s` | W sqrt(s) | Optional thermal noise amplitude |

## Voltage model

| TOML path | Unit | Meaning |
|---|---:|---|
| `input.amplitude_V` | V | Constant source voltage |
| `electrical.R_series_kohm` | kOhm | External load/series resistor |
| `electrical.C_pF` | pF | Parasitic node capacitance |
| `thermal.couple_factor` | dimensionless | Array neighbor-coupling fraction |
| `thermal.C_th_factor` | dimensionless | Per-device thermal scaling |
| `thermal.noise_K_per_ns` | K/ns | Legacy upstream noise convention |

## Laboratory replay output

A current-model recipe with a `[replay]` table compares the simulated device with
every measured trace in `replay.data_directory`. Every `analyze replay-lab` run
always writes:

- `GIFs/`: one vertically stacked animation per current, with input current in A,
  measured and simulated voltage, and the evolving simulated R(T) trajectory;
- `Figures/`: the final frame of each animation as a PNG;
- `START_HERE.html`: an offline current selector;
- `All_current_GIFs.zip`: all animations; and
- `major_branches.csv`: the heating and cooling R(T) guides used in the figures.

`replay.scope_frames` controls visual playback sampling and defaults to 96. It does
not change the integration timestep or numerical tables. The old two-panel cursor
animations are produced only when `replay.legacy_cursor_animations = true`.
Pure current-source recipes can request the same synchronized layout without a
measured trace:

```toml
[output]
scope_gif = true
scope_frames = 96
```

This writes `GIFs/current_<amplitude_A>_A.gif` and a matching final-frame PNG in
`Figures/`. The panels show imposed input current in A, simulated voltage, and the
evolving simulated R(T) trajectory over the heating/cooling major branches.

## Sweeps

A sweep points to a simulation recipe and declares one to three axes:

```toml
schema_version = 1
name = "Frequency versus C and Cth"
kind = "sweep"
model = "current"

[sweep]
base_config = "../current/sweep_base.toml"
max_points = 500

[[sweep.axes]]
path = "electrical.C_pF"
values = [0.0, 25.0, 145.346]

[[sweep.axes]]
path = "thermal.C_th_pJ_per_K"
start = 5.0
stop = 100.0
step = 5.0
```

The Cartesian product is evaluated. One axis produces a line, two produce a heatmap,
and three produce heatmaps faceted by the first axis. Larger studies can still be
split into multiple recipes with explicit names.

## Steady extrema fitting and screened parameter maps

`neuristor analyze fit-steady` accepts `[steady] feature_method = "extrema"`.
Only complete cycles in `steady.window_ns` enter the voltage objective. Median
maximum-minus-minimum voltage defines Vpp; median maximum-to-maximum time defines
frequency. Missing persistent cycles and frequencies exceeding
`steady.maximum_frequency_MHz` receive a finite penalty. Static R(T) constraints
remain separate. Historical recipes retain the spectral method by default.

`neuristor analyze physical-maps` uses a current simulation recipe with `[maps]`.
`window_ns`, `maximum_frequency_MHz`, two successively smaller
`verification_dt_ns`, and `convergence_fraction` specify the screening protocol.
Each `[maps.axes.NAME]` defines `low`, `high`, `count` for a logarithmic grid and
optional `extra_values`, or an explicit `values` array. Names are `C_pF`,
`C_th_pJ_per_K`, and `S_e_mW_per_K`; their units match simulation fields.
`[maps.fixed_values]` selects one actual grid value per parameter for the 2D slices.
These are the initial HTML slider positions and the static PNG slices. Each HTML
slider can select all sampled values, updating both metrics and their per-slice
colorbar ranges. The original metric units remain MHz/mV; values are not converted
to dimensionless 0–1 scores. Empty slices have no displayed colorbar.
Blank cells are omitted candidates, not invented interpolated values or zero
frequency measurements. `--reuse-numerics DIRECTORY` writes a new rendering bundle
only when physical parameters, input current, measurement settings and timesteps
match the completed source bundle; original numerical CSV bytes are preserved.

## Path resolution and overrides

Preset and base-config paths resolve relative to the TOML file containing them.
`output.root` resolves relative to the repository root. `--set path=value` parses TOML
scalar syntax and only updates existing paths, so typos fail rather than creating
silent parameters.

### Controlled frequency and amplitude comparison

The paired recipes `specimen_frequency_only_20261020.toml` and
`specimen_frequency_amplitude_20261020.toml` use `loss_function = "relative_squared"`.
`frequency_weight` and `amplitude_weight` multiply squared relative errors
(defaults: 1 and 1, preserving the historical objective). Set amplitude weight
to zero for frequency-only optimization; amplitude remains a reported diagnostic.
Weights must be finite, nonnegative and not both zero. A missing frequency has
null loss/error and fails eligibility rather than becoming an invented frequency.

`target_current_range_uA` selects records by baseline-corrected measured current
averaged over 50–250 ns, not by filename numbers. `fixed_parameters` names
internal search coordinates in `joint_inference.NAMES` to keep at their baseline
values; `fixed_resistance_restarts` still controls the six static quantities.
`maximum_evaluations` caps unique candidate evaluations as well as the time cap.
The supplied pair fixes all six R(T) values and T0, leaving four physical unknowns.
Tau_th is an internal coordinate for Cth/Se, not another independent parameter.

Run either with `neuristor analyze fit-steady --config FILE.toml`. Each bundle
records measured currents, feature definitions, search history and three-step
all-current verification. `objective_loss` is null unless every selected target
passes the cycle, frequency, temperature and configured proximity screens.
`mean_dynamic_loss` remains a diagnostic over defined per-record errors and must
not be mistaken for an accepted all-target objective on a rejected attempt.

`neuristor analyze compare-steady --frequency-run RUN --combined-run RUN`
writes a separate comparison bundle from completed saved verification. It checks
matching protocols and measured targets, cross-evaluates both scientific losses,
and plots the lowest, middle and highest target currents selected by current
rank. Missing frequencies stay undefined; diagnostics show their denominator.
Plots and headings use measured current in µA, while acquisition labels remain
in the provenance tables. Both source bundles remain unchanged.

### Optimization sensitivity and gradient audit

`neuristor analyze audit-optimizer --config
experiments/current/specimen_optimizer_audit_20261020.toml` probes both saved
20 October candidates. `[optimizer_audit]` declares completed parent `sources`,
`normalized_steps`, integration `dt_ns`, two `gradient_steps`,
`gradient_dt_ns`, `gradient_max_evaluations`, `gradient_max_iterations`,
`gradient_max_line_search` and a total wall-time cap. Coordinates use the same
normalized log C, Se, Cth/Se and gamma ranges as the existing search. A perturbation
of Se at fixed Cth/Se therefore changes Cth; no fifth independent parameter is added.

Forward, backward and centered slopes of the finite optimizer score are recorded
separately from slopes of a defined all-target scientific loss. Missing frequencies
make the latter undefined. Differences across step sizes and timesteps are raw
diagnostics, not a theorem of differentiability. If a profile point is eligible,
it supplies the gradient starting point; otherwise L-BFGS-B explicitly tests only
the same finite rejection score. Both cases record the count of feasible and
infeasible gradient evaluation requests, the termination reason and the best
sampled candidate. Optimizer convergence is not scientific acceptance.

The expanded paired recipes `specimen_frequency_only_expanded_20261020.toml` and
`specimen_frequency_amplitude_expanded_20261020.toml` preserve the original
objectives and bounds, use four fixed-static restarts and a 768-evaluation cap,
and share both earlier best attempts as starting vectors. `initial_seed_count`
(default 3) controls how many declared seed vectors enter each initial population.
All selected outcomes are checked on all 22 records at the same three timesteps.
The display, data and physical equations remain unchanged.

`neuristor analyze review-optimizer --audit-run RUN --frequency-run RUN
--combined-run RUN` creates a separate reviewed comparison of the old centers,
expanded searches and gradient-selected candidates. It computes saved per-input
frequency slopes even where another record makes the full loss undefined,
checks identical measured features across methods, and retains all-source hashes.
The direct frequency-profile figure uses the middle measured current and
capacitance perturbations selected before inspecting errors. Scientific loss,
finite rejection score, and optimizer termination remain separate fields.

## Fresh parameter-trend validation

```bash
neuristor analyze validate-trends --config experiments/current/validate_parameter_trends_20261020.toml
```

`[trend_validation]` declares anchors (C pF, Cth pJ/K, Se mW/K), operating
scenarios (current µA, ambient K), gamma values, local multipliers, common-scale
multipliers, three decreasing `dt_ns`, initial/maximum nominal horizons,
convergence and direction thresholds, batch/sample budgets, and optional further
timestep halvings. Starts use the base recipe's voltage, ambient temperature and
branch; thermal noise, prehistory and finite pulses are rejected. The measured
resistance interval remains explicit in `[resistance.parameters]`.

Both fixed-physical-dt and co-scaled-dt protocols multiply observation duration
by the common capacitance scale, preserving comparable startup time. Repeated
physical integrations are cached without losing logical case IDs. Every case,
including rejected and unresolved trajectories, remains in the bundle. Numerical
feature comparisons, sampled unaligned clock errors and all source snapshots
are exported. `[trend_validation]` validation is enforced by the workflow.
See [methods and results](FRESH_PARAMETER_TRENDS_20261020.md).

Use `--reuse-integrations COMPLETED_BUNDLE` with an otherwise identical trend
recipe to extend only the observation cap. The loader checks the source's
completed status, base SI physics, all numerical settings except duration/batch
budgets, exact solver/extrema bytes and the integration/classifier syntax tree.
Raw numerical records and sampled scaling paths are cached; parent tables,
hashes and frozen source remain in the new bundle. Missing longer integrations
are new simulations; parent results are never edited. A lower observation cap
or changed gamma, anchors, current, ambient, tolerances or sampled-path controls
is rejected. Metrics distinguish new from reused integrations.
