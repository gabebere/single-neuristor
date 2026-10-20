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
