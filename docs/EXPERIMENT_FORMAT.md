# Experiment TOML format

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

## Path resolution and overrides

Preset and base-config paths resolve relative to the TOML file containing them.
`output.root` resolves relative to the repository root. `--set path=value` parses TOML
scalar syntax and only updates existing paths, so typos fail rather than creating
silent parameters.
