# Experiment recipes

Every simulation and sweep is described by a versioned TOML file. The file is
the scientific input; the CLI command is only the executor. Units are written
into parameter names so a recipe can be reviewed without reading Python.

- `current/`: ideal-current-source simulations.
- `voltage/`: Yuanhang voltage-source circuit simulations.
- `sweeps/`: Cartesian parameter studies based on one simulation recipe.
- `studies/`: complete measured R(T) → screened map → standalone GIF-enabled HTML.

New sample users should start with [studies/sample_parameter_map.toml](studies/sample_parameter_map.toml).
The [paper reproduction index](../paper/README.md) identifies supporting recipes
without duplicating them or assigning premature final paper figure numbers.

Run a recipe from the repository root:

```bash
neuristor simulate current --config experiments/current/nonzero_voltage_valley.toml
neuristor simulate voltage --config experiments/voltage/yuanhang_oscillator.toml
neuristor sweep run --config experiments/sweeps/capacitance_vs_thermal_600uA.toml
```

Temporary changes do not require a new file. Repeat `--set` with a dotted path:

```bash
neuristor simulate current \
  --config experiments/current/nonzero_voltage_valley.toml \
  --set input.amplitude_uA=700 \
  --set electrical.C_pF=10
```

The resolved recipe, including overrides, is copied into every run bundle.
