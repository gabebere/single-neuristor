# Repo context for Codex: VO₂ neuristor simulator + analysis

## What this repo is
A Python codebase that simulates a single VO₂ neuristor (and optionally an Nx×Ny lattice) using a paper-faithful model (Zhang–Qiu–Di Ventra / “Yuanhang”).
Primary outputs: time traces of device voltage/current/temperature/resistance and hysteresis variable `g(T)`.

## Design goal (refactor request)
We want one authoritative module: `model.py`, acting as the **single source of truth** for:
- parameter dataclasses (resistance + circuit/thermal)
- hysteretic resistance model `R(T)` and `g(T)` (major + minor loops)
- simulator integration (Euler)
- shared analysis helpers (series selection, spike detection, oscillation checks)
- generic sweep utilities (1D sweeps + find oscillatory band)

Other files (analysis scripts and CLIs) should import from `model.py` instead of duplicating the simulator.

## Key files today
- `histogram.py`: currently contains the full simulator implementation + CLI plotting.
- `plots.py`: analysis utilities; currently imports simulator/helpers from `model.py`.
- `model.py`: currently empty or not yet authoritative (to be implemented).

## Model summary (must not change)
### Electrical ODE
Let `V` be the node/device voltage across VO₂, `Vin` applied bias, `R_series` load resistor, `C_par` capacitor, and `R_vo2(T,g)` device resistance.
- `dV/dt = (Vin - V)/(R_series*C_par) - V/(R_vo2*C_par)`

### Thermal ODE
Let `T` be device temperature (Kelvin). Let `C_th` be thermal capacitance, `S_env` environmental thermal conductance,
`S_couple` coupling conductance to neighbors, and `laplacian(T)` the discrete Neumann-edge laplacian for lattice coupling.
Let `noise_strength` be additive noise term (Gaussian), and divide by `Cth_factor`.
- `P = V^2 / R_vo2`
- `dT/dt = ( P - S_env*(T - T_base) + S_couple*laplacian(T) )/C_th + noise`
- then `/ Cth_factor`

### Resistance / hysteresis law
- `R(T) = R0 * exp(Ea_over_k / T) * g(T) + Rm`, where `Rm = Rm0 * Rm_factor`
- `g(T)` uses Almeida-style hysteresis with:
  - `delta = sign(dT/dt)` selecting heating/cooling branch
  - last reversal state `(T_r, g_r)` and proximity shift `T_pr`
  - window `P(x; gamma)` controlling minor-loop proximity
- Temperatures are clamped to `[T_min_K, T_max_K]` inside the hysteresis evaluation for stability.

## Units / conversions (must not change)
- Temperature: Kelvin everywhere
- `R_series_kohm` -> Ohm: `* 1e3`
- `C_par_pF` -> Farad: `* 1e-12`
- `Cth_mW_ns_per_K` -> J/K: `* 1e-12` (because mW*ns = 1e-12 J)
- `Sth_mW_per_K` -> W/K: `* 1e-3`

## Output format (must not change)
Return a `SimOut` dict with keys:
- `time_s`, `V_node`, `I_load`, `I_vo2`, `T_K`, `R_vo2`, `g`, `grid_shape`
For single device: each trace is `List[float]`
For multi-device lattice: each trace is `List[List[float]]`, one list per device (consistent with existing code).

## Public API desired in model.py
Expose these as public functions (no leading underscore):
- `simulate_yuanhang(...) -> SimOut`
- `series_first(series) -> List[float]`
- `series_mean(series) -> List[float]`
- `detect_spike_times(time_s, I_vo2, threshold_A=1e-3) -> List[float]` (peak detect on |I|)
- `is_oscillatory(simout, t_start_us=25, t_end_us=300, threshold_A=1e-3, min_spikes=4) -> bool`
- sweep utilities:
  - `sweep_1d(run_one, values) -> Dict[value, result]`
  - `find_oscillatory_band_1d(run_sim, start, stop, step, osc_check=is_oscillatory) -> (min, max)`

## Refactor plan
1) Move/copy simulator core + hysteresis + dataclasses from `histogram.py` into `model.py`.
2) Update `plots.py` to import from `model.py` and replace calls to underscored helpers.
3) Optionally update `histogram.py` to import from `model.py` for simulation, leaving plotting/CLI in histogram.py.

## Guardrails
- Don’t “improve” the physics. Don’t change numerical methods unless asked.
- Keep diffs minimal. Avoid renaming variables.
- Avoid circular imports.
- After refactor: `python plots.py --help` should run without import errors.
