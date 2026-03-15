# Single-Neuristor vs. Yuanhang Zhang Thermal Neuristor Models

## Overview
- **Scope:** Compare the single-device simulator in `single-neuristor/singleneuristor.py` with the network model published by Yuanhang Zhang et al. in `yuanhangzhang98-collective_dynamics_neuristor-217d4f0/model.py`.
- **Focus areas:** (1) resistance / hysteresis formulation and (2) coupled electrical–thermal ordinary differential equations (ODEs).
- **Key context:** Your code targets a single VO₂ device driven by a fixed supply, while the paper implements a GPU-friendly reservoir of many thermally coupled neuristors for statistical studies.

## Resistance and Hysteresis Models

### `single-neuristor/singleneuristor.py`
- **Window function:** Implements Π(x;γ) = 0.5·(1 − sin(γx))·(1 + tanh(π − 2πx)) with clipping safeguards to keep Π ∈ [0,1] (`singleneuristor.py:25-41`).
- **Branch tracking:** Heating/cooling direction `d`, reversal temperature `Tr`, and normalization `Tpr` follow the formulation in “Modeling of the hysteretic metal–insulator transition,” updated whenever the temperature derivative changes sign (`singleneuristor.py:81-105`).
- **Phase fraction:** On each branch, `g(T) = 0.5 + 0.5·tanh[b( d·w/2 + Tc − (T + Tpr·Π ))]`, explicitly clamped to [0,1] (`singleneuristor.py:107-110`).
- **Resistance mixture:** Arrhenius insulating state `Rs = R0·exp(Ea/(k·TK))` is blended with fixed metallic resistance `Rm` as `R = g·Rs + (1 − g)·Rm`, so `g→1` recovers the full insulating resistance and `g→0` recovers the metallic floor (`singleneuristor.py:111-118`).
- **Units:** Temperatures are stored in °C (shifted to Kelvin only inside the Arrhenius term), and parameters retain SI units (Ohm, Joule, etc.) (`singleneuristor.py:7-15`).

### `yuanhangzhang98-collective_dynamics_neuristor-217d4f0/model.py`
- **Window function:** Uses Π(x;γ) = 0.5·(1 − sin(γx))·(1 + tanh(π² − 2πx)), i.e., the tanh window is centered at π² rather than π, which narrows the window and delays the onset of the minor loop (`model.py:7-8`).
- **Branch tracking:** Maintains vectorized tensors for `delta` (heating/cooling sign), `Tr`, `gr`, and `Tpr`; reversals are triggered only when |ΔT| > 0.01 K to avoid noise-driven flips, and the proximity offset `Tp` is applied only after a reversal flag is set (`model.py:24-66`).
- **Phase fraction:** Similar tanh form but multiplied by `self.reversed`, so before any reversal each device follows the major loop exactly; minor-loop corrections start only after a true reversal has been observed (`model.py:63-66`).
- **Resistance formula:** `R(T) = [R0·exp(Ea/T)·g(T) + Rm] / 1000`, where `Ea` is already in Kelvin units (≈5220 K) and the result is converted to kΩ (`model.py:67-70`). Metallic resistance is **added**, not weighted by (1 − g), so the metallic branch acts as a constant offset.
- **Units:** All temperatures stay in Kelvin; electrical quantities mix SI volts with kΩ and pF, reflecting the paper’s empirical parameterization.

### Summary of Resistance-Model Differences
- **Window placement:** Your Π(x) centers the tanh at π, whereas theirs centers it at π², widening the temperature range before proximity feedback engages.
- **Minor-loop activation:** You always evaluate Π (with safeguards) even without explicit reversals, while their model gates Π by `self.reversed`, delaying hysteretic corrections until a real reversal occurs.
- **Mixture vs. additive R:** Your mixture enforces `R ∈ [Rm, Rs]`. Their additive formula effectively keeps a metallic shunt in series with the insulating branch, so the on/off ratio collapses to `(R0·exp(Ea/T) + Rm) / Rm ≈ 1 + Rs/Rm`.
- **Temperature units:** Celsius vs. Kelvin handling requires care when porting parameters; their `Ea` already absorbs Boltzmann’s constant.

### Additional Hysteresis Implementation Details
- **Normalization sign:** `single-neuristor/singleneuristor.py:91-105` now matches Zhang’s `Tpr = d·w/2 + Tc − (1/b)·atanh(2gr − 1) − Tr` expression (`model.py:53-55`). Earlier revisions added the atanh term, which mirrored the minor-loop offsets.
- **Temperature clamping:** The Torch model clamps both reversal detection and the resistance lookup to `305–370 K` (`model.py:39-70`), preventing runaway Arrhenius growth; your solver lets T wander and only guards the tanh via `_clamp01`.
- **Width/metal scaling knobs:** Zhang exposes `width_factor` and `Rm_factor` so that w and Rm can vary per device (`model.py:12-24`). The single-device script fixes w and Rm at compile time.
- **Window safeguards:** You soft-clip the normalized proximity `x` before calling Π (`singleneuristor.py:33-41`), while their implementation leaves tanh/sin unclipped but regularizes the denominator with `Tpr + 1e-6` (`model.py:63-65`).
- **Last-temperature memory:** Zhang tracks `T_last` and requires |ΔT| > 0.01 K before flipping `delta` (`model.py:38-44`); your direction logic flips as soon as the discrete derivative changes sign, so it can chatter under noisy heating.

## Electrical ODEs

### `single-neuristor/singleneuristor.py`
- **Circuit topology:** Single node with series load `R_load` feeding the VO₂ branch shunted by a parasitic capacitance `C_par` (`singleneuristor.py:172-210`).
- **ODE:** `dVn/dt = (Vin − Vn)/(R_load·C_par) − Vn/(R_vo2·C_par)` derived from KCL, integrated via forward Euler with fixed timestep (`singleneuristor.py:185-273`).
- **Outputs:** Stores node voltage, load/device currents, instantaneous resistance, temperature, and `g` for later plotting (`singleneuristor.py:250-301`).

### `yuanhangzhang98-collective_dynamics_neuristor-217d4f0/model.py`
- **Circuit topology:** Array of neuristors, each with its own node voltage `V1`, biased by `V0` through a fixed series resistor `R0` and capacitor `C0` (145 pF). States are batched for GPU execution (`model.py:75-124`).
- **ODE:** `dV1/dt = V0/(R0·C0) − V1/(R0·C0) − V1/(R·C0)`; mathematically identical to yours but expressed in kΩ·pF units and broadcast across N devices (`model.py:109-124`).
- **Noise & batches:** No electrical noise is added, but the structure allows multiple inputs/batches simultaneously, unlike the single-scalar loop in your script.

### Electrical Differences
- Your simulator enforces SI units and exposes `Vin` sweeps directly, which is convenient for hardware-matching studies. The paper’s code fixes `V0`, `R0`, and `C0` constants inside the class and expects batched tensors; adapting it to a single neuristor requires unrolling those tensors and ensuring consistent units.

## Thermal ODEs

### `single-neuristor/singleneuristor.py`
- **Heating term:** `P_vo2 = Vn² / R_vo2`; temperature evolves via `dT/dt = [−G_th·(T − T_base) + η·P_vo2]/C_th` in Celsius units (`singleneuristor.py:185-277`).
- **Assumptions:** No stochastic noise, no spatial coupling, and the thermal capacitance/conductance are scalars supplied by `ThermalElectricalParams`.

### `yuanhangzhang98-collective_dynamics_neuristor-217d4f0/model.py`
- **Heating term:** Uses `QR = I²·R` (with R in kΩ) as the power input. Thermal evolution couples neighbors through a discrete Laplacian and includes configurable Gaussian noise (`model.py:115-124`).
- **Equation:** `dT/dt = { [QR − S_env·(T − T_base) + S_couple·∇²T]/C_th + σ·ξ } / Cth_factor`, where `S_couple = couple_factor·S_th` controls lateral heat spreading and `Cth_factor` rescales the net thermal inertia.
- **Dimensionality:** The 2D variant (`Circuit2D`) pads the lattice and applies a 4-neighbor Laplacian (`model.py:165-192`).

### Thermal Differences
- Your model treats the neuristor as thermally isolated except for a single bath conductance, matching a standalone device on a probe station. The paper models mesoscale coupling (neighbor heat sharing), environmental leakage adjustment (`S_env`), and noise-driven avalanches—critical for collective dynamics but unnecessary for a single device.
- **Numerical considerations:** Your scalar integrator is simple but may require very small `dt` for stiff regimes. The Torch implementation benefits from GPU acceleration and vectorized updates but also relies on clamping (`T.clamp(305, 370)`) to avoid divergence—behavior you currently handle via `_clamp01` and `_EPS` safeguards instead of explicit T bounds.

## Parameter Scaling & Noise Controls
- **Environmental vs. coupling loss:** Zhang decomposes the thermal leakage into `S_env = S_th·(1 − 2·d·couple_factor)` and `S_couple = couple_factor·S_th` (`model.py:82-91, 121-124`), letting batches explore different amounts of bath loss versus neighbor coupling; you have a single scalar `G_th`.
- **Effective heat capacity:** `Cth_factor` rescales the total thermal inertia per device (`model.py:84-107`), whereas your implementation hard-codes `C_th`.
- **Noise injection:** `noise_strength * randn` is added inside the thermal derivative before dividing by `Cth_factor` (`model.py:122-123`), supporting stochastic avalanches; your solver is deterministic.
- **Input hooks:** `Circuit.set_input` allows hot-swapping the bias tensor `V0` or `Cth_factor` without rebuilding the circuit (`model.py:103-107`); your CLI rebuilds the whole simulation for each Vin sweep.
- **Spatial variants:** `Circuit` pads the 1D chain to build a discrete Laplacian (`model.py:109-124`), and `Circuit2D` upgrades that to a 4-neighbor stencil with replicate padding (`model.py:165-191`). Your code has no notion of dimensionality or boundary conditions.

## Practical Implications
- **Parameter translation:** Directly copying `Ea`, `Rm`, or `w` between codes requires unit conversions (Kg vs. eV, °C vs. K). Failing to convert leads to large resistance errors.
- **Hysteresis behavior:** Because your model always blends Rs and Rm, it can represent continuous intermediate resistances, while theirs effectively keeps a metallic “floor” in series, which may understate the insulating high-R state when `g→1`.
- **Dynamic richness:** The paper’s Laplacian and noise terms are responsible for avalanche statistics and long-range order; omitting them (as in your single-device model) means you cannot reproduce those collective effects without explicitly extending the thermal equation.

## Suggested Next Steps
1. Decide whether you need additive metallic resistance (paper) or convex blending (current code) and adjust `update_R` accordingly.
2. If you plan to study coupled neuristors, extend your thermal ODE with Laplacian coupling and optional stochastic forcing, mirroring the structure in `model.py`.
3. Align parameter units (especially `Ea`, `Tc`, `R0`, and `C_th`) before running cross-validation against published results.
