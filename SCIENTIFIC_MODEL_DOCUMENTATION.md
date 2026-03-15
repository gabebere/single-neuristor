# Single-Neuristor Scientific Documentation

## 1. Scope and goal

This document explains, in a traceable way, how the equations in this repository were obtained and implemented:

1. The original continuous-time equations from the reference papers.
2. The meaning of every term and parameter.
3. The assumptions used to move from voltage drive to current drive.
4. The ideal-current-source limit.
5. The exact discretization used in code.
6. The step-by-step simulation and fitting algorithms.

Primary code sources:

- `model.py` (authoritative voltage-driven simulator and hysteresis model)
- `current_drive_sim.py` (current-driven simulator)
- `resistance_custom_analysis.py` (specimen-specific hysteresis/resistance fitting)
- `app.py` (paper presets, ideal-source convention, diagnostics)

Primary paper sources:

- `collective Dynamics.pdf` (Nature Communications 2024, Zhang et al.)
- `2582_1.pdf` (Optical Engineering 2002, de Almeida et al.)

---

## 2. Scientific origin: where the model comes from

### 2.1 2002 hysteresis model (de Almeida et al., `2582_1.pdf`)

The resistance model starts from a semiconducting Arrhenius law and an effective-medium mixture:

\[
R_s(T) = R_0 \exp\left(\frac{E_a}{k_B T}\right)
\]

\[
R(T) \approx g(T)\,R_s(T) + R_m
\]

where \(g(T)\in[0,1]\) is the semiconducting volume fraction (with hysteresis and minor loops).

The paper defines:

- A major-loop model with parameters \(w, T_c, \beta\), and branch sign \(\delta\).
- Reversal-state memory through \((T_r, g_r)\) and \(T_{pr}\).
- A proximity function \(P(x)\) with parameter \(\gamma\), producing realistic minor loops.

The canonical proximity function used in this project lineage is:

\[
P(x)=\frac{1}{2}\left(1-\sin(\gamma x)\right)\left(1+\tanh(\pi^2-2\pi x)\right)
\]

### 2.2 2024 thermal-neuristor network model (Zhang et al., `collective Dynamics.pdf`)

The network dynamics are given in the paper as:

\[
C\frac{dV_i}{dt}=\frac{V^{in}_i}{R^{load}_i}-V_i\left(\frac{1}{R_i}+\frac{1}{R^{load}_i}\right)
\]

\[
C_{th}\frac{dT_i}{dt}=\frac{V_i^2}{R_i}-S_e(T_i-T_0)+S_c\nabla^2T_i+\sigma\eta_i(t)
\]

with resistance \(R_i=R(T_i)\) computed from the hysteresis model above.

The methods section also states Euler-Maruyama integration with \(dt=10\) ns and reports parameter values (C, \(R_{load}\), \(C_{th}\), \(S_e\), \(S_c\), \(R_0\), \(E_a\), \(R_m\), \(w\), \(T_c\), \(\beta\), \(\gamma\)).

### 2.3 This repository

This project refactors and extends the above into:

1. A single-source-of-truth voltage-driven simulator (`model.py`).
2. A current-driven simulator (`current_drive_sim.py`) derived from the same KCL/thermal physics and hysteresis.
3. A specimen-specific resistance fitting pipeline (`resistance_custom_analysis.py`) tied to your measured data.

---

## 3. Nomenclature and unit mapping

| Symbol | Physical meaning | Units | Code variable(s) |
|---|---|---|---|
| \(V\), \(V_i\) | Node voltage across VO2/capacitor | V | `Vn`, `V_node`, `V_vo2` |
| \(V_{in}\) | Applied voltage source | V | `Vin`, `V_bias` |
| \(I_{in}(t)\) | Injected current source waveform | A | `I_in` |
| \(R_{load}\), \(R_{series}\) | Series/load resistor in voltage-drive form | Ohm | `R_series_ohm` |
| \(R_{out}\) | Shunt/output resistance in current-drive Norton form | Ohm | `R_out_ohm` |
| \(C\) | Electrical capacitance | F | `C_par_F`, `C_F` |
| \(T\), \(T_i\) | Device temperature | K | `T_K`, `T` |
| \(T_0\) | Ambient/base temperature | K | `T_base_K`, `T0_K` |
| \(C_{th}\) | Thermal capacitance | J/K | `C_th_J_per_K`, `Cth_J_per_K` |
| \(S_e\) | Thermal conductance to environment | W/K | `S_e_W_per_K`, `S_env` |
| \(S_c\) | Thermal coupling to neighbors | W/K | `S_couple` |
| \(R(T)\) | VO2 resistance | Ohm | `R_vo2`, `R` |
| \(R_0\) | Arrhenius prefactor | Ohm | `R0` |
| \(E_a/k_B\) | Activation-energy-over-Boltzmann term | K | `Ea_over_k` |
| \(R_m\) | Metallic resistance floor | Ohm | `Rm = Rm0*Rm_factor` |
| \(g(T)\) | Hysteretic semiconducting fraction | 1 | `g` |
| \(w\) | Hysteresis loop width | K | `w`, `w_eff` |
| \(T_c\) | Hysteresis center/critical temperature | K | `Tc_K` |
| \(\beta\) | Branch sharpness | 1/K | `beta` |
| \(\gamma\) | Proximity-window parameter | 1 | `gamma` |
| \(\delta\) | Branch sign (heating/cooling) | 1 | `delta` |
| \(T_r\), \(g_r\) | Last reversal temperature/fraction | K, 1 | `Tr`, `gr` |
| \(T_{pr}\) | Reversal-derived proximity scale | K | `Tpr` |
| \(\sigma\eta\) | Thermal noise term | model-dependent | `noise_strength` or `sigma_W_sqrt_s` |

### 3.1 Unit conversions in code

In `model.py`, user-facing parameters are converted to SI:

- `R_series_kohm -> R_series_ohm = *1e3`
- `C_par_pF -> C_par_F = *1e-12`
- `Cth_mW_ns_per_K -> Cth_J_per_K = *1e-12`
- `Sth_mW_per_K -> W_per_K = *1e-3`

This keeps the integrator in SI units.

---

## 4. Base continuous-time equations used in this project

## 4.1 Voltage-driven model (non-discretized)

### Electrical equation

\[
C\frac{dV}{dt}=\frac{V_{in}}{R_{series}}-V\left(\frac{1}{R(T,\mathcal{H})}+\frac{1}{R_{series}}\right)
\]

Equivalent rearrangement used in `model.py`:

\[
\frac{dV}{dt}=\frac{V_{in}-V}{R_{series}C}-\frac{V}{R(T,\mathcal{H})C}
\]

### Thermal equation (single device)

\[
C_{th}\frac{dT}{dt}= \underbrace{\frac{V^2}{R(T,\mathcal{H})}}_{\text{Joule heating}}
-\underbrace{S_e(T-T_0)}_{\text{cooling to bath}}
+\underbrace{\sigma\eta(t)}_{\text{optional noise}}
\]

### Thermal equation (array/lattice)

\[
C_{th}\frac{dT_i}{dt}=
\frac{V_i^2}{R_i}
-S_e(T_i-T_0)
+S_c\nabla^2T_i
+\sigma\eta_i(t)
\]

with Neumann boundary Laplacian in the implementation.

## 4.2 Hysteretic resistance model (non-discretized)

This repository uses the Almeida-type formulation (major + minor loops):

\[
R(T,\mathcal{H})=R_0\exp\left(\frac{E_a/k_B}{T}\right)g(T,\mathcal{H})+R_m
\]

\[
g(T,\mathcal{H})=
\frac{1}{2}+\frac{1}{2}\tanh\left[
\beta\left(
\delta\frac{w}{2}+T_c-\left(T+T_p\right)
\right)\right]
\]

\[
T_p = T_{pr}\,P\!\left(\frac{T-T_r}{T_{pr}}\right),
\quad
P(x)=\frac{1}{2}(1-\sin(\gamma x))(1+\tanh(\pi^2-2\pi x))
\]

\[
T_{pr}=\delta\frac{w}{2}+T_c-\frac{1}{\beta}\operatorname{arctanh}(2g_r-1)-T_r
\]

where:

- \(\delta=+1\) on heating, \(-1\) on cooling.
- \((T_r,g_r)\) are updated at each detected reversal.
- \(g\) is clipped to \([0,1]\) in code.
- \(T\) is clamped to `[T_min_K, T_max_K]` in the evaluator for stability.

---

## 5. Derivation of the current-driven module from the original model

This is the key transition your professor asked to document.

## 5.1 Start from voltage-drive KCL

From Section 4.1:

\[
C\frac{dV}{dt}=\frac{V_{in}}{R_{series}}-V\left(\frac{1}{R(T,\mathcal{H})}+\frac{1}{R_{series}}\right)
\]

Define the source current in Thevenin form:

\[
I_{src}\equiv \frac{V_{in}}{R_{series}}
\]

Then:

\[
C\frac{dV}{dt}=I_{src}-V\left(\frac{1}{R(T,\mathcal{H})}+\frac{1}{R_{series}}\right)
\]

## 5.2 Generalize to explicit current source (Norton-equivalent form)

Replace \(I_{src}\) by an injected waveform \(I_{in}(t)\), and keep a finite source/output resistance \(R_{out}\):

\[
\boxed{
C\frac{dV}{dt}=I_{in}(t)-V\left(\frac{1}{R(T,\mathcal{H})}+\frac{1}{R_{out}}\right)
}
\]

This is exactly what `current_drive_sim.py` implements.

## 5.3 Ideal-current-source limit

Ideal source means infinite output resistance:

\[
R_{out}\rightarrow\infty \quad\Rightarrow\quad \frac{1}{R_{out}}\rightarrow0
\]

So the electrical equation becomes:

\[
\boxed{
C\frac{dV}{dt}=I_{in}(t)-\frac{V}{R(T,\mathcal{H})}
}
\]

In this repo/UI, ideal source is encoded as `R_out_ohm <= 0`, and internally mapped to `1/R_out = 0`.

## 5.4 Current-drive thermal equation

The current module keeps the same thermal physics structure:

\[
C_{th}\frac{dT}{dt}=\frac{V^2}{R(T,\mathcal{H})}-S_e(T-T_0)+\sigma\eta(t)
\]

with optional stochastic term.

---

## 6. Discretization used in code

## 6.1 Voltage-driven simulator (`model.py`)

Explicit Euler:

\[
V_{n+1}=V_n+\Delta t\left[
\frac{V_{in}-V_n}{R_{series}C}
-\frac{V_n}{R_n C}
\right]
\]

\[
T_{n+1}=T_n+\Delta t\left\{
\frac{1}{C_{th}}
\left[
\frac{V_n^2}{R_n}
-S_e(T_n-T_0)
+S_c(\nabla^2T)_n
\right]
\;+\;\text{noise}_n
\right\}\frac{1}{Cth\_factor}
\]

Notes:

- `R_n = R(T_n, H_n)` from hysteresis evaluator.
- `noise_strength` path in `model.py` preserves original project behavior (`noise * dt` scaling), not explicit \(\sqrt{\Delta t}\) Brownian scaling.
- Temperature is clamped inside resistance/hysteresis evaluation, not globally.

## 6.2 Current-driven simulator (`current_drive_sim.py`)

Electrical update (Euler):

\[
V_{n+1}=V_n+\frac{\Delta t}{C}\left[I_{in,n}-V_n\left(\frac{1}{R_n}+\frac{1}{R_{out}}\right)\right]
\]

Thermal update (Euler-Maruyama style in this module):

\[
T_{n+1}=T_n+\frac{\Delta t}{C_{th}}\left[\frac{V_n^2}{R_n}-S_e(T_n-T_0)\right]
+\frac{\sigma}{C_{th}}\sqrt{\Delta t}\,\xi_n,\quad \xi_n\sim\mathcal{N}(0,1)
\]

This matches the current-input plan and the module docstring.

---

## 7. Hysteresis state-update algorithm (discrete logic)

At each step, the evaluator in `HysteresisArray` performs:

1. Clamp \(T\) to `[T_min_K, T_max_K]`.
2. Compute \(\Delta T = T - T_{last}\).
3. If \(|\Delta T| \le \text{reversal_threshold}\), keep branch state.
4. Else set proposed branch \(\delta_{new}=sign(\Delta T)\).
5. If branch flips (\(\delta_{new}\neq\delta\)):
   - Compute current \(g\) at reversal.
   - Store \(g_r, T_r\).
   - Recompute \(T_{pr}\).
   - Mark `reversed = 1` to activate minor-loop shift.
6. Evaluate \(g(T)\), then \(R(T)\).

This implements major-loop + minor-loop memory from 2582_1.

---

## 8. End-to-end algorithms for each simulation workflow

## 8.1 Voltage-driven run (`simulate_yuanhang`)

For each time step:

1. Evaluate hysteretic resistance \(R_n\) and fraction \(g_n\).
2. Compute load and VO2 currents.
3. Compute \(dV/dt\) and \(dT/dt\).
4. Euler update \(V_{n+1},T_{n+1}\).
5. Store all traces (`V`, `I_load`, `I_vo2`, `T`, `R`, `g`).

Supports single device and Nx x Ny arrays via the same code path.

## 8.2 Current-step run (`simulate_current_step`)

For each time step:

1. Read waveform sample \(I_{in,n}\) (step or pulse).
2. Evaluate \(R_n\) from hysteresis at \(T_n\).
3. Compute \(P_n=V_n^2/R_n\).
4. Update voltage using current-drive KCL.
5. Update temperature using deterministic + stochastic term.
6. Update hysteresis reversal state using \((T_n,T_{n+1})\).

## 8.3 Current sweep + GIF (`run_sweep_make_gif`)

1. Loop current from 50 to 2000 microamp (default 50-step).
2. Run one time-domain simulation per current.
3. Save dual-axis frame: \(I_{in}(t)\), \(V_{vo2}(t)\).
4. Build GIF with chosen frame duration.

## 8.4 Oscillation-domain explorer (`app.py`)

1. Pick one scan parameter (`C`, `C_th`, `S_e`, `R_out`, `dt`, etc.).
2. For each scan value, run full current sweep.
3. Classify oscillatory runs by turning-point count and Vpp thresholds.
4. Output summary/detail CSV tables.

---

## 9. How the resistance fitting from the second paper is implemented

This corresponds to your request about "how we used the other paper to calculate hysteresis based on our document/data."

## 9.1 Input data

File:

- `data/experimental/100425_chip1_gap3.tsv` (264 usable samples after parsing)

Columns used:

- `Temperature`, `Resistance` (and `Time` for sorting when present)

## 9.2 Model form being fit

The fitted model is the same simulator model:

\[
R(T)=R_0\exp\left(\frac{E_a/k_B}{T}\right)g(T)+R_m
\]

with hysteresis parameters \((w,T_c,\beta,\gamma)\) and branch state.

## 9.3 Fitting strategy in `resistance_custom_analysis.py`

1. Parse and clean data, convert Celsius to Kelvin if needed.
2. Estimate initial \(R_0,E_a,R_m\) from low-temperature window via semilog regression and \(R_m\) scan.
3. Build seed \(g_{exp}(T)\) from measured \(R(T)\):
   \[
   g_{exp}=\frac{R-R_m}{R_0\exp(E_a/T)}
   \]
   then clip to \([0,1]\).
4. Estimate rough \(w,T_c\) from \(g\approx 0.5\) points on heating/cooling.
5. Optimize parameters using hybrid search:
   - random search (`random_iters`)
   - local coordinate descent (`local_passes`)
6. Objective:
   \[
   J = RMSE_{\log10}(R_{pred},R_{meas}) + \lambda\,RMSE(g_{pred},g_{exp}) + \text{optional penalty}
   \]
   with \(\lambda=\) `g_weight`.
7. Evaluate both initial branches (`insulator`, `metal`) and choose lower loss.
8. Save fitted params and metrics to JSON preset.

## 9.4 Result currently stored in preset

From `presets/resistance_100425_chip1_gap3.json`:

- `start_branch = metal`
- `rmse_log10 = 0.03756`
- Fitted key parameters:
  - `R0 = 0.5297 Ohm`
  - `Ea_over_k = 2658 K`
  - `Rm0 = 18.20 Ohm`
  - `w = 6.900 K`
  - `Tc_K = 333.57 K`
  - `beta = 0.297`
  - `gamma = 0.95627` (fixed default unless `--fit-gamma`)

---

## 10. Traceability matrix: equation -> implementation

| Topic | Equation / logic | Code location |
|---|---|---|
| Proximity function \(P(x)\) | \(0.5(1-\sin(\gamma x))(1+\tanh(\pi^2-2\pi x))\) | `model.py` lines 31-33 and 138-140 |
| Hysteresis \(T_{pr}\) | \(\delta w/2 + T_c - (1/\beta)\operatorname{arctanh}(2g_r-1)-T_r\) | `model.py` lines 175-179 |
| Resistance law | \(R=R_0 e^{E_a/T} g + R_m\) | `model.py` lines 197-206 |
| Voltage ODE | \(\dot V = (V_{in}-V)/(R_{series}C) - V/(RC)\) | `model.py` line 338 |
| Thermal ODE + coupling | \(C_{th}\dot T = V^2/R - S_e(T-T_0)+S_c\nabla^2T + noise\) | `model.py` lines 345-348 |
| Current-drive KCL | \(C\dot V = I_{in} - V(1/R + 1/R_{out})\) | `current_drive_sim.py` lines 192-193 |
| Ideal current source convention | `R_out <= 0` means \(1/R_{out}=0\) | `current_drive_sim.py` lines 123-127; `app.py` lines 1096-1098 |
| Current-drive thermal noise | \(+(\sigma/C_{th})\sqrt{dt}\,\xi\) | `current_drive_sim.py` lines 195-197 |
| Numerics diagnostics | \(dt/\tau\), thermal jump vs reversal threshold | `app.py` lines 1223-1272 |
| Specimen fit objective | RMSE in log-resistance + weighted g-consistency | `resistance_custom_analysis.py` lines 478-490 |

---

## 11. Assumptions made in this project

1. Lumped-element node model for electrical dynamics.
2. Lumped thermal dynamics per device, optional nearest-neighbor diffusive coupling.
3. Hysteresis memory is represented by reversal-state variables (\(T_r,g_r,T_{pr},\delta\)).
4. Temperature clamping is used inside hysteresis evaluation for numerical stability.
5. Current-drive model is obtained by source transformation (Thevenin/Norton-equivalent interpretation), not by changing VO2 physics.
6. Ideal current source is represented as \(R_{out}\to\infty\), encoded as `R_out <= 0` in UI/code.

---

## 12. Reproducibility commands

Run resistance fitting:

```bash
python resistance_custom_analysis.py \
  --data data/experimental/100425_chip1_gap3.tsv \
  --save-json presets/resistance_100425_chip1_gap3.json \
  --save-plot outputs/resistance_fit_100425_chip1_gap3.png
```

Run a current sweep and GIF:

```bash
python current_drive_sim.py
```

Run Streamlit UI:

```bash
streamlit run app.py
```

---

## 13. Practical interpretation for thesis/report writing

If you want to present this in your project report, the clean narrative is:

1. Start from Zhang et al. Eq. (1)-(2) for voltage + thermal dynamics.
2. Explain that \(R(T)\) uses the Almeida 2582 hysteresis formalism.
3. Show how hysteresis state variables generate major/minor loops.
4. Derive current-drive KCL by replacing \(V_{in}/R_{load}\) with \(I_{in}(t)\) and introducing \(R_{out}\).
5. Take \(R_{out}\to\infty\) to define ideal-current-source limit.
6. Present Euler (voltage module) and Euler/Euler-Maruyama (current module) discretizations.
7. Close with your specimen-fit methodology and fitted parameters.

That sequence is fully consistent with both papers and this codebase.
