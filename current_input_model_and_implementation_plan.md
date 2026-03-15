# Current-Input VO2 Neuristor Simulation (di Ventra dynamics + 2582_1 hysteretic resistance)

Historical note:
- An earlier draft of this plan described a Norton-style shunt resistance `R_out`.
- That is not the model used in the current codebase or manuscript.
- The implemented model is a separate direct current-source experiment with an ideal imposed current waveform at the VO2 node.

This document specifies a separate **current-driven** simulation entrypoint that uses the same VO₂ physics as the existing voltage-driven model:

- Electrical and thermal dynamics from the **di Ventra VO₂ oscillator** model.
- Hysteretic resistance model from **2582_1** (major loop + minor loops via reversal memory).

It also specifies the deliverables needed to recreate the reference-style plots you provided and to generate a GIF across a current sweep.

This is intended to be handed to Codex and implemented as a **separate simulation module**, without changing your existing voltage-driven workflow.

---

## 1) What we want to reproduce (deliverables)

You provided reference images showing:

1) Time-domain traces with:
- **Input current** (green), step input.
- **Output voltage** across VO₂ (purple), denoted here as `V_vo2(t)`.

2) A sweep behavior where the response changes with current.

### Required output for this task

A current sweep from **50 µA to 2000 µA** in **50 µA** increments:

\[
I_{\mathrm{in}} \in \{50,100,150,\dots,2000\}\ \mu\text{A}
\]

For each current value:
- Run a time-domain simulation with a step input.
- Save a figure similar in layout to the reference: `I_in(t)` and `V_vo2(t)` on a shared time axis.
- Save frames as PNGs.
- Create a GIF where each frame is displayed for **0.5 seconds**.

---

## 2) Model definitions

### State variables
Single device:

- \(V(t)\) [V], voltage across the VO₂ device and capacitor (this is `V_vo2`)
- \(T(t)\) [K], device temperature
- \(\mathcal H(t)\), hysteresis internal state (reversal memory, branch, etc.)

### 2582_1 resistance model (unchanged)
Use the existing hysteresis code you already have.

The resistance is:

\[
R(T,\mathcal H)=R_m+R_0\exp\!\left(\frac{E_a}{k_B T}\right)\,F(T,\mathcal H)
\]

where \(F(T,\mathcal H)\in[0,1]\) is the 2582_1 hysteresis operator.

---

## 3) Electrical dynamics with current input

This current-driven mode is not obtained by a Thevenin/Norton conversion of the voltage-driven circuit.
It represents a separate experiment in which the voltage source is replaced by an ideal programmed current source.
Any external source-side series resistor is omitted from the reduced equations because the imposed current waveform is treated as ideal.

### Node KCL

Currents leaving the node:
- through VO₂: \(V/R(T,\mathcal H)\)
- through capacitor: \(C\,dV/dt\)

Current entering the node:
- injected source current: \(I_{\mathrm{in}}(t)\)

So:

\[
I_{\mathrm{in}}(t)=\frac{V}{R(T,\mathcal H)}+C\frac{dV}{dt}
\]

Rearrange:

\[
\boxed{
 C\frac{dV}{dt}=I_{\mathrm{in}}(t)-\frac{V}{R(T,\mathcal H)}
}
\]

Notes:
- The reduced model assumes the imposed current waveform is ideal.

---

## 4) Thermal dynamics (di Ventra form)

Single device (no spatial coupling):

\[
\boxed{
C_{\mathrm{th}}\frac{dT}{dt}=\frac{V^2}{R(T,\mathcal H)}-S_e(T-T_0)+\sigma\,\eta(t)
}
\]

- Joule heating is \(P(t)=V^2/R(T,\mathcal H)\).
- Cooling is linear with coefficient \(S_e\).
- Noise is additive Gaussian white noise.

Array / PDE-style coupling (optional later):
\[
C_{\mathrm{th}}\frac{dT_i}{dt}=\frac{V_i^2}{R(T_i,\mathcal H_i)}-S_e(T_i-T_0)+S_c(\nabla^2T)_i+\sigma\,\eta_i(t)
\]

This implementation request is for a single device, but the code should be structured so the array version can be added later.

---

## 5) Euler–Maruyama discretization (what to implement)

Let \(t_n=n\Delta t\), and \(\xi_n\sim\mathcal N(0,1)\).

Electrical update (Euler):
\[
V_{n+1}=V_n+\frac{\Delta t}{C}\left[I_{\mathrm{in}}(t_n)-\frac{V_n}{R_n}\right]
\]
where \(R_n=R(T_n,\mathcal H_n)\).

Thermal update (Euler–Maruyama):
\[
T_{n+1}=T_n+\frac{\Delta t}{C_{\mathrm{th}}}\left[\frac{V_n^2}{R_n}-S_e(T_n-T_0)\right]+\frac{\sigma}{C_{\mathrm{th}}}\sqrt{\Delta t}\,\xi_n
\]

Then update the hysteresis state \(\mathcal H\) using your existing 2582_1 reversal and minor-loop logic.

---

## 6) Simulation protocol for the sweep

For each current \(I_{\mathrm{in}}\):

1) Define a step current waveform:
- \(I_{\mathrm{in}}(t)=0\) for \(t<0\) (optional pre-time `t_pre`)
- \(I_{\mathrm{in}}(t)=I_{\mathrm{target}}\) for \(t\ge 0\)

2) Run the simulation for a fixed window:
- Use a time step `dt` consistent with your existing voltage-driven runs.
- Use `t_end` long enough to see the behavior (for example 600 ns like the reference plots).

3) Save a figure with:
- Left axis: `I_in (µA)` vs time (ns), in green
- Right axis: `V_vo2 (mV)` vs time (ns), in purple
- Title: `I_in = XXXX µA`

4) Save as PNG:
- `outputs/current_sweep_frames/frame_000_I0050uA.png`, etc.

5) Make a GIF:
- Duration per frame: 0.5 seconds
- Save to `outputs/current_sweep.gif`

---

## 7) Repository integration plan

Implement this as a separate module so it does not affect existing voltage-driven simulations.

### New files
- `current_drive_sim.py`
  - simulation and sweep driver
- `plots_current_drive.py` (optional)
  - plotting helpers if you want to keep plotting code clean

### Output folders
- `outputs/current_sweep_frames/`
- `outputs/current_sweep.gif`

### Return structure for a single run
A dict with numpy arrays:
- `t` [s]
- `I_in` [A]
- `V_vo2` [V]
- `T` [K]
- `R` [ohm]
- `P` [W]

---

## 8) Reference implementation outline (Codex should implement this)

Codex should implement something equivalent to the following structure.

Key requirements:
- Use the existing hysteresis object for `evaluate(T)` and `update(T_prev, T_new)`.
- Use Euler for the electrical update.
- Use Euler–Maruyama noise scaling `sqrt(dt)` in the temperature update if `sigma > 0`.
- Use `imageio` to assemble the GIF.

Pseudo-code outline:

```python
simulate_current_step(I_uA, params) -> out_dict:
    build time grid
    build I_in(t) array: 0 during pre-time, I_target after
    init V=0, T=T0, hysteresis.reset(T0)

    for each timestep:
        R = hysteresis.evaluate(T)
        dV_dt = (I_in - V/R)/C
        V_next = V + dV_dt*dt

        P = V^2 / R
        dT_dt = (P - S_e*(T - T0))/C_th
        T_next = T + dT_dt*dt + (sigma/C_th)*sqrt(dt)*N(0,1)

        hysteresis.update(T_prev=T, T_new=T_next)
        commit V, T

    return dict with arrays
```

Then:

```python
for I_uA in range(50, 2000+1, 50):
    out = simulate_current_step(I_uA)
    plot I_in(t) and V_vo2(t)
    save frame PNG

make GIF with duration 0.5 seconds per frame
```

---

## 9) Practical notes (to avoid confusion)

1) Current magnitude matters:
- Start near hundreds of microamps.
- If current is too large, temperature can blow up to a hot steady state.

2) Current-drive interpretation:
- This mode is a separate direct-current experiment, not a Norton-equivalent rewrite of the voltage-driven circuit.

3) Voltage output:
- `V_vo2` is simply the node voltage `V(t)` across VO₂ and the capacitor.

---

## 10) Inputs you must specify in code

Simulation configuration:
- `dt`, `t_end`, `t_pre`
- `seed` for noise reproducibility

Circuit parameters:
- `C`

Thermal parameters:
- `C_th`
- `S_e`
- `T0`
- `sigma` (optional noise)

VO₂ resistance model parameters:
- should come from your existing 2582_1 hysteresis implementation and configuration

---

## 11) Files included with this task

- `reference_graphs.zip` contains the provided reference plots.
- This markdown file documents the model and implementation plan for Codex.
