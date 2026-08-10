# One-Year Project Presentation Outline

Working title:
**From the Yuanhang VO2 Neuristor Model to Current-Source Oscillations**

Audience:
Professor, Amir, and lab members who know the device physics but may not know the implementation history.

Recommended length:
20-30 minutes, about 20-24 slides. If the talk is shorter, keep slides 1-16 and move the AI/process slides to backup.

> **Archive status (August 10, 2026):** This outline records the July presentation
> narrative. The current quantitative authority is
> `docs/CURRENT_DRIVE_CALIBRATION.md`. The later audit established that lowering
> electrical capacitance speeds the drop but cannot raise the ideal-current `I*Rm`
> voltage floor, corrected the hysteresis endpoint handling, and found no sustained
> `C=0` thermal-only cycle for the Yuanhang parameter study.

## Core Story

The presentation should not be "I wrote code." The story should be:

1. We started from the published Yuanhang voltage-source neuristor model.
2. We rebuilt it locally in a transparent, inspectable simulator.
3. We validated the R(T) hysteresis implementation and fitted our measured specimen data.
4. We extended the framework to an ideal current-source version.
5. Because the current-source domain was not already known, we used automated/AI-assisted domain search to find physically interpretable oscillatory regimes.
6. The current-source model reproduces the key qualitative behavior: low-current no-switching, intermediate oscillations, high-current lock/clamping.
7. The remaining gap is absolute calibration: matching current, voltage, frequency, and energy scales simultaneously.

## Slide Deck Structure

### Slide 1: Title

Title:
**Single VO2 Neuristor Simulations: From Yuanhang Voltage Source to Current-Source Oscillations**

Content:
- Your name
- Lab name
- Advisor/professor and Amir
- One-sentence subtitle: "A year-long modeling, implementation, validation, and AI-assisted domain-search project."

Speaker move:
Open by saying this is both a physics/modeling project and a computational infrastructure project.

### Slide 2: Motivation

Question:
**Can a compact VO2 electrothermal model reproduce neuristor-like oscillations and help interpret current-driven experiments?**

Content:
- VO2 has a hysteretic insulator-metal transition.
- Electrical and thermal dynamics are coupled.
- Oscillations arise from charging, Joule heating, resistance collapse, cooling, and recovery.

Visual:
- Simple device/circuit cartoon.
- Optional: R(T) hysteresis loop thumbnail.

### Slide 3: Starting Point: Yuanhang Voltage-Source Model

Content:
- The initial source was the Yuanhang Zhang / Qiu / Di Ventra collective-dynamics neuristor model.
- Original model: voltage input, series/load resistance, VO2 node voltage, thermal state, hysteretic R(T).
- Our first goal: reproduce the single-device dynamics locally.

Use this file as a source:
- `references/yuanhangzhang98-collective_dynamics_neuristor-217d4f0/model.py`

Speaker move:
Emphasize that the project began by understanding and porting an existing model, not inventing a new one.

### Slide 4: Voltage-Source Electrical Equation

Show:
\[
C\frac{dV}{dt} =
\frac{V_{in}}{R_{series}}
- V\left(\frac{1}{R_{vo2}}+\frac{1}{R_{series}}\right)
\]

Equivalent:
\[
\frac{dV}{dt}
=
\frac{V_{in}-V}{R_{series}C}
- \frac{V}{R_{vo2}C}
\]

Explain terms:
- \(C dV/dt\): node capacitance.
- \((V_{in}-V)/R_{series}\): current supplied by voltage source through the load.
- \(V/R_{vo2}\): current through VO2.
- \(R_{vo2}=R(T,\mathcal H)\): nonlinear hysteretic resistance.

Suggested visual:
- Circuit diagram from `docs/manuscript/theory_behind_simulations.tex`.

### Slide 5: Thermal Equation

Show:
\[
C_{th}\frac{dT}{dt}
=
\frac{V^2}{R_{vo2}}
-S_e(T-T_0)
+\sigma\eta(t)
\]

Explain:
- \(V^2/R\): Joule heating.
- \(S_e(T-T_0)\): cooling to environment/substrate.
- \(C_{th}\): thermal inertia.
- \(\sigma\eta(t)\): optional thermal noise.

Speaker move:
Make clear this is where oscillations become possible: the resistance affects heating, and heating affects resistance.

### Slide 6: Resistance Model: Arrhenius + Hysteresis

Show:
\[
R_{vo2}(T,\mathcal H)
=
R_0\exp\left(\frac{E_a/k_B}{T}\right)g(T,\mathcal H)+R_m
\]

Show major-loop fraction:
\[
g_{\mathrm{major}}(T,\delta)
=
\frac{1}{2}
+\frac{1}{2}\tanh\left[
\beta\left(\delta\frac{w}{2}+T_c-T\right)
\right]
\]

Explain:
- \(g\): semiconducting fraction.
- \(R_m\): metallic floor.
- \(T_c,w,\beta\): transition center, hysteresis width, steepness.
- \(\mathcal H\): memory state, including branch and reversal point.

Visual:
- `docs/manuscript/figures/hysteresis.png`

### Slide 7: Fitting Measured R(T) Data

Content:
- We calibrated the resistance model to the measured specimen.
- Fitting target: heating/cooling branches of measured R(T).
- Stored specimen preset:
  `presets/resistance_100425_chip1_gap3.json`

Useful numbers:
- Fit RMSE log10: about `0.0373`
- R2 log10: about `0.9987`
- Fitted transition center: about `333.47 K`
- Fitted metallic floor: about `18.3 Ohm`

Visuals:
- `docs/manuscript/figures/fit_resistance_rt.png`
- App screenshot of measured R-T data and fitted model, if you want a more polished visual.

Speaker move:
This is where the model becomes connected to your lab sample, not only the original paper.

### Slide 8: Discrete Algorithm / Time-Step Order

Title:
**What Happens at Each Time Step?**

Show this order:

1. Start with current state \(V_k,T_k,\mathcal H_k\).
2. Evaluate hysteretic resistance:
   \[
   R_k,g_k = R(T_k,\mathcal H_k)
   \]
3. Compute electrical update:
   \[
   V_{k+1}=V_k+\Delta t \cdot f_V(V_k,T_k,R_k)
   \]
4. Compute Joule power:
   \[
   P_k=\frac{V_k^2}{R_k}
   \]
5. Compute thermal update:
   \[
   T_{k+1}=T_k+\frac{\Delta t}{C_{th}}
   \left(P_k-S_e(T_k-T_0)\right)
   \]
6. Update hysteresis branch/reversal memory based on temperature movement.
7. Save traces: \(V,T,R,g,P,I\).

Speaker move:
Say this slide matters because many apparent oscillations can be numerical artifacts if the order/timestep is wrong.

### Slide 9: Local Simulator and Streamlit App

Content:
- Core voltage model: `src/neuristor/model.py`
- Current-source model: `src/neuristor/current_drive_sim.py`
- Search/analysis backend: `src/neuristor/current_domain_search.py`
- GUI and history system: `app.py`
- Public shareable runs: `public_jobs/`

Visual:
- Screenshot of app interface or History tab.

Speaker move:
This was not just a script. It became a reusable tool where simulations, presets, and history can be inspected.

### Slide 10: Voltage-Source Validation

Content:
- Reproduced voltage-driven oscillatory behavior.
- Tested frequency convergence with timestep refinement.
- Preserved upstream-faithful hysteresis behavior.

Mention tests:
- `test_voltage_oscillator_frequency_converges`
- `test_saved_sample_fits_replay_to_stored_metrics`
- `test_matches_sample_calibrated_reference_trace`

Visual:
- Voltage-source trace showing spiking.
- Optional frequency vs parameter sweep from app/manual outputs.

### Slide 11: Why Move to Current Source?

Content:
- Amir/professor's paper and experiments include current-driven behavior.
- The physics changes because input current is imposed directly.
- Under ideal current drive, voltage is:
  \[
  V \approx I R(T,\mathcal H)
  \]
  after the electrical transient.
- The current-source question becomes: can electrothermal feedback alone produce relaxation oscillations?

Speaker move:
This is the pivot from reproducing known voltage-source behavior to exploring a less constrained domain.

### Slide 12: Current-Source Derivation

Start from node current balance:
\[
I_{in}=C\frac{dV}{dt}+\frac{V}{R_{vo2}}
\]

Then:
\[
C\frac{dV}{dt}=I_{in}-\frac{V}{R_{vo2}}
\]

Thermal equation:
\[
C_{th}\frac{dT}{dt}
=
\frac{V^2}{R_{vo2}}
-S_e(T-T_0)
+\sigma\eta(t)
\]

Assumptions:
- One VO2 domain.
- One thermal node.
- Ideal imposed current.
- Same upstream-faithful Yuanhang hysteresis.
- No dynamic/quasistatic/multidomain current-source modes in final model.

### Slide 13: New Challenge: Unknown Oscillatory Domain

Content:
- In the voltage-source model, the original paper gave strong parameter guidance.
- In the current-source version, the right oscillatory domain was not known.
- We needed to search over physically meaningful parameters:
  - current amplitude
  - capacitance \(C\)
  - thermal capacitance \(C_{th}\)
  - thermal conductance \(S_e\)
  - base temperature \(T_0\)
  - fitted R(T) parameters

Speaker move:
This is where computational search became scientifically useful.

### Slide 14: What Counts as a Real Oscillation?

Content:
Acceptance criteria:
- Not startup overshoot only.
- Sustained late-window cycles.
- Finite voltage/temperature/resistance traces.
- Not timestep zigzag/Nyquist artifact.
- Repeatable under smaller timestep.
- Physically interpretable R(T) path.

Mention:
- Old apparent oscillations were rejected when they depended on outdated reversal modes or timestep artifacts.
- Final model uses the hardcoded Yuanhang current-source standard.

Source:
- `docs/HYSTERESIS_IMPLEMENTATION_AUDIT.md`

### Slide 15: AI-Assisted Domain Search

Title:
**How I Used AI as a Research/Engineering Assistant**

Content:
- Used Codex in the local repo, with access to code, tests, and generated outputs.
- Prompted it to act like a research scientist:
  - inspect the existing simulator
  - write scripts using the existing model APIs
  - sweep parameter domains
  - classify traces as oscillatory/non-oscillatory
  - generate figures/GIFs
  - save successful runs into app History
  - test numerical stability

Important framing:
AI did not replace validation. It accelerated iteration and helped connect code, plots, and physical interpretation.

### Slide 16: Example AI Prompt

Show a shortened version:

> "Act as a research scientist. Use the existing current-source simulator without changing the model. Sweep physical parameters such as current, capacitance, thermal capacitance, and thermal conductance. Find domains where the current-source system shows sustained oscillatory behavior. Reject startup overshoot and numerical artifacts. Save successful simulations into app History and report the parameter domains."

Then explain:
- The prompt gave Codex a scientific role, constraints, and success criteria.
- The most important instruction was: **do not change the model while searching.**

### Slide 17: Domain Search Workflow

Show pipeline:

1. Load fitted R(T) specimen parameters.
2. Choose physical parameter ranges.
3. Run current-source simulations.
4. Extract metrics:
   - \(V_{pp}\)
   - turn/cycle count
   - dominant frequency
   - energy per cycle
   - average voltage/current
   - R(T) trajectory
5. Classify regimes:
   - insulating/no switching
   - transient
   - sustained oscillation
   - metallic lock
6. Save selected runs to `public_jobs/`.

Visual:
- Flowchart or screenshots from History tab.

### Slide 18: Main Current-Source Result

Content:
Paper-frequency analog:
- `C = 4 pF`
- `C_th = 1 pJ/K`
- `S_e = 0.10 mW/K`
- 300 ns pulse from 150 ns to 450 ns
- Oscillatory window: `1400-2100 uA`
- Frequency: `35.7-60.7 MHz`

Visuals:
- `public_jobs/20260707_145140_paper_frequency_f881d2/paper_frequency_operating_window.png`
- `public_jobs/20260707_145140_paper_frequency_f881d2/paper_frequency_three_regimes.png`

Speaker move:
Say this matches the paper's frequency scale and qualitative regime structure, not the absolute energy/current calibration.

### Slide 19: Three Regimes

Content:
Show:
- low current: no IMT / insulating response
- intermediate current: relaxation oscillations
- high current: metallic lock/clamping

Visual:
- `paper_frequency_three_regimes.png`

Speaker move:
This is probably the most important "physics result" slide. Spend time explaining the R(T) dot/trajectory if shown.

### Slide 20: Hidden State Visualization: R(T) Trajectory

Content:
- The experiment measures voltage/current, but the simulation reveals the hidden motion on the R(T) hysteresis curve.
- During oscillation, the state repeatedly traverses the transition region.
- This helps explain why oscillation happens.

Visual:
- `public_jobs/20260707_145140_paper_frequency_f881d2/current_sweep.gif`
- `public_jobs/20260707_145140_paper_frequency_f881d2/paper_frequency_1800uA_time_evolution_10s.gif`

Speaker move:
This is a "value of simulation" slide: seeing internal state not directly observable in real time.

### Slide 21: Sample-Scale Control

Content:
Validated sample-scale run:
- `C = 25.930953 pF`
- `C_th = 5 pJ/K`
- `S_e = 0.10 mW/K`
- Oscillatory window: `1600-2800 uA`
- Frequency: `7.1-14.3 MHz`

Visual:
- `public_jobs/20260707_145140_sample_scale_89ffb4/validated_sample_operating_window.png`

Speaker move:
Use this as an honesty/control slide: same mechanism, slower time scale.

### Slide 22: What Did Not Match Yet?

Content:
- The paper-frequency analog reaches `35.7-60.7 MHz`, but full-cycle energy is about `36.3-53.9 pJ`, larger than paper's reported `1-2 pJ`.
- The absolute current window is higher than the paper's `200-600 uA`.
- A 390 uA nonzero-voltage-floor trial can reach `156-210 mV`, but it damps to a fixed point instead of sustained oscillation.

Interpretation:
- The minimal one-node ideal-current model captures the oscillatory mechanism.
- Absolute calibration likely needs better device geometry, circuit/readout model, or additional experimental constraints.

Visual:
- Optional: `public_jobs/20260707_154737_390uA_valley_ff1d22/390uA_nonzero_valley_trial.png`

### Slide 23: Engineering Contributions

Content:
- Local model implementation.
- Streamlit app.
- Sample R(T) fitting workflow.
- Public/private job history.
- Reproducible simulation packages.
- Tests for convergence and legacy compatibility.
- Documentation:
  - `docs/manuscript/theory_behind_simulations.tex`
  - `docs/HYSTERESIS_IMPLEMENTATION_AUDIT.md`
  - `README.md`

Speaker move:
This is where you make the year of work visible beyond the final plot.

### Slide 24: Next Steps

Scientific next steps:
- Calibrate geometry/device area to shift absolute current and energy scale.
- Include measured circuit/output stage if the plotted voltage is not pure VO2 voltage.
- Fit current-pulse data directly, not only R(T).
- Compare pump-probe/recovery behavior.
- Explore whether a small extension beyond one thermal node is required, but only after the one-node model's limits are documented.

Engineering next steps:
- Build a "paper comparison" page in the app.
- Add exportable slide-ready figures.
- Keep public jobs curated for professor/Amir review.

### Slide 25: Conclusion

Suggested final message:

> We successfully rebuilt and validated the Yuanhang VO2 neuristor model, calibrated it to measured R(T) data, extended it to an ideal current-source formulation, and used AI-assisted parameter-domain search to find physically interpretable current-source oscillations. The final model reproduces the qualitative low/intermediate/high-current regimes and reaches the paper's frequency scale, while absolute current and energy calibration remain the key next step.

## Backup Slides

Use these only if asked.

1. Full resistance/hysteresis equations.
2. Hysteresis implementation audit.
3. Tests and numerical convergence.
4. Streamlit job/history architecture.
5. Public job artifacts and reproducibility.
6. Why old dynamic/quasistatic current-source variants were removed.
7. 390 uA voltage-floor exploration and why it damped.

## Figures to Use

High-priority:

- `docs/manuscript/figures/hysteresis.png`
- `docs/manuscript/figures/fit_resistance_rt.png`
- `public_jobs/20260707_145140_paper_frequency_f881d2/paper_frequency_three_regimes.png`
- `public_jobs/20260707_145140_paper_frequency_f881d2/paper_frequency_operating_window.png`
- `public_jobs/20260707_145140_paper_frequency_f881d2/paper_frequency_energy_cycle.png`
- `public_jobs/20260707_145140_paper_frequency_f881d2/current_sweep.gif`
- `public_jobs/20260707_145140_paper_frequency_f881d2/paper_frequency_1800uA_time_evolution_10s.gif`
- `public_jobs/20260707_145140_sample_scale_89ffb4/validated_sample_operating_window.png`

Optional:

- `public_jobs/20260707_154737_390uA_valley_ff1d22/390uA_nonzero_valley_trial.png`
- Streamlit screenshots from Samples, Experiment, and History.

## Prompt Pack for ChatGPT

Use these prompts to generate speaker notes, slide text, or diagrams.

### Prompt 1: Generate Speaker Notes from the Outline

```text
I am preparing a 20-30 minute presentation for my professor and a PhD student named Amir.
The project is a one-year VO2 neuristor simulation project.

My story:
1. I started from the Yuanhang voltage-source VO2 neuristor model.
2. I implemented it locally in Python.
3. I validated the voltage-source equations and R(T) hysteresis model.
4. I fitted measured specimen R(T) data.
5. I derived an ideal current-source version:
   C dV/dt = I_in - V/R_vo2
   C_th dT/dt = V^2/R_vo2 - S_e(T-T0)
6. I used AI/Codex to search parameter domains for current-source oscillations without changing the model.
7. I found a paper-frequency analog with C=4 pF, C_th=1 pJ/K, S_e=0.10 mW/K, oscillatory window 1400-2100 uA, frequency 35.7-60.7 MHz.
8. I also found a validated sample-scale control with C=25.930953 pF, C_th=5 pJ/K, frequency 7.1-14.3 MHz.
9. The main limitation is that the model matches qualitative regimes and frequency scale, but not absolute current/energy calibration.

Please write concise speaker notes for each slide.
Tone: technically serious, but clear and student-presentable.
Avoid overclaiming. Emphasize what was validated and what remains open.
```

### Prompt 2: Make Slide Text Concise

```text
Here is a draft slide outline for a VO2 neuristor simulation project.
Turn each slide into 3-5 concise bullet points suitable for PowerPoint.
Keep equations where necessary, but avoid long paragraphs.
Make sure each slide has one main message.
```

### Prompt 3: Explain the Equations Simply

```text
Explain these equations for a lab presentation:

Voltage source:
C dV/dt = V_in/R_series - V(1/R_vo2 + 1/R_series)

Current source:
C dV/dt = I_in - V/R_vo2

Thermal:
C_th dT/dt = V^2/R_vo2 - S_e(T-T0)

Resistance:
R_vo2(T,H) = R0 exp((Ea/kB)/T) g(T,H) + Rm

Explain what each term means physically and why these equations can produce relaxation oscillations.
Write it as speaker notes, not as a textbook.
```

### Prompt 4: Explain AI Use Without Sounding Like Hype

```text
I need one slide explaining how I used AI/Codex in a physics simulation project.

Facts:
- I used Codex inside the local codebase.
- It inspected code, wrote search scripts, generated plots/GIFs, and saved selected runs to Streamlit history.
- It helped sweep current-source parameter domains.
- I constrained it not to change the physical model while searching.
- I validated results with tests, timestep convergence, and physical interpretation.

Write a professional slide plus speaker notes.
Avoid sounding like AI magically solved the science.
Frame it as accelerated computational exploration and reproducible engineering.
```

### Prompt 5: Prepare for Professor Questions

```text
Based on this VO2 current-source simulation project, list likely questions my professor might ask.
Include questions about:
- model assumptions
- why current source differs from voltage source
- numerical timestep convergence
- R(T) fitting
- why absolute current/energy do not match yet
- how AI was used responsibly
- next experiments

For each question, give a concise answer I can say out loud.
```

### Prompt 6: Turn the Deck Into a Narrative

```text
Given this slide sequence, write a 2-minute opening narrative and a 1-minute closing narrative.
The opening should explain the project arc from reproducing an existing voltage model to discovering current-source oscillatory domains.
The closing should emphasize achievements, limitations, and next steps.
```

## Recommended Spoken Framing for AI/Codex

Say:

"I used Codex as a computational research assistant inside the repository. The important part was that I constrained it to use the existing model and existing APIs. It helped generate search scripts, run sweeps, classify traces, create plots, and save results into the app history. But the acceptance criteria were physical and numerical: late-window oscillations, timestep robustness, finite traces, and interpretable R(T) trajectories."

Do not say:

"AI discovered the physics."

Better:

"AI accelerated the domain search and helped make the workflow reproducible."

## Recommended Emphasis

Most important:
- The equations and their physical meaning.
- The R(T) fit and hysteresis implementation.
- The current-source derivation.
- The domain-search problem.
- The successful oscillatory regimes.
- Honest limitations.

Less important:
- Every code detail.
- Every old dead-end model.
- Every deleted preset/script.

## One-Slide Summary Version

If you need a single summary slide:

**What I built**
- Local VO2 neuristor simulator based on Yuanhang model.
- Streamlit interface with sample fitting, sweeps, and saved history.
- Ideal current-source extension.

**What I validated**
- R(T) hysteresis fit to measured specimen data.
- Voltage-source oscillator convergence.
- Current-source oscillator convergence.

**What I found**
- Paper-frequency current-source analog: `35.7-60.7 MHz`.
- Three regimes: insulating, oscillatory, metallic lock.

**What remains**
- Absolute current/energy calibration.
- Better mapping to measured current-source experiment.
