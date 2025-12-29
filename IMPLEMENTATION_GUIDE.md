VO₂ Neuristor Simulator – Implementation & Analysis Specification

(for Codex / automated refactoring & feature implementation)

0. Purpose of this document

This document specifies what to implement, what not to change, and how the codebase should be structured, so that an automated coding agent (Codex) can extend and refactor the VO₂ neuristor simulator without access to the original scientific papers or presentations.

The simulator is already physics-faithful. The goal is organization, reuse, extensibility, and analysis, not inventing new physics.

⸻

1. High-level goals
	1.	Provide a single source of truth for the neuristor model (physics + numerics).
	2.	Support systematic analysis and plotting requested by the PI.
	3.	Enable generic parameter sweeps (any parameter can be the free variable).
	4.	Prepare the codebase for a GUI / UI that lets users run simulations without touching code.
	5.	Add an optional heater model via a second thermal equation.

⸻

2. Non-negotiable constraints (DO NOT BREAK)

2.1 Physics invariants
	•	Do not change the governing equations already implemented.
	•	Do not change hysteresis logic, resistance law, or numerical integration scheme unless explicitly requested.
	•	All temperatures are in Kelvin.
	•	The electrical and thermal ODEs must remain consistent with the current implementation.

2.2 Units & conventions
	•	R_series_kohm → Ω via ×1e3
	•	C_par_pF → F via ×1e-12
	•	Cth_mW_ns_per_K → J/K via ×1e-12
	•	Sth_mW_per_K → W/K via ×1e-3
	•	Device voltage = V_node
	•	Device current = I_vo2
	•	Power = P = V_node * I_vo2

2.3 Output format

All simulations must return a dictionary (SimOut) with exactly these keys:

{
  "time_s",
  "V_node",
  "I_load",
  "I_vo2",
  "T_K",
  "R_vo2",
  "g",
  "grid_shape"
}

Single device → lists of floats
Multiple devices → list of lists (one per device)

⸻

3. Core architecture (required refactor)

3.1 model.py (mandatory)

model.py must be the single authoritative module containing:

Physics & numerics
	•	Parameter dataclasses:
	•	YuanhangResistParams
	•	YuanhangCircuitParams
	•	Hysteresis + resistance model R(T, g)
	•	Euler simulator for:
	•	single device
	•	Nx×Ny lattice
	•	Optional thermal noise

Public API (no leading underscores)

simulate_yuanhang(...)
series_first(...)
series_mean(...)
detect_spike_times(...)
is_oscillatory(...)
sweep_1d(...)
find_oscillatory_band_1d(...)

All analysis and GUI code must call only these APIs.

⸻

4. Analysis tasks requested by the PI

Each item below must be implemented as a reusable function, not hard-coded into a script.

⸻

4.1 Baselines of temperature and voltage vs time

Task
Plot the minimum points per oscillation cycle (baselines) of:
	•	device temperature T_K
	•	device voltage V_node

Definition
	•	Baseline = local minimum between successive oscillation peaks
	•	Use the same local-extrema logic consistently across all plots

Output
	•	Time-domain plot:
	•	raw signal
	•	overlaid baseline points
	•	Implement as a function:

plot_baselines_T_and_V_vs_time(simout, t_start_us, t_end_us)


⸻

4.2 Peak device voltage vs input voltage

Task
Plot:

V_max (peak V_node) vs V_in

Procedure
	•	For each V_in:
	•	run simulation
	•	restrict to steady-state window (e.g. 25–300 µs)
	•	extract maximum V_node

Output

plot_Vmax_vs_Vin(vin_list, sim_results)


⸻

4.3 Power vs time and power extrema vs input voltage

Definitions
	•	Power:

P(t) = V_node(t) * I_vo2(t)


	•	(Equivalently V_switch^2 / R_metal in the metallic phase.)

Tasks
	1.	Plot P(t) vs time for a representative V_in
	2.	Plot:

P_max(V_in) and P_min(V_in)



Physical intuition (must be preserved)

Lower temperature → higher switching voltage → higher power dissipation

Output

plot_power_time_and_peaks_vs_Vin(...)


⸻

4.4 Capacitance sweep: power dependence

Task
	•	Choose multiple values of electric capacitance C_par
	•	For each C:
	•	run simulation at fixed V_in
	•	compute and plot P(t)

Expected trend
	•	Larger capacitance → stronger power dissipation effect

Output

plot_capacitance_effect_on_power(vin, C_values, ...)


⸻

4.5 Multi-parameter sweep: frequency as a function of two parameters

Task
	•	Choose a single observable (default: oscillation frequency)
	•	Sweep two parameters, e.g.:
	•	C_par
	•	R_series

Procedure
	1.	For each (C, R) pair:
	•	run simulation
	•	detect oscillation frequency using spike detection
	2.	Plot a 3D scatter or surface:

frequency = f(C, R)



Output

plot_frequency_3d_vs_C_and_Rload(...)


⸻

4.6 Resistance in insulating state vs time

Task
Plot R_vo2(t) only when the device is in the insulating state.

Definition
	•	Use hysteresis variable g(t)
	•	Insulating state ≈ g ≥ threshold (e.g. 0.9)

Output

plot_R_insulating_vs_time(simout, g_threshold)


⸻

5. Generic sweep engine (critical for GUI)

5.1 Design requirement

Any single parameter must be sweepable without rewriting logic.

Examples:
	•	Sweep V_in
	•	Sweep C_par
	•	Sweep R_series
	•	Sweep thermal parameters
	•	Sweep noise strength

5.2 API design

sweep_1d(
  run_one: Callable[[float], SimOut],
  values: Sequence[float]
)

And a generic oscillation domain detector:

find_oscillatory_band_1d(
  run_sim,
  start,
  stop,
  step,
  osc_check=is_oscillatory
)

This is required so a future GUI can:
	•	lock all parameters
	•	choose one free variable
	•	automatically find oscillatory regimes

⸻

6. Heater model (new physics extension)

6.1 Goal

Add a heater contribution with a second thermal equation, similar to the device heat equation, but:
	•	Fixed resistance (no phase transition)
	•	No hysteresis
	•	Acts as an additional heat source coupled to the device

6.2 Conceptual model

Let:
	•	T_dev = device temperature
	•	T_heater = heater temperature

Add:

C_h dT_heater/dt = P_heater − G_h (T_heater − T_base) − G_c (T_heater − T_dev)

And modify device thermal equation to include:

+ G_c (T_heater − T_dev)

6.3 Implementation rules
	•	Heater must be optional
	•	Default behavior (no heater) must reproduce current results exactly
	•	Heater parameters must be cleanly encapsulated (e.g. HeaterParams)

⸻

7. File organization (target state)

Minimal but clean:

model.py        ← physics + numerics + sweeps (authoritative)
plots.py ← analysis & plots (imports only from model.py)
histogram.py    ← CLI / demo (may call model.py)
heater.py       ← optional heater extension (or merged later)

No duplicated simulators.

⸻

8. Future GUI considerations (do not implement yet)

Design choices should assume:
	•	sliders for parameters
	•	dropdown for sweep variable
	•	buttons:
	•	“simulate”
	•	“find oscillatory domain”
	•	“plot analysis X”

This is why:
	•	functions must be stateless
	•	inputs must be explicit
	•	outputs must be structured (SimOut)

⸻

9. Definition of “done”

Codex’s work is considered correct if:
	•	All requested plots are implemented as callable functions
	•	model.py is the single source of truth
	•	No physics behavior changes
	•	plots.py runs without importing from histogram.py
	•	Sweeps work for any scalar parameter
	•	Heater model can be enabled without breaking legacy simulations

⸻

10. GUI Specification (required)

10.1 Purpose

Build a GUI that allows any user (including non-programmers) to:
	1.	configure the VO₂ neuristor model parameters
	2.	run single simulations
	3.	run parameter sweeps
	4.	automatically detect oscillatory regimes
	5.	execute the PI-requested analyses/plots
	6.	export results (figures + CSV)

The GUI is a front-end; it must rely exclusively on public APIs in model.py (and analysis functions in plots.py). It must not duplicate simulation logic.

⸻

10.2 Recommended GUI framework (best choice)

Use Streamlit.

Why Streamlit
	•	fastest to build and iterate
	•	extremely clear UI for scientific tools
	•	built-in widgets (sliders, dropdowns, tabs)
	•	can export figures and files easily
	•	works locally and can be deployed later (e.g., Streamlit Community Cloud)

Implementation target file:

app.py

Run command:

streamlit run app.py


⸻

10.3 UI layout (high-level)

The GUI should have:

A) Left sidebar: “Simulation Setup”

Organize inputs into collapsible sections:

(1) Experiment mode selector
	•	Mode (radio):
	•	“Single run”
	•	“1D sweep”
	•	“2D sweep”
	•	“Oscillation domain finder”
	•	“Batch runner” (optional)

(2) Simulation time settings
	•	t_end_us (float input)
	•	dt_ns (float input)
	•	Steady-state window for analysis:
	•	t_start_us
	•	t_end_us

(3) Electrical parameters
	•	Vin (float input)
	•	R_series_kohm
	•	C_par_pF

(4) Thermal parameters
	•	T_base_K
	•	Cth_mW_ns_per_K
	•	Sth_mW_per_K
	•	Cth_factor
	•	noise_strength

(5) Lattice settings (advanced)
	•	nx, ny
	•	couple_factor
	•	dimension should be inferred if possible; otherwise hidden.

(6) Hysteresis / resistance parameters (advanced)
	•	R0
	•	Ea_over_k
	•	Rm0
	•	Rm_factor
	•	w
	•	Tc_K
	•	beta
	•	gamma
	•	T_min_K, T_max_K
	•	reversal_threshold_K
	•	start_branch (dropdown: “insulator”, “metal”)

(7) Presets
	•	Preset selector:
	•	“Default”
	•	“Paper preset”
	•	“Custom”
	•	“Load preset” button applies values to widgets
	•	“Reset all” returns to defaults

⸻

10.4 Main panel: Tabs (must-have)

Tab 1: “Run”

A single-click workflow:
	•	Button: Run simulation
	•	After run:
	•	show key plots
	•	show computed summary metrics
	•	allow export

Default plots on this tab (for single device)
	•	V_node(t) vs time
	•	I_vo2(t) vs time
	•	T_K(t) vs time
	•	R_vo2(t) vs time
	•	P(t)=V*I vs time

All plots must allow user to choose:
	•	full time window
	•	steady-state window

Summary metrics (computed & displayed)
	•	oscillatory? (True/False)
	•	spike count in window
	•	mean frequency (MHz) if oscillatory
	•	mean baseline temperature and voltage
	•	Vmax
	•	Pmax/Pmin

⸻

Tab 2: “PI Analyses”

This tab implements the professor’s required plots as checkboxes + run buttons.

Checkbox list:
	1.	Baselines of T and V vs time
	2.	Vmax vs Vin (requires Vin sweep)
	3.	Power vs time and power peaks vs Vin
	4.	Capacitance sweep power overlay
	5.	3D frequency vs (C, R_load)
	6.	Resistance in insulating state vs time

UI behavior:
	•	When user checks an analysis, show only the inputs needed for it.
	•	Provide a “Run selected analyses” button.
	•	Display plots inline with captions.

⸻

Tab 3: “Sweeps”

1D sweep
User selects:
	•	sweep parameter (dropdown)
	•	Vin
	•	C_par_pF
	•	R_series_kohm
	•	T_base_K
	•	Cth_factor
	•	noise_strength
	•	etc. (any scalar parameter)
	•	start, stop, step
	•	“Run sweep” button

Outputs:
	•	preview a sweep table (value → oscillatory?, freq, Vmax, Pmax)
	•	allow selecting a value from the table and plotting its full time traces

2D sweep
User selects:
	•	parameter A (dropdown)
	•	parameter B (dropdown)
	•	ranges for each
	•	“Run 2D sweep” button

Outputs:
	•	3D scatter OR heatmap (preferred: heatmap for frequency)
	•	allow exporting the sweep results as CSV

⸻

Tab 4: “Oscillation Domain Finder”

Goal: find region of parameter space where oscillations occur.

1D domain finder
	•	Choose sweep parameter (dropdown)
	•	start/stop/step
	•	spike threshold, min_spikes
	•	“Find oscillatory band” button

Outputs:
	•	show (min_osc, max_osc)
	•	plot oscillatory classification vs parameter value
	•	optionally automatically run a fine sweep within the band

⸻

Tab 5: “Export”

After any run/sweep:
	•	Download CSV for:
	•	last single simulation
	•	last sweep results
	•	Download plots (PNG)
	•	Save a “session config” JSON (all parameters)

⸻

10.5 Parameter binding (critical implementation detail)

The GUI must build parameter objects from widget values:
	•	YuanhangResistParams(...)
	•	YuanhangCircuitParams(...)

Then call:

simout = simulate_yuanhang(
    Vin=...,
    t_end=...,
    dt=...,
    resist_params=resist_params,
    circuit_params=circuit_params,
    lattice_shape=(nx, ny),
    start_branch=start_branch,
    noise_seed=seed,
)

For sweeps:
	•	update only the swept parameter
	•	keep everything else fixed
	•	implement with generic sweep utilities (no hardcoding Vin-only sweeps)

⸻

10.6 Robustness requirements
	•	Validate input ranges and show friendly error messages:
	•	dt > 0
	•	t_end > dt
	•	step > 0
	•	nx, ny ≥ 1
	•	Show progress bars for sweeps
	•	Cache results when possible (Streamlit caching) using a hashable config dictionary
	•	Never freeze UI during large sweeps: use Streamlit progress + incremental updates

⸻

10.7 Performance strategy (important)

Sweeps can be expensive. Implement:
	•	coarse/fine sweep options
	•	optional downsampling for plotting only (never for internal computations)
	•	a cap on max sweep points unless user explicitly overrides (“advanced”)

⸻

10.8 Heater model integration (GUI-ready design)

Once heater physics is added:
	•	add a toggle: “Enable heater”
	•	show heater parameters only when enabled
	•	add plot: T_heater(t) vs time
	•	add new metrics: heater power, coupling heat flow

⸻

10.9 Definition of GUI “done”

The GUI is successful if:
	•	a new user can reproduce all PI-requested plots with no code edits
	•	all simulation parameters are editable
	•	generic 1D sweeps work for any scalar parameter
	•	domain finder correctly identifies oscillatory bands
	•	exports work (CSV + PNG + config JSON)

Absolutely — add the following to the GUI spec. It tells Codex exactly how to keep the user informed during long runs (single runs, sweeps, domain finding), with progress bars, live logs, and intermediate results.

⸻

10.10 Long-run UX: Progress, status, and live results (mandatory)

10.10.1 Principle

Simulations and sweeps can take a while. The GUI must always show the user:
	•	what is currently running
	•	which parameter/value is being simulated
	•	how many runs are completed / remaining
	•	estimated completion in terms of completed fraction (avoid time estimates if unreliable)
	•	partial results as they arrive

This must be implemented for:
	•	single simulations
	•	1D sweeps
	•	2D sweeps
	•	oscillation-domain finding (coarse + fine stages)

⸻

10.10.2 Required UI elements (Streamlit)

A) Global status area (top of main panel)
A persistent “Run Status” box containing:
	•	State: Idle / Running / Completed / Error
	•	Mode: Single / Sweep / Domain finder
	•	Current step: e.g. “Simulating Vin = 14.35 V (17/121)”
	•	Last completed value and quick metrics (oscillatory? frequency)

Implementation guidance:
	•	Use st.status() (preferred) or a combination of st.info()/st.success()/st.error()
	•	Keep this visible even when switching tabs by storing run state in st.session_state

⸻

B) Progress bar
For any multi-run operation:
	•	show a progress bar with fraction completed

Implementation:
	•	progress = st.progress(0.0)
	•	update with progress.progress((i+1)/N)

For two-stage operations (coarse → fine):
	•	show two stacked progress bars:
	•	coarse progress
	•	fine progress (appears after coarse completes)

⸻

C) Live log / console output
Show a scrolling or updating log panel with:
	•	start time
	•	parameter values being simulated
	•	detection results (oscillatory vs not)
	•	any warnings (e.g., “no spikes detected in window”)

Implementation options (Streamlit):
	•	log_box = st.empty() and update it with a growing string
	•	or st.code(log_text) / st.text_area(log_text, height=...)
	•	store logs in st.session_state["logs"] to persist across reruns

Example log lines:
	•	[sweep] 17/121: Vin=14.35 V → oscillatory, f=2.18 MHz
	•	[domain] coarse: C=130 pF → non-oscillatory
	•	[domain] detected band: 145–205 pF (coarse)

⸻

D) Partial results table (updates live)
During sweeps/domain finding, show a table that fills as runs complete.

For 1D sweeps:
Columns:
	•	sweep_value
	•	oscillatory (bool)
	•	spike_count
	•	frequency_MHz (NaN if non-osc)
	•	Vmax
	•	Pmax
	•	Pmin

Implementation:
	•	Maintain a list of dict rows in session_state
	•	Render with st.dataframe(rows_df) and update each iteration
	•	Always show “completed so far” even before sweep finishes

⸻

E) Live preview plot (optional but strongly recommended)
Update a preview plot during sweeps:
	•	e.g., frequency vs sweep variable for completed points so far
	•	or oscillatory classification vs sweep variable

Implementation:
	•	Use st.pyplot(fig) with plot_container = st.empty()
	•	Update every k iterations (e.g., every 5–10 points) to reduce overhead

⸻

10.10.3 Cancellation / stop control (strongly recommended)

Provide a Stop button to cancel long sweeps.

Implementation guidance:
	•	Use st.button("Stop") to set st.session_state["cancel_requested"]=True
	•	In sweep loop: check that flag each iteration and break gracefully
	•	Mark status as “Cancelled by user” and keep partial results available for export

⸻

10.10.4 Caching + reuse (performance + UX)

To avoid repeating expensive computations when the user reruns with identical settings:
	•	Cache single simulations by a hashable config dict (Vin, dt, t_end, params)
	•	Cache sweep results by config + sweep definition

Streamlit guidance:
	•	Prefer st.cache_data for results that are pure (inputs → outputs)
	•	Use stable serialization for parameter structs (convert dataclasses to dict)

Important:
	•	Caching must not hide progress for new runs; it should show “Loaded from cache” clearly in the status area.

⸻

10.10.5 Error handling UX

If a run fails:
	•	show the failure in the status box
	•	include the last log lines
	•	do not lose partial sweep data already computed
	•	show the exact exception message in a collapsible section (“Details”)

⸻

10.10.6 Definition of “done” for progress UX

The GUI meets the requirement if:
	•	every operation shows progress and current parameter value
	•	logs update continuously
	•	partial results are visible before completion
	•	the app remains responsive (no blank screen / frozen state)
	•	user can cancel long sweeps and still export partial results
