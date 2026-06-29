# Single VO₂ Neuristor Simulation

This repository contains simulation code for a single VO₂ neuristor, including:
- A physics‑faithful simulator in `src/neuristor/model.py`
- An ideal current-driven simulator in `src/neuristor/current_drive_sim.py`
- A current-domain search backend in `src/neuristor/current_domain_search.py`
- Analysis/plotting utilities in `src/neuristor/plots.py`
- A manual CLI in `scripts/manual.py` (single runs, 1D sweeps, 2D frequency sweeps)
- A Streamlit GUI in `app.py`
- A specimen-fitting utility in `src/neuristor/resistance_custom_analysis.py`

Created by Gabriel Berezovsky under the supervision of PhD candidate Amir Gildor in the Quantum Materials for Neuromorphic Computation Lab at the Technion.

This project models the electrical and thermal dynamics of a VO₂ neuristor and reproduces spiking behavior.

## Repository layout

These are the main files worth reading first:

- `app.py`: Streamlit interface and job orchestration
- `src/neuristor/model.py`: authoritative voltage-driven model and hysteresis implementation
- `src/neuristor/current_drive_sim.py`: ideal current-source single-device simulator
- `src/neuristor/current_domain_search.py`: parameter/domain search backend for current-driven runs
- `src/neuristor/resistance_custom_analysis.py`: fit resistance/hysteresis parameters from measured `R(T)` data
- `scripts/manual.py`: CLI entrypoint for voltage-driven runs and sweeps
- `src/neuristor/plots.py`: post-processing and plotting helpers
- `scripts/`: one-off analysis utilities for validation, figure generation, and current-drive studies
- `presets/`: saved fitted/sample parameter sets
- `data/experimental/`: measured data used for fitting
- `docs/manuscript/`: manuscript source, compiled PDF, and manuscript figures
- `references/papers/`: paper PDFs used as modeling references
- `references/yuanhangzhang98-collective_dynamics_neuristor-217d4f0/`: upstream reference code
- `docs/HYSTERESIS_IMPLEMENTATION_AUDIT.md`: hysteresis implementation, branch conventions, provenance, and validation rules
- `jobs/`: local-only Streamlit run history, ignored by Git
- `public_jobs/`: curated Streamlit run history that can be committed and shared

The root is intentionally minimal. `app.py` is the only top-level application entrypoint.
Implementation code lives under `src/neuristor/`, and auxiliary command-line tools live under `scripts/`.

## Job History Storage

Streamlit run history is split into two lanes:

- Local/private jobs are saved under `jobs/`; this folder is ignored and should be used for exploratory sweeps.
- Public jobs are saved under `public_jobs/`; this folder is intentionally not ignored so selected runs can be reviewed through GitHub.

Use the **Save in public history** toggle in the experiment forms only for runs that should be shared. The History tab can filter between both lanes.

## Custom Resistance Calibration (Experimental Specimen)

This repo includes a specimen-fitting utility for calibrating `YuanhangResistParams` from measured `R(T)` data:

```bash
python scripts/fit_resistance.py \
  --data data/experimental/100425_chip1_gap3.tsv \
  --save-json presets/resistance_100425_chip1_gap3.json \
  --save-plot outputs/resistance_fit_100425_chip1_gap3.png
```

The generated preset can be loaded in the Streamlit app via the sidebar button:
- `Load specimen resistance preset`

This button only updates resistance/hysteresis parameters (and initial branch), so it does not overwrite your circuit/time controls.

For current-input runs, the app also provides a one-click combined preset in the **Current-Driven Sweep** section:
- `Load professor preset`

This applies paper current/thermal defaults plus the specimen RT-fitted resistance parameters together.

Current-drive ODE assumption used in this repo: ideal current source at the VO2 node
(`dV/dt = (I_in - V/R_vo2)/C`). External/source series resistance is not part of that reduced model.

## Performance and fidelity

- The executable hysteresis path is the upstream-faithful Yuanhang float32 implementation.
- Torch evaluates the hysteresis transcendental functions; NumPy stores and integrates simulation state.
- Deterministic quasistatic current sweeps are vectorized across current amplitudes and are tested for exact equality with serial traces.
- Stochastic, dynamic-phase, and multidomain sweeps retain the serial path to preserve their existing semantics.
- Streamlit caches unchanged sample presets and completed job CSVs. Cache keys include file timestamps and sizes, so edited files invalidate automatically.

## Installation and GUI (Streamlit)

Below is a thorough, step-by-step guide for first-time users (PI/PhD/student).

### 0) Prerequisites (one-time)

You need:
- Git (for cloning the repo)
- Python 3.10+ (required because the code uses modern typing syntax)

Check Python:
```
python --version
```

If Python is missing, install it from https://www.python.org/downloads/ or use Anaconda/Miniconda.

Check Git:
```
git --version
```

### 1) Fork the repository (recommended)

1) Open this repo in GitHub.
2) Click **Fork** (top right) to create your own copy under your GitHub account.

Why fork?
- You can pull updates from the main repo but keep your own changes in your fork.

### 2) Clone your fork to your computer

Open a terminal and run:
```
git clone https://github.com/<your-username>/<your-fork>.git
cd <your-fork>
```

If you do not need to modify the code, you can clone the main repo directly:
```
git clone https://github.com/gabebere/single-neuristor.git
cd single-neuristor
```

### 3) Create and activate a virtual environment

From the repo root:
```
python -m venv .venv
```

Activate it:
```
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

You should now see the environment name in your terminal prompt (e.g., `.venv`).

### 4) Install Python dependencies

```
pip install -r requirements.txt
```

If you use Anaconda:
```
conda create -n neuristor python=3.11
conda activate neuristor
pip install -r requirements.txt
```

### 5) Run the GUI (Streamlit)

From the repo root:
```
streamlit run app.py
```

Streamlit will print a URL (usually `http://localhost:8501`). Open it in your browser.

### 6) (Optional) Update your local copy later

If you forked and want the latest changes:
```
git pull origin main
```

If you cloned the main repo directly:
```
git pull origin main
```

### Troubleshooting

- **`ModuleNotFoundError`**: dependencies aren’t installed. Re-run `pip install -r requirements.txt`.
- **`streamlit` not found**: your environment isn’t active. Activate `.venv`, then retry.
- **Port already in use**: run `streamlit run app.py --server.port 8502`.
- **Click-to-run points not working**: ensure `streamlit-plotly-events` is installed (it is included in `requirements.txt`).

## Manual CLI (`scripts/manual.py`)

Run from the repo root:
```
python scripts/manual.py --help
```

### Commands

1) Single run (one Vin or Vin list)
```
python scripts/manual.py single --vin 14.5
python scripts/manual.py single --vin_list "10.5,12.5,14.5"
```

2) 1D sweep (coarse → fine) over any scalar parameter
```
python scripts/manual.py sweep1d --param Vin --start 0 --stop 20 --coarse-step 0.5 --fine-step 0.05
python scripts/manual.py sweep1d --param C_par_pF --start 80 --stop 250 --coarse-step 20 --fine-step 5 --vin 14.5
```

Outputs: Vmax, Pmax/Pmin, Tmax/Tmin, frequency, mean ISI vs the free variable.

3) 2D sweep (frequency vs two parameters, 3D scatter + heatmap)
```
python scripts/manual.py sweep2d --param-x Vin --param-y C_par_pF --x-start 0 --x-stop 20 --x-step 0.5 --y-start 80 --y-stop 250 --y-step 10
```

If `x_stop` or `y_stop` is omitted, the code scans up to 100 coarse steps. If oscillations never terminate, it uses the full scanned range.

### Common options (all commands)

- Time/analysis: `--t_end_us`, `--dt_ns`, `--t_start_us`, `--t_end_window_us`, `--threshold_A`
- Lattice: `--nx`, `--ny`, `--dimension`
- Hysteresis/Resistance (YuanhangResistParams):  
  `--R0`, `--Ea_over_k`, `--Rm0`, `--Rm_factor`, `--w`, `--Tc_K`, `--beta`, `--gamma`,  
  `--width_factor`, `--T_min_K`, `--T_max_K`, `--reversal_threshold_K`
- Circuit/Thermal (YuanhangCircuitParams):  
  `--R_series_kohm`, `--C_par_pF`, `--Cth_mW_ns_per_K`, `--Sth_mW_per_K`,  
  `--couple_factor`, `--Cth_factor`, `--noise_strength`, `--T_base_K`
- Preset: `--paper`

Use `python scripts/manual.py <command> --help` to list all flags for that command.

### Plotting and export flags

- `--no-plots` (all commands) disables plot windows.
- `single`: `--save-csv` and `--out-dir`
- `sweep1d`: `--save-csv` and `--out-csv`
- `sweep2d`: `--save-csv` and `--out-csv`

### Example: fixed parameters + Vin sweep
```
python scripts/manual.py sweep1d \
  --param Vin --start 10.5 --stop 15.0 --coarse-step 0.5 --fine-step 0.05 \
  --R_series_kohm 12 --C_par_pF 145.34619293
```

## Quick start (`scripts/manual.py`)

1) Single run + plots (default Vin=14.5 V):
```
python scripts/manual.py single
```

2) Single run + CSV export:
```
python scripts/manual.py single --vin 18.85 --C_par_pF 198 --save-csv --out-dir outputs
```
Outputs:
- `outputs/sim_Vin_18p850.csv`

3) 1D sweep + plots (Vin sweep):
```
python scripts/manual.py sweep1d --param Vin --start 10.5 --stop 15.0 --coarse-step 0.5 --fine-step 0.05
```

4) 1D sweep + CSV export:
```
python scripts/manual.py sweep1d --param C_par_pF --start 80 --stop 250 --coarse-step 20 --fine-step 5 --vin 14.5 --save-csv --out-csv sweep1d_results.csv
```
Outputs:
- `sweep1d_results.csv`

5) 2D frequency sweep + plots (3D + heatmap):
```
python scripts/manual.py sweep2d --param-x Vin --param-y C_par_pF --x-start 10.5 --x-stop 15.0 --x-step 0.5 --y-start 120 --y-stop 210 --y-step 10
```

6) 2D sweep + CSV export:
```
python scripts/manual.py sweep2d --param-x Vin --param-y C_par_pF --x-start 10.5 --x-stop 15.0 --x-step 0.5 --y-start 120 --y-stop 210 --y-step 10 --save-csv --out-csv sweep2d_frequency.csv
```
Outputs:
- `sweep2d_frequency.csv`
