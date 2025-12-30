# Single VO₂ Neuristor Simulation

This repository contains simulation code for a single VO₂ neuristor, including:
- A physics‑faithful simulator in `model.py`
- Analysis/plotting utilities in `plots.py`
- A manual CLI in `manual.py` (single runs, 1D sweeps, 2D frequency sweeps)
- A Streamlit GUI in `app.py`

Created by Gabriel Berezovsky under the supervision of PhD candidate Amir Gildor in the Quantum Materials for Neuromorphic Computation Lab at the Technion.

This project models the electrical and thermal dynamics of a VO₂ neuristor and reproduces spiking behavior.

## Installation and GUI (Streamlit)

### Fork and clone

1) In GitHub, open this repo and click **Fork** (top right).
2) Clone your fork:
```
git clone https://github.com/<your-username>/<your-fork>.git
cd <your-fork>
```

### Local setup

1) Create a virtual environment:
```
python -m venv .venv
```
2) Activate it:
```
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```
3) Install dependencies:
```
pip install -r requirements.txt
```

### Run the app

From the repo root:
```
streamlit run app.py
```
Then open the URL printed in the terminal (usually `http://localhost:8501`).

## Manual CLI (manual.py)

Run from the repo root:
```
python manual.py --help
```

### Commands

1) Single run (one Vin or Vin list)
```
python manual.py single --vin 14.5
python manual.py single --vin_list "10.5,12.5,14.5"
```

2) 1D sweep (coarse → fine) over any scalar parameter
```
python manual.py sweep1d --param Vin --start 0 --stop 20 --coarse-step 0.5 --fine-step 0.05
python manual.py sweep1d --param C_par_pF --start 80 --stop 250 --coarse-step 20 --fine-step 5 --vin 14.5
```

Outputs: Vmax, Pmax/Pmin, Tmax/Tmin, frequency, mean ISI vs the free variable.

3) 2D sweep (frequency vs two parameters, 3D scatter + heatmap)
```
python manual.py sweep2d --param-x Vin --param-y C_par_pF --x-start 0 --x-stop 20 --x-step 0.5 --y-start 80 --y-stop 250 --y-step 10
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

Use `python manual.py <command> --help` to list all flags for that command.

### Plotting and export flags

- `--no-plots` (all commands) disables plot windows.
- `single`: `--save-csv` and `--out-dir`
- `sweep1d`: `--save-csv` and `--out-csv`
- `sweep2d`: `--save-csv` and `--out-csv`

### Example: fixed parameters + Vin sweep
```
python manual.py sweep1d \
  --param Vin --start 10.5 --stop 15.0 --coarse-step 0.5 --fine-step 0.05 \
  --R_series_kohm 12 --C_par_pF 145.34619293
```

## Quick start (manual.py)

1) Single run + plots (default Vin=14.5 V):
```
python manual.py single
```

2) Single run + CSV export:
```
python manual.py single --vin 18.85 --C_par_pF 198 --save-csv --out-dir outputs
```
Outputs:
- `outputs/sim_Vin_18p850.csv`

3) 1D sweep + plots (Vin sweep):
```
python manual.py sweep1d --param Vin --start 10.5 --stop 15.0 --coarse-step 0.5 --fine-step 0.05
```

4) 1D sweep + CSV export:
```
python manual.py sweep1d --param C_par_pF --start 80 --stop 250 --coarse-step 20 --fine-step 5 --vin 14.5 --save-csv --out-csv sweep1d_results.csv
```
Outputs:
- `sweep1d_results.csv`

5) 2D frequency sweep + plots (3D + heatmap):
```
python manual.py sweep2d --param-x Vin --param-y C_par_pF --x-start 10.5 --x-stop 15.0 --x-step 0.5 --y-start 120 --y-stop 210 --y-step 10
```

6) 2D sweep + CSV export:
```
python manual.py sweep2d --param-x Vin --param-y C_par_pF --x-start 10.5 --x-stop 15.0 --x-step 0.5 --y-start 120 --y-stop 210 --y-step 10 --save-csv --out-csv sweep2d_frequency.csv
```
Outputs:
- `sweep2d_frequency.csv`
