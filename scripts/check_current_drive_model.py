"""
Sanity checks for the direct current-source VO2 model.

This script verifies the two key assumptions used by the current-driven module:
1) The first electrical Euler step matches dV/dt = I_in / C when V_init = 0.
2) Legacy R_out input is ignored and does not affect the simulated trajectory.

Usage:
  python scripts/check_current_drive_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step


def main() -> None:
    params = CurrentDriveParams(
        dt_s=10e-9,
        t_end_s=100e-9,
        t_pre_s=0.0,
        pulse_on_s=0.0,
        pulse_off_s=None,
        V_init_V=0.0,
        sigma_W_sqrt_s=0.0,
    )
    i_uA = 500.0
    out = simulate_current_step(i_uA, params=params, seed=1)

    c_f = float(params.C_F)
    dt = float(params.dt_s)
    i_in_a = i_uA * 1e-6
    expected_v1 = dt * i_in_a / c_f
    actual_v1 = float(out["V_vo2"][1])
    first_step_error = abs(actual_v1 - expected_v1)

    params_r0 = CurrentDriveParams(R_out_ohm=0.0)
    params_r1 = CurrentDriveParams(R_out_ohm=12345.0)
    out_r0 = simulate_current_step(i_uA, params=params_r0, seed=1)
    out_r1 = simulate_current_step(i_uA, params=params_r1, seed=1)
    invariance_error = float(np.max(np.abs(out_r0["V_vo2"] - out_r1["V_vo2"])))

    print("Current-drive sanity checks")
    print(f"expected first-step V[1] = {expected_v1:.12e} V")
    print(f"actual   first-step V[1] = {actual_v1:.12e} V")
    print(f"|error|                  = {first_step_error:.12e} V")
    print(f"max |delta V| when changing legacy R_out input = {invariance_error:.12e} V")

    if first_step_error > 1e-12:
        raise SystemExit("FAILED: first-step Euler update does not match direct current-source ODE.")
    if invariance_error > 1e-15:
        raise SystemExit("FAILED: R_out metadata is affecting the trajectory.")

    print("PASS: current-drive implementation matches the direct ideal-current-source model.")


if __name__ == "__main__":
    main()
