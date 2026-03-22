"""
manual.py - CLI for running neuristor simulations without the GUI.

Supports:
  - single: single run for one Vin or a list of Vin
  - sweep1d: coarse->fine sweep over one free variable
  - sweep2d: 2D sweep for oscillation frequency (with heatmap + 3D scatter)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np

import neuristor.plots as plots
from neuristor.model import (
    YuanhangCircuitParams,
    YuanhangResistParams,
    series_first,
    simulate_vin_sweep,
    simulate_yuanhang,
)


def _parse_csv_floats(text: str) -> List[float]:
    if not text.strip():
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _paper_params() -> tuple[YuanhangResistParams, YuanhangCircuitParams]:
    resist = YuanhangResistParams()
    circuit = YuanhangCircuitParams()
    resist.R0 = 5.35882879e-3
    resist.Ea_over_k = 5.22047417e3
    resist.Rm0 = 262.5
    resist.Rm_factor = 4.90025335
    resist.w = 7.19357064
    resist.Tc_K = 3.32805839e2
    resist.beta = 2.52796285e-1
    resist.gamma = 9.56269682e-1
    resist.width_factor = 1.0
    resist.T_min_K = 305.0
    resist.T_max_K = 370.0
    resist.reversal_threshold_K = 0.01

    circuit.R_series_kohm = 12.0
    circuit.C_par_pF = 145.34619293
    circuit.Cth_mW_ns_per_K = 49.62776831
    circuit.Sth_mW_per_K = 0.20558726
    circuit.couple_factor = 0.0
    circuit.Cth_factor = 1.0
    circuit.noise_strength = 1e-3
    circuit.dimension = 1
    circuit.T_base_K = 325.0
    return resist, circuit


def _add_dataclass_args(parser: argparse.ArgumentParser, cls) -> None:
    for f in fields(cls):
        arg = f"--{f.name}"
        if f.type in (int, float):
            parser.add_argument(arg, type=f.type, default=None)
        else:
            parser.add_argument(arg, type=float, default=None)


def _apply_overrides(obj, args: argparse.Namespace) -> None:
    for f in fields(obj):
        val = getattr(args, f.name, None)
        if val is not None:
            setattr(obj, f.name, val)


def _write_sim_csv(simout: Dict, outpath: str) -> None:
    keys = ["time_s", "V_node", "I_load", "I_vo2", "T_K", "R_vo2", "g"]
    series = {k: series_first(simout[k]) for k in keys if k != "time_s"}
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i, t in enumerate(simout["time_s"]):
            row = [t] + [series[k][i] for k in keys if k != "time_s"]
            writer.writerow(row)


def _build_params(args: argparse.Namespace) -> tuple[YuanhangResistParams, YuanhangCircuitParams, tuple[int, int]]:
    if args.paper:
        resist, circuit = _paper_params()
    else:
        resist = YuanhangResistParams()
        circuit = YuanhangCircuitParams()

    _apply_overrides(resist, args)
    _apply_overrides(circuit, args)

    nx = max(1, int(args.nx))
    ny = max(1, int(args.ny))
    if args.dimension is not None:
        circuit.dimension = int(args.dimension)
    else:
        circuit.dimension = 1 if (nx == 1 or ny == 1) else 2
    return resist, circuit, (nx, ny)


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--paper", action="store_true", help="Use paper preset parameters")
    parser.add_argument("--t_end_us", type=float, default=300.0, help="Simulation duration (us)")
    parser.add_argument("--dt_ns", type=float, default=10.0, help="Timestep (ns)")
    parser.add_argument("--t_start_us", type=float, default=25.0, help="Steady-state window start (us)")
    parser.add_argument("--t_end_window_us", type=float, default=300.0, help="Steady-state window end (us)")
    parser.add_argument("--threshold_A", type=float, default=1e-3, help="Spike threshold (A)")
    parser.add_argument("--noise_seed", type=int, default=None, help="Noise seed (optional)")
    parser.add_argument("--start_branch", choices=["insulator", "metal"], default="insulator")
    parser.add_argument("--nx", type=int, default=1, help="Lattice Nx")
    parser.add_argument("--ny", type=int, default=1, help="Lattice Ny")
    _add_dataclass_args(parser, YuanhangResistParams)
    _add_dataclass_args(parser, YuanhangCircuitParams)
    return parser


def cmd_single(args: argparse.Namespace) -> None:
    resist, circuit, lattice_shape = _build_params(args)
    t_end = args.t_end_us * 1e-6
    dt = args.dt_ns * 1e-9

    vins = _parse_csv_floats(args.vin_list) if args.vin_list else [float(args.vin)]
    results = {}
    if len(vins) == 1:
        sim = simulate_yuanhang(
            Vin=vins[0],
            t_end=t_end,
            dt=dt,
            resist_params=resist,
            circuit_params=circuit,
            start_branch=args.start_branch,
            lattice_shape=lattice_shape,
            noise_seed=args.noise_seed,
        )
        results[vins[0]] = sim
    else:
        results = simulate_vin_sweep(
            vins,
            t_end=t_end,
            dt=dt,
            resist_params=resist,
            circuit_params=circuit,
            start_branch=args.start_branch,
            lattice_shape=lattice_shape,
            noise_seed=args.noise_seed,
        )

    if args.save_csv:
        out_dir = args.out_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        for v, sim in results.items():
            outpath = os.path.join(out_dir, f"sim_Vin_{v:.3f}.csv".replace(".", "p"))
            _write_sim_csv(sim, outpath)
            print(f"Wrote {outpath}")

    if not args.no_plots:
        for v, sim in results.items():
            plots.plot_time_traces(
                sim,
                R_series_kohm=circuit.R_series_kohm,
                title=f"Time traces (Vin={v:.3f} V)",
            )
        plt.show()


def cmd_sweep1d(args: argparse.Namespace) -> None:
    resist, circuit, lattice_shape = _build_params(args)
    t_end = args.t_end_us * 1e-6
    dt = args.dt_ns * 1e-9

    result = plots.sweep_free_variable_coarse_fine(
        param_name=args.param,
        start=args.start,
        stop=args.stop,
        coarse_step=args.coarse_step,
        fine_step=args.fine_step,
        Vin=float(args.vin),
        t_end=t_end,
        dt=dt,
        resist_params=resist,
        circuit_params=circuit,
        start_branch=args.start_branch,
        lattice_shape=lattice_shape,
        noise_seed=args.noise_seed,
        t_start_us=args.t_start_us,
        t_end_us=args.t_end_window_us,
        threshold_A=args.threshold_A,
    )

    if result["fine_results"] is None:
        print("No oscillatory band found in the coarse scan.")
        return

    print(f"Coarse band: {result['band_min']} – {result['band_max']}")
    if args.save_csv:
        outpath = args.out_csv or "sweep1d_results.csv"
        keys = ["values", "Vmax", "Pmax", "Pmin", "Tmax", "Tmin", "ISI_mean_us", "freq_MHz"]
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["value"] + keys[1:])
            for i, val in enumerate(result["fine_results"]["values"]):
                writer.writerow([val] + [result["fine_results"][k][i] for k in keys[1:]])
        print(f"Wrote {outpath}")

    if not args.no_plots:
        plots.plot_sweep_metrics(
            result["fine_results"],
            free_label=args.param,
            title_prefix=f"Sweep ({args.param})",
        )
        plt.show()


def cmd_sweep2d(args: argparse.Namespace) -> None:
    resist, circuit, lattice_shape = _build_params(args)
    t_end = args.t_end_us * 1e-6
    dt = args.dt_ns * 1e-9

    result = plots.sweep_frequency_2d(
        param_x=args.param_x,
        param_y=args.param_y,
        x_start=args.x_start,
        x_stop=args.x_stop,
        x_step=args.x_step,
        y_start=args.y_start,
        y_stop=args.y_stop,
        y_step=args.y_step,
        Vin=float(args.vin),
        t_end=t_end,
        dt=dt,
        resist_params=resist,
        circuit_params=circuit,
        start_branch=args.start_branch,
        lattice_shape=lattice_shape,
        noise_seed=args.noise_seed,
        t_start_us=args.t_start_us,
        t_end_us=args.t_end_window_us,
        threshold_A=args.threshold_A,
        row_early_stop=not args.no_row_stop,
        col_early_stop=not args.no_col_stop,
    )

    if args.save_csv:
        outpath = args.out_csv or "sweep2d_frequency.csv"
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.param_x, args.param_y, "freq_MHz"])
            for yi, y in enumerate(result["y_values"]):
                for xi, x in enumerate(result["x_values"]):
                    writer.writerow([x, y, result["freq_MHz"][yi][xi]])
        print(f"Wrote {outpath}")

    if not args.no_plots:
        plots.plot_frequency_2d(
            result,
            x_label=args.param_x,
            y_label=args.param_y,
            title_prefix=f"2D sweep ({args.param_x}, {args.param_y})",
            log_scale=args.log_scale,
            vmin=args.vmin,
            vmax=args.vmax,
            cmap_name=args.cmap,
            nan_color=args.nan_color,
        )
        plt.show()


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual CLI for neuristor simulations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_single = subparsers.add_parser("single", parents=[_common_parser()], help="Single run (Vin or Vin list)")
    p_single.add_argument("--vin", type=float, default=14.5, help="Vin (V)")
    p_single.add_argument("--vin_list", type=str, default="", help="Comma-separated Vin list")
    p_single.add_argument("--no-plots", action="store_true", help="Disable plots")
    p_single.add_argument("--save-csv", action="store_true", help="Write CSV output for each Vin")
    p_single.add_argument("--out-dir", type=str, default="", help="Output directory for CSVs")
    p_single.set_defaults(func=cmd_single)

    p_sweep = subparsers.add_parser("sweep1d", parents=[_common_parser()], help="Coarse/fine 1D sweep")
    p_sweep.add_argument("--param", type=str, required=True, help="Free variable to sweep")
    p_sweep.add_argument("--start", type=float, required=True, help="Coarse start")
    p_sweep.add_argument("--stop", type=float, required=True, help="Coarse stop")
    p_sweep.add_argument("--coarse-step", type=float, required=True, help="Coarse step size")
    p_sweep.add_argument("--fine-step", type=float, required=True, help="Fine step size")
    p_sweep.add_argument("--vin", type=float, default=14.5, help="Vin (fixed unless sweeping Vin)")
    p_sweep.add_argument("--no-plots", action="store_true", help="Disable plots")
    p_sweep.add_argument("--save-csv", action="store_true", help="Write fine results CSV")
    p_sweep.add_argument("--out-csv", type=str, default="", help="Output CSV path")
    p_sweep.set_defaults(func=cmd_sweep1d)

    p_sweep2d = subparsers.add_parser("sweep2d", parents=[_common_parser()], help="2D frequency sweep")
    p_sweep2d.add_argument("--param-x", type=str, required=True, help="X parameter")
    p_sweep2d.add_argument("--param-y", type=str, required=True, help="Y parameter")
    p_sweep2d.add_argument("--x-start", type=float, default=None, help="X start (optional)")
    p_sweep2d.add_argument("--x-stop", type=float, default=None, help="X stop (optional)")
    p_sweep2d.add_argument("--x-step", type=float, required=True, help="X step size")
    p_sweep2d.add_argument("--y-start", type=float, default=None, help="Y start (optional)")
    p_sweep2d.add_argument("--y-stop", type=float, default=None, help="Y stop (optional)")
    p_sweep2d.add_argument("--y-step", type=float, required=True, help="Y step size")
    p_sweep2d.add_argument("--vin", type=float, default=14.5, help="Vin (fixed unless sweeping Vin)")
    p_sweep2d.add_argument("--no-row-stop", action="store_true", help="Disable row early-stop")
    p_sweep2d.add_argument("--no-col-stop", action="store_true", help="Disable column early-stop")
    p_sweep2d.add_argument("--log-scale", action="store_true", help="Log scale for frequency heatmap")
    p_sweep2d.add_argument("--vmin", type=float, default=None, help="Colorbar vmin")
    p_sweep2d.add_argument("--vmax", type=float, default=None, help="Colorbar vmax")
    p_sweep2d.add_argument("--cmap", type=str, default="viridis", help="Colormap")
    p_sweep2d.add_argument("--nan-color", type=str, default="lightgray", help="NaN color")
    p_sweep2d.add_argument("--no-plots", action="store_true", help="Disable plots")
    p_sweep2d.add_argument("--save-csv", action="store_true", help="Write frequency CSV")
    p_sweep2d.add_argument("--out-csv", type=str, default="", help="Output CSV path")
    p_sweep2d.set_defaults(func=cmd_sweep2d)

    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
