"""
Quick test script to exercise time-trace plots and 1D coarse/fine sweeps.

Use --mode time-trace or --mode sweep. Run with --show-prompts for example commands.
"""
from __future__ import annotations

import argparse
import matplotlib.pyplot as plt

import plots as analysis
from model import YuanhangCircuitParams, YuanhangResistParams, simulate_yuanhang


def _example_prompts() -> str:
    return "\n".join(
        [
            "python test.py --mode time-trace --vin 14.5",
            "python test.py --mode sweep --param Vin --start 0 --stop 20 --coarse-step 0.5 --fine-step 0.05",
            "python test.py --mode sweep --param C_par_pF --start 80 --stop 250 --coarse-step 20 --fine-step 5 --vin 14.5",
            "python test.py --mode sweep2d --param-x Vin --param-y C_par_pF --x-start 0 --x-stop 20 --x-step 0.5 --y-start 80 --y-stop 250 --y-step 10",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test neuristor plots and sweep metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n" + _example_prompts(),
    )
    parser.add_argument("--mode", choices=["time-trace", "sweep", "sweep2d"], default="time-trace")
    parser.add_argument("--vin", type=float, default=14.5, help="Bias voltage Vin (V)")
    parser.add_argument("--t_end_us", type=float, default=300.0, help="Simulation duration (us)")
    parser.add_argument("--dt_ns", type=float, default=10.0, help="Timestep (ns)")
    parser.add_argument("--t_start_us", type=float, default=25.0, help="Steady-state window start (us)")
    parser.add_argument("--t_end_window_us", type=float, default=300.0, help="Steady-state window end (us)")
    parser.add_argument("--threshold_A", type=float, default=1e-3, help="Spike detection threshold (A)")
    parser.add_argument("--param", type=str, default="Vin", help="Free variable to sweep (e.g., Vin, C_par_pF)")
    parser.add_argument("--start", type=float, default=0.0, help="Coarse sweep start")
    parser.add_argument("--stop", type=float, default=20.0, help="Coarse sweep stop")
    parser.add_argument("--coarse-step", type=float, default=0.5, dest="coarse_step", help="Coarse step size")
    parser.add_argument("--fine-step", type=float, default=0.05, dest="fine_step", help="Fine step size")
    parser.add_argument("--param-x", type=str, default="Vin", help="2D sweep param X")
    parser.add_argument("--param-y", type=str, default="C_par_pF", help="2D sweep param Y")
    parser.add_argument("--x-start", type=float, default=None, help="X sweep start (leave blank for auto)")
    parser.add_argument("--x-stop", type=float, default=None, help="X sweep stop (leave blank for auto)")
    parser.add_argument("--x-step", type=float, default=0.5, help="X sweep step")
    parser.add_argument("--y-start", type=float, default=None, help="Y sweep start (leave blank for auto)")
    parser.add_argument("--y-stop", type=float, default=None, help="Y sweep stop (leave blank for auto)")
    parser.add_argument("--y-step", type=float, default=10.0, help="Y sweep step")
    parser.add_argument("--show-prompts", action="store_true", help="Print example commands and exit")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.show_prompts:
        print(_example_prompts())
        return

    resist = YuanhangResistParams()
    circuit = YuanhangCircuitParams()

    if args.mode == "time-trace":
        sim = simulate_yuanhang(
            Vin=args.vin,
            t_end=args.t_end_us * 1e-6,
            dt=args.dt_ns * 1e-9,
            resist_params=resist,
            circuit_params=circuit,
        )
        analysis.plot_time_traces(
            sim,
            R_series_kohm=circuit.R_series_kohm,
            title="Time traces (V_vo2/V_load, T_vo2, I_vo2, P_vo2)",
        )
        plt.show()
        return

    if args.mode == "sweep":
        result = analysis.sweep_free_variable_coarse_fine(
            param_name=args.param,
            start=args.start,
            stop=args.stop,
            coarse_step=args.coarse_step,
            fine_step=args.fine_step,
            Vin=args.vin,
            t_end=args.t_end_us * 1e-6,
            dt=args.dt_ns * 1e-9,
            resist_params=resist,
            circuit_params=circuit,
            t_start_us=args.t_start_us,
            t_end_us=args.t_end_window_us,
            threshold_A=args.threshold_A,
        )

        if result["fine_results"] is None:
            print("No oscillatory band found in the coarse scan.")
            return

        band_min = result["band_min"]
        band_max = result["band_max"]
        print(f"Coarse band: {band_min} – {band_max}")
        analysis.plot_sweep_metrics(
            result["fine_results"],
            free_label=args.param,
            title_prefix=f"Sweep ({args.param})",
        )
        plt.show()
        return

    if args.mode == "sweep2d":
        result2d = analysis.sweep_frequency_2d(
            param_x=args.param_x,
            param_y=args.param_y,
            x_start=args.x_start,
            x_stop=args.x_stop,
            x_step=args.x_step,
            y_start=args.y_start,
            y_stop=args.y_stop,
            y_step=args.y_step,
            Vin=args.vin,
            t_end=args.t_end_us * 1e-6,
            dt=args.dt_ns * 1e-9,
            resist_params=resist,
            circuit_params=circuit,
            t_start_us=args.t_start_us,
            t_end_us=args.t_end_window_us,
            threshold_A=args.threshold_A,
        )
        analysis.plot_frequency_2d(
            result2d,
            x_label=args.param_x,
            y_label=args.param_y,
            title_prefix=f"2D sweep ({args.param_x}, {args.param_y})",
        )
        plt.show()
        return


if __name__ == "__main__":
    main()
