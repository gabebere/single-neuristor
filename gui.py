"""
GUI wrapper for the VO₂ neuristor simulator/analysis.

Provides buttons to run common plots from plots.py using the simulator in model.py:
- Baselines (minima) of T and V vs time
- V_max vs Vin
- Power vs time and power peaks vs Vin
- Capacitance overlays and cycle-averaged power extrema vs C
- Frequency vs (C, R_load) 3D scatter
- Resistance in insulating state vs time

Heater note: the GUI currently wraps the existing simulator; a heater-coupled
thermal equation is not yet implemented and is exposed as a stub action here.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")  # Use Tk backend for matplotlib UI
import matplotlib.pyplot as plt  # noqa: E402

from model import (
    YuanhangCircuitParams,
    YuanhangResistParams,
    is_oscillatory,
    simulate_yuanhang,
)
import plots as analysis


def _parse_float(value: str, default: float) -> float:
    return default if not value.strip() else float(value)


def _parse_int(value: str, default: int | None) -> int | None:
    return default if not value.strip() else int(value)


def _parse_csv_floats(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(x.strip()) for x in value.split(",") if x.strip()]


class NeuristorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("VO₂ Neuristor Simulator GUI")

        self.paper_var = tk.BooleanVar(value=False)
        self.branch_var = tk.StringVar(value="insulator")
        self.vin_var = tk.StringVar(value="14.5")
        self.vin_list_var = tk.StringVar(value="10.5,12.5,14.5,16.5")
        self.example_vin_var = tk.StringVar(value="14.5")
        self.t_end_us_var = tk.StringVar(value="300.0")
        self.dt_ns_var = tk.StringVar(value="10.0")
        self.t_start_us_var = tk.StringVar(value="25.0")
        self.t_window_end_us_var = tk.StringVar(value="300.0")
        self.threshold_var = tk.StringVar(value="1e-3")
        self.rload_var = tk.StringVar(value="")
        self.cpar_var = tk.StringVar(value="")
        self.nx_var = tk.StringVar(value="1")
        self.ny_var = tk.StringVar(value="1")
        self.noise_seed_var = tk.StringVar(value="")
        self.c_list_var = tk.StringVar(value="100,150,200")
        self.c_start_var = tk.StringVar(value="80")
        self.c_stop_var = tk.StringVar(value="250")
        self.c_step_var = tk.StringVar(value="10")
        self.r_list_var = tk.StringVar(value="10,12,14,16")

        self._build_layout()

    # ------------------------------------------------------------------ UI helpers
    def _build_layout(self) -> None:
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Simulation parameters
        params = ttk.LabelFrame(frm, text="Simulation parameters")
        params.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        ttk.Label(params, text="Vin (V)").grid(row=0, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.vin_var, width=12).grid(row=0, column=1, sticky="w")
        ttk.Label(params, text="Vin list (csv)").grid(row=0, column=2, sticky="w")
        ttk.Entry(params, textvariable=self.vin_list_var, width=28).grid(row=0, column=3, sticky="w")
        ttk.Label(params, text="Example Vin (for power plots)").grid(row=0, column=4, sticky="w")
        ttk.Entry(params, textvariable=self.example_vin_var, width=10).grid(row=0, column=5, sticky="w")

        ttk.Label(params, text="t_end (µs)").grid(row=1, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.t_end_us_var, width=12).grid(row=1, column=1, sticky="w")
        ttk.Label(params, text="dt (ns)").grid(row=1, column=2, sticky="w")
        ttk.Entry(params, textvariable=self.dt_ns_var, width=12).grid(row=1, column=3, sticky="w")
        ttk.Label(params, text="Noise seed").grid(row=1, column=4, sticky="w")
        ttk.Entry(params, textvariable=self.noise_seed_var, width=10).grid(row=1, column=5, sticky="w")

        ttk.Label(params, text="R_load (kΩ)").grid(row=2, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.rload_var, width=12).grid(row=2, column=1, sticky="w")
        ttk.Label(params, text="C_par (pF)").grid(row=2, column=2, sticky="w")
        ttk.Entry(params, textvariable=self.cpar_var, width=12).grid(row=2, column=3, sticky="w")
        ttk.Label(params, text="Grid Nx, Ny").grid(row=2, column=4, sticky="w")
        ttk.Entry(params, textvariable=self.nx_var, width=5).grid(row=2, column=5, sticky="w")
        ttk.Entry(params, textvariable=self.ny_var, width=5).grid(row=2, column=6, sticky="w")

        ttk.Checkbutton(params, text="Paper preset", variable=self.paper_var).grid(row=3, column=0, sticky="w")
        ttk.Label(params, text="Start branch").grid(row=3, column=2, sticky="e")
        ttk.Combobox(params, textvariable=self.branch_var, values=["insulator", "metal"], width=10).grid(
            row=3, column=3, sticky="w"
        )

        # Window / threshold
        window = ttk.LabelFrame(frm, text="Steady-state window / spike detection")
        window.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(window, text="t_start (µs)").grid(row=0, column=0, sticky="w")
        ttk.Entry(window, textvariable=self.t_start_us_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(window, text="t_end (µs)").grid(row=0, column=2, sticky="w")
        ttk.Entry(window, textvariable=self.t_window_end_us_var, width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(window, text="Spike threshold (A)").grid(row=0, column=4, sticky="w")
        ttk.Entry(window, textvariable=self.threshold_var, width=12).grid(row=0, column=5, sticky="w")

        # Sweep parameters
        sweep = ttk.LabelFrame(frm, text="Sweep parameters")
        sweep.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(sweep, text="C list (pF)").grid(row=0, column=0, sticky="w")
        ttk.Entry(sweep, textvariable=self.c_list_var, width=20).grid(row=0, column=1, sticky="w")
        ttk.Label(sweep, text="C sweep start/stop/step (pF)").grid(row=0, column=2, sticky="w")
        ttk.Entry(sweep, textvariable=self.c_start_var, width=8).grid(row=0, column=3, sticky="w")
        ttk.Entry(sweep, textvariable=self.c_stop_var, width=8).grid(row=0, column=4, sticky="w")
        ttk.Entry(sweep, textvariable=self.c_step_var, width=8).grid(row=0, column=5, sticky="w")
        ttk.Label(sweep, text="R_load list (kΩ) for 3D").grid(row=1, column=0, sticky="w")
        ttk.Entry(sweep, textvariable=self.r_list_var, width=30).grid(row=1, column=1, columnspan=2, sticky="w")

        # Buttons
        btns = ttk.LabelFrame(frm, text="Actions")
        btns.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        ttk.Button(btns, text="Baselines: T & V vs time", command=self.do_baselines).grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="V_max vs Vin", command=self.do_vmax_vs_vin).grid(row=0, column=1, sticky="ew")
        ttk.Button(btns, text="Power time + peaks vs Vin", command=self.do_power_vs_vin).grid(
            row=0, column=2, sticky="ew"
        )
        ttk.Button(btns, text="Capacitance overlay (power)", command=self.do_c_power_overlay).grid(
            row=1, column=0, sticky="ew"
        )
        ttk.Button(btns, text="Power extrema vs C", command=self.do_c_power_extrema).grid(
            row=1, column=1, sticky="ew"
        )
        ttk.Button(btns, text="Frequency 3D vs (C, R_load)", command=self.do_freq_3d).grid(
            row=1, column=2, sticky="ew"
        )
        ttk.Button(btns, text="R_insulating vs time", command=self.do_r_ins).grid(row=2, column=0, sticky="ew")
        ttk.Button(btns, text="Oscillation check", command=self.do_is_oscillatory).grid(
            row=2, column=1, sticky="ew"
        )
        ttk.Button(btns, text="Heater (stub)", command=self.do_heater_stub).grid(row=2, column=2, sticky="ew")

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frm, textvariable=self.status_var, foreground="blue").grid(row=4, column=0, sticky="w")

    # ------------------------------------------------------------------ Core helpers
    def _build_params(self) -> tuple[YuanhangResistParams, YuanhangCircuitParams]:
        resist = YuanhangResistParams()
        circuit = YuanhangCircuitParams()
        if not self.paper_var.get():
            if self.rload_var.get().strip():
                circuit.R_series_kohm = float(self.rload_var.get())
            if self.cpar_var.get().strip():
                circuit.C_par_pF = float(self.cpar_var.get())
        return resist, circuit

    def _common_args(self) -> dict:
        return {
            "t_end_us": float(self.t_end_us_var.get()),
            "dt_ns": float(self.dt_ns_var.get()),
            "t_start_us": float(self.t_start_us_var.get()),
            "t_end_window_us": float(self.t_window_end_us_var.get()),
            "threshold_A": float(self.threshold_var.get()),
            "noise_seed": _parse_int(self.noise_seed_var.get(), None),
            "branch": self.branch_var.get(),
            "lattice_shape": (max(1, int(self.nx_var.get())), max(1, int(self.ny_var.get()))),
        }

    def _run_single(self) -> analysis.SimOut:
        resist, circuit = self._build_params()
        args = self._common_args()
        return simulate_yuanhang(
            Vin=float(self.vin_var.get()),
            t_end=args["t_end_us"] * 1e-6,
            dt=args["dt_ns"] * 1e-9,
            resist_params=resist,
            circuit_params=circuit,
            start_branch=args["branch"],
            lattice_shape=args["lattice_shape"],
            noise_seed=args["noise_seed"],
        )

    def _run_vin_sweep(self, vin_list: list[float]) -> dict[float, analysis.SimOut]:
        resist, circuit = self._build_params()
        args = self._common_args()
        results: dict[float, analysis.SimOut] = {}
        for v in vin_list:
            results[v] = simulate_yuanhang(
                Vin=v,
                t_end=args["t_end_us"] * 1e-6,
                dt=args["dt_ns"] * 1e-9,
                resist_params=resist,
                circuit_params=circuit,
                start_branch=args["branch"],
                lattice_shape=args["lattice_shape"],
                noise_seed=None if args["noise_seed"] is None else args["noise_seed"] + int(v * 10),
            )
        return results

    def _show_status(self, msg: str) -> None:
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _get_vin_list(self) -> list[float]:
        vins = _parse_csv_floats(self.vin_list_var.get())
        if not vins:
            raise ValueError("Vin list is required for this action.")
        return vins

    # ------------------------------------------------------------------ Actions
    def do_baselines(self) -> None:
        try:
            data = self._run_single()
            args = self._common_args()
            analysis.plot_baselines_T_and_V_vs_time(
                data,
                t_start_us=args["t_start_us"],
                t_end_us=args["t_end_window_us"],
            )
            self._show_status("Baselines plotted.")
            plt.show()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", str(e))

    def do_vmax_vs_vin(self) -> None:
        try:
            vins = self._get_vin_list()
            results = self._run_vin_sweep(vins)
            args = self._common_args()
            analysis.plot_Vmax_vs_Vin(vins, results, t_start_us=args["t_start_us"], t_end_us=args["t_end_window_us"])
            self._show_status("V_max vs Vin plotted.")
            plt.show()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", str(e))

    def do_power_vs_vin(self) -> None:
        try:
            vins = self._get_vin_list()
            results = self._run_vin_sweep(vins)
            args = self._common_args()
            example_v = float(self.example_vin_var.get() or vins[0])
            analysis.plot_power_time_and_peaks_vs_Vin(
                vins,
                results,
                example_vin=example_v,
                t_start_us=args["t_start_us"],
                t_end_us=args["t_end_window_us"],
            )
            self._show_status("Power plots generated.")
            plt.show()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", str(e))

    def do_c_power_overlay(self) -> None:
        try:
            C_vals = _parse_csv_floats(self.c_list_var.get())
            if not C_vals:
                raise ValueError("Provide a comma-separated C list.")
            resist, circuit = self._build_params()
            args = self._common_args()
            analysis.plot_capacitance_effect_on_power(
                vin=float(self.vin_var.get()),
                C_values_pF=C_vals,
                resist_params=resist,
                circuit_params=circuit,
                t_end=args["t_end_us"] * 1e-6,
                dt=args["dt_ns"] * 1e-9,
                t_start_us=args["t_start_us"],
                t_end_us=args["t_end_window_us"],
            )
            self._show_status("Capacitance overlay plotted.")
            plt.show()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", str(e))

    def do_c_power_extrema(self) -> None:
        try:
            resist, circuit = self._build_params()
            args = self._common_args()
            analysis.plot_capacitance_sweep_power_extrema(
                vin=float(self.vin_var.get()),
                C_start_pF=float(self.c_start_var.get()),
                C_stop_pF=float(self.c_stop_var.get()),
                C_step_pF=float(self.c_step_var.get()),
                resist_params=resist,
                circuit_params=circuit,
                t_end=args["t_end_us"] * 1e-6,
                dt=args["dt_ns"] * 1e-9,
                t_start_us=args["t_start_us"],
                t_end_us=args["t_end_window_us"],
            )
            self._show_status("Power extrema vs C plotted.")
            plt.show()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", str(e))

    def do_freq_3d(self) -> None:
        try:
            C_vals = _parse_csv_floats(self.c_list_var.get())
            R_vals = _parse_csv_floats(self.r_list_var.get())
            if not C_vals or not R_vals:
                raise ValueError("Provide C list and R_load list for 3D plot.")
            resist, circuit = self._build_params()
            args = self._common_args()
            analysis.plot_frequency_3d_vs_C_and_Rload(
                vin=float(self.vin_var.get()),
                C_values_pF=C_vals,
                Rload_values_kohm=R_vals,
                resist_params=resist,
                circuit_params=circuit,
                t_end=args["t_end_us"] * 1e-6,
                dt=args["dt_ns"] * 1e-9,
                threshold_A=float(self.threshold_var.get()),
            )
            self._show_status("Frequency 3D plotted.")
            plt.show()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", str(e))

    def do_r_ins(self) -> None:
        try:
            data = self._run_single()
            args = self._common_args()
            analysis.plot_R_insulating_vs_time(
                data,
                t_start_us=args["t_start_us"],
                t_end_us=args["t_end_window_us"],
            )
            self._show_status("R insulating vs time plotted.")
            plt.show()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", str(e))

    def do_is_oscillatory(self) -> None:
        try:
            data = self._run_single()
            args = self._common_args()
            osc = is_oscillatory(
                data,
                t_start_us=args["t_start_us"],
                t_end_us=args["t_end_window_us"],
                threshold_A=float(self.threshold_var.get()),
                min_spikes=4,
            )
            msg = "Oscillatory in window" if osc else "Not oscillatory in window"
            self._show_status(msg)
            messagebox.showinfo("Oscillation check", msg)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", str(e))

    def do_heater_stub(self) -> None:
        message = (
            "Heater-coupled simulation not implemented yet.\n"
            "Add a second thermal equation with fixed-resistance heater to model.py, "
            "then wire it here."
        )
        self._show_status("Heater stub invoked.")
        messagebox.showinfo("Heater", message)


def main() -> None:
    root = tk.Tk()
    NeuristorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
