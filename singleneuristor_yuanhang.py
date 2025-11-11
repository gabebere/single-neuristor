"""
Single VO₂ neuristor simulator (Yuanhang Zhang et al. variant).

This script mirrors the hysteresis, unit conventions, and RC/thermal ODE
coefficients from `yuanhangzhang98-collective_dynamics_neuristor-217d4f0/model.py`
while keeping the single-device workflow, CSV export, and plotting helpers from
`singleneuristor.py`.
"""
from __future__ import annotations

import math
import random

from dataclasses import dataclass
from typing import Dict, List, TypedDict


_EPS: float = 1e-12
_PI: float = math.pi
_KELVIN_OFFSET: float = 273.15


def P(x: float, gamma: float) -> float:
    """Proximity window Π(x;γ) used by the hysteresis model (Zhang et al.)."""
    return 0.5 * (1.0 - math.sin(gamma * x)) * (1.0 + math.tanh(_PI * _PI - 2.0 * _PI * x))


@dataclass
class YuanhangResistParams:
    R0: float = 5.35882879e-3          # Ohm
    Ea_over_k: float = 5.22047417e3    # Kelvin (Ea/kB)
    Rm0: float = 262.5                 # Ohm
    Rm_factor: float = 4.90025335      # dimensionless
    w: float = 7.19357064              # Kelvin (loop width)
    Tc_K: float = 3.32805839e2         # Kelvin (center temperature)
    beta: float = 2.52796285e-1        # 1/K (branch sharpness)
    gamma: float = 9.56269682e-1       # window gamma
    width_factor: float = 1.0          # scales w
    T_min_K: float = 305.0             # clamp lower bound
    T_max_K: float = 370.0             # clamp upper bound
    reversal_threshold_K: float = 0.01 # |ΔT| needed to trigger reversal

    @property
    def w_eff(self) -> float:
        return self.w * self.width_factor

    @property
    def Rm(self) -> float:
        return self.Rm0 * self.Rm_factor


@dataclass
class YuanhangCircuitParams:
    R_series_kohm: float = 12.0
    C_par_pF: float = 145.34619293
    Cth_mW_ns_per_K: float = 49.62776831
    Sth_mW_per_K: float = 0.20558726
    couple_factor: float = 0.0
    Cth_factor: float = 1.0
    noise_strength: float = 0.0
    dimension: int = 1
    T_base_K: float = 325.0

    @property
    def R_series_ohm(self) -> float:
        return self.R_series_kohm * 1e3

    @property
    def C_par_F(self) -> float:
        return self.C_par_pF * 1e-12

    @property
    def Cth_J_per_K(self) -> float:
        return self.Cth_mW_ns_per_K * 1e-12

    @property
    def S_env_W_per_K(self) -> float:
        base = self.Sth_mW_per_K * 1e-3
        return base * (1.0 - 2.0 * self.dimension * self.couple_factor)

    @property
    def S_couple_W_per_K(self) -> float:
        return self.Sth_mW_per_K * 1e-3 * self.couple_factor


@dataclass
class HysteresisState:
    delta: float
    Tr: float
    gr: float
    Tpr: float
    g_last: float
    reversed_flag: float
    T_last: float


class SimOut(TypedDict):
    time_s: List[float]
    V_node: List[float]
    I_load: List[float]
    I_vo2: List[float]
    T_C: List[float]
    R_vo2: List[float]
    g: List[float]


def _clamp_temperature(T_K: float, params: YuanhangResistParams) -> float:
    return max(params.T_min_K, min(params.T_max_K, T_K))


def _g_major(T_K: float, delta: float, params: YuanhangResistParams) -> float:
    arg = params.beta * (delta * params.w_eff / 2.0 + params.Tc_K - T_K)
    g = 0.5 + 0.5 * math.tanh(arg)
    return min(max(g, 0.0), 1.0)


def _solve_Tpr(delta: float, gr: float, Tr: float, params: YuanhangResistParams) -> float:
    gr_safe = min(max(gr, 1e-12), 1.0 - 1e-12)
    return delta * (params.w_eff / 2.0) + params.Tc_K - (1.0 / params.beta) * math.atanh(2.0 * gr_safe - 1.0) - Tr


def _init_state(T_init_K: float, params: YuanhangResistParams, start_branch: str) -> HysteresisState:
    delta = -1.0 if start_branch.lower() == "metal" else 1.0
    T0 = _clamp_temperature(T_init_K, params)
    g0 = _g_major(T0, delta, params)
    Tpr0 = _solve_Tpr(delta, g0, T0, params)
    reversed_flag = 0.0
    return HysteresisState(delta=delta, Tr=T0, gr=g0, Tpr=Tpr0, g_last=g0, reversed_flag=reversed_flag, T_last=T0)


def _g_value(T_K: float, state: HysteresisState, params: YuanhangResistParams) -> float:
    Tp = 0.0
    if state.reversed_flag:
        denom = state.Tpr
        if abs(denom) < 1e-9:
            denom = 1e-9 if denom >= 0 else -1e-9
        Tp = state.Tpr * P((T_K - state.Tr) / denom, params.gamma)
    arg = params.beta * (state.delta * params.w_eff / 2.0 + params.Tc_K - (T_K + Tp))
    g = 0.5 + 0.5 * math.tanh(arg)
    return min(max(g, 0.0), 1.0)


def _update_reversal(T_K: float, state: HysteresisState, params: YuanhangResistParams) -> float:
    T_clamped = _clamp_temperature(T_K, params)
    dT = T_clamped - state.T_last
    if abs(dT) > params.reversal_threshold_K:
        new_delta = 1.0 if dT > 0 else -1.0
        if new_delta != state.delta:
            g_at_T = _g_value(T_clamped, state, params)
            state.gr = g_at_T
            state.delta = new_delta
            state.reversed_flag = 1.0
            state.Tr = T_clamped
            state.Tpr = _solve_Tpr(state.delta, state.gr, state.Tr, params)
    state.T_last = T_clamped
    return T_clamped


def _evaluate_resistance(T_K: float, state: HysteresisState, params: YuanhangResistParams) -> tuple[float, float]:
    T_effective = _update_reversal(T_K, state, params)
    g_val = _g_value(T_effective, state, params)
    state.g_last = g_val
    exp_arg = params.Ea_over_k / max(T_effective, 1e-3)
    Rs = params.R0 * math.exp(exp_arg) * g_val
    R_total = Rs + params.Rm
    return R_total, g_val


def _simulate_single_neuristor(
    Vin: float,
    t_end: float = 60e-6,
    dt: float = 10e-9,
    resist_params: YuanhangResistParams | None = None,
    circuit_params: YuanhangCircuitParams | None = None,
    init: Dict[str, float] | None = None,
    start_branch: str = "insulator",
) -> SimOut:
    resist = resist_params or YuanhangResistParams()
    circuit = circuit_params or YuanhangCircuitParams()

    Vn = init.get("Vn") if init else 0.0
    if init and "T_K" in init:
        T_K = init["T_K"]
    elif init and "T_C" in init:
        T_K = init["T_C"] + _KELVIN_OFFSET
    else:
        T_K = circuit.T_base_K - 0.1
    state = _init_state(T_K, resist, start_branch)

    n_steps = int(t_end / dt)
    time_s: List[float] = [0.0] * (n_steps + 1)
    V_node: List[float] = [0.0] * (n_steps + 1)
    I_load: List[float] = [0.0] * (n_steps + 1)
    I_vo2: List[float] = [0.0] * (n_steps + 1)
    T_series_C: List[float] = [T_K - _KELVIN_OFFSET] * (n_steps + 1)
    R_series: List[float] = [0.0] * (n_steps + 1)
    g_series: List[float] = [state.g_last] * (n_steps + 1)

    Vin_scalar = float(Vin)
    R_series_ohm = max(circuit.R_series_ohm, _EPS)
    C_par_F = max(circuit.C_par_F, _EPS)
    inv_Cpar = 1.0 / C_par_F
    Cth = max(circuit.Cth_J_per_K, _EPS)
    Cth_factor = max(circuit.Cth_factor, _EPS)
    S_env = circuit.S_env_W_per_K
    S_couple = circuit.S_couple_W_per_K

    t = 0.0
    for i in range(n_steps + 1):
        R_vo2, g_val = _evaluate_resistance(T_K, state, resist)
        I_through_load = (Vin_scalar - Vn) / R_series_ohm
        I_through_vo2 = Vn / max(R_vo2, _EPS)
        dVn_dt = (Vin_scalar - Vn) / (R_series_ohm * C_par_F) - Vn / (R_vo2 * C_par_F)
        laplacian = 0.0  # single device, no neighbors
        P_vo2 = (Vn * Vn) / max(R_vo2, _EPS)
        noise_term = circuit.noise_strength * random.gauss(0.0, 1.0)
        dT_dt = ((P_vo2 - S_env * (T_K - circuit.T_base_K) + S_couple * laplacian) / Cth + noise_term) / Cth_factor

        time_s[i] = t
        V_node[i] = Vn
        I_load[i] = I_through_load
        I_vo2[i] = I_through_vo2
        T_series_C[i] = T_K - _KELVIN_OFFSET
        R_series[i] = R_vo2
        g_series[i] = g_val

        t += dt
        Vn += dVn_dt * dt
        T_K += dT_dt * dt

    return {
        "time_s": time_s,
        "V_node": V_node,
        "I_load": I_load,
        "I_vo2": I_vo2,
        "T_C": T_series_C,
        "R_vo2": R_series,
        "g": g_series,
    }


def run_and_save_csv(Vin: float, t_end: float = 60e-6, dt: float = 10e-9, filename: str | None = None) -> str:
    """Run a simulation and save the traces to CSV."""
    import csv
    import os

    data = _simulate_single_neuristor(Vin=Vin, t_end=t_end, dt=dt)
    if filename is None:
        filename = f"neuristor_yuanhang_Vin_{Vin:.3f}V_60us_10ns.csv".replace("/", "_")
    outpath = os.path.join(os.path.dirname(__file__), filename)

    keys = ["time_s", "V_node", "I_load", "I_vo2", "T_C", "R_vo2", "g"]
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for idx in range(len(data["time_s"])):
            writer.writerow([data[k][idx] for k in keys])
    return outpath


def _run_vin_sweep(vins, t_end, dt, start_branch: str = "insulator"):
    results = {}
    for v in vins:
        results[v] = _simulate_single_neuristor(Vin=v, t_end=t_end, dt=dt, start_branch=start_branch)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate a single VO₂ neuristor (Yuanhang Zhang model). Default sweep: 9,11,13,15,17 V."
    )
    parser.add_argument("--t_end_us", type=float, default=60.0, help="Simulation duration in microseconds")
    parser.add_argument("--dt_ns", type=float, default=10.0, help="Timestep in nanoseconds")
    parser.add_argument("--vin_list", type=str, default="", help="Comma-separated Vin list, e.g. '9,11,13,15,17'")
    parser.add_argument("--save_png", type=str, default="", help="If provided, save combined sweep plot to this path")
    parser.add_argument("--save_dir", type=str, default="", help="If provided, dump per-Vin plots into this directory")
    parser.add_argument("--no_combined", action="store_true", help="Skip the combined sweep figure")
    parser.add_argument("--start_branch", choices=["insulator", "metal"], default="insulator", help="Initial hysteresis bias")
    args = parser.parse_args()

    t_end = args.t_end_us * 1e-6
    dt = args.dt_ns * 1e-9

    if args.vin_list.strip():
        try:
            vins = [float(val.strip()) for val in args.vin_list.split(",") if val.strip()]
        except ValueError:
            raise SystemExit("Could not parse --vin_list. Use comma-separated floats like '9,11,13'.")
    else:
        vins = [9.0, 11.0, 13.0, 15.0, 17.0]

    import matplotlib.pyplot as plt
    import os

    results = _run_vin_sweep(vins, t_end=t_end, dt=dt, start_branch=args.start_branch)

    if not args.no_combined:
        plt.figure(figsize=(9, 5))
        for v in vins:
            data = results[v]
            t_us = [t * 1e6 for t in data["time_s"]]
            i_mA = [i * 1e3 for i in data["I_load"]]
            plt.plot(t_us, i_mA, label=f"Vin = {v:g} V")
        plt.xlabel("Time (µs)")
        plt.ylabel("Current through neuristor (mA)")
        plt.title("Single VO₂ Neuristor – Zhang model (Vin sweep)")
        plt.grid(True)
        plt.legend(title="Bias")
        plt.tight_layout()
        if args.save_png:
            plt.savefig(args.save_png, dpi=200)
            print(f"Saved plot to: {args.save_png}")

    for v in vins:
        data = results[v]
        t_us = [t * 1e6 for t in data["time_s"]]
        T_C = data["T_C"]
        I_mA = [i * 1e3 for i in data["I_load"]]
        V_node = data["V_node"]

        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(9, 8))
        axes[0].plot(t_us, T_C)
        axes[0].set_ylabel("Temp (°C)")
        axes[0].set_title(f"VO₂ Neuristor – Vin = {v:g} V")
        axes[0].grid(True)

        axes[1].plot(t_us, I_mA)
        axes[1].set_ylabel("Current (mA)")
        axes[1].grid(True)

        axes[2].plot(t_us, V_node)
        axes[2].set_xlabel("Time (µs)")
        axes[2].set_ylabel("Voltage (V)")
        axes[2].grid(True)

        fig.tight_layout()
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            fig.savefig(os.path.join(args.save_dir, f"Vin_{v:g}V_triplet.png"), dpi=200)

    plt.show()
