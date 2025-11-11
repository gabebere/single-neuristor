"""
Single VO₂ neuristor simulator (literal Torch port of Zhang et al.).

This script copies the VO₂ hysteresis and Circuit integrator from
`yuanhangzhang98-collective_dynamics_neuristor-217d4f0/model.py` and wraps it in
the same single-device CLI used across this repository.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, TypedDict

import numpy as np
import torch

pi = np.pi


def P(x: torch.Tensor, gamma: torch.Tensor | float):
    return 0.5 * (1 - torch.sin(gamma * x)) * (1 + torch.tanh(pi ** 2 - 2 * pi * x))


class VO2:
    def __init__(self, N, width_factor=1):
        self.N = N
        self.width_factor = width_factor
        self.w = 7.19357064e00 * width_factor
        self.Tc = 3.32805839e02
        self.beta = 2.52796285e-01
        self.R0 = 5.35882879e-03
        self.Ea = 5.22047417e03
        self.gamma = 9.56269682e-01
        self.Rm0 = 262.5
        self.Rm_factor = 4.90025335
        self.Rm = self.Rm0 * self.Rm_factor
        self.delta = torch.ones(N)
        self.reversed = torch.zeros(N)
        self.Tr = None
        self.gr = None
        self.Tpr = None
        self.T_last = None

    def initialize(self, T0):
        T = T0 * torch.ones(self.N)
        self.gr = self.g_major(T)
        self.Tr = T
        self.Tpr = self.Tpr_func()
        self.T_last = T

    def reversal(self, T):
        T = T.clamp(305, 370)
        dT = T - self.T_last
        if dT.abs().max() > 0.01:
            delta = torch.sign(dT)
            reversal_mask = (delta != self.delta) & (delta != 0)
            if reversal_mask.any():
                self.gr[reversal_mask] = self.g(T)[reversal_mask]
                self.delta[reversal_mask] = delta[reversal_mask]
                self.reversed[reversal_mask] = 1
                self.Tr[reversal_mask] = T[reversal_mask]
                self.Tpr[reversal_mask] = self.Tpr_func()[reversal_mask]
            self.T_last = T

    def Tpr_func(self):
        return self.delta * self.w / 2 + self.Tc - torch.arctanh(2 * self.gr - 1) / self.beta - self.Tr

    def g_major(self, T):
        return 0.5 + 0.5 * torch.tanh(self.beta * (self.delta * self.w / 2 + self.Tc - T))

    def g(self, T):
        Tp = self.Tpr * P((T - self.Tr) / (self.Tpr + 1e-6), self.gamma) * self.reversed
        return 0.5 + 0.5 * torch.tanh(self.beta * (self.delta * self.w / 2 + self.Tc - (T + Tp)))

    def R(self, T):
        T = T.clamp(305, 370)
        return (self.R0 * torch.exp(self.Ea / T) * self.g(T) + self.Rm) / 1000


class Circuit:
    def __init__(self, batch, N, V, R, noise_strength, Cth_factor, couple_factor, width_factor, T_base=325):
        self.batch = batch
        self.N = N
        self.d = 1
        self.V0 = V * torch.ones(self.batch, self.N)
        self.R0 = R
        self.C0 = 145.34619293
        self.R0C0 = self.R0 * self.C0
        self.Cth_factor = Cth_factor
        self.Cth = 49.62776831
        self.Sth = 0.20558726
        self.couple_factor = couple_factor
        self.S_env = self.Sth * (1 - 2 * self.d * self.couple_factor)
        self.S_couple = self.couple_factor * self.Sth
        self.noise_strength = noise_strength
        self.width_factor = width_factor
        self.T_base = T_base

        self.VO2 = VO2(batch * N, width_factor)
        self.VO2.initialize(self.T_base - 0.1)

        self.IR = None
        self.T = None
        self.R = None

        self.compiled_step = None

    def dydt(self, t, y):
        V1 = y[:, 0, :]
        T = y[:, 1, :]
        T_left = T[:, :1]
        T_right = T[:, -1:]
        T_padded = torch.cat([T_left, T, T_right], dim=1)
        laplacian = T_padded[:, :-2] - 2 * T + T_padded[:, 2:]

        R = self.VO2.R(T.reshape(-1)).reshape(self.batch, self.N)
        IR = V1 / R
        self.IR = IR
        self.T = T
        self.R = R
        QR = IR ** 2 * R
        dV1 = self.V0 / self.R0C0 - V1 / self.R0C0 - V1 / (R * self.C0)
        dT = ((QR - self.S_env * (T - self.T_base) + self.S_couple * laplacian) / self.Cth
              + self.noise_strength * torch.randn_like(T)) / self.Cth_factor
        return torch.stack([dV1, dT], dim=1)

    def step(self, t, y):
        self.VO2.reversal(y[:, 1].reshape(-1))
        return self.dydt(t, y)

    @torch.no_grad()
    def solve(self, y0, t_max, dt):
        t = 0.0
        y = y0.clone()
        t_traj: List[float] = []
        I_traj = []
        V_traj = []
        T_traj = []
        R_traj = []
        n_max = int(t_max / dt)

        self.compiled_step = self.step

        for _ in range(n_max):
            dy = self.compiled_step(t, y)
            t += dt
            y += dy * dt
            t_traj.append(t)
            I_traj.append(self.IR.detach().clone())
            V_traj.append(y[:, 0, :].detach().clone())
            T_traj.append(y[:, 1, :].detach().clone())
            R_traj.append(self.R.detach().clone())

        return y, torch.stack(I_traj, dim=-1), torch.stack(V_traj, dim=-1), torch.stack(T_traj, dim=-1), torch.stack(R_traj, dim=-1), torch.tensor(t_traj)


class Circuit2D(Circuit):
    def __init__(self, batch, Nx, Ny, V, R, noise_strength, Cth_factor, couple_factor, width_factor, T_base=325):
        N = Nx * Ny
        super().__init__(batch, N, V, R, noise_strength, Cth_factor, couple_factor, width_factor, T_base)
        self.Nx = Nx
        self.Ny = Ny
        self.d = 2
        self.S_env = self.Sth * (1 - 2 * self.d * self.couple_factor)

    def dydt(self, t, y):
        V1 = y[:, 0, :]
        T = y[:, 1, :]
        T_2D = T.view(self.batch, self.Nx, self.Ny)
        T_padded = torch.nn.functional.pad(T_2D, (1, 1, 1, 1), mode="replicate")
        laplacian = T_padded[:, :-2, 1:-1] + T_padded[:, 2:, 1:-1] + T_padded[:, 1:-1, :-2] + T_padded[:, 1:-1, 2:] - 4 * T_2D
        laplacian = laplacian.view(self.batch, self.N)

        R = self.VO2.R(T.reshape(-1)).reshape(self.batch, self.N)
        IR = V1 / R
        self.IR = IR
        self.T = T
        self.R = R
        QR = IR ** 2 * R
        dV1 = self.V0 / self.R0C0 - V1 / self.R0C0 - V1 / (R * self.C0)
        dT = ((QR - self.S_env * (T - self.T_base) + self.S_couple * laplacian) / self.Cth
              + self.noise_strength * torch.randn_like(T)) / self.Cth_factor
        return torch.stack([dV1, dT], dim=1)


class SimOut(TypedDict):
    time_s: List[float]
    V_node: List[float]
    I_load: List[float]
    I_vo2: List[float]
    T_C: List[float]
    R_vo2: List[float]


def simulate_single_neuristor_literal(
    Vin: float,
    t_end_ns: float = 60_000.0,
    dt_ns: float = 10.0,
    width_factor: float = 1.0,
    noise_strength: float = 0.0,
    couple_factor: float = 0.0,
    Cth_factor: float = 1.0,
    T_base_K: float = 325.0,
) -> SimOut:
    circuit = Circuit2D(
        batch=1,
        Nx=1,
        Ny=1,
        V=Vin,
        R=12.0,
        noise_strength=noise_strength,
        Cth_factor=Cth_factor,
        couple_factor=couple_factor,
        width_factor=width_factor,
        T_base=T_base_K,
    )

    y0 = torch.zeros((1, 2, 1), dtype=torch.float32)
    y0[:, 1, :] = T_base_K - 0.1

    _, I_traj, V_traj, T_traj, R_traj, t_traj = circuit.solve(y0, t_end_ns, dt_ns)

    times_s = [float(t) * 1e-9 for t in t_traj]
    V_node = [float(v) for v in V_traj[0, 0]]
    I_vo2 = [float(i) * 1e-3 for i in I_traj[0, 0]]
    I_load = [float((Vin - v) / circuit.R0) * 1e-3 for v in V_traj[0, 0]]
    T_C = [float(T - 273.15) for T in T_traj[0, 0]]
    R_vo2 = [float(r) * 1e3 for r in R_traj[0, 0]]

    return {
        "time_s": times_s,
        "V_node": V_node,
        "I_load": I_load,
        "I_vo2": I_vo2,
        "T_C": T_C,
        "R_vo2": R_vo2,
    }


def run_and_save_csv(Vin: float, t_end_ns: float = 60_000.0, dt_ns: float = 10.0, filename: str | None = None) -> str:
    import csv

    data = simulate_single_neuristor_literal(Vin=Vin, t_end_ns=t_end_ns, dt_ns=dt_ns)
    if filename is None:
        filename = f"neuristor_yuanhang_literal_Vin_{Vin:.3f}V_{t_end_ns:.0f}ns_{dt_ns:.0f}ns.csv".replace("/", "_")
    outpath = os.path.join(os.path.dirname(__file__), filename)

    keys = ["time_s", "V_node", "I_load", "I_vo2", "T_C", "R_vo2"]
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for idx in range(len(data["time_s"])):
            writer.writerow([data[k][idx] for k in keys])
    return outpath


def _run_vin_sweep(vins, t_end_ns, dt_ns):
    return {v: simulate_single_neuristor_literal(v, t_end_ns=t_end_ns, dt_ns=dt_ns) for v in vins}


def main():
    parser = argparse.ArgumentParser(
        description="Literal Torch simulation of a single VO₂ neuristor (Zhang et al.)."
    )
    parser.add_argument("--t_end_us", type=float, default=60.0, help="Duration in microseconds")
    parser.add_argument("--dt_ns", type=float, default=10.0, help="Timestep in nanoseconds")
    parser.add_argument("--vin_list", type=str, default="", help="Comma-separated Vin list")
    parser.add_argument("--save_png", type=str, default="", help="Save combined plot here")
    parser.add_argument("--save_dir", type=str, default="", help="Optional directory for per-Vin plots")
    parser.add_argument("--no_combined", action="store_true", help="Skip combined sweep plot")
    args = parser.parse_args()

    t_end_ns = args.t_end_us * 1e3
    dt_ns = args.dt_ns

    if args.vin_list.strip():
        vins = [float(v.strip()) for v in args.vin_list.split(",") if v.strip()]
    else:
        vins = [9.0, 11.0, 13.0, 15.0, 17.0]

    import matplotlib.pyplot as plt

    results = _run_vin_sweep(vins, t_end_ns=t_end_ns, dt_ns=dt_ns)

    if not args.no_combined:
        plt.figure(figsize=(9, 5))
        for v in vins:
            data = results[v]
            t_us = [t * 1e6 for t in data["time_s"]]
            i_mA = [i * 1e3 for i in data["I_load"]]
            plt.plot(t_us, i_mA, label=f"Vin = {v:g} V")
        plt.xlabel("Time (µs)")
        plt.ylabel("Current through neuristor (mA)")
        plt.title("VO₂ Neuristor – Literal Torch model")
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
        axes[0].set_title(f"VO₂ Neuristor – Vin = {v:g} V (Torch)")
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
            fig.savefig(os.path.join(args.save_dir, f"Vin_{v:g}V_torch_triplet.png"), dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
