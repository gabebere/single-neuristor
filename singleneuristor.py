"""
Single VO₂ neuristor simulator (paper-faithful Zhang–Qiu–Di Ventra model).

This script implements the electrical/thermal ODEs and the hysteretic R(T) model used for
single-device VO₂ oscillators. It is **vectorized, NumPy-only**, and produces the same
currents/voltages/temperatures as the original code with straightforward parameters.

Model summary
-------------
Electrical ODE (node voltage V across VO₂, series load R_load, capacitor C):
  C dV/dt = Vin/R_load − V/R_VO2 − V/R_load

Thermal ODE (device temperature T):
  C_th dT/dt = V²/R_VO2 − S_e (T − T_base) + S_c ∇²T + noise

Hysteresis / resistance law (Almeida-form, as used in the paper):
  R(T) = R0 * exp(Ea_over_k / T) * g(T) + Rm
  g(T) = 0.5 + 0.5 * tanh[ β( δ*w/2 + Tc − (T + T_pr * P((T − T_r)/T_pr)) ) ]
  δ = sign(dT/dt) selects heating (δ=+1) or cooling (δ=−1) major branch
  (T_r, g_r) are stored at last reversal to define the minor-loop proximity shift T_pr.

Key conventions
---------------
• Temperatures are **Kelvin** throughout the simulation and in outputs/plots.
• "Series current" I_load = (Vin − V)/R_load.  "Device current" I_vo2 = V/R_VO2.
• CSV/plots expose: time_s, V_node (V across VO₂), I_load, I_vo2, T_K, R_vo2, g.
• Per‑Vin figures show: T_K, I_vo2, V_VO2 + V_Rload. Combined figure plots I_vo2 for all V.
"""
from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, TypedDict

import numpy as np


_EPS: float = 1e-12
_PI: float = math.pi
_KELVIN_OFFSET: float = 273.15


def P(x: float, gamma: float) -> float:
    """Smooth proximity window P(x; γ) controlling minor-loop influence.
    Matches the paper’s form: 0.5*(1−sin(γx))*(1+tanh(π²−2πx))."""
    return 0.5 * (1.0 - math.sin(gamma * x)) * (1.0 + math.tanh(_PI * _PI - 2.0 * _PI * x))


@dataclass
class YuanhangResistParams:
    """Parameters for the VO₂ resistance model R(T) and hysteresis shape.
    All temperatures are in Kelvin; beta has units 1/K; gamma is unitless."""
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
    """Electrical and thermal circuit parameters for a single neuristor.
    R_series is the load resistor; C_par the node capacitance; thermal parameters use SI."""
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
class SimOut(TypedDict):
    """Structured output of a simulation run (single device or lattice average/first)."""
    time_s: List[float]
    V_node: List[float] | List[List[float]]
    I_load: List[float] | List[List[float]]
    I_vo2: List[float] | List[List[float]]
    T_K: List[float] | List[List[float]]
    R_vo2: List[float] | List[List[float]]
    g: List[float] | List[List[float]]
    grid_shape: Tuple[int, int]


# ---------------------------------------------------------------------------
# Hysteresis helper (vectorized, Torch-free)
# ---------------------------------------------------------------------------


def _broadcast_array(value: float | Sequence[float], size: int, name: str) -> np.ndarray:
    """Return a 1D float array of length `size`, broadcasting scalars or validating sequences."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size != size:
            raise ValueError(f"{name} must have length {size}, got {arr.size}")
        return arr.copy()
    return np.full(size, float(value), dtype=float)


def _clamp_temperature_array(T: np.ndarray, params: YuanhangResistParams) -> np.ndarray:
    """Clamp device temperature to [T_min_K, T_max_K] to stabilize the resistance law."""
    return np.clip(T, params.T_min_K, params.T_max_K)


def _P_vec(x: np.ndarray, gamma: float) -> np.ndarray:
    """Vectorized version of P(x; γ) used within the hysteresis implementation."""
    return 0.5 * (1.0 - np.sin(gamma * x)) * (1.0 + np.tanh(math.pi * math.pi - 2.0 * math.pi * x))


class HysteresisArray:
    """Evaluate g(T) with Almeida hysteresis (major branches + minor-loop proximity).
    Keeps per-device state: current branch sign δ, last reversal (T_r, g_r), and T_pr."""

    def __init__(self, params: YuanhangResistParams, size: int, start_branch: str = "insulator") -> None:
        self.params = params
        branch = start_branch.lower()
        delta0 = 1.0 if branch == "insulator" else -1.0
        self.delta = np.full(size, delta0, dtype=float)
        self.reversed = np.zeros(size, dtype=float)
        self.Tr = np.zeros(size, dtype=float)
        self.gr = np.zeros(size, dtype=float)
        self.Tpr = np.zeros(size, dtype=float)
        self.g_last = np.zeros(size, dtype=float)
        self.T_last = np.zeros(size, dtype=float)

    def initialize(self, T_init_K: np.ndarray) -> None:
        """Initialize per-device hysteresis state at T_init_K (clamped)."""
        params = self.params
        T0 = _clamp_temperature_array(T_init_K, params)
        self.Tr = T0.copy()
        self.T_last = T0.copy()
        self.gr = self._g_major(T0, self.delta)
        self.Tpr = self._solve_Tpr(self.delta, self.gr, self.Tr)
        self.g_last = self.gr.copy()
        self.reversed.fill(0.0)

    def _g_major(self, T: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Major-branch fraction g for a given T and branch sign δ (no minor-loop shift)."""
        params = self.params
        arg = params.beta * (delta * params.w_eff / 2.0 + params.Tc_K - T)
        return np.clip(0.5 + 0.5 * np.tanh(arg), 0.0, 1.0)

    def _solve_Tpr(self, delta: np.ndarray, gr: np.ndarray, Tr: np.ndarray) -> np.ndarray:
        """Compute T_pr at reversal from (δ, g_r, T_r) per the paper’s formula."""
        params = self.params
        gr_safe = np.clip(gr, 1e-12, 1.0 - 1e-12)
        return delta * (params.w_eff / 2.0) + params.Tc_K - (1.0 / params.beta) * np.arctanh(2.0 * gr_safe - 1.0) - Tr

    def g(self, T: np.ndarray) -> np.ndarray:
        """Evaluate g(T) with the current reversal window: g_major(T + T_p), where T_p=T_pr*P(...)."""
        params = self.params
        if np.any(self.reversed):
            # Avoid zero denominator at reversal pivot; when Tpr≈0 use a signed epsilon,
            # choosing +eps when Tpr>=0 and -eps otherwise. This prevents division-by-zero
            # and spurious huge Tp during branch transitions.
            eps = 1e-9
            denom = np.where(
                np.abs(self.Tpr) < eps,
                np.where(self.Tpr >= 0.0, eps, -eps),
                self.Tpr,
            )
            Tp = self.Tpr * _P_vec((T - self.Tr) / denom, params.gamma) * self.reversed
        else:
            Tp = 0.0
        arg = params.beta * (self.delta * params.w_eff / 2.0 + params.Tc_K - (T + Tp))
        return np.clip(0.5 + 0.5 * np.tanh(arg), 0.0, 1.0)

    def evaluate(self, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (R_vo2, g) at temperature T. Updates reversal state before evaluating g."""
        params = self.params
        T_clamped = _clamp_temperature_array(T, params)
        self._update_reversal(T_clamped)
        g_val = self.g(T_clamped)
        self.g_last = g_val.copy()
        exp_arg = params.Ea_over_k / np.maximum(T_clamped, 1e-9)
        Rs = params.R0 * np.exp(exp_arg) * g_val
        return Rs + params.Rm, g_val

    def _update_reversal(self, T_clamped: np.ndarray) -> None:
        """Detect heating/cooling changes (δ flip) when |ΔT| exceeds threshold; update minor-loop state."""
        params = self.params
        dT = T_clamped - self.T_last
        mask = np.abs(dT) > params.reversal_threshold_K
        if not np.any(mask):
            self.T_last = T_clamped
            return
        # delta = +1 for heating (dT>0), -1 for cooling (dT<0); matches major-branch bias in _g_major
        delta_new = np.sign(dT)
        delta_new[delta_new == 0.0] = self.delta[delta_new == 0.0]
        reversal_mask = mask & (delta_new != self.delta)
        if np.any(reversal_mask):
            g_at_T = self.g(T_clamped)
            self.gr[reversal_mask] = g_at_T[reversal_mask]
            self.delta[reversal_mask] = delta_new[reversal_mask]
            self.reversed[reversal_mask] = 1.0
            self.Tr[reversal_mask] = T_clamped[reversal_mask]
            self.Tpr[reversal_mask] = self._solve_Tpr(
                self.delta[reversal_mask], self.gr[reversal_mask], self.Tr[reversal_mask]
            )
        self.T_last = T_clamped


def _compute_laplacian(T: np.ndarray, Nx: int, Ny: int) -> np.ndarray:
    """Neumann (edge-replicated) discrete Laplacian for 1D/2D device arrays."""
    if Nx * Ny == 1:
        return np.zeros_like(T)
    if Ny == 1:
        arr = T.reshape(Nx, 1)
        padded = np.concatenate([arr[:1], arr, arr[-1:]], axis=0)
        lap = padded[:-2] - 2.0 * arr + padded[2:]
        return lap.reshape(-1)
    arr = T.reshape(Nx, Ny)
    padded = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * arr
    )
    return lap.reshape(-1)


class YuanhangArraySimulator:
    """Array simulator for VO₂ neuristors (NumPy, no Torch). Handles ODE integration and logging."""

    def __init__(
        self,
        Vin: float | Sequence[float],
        resist_params: YuanhangResistParams | None,
        circuit_params: YuanhangCircuitParams | None,
        lattice_shape: Tuple[int, int],
        start_branch: str,
        init: Dict[str, float] | None,
    ) -> None:
        """Build a simulator with provided Vin and parameter structs.
        Args map 1:1 to dataclasses; lattice_shape defines Nx×Ny devices; start_branch picks δ=+1/−1."""
        self.resist = resist_params or YuanhangResistParams()
        self.circuit = circuit_params or YuanhangCircuitParams()
        self.Nx, self.Ny = lattice_shape
        self.N = self.Nx * self.Ny
        if self.N <= 0:
            raise ValueError("Lattice must contain at least one device")
        self.dimension = 1 if (self.Nx == 1 or self.Ny == 1) else 2
        self.V_bias = _broadcast_array(Vin, self.N, "Vin")
        init = init or {}
        self.Vn = _broadcast_array(init.get("Vn", 0.0), self.N, "Vn")
        if "T_K" in init:
            T_init = _broadcast_array(init["T_K"], self.N, "T_K")
        elif "T_C" in init:
            T_init = _broadcast_array(init["T_C"] + _KELVIN_OFFSET, self.N, "T_C")
        else:
            T_init = np.full(self.N, self.circuit.T_base_K - 0.1, dtype=float)
        self.T_K = T_init
        self.hysteresis = HysteresisArray(self.resist, self.N, start_branch=start_branch)
        self.hysteresis.initialize(T_init)
        self.Cth_factor = _broadcast_array(self.circuit.Cth_factor, self.N, "Cth_factor")
        self.noise_strength = self.circuit.noise_strength
        self.S_env = self.circuit.Sth_mW_per_K * 1e-3 * (1.0 - 2.0 * self.dimension * self.circuit.couple_factor)
        self.S_couple = self.circuit.Sth_mW_per_K * 1e-3 * self.circuit.couple_factor
        self.R_series_ohm = max(self.circuit.R_series_ohm, _EPS)
        self.C_par_F = max(self.circuit.C_par_F, _EPS)
        self.C_th_J_per_K = max(self.circuit.Cth_J_per_K, _EPS)
        self.T_base = self.circuit.T_base_K

    def set_inputs(
        self,
        V: float | Sequence[float] | None = None,
        Cth_factor: float | Sequence[float] | None = None,
    ) -> None:
        """Update Vin and/or Cth scaling at runtime (broadcasting scalars or sequences)."""
        if V is not None:
            self.V_bias = _broadcast_array(V, self.N, "Vin")
        if Cth_factor is not None:
            self.Cth_factor = _broadcast_array(Cth_factor, self.N, "Cth_factor")

    def run(self, t_end: float, dt: float, noise_seed: int | None = None) -> SimOut:
        """Integrate the ODEs for t∈[0, t_end] with step dt (Euler). Returns SimOut with traces."""
        steps = int(t_end / dt)
        if steps <= 0:
            raise ValueError("t_end/dt must yield at least one timestep")
        time_s = [0.0] * (steps + 1)
        history_V = np.zeros((steps + 1, self.N))
        history_Iload = np.zeros_like(history_V)
        history_Ivo2 = np.zeros_like(history_V)
        history_T = np.zeros_like(history_V)
        history_R = np.zeros_like(history_V)
        history_g = np.zeros_like(history_V)

        history_V[0, :] = self.Vn
        R0, g0 = self.hysteresis.evaluate(self.T_K)
        history_Iload[0, :] = (self.V_bias - self.Vn) / self.R_series_ohm
        history_Ivo2[0, :] = self.Vn / np.maximum(R0, _EPS)
        history_T[0, :] = self.T_K
        history_R[0, :] = R0
        history_g[0, :] = g0

        rng = np.random.default_rng(noise_seed)
        t = 0.0
        for idx in range(1, steps + 1):
            R_vo2, g_val = self.hysteresis.evaluate(self.T_K)
            I_load = (self.V_bias - self.Vn) / self.R_series_ohm
            I_vo2 = self.Vn / np.maximum(R_vo2, _EPS)
            dV_dt = (self.V_bias - self.Vn) / (self.R_series_ohm * self.C_par_F) - self.Vn / (R_vo2 * self.C_par_F)
            laplacian = _compute_laplacian(self.T_K, self.Nx, self.Ny)
            P_vo2 = (self.Vn * self.Vn) / np.maximum(R_vo2, _EPS)
            if self.noise_strength > 0.0:
                noise = self.noise_strength * rng.standard_normal(self.N)
            else:
                noise = 0.0
            dT_dt = (
                (P_vo2 - self.S_env * (self.T_K - self.T_base) + self.S_couple * laplacian) / self.C_th_J_per_K
                + noise
            ) / self.Cth_factor

            self.Vn = self.Vn + dV_dt * dt
            self.T_K = self.T_K + dT_dt * dt
            t += dt

            time_s[idx] = t
            history_V[idx, :] = self.Vn
            history_Iload[idx, :] = I_load
            history_Ivo2[idx, :] = I_vo2
            history_T[idx, :] = self.T_K
            history_R[idx, :] = R_vo2
            history_g[idx, :] = g_val

        return {
            "time_s": time_s,
            "V_node": _series_to_output(history_V),
            "I_load": _series_to_output(history_Iload),
            "I_vo2": _series_to_output(history_Ivo2),
            "T_K": _series_to_output(history_T),
            "R_vo2": _series_to_output(history_R),
            "g": _series_to_output(history_g),
            "grid_shape": (self.Nx, self.Ny),
        }


def _series_to_output(matrix: np.ndarray) -> List[float] | List[List[float]]:
    """Convert (steps×N) arrays to a list or list-of-lists compatible with SimOut."""
    if matrix.shape[1] == 1:
        return matrix[:, 0].tolist()
    return [matrix[:, idx].tolist() for idx in range(matrix.shape[1])]


def _series_mean(series: List[float] | List[List[float]]) -> List[float]:
    """Average across devices for multi-device lattices; pass-through for single-device."""
    if not series:
        return []
    if isinstance(series[0], list):
        length = len(series[0])
        return [sum(device[i] for device in series) / len(series) for i in range(length)]
    return series  # type: ignore[return-value]


def _series_first(series: List[float] | List[List[float]]) -> List[float]:
    """Return first device’s series for multi-device lattices; pass-through for single-device."""
    if not series:
        return []
    if isinstance(series[0], list):
        return series[0]
    return series  # type: ignore[return-value]


def _simulate_single_neuristor(
    Vin: float | Sequence[float],
    t_end: float = 60e-6,
    dt: float = 10e-9,
    resist_params: YuanhangResistParams | None = None,
    circuit_params: YuanhangCircuitParams | None = None,
    init: Dict[str, float] | None = None,
    start_branch: str = "insulator",
    lattice_shape: Tuple[int, int] = (1, 1),
    noise_seed: int | None = None,
) -> SimOut:
    """Convenience wrapper. Create a simulator and run it once; returns SimOut."""
    simulator = YuanhangArraySimulator(
        Vin=Vin,
        resist_params=resist_params,
        circuit_params=circuit_params,
        lattice_shape=lattice_shape,
        start_branch=start_branch,
        init=init,
    )
    return simulator.run(t_end=t_end, dt=dt, noise_seed=noise_seed)


def run_and_save_csv(
    Vin: float | Sequence[float],
    t_end: float = 60e-6,
    dt: float = 10e-9,
    filename: str | None = None,
    lattice_shape: Tuple[int, int] = (1, 1),
) -> str:
    """Run a single-device simulation and write traces to CSV.
    Columns: time_s, V_node, I_load, I_vo2, T_K, R_vo2, g (one row per step)."""
    import csv
    import os

    if lattice_shape != (1, 1):
        raise ValueError("CSV export currently supports only single-device simulations (grid_shape == (1, 1)).")
    if not isinstance(Vin, (int, float)):
        raise ValueError("Scalar Vin is required for CSV export.")
    data = _simulate_single_neuristor(Vin=Vin, t_end=t_end, dt=dt, lattice_shape=lattice_shape)
    if filename is None:
        filename = f"neuristor_yuanhang_Vin_{Vin:.3f}V_60us_10ns.csv".replace("/", "_")
    outpath = os.path.join(os.path.dirname(__file__), filename)

    keys = ["time_s", "V_node", "I_load", "I_vo2", "T_K", "R_vo2", "g"]
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for idx in range(len(data["time_s"])):
            writer.writerow([data[k][idx] for k in keys])
    return outpath


def _run_vin_sweep(
    vins,
    t_end,
    dt,
    start_branch: str = "insulator",
    lattice_shape: Tuple[int, int] = (1, 1),
    noise_seed: int | None = None,
    resist_params: YuanhangResistParams | None = None,
    circuit_params: YuanhangCircuitParams | None = None,
):
    """Run a Vin sweep and return a dict {Vin: SimOut}. Pass-through of parameter objects."""
    results = {}
    for v in vins:
        results[v] = _simulate_single_neuristor(
            Vin=v,
            t_end=t_end,
            dt=dt,
            resist_params=resist_params,
            circuit_params=circuit_params,
            start_branch=start_branch,
            lattice_shape=lattice_shape,
            noise_seed=None if noise_seed is None else noise_seed + int(v * 10),
        )
    return results


if __name__ == "__main__":
    # CLI: use --paper to lock paper constants; --vin_list "9,11,13,15,17"; --save_dir to dump per‑Vin PNGs.
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate a single VO₂ neuristor (Yuanhang Zhang model). Default sweep: 9,11,13,15,17 V."
    )
    parser.add_argument("--t_end_us", type=float, default=85.0, help="Simulation duration in microseconds")
    parser.add_argument("--dt_ns", type=float, default=10.0, help="Timestep in nanoseconds")
    parser.add_argument("--vin_list", type=str, default="", help="Comma-separated Vin list, e.g. '9,11,13,15,17'")
    parser.add_argument("--save_png", type=str, default="", help="If provided, save combined sweep plot to this path")
    parser.add_argument("--save_dir", type=str, default="", help="If provided, dump per-Vin plots into this directory")
    parser.add_argument("--no_combined", action="store_true", help="Skip the combined sweep figure")
    parser.add_argument("--start_branch", choices=["insulator", "metal"], default="insulator", help="Initial hysteresis bias")
    parser.add_argument("--nx", type=int, default=1, help="Number of devices along x (>=1)")
    parser.add_argument("--ny", type=int, default=1, help="Number of devices along y (>=1)")
    parser.add_argument("--noise_seed", type=int, default=None, help="Seed for the thermal noise RNG")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Use paper-exact preset (Zhang–Qiu–Di Ventra): additive mixing, Table-1 constants, zero coupling, plot I_load with 0–5 mA scale on per-V plots.",
    )
    args = parser.parse_args()

    t_end = args.t_end_us * 1e-6
    dt = args.dt_ns * 1e-9
    lattice_shape = (max(1, args.nx), max(1, args.ny))

    if args.vin_list.strip():
        try:
            vins = [float(val.strip()) for val in args.vin_list.split(",") if val.strip()]
        except ValueError:
            raise SystemExit("Could not parse --vin_list. Use comma-separated floats like '9,11,13'.")
    else:
        vins = [9.0, 11.0, 13.0, 15.0, 17.0]

    import matplotlib.pyplot as plt
    import os

    resist_params = YuanhangResistParams()
    circuit_params = YuanhangCircuitParams()
    if args.paper:
        # Paper-exact constants (matching Methods/Table 1; values already match defaults)
        resist_params.R0 = 5.35882879e-3
        resist_params.Ea_over_k = 5.22047417e3
        resist_params.Rm0 = 262.5
        resist_params.Rm_factor = 4.90025335  # Rm ≈ 1286 Ω
        resist_params.w = 7.19357064
        resist_params.Tc_K = 3.32805839e2
        resist_params.beta = 2.52796285e-1
        resist_params.gamma = 9.56269682e-1
        resist_params.width_factor = 1.0
        resist_params.T_min_K = 305.0
        resist_params.T_max_K = 370.0
        resist_params.reversal_threshold_K = 0.01

        circuit_params.R_series_kohm = 12.0
        circuit_params.C_par_pF = 145.34619293
        circuit_params.Cth_mW_ns_per_K = 49.62776831
        circuit_params.Sth_mW_per_K = 0.20558726
        circuit_params.couple_factor = 0.0
        circuit_params.Cth_factor = 1.0
        circuit_params.noise_strength = 0.0
        circuit_params.dimension = 1
        circuit_params.T_base_K = 325.0

    results = _run_vin_sweep(
        vins,
        t_end=t_end,
        dt=dt,
        start_branch=args.start_branch,
        lattice_shape=lattice_shape,
        noise_seed=args.noise_seed,
        resist_params=resist_params,
        circuit_params=circuit_params,
    )

    multi_device = lattice_shape != (1, 1)
    if multi_device:
        print(
            "Multi-device lattice detected: combined plots show average currents; per-V plots show device (0,0)."
        )

    if not args.no_combined:
        plt.figure(figsize=(9, 5))
        for v in vins:
            data = results[v]
            t_us = [t * 1e6 for t in data["time_s"]]
            # I_vo2 (device current)
            ivo2_series = _series_mean(data["I_vo2"])
            ivo2_mA = [i * 1e3 for i in ivo2_series]
            # Show only 25–85 µs
            mask = [(t >= 25.0) and (t <= 85.0) for t in t_us]
            t_us_plot = [t for t, m in zip(t_us, mask) if m]
            ivo2_mA_plot = [i for i, m in zip(ivo2_mA, mask) if m]
            # Plot only I_vo2
            plt.plot(t_us_plot, ivo2_mA_plot, label=f"Vin = {v:g} V")
        plt.xlabel("Time (µs)")
        plt.ylabel("Device current I_vo2 (mA)")
        plt.title("Single VO₂ Neuristor – Zhang model (Vin sweep)")
        plt.grid(True)
        plt.legend(title="Bias")
        if args.paper:
            plt.ylim(0.0, 5.0)
        plt.tight_layout()
        if args.save_png:
            plt.savefig(args.save_png, dpi=200)
            print(f"Saved plot to: {args.save_png}")

    for v in vins:
        data = results[v]
        t_us = [t * 1e6 for t in data["time_s"]]
        T_K = _series_first(data["T_K"])
        I_vo2_mA = [i * 1e3 for i in _series_first(data["I_vo2"])]
        V_node = _series_first(data["V_node"])

        # Show only 25–85 µs
        mask = [(t >= 25.0) and (t <= 85.0) for t in t_us]
        t_us_plot = [t for t, m in zip(t_us, mask) if m]
        T_K_plot = [x for x, m in zip(T_K, mask) if m]
        I_vo2_mA_plot = [x for x, m in zip(I_vo2_mA, mask) if m]
        V_node_plot = [x for x, m in zip(V_node, mask) if m]

        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(9, 8))
        axes[0].plot(t_us_plot, T_K_plot)
        axes[0].set_ylabel("Temp (K)")
        axes[0].set_title(f"VO₂ Neuristor – Vin = {v:g} V")
        axes[0].grid(True)

        axes[1].plot(t_us_plot, I_vo2_mA_plot)
        axes[1].set_ylabel("Device current I_vo2 (mA)")
        if args.paper:
            axes[1].set_ylim(0.0, 5.0)
        axes[1].grid(True)

        # Plot VO2 voltage (solid) and series resistor voltage (dashed)
        Vin_value = v  # since each run uses a single Vin
        V_load_plot = [Vin_value - vn for vn in V_node_plot]

        axes[2].plot(t_us_plot, V_node_plot, label="V_VO2 (V)")
        axes[2].plot(t_us_plot, V_load_plot, linestyle="--", label="V_Rload (V)")
        axes[2].set_xlabel("Time (µs)")
        axes[2].set_ylabel("Voltage (V)")
        axes[2].grid(True)
        axes[2].legend()

        fig.tight_layout()
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            fig.savefig(os.path.join(args.save_dir, f"Vin_{v:g}V_triplet.png"), dpi=200)

    plt.show()
