"""
Core VO₂ neuristor simulator (Yuanhang Zhang–Qiu–Di Ventra model) and shared helpers.

This module is the single source of truth for:
- parameter dataclasses
- hysteretic resistance law g(T)
- ODE integration (electrical + thermal)
- helper utilities (series selectors, spike/oscillation detection, sweeps)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, TypedDict, TypeVar

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for reference-faithful float32 hysteresis
    torch = None  # type: ignore[assignment]


_EPS: float = 1e-12
_PI: float = math.pi
_KELVIN_OFFSET: float = 273.15
_NS_PER_S: float = 1e9
_SIM_DTYPE = np.float32
_TORCH_HYSTERESIS_AVAILABLE = torch is not None
T = TypeVar("T")


class SimulationCancelled(Exception):
    """Raised when a running simulation is cancelled."""



def P(x: float, gamma: float) -> float:
    """Smooth proximity window P(x; γ) controlling minor-loop influence."""
    return 0.5 * (1.0 - math.sin(gamma * x)) * (1.0 + math.tanh(_PI * _PI - 2.0 * _PI * x))


@dataclass
class YuanhangResistParams:
    """Parameters for the VO₂ resistance model R(T) and hysteresis shape."""

    R0: float = 5.35882879e-3  # Ohm
    Ea_over_k: float = 5.22047417e3  # Kelvin (Ea/kB)
    Rm0: float = 262.5  # Ohm
    Rm_factor: float = 4.90025335  # dimensionless
    w: float = 7.19357064  # Kelvin (loop width)
    Tc_K: float = 3.32805839e2  # Kelvin (center temperature)
    beta: float = 2.52796285e-1  # 1/K (branch sharpness)
    gamma: float = 9.56269682e-1  # window gamma
    width_factor: float = 1.0  # scales w
    T_min_K: float = 305.0  # clamp lower bound
    T_max_K: float = 370.0  # clamp upper bound
    reversal_threshold_K: float = 0.01  # |ΔT| needed to trigger reversal

    @property
    def w_eff(self) -> float:
        return self.w * self.width_factor

    @property
    def Rm(self) -> float:
        return self.Rm0 * self.Rm_factor


@dataclass
class YuanhangCircuitParams:
    """Electrical and thermal circuit parameters for a single neuristor."""

    R_series_kohm: float = 14.0
    C_par_pF: float = 200.0
    Cth_mW_ns_per_K: float = 49.62776831
    Sth_mW_per_K: float = 0.20558726
    couple_factor: float = 0.0
    Cth_factor: float = 1.0
    # Legacy paper noise term in K/ns (converted internally to K/s for SI integration).
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

    @property
    def noise_strength_K_per_s(self) -> float:
        # Original model integrates in ns and adds `noise_strength * randn` to dT/dt.
        # Convert that legacy term into SI seconds so the same numeric values behave identically.
        return self.noise_strength * _NS_PER_S


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
# Hysteresis helper (vectorized, float32-faithful)
# ---------------------------------------------------------------------------


def _broadcast_array(value: float | Sequence[float], size: int, name: str) -> np.ndarray:
    """Return a 1D float array of length `size`, broadcasting scalars or validating sequences."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=_SIM_DTYPE).reshape(-1)
        if arr.size != size:
            raise ValueError(f"{name} must have length {size}, got {arr.size}")
        return arr.copy()
    return np.full(size, float(value), dtype=_SIM_DTYPE)


def _clamp_temperature_array(T: np.ndarray, params: YuanhangResistParams) -> np.ndarray:
    """Clamp device temperature to [T_min_K, T_max_K] to stabilize the resistance law."""
    T_arr = np.asarray(T, dtype=_SIM_DTYPE)
    return np.clip(T_arr, params.T_min_K, params.T_max_K).astype(_SIM_DTYPE, copy=False)


def _torch_tensor(x: np.ndarray) -> "torch.Tensor":
    """Convert a float32 NumPy array into a CPU torch tensor without changing values."""
    arr = np.asarray(x, dtype=_SIM_DTYPE)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    if torch is None:  # pragma: no cover - guarded by _TORCH_HYSTERESIS_AVAILABLE
        raise RuntimeError("torch is not available")
    return torch.from_numpy(arr)


def _P_vec(x: np.ndarray, gamma: float) -> np.ndarray:
    """Vectorized version of P(x; γ) used within the hysteresis implementation."""
    x_arr = np.asarray(x, dtype=_SIM_DTYPE)
    if _TORCH_HYSTERESIS_AVAILABLE:
        x_t = _torch_tensor(x_arr)
        out_t = 0.5 * (1.0 - torch.sin(gamma * x_t)) * (1.0 + torch.tanh(_PI * _PI - 2.0 * _PI * x_t))
        return np.asarray(out_t.cpu().numpy(), dtype=_SIM_DTYPE)
    out = 0.5 * (1.0 - np.sin(gamma * x)) * (1.0 + np.tanh(math.pi * math.pi - 2.0 * math.pi * x))
    return np.asarray(out, dtype=_SIM_DTYPE)


class HysteresisArray:
    """Evaluate g(T) with Almeida hysteresis (major branches + minor-loop proximity)."""

    def __init__(self, params: YuanhangResistParams, size: int, start_branch: str = "insulator") -> None:
        self.params = params
        branch = start_branch.lower()
        delta0 = 1.0 if branch == "insulator" else -1.0
        self.delta = np.full(size, delta0, dtype=_SIM_DTYPE)
        self.reversed = np.zeros(size, dtype=_SIM_DTYPE)
        self.Tr = np.zeros(size, dtype=_SIM_DTYPE)
        self.gr = np.zeros(size, dtype=_SIM_DTYPE)
        self.Tpr = np.zeros(size, dtype=_SIM_DTYPE)
        self.g_last = np.zeros(size, dtype=_SIM_DTYPE)
        self.T_last = np.zeros(size, dtype=_SIM_DTYPE)

    def initialize(self, T_init_K: np.ndarray) -> None:
        """Initialize per-device hysteresis state at T_init_K (clamped)."""
        params = self.params
        T0 = _clamp_temperature_array(np.asarray(T_init_K, dtype=_SIM_DTYPE), params)
        self.Tr = T0.copy()
        self.T_last = T0.copy()
        self.gr = self._g_major(T0, self.delta)
        self.Tpr = self._solve_Tpr(self.delta, self.gr, self.Tr)
        self.g_last = self.gr.copy()
        self.reversed.fill(_SIM_DTYPE(0.0))

    def _g_major(self, T: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Major-branch fraction g for a given T and branch sign δ (no minor-loop shift)."""
        params = self.params
        T_arr = np.asarray(T, dtype=_SIM_DTYPE)
        delta_arr = np.asarray(delta, dtype=_SIM_DTYPE)
        if _TORCH_HYSTERESIS_AVAILABLE:
            T_t = _torch_tensor(T_arr)
            delta_t = _torch_tensor(delta_arr)
            out_t = 0.5 + 0.5 * torch.tanh(params.beta * (delta_t * params.w_eff / 2.0 + params.Tc_K - T_t))
            return np.asarray(out_t.cpu().numpy(), dtype=_SIM_DTYPE)
        arg = params.beta * (delta_arr * params.w_eff / 2.0 + params.Tc_K - T_arr)
        return np.asarray(0.5 + 0.5 * np.tanh(arg), dtype=_SIM_DTYPE)

    def _solve_Tpr(self, delta: np.ndarray, gr: np.ndarray, Tr: np.ndarray) -> np.ndarray:
        """Compute T_pr at reversal from (δ, g_r, T_r) per the paper’s formula."""
        params = self.params
        delta_arr = np.asarray(delta, dtype=_SIM_DTYPE)
        gr_arr = np.asarray(gr, dtype=_SIM_DTYPE)
        Tr_arr = np.asarray(Tr, dtype=_SIM_DTYPE)
        if _TORCH_HYSTERESIS_AVAILABLE:
            delta_t = _torch_tensor(delta_arr)
            gr_t = _torch_tensor(gr_arr)
            Tr_t = _torch_tensor(Tr_arr)
            out_t = delta_t * (params.w_eff / 2.0) + params.Tc_K - torch.arctanh(2.0 * gr_t - 1.0) / params.beta - Tr_t
            return np.asarray(out_t.cpu().numpy(), dtype=_SIM_DTYPE)
        out = delta_arr * (params.w_eff / 2.0) + params.Tc_K - (1.0 / params.beta) * np.arctanh(2.0 * gr_arr - 1.0) - Tr_arr
        return np.asarray(out, dtype=_SIM_DTYPE)

    def g(self, T: np.ndarray) -> np.ndarray:
        """Evaluate g(T) with the current reversal window: g_major(T + T_p), where T_p=T_pr*P(...)."""
        params = self.params
        T_arr = np.asarray(T, dtype=_SIM_DTYPE)
        if np.any(self.reversed):
            Tp = self.Tpr * _P_vec((T_arr - self.Tr) / (self.Tpr + 1e-6), params.gamma) * self.reversed
        else:
            Tp = _SIM_DTYPE(0.0)
        if _TORCH_HYSTERESIS_AVAILABLE:
            T_t = _torch_tensor(T_arr)
            Tp_t = _torch_tensor(np.asarray(Tp, dtype=_SIM_DTYPE))
            delta_t = _torch_tensor(self.delta)
            out_t = 0.5 + 0.5 * torch.tanh(params.beta * (delta_t * params.w_eff / 2.0 + params.Tc_K - (T_t + Tp_t)))
            return np.asarray(out_t.cpu().numpy(), dtype=_SIM_DTYPE)
        arg = params.beta * (self.delta * params.w_eff / 2.0 + params.Tc_K - (T_arr + Tp))
        return np.asarray(0.5 + 0.5 * np.tanh(arg), dtype=_SIM_DTYPE)

    def evaluate(self, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (R_vo2, g) at temperature T. Updates reversal state before evaluating g."""
        params = self.params
        T_clamped = _clamp_temperature_array(np.asarray(T, dtype=_SIM_DTYPE), params)
        self._update_reversal(T_clamped)
        g_val = self.g(T_clamped)
        self.g_last = g_val.copy()
        if _TORCH_HYSTERESIS_AVAILABLE:
            T_t = _torch_tensor(T_clamped)
            g_t = _torch_tensor(g_val)
            Rs_t = params.R0 * torch.exp(params.Ea_over_k / T_t) * g_t
            Rs = np.asarray(Rs_t.cpu().numpy(), dtype=_SIM_DTYPE)
        else:
            exp_arg = params.Ea_over_k / T_clamped
            Rs = np.asarray(params.R0 * np.exp(exp_arg) * g_val, dtype=_SIM_DTYPE)
        return np.asarray(Rs + params.Rm, dtype=_SIM_DTYPE), g_val

    def _update_reversal(self, T_clamped: np.ndarray) -> None:
        """Detect heating/cooling changes (δ flip) when |ΔT| exceeds threshold; update minor-loop state."""
        params = self.params
        dT = T_clamped - self.T_last
        mask = np.abs(dT) > params.reversal_threshold_K
        if not np.any(mask):
            self.T_last = T_clamped
            return
        delta_new = np.sign(dT)
        delta_new[delta_new == 0.0] = self.delta[delta_new == 0.0]
        reversal_mask = mask & (delta_new != self.delta)
        if np.any(reversal_mask):
            g_at_T = self.g(T_clamped)
            self.gr[reversal_mask] = g_at_T[reversal_mask]
            self.delta[reversal_mask] = delta_new[reversal_mask]
            self.reversed[reversal_mask] = _SIM_DTYPE(1.0)
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
    """Array simulator for VO₂ neuristors (NumPy, no Torch)."""

    def __init__(
        self,
        Vin: float | Sequence[float],
        resist_params: YuanhangResistParams | None,
        circuit_params: YuanhangCircuitParams | None,
        lattice_shape: Tuple[int, int],
        start_branch: str,
        init: Dict[str, float] | None,
    ) -> None:
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
            T_state_init = _broadcast_array(init["T_K"], self.N, "T_K")
            T_hysteresis_init = T_state_init.copy()
        elif "T_C" in init:
            T_state_init = _broadcast_array(init["T_C"] + _KELVIN_OFFSET, self.N, "T_C")
            T_hysteresis_init = T_state_init.copy()
        else:
            T_state_init = np.full(self.N, self.circuit.T_base_K, dtype=_SIM_DTYPE)
            T_hysteresis_init = np.full(self.N, self.circuit.T_base_K - 0.1, dtype=_SIM_DTYPE)
        self.T_K = T_state_init
        self.hysteresis = HysteresisArray(self.resist, self.N, start_branch=start_branch)
        self.hysteresis.initialize(T_hysteresis_init)
        self.Cth_factor = _broadcast_array(self.circuit.Cth_factor, self.N, "Cth_factor")
        self.noise_strength_K_per_s = _SIM_DTYPE(self.circuit.noise_strength_K_per_s)
        self.S_env = _SIM_DTYPE(
            self.circuit.Sth_mW_per_K * 1e-3 * (1.0 - 2.0 * self.dimension * self.circuit.couple_factor)
        )
        self.S_couple = _SIM_DTYPE(self.circuit.Sth_mW_per_K * 1e-3 * self.circuit.couple_factor)
        self.R_series_ohm = max(self.circuit.R_series_ohm, _EPS)
        self.C_par_F = max(self.circuit.C_par_F, _EPS)
        self.C_th_J_per_K = max(self.circuit.Cth_J_per_K, _EPS)
        self.T_base = _SIM_DTYPE(self.circuit.T_base_K)

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

    def run(
        self,
        t_end: float,
        dt: float,
        noise_seed: int | None = None,
        cancel_cb: Callable[[], bool] | None = None,
    ) -> SimOut:
        """Integrate the ODEs for t∈[0, t_end] with step dt (Euler). Returns SimOut with traces."""
        steps = int(t_end / dt)
        if steps <= 0:
            raise ValueError("t_end/dt must yield at least one timestep")
        time_s = [0.0] * (steps + 1)
        history_V = np.zeros((steps + 1, self.N), dtype=_SIM_DTYPE)
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
            if cancel_cb is not None and cancel_cb():
                raise SimulationCancelled("cancelled")
            R_vo2, g_val = self.hysteresis.evaluate(self.T_K)
            I_load = (self.V_bias - self.Vn) / self.R_series_ohm
            I_vo2 = self.Vn / np.maximum(R_vo2, _EPS)
            dV_dt = (self.V_bias - self.Vn) / (self.R_series_ohm * self.C_par_F) - self.Vn / (R_vo2 * self.C_par_F)
            laplacian = _compute_laplacian(self.T_K, self.Nx, self.Ny)
            P_vo2 = (self.Vn * self.Vn) / np.maximum(R_vo2, _EPS)
            if self.noise_strength_K_per_s > 0.0:
                noise = np.asarray(self.noise_strength_K_per_s * rng.standard_normal(self.N), dtype=_SIM_DTYPE)
            else:
                noise = _SIM_DTYPE(0.0)
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


def series_mean(series: List[float] | List[List[float]]) -> List[float]:
    """Average across devices for multi-device lattices; pass-through for single-device."""
    if not series:
        return []
    if isinstance(series[0], list):
        length = len(series[0])
        return [sum(device[i] for device in series) / len(series) for i in range(length)]
    return series  # type: ignore[return-value]


def series_first(series: List[float] | List[List[float]]) -> List[float]:
    """Return first device’s series for multi-device lattices; pass-through for single-device."""
    if not series:
        return []
    if isinstance(series[0], list):
        return series[0]
    return series  # type: ignore[return-value]


def detect_spike_times(
    time_s: List[float],
    I_vo2: List[float],
    threshold_A: float = 1e-3,
) -> List[float]:
    """Detect spike times (in seconds) in I_vo2 as local maxima of |I| above `threshold_A`."""
    n = len(time_s)
    if n < 3:
        return []
    t = np.asarray(time_s, dtype=float)
    I = np.asarray(I_vo2, dtype=float)
    mag = np.abs(I)
    spikes: List[float] = []
    for i in range(1, n - 1):
        if mag[i] > threshold_A and mag[i] > mag[i - 1] and mag[i] >= mag[i + 1]:
            spikes.append(t[i])
    return spikes


def is_oscillatory(
    data: SimOut,
    t_start_us: float = 25.0,
    t_end_us: float = 300.0,
    threshold_A: float = 1e-3,
    min_spikes: int = 4,
) -> bool:
    """Return True if the given SimOut exhibits at least `min_spikes` spikes in I_vo2 within the window."""
    time_s = data["time_s"]
    I_vo2 = series_first(data["I_vo2"])
    if not time_s or not I_vo2:
        return False
    t_arr = np.asarray(time_s, dtype=float)
    I_arr = np.asarray(I_vo2, dtype=float)
    t_us = t_arr * 1e6
    mask = (t_us >= t_start_us) & (t_us <= t_end_us)
    if not np.any(mask):
        return False
    t_win = t_arr[mask]
    I_win = I_arr[mask]
    spike_times = detect_spike_times(t_win.tolist(), I_win.tolist(), threshold_A=threshold_A)
    return len(spike_times) >= min_spikes


def simulate_yuanhang(
    Vin: float | Sequence[float],
    t_end: float = 60e-6,
    dt: float = 10e-9,
    resist_params: YuanhangResistParams | None = None,
    circuit_params: YuanhangCircuitParams | None = None,
    init: Dict[str, float] | None = None,
    start_branch: str = "insulator",
    lattice_shape: Tuple[int, int] = (1, 1),
    noise_seed: int | None = None,
    cancel_cb: Callable[[], bool] | None = None,
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
    return simulator.run(t_end=t_end, dt=dt, noise_seed=noise_seed, cancel_cb=cancel_cb)


def sweep_1d(run_one: Callable[[T], SimOut], values: Iterable[T]) -> Dict[T, SimOut]:
    """Run a 1D sweep over iterable `values`, returning {value: SimOut}."""
    results: Dict[T, SimOut] = {}
    vals = list(values)
    total = len(vals)
    for idx, v in enumerate(vals, start=1):
        print(f"[sweep] Simulating {v} ({idx}/{total})")
        results[v] = run_one(v)
    print("[sweep] Completed sweep.")
    return results


def simulate_vin_sweep(
    vins: Sequence[float],
    t_end: float = 60e-6,
    dt: float = 10e-9,
    resist_params: YuanhangResistParams | None = None,
    circuit_params: YuanhangCircuitParams | None = None,
    init: Dict[str, float] | None = None,
    start_branch: str = "insulator",
    lattice_shape: Tuple[int, int] = (1, 1),
    noise_seed: int | None = None,
    seed_offset: int = 1,
    cancel_cb: Callable[[], bool] | None = None,
) -> Dict[float, SimOut]:
    """Convenience sweep over a list of Vin values. Returns {Vin: SimOut}.

    Each Vin is simulated independently using `simulate_yuanhang`. If a base `noise_seed`
    is provided, runs are de-correlated by adding `seed_offset * idx` (idx is the Vin index).
    """
    results: Dict[float, SimOut] = {}
    vins_list = list(vins)
    total = len(vins_list)
    for idx, v in enumerate(vins_list):
        run_seed = None if noise_seed is None else noise_seed + seed_offset * idx
        results[v] = simulate_yuanhang(
            Vin=v,
            t_end=t_end,
            dt=dt,
            resist_params=resist_params,
            circuit_params=circuit_params,
            init=init,
            start_branch=start_branch,
            lattice_shape=lattice_shape,
            noise_seed=run_seed,
            cancel_cb=cancel_cb,
        )
    return results


def find_oscillatory_band_1d(
    run_sim: Callable[[float], SimOut],
    start: float,
    stop: float,
    step: float,
    osc_check: Callable[[SimOut], bool] = is_oscillatory,
) -> Tuple[float | None, float | None]:
    """Coarsely scan from `start` to `stop` in steps of `step` and detect first/last oscillatory points."""
    if step <= 0:
        raise ValueError("step must be > 0")
    n_steps = int(math.floor((stop - start) / step))
    print(f"[auto_domain] Coarse sweep: from {start:.2f} to {stop:.2f} in steps of {step:.2f} ({n_steps + 1} points)")
    min_osc: float | None = None
    max_osc: float | None = None
    for i in range(n_steps + 1):
        val = start + i * step
        sim = run_sim(val)
        if osc_check(sim):
            print(f"[auto_domain]   value = {val:.2f} → oscillatory")
            if min_osc is None:
                min_osc = val
            max_osc = val
        elif min_osc is not None:
            print(f"[auto_domain]   value = {val:.2f} → non-oscillatory (exiting band, stopping coarse scan)")
            break
        else:
            print(f"[auto_domain]   value = {val:.2f} → non-oscillatory")
    return min_osc, max_osc
