"""
Single VO₂ neuristor simulator (constant Vin).

This script integrates the coupled electrical (RC) and thermal ODEs for a single
Mott device with a hysteretic resistance model.

Conventions / Units
-------------------
• Temperatures stored as Celsius (°C). Absolute (Kelvin) used only for Arrhenius.
• Voltages in Volts (V), currents in Amperes (A), resistances in Ohms (Ω).
• Capacitances in Farads (F); thermal C_th in J/K; thermal G_th in W/K.
"""
import math

from dataclasses import dataclass
from typing import Tuple, List, Dict, TypedDict

# -----------------------
# Numerical constants
# -----------------------
_EPS: float = 1e-12        # small epsilon to avoid division by zero
_LARGE: float = 1e6        # soft clip for arguments
_KELVIN_OFFSET: float = 273.15  # °C → K

# "Modeling of the hysteretic metal-insulator transition"
def P(x: float, gamma: float) -> float:
    """
    Proximity window function Π(x;γ) used by the hysteresis model.
    
    It multiplies a smooth "tanh window" by a sinusoidal branch shaper:
        Π = 0.5 * (1 - sin(γ x)) * (1 + tanh(π - 2π x))
    
    The input x is a normalized proximity (dimensionless). We soft-clip x to
    ±_LARGE to prevent numerical overflow and keep Π ∈ [0, 1].
    """
    # Soft-limit x so that tanh() and sin() behave well and keep Π in [0,1]
    if x > _LARGE:
        x = _LARGE
    elif x < -_LARGE:
        x = -_LARGE
    return 0.5 * (1 - math.sin(gamma * x)) * (1 + math.tanh(math.pi - 2 * math.pi * x))


# Notes on units for resistance update:
#  - Temperatures (T_celsius, prev_T_celsius, Tc, Tr, Tpr, w) are in °C; differences equal K.
#  - TK = T_celsius + _KELVIN_OFFSET is used only in the Arrhenius term (Kelvin).
#  - Ea must be in eV if k is provided in eV/K (k ≈ 8.617333262e-5 eV/K).
def update_R(T_celsius: float, prev_T_celsius: float, state: Tuple[float, float, float, float, float], params: Tuple[float, float, float, float, float, float, float, float]) -> Tuple[float, Tuple[float, float, float, float, float]]:
    """
    Update the VO₂ resistance using the hysteresis model.
    
    Parameters
    ----------
    T_celsius : float
        Current device temperature (°C).
    prev_T_celsius : float
        Previous step device temperature (°C) to determine heating/cooling direction.
    state : tuple
        Hysteresis internal state: (d_prev, Tr, gr, Tpr, g_prev)
          d_prev : +1 / -1 / 0  → previous heating/cooling direction
          Tr     : last reversal temperature (°C)
          gr     : g(Tr) value at reversal (∈ [0,1])
          Tpr    : proximity normalization (°C), avoids division by ~0
          g_prev : previous g value (∈ [0,1])
    params : tuple
        Resistance parameters: (R0, Ea, k, Rm, w, Tc, b, gamma)
    Returns
    -------
    (R, new_state) : (float, tuple)
        R : instantaneous device resistance (Ω)
        new_state : updated hysteresis state as above
    """
    d_prev, Tr, gr, Tpr, g_prev = state
    R0, Ea, k, Rm, w, Tc, b, gamma = params

    # --- Numerical/physical guards ---
    # Ensure Boltzmann constant and absolute temperature are valid.
    if k <= 0:
        raise ValueError("Boltzmann constant k must be positive (in eV/K if Ea is in eV).")

    # 1) sign of heating/cooling (EPS is use to create a band of acceptable "0"'s)
    dT = T_celsius - prev_T_celsius
    if dT > _EPS:
        d = 1
    elif dT < -_EPS:
        d = -1
    else:
        d = d_prev

    # 2) reversal?
    if d != d_prev:
        Tr = prev_T_celsius
        # Clamp previous g to (0,1) to avoid atanh(±1)
        gr = _clamp01(g_prev)
        # Prevent exactly 0 or 1 which would make atanh blow up
        gr_safe = min(max(gr, 1e-12), 1 - 1e-12)
        Tpr = d * (w / 2.0) + Tc - (1.0 / b) * math.atanh(2.0 * gr_safe - 1.0) - Tr

    # Avoid division by ~0 in proximity normalization
    if abs(Tpr) < _EPS:
        Tpr = _EPS if Tpr >= 0 else -_EPS

    # 3) proximity
    x = (T_celsius - Tr) / Tpr
    Pi = P(x, gamma)

    # 4) g(T) on current branch (Eq. (18))
    g = 0.5 + 0.5 * math.tanh(b * (d * (w / 2.0) + Tc - (T_celsius + Tpr * Pi)))
    g = _clamp01(g)

    # 5) phase resistances
    TK = T_celsius + _KELVIN_OFFSET
    if TK <= 0:
        raise ValueError("Absolute temperature must be > 0 K. Check input T_celsius.")
    Rs = R0 * math.exp(Ea / (k * TK))

    # 6) mixture
    R = g * Rs + (1 - g) * Rm

    # update state and return
    return R, (d, Tr, gr, Tpr, g)


def _clamp01(x: float) -> float:
    """Clamp to [0, 1] to avoid atanh/divergence issues."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# =======================
# Dataclasses (parameters)
# =======================
@dataclass
class ResistParams:
    """Parameters for the VO₂ resistance hysteresis model."""
    # Resistance-model parameters (DO NOT change model above)
    R0: float      # Ohm
    Ea: float      # eV
    k: float       # eV/K
    Rm: float      # Ohm (metallic-state resistance)
    w: float       # deg C (width of major loop)
    Tc: float      # deg C (center temperature)
    b: float       # 1/deg C (branch sharpness)
    gamma: float   # window gamma


@dataclass
class ThermalElectricalParams:
    """Circuit and thermal parameters for the single-neuristor setup."""
    # Circuit + thermal parameters for the single neuristor (di Ventra/Qiu model)
    R_load: float      # Ohm (series load)
    C_par: float       # F   (parasitic capacitance in parallel with VO2)
    C_th: float        # J/K (thermal capacitance of active region)
    G_th: float        # W/K (thermal conductance to bath)
    T_base_C: float    # deg C (ambient / substrate temperature)
    eta: float = 1.0   # dimensionless scaling of how much Joule power heats the VO2 region


class SimOut(TypedDict):
    time_s: List[float]
    V_node: List[float]
    I_load: List[float]
    I_vo2: List[float]
    T_C: List[float]
    R_vo2: List[float]
    g: List[float]


def _simulate_single_neuristor(
    Vin: float,
    t_end: float = 60e-6,
    dt: float = 10e-9,
    resist_params: ResistParams = None,
    te_params: ThermalElectricalParams = None,
    init: Dict[str, float] | None = None,
    start_branch: str = "insulator",
) -> SimOut:
    """
    Simulate a single VO₂ neuristor under constant Vin for duration t_end with step dt.
    
    Algorithm (forward Euler at node):
    1) Evaluate R_vo2(T) from hysteresis (update_R) and obtain branch g.
    2) KCL: I_load = (Vin - Vn)/R_load, I_vo2 = Vn/R_vo2, I_cap = I_load - I_vo2.
    3) Node voltage: dVn/dt = I_cap / C_par → Vn ← Vn + dVn/dt·dt.
    4) Thermal: dT/dt = (-G_th (T - T_base) + η·P_vo2) / C_th with P_vo2 = Vn²/R_vo2 → T ← T + dT/dt·dt.
    5) Store traces; repeat.

    Start state:
    The hysteresis internal state can be biased via `start_branch` ∈ {"insulator","metal"}.
    This affects only the initial hysteresis memory (g_prev, etc.); equations are unchanged.
    """
    # ---- defaults (safe placeholders; replace with your device parameters) ----
    if resist_params is None:
        resist_params = ResistParams(
            R0=5.36e-3,            # Ohm (insulating prefactor; keep as in paper)
            Ea=0.449,              # eV (Ea ≈ 5220 K * k_B ≈ 0.449 eV)
            k=8.617333262e-5,      # eV/K (Boltzmann)
            Rm=1286.0,             # Ohm (metallic-state resistance)
            w=7.19,                # deg C (width of major loop; ≈ 7.19 K)
            Tc=332.8 - 273.15,     # deg C (center temperature ≈ 59.65 °C)
            b=0.253,               # 1/deg C (branch sharpness β)
            gamma=0.956,           # window gamma
        )
    if te_params is None:
        te_params = ThermalElectricalParams(
            R_load=12_000.0,     # Ohm (per Table 1)
            C_par=145e-12,       # F (≈145 pF parasitic)
            C_th=49.6e-12,       # J/K (thermal capacitance ≈ 49.6 pJ/K)
            G_th=0.201e-3,       # W/K (Se ≈ 0.201 mW/K)
            T_base_C=325.0 - 273.15,  # deg C (ambient ≈ 51.85 °C)
            eta=1.0,
        )

    # Initial conditions
    Vn = (init.get("Vn") if init else 0.0)  # node voltage across VO2/C
    T_C = (init.get("T_C") if init else te_params.T_base_C)
    prev_T_C = T_C

    # Hysteresis state: (d_prev, Tr, gr, Tpr, g_prev)
    # Bias initial memory toward insulating or metallic branch without changing equations.
    if start_branch.lower() == "insulator":
        g_prev = 0.999   # mostly insulating
        d_prev = -1      # cooling direction bias
    else:
        g_prev = 0.001   # mostly metallic
        d_prev = +1      # heating direction bias
    Tr = T_C            # last reversal at current temperature
    gr = g_prev         # g at reversal matches our bias
    Tpr = resist_params.w / 2.0 if resist_params.w != 0 else 1.0  # reasonable non-zero normalization
    state = (d_prev, Tr, gr, Tpr, g_prev)

    # Convenience aliases
    R0, Ea, kB, Rm, w, Tc, b, gamma = (
        resist_params.R0,
        resist_params.Ea,
        resist_params.k,
        resist_params.Rm,
        resist_params.w,
        resist_params.Tc,
        resist_params.b,
        resist_params.gamma,
    )

    # Output buffers
    t = 0.0
    n_steps = int(t_end / dt)
    time_s: List[float] = [0.0] * (n_steps + 1)
    V_node: List[float] = [0.0] * (n_steps + 1)
    I_load: List[float] = [0.0] * (n_steps + 1)
    I_vo2: List[float] = [0.0] * (n_steps + 1)
    T_series_C: List[float] = [T_C] * (n_steps + 1)
    R_series: List[float] = [0.0] * (n_steps + 1)
    g_series: List[float] = [0.0] * (n_steps + 1)

    # Precompute constants
    inv_Cpar = 1.0 / max(te_params.C_par, _EPS)
    
    for i in range(n_steps + 1):
        # -- (1) Resistance update via hysteresis model --
        R_vo2, state = update_R(T_C, prev_T_C, state, (R0, Ea, kB, Rm, w, Tc, b, gamma))

        # -- (2) Currents at this instant (KCL) --
        I_through_load = (Vin - Vn) / te_params.R_load
        I_through_vo2 = Vn / max(R_vo2, _EPS)
        I_cap = I_through_load - I_through_vo2  # by KCL at node

        # -- (3) Node voltage update (Forward Euler) --
        dVn_dt = I_cap * inv_Cpar
        Vn_next = Vn + dVn_dt * dt

        # -- (4) Thermal update (Forward Euler) --
        P_vo2 = (Vn * Vn) / max(R_vo2, _EPS)  # W
        dT_dt = ( -te_params.G_th * (T_C - te_params.T_base_C) + te_params.eta * P_vo2 ) / max(te_params.C_th, _EPS)
        T_next = T_C + dT_dt * dt

        # -- (5) Store and advance time/state --
        time_s[i] = t
        V_node[i] = Vn
        I_load[i] = I_through_load
        I_vo2[i] = I_through_vo2
        T_series_C[i] = T_C
        R_series[i] = R_vo2
        g_series[i] = state[-1]  # g

        t += dt
        prev_T_C = T_C
        T_C = T_next
        Vn = Vn_next

    return {
        "time_s": time_s,
        "V_node": V_node,
        "I_load": I_load,     # circuit current through R_load
        "I_vo2": I_vo2,       # device branch current
        "T_C": T_series_C,
        "R_vo2": R_series,
        "g": g_series,
    }


# CSV output columns: ["time_s", "V_node", "I_load", "I_vo2", "T_C", "R_vo2", "g"]
def run_and_save_csv(Vin: float, t_end: float = 60e-6, dt: float = 10e-9, filename: str = None) -> str:
    """Run a 60 µs simulation at 10 ns and save CSV to disk. Returns path."""
    import csv, os

    data = _simulate_single_neuristor(Vin=Vin, t_end=t_end, dt=dt)
    if filename is None:
        filename = f"neuristor_Vin_{Vin:.3f}V_60us_10ns.csv".replace("/", "_")
    outpath = os.path.join(os.path.dirname(__file__), filename)

    keys = ["time_s", "V_node", "I_load", "I_vo2", "T_C", "R_vo2", "g"]
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row_idx in range(len(data["time_s"])):
            w.writerow([data[k][row_idx] for k in keys])
    return outpath



# ---- Helper function for Vin sweep ----
def _run_vin_sweep(vins, t_end, dt, start_branch: str = "insulator"):
    """Run a sweep over multiple Vin values; returns dict[vin]->data dict."""
    results = {}
    for v in vins:
        data = _simulate_single_neuristor(Vin=v, t_end=t_end, dt=dt, start_branch=start_branch)
        results[v] = data
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate a single VO₂ neuristor (constant Vin). Default: sweep 9,11,13,15,17 V. Example: --vin_list '9,11,13,15,17' --t_end_us 60 --dt_ns 10"
    )
    parser.add_argument("--t_end_us", type=float, default=60.0, help="Simulation duration in microseconds")
    parser.add_argument("--dt_ns", type=float, default=10.0, help="Timestep in nanoseconds")
    parser.add_argument("--vin_list", type=str, default="", help="Comma-separated Vin list to sweep, e.g. '9,11,13,15,17'")
    parser.add_argument("--save_png", type=str, default="", help="If provided, save the generated plot to this PNG path")
    parser.add_argument("--save_dir", type=str, default="", help="If provided, save per-Vin plots into this directory")
    parser.add_argument("--no_combined", action="store_true", help="If set, skip the combined sweep figure")
    parser.add_argument("--start_branch", choices=["insulator","metal"], default="insulator", help="Initial VO₂ phase bias for hysteresis state")
    args = parser.parse_args()

    t_end = args.t_end_us * 1e-6
    dt = args.dt_ns * 1e-9

    # Determine Vin list: default sweep or user-provided list
    if args.vin_list.strip():
        try:
            vins = [float(s.strip()) for s in args.vin_list.split(",") if s.strip()]
        except ValueError:
            raise SystemExit("Could not parse --vin_list. Use a comma-separated list like '9,11,13,15,17'.")
    else:
        vins = [9.0, 11.0, 13.0, 15.0, 17.0]

    # ---- Sweep path: run without CSV (faster), single figure with all traces ----
    import matplotlib.pyplot as plt

    results = _run_vin_sweep(vins, t_end=t_end, dt=dt, start_branch=args.start_branch)

    if not args.no_combined:
        plt.figure(figsize=(9, 5))
        for v in vins:
            data = results[v]
            # Convert to mA and µs for plotting
            t_us = [t * 1e6 for t in data['time_s']]
            i_mA = [i * 1e3 for i in data['I_load']]
            plt.plot(t_us, i_mA, label=f"Vin = {v:g} V")
        plt.xlabel("Time (µs)")
        plt.ylabel("Current through neuristor (mA)")
        plt.title("Single VO₂ Neuristor – I_load vs Time (Vin sweep)")
        plt.grid(True)
        plt.legend(title="Bias")
        plt.tight_layout()
        if args.save_png:
            plt.savefig(args.save_png, dpi=200)
            print(f"Saved plot to: {args.save_png}")

    # ---- Per-Vin triplet plots: Temperature, Current, Voltage vs time (stacked, shared x) ----
    import os
    for v in vins:
        data = results[v]
        t_us = [t * 1e6 for t in data['time_s']]
        T_C = data['T_C']
        I_mA = [i * 1e3 for i in data['I_load']]
        V_node = data['V_node']

        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(9, 8))

        # Temperature vs time
        axes[0].plot(t_us, T_C)
        axes[0].set_ylabel("Temp (°C)")
        axes[0].set_title(f"VO₂ Neuristor – Vin = {v:g} V")
        axes[0].grid(True)

        # Current vs time
        axes[1].plot(t_us, I_mA)
        axes[1].set_ylabel("Current (mA)")
        axes[1].grid(True)

        # Voltage vs time (node voltage across VO₂)
        axes[2].plot(t_us, V_node)
        axes[2].set_xlabel("Time (µs)")
        axes[2].set_ylabel("Voltage (V)")
        axes[2].grid(True)

        fig.tight_layout()
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            fig.savefig(os.path.join(args.save_dir, f"Vin_{v:g}V_triplet.png"), dpi=200)

    plt.show()
