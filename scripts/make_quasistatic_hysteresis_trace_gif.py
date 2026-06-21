from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import neuristor.model as model
from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step
from neuristor.sample_library import load_sample_json, params_from_dict


OUTPUT_DIR = ROOT / "outputs" / "yuanhang_reference_hysteresis_trace_1100_to_2470_50frame_gif"
FRAME_DIR = OUTPUT_DIR / "frames"
GIF_PATH = OUTPUT_DIR / "yuanhang_reference_hysteresis_trace_1100_to_2470_50frame_uA.gif"
CURRENT_MIN_UA = 1100.0
CURRENT_MAX_UA = 2470.0
N_CURRENTS = 50
PULSE_ON_NS = 150.0
PULSE_OFF_NS = 1200.0
T_END_NS = 1300.0
FRAME_DURATION_MS = 2000


def _major_branch(
    T_K: np.ndarray,
    resist: model.YuanhangResistParams,
    *,
    delta: float,
) -> np.ndarray:
    arg = resist.beta * (delta * resist.w_eff / 2.0 + resist.Tc_K - T_K)
    g = 0.5 + 0.5 * np.tanh(arg)
    return resist.Rm + resist.R0 * np.exp(resist.Ea_over_k / T_K) * g


def _plot_frame(
    *,
    out: dict[str, np.ndarray],
    current_uA: float,
    frame_number: int,
    current_ylim: tuple[float, float],
    voltage_ylim: tuple[float, float],
    rt_temperature_xlim: tuple[float, float],
    rt_resistance_ylim: tuple[float, float],
    branch_temperature: np.ndarray,
    heating_resistance: np.ndarray,
    cooling_resistance: np.ndarray,
    path: Path,
) -> None:
    t_ns = out["t"] * 1e9
    i_mA = out["I_in"] * 1e3
    v_v = out["V_vo2"]
    T_K = out["T"]
    R_ohm = out["R"]
    pulse = (t_ns >= PULSE_ON_NS) & (t_ns <= PULSE_OFF_NS)
    pulse_indices = np.flatnonzero(pulse)
    pulse_off_index = int(pulse_indices[-1])
    trace_stride = max(1, int(math.ceil(pulse_indices.size / 450)))
    trace_indices = pulse_indices[::trace_stride]

    fig = plt.figure(figsize=(15.5, 6.2), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.32)
    ax_i = fig.add_subplot(grid[0, 0])
    ax_v = ax_i.twinx()
    ax_rt = fig.add_subplot(grid[0, 1])

    ax_i.step(t_ns, i_mA, where="post", color="#16833b", linewidth=2.2, label="Source current")
    ax_v.plot(t_ns, v_v, color="#7b2cbf", linewidth=2.0, label="VO$_2$ voltage")
    ax_i.set_xlim(0.0, T_END_NS)
    ax_i.set_ylim(*current_ylim)
    ax_v.set_ylim(*voltage_ylim)
    ax_i.set_xlabel("Time (ns)")
    ax_i.set_ylabel("Source current (mA)", color="#16833b")
    ax_v.set_ylabel("VO$_2$ voltage (V)", color="#7b2cbf")
    ax_i.tick_params(axis="y", colors="#16833b")
    ax_v.tick_params(axis="y", colors="#7b2cbf")
    ax_i.grid(True, color="#c8c8c8", alpha=0.45, linewidth=0.8)
    ax_i.axvline(PULSE_ON_NS, color="#16833b", alpha=0.28, linewidth=1.0)
    ax_i.axvline(PULSE_OFF_NS, color="#16833b", alpha=0.28, linewidth=1.0)
    ax_i.legend(
        [ax_i.lines[0], ax_v.lines[0]],
        ["Source current", "VO$_2$ voltage"],
        loc="upper left",
        frameon=False,
    )

    ax_rt.plot(branch_temperature, heating_resistance, color="#ed5a49", linewidth=2.0, label="Heating branch")
    ax_rt.plot(branch_temperature, cooling_resistance, color="#3478d4", linewidth=2.0, label="Cooling branch")
    ax_rt.plot(T_K[pulse], R_ohm[pulse], color="#d7191c", alpha=0.42, linewidth=1.0)
    ax_rt.scatter(
        T_K[trace_indices],
        R_ohm[trace_indices],
        color="#d7191c",
        s=12,
        alpha=0.60,
        edgecolors="none",
        label="Simulated path",
        zorder=4,
    )
    ax_rt.scatter(
        [T_K[pulse_off_index]],
        [R_ohm[pulse_off_index]],
        color="#d7191c",
        s=90,
        edgecolors="white",
        linewidths=1.2,
        label="State at pulse off",
        zorder=6,
    )
    ax_rt.set_xlim(*rt_temperature_xlim)
    ax_rt.set_ylim(*rt_resistance_ylim)
    ax_rt.set_yscale("log")
    ax_rt.set_xlabel("Temperature (K)")
    ax_rt.set_ylabel("Resistance (Ohm)")
    ax_rt.set_title("Hysteresis R(T) and simulated state path")
    ax_rt.grid(True, which="both", color="#c8c8c8", alpha=0.38, linewidth=0.8)
    ax_rt.legend(loc="upper right", frameon=False, fontsize=9)

    fig.suptitle(
        f"Quasistatic + Yuanhang reference reversal | frame {frame_number:02d}/{N_CURRENTS} | "
        f"pulse amplitude {current_uA / 1000.0:.4f} mA",
        fontsize=15,
    )
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.06, right=0.94)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    sample = load_sample_json(ROOT / "presets" / "resistance_100425_chip1_gap3.json")
    resist = params_from_dict(sample["resist_params"])
    params = CurrentDriveParams(
        dt_s=0.00791e-9,
        t_end_s=T_END_NS * 1e-9,
        pulse_on_s=PULSE_ON_NS * 1e-9,
        pulse_off_s=PULSE_OFF_NS * 1e-9,
        T0_K=311.21937437938016,
        T_init_K=311.21937437938016,
        C_F=25.930953e-12,
        C_th_J_per_K=5.0e-12,
        S_e_W_per_K=0.1e-3,
        phase_mode="quasistatic",
        resist_params=resist,
        start_branch="insulator",
    )
    currents_uA = np.linspace(CURRENT_MIN_UA, CURRENT_MAX_UA, N_CURRENTS)
    runs: list[tuple[float, dict[str, np.ndarray]]] = []
    summaries: list[dict[str, float]] = []
    voltage_min = math.inf
    voltage_max = -math.inf
    rt_temperature_min = math.inf
    rt_temperature_max = -math.inf
    rt_resistance_min = math.inf
    rt_resistance_max = -math.inf

    for index, current_uA in enumerate(currents_uA, start=1):
        out = simulate_current_step(float(current_uA), params=params, seed=0)
        runs.append((float(current_uA), out))
        voltage_min = min(voltage_min, float(np.min(out["V_vo2"])))
        voltage_max = max(voltage_max, float(np.max(out["V_vo2"])))
        rt_temperature_min = min(rt_temperature_min, float(np.min(out["T"])))
        rt_temperature_max = max(rt_temperature_max, float(np.max(out["T"])))
        rt_resistance_min = min(rt_resistance_min, float(np.min(out["R"])))
        rt_resistance_max = max(rt_resistance_max, float(np.max(out["R"])))
        pulse = (out["t"] >= params.pulse_on_s) & (out["t"] <= float(params.pulse_off_s))
        pulse_off_idx = int(np.flatnonzero(pulse)[-1])
        summaries.append(
            {
                "frame": index,
                "current_uA": float(current_uA),
                "pulse_off_T_K": float(out["T"][pulse_off_idx]),
                "pulse_off_R_ohm": float(out["R"][pulse_off_idx]),
                "T_min_during_pulse_K": float(np.min(out["T"][pulse])),
                "T_max_during_pulse_K": float(np.max(out["T"][pulse])),
                "R_min_during_pulse_ohm": float(np.min(out["R"][pulse])),
                "R_max_during_pulse_ohm": float(np.max(out["R"][pulse])),
            }
        )
        print(f"simulated {index:02d}/{N_CURRENTS}: {current_uA:.3f} uA", flush=True)

    branch_temperature = np.linspace(
        min(resist.T_min_K, rt_temperature_min) - 1.0,
        max(resist.T_max_K, rt_temperature_max) + 1.0,
        900,
    )
    heating_resistance = _major_branch(branch_temperature, resist, delta=1.0)
    cooling_resistance = _major_branch(branch_temperature, resist, delta=-1.0)
    rt_resistance_min = min(rt_resistance_min, float(np.min(heating_resistance)), float(np.min(cooling_resistance)))
    rt_resistance_max = max(rt_resistance_max, float(np.max(heating_resistance)), float(np.max(cooling_resistance)))

    voltage_span = max(voltage_max - voltage_min, 0.1)
    voltage_ylim = (min(-0.05, voltage_min - 0.06 * voltage_span), voltage_max + 0.06 * voltage_span)
    current_ylim = (-0.08, CURRENT_MAX_UA * 1e-3 * 1.08)
    rt_temperature_xlim = (float(branch_temperature[0]), float(branch_temperature[-1]))
    rt_resistance_ylim = (max(1.0, 0.7 * rt_resistance_min), 1.35 * rt_resistance_max)

    frame_paths: list[Path] = []
    for index, (current_uA, out) in enumerate(runs, start=1):
        frame_path = FRAME_DIR / f"frame_{index:02d}_{current_uA:.3f}uA.png"
        _plot_frame(
            out=out,
            current_uA=current_uA,
            frame_number=index,
            current_ylim=current_ylim,
            voltage_ylim=voltage_ylim,
            rt_temperature_xlim=rt_temperature_xlim,
            rt_resistance_ylim=rt_resistance_ylim,
            branch_temperature=branch_temperature,
            heating_resistance=heating_resistance,
            cooling_resistance=cooling_resistance,
            path=frame_path,
        )
        frame_paths.append(frame_path)
        print(f"rendered {index:02d}/{N_CURRENTS}: {frame_path.name}", flush=True)

    imageio.mimsave(GIF_PATH, [imageio.imread(path) for path in frame_paths], duration=FRAME_DURATION_MS, loop=0)
    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "simulation_summary.csv", index=False)
    metadata = {
        "model": "quasistatic ideal-current equations + Yuanhang reference reversal",
        "sample": "100425 chip1 gap3 - top-aware fitted R-T",
        "n_currents": N_CURRENTS,
        "currents_uA": currents_uA.tolist(),
        "pulse_on_ns": PULSE_ON_NS,
        "pulse_off_ns": PULSE_OFF_NS,
        "duration_ns": T_END_NS,
        "frame_duration_s": FRAME_DURATION_MS / 1000.0,
        "fixed_current_axis_mA": list(current_ylim),
        "fixed_voltage_axis_V": list(voltage_ylim),
        "fixed_rt_temperature_axis_K": list(rt_temperature_xlim),
        "fixed_rt_resistance_axis_ohm": list(rt_resistance_ylim),
        "red_trace": "all simulated T,R states during the current pulse; large red dot is the pulse-off state",
        "gif_path": str(GIF_PATH),
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
