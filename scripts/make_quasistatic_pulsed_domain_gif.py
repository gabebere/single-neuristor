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

from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step
from neuristor.sample_library import load_sample_json, params_from_dict


OUTPUT_DIR = ROOT / "outputs" / "quasistatic_pulsed_1100_to_1200_gif"
FRAME_DIR = OUTPUT_DIR / "frames"
GIF_PATH = OUTPUT_DIR / "quasistatic_pulsed_1100_to_1200_uA.gif"
CURRENT_MIN_UA = 1100.0
CURRENT_MAX_UA = 1200.0
N_CURRENTS = 20
PULSE_ON_NS = 150.0
PULSE_OFF_NS = 500.0
T_END_NS = 600.0
FRAME_DURATION_S = 1.0


def _plot_frame(
    *,
    out: dict[str, np.ndarray],
    current_uA: float,
    frame_number: int,
    current_ylim: tuple[float, float],
    voltage_ylim: tuple[float, float],
    path: Path,
) -> None:
    t_ns = out["t"] * 1e9
    i_mA = out["I_in"] * 1e3
    v_v = out["V_vo2"]

    fig, ax_i = plt.subplots(figsize=(10.5, 5.8), facecolor="white")
    ax_v = ax_i.twinx()
    ax_i.step(t_ns, i_mA, where="post", color="#16833b", linewidth=2.3, label="Source current")
    ax_v.plot(t_ns, v_v, color="#7b2cbf", linewidth=2.1, label="VO$_2$ voltage")

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
    ax_i.set_title(
        f"Quasistatic Yuanhang model | frame {frame_number:02d}/{N_CURRENTS} | "
        f"pulse amplitude {current_uA / 1000.0:.3f} mA"
    )
    lines = [ax_i.lines[0], ax_v.lines[0]]
    ax_i.legend(lines, ["Source current", "VO$_2$ voltage"], loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
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

    for index, current_uA in enumerate(currents_uA, start=1):
        out = simulate_current_step(float(current_uA), params=params, seed=0)
        runs.append((float(current_uA), out))
        voltage_min = min(voltage_min, float(np.min(out["V_vo2"])))
        voltage_max = max(voltage_max, float(np.max(out["V_vo2"])))
        pulse = (out["t"] >= params.pulse_on_s) & (out["t"] <= float(params.pulse_off_s))
        summaries.append(
            {
                "frame": index,
                "current_uA": float(current_uA),
                "pulse_on_ns": PULSE_ON_NS,
                "pulse_off_ns": PULSE_OFF_NS,
                "duration_ns": T_END_NS,
                "V_min_V": float(np.min(out["V_vo2"])),
                "V_max_V": float(np.max(out["V_vo2"])),
                "V_pp_during_pulse_V": float(np.ptp(out["V_vo2"][pulse])),
                "T_min_during_pulse_K": float(np.min(out["T"][pulse])),
                "T_max_during_pulse_K": float(np.max(out["T"][pulse])),
            }
        )
        print(f"simulated {index:02d}/{N_CURRENTS}: {current_uA:.3f} uA", flush=True)

    voltage_span = max(voltage_max - voltage_min, 0.1)
    voltage_margin = 0.06 * voltage_span
    voltage_ylim = (min(-0.05, voltage_min - voltage_margin), voltage_max + voltage_margin)
    current_ylim = (-0.08, CURRENT_MAX_UA * 1e-3 * 1.08)

    frame_paths: list[Path] = []
    for index, (current_uA, out) in enumerate(runs, start=1):
        frame_path = FRAME_DIR / f"frame_{index:02d}_{current_uA:.3f}uA.png"
        _plot_frame(
            out=out,
            current_uA=current_uA,
            frame_number=index,
            current_ylim=current_ylim,
            voltage_ylim=voltage_ylim,
            path=frame_path,
        )
        frame_paths.append(frame_path)
        print(f"rendered {index:02d}/{N_CURRENTS}: {frame_path.name}", flush=True)

    images = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(GIF_PATH, images, duration=int(round(FRAME_DURATION_S * 1000.0)), loop=0)
    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "simulation_summary.csv", index=False)
    metadata = {
        "model": "quasistatic ideal-current equations + Yuanhang hysteresis",
        "sample": "100425 chip1 gap3 - top-aware fitted R-T",
        "n_currents": N_CURRENTS,
        "currents_uA": currents_uA.tolist(),
        "pulse_on_ns": PULSE_ON_NS,
        "pulse_off_ns": PULSE_OFF_NS,
        "duration_ns": T_END_NS,
        "frame_duration_s": FRAME_DURATION_S,
        "fixed_current_axis_mA": list(current_ylim),
        "fixed_voltage_axis_V": list(voltage_ylim),
        "params": {
            "dt_ns": params.dt_s * 1e9,
            "C_pF": params.C_F * 1e12,
            "C_th_pJ_per_K": params.C_th_J_per_K * 1e12,
            "S_e_mW_per_K": params.S_e_W_per_K * 1e3,
            "T0_K": params.T0_K,
            "phase_mode": params.phase_mode,
        },
        "gif_path": str(GIF_PATH),
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
