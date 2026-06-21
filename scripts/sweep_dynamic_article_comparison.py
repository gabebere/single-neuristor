from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuristor.current_domain_search import analyze_current_trace
from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step
from neuristor.model import YuanhangResistParams


OUTPUT_DIR = ROOT / "outputs" / "dynamic_article_comparison"
RESISTANCE_PRESET = ROOT / "presets" / "resistance_100425_chip1_gap3.json"

BASE = {
    "I_target_uA": 1500.0,
    "C_pF": 80.0,
    "C_th_pJ_per_K": 3.0,
    "S_e_mW_per_K": 0.052838,
    "tau_g_ns": 10.0,
    "T0_K": 325.0,
}

SWEEPS = {
    "current": ("I_target_uA", [400, 600, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1800, 2000]),
    "electrical_capacitance": ("C_pF", [30, 50, 80, 120, 180]),
    "thermal_capacitance": ("C_th_pJ_per_K", [0.5, 1, 2, 3, 5, 8]),
    "thermal_conductance": ("S_e_mW_per_K", [0.015, 0.025, 0.035, 0.052838, 0.075, 0.10]),
    "phase_time": ("tau_g_ns", [1, 2.5, 5, 10, 15, 25]),
}


def run_case(resist: YuanhangResistParams, values: dict[str, float]) -> dict[str, float]:
    params = CurrentDriveParams(
        dt_s=0.05e-9,
        t_end_s=3_000e-9,
        T0_K=values["T0_K"],
        T_init_K=values["T0_K"],
        C_F=values["C_pF"] * 1e-12,
        C_th_J_per_K=values["C_th_pJ_per_K"] * 1e-12,
        S_e_W_per_K=values["S_e_mW_per_K"] * 1e-3,
        phase_mode="dynamic",
        tau_g_s=values["tau_g_ns"] * 1e-9,
        resist_params=resist,
        start_branch="insulator",
    )
    out = simulate_current_step(values["I_target_uA"], params=params, seed=0)
    metrics = analyze_current_trace(
        out,
        params=params,
        min_vpp_mV=12.0,
        max_vpp_mV=2_000.0,
        min_cycles=5,
        pulse_on_ns=0.0,
        pulse_off_ns=None,
    )
    late = out["t"] >= 0.60 * out["t"][-1]
    late_out = {key: value[late] for key, value in out.items() if isinstance(value, np.ndarray) and value.shape[0] == late.shape[0]}
    late_metrics = analyze_current_trace(
        late_out,
        params=params,
        min_vpp_mV=12.0,
        max_vpp_mV=2_000.0,
        min_cycles=4,
        pulse_on_ns=float(late_out["t"][0] * 1e9),
        pulse_off_ns=None,
    )
    mean_power_uW = float(np.mean(out["P"][late]) * 1e6)
    frequency_mhz = float(metrics["dominant_freq_MHz"])
    energy_pj = mean_power_uW / frequency_mhz if frequency_mhz > 0.0 else float("nan")
    return {
        **values,
        **{str(k): float(v) for k, v in metrics.items()},
        **{f"late_{k}": float(v) for k, v in late_metrics.items()},
        "sustained_strict": float(
            late_metrics["oscillatory"] > 0.5
            and late_metrics["dominant_freq_MHz"] >= 1.0
            and late_metrics["late_to_early_vpp"] >= 0.50
        ),
        "late_V_pp_mV": float(np.ptp(out["V_vo2"][late]) * 1e3),
        "late_T_min_K": float(np.min(out["T"][late])),
        "late_T_max_K": float(np.max(out["T"][late])),
        "late_g_min": float(np.min(out["g_dyn"][late])),
        "late_g_max": float(np.max(out["g_dyn"][late])),
        "mean_power_uW": mean_power_uW,
        "energy_pJ_per_cycle": energy_pj,
    }


def make_plot(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (sweep_name, (variable, _)) in zip(axes.flat, SWEEPS.items()):
        part = results[results["sweep"] == sweep_name].sort_values(variable)
        accepted = part["sustained_strict"] > 0.5
        ax.plot(part[variable], part["dominant_freq_MHz"], color="#777777", linewidth=1.2)
        ax.scatter(
            part.loc[~accepted, variable],
            part.loc[~accepted, "dominant_freq_MHz"],
            color="#D84A3A",
            label="not sustained",
        )
        ax.scatter(
            part.loc[accepted, variable],
            part.loc[accepted, "dominant_freq_MHz"],
            color="#147D64",
            label="sustained",
        )
        if variable in {"C_pF", "C_th_pJ_per_K", "S_e_mW_per_K", "tau_g_ns"}:
            ax.set_xscale("log")
        ax.axhspan(40, 60, color="#4C78A8", alpha=0.10)
        ax.set_title(sweep_name.replace("_", " ").title())
        ax.set_xlabel(variable)
        ax.set_ylabel("Dominant frequency (MHz)")
        ax.grid(alpha=0.22)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")
    fig.suptitle("Dynamic-phase model sweeps using the validated 100425 resistance fit")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sweep_frequency_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resist_payload = json.loads(RESISTANCE_PRESET.read_text())
    resist = YuanhangResistParams(**resist_payload["resist_params"])

    rows: list[dict[str, float | str]] = []
    total = sum(len(values) for _, values in SWEEPS.values())
    index = 0
    for sweep_name, (variable, sweep_values) in SWEEPS.items():
        for sweep_value in sweep_values:
            index += 1
            values = dict(BASE)
            values[variable] = float(sweep_value)
            row = run_case(resist, values)
            row["sweep"] = sweep_name
            row["swept_variable"] = variable
            rows.append(row)
            print(
                f"{index}/{total} {sweep_name} {variable}={sweep_value}: "
                f"osc={int(row['oscillatory'])} f={row['dominant_freq_MHz']:.3f} MHz",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / "sweep_checkpoint.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "sweep_results.csv", index=False)
    make_plot(results)

    current = results[results["sweep"] == "current"]
    accepted = current[current["sustained_strict"] > 0.5]
    summary = {
        "model": "dynamic phase extension",
        "resistance_preset": str(RESISTANCE_PRESET),
        "base_parameters": BASE,
        "paper_targets": {
            "current_window_uA": [200.0, 600.0],
            "frequency_MHz": [40.0, 60.0],
            "power_uW": [60.0, 100.0],
            "energy_pJ_per_cycle": [1.2, 1.9],
        },
        "sustained_current_points_uA": accepted["I_target_uA"].tolist(),
        "observed_current_window_uA": (
            [float(accepted["I_target_uA"].min()), float(accepted["I_target_uA"].max())]
            if not accepted.empty
            else None
        ),
        "observed_frequency_MHz": (
            [float(accepted["dominant_freq_MHz"].min()), float(accepted["dominant_freq_MHz"].max())]
            if not accepted.empty
            else None
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
