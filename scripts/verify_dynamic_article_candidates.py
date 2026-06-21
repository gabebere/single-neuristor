from __future__ import annotations

import json
import sys
from pathlib import Path

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
CASES = [
    ("low_fixed_point", 600.0),
    ("low_oscillatory", 800.0),
    ("bifurcation_gap", 1100.0),
    ("central_oscillatory", 1500.0),
    ("high_oscillatory", 1800.0),
    ("high_fixed_point", 2000.0),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resist = YuanhangResistParams(**json.loads(RESISTANCE_PRESET.read_text())["resist_params"])
    rows: list[dict[str, float | str]] = []
    for label, current_uA in CASES:
        dt_values = (0.05, 0.025, 0.0125) if label in {
            "low_oscillatory",
            "bifurcation_gap",
            "central_oscillatory",
            "high_oscillatory",
        } else (0.05, 0.025)
        for dt_ns in dt_values:
            params = CurrentDriveParams(
                dt_s=dt_ns * 1e-9,
                t_end_s=5_000e-9,
                T0_K=325.0,
                T_init_K=325.0,
                C_F=80e-12,
                C_th_J_per_K=3e-12,
                S_e_W_per_K=0.052838e-3,
                phase_mode="dynamic",
                tau_g_s=10e-9,
                resist_params=resist,
                start_branch="insulator",
            )
            out = simulate_current_step(current_uA, params=params, seed=0)
            late = out["t"] >= 0.60 * out["t"][-1]
            late_out = {
                key: value[late]
                for key, value in out.items()
                if isinstance(value, np.ndarray) and value.shape[0] == late.shape[0]
            }
            metrics = analyze_current_trace(
                late_out,
                params=params,
                min_vpp_mV=12.0,
                max_vpp_mV=2_000.0,
                min_cycles=6,
                pulse_on_ns=float(late_out["t"][0] * 1e9),
                pulse_off_ns=None,
            )
            frequency_mhz = float(metrics["dominant_freq_MHz"])
            mean_power_uW = float(np.mean(out["P"][late]) * 1e6)
            row = {
                "label": label,
                "I_target_uA": current_uA,
                "dt_ns": dt_ns,
                **{str(k): float(v) for k, v in metrics.items()},
                "late_T_min_K": float(np.min(out["T"][late])),
                "late_T_max_K": float(np.max(out["T"][late])),
                "late_g_min": float(np.min(out["g_dyn"][late])),
                "late_g_max": float(np.max(out["g_dyn"][late])),
                "mean_power_uW": mean_power_uW,
                "energy_pJ_per_cycle": mean_power_uW / frequency_mhz if frequency_mhz > 1.0 else float("nan"),
            }
            row["verified_sustained"] = float(
                metrics["oscillatory"] > 0.5 and metrics["late_to_early_vpp"] >= 0.50
            )
            rows.append(row)
            print(
                f"{label} I={current_uA:g} uA dt={dt_ns:g} ns: "
                f"verified={int(row['verified_sustained'])} f={frequency_mhz:.3f} MHz "
                f"Vpp={metrics['V_pp_mV']:.1f} mV",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / "verification_checkpoint.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "verification_results.csv", index=False)


if __name__ == "__main__":
    main()
