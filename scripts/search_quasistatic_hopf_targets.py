from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import neuristor.model as model
from neuristor.current_domain_search import analyze_current_trace
from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step
from neuristor.sample_library import load_sample_json, params_from_dict


def _major_heating_resistance(T_K: np.ndarray, resist: model.YuanhangResistParams) -> np.ndarray:
    arg = resist.beta * (resist.w_eff / 2.0 + resist.Tc_K - T_K)
    g = 0.5 + 0.5 * np.tanh(arg)
    return resist.Rm + resist.R0 * np.exp(resist.Ea_over_k / T_K) * g


def _late_trace(out: dict[str, np.ndarray], fraction: float = 0.5) -> dict[str, np.ndarray]:
    mask = out["t"] >= fraction * out["t"][-1]
    return {
        key: value[mask]
        for key, value in out.items()
        if isinstance(value, np.ndarray) and value.ndim == 1 and value.shape[0] == mask.shape[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-candidates", type=int, default=80)
    args = parser.parse_args()

    sample = load_sample_json(args.sample)
    resist = params_from_dict(sample["resist_params"])
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    T = np.linspace(resist.Tc_K - 10.0, resist.Tc_K + 8.0, 361)
    R = _major_heating_resistance(T, resist)
    dR_dT = np.gradient(R, T)

    candidates: list[dict[str, float]] = []
    for index in range(T.size):
        if dR_dT[index] >= 0.0:
            continue
        for delta_T in (3.0, 8.0, 15.0, 25.0):
            T0 = float(T[index] - delta_T)
            if T0 < max(290.0, resist.T_min_K + 1.0):
                continue
            for G_mW_per_K in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
                G = G_mW_per_K * 1e-3
                current_A = math.sqrt(G * delta_T / R[index])
                current_uA = current_A * 1e6
                if not 50.0 <= current_uA <= 5000.0:
                    continue
                destabilizing = -(current_A * current_A) * dR_dT[index] - G
                if destabilizing <= 0.0:
                    continue
                for Cth_pJ_per_K in (0.05, 0.2, 1.0, 5.0, 20.0):
                    Cth = Cth_pJ_per_K * 1e-12
                    Ccrit = Cth / (R[index] * destabilizing)
                    for c_factor in (1.15, 1.5, 2.5):
                        C_pF = c_factor * Ccrit * 1e12
                        if not 5.0 <= C_pF <= 1500.0:
                            continue
                        C = C_pF * 1e-12
                        j11 = -1.0 / (C * R[index])
                        j12 = current_A * dR_dT[index] / (C * R[index])
                        j21 = 2.0 * current_A / Cth
                        j22 = (-(current_A * current_A) * dR_dT[index] - G) / Cth
                        trace = j11 + j22
                        det = j11 * j22 - j12 * j21
                        disc = trace * trace - 4.0 * det
                        if trace <= 0.0 or det <= 0.0 or disc >= 0.0:
                            continue
                        freq_MHz = math.sqrt(-disc) / (4.0 * math.pi) * 1e-6
                        growth_ns = 2.0e9 / trace
                        if not 0.05 <= freq_MHz <= 200.0 or growth_ns > 3000.0:
                            continue
                        candidates.append(
                            {
                                "target_T_K": float(T[index]),
                                "T0_K": T0,
                                "target_R_ohm": float(R[index]),
                                "dR_dT_ohm_per_K": float(dR_dT[index]),
                                "current_uA": current_uA,
                                "S_e_mW_per_K": G_mW_per_K,
                                "C_th_pJ_per_K": Cth_pJ_per_K,
                                "C_pF": C_pF,
                                "C_over_Ccrit": c_factor,
                                "linear_trace_per_s": trace,
                                "linear_det_per_s2": det,
                                "predicted_freq_MHz": freq_MHz,
                                "predicted_growth_ns": growth_ns,
                            }
                        )

    candidates.sort(key=lambda x: (x["predicted_growth_ns"], -x["predicted_freq_MHz"]))
    if len(candidates) > args.max_candidates:
        # Preserve coverage across the ranked instability candidates.
        selection = np.linspace(0, len(candidates) - 1, args.max_candidates, dtype=int)
        candidates = [candidates[i] for i in selection]

    rows: list[dict[str, float]] = []
    for index, candidate in enumerate(candidates):
        tau_e_ns = candidate["C_pF"] * resist.Rm * 1e-3
        tau_th_ns = candidate["C_th_pJ_per_K"] / candidate["S_e_mW_per_K"]
        dt_ns = max(0.002, min(0.025, tau_e_ns / 30.0, tau_th_ns / 100.0))
        predicted_period_ns = 1000.0 / candidate["predicted_freq_MHz"]
        t_end_ns = min(3000.0, max(700.0, 18.0 * predicted_period_ns, 8.0 * candidate["predicted_growth_ns"]))
        n_steps = int(t_end_ns / dt_ns)
        if n_steps > 500_000:
            dt_ns = t_end_ns / 500_000.0

        params = CurrentDriveParams(
            dt_s=dt_ns * 1e-9,
            t_end_s=t_end_ns * 1e-9,
            T0_K=candidate["T0_K"],
            T_init_K=candidate["T0_K"],
            C_F=candidate["C_pF"] * 1e-12,
            C_th_J_per_K=candidate["C_th_pJ_per_K"] * 1e-12,
            S_e_W_per_K=candidate["S_e_mW_per_K"] * 1e-3,
            phase_mode="quasistatic",
            resist_params=resist,
            start_branch="insulator",
        )
        out = simulate_current_step(candidate["current_uA"], params=params, seed=0)
        late = _late_trace(out)
        metrics = analyze_current_trace(
            late,
            params=params,
            min_vpp_mV=5.0,
            max_vpp_mV=2000.0,
            min_cycles=3,
            pulse_on_ns=float(late["t"][0] * 1e9),
            pulse_off_ns=None,
        )
        row = {
            "candidate": index,
            **candidate,
            "dt_ns": dt_ns,
            "t_end_ns": t_end_ns,
            **{key: float(value) for key, value in metrics.items()},
        }
        rows.append(row)
        if metrics["oscillatory"] > 0.5 or metrics["n_cycles"] >= 2.0:
            np.savez_compressed(output_dir / f"candidate_{index:03d}.npz", **out)
        if (index + 1) % 10 == 0:
            accepted = sum(row["oscillatory"] > 0.5 for row in rows)
            print(f"{index + 1}/{len(candidates)} accepted={accepted}", flush=True)
            pd.DataFrame(rows).to_csv(output_dir / "checkpoint.csv", index=False)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["oscillatory", "n_cycles", "trace_score"], ascending=False)
    result.to_csv(output_dir / "screen.csv", index=False)
    summary = {
        "sample_path": args.sample,
        "model": "corrected original quasistatic ideal-current model",
        "selection_method": "linearized unstable-focus targets on fixed fitted heating branch",
        "n_candidates": len(rows),
        "n_accepted": int(sum(row["oscillatory"] > 0.5 for row in rows)),
        "best": {} if result.empty else result.iloc[0].to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
