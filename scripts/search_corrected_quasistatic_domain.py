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

from neuristor.current_domain_search import analyze_current_trace
from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step
from neuristor.sample_library import load_sample_json, params_from_dict


def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(10.0 ** rng.uniform(math.log10(low), math.log10(high)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-random", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    sample = load_sample_json(args.sample)
    resist = params_from_dict(sample["resist_params"])
    rng = np.random.default_rng(args.seed)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [
        (80.0, 3.0, 0.052838, 325.0),
        (145.0, 49.6, 0.2056, 325.0),
        (20.0, 0.2, 0.005, min(resist.Tc_K - 2.0, 335.0)),
        (200.0, 10.0, 0.02, min(resist.Tc_K - 5.0, 330.0)),
    ]
    candidates = list(seeds)
    for _ in range(args.n_random):
        candidates.append(
            (
                _loguniform(rng, 8.0, 600.0),
                _loguniform(rng, 0.02, 150.0),
                _loguniform(rng, 0.0002, 1.0),
                float(rng.uniform(295.0, min(resist.Tc_K + 3.0, 347.0))),
            )
        )

    rows: list[dict[str, float]] = []
    for index, (c_pf, cth_pj, se_mw, t0_k) in enumerate(candidates):
        dt_ns = min(0.025, 0.025 * c_pf * resist.Rm * 1e-3)
        dt_ns = max(dt_ns, 0.005)
        broad_current = float(rng.uniform(650.0, 3500.0))
        currents = [250.0, 400.0, 550.0, broad_current]
        row: dict[str, float] = {
            "candidate": index,
            "C_pF": c_pf,
            "C_th_pJ_per_K": cth_pj,
            "S_e_mW_per_K": se_mw,
            "T0_K": t0_k,
            "dt_ns": dt_ns,
            "broad_current_uA": broad_current,
        }
        best_score = 0.0
        accepted = 0
        for current_uA in currents:
            params = CurrentDriveParams(
                dt_s=dt_ns * 1e-9,
                t_end_s=1_000e-9,
                T0_K=t0_k,
                T_init_K=t0_k,
                C_F=c_pf * 1e-12,
                C_th_J_per_K=cth_pj * 1e-12,
                S_e_W_per_K=se_mw * 1e-3,
                phase_mode="quasistatic",
                resist_params=resist,
                start_branch="insulator",
            )
            out = simulate_current_step(current_uA, params=params, seed=0)
            late = out["t"] >= 0.50 * out["t"][-1]
            late_out = {
                key: value[late]
                for key, value in out.items()
                if isinstance(value, np.ndarray) and value.shape[0] == late.shape[0]
            }
            metrics = analyze_current_trace(
                late_out,
                params=params,
                min_vpp_mV=10.0,
                max_vpp_mV=2_000.0,
                min_cycles=4,
                pulse_on_ns=float(late_out["t"][0] * 1e9),
                pulse_off_ns=None,
            )
            tag = f"I{int(round(current_uA))}"
            for key in ("oscillatory", "trace_score", "n_cycles", "V_pp_mV", "dominant_freq_MHz", "period_cv"):
                row[f"{tag}_{key}"] = float(metrics[key])
            best_score = max(best_score, float(metrics["trace_score"]))
            accepted += int(metrics["oscillatory"] > 0.5)
        row["best_late_score"] = best_score
        row["n_accepted_currents"] = accepted
        rows.append(row)
        if (index + 1) % 10 == 0:
            print(
                f"{index + 1}/{len(candidates)} accepted_candidates="
                f"{sum(r['n_accepted_currents'] > 0 for r in rows)} best={max(r['best_late_score'] for r in rows):.4f}",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(output_dir / "checkpoint.csv", index=False)

    result = pd.DataFrame(rows).sort_values(["n_accepted_currents", "best_late_score"], ascending=False)
    result.to_csv(output_dir / "screen.csv", index=False)
    summary = {
        "sample": sample,
        "model": "corrected original quasistatic ideal-current model",
        "n_candidates": len(candidates),
        "n_candidates_with_accepted_current": int((result["n_accepted_currents"] > 0).sum()),
        "best": result.iloc[0].to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
