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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search the paper-targeted dynamic current-drive domain.")
    parser.add_argument("--sample", required=True, help="Sample preset JSON.")
    parser.add_argument("--n-random", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    return parser


def _energy_pj(out: dict[str, np.ndarray], frequency_mhz: float, late_mask: np.ndarray) -> float:
    if frequency_mhz <= 0.0:
        return float("nan")
    mean_power_w = float(np.mean(out["P"][late_mask]))
    return float(mean_power_w / (frequency_mhz * 1e6) * 1e12)


def main() -> None:
    args = build_parser().parse_args()
    sample = load_sample_json(args.sample)
    resist = params_from_dict(sample["resist_params"])
    rng = np.random.default_rng(int(args.seed))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    currents = [250.0, 400.0, 550.0]
    seeds = [
        (18.0, 0.08, 0.0040, 1.5, 325.0),
        (25.0, 0.12, 0.0060, 2.0, 325.0),
        (12.0, 0.05, 0.0025, 0.8, 325.0),
    ]
    candidates = list(seeds)
    for _ in range(int(args.n_random)):
        candidates.append(
            (
                _loguniform(rng, 8.0, 50.0),
                _loguniform(rng, 0.02, 0.8),
                _loguniform(rng, 0.001, 0.03),
                _loguniform(rng, 0.2, 6.0),
                float(rng.uniform(315.0, min(334.0, resist.Tc_K))),
            )
        )

    for index, (c_pf, cth_pj, se_mw, tau_ns, t0_k) in enumerate(candidates):
        dt_ns = min(0.01, 0.025 * c_pf * resist.Rm * 1e-3, 0.025 * tau_ns)
        dt_ns = max(dt_ns, 0.002)
        params = CurrentDriveParams(
            dt_s=dt_ns * 1e-9,
            t_end_s=500e-9,
            T0_K=t0_k,
            T_init_K=t0_k,
            C_F=c_pf * 1e-12,
            C_th_J_per_K=cth_pj * 1e-12,
            S_e_W_per_K=se_mw * 1e-3,
            phase_mode="dynamic",
            tau_g_s=tau_ns * 1e-9,
            resist_params=resist,
            start_branch="insulator",
        )
        row: dict[str, float] = {
            "candidate": float(index),
            "C_pF": c_pf,
            "C_th_pJ_per_K": cth_pj,
            "S_e_mW_per_K": se_mw,
            "tau_g_ns": tau_ns,
            "T0_K": t0_k,
            "dt_ns": dt_ns,
        }
        scores: list[float] = []
        for current in currents:
            out = simulate_current_step(current, params=params, seed=0)
            late_mask = out["t"] >= 250e-9
            metrics = analyze_current_trace(
                out,
                params=params,
                min_vpp_mV=12.0,
                max_vpp_mV=1500.0,
                min_cycles=5,
                pulse_on_ns=0.0,
                pulse_off_ns=None,
            )
            freq = float(metrics["dominant_freq_MHz"])
            energy = _energy_pj(out, freq, late_mask)
            frequency_score = math.exp(-0.5 * ((freq - 50.0) / 15.0) ** 2) if freq > 0.0 else 0.0
            energy_score = math.exp(-0.5 * ((energy - 1.5) / 1.0) ** 2) if np.isfinite(energy) else 0.0
            score = (
                0.40 * float(metrics["trace_score"])
                + 0.30 * float(metrics["oscillatory"])
                + 0.20 * frequency_score
                + 0.10 * energy_score
            )
            scores.append(score)
            for key in (
                "oscillatory",
                "trace_score",
                "V_pp_mV",
                "n_cycles",
                "dominant_freq_MHz",
                "period_cv",
                "spectral_purity",
                "late_to_early_vpp",
                "plateau_mean_mV",
            ):
                row[f"I{int(current)}_{key}"] = float(metrics[key])
            row[f"I{int(current)}_energy_pJ"] = energy
        row["paper_score"] = float(np.mean(scores))
        row["n_oscillatory_currents"] = float(
            sum(row[f"I{int(current)}_oscillatory"] > 0.5 for current in currents)
        )
        rows.append(row)
        if (index + 1) % 10 == 0:
            print(f"{index + 1}/{len(candidates)} best={max(r['paper_score'] for r in rows):.4f}", flush=True)

    result = pd.DataFrame(rows).sort_values(
        ["paper_score", "n_oscillatory_currents"], ascending=[False, False]
    )
    result.to_csv(output_dir / "screen.csv", index=False)
    best = result.iloc[0].to_dict()
    (output_dir / "best_candidate.json").write_text(
        json.dumps(
            {
                "sample": sample,
                "candidate": best,
                "model": "dynamic phase extension",
                "paper_targets": {
                    "current_window_uA": [200.0, 600.0],
                    "frequency_MHz": [40.0, 60.0],
                    "energy_pJ_per_cycle": [1.2, 1.9],
                },
            },
            indent=2,
        )
    )
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
