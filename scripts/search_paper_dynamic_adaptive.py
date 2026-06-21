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


def _energy_pj(out: dict[str, np.ndarray], frequency_mhz: float) -> float:
    if frequency_mhz <= 0.0:
        return float("nan")
    late = out["t"] >= 0.60 * out["t"][-1]
    return float(np.mean(out["P"][late]) / (frequency_mhz * 1e6) * 1e12)


def _simulate(
    *,
    current_uA: float,
    resist,
    c_pf: float,
    cth_pj: float,
    se_mw: float,
    tau_ns: float,
    t0_k: float,
    dt_ns: float,
    duration_ns: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    params = CurrentDriveParams(
        dt_s=dt_ns * 1e-9,
        t_end_s=duration_ns * 1e-9,
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
    out = simulate_current_step(current_uA, params=params, seed=0)
    metrics = analyze_current_trace(
        out,
        params=params,
        min_vpp_mV=8.0,
        max_vpp_mV=2_000.0,
        min_cycles=6,
        pulse_on_ns=0.0,
        pulse_off_ns=None,
    )
    metrics["energy_pJ"] = _energy_pj(out, float(metrics["dominant_freq_MHz"]))
    late = out["t"] >= 0.60 * out["t"][-1]
    metrics["late_V_pp_mV"] = float(np.ptp(out["V_vo2"][late]) * 1e3)
    metrics["late_T_pp_K"] = float(np.ptp(out["T"][late]))
    metrics["late_g_pp"] = float(np.ptp(out["g_dyn"][late]))
    return out, metrics


def _paper_score(metrics: dict[str, float]) -> float:
    freq = float(metrics["dominant_freq_MHz"])
    energy = float(metrics["energy_pJ"])
    freq_score = math.exp(-0.5 * ((freq - 50.0) / 15.0) ** 2) if freq > 0.0 else 0.0
    energy_score = math.exp(-0.5 * ((energy - 1.55) / 0.7) ** 2) if np.isfinite(energy) else 0.0
    persistence = min(float(metrics["late_to_early_vpp"]), 1.0)
    return float(
        0.35 * float(metrics["oscillatory"])
        + 0.25 * float(metrics["trace_score"])
        + 0.20 * freq_score
        + 0.10 * energy_score
        + 0.10 * persistence
    )


def _candidate_rows(rng: np.random.Generator, n_random: int, tc_k: float):
    seeds = [
        (80.0, 3.0, 0.052838, 10.0, 325.0),
        (80.0, 2.0, 0.032844, 10.0, 325.0),
        (120.0, 3.0, 0.036205, 5.0, 325.0),
        (24.0, 0.08, 0.0035, 1.5, min(330.0, tc_k - 0.5)),
        (40.0, 0.20, 0.0040, 3.0, min(330.0, tc_k - 0.5)),
    ]
    yield from seeds
    for _ in range(n_random):
        yield (
            _loguniform(rng, 15.0, 250.0),
            _loguniform(rng, 0.005, 8.0),
            _loguniform(rng, 0.00005, 0.15),
            _loguniform(rng, 0.30, 30.0),
            float(rng.uniform(300.0, min(tc_k + 2.0, 344.0))),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-random", type=int, default=240)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--screen-current-uA", type=float, default=400.0)
    args = parser.parse_args()

    sample = load_sample_json(args.sample)
    resist = params_from_dict(sample["resist_params"])
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, float]] = []
    candidates = list(_candidate_rows(rng, args.n_random, resist.Tc_K))
    for index, (c_pf, cth_pj, se_mw, tau_ns, t0_k) in enumerate(candidates):
        dt_ns = min(0.025, 0.06 * tau_ns, 0.025 * c_pf * resist.Rm * 1e-3)
        dt_ns = max(dt_ns, 0.008)
        _, metrics = _simulate(
            current_uA=args.screen_current_uA,
            resist=resist,
            c_pf=c_pf,
            cth_pj=cth_pj,
            se_mw=se_mw,
            tau_ns=tau_ns,
            t0_k=t0_k,
            dt_ns=dt_ns,
            duration_ns=700.0,
        )
        row = {
            "candidate": index,
            "C_pF": c_pf,
            "C_th_pJ_per_K": cth_pj,
            "S_e_mW_per_K": se_mw,
            "tau_g_ns": tau_ns,
            "T0_K": t0_k,
            "dt_ns": dt_ns,
            **{str(k): float(v) for k, v in metrics.items()},
        }
        row["paper_score"] = _paper_score(metrics)
        rows.append(row)
        if (index + 1) % 20 == 0:
            accepted = sum(r["oscillatory"] > 0.5 for r in rows)
            print(f"{index + 1}/{len(candidates)} accepted={accepted} best={max(r['paper_score'] for r in rows):.4f}", flush=True)
            pd.DataFrame(rows).to_csv(output_dir / "screen_checkpoint.csv", index=False)

    screen = pd.DataFrame(rows).sort_values(["oscillatory", "paper_score"], ascending=False)
    screen.to_csv(output_dir / "screen_400uA.csv", index=False)

    finalists = screen[screen["oscillatory"] > 0.5].head(8)
    if finalists.empty:
        finalists = screen.head(6)
    validation_rows: list[dict[str, float]] = []
    for _, candidate in finalists.iterrows():
        for current_uA in (200.0, 300.0, 400.0, 500.0, 600.0):
            for dt_factor, duration_ns in ((1.0, 1_600.0), (0.5, 1_600.0)):
                _, metrics = _simulate(
                    current_uA=current_uA,
                    resist=resist,
                    c_pf=float(candidate["C_pF"]),
                    cth_pj=float(candidate["C_th_pJ_per_K"]),
                    se_mw=float(candidate["S_e_mW_per_K"]),
                    tau_ns=float(candidate["tau_g_ns"]),
                    t0_k=float(candidate["T0_K"]),
                    dt_ns=float(candidate["dt_ns"]) * dt_factor,
                    duration_ns=duration_ns,
                )
                validation_rows.append(
                    {
                        "candidate": int(candidate["candidate"]),
                        "I_target_uA": current_uA,
                        "dt_factor": dt_factor,
                        "duration_ns": duration_ns,
                        "C_pF": float(candidate["C_pF"]),
                        "C_th_pJ_per_K": float(candidate["C_th_pJ_per_K"]),
                        "S_e_mW_per_K": float(candidate["S_e_mW_per_K"]),
                        "tau_g_ns": float(candidate["tau_g_ns"]),
                        "T0_K": float(candidate["T0_K"]),
                        "dt_ns": float(candidate["dt_ns"]) * dt_factor,
                        **{str(k): float(v) for k, v in metrics.items()},
                        "paper_score": _paper_score(metrics),
                    }
                )
    validation = pd.DataFrame(validation_rows).sort_values(
        ["oscillatory", "paper_score"], ascending=False
    )
    validation.to_csv(output_dir / "validation.csv", index=False)
    summary = {
        "sample": sample,
        "model": "dynamic phase extension",
        "paper_targets": {
            "current_window_uA": [200.0, 600.0],
            "frequency_MHz": [40.0, 60.0],
            "energy_pJ_per_cycle": [1.2, 1.9],
        },
        "screen_accepted": int((screen["oscillatory"] > 0.5).sum()),
        "validated_accepted": int((validation["oscillatory"] > 0.5).sum()),
        "best_validation": validation.iloc[0].to_dict() if not validation.empty else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
