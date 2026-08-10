from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import neuristor.model as model
from neuristor.sample_library import (
    compute_rt_fit_metrics,
    get_sample,
    load_experimental_rt_path,
    normalize_sample_payload,
    params_from_dict,
    params_to_dict,
    predict_resistance_trace,
    save_sample,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Top-aware manual-style refit using the unchanged R-T model.")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=55)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output-dir", default="outputs/manual_top_aware_refits")
    return parser


def _masked_rmse(error: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error[mask] ** 2))) if np.any(mask) else float("nan")


def main() -> None:
    args = build_parser().parse_args()
    sample = get_sample(args.sample_id)
    if sample is None:
        raise ValueError(f"Unknown sample: {args.sample_id}")
    source_path = Path(str(sample.get("source_path", "")))
    if not source_path.is_file():
        raise ValueError(f"Sample has no measured R-T file: {args.sample_id}")

    output_dir = ROOT / args.output_dir / args.sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_experimental_rt_path(source_path)
    t = df["Temperature"].to_numpy(dtype=float)
    r = np.maximum(df["Resistance"].to_numpy(dtype=float), 1e-12)
    log_r = np.log10(r)
    old_params = params_from_dict(sample["resist_params"])
    branch = str(sample["start_branch"])

    # The upper/insulating branch is the high-resistance half of the measured
    # trajectory. The transition mask avoids overfitting only the plateaus.
    top_mask = r >= float(np.quantile(r, 0.50))
    lo_log, hi_log = float(np.min(log_r)), float(np.max(log_r))
    transition_mask = (log_r >= lo_log + 0.8) & (log_r <= hi_log - 0.45)

    def params_from_x(x: np.ndarray) -> model.YuanhangResistParams:
        p = params_from_dict(sample["resist_params"])
        p.R0 = float(10.0 ** x[0])
        p.Ea_over_k = float(x[1])
        p.Rm0 = float(10.0 ** x[2])
        p.Rm_factor = 1.0
        p.w = float(x[3])
        p.Tc_K = float(x[4])
        p.beta = float(10.0 ** x[5])
        return p

    def objective(x: np.ndarray) -> float:
        pred, _ = predict_resistance_trace(t, params_from_x(x), start_branch=branch)
        error = np.log10(np.maximum(pred, 1e-12)) - log_r
        weights = np.ones_like(error)
        weights[top_mask] *= 4.0
        weights[transition_mask] *= 2.0
        weighted_rmse = float(np.sqrt(np.sum(weights * error * error) / np.sum(weights)))
        return weighted_rmse + 0.08 * float(np.max(np.abs(error)))

    bounds = [
        (-5.0, 1.3),
        (1200.0, 9000.0),
        (0.7, 2.7),
        (2.0, 20.0),
        (float(np.min(t)) + 8.0, float(np.max(t)) - 8.0),
        (-2.0, 0.6),
    ]
    torch_available = model._TORCH_HYSTERESIS_AVAILABLE
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=int(args.seed),
            popsize=int(args.popsize),
            maxiter=int(args.maxiter),
            polish=True,
            workers=1,
        )
        new_params = params_from_x(result.x)
        old_metrics, old_pred, _ = compute_rt_fit_metrics(df, old_params, start_branch=branch)
        new_metrics, new_pred, _ = compute_rt_fit_metrics(df, new_params, start_branch=branch)
    finally:
        model._TORCH_HYSTERESIS_AVAILABLE = torch_available

    old_error = np.log10(np.maximum(old_pred, 1e-12)) - log_r
    new_error = np.log10(np.maximum(new_pred, 1e-12)) - log_r
    old_extra = {
        "top_rmse_log10": _masked_rmse(old_error, top_mask),
        "transition_rmse_log10": _masked_rmse(old_error, transition_mask),
    }
    new_extra = {
        "top_rmse_log10": _masked_rmse(new_error, top_mask),
        "transition_rmse_log10": _masked_rmse(new_error, transition_mask),
    }
    max_limit = max(0.40, 1.30 * float(old_metrics["max_abs_log10_error"]))
    accepted = (
        float(new_metrics["rmse_log10"]) < float(old_metrics["rmse_log10"])
        and new_extra["top_rmse_log10"] < old_extra["top_rmse_log10"]
        and new_extra["transition_rmse_log10"] <= 1.15 * old_extra["transition_rmse_log10"]
        and float(new_metrics["max_abs_log10_error"]) <= max_limit
    )

    payload = normalize_sample_payload(sample)
    payload.pop("_path", None)
    payload.pop("_legacy", None)
    payload["resist_params"] = params_to_dict(new_params)
    payload["fit_metrics"] = new_metrics
    payload["notes"] = (
        str(payload.get("notes", "")).rstrip()
        + "\nTop-aware manual-style refit using the unchanged Yuanhang R-T model."
    ).strip()
    (output_dir / "candidate.json").write_text(json.dumps(payload, indent=2))
    if accepted and args.apply:
        save_sample(payload)

    summary = {
        "sample_id": args.sample_id,
        "display_name": sample["display_name"],
        "accepted": bool(accepted),
        "applied": bool(accepted and args.apply),
        "old_metrics": {**old_metrics, **old_extra},
        "new_metrics": {**new_metrics, **new_extra},
        "old_params": params_to_dict(old_params),
        "new_params": params_to_dict(new_params),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    d_t = np.diff(t, prepend=t[0])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, pred, title in (
        (axes[0], old_pred, "Before"),
        (axes[1], new_pred, "Top-aware candidate"),
    ):
        ax.semilogy(t[d_t < 0], r[d_t < 0], ".", ms=3, label="cooling")
        ax.semilogy(t[d_t > 0], r[d_t > 0], ".", ms=3, label="heating")
        ax.semilogy(t, pred, "k-", linewidth=1.5, label="model")
        ax.set_title(title)
        ax.set_xlabel("Temperature (K)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Resistance (Ohm)")
    axes[1].legend()
    fig.suptitle(f"{sample['display_name']} | accepted={accepted}")
    fig.tight_layout()
    fig.savefig(output_dir / "before_after.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
