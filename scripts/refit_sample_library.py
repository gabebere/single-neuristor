from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import neuristor.model as model
from neuristor.resistance_custom_analysis import fit_resistance_params
from neuristor.sample_library import (
    compute_rt_fit_metrics,
    list_samples,
    load_experimental_rt_path,
    normalize_sample_payload,
    save_sample,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refit and validate every measured R-T sample.")
    parser.add_argument("--random-iters", type=int, default=1000)
    parser.add_argument("--local-passes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply", action="store_true", help="Update samples whose exact validated fit improves.")
    parser.add_argument("--output-dir", default="outputs/sample_refit_audit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    # NumPy is much faster for optimization. Every candidate is validated again
    # through the normal app evaluator before it can replace a saved sample.
    torch_available = model._TORCH_HYSTERESIS_AVAILABLE
    try:
        for index, sample in enumerate(list_samples()):
            source_path = Path(str(sample.get("source_path", "")))
            if not source_path.is_file():
                continue
            df = load_experimental_rt_path(source_path)
            old_metrics, _, _ = compute_rt_fit_metrics(
                df,
                model.YuanhangResistParams(**sample["resist_params"]),
                start_branch=str(sample["start_branch"]),
            )
            fit, _ = fit_resistance_params(
                df,
                seed=int(args.seed) + index,
                random_iters=int(args.random_iters),
                local_passes=int(args.local_passes),
                fit_gamma=True,
                g_weight=0.2,
                high_res_weight=0.65,
            )
            model._TORCH_HYSTERESIS_AVAILABLE = torch_available
            new_metrics, _, _ = compute_rt_fit_metrics(df, fit.params, start_branch=fit.start_branch)

            improved = float(new_metrics["rmse_log10"]) < float(old_metrics["rmse_log10"])
            accepted = improved and float(new_metrics["max_abs_log10_error"]) <= 0.5
            row = {
                "sample_id": sample["sample_id"],
                "display_name": sample["display_name"],
                "old_rmse_log10": old_metrics["rmse_log10"],
                "new_rmse_log10": new_metrics["rmse_log10"],
                "old_r2_log10": old_metrics["r2_log10"],
                "new_r2_log10": new_metrics["r2_log10"],
                "old_max_abs_log10_error": old_metrics["max_abs_log10_error"],
                "new_max_abs_log10_error": new_metrics["max_abs_log10_error"],
                "accepted": accepted,
                "applied": bool(accepted and args.apply),
            }
            rows.append(row)
            print(json.dumps(row))

            candidate = normalize_sample_payload(sample)
            candidate.pop("_path", None)
            candidate.pop("_legacy", None)
            candidate["start_branch"] = fit.start_branch
            candidate["fit_metrics"] = new_metrics
            candidate["resist_params"] = {
                field: float(getattr(fit.params, field))
                for field in model.YuanhangResistParams.__dataclass_fields__
            }
            (output_dir / f"{sample['sample_id']}_candidate.json").write_text(json.dumps(candidate, indent=2))
            if accepted and args.apply:
                save_sample(candidate)
    finally:
        model._TORCH_HYSTERESIS_AVAILABLE = torch_available

    pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
