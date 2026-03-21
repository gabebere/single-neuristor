from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from current_domain_search import DomainSearchConfig, SearchRange, save_search_results, search_current_domain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search oscillatory domains for the ideal current-driven VO2 model.")
    parser.add_argument(
        "--preset",
        default="presets/resistance_100425_chip1_gap3.json",
        help="Resistance preset JSON to use as the sample-specific hysteresis law.",
    )
    parser.add_argument("--n-random", type=int, default=18, help="Number of random candidates before local refinement.")
    parser.add_argument("--top-k", type=int, default=4, help="How many top candidates to refine locally.")
    parser.add_argument("--refine", type=int, default=4, help="Refinement samples per top candidate.")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for candidate sampling.")
    parser.add_argument("--current-start", type=int, default=50, help="Start of current sweep in uA.")
    parser.add_argument("--current-stop", type=int, default=2000, help="End of current sweep in uA.")
    parser.add_argument("--coarse-step", type=int, default=100, help="Coarse current step in uA.")
    parser.add_argument("--fine-step", type=int, default=25, help="Refinement current step in uA.")
    parser.add_argument("--fine-window", type=int, default=200, help="Half-window around best coarse current in uA.")
    parser.add_argument("--t-end-ns", type=float, default=600.0, help="Simulation duration after the step.")
    parser.add_argument("--t-pre-ns", type=float, default=0.0, help="Pre-step duration.")
    parser.add_argument("--pulse-off-ns", type=float, default=None, help="Optional pulse-off time; omit for constant current.")
    parser.add_argument("--min-T0-K", type=float, default=298.0, help="Minimum ambient/base temperature.")
    parser.add_argument("--max-T0-K", type=float, default=298.0, help="Maximum ambient/base temperature.")
    parser.add_argument("--min-c-pF", type=float, default=5.0)
    parser.add_argument("--max-c-pF", type=float, default=500.0)
    parser.add_argument("--min-cth", type=float, default=2.0, help="Minimum thermal capacitance in mW*ns/K.")
    parser.add_argument("--max-cth", type=float, default=500.0, help="Maximum thermal capacitance in mW*ns/K.")
    parser.add_argument("--min-se", type=float, default=0.02, help="Minimum S_e in mW/K.")
    parser.add_argument("--max-se", type=float, default=5.0, help="Maximum S_e in mW/K.")
    parser.add_argument("--min-tinit", type=float, default=298.0, help="Minimum initial temperature in K.")
    parser.add_argument("--max-tinit", type=float, default=338.0, help="Maximum initial temperature in K.")
    parser.add_argument("--min-sigma", type=float, default=0.0, help="Minimum thermal-noise sigma in W*sqrt(s).")
    parser.add_argument("--max-sigma", type=float, default=0.0, help="Maximum thermal-noise sigma in W*sqrt(s).")
    parser.add_argument("--min-rm-scale", type=float, default=1.0, help="Minimum multiplicative scale on fitted Rm_factor.")
    parser.add_argument("--max-rm-scale", type=float, default=1.0, help="Maximum multiplicative scale on fitted Rm_factor.")
    parser.add_argument("--min-tc-shift", type=float, default=0.0, help="Minimum additive shift on fitted Tc_K.")
    parser.add_argument("--max-tc-shift", type=float, default=0.0, help="Maximum additive shift on fitted Tc_K.")
    parser.add_argument("--min-w-scale", type=float, default=1.0, help="Minimum multiplicative scale on fitted hysteresis width w.")
    parser.add_argument("--max-w-scale", type=float, default=1.0, help="Maximum multiplicative scale on fitted hysteresis width w.")
    parser.add_argument("--min-beta-scale", type=float, default=1.0, help="Minimum multiplicative scale on fitted beta.")
    parser.add_argument("--max-beta-scale", type=float, default=1.0, help="Maximum multiplicative scale on fitted beta.")
    parser.add_argument(
        "--min-reversal-threshold",
        type=float,
        default=0.01,
        help="Minimum hysteresis reversal threshold in K.",
    )
    parser.add_argument(
        "--max-reversal-threshold",
        type=float,
        default=0.01,
        help="Maximum hysteresis reversal threshold in K.",
    )
    parser.add_argument("--output-dir", default="", help="Optional output directory. Defaults to outputs/current_domain_search_<timestamp>.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = DomainSearchConfig(
        resistance_preset_path=args.preset,
        current_start_uA=args.current_start,
        current_stop_uA=args.current_stop,
        coarse_current_step_uA=args.coarse_step,
        refine_current_step_uA=args.fine_step,
        refine_half_window_uA=args.fine_window,
        t_end_ns=args.t_end_ns,
        t_pre_ns=args.t_pre_ns,
        pulse_off_ns=args.pulse_off_ns,
        n_random_candidates=args.n_random,
        top_k_for_refine=args.top_k,
        refine_samples_per_top=args.refine,
        seed=args.seed,
        t0_K=SearchRange(args.min_T0_K, args.max_T0_K, "linear"),
        c_pF=SearchRange(args.min_c_pF, args.max_c_pF, "log"),
        c_th_mW_ns_per_K=SearchRange(args.min_cth, args.max_cth, "log"),
        s_e_mW_per_K=SearchRange(args.min_se, args.max_se, "log"),
        t_init_K=SearchRange(args.min_tinit, args.max_tinit, "linear"),
        sigma_W_sqrt_s=SearchRange(args.min_sigma, args.max_sigma, "linear"),
        rm_factor_scale=SearchRange(args.min_rm_scale, args.max_rm_scale, "log"),
        tc_shift_K=SearchRange(args.min_tc_shift, args.max_tc_shift, "linear"),
        w_scale=SearchRange(args.min_w_scale, args.max_w_scale, "linear"),
        beta_scale=SearchRange(args.min_beta_scale, args.max_beta_scale, "linear"),
        reversal_threshold_K=SearchRange(args.min_reversal_threshold, args.max_reversal_threshold, "log"),
    )

    results = search_current_domain(cfg)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"current_domain_search_{ts}"
    save_search_results(results, output_dir)

    summary_df = results["summary_df"]
    print(f"Saved search results to: {output_dir}")
    if summary_df.empty:
        print("No candidates were evaluated.")
        return

    best = summary_df.iloc[0].to_dict()
    print("Top candidate:")
    print(json.dumps(best, indent=2))
    print()
    if "target_score" in summary_df.columns:
        best_target = summary_df.sort_values(["target_score", "candidate_score"], ascending=[False, False]).iloc[0].to_dict()
        print("Best target-match candidate:")
        print(json.dumps(best_target, indent=2))
        print()
    cols = [
        "candidate_score",
        "target_score",
        "generic_score",
        "best_trace_score",
        "n_good_currents",
        "first_good_uA",
        "last_good_uA",
        "best_current_uA",
        "best_turn_count",
        "best_vpp_mV",
        "best_freq_MHz",
        "plateau_mean_mV_target",
        "plateau_std_mV_target",
        "low_current_peak_mV",
        "low_current_peak_uA",
        "low_current_bump_mV",
        "rep_plateau_mV",
        "rep_overshoot_mV",
        "rep_turnoff_undershoot_mV",
        "rep_plateau_vpp_mV",
        "rep_plateau_turn_count",
        "rep_plateau_ring_freq_MHz",
        "rep_plateau_damping_ratio",
        "rep_plateau_underdamped",
        "C_pF",
        "C_th_mW_ns_per_K",
        "S_e_mW_per_K",
        "T0_K",
        "T_init_K",
        "sigma_W_sqrt_s",
        "Rm_factor_scale",
        "Tc_shift_K",
        "w_scale",
        "beta_scale",
        "Rm_eff_ohm",
        "Tc_eff_K",
        "w_eff_K",
        "beta_eff_per_K",
        "reversal_threshold_K",
        "dt_ns",
        "feasible",
    ]
    cols = [c for c in cols if c in summary_df.columns]
    print("Top 10 candidates:")
    print(summary_df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
