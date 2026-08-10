"""Estimate identifiable current-drive parameters from digitized lab traces.

This utility intentionally separates direct estimates from assumptions:

* C is estimated from the cold, early-time slope C ~= I/(dV/dt).
* S_e is evaluated from S_e ~= P_switch/(T_switch-T0) for candidate T0.
* C_th is reported as scenarios C_th=S_e*tau_th because voltage screenshots
  alone do not independently measure the thermal recovery time tau_th.
* Effective plateau resistance V/I is reported to diagnose the nonzero voltage
  floor and whether a fixed metallic resistance can explain it.

Raw oscilloscope CSVs should replace the digitized screenshot summary when
available.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuristor.current_results_digitizer import digitize_directory
from neuristor.model import YuanhangResistParams


DEFAULT_IMAGE_DIR = ROOT / "data" / "Current Results"
DEFAULT_SAMPLE = ROOT / "presets" / "resistance_100425_chip1_gap3.json"
DEFAULT_PRIOR_FIT = ROOT / "presets" / "current_drive" / "lab_image_fit_20260323.json"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_resistance(path: Path) -> YuanhangResistParams:
    payload = json.loads(path.read_text())
    return YuanhangResistParams(**payload["resist_params"])


def _find_switching_bracket(summary: pd.DataFrame, *, ripple_threshold_mV: float) -> tuple[pd.Series, pd.Series]:
    ordered = summary.sort_values("current_inferred_uA").reset_index(drop=True)
    switched = np.flatnonzero(ordered["v_plateau_vpp_mV"].to_numpy(dtype=float) >= ripple_threshold_mV)
    if switched.size == 0 or int(switched[0]) == 0:
        raise ValueError("Could not bracket switching onset from plateau ripple.")
    high_idx = int(switched[0])
    return ordered.iloc[high_idx - 1], ordered.iloc[high_idx]


def _estimate_electrical_capacitance(summary: pd.DataFrame, *, onset_current_uA: float) -> pd.DataFrame:
    d = summary.copy()
    d["C_slope_pF"] = d["current_inferred_uA"] / d["v_slope_0_30_mV_per_ns"]
    valid = (
        (d["current_inferred_uA"] > 0.0)
        & (d["current_inferred_uA"] <= onset_current_uA)
        & (d["v_slope_0_30_mV_per_ns"] > 1.0)
        & (d["v_plateau_vpp_mV"] < 20.0)
        & np.isfinite(d["C_slope_pF"])
    )
    return d.loc[
        valid,
        ["frame_index", "current_inferred_uA", "v_slope_0_30_mV_per_ns", "C_slope_pF"],
    ].reset_index(drop=True)


def _thermal_conductance_scenarios(
    *,
    pre_switch: pd.Series,
    T_switch_K: float,
    ambient_values_K: list[float],
) -> pd.DataFrame:
    current_uA = float(pre_switch["current_inferred_uA"])
    voltage_mV = float(pre_switch["v_plateau_mean_mV"])
    power_mW = current_uA * voltage_mV * 1e-6
    rows = []
    for ambient_K in ambient_values_K:
        delta_T = T_switch_K - ambient_K
        rows.append(
            {
                "T0_K": float(ambient_K),
                "T_switch_K": float(T_switch_K),
                "Delta_T_K": float(delta_T),
                "I_pre_switch_uA": current_uA,
                "V_pre_switch_mV": voltage_mV,
                "P_pre_switch_mW": float(power_mW),
                "S_e_mW_per_K": float(power_mW / delta_T) if delta_T > 0.0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _thermal_capacitance_scenarios(conductance: pd.DataFrame, tau_values_ns: list[float]) -> pd.DataFrame:
    rows = []
    for item in conductance.to_dict("records"):
        for tau_ns in tau_values_ns:
            S_e = float(item["S_e_mW_per_K"])
            rows.append(
                {
                    "T0_K": float(item["T0_K"]),
                    "S_e_mW_per_K": S_e,
                    "assumed_tau_th_ns": float(tau_ns),
                    # mW/K * ns = pJ/K numerically.
                    "C_th_pJ_per_K": float(S_e * tau_ns),
                }
            )
    return pd.DataFrame(rows)


def _effective_resistance(summary: pd.DataFrame) -> pd.DataFrame:
    d = summary[
        ["frame_index", "current_inferred_uA", "v_plateau_mean_mV", "v_plateau_vpp_mV"]
    ].copy()
    d["R_effective_ohm"] = 1000.0 * d["v_plateau_mean_mV"] / d["current_inferred_uA"]
    return d


def _plot_summary(
    *,
    capacitance: pd.DataFrame,
    conductance: pd.DataFrame,
    resistance: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7))
    ax = axes[0]
    ax.scatter(capacitance["current_inferred_uA"], capacitance["C_slope_pF"], s=55, color="#7c3aed")
    median_c = float(capacitance["C_slope_pF"].median())
    ax.axhline(median_c, color="black", linestyle="--", label=f"median {median_c:.1f} pF")
    ax.set_xlabel("Current (uA)")
    ax.set_ylabel("C from I/(dV/dt) (pF)")
    ax.set_title("Cold electrical transient")
    ax.legend()

    ax = axes[1]
    ax.plot(conductance["T0_K"], conductance["S_e_mW_per_K"], "o-", color="#ea580c")
    ax.set_xlabel("Assumed ambient T0 (K)")
    ax.set_ylabel("S_e = P_switch/DeltaT (mW/K)")
    ax.set_title("Ambient/cooling degeneracy")

    ax = axes[2]
    ax.plot(resistance["current_inferred_uA"], resistance["R_effective_ohm"], "o-", color="#0891b2")
    ax.set_yscale("log")
    ax.set_xlabel("Current (uA)")
    ax.set_ylabel("Plateau V/I (Ohm)")
    ax.set_title("Measured nonzero voltage floor")
    for item in axes:
        item.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def _plot_voltage_floor_comparison(
    *,
    resistance: pd.DataFrame,
    specimen_Rm_ohm: float,
    out_path: Path,
) -> None:
    """Compare measured plateaus with ideal-current I*Rm predictions."""

    current_uA = resistance["current_inferred_uA"].to_numpy(dtype=float)
    measured_mV = resistance["v_plateau_mean_mV"].to_numpy(dtype=float)
    current_grid_uA = np.linspace(0.0, max(1400.0, float(np.max(current_uA)) * 1.03), 400)
    yuanhang_Rm_ohm = float(YuanhangResistParams().Rm)

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    ax.scatter(
        current_uA,
        measured_mV,
        s=34,
        color="#111827",
        label="Digitized lab plateau",
        zorder=5,
    )
    ax.plot(
        current_grid_uA,
        current_grid_uA * specimen_Rm_ohm / 1000.0,
        linewidth=2.2,
        color="#dc2626",
        label=f"Ideal current: fitted specimen Rm={specimen_Rm_ohm:.1f} Ohm",
    )
    ax.plot(
        current_grid_uA,
        current_grid_uA * yuanhang_Rm_ohm / 1000.0,
        linewidth=2.2,
        color="#2563eb",
        label=f"Ideal current: Yuanhang Rm={yuanhang_Rm_ohm:.0f} Ohm",
    )
    ax.plot(
        current_grid_uA,
        current_grid_uA * 250.0 / 1000.0,
        linewidth=1.8,
        linestyle="--",
        color="#7c3aed",
        label="Illustrative fixed 250 Ohm floor",
    )
    ax.axhline(190.0, color="#059669", linestyle=":", linewidth=1.8, label="Observed high-current plateau ~190 mV")
    ax.set_xlim(0.0, float(np.max(current_grid_uA)))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Applied current (uA)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title("A low fitted metallic resistance mathematically forces a near-zero ideal-current valley")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(
        0.99,
        0.02,
        "C changes the approach time; these steady-state lines do not depend on C.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    *,
    out_dir: Path,
    summary_path: Path,
    sample_path: Path,
    pre_switch: pd.Series,
    post_switch: pd.Series,
    capacitance: pd.DataFrame,
    conductance: pd.DataFrame,
    resistance: pd.DataFrame,
    prior_fit_path: Path | None,
) -> None:
    c_median = float(capacitance["C_slope_pF"].median())
    c_low = float(capacitance["C_slope_pF"].min())
    c_high = float(capacitance["C_slope_pF"].max())
    high = resistance[resistance["current_inferred_uA"] >= 350.0]
    r_eff_low = float(high["R_effective_ohm"].min())
    r_eff_high = float(high["R_effective_ohm"].max())
    prior_text = "No prior image fit supplied."
    if prior_fit_path is not None and prior_fit_path.is_file():
        prior = json.loads(prior_fit_path.read_text())
        prior_text = (
            f"The prior image-based joint fit used C={float(prior['best_C_pF']):.2f} pF, "
            f"C_th={float(prior['best_C_th_mW_ns_per_K']):.2f} pJ/K, "
            f"S_e={float(prior['best_S_e_mW_per_K']):.4f} mW/K, and "
            f"T0={float(prior['best_T0_K']):.2f} K. Treat these as correlated starting values, not independent measurements."
        )

    lines = [
        "# Lab current-trace parameter estimates",
        "",
        f"Digitized source: `{_display_path(summary_path)}`",
        f"Resistance preset: `{_display_path(sample_path)}`",
        "",
        "## Directly supported by the digitized traces",
        "",
        f"- Switching onset is bracketed between {float(pre_switch['current_inferred_uA']):.1f} and {float(post_switch['current_inferred_uA']):.1f} uA, where plateau ripple rises sharply.",
        f"- Cold-slope electrical capacitance estimates span {c_low:.1f}-{c_high:.1f} pF; median C={c_median:.1f} pF.",
        f"- Above 350 uA, measured plateau V/I spans {r_eff_low:.1f}-{r_eff_high:.1f} Ohm. The specimen R(T) fit has Rm=18.3 Ohm, so using that bare Rm as the switched-state floor forces voltage far below the observed plateau.",
        "- The high-current plateau remains near 190 mV while current changes substantially, so one constant series resistance alone cannot reproduce the entire family.",
        "",
        "## Thermal conductance and ambient temperature",
        "",
        "Using the last pre-switch point and S_e=P_switch/(T_switch-T0), the estimate depends strongly on the actual stage/substrate temperature:",
        "",
        conductance.to_markdown(index=False, floatfmt=".5g"),
        "",
        "T0 must therefore be measured or fixed from the experiment before fitting S_e. In a single-device simulation, this S_e is the environment/substrate coupling. Neighbor coupling S_c is zero and cannot be inferred from one device.",
        "",
        "## Thermal capacitance",
        "",
        "The screenshots do not provide an independent temperature-recovery time. Once tau_th is measured from a pump-probe or cooling transient, use C_th=S_e*tau_th. The generated scenario CSV tabulates this relation for 10, 20, 50, and 100 ns.",
        "",
        prior_text,
        "",
        "## Recommended next measurement",
        "",
        "Export raw time, source current, and voltage CSVs together with stage temperature and the exact measured voltage nodes. Fit the cold electrical edge first, then the post-pulse thermal recovery, and only then fit the coupled switching waveform.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=None, help="Optional precomputed digitized summary CSV.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--sample-preset", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--prior-fit", type=Path, default=DEFAULT_PRIOR_FIT)
    parser.add_argument("--ambient-K", default="298,325,330,333")
    parser.add_argument("--thermal-times-ns", default="10,20,50,100")
    parser.add_argument("--ripple-threshold-mV", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "lab_parameter_estimates")
    args = parser.parse_args()

    sample_path = args.sample_preset.resolve()
    if args.summary is not None:
        summary_source = args.summary.resolve()
        summary = pd.read_csv(summary_source)
    else:
        summary_source = args.image_dir.resolve()
        _, _, summary = digitize_directory(summary_source)
    resist = _load_resistance(sample_path)
    pre_switch, post_switch = _find_switching_bracket(summary, ripple_threshold_mV=args.ripple_threshold_mV)
    capacitance = _estimate_electrical_capacitance(
        summary,
        onset_current_uA=float(pre_switch["current_inferred_uA"]),
    )
    if capacitance.empty:
        raise SystemExit("No valid pre-switch slope points were found.")
    ambient = [float(value) for value in str(args.ambient_K).split(",")]
    thermal_times = [float(value) for value in str(args.thermal_times_ns).split(",")]
    conductance = _thermal_conductance_scenarios(
        pre_switch=pre_switch,
        T_switch_K=float(resist.Tc_K + 0.5 * resist.w_eff),
        ambient_values_K=ambient,
    )
    thermal_capacitance = _thermal_capacitance_scenarios(conductance, thermal_times)
    resistance = _effective_resistance(summary)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    capacitance.to_csv(out_dir / "electrical_capacitance_estimates.csv", index=False)
    conductance.to_csv(out_dir / "thermal_conductance_vs_ambient.csv", index=False)
    thermal_capacitance.to_csv(out_dir / "thermal_capacitance_scenarios.csv", index=False)
    resistance.to_csv(out_dir / "effective_resistance_vs_current.csv", index=False)
    _plot_summary(
        capacitance=capacitance,
        conductance=conductance,
        resistance=resistance,
        out_path=out_dir / "lab_parameter_estimates.png",
    )
    _plot_voltage_floor_comparison(
        resistance=resistance,
        specimen_Rm_ohm=float(resist.Rm),
        out_path=out_dir / "voltage_floor_comparison.png",
    )
    _write_report(
        out_dir=out_dir,
        summary_path=summary_source,
        sample_path=sample_path,
        pre_switch=pre_switch,
        post_switch=post_switch,
        capacitance=capacitance,
        conductance=conductance,
        resistance=resistance,
        prior_fit_path=args.prior_fit.resolve() if args.prior_fit else None,
    )
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
