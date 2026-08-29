"""Generate basic current-input/voltage-output examples with nonzero valleys.

The figure contains two deliberately different demonstrations:

1. A Yuanhang-centered case that stays inside the published 305--370 K R(T)
   calibration range.
2. A lab-scale diagnostic comparison using the fitted specimen R(T) shape and
   measured C estimate.  The control retains the fitted Rm=18.3 Ohm; the
   candidate uses an effective switched resistance of 150 Ohm.  The latter is
   an explicit hypothesis for unresolved device/contact/source behavior, not a
   refit of the measured R(T) data.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
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

from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step
from neuristor.model import YuanhangResistParams


SAMPLE_PRESET = ROOT / "presets" / "resistance_100425_chip1_gap3.json"


def _load_sample_resistance() -> YuanhangResistParams:
    payload = json.loads(SAMPLE_PRESET.read_text())
    return YuanhangResistParams(**payload["resist_params"])


def _late_metrics(out: dict[str, np.ndarray], start_s: float, stop_s: float) -> dict[str, float]:
    mask = (out["t"] >= start_s) & (out["t"] <= stop_s)
    t = np.asarray(out["t"][mask], dtype=float)
    voltage = np.asarray(out["V_vo2"][mask], dtype=float)
    temperature = np.asarray(out["T"][mask], dtype=float)
    low, high = np.quantile(voltage, [0.1, 0.9])
    threshold = 0.5 * (low + high)
    crossings = np.flatnonzero((voltage[:-1] < threshold) & (voltage[1:] >= threshold)) + 1
    periods = np.diff(t[crossings])
    period_s = float(np.mean(periods)) if periods.size else float("nan")
    frequency_MHz = 1e-6 / period_s if np.isfinite(period_s) and period_s > 0.0 else 0.0
    return {
        "frequency_MHz": float(frequency_MHz),
        "cycles_analyzed": float(max(0, crossings.size - 1)),
        "V_min_V": float(np.min(voltage)),
        "V_max_V": float(np.max(voltage)),
        "V_pp_V": float(np.ptp(voltage)),
        "T_min_K": float(np.min(temperature)),
        "T_max_K": float(np.max(temperature)),
    }


def _run_case(
    *,
    name: str,
    current_uA: float,
    params: CurrentDriveParams,
    analysis_start_s: float,
    analysis_stop_s: float,
    interpretation: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    out = simulate_current_step(current_uA, params=params, seed=0)
    metrics = _late_metrics(out, analysis_start_s, analysis_stop_s)
    pulse_stop_s = params.pulse_off_s if params.pulse_off_s is not None else float(out["t"][-1])
    pulse_mask = (out["t"] >= params.pulse_on_s) & (out["t"] <= pulse_stop_s)
    pulse_T_min_K = float(np.min(out["T"][pulse_mask]))
    pulse_T_max_K = float(np.max(out["T"][pulse_mask]))
    row: dict[str, object] = {
        "case": name,
        "I_uA": current_uA,
        "C_pF": params.C_F * 1e12,
        "C_th_pJ_per_K": params.C_th_J_per_K * 1e12,
        "S_e_mW_per_K": params.S_e_W_per_K * 1e3,
        "T0_K": params.T0_K,
        "Rm_ohm": params.resist_params.Rm,
        "tau_th_ns": params.C_th_J_per_K / params.S_e_W_per_K * 1e9,
        "inside_R_T_calibration": bool(
            pulse_T_min_K >= params.resist_params.T_min_K
            and pulse_T_max_K <= params.resist_params.T_max_K
        ),
        "T_pulse_min_K": pulse_T_min_K,
        "T_pulse_max_K": pulse_T_max_K,
        "interpretation": interpretation,
        **metrics,
    }
    return out, row


def _plot_examples(
    *,
    reference: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    rows: list[dict[str, object]],
    out_path: Path,
) -> None:
    reference_row, control_row, candidate_row = rows
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.2), gridspec_kw={"height_ratios": [1.0, 2.3]})

    ref_t_us = reference["t"] * 1e6
    axes[0, 0].plot(ref_t_us, reference["I_in"] * 1e6, color="#111827", linewidth=1.6)
    axes[1, 0].plot(ref_t_us, reference["V_vo2"], color="#2563eb", linewidth=1.25)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 0].axhline(
        float(reference_row["V_min_V"]),
        color="#2563eb",
        linestyle="--",
        linewidth=1.0,
        label=f"late minimum = {float(reference_row['V_min_V']):.3f} V",
    )
    axes[0, 0].set_title("A. Yuanhang-centered case inside calibrated R(T) range")
    axes[0, 0].set_ylabel("Current (uA)")
    axes[1, 0].set_ylabel("VO2 voltage (V)")
    axes[1, 0].set_xlabel("Time (us)")
    axes[1, 0].legend(loc="upper right", fontsize=9)
    axes[1, 0].text(
        0.02,
        0.96,
        f"f = {float(reference_row['frequency_MHz']):.3f} MHz\n"
        f"T = {float(reference_row['T_min_K']):.1f}--{float(reference_row['T_max_K']):.1f} K",
        transform=axes[1, 0].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d1d5db"},
    )

    lab_t_us = candidate["t"] * 1e6
    axes[0, 1].plot(lab_t_us, candidate["I_in"] * 1e6, color="#111827", linewidth=1.6)
    axes[1, 1].plot(
        control["t"] * 1e6,
        control["V_vo2"],
        color="#dc2626",
        linewidth=1.0,
        alpha=0.88,
        label=f"unchanged fit: Rm={float(control_row['Rm_ohm']):.1f} Ohm, min={float(control_row['V_min_V']):.3f} V",
    )
    axes[1, 1].plot(
        lab_t_us,
        candidate["V_vo2"],
        color="#059669",
        linewidth=1.15,
        label=f"effective candidate: Rm={float(candidate_row['Rm_ohm']):.0f} Ohm, min={float(candidate_row['V_min_V']):.3f} V",
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 1].axhline(0.190, color="#7c3aed", linestyle=":", linewidth=1.4, label="observed plateau ~0.190 V")
    axes[0, 1].set_title("B. Same lab-scale dynamics; switched resistance controls the valley")
    axes[0, 1].set_ylabel("Current (uA)")
    axes[1, 1].set_ylabel("VO2 voltage (V)")
    axes[1, 1].set_xlabel("Time (us)")
    axes[1, 1].legend(loc="upper right", fontsize=8.0, framealpha=1.0)
    axes[1, 1].text(
        0.02,
        0.96,
        f"candidate f = {float(candidate_row['frequency_MHz']):.2f} MHz\n"
        f"candidate T = {float(candidate_row['T_min_K']):.1f}--{float(candidate_row['T_max_K']):.1f} K",
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d1d5db"},
    )

    for ax in axes.flat:
        ax.grid(True, alpha=0.22)
    for ax in axes[0, :]:
        ax.set_ylim(bottom=-0.04 * max(float(np.max(reference["I_in"] * 1e6)), 1000.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _write_report(out_dir: Path, rows: list[dict[str, object]], configs: dict[str, CurrentDriveParams]) -> None:
    ref, control, candidate = rows
    lines = [
        "# Nonzero-valley current-input examples",
        "",
        "These examples answer a narrow question: can the supported ideal-current model sustain oscillations without the voltage reaching zero? Yes, but the interpretation depends on the resistance model.",
        "",
        "## A. Yuanhang-centered, in-range demonstration",
        "",
        f"- Input pulse: {float(ref['I_uA']):.0f} uA",
        f"- C={float(ref['C_pF']):.3f} pF, C_th={float(ref['C_th_pJ_per_K']):.3f} pJ/K, S_e={float(ref['S_e_mW_per_K']):.6f} mW/K",
        f"- Frequency={float(ref['frequency_MHz']):.4f} MHz; late voltage range={float(ref['V_min_V']):.3f}--{float(ref['V_max_V']):.3f} V",
        f"- Full-pulse temperature range={float(ref['T_pulse_min_K']):.2f}--{float(ref['T_pulse_max_K']):.2f} K, inside the published 305--370 K R(T) calibration range.",
        "- This uses Yuanhang's R(T), C, S_e, and T0, with C_th set to four times the nominal value to keep the entire pulse trajectory inside the calibrated temperature range.",
        "",
        "## B. Lab-scale diagnostic comparison",
        "",
        f"Both traces use I={float(candidate['I_uA']):.0f} uA, the fitted specimen R(T) shape, measured C estimate of {float(candidate['C_pF']):.1f} pF, C_th={float(candidate['C_th_pJ_per_K']):.1f} pJ/K, and S_e={float(candidate['S_e_mW_per_K']):.2f} mW/K.",
        "",
        f"- Unchanged fitted Rm={float(control['Rm_ohm']):.1f} Ohm: minimum={float(control['V_min_V']):.3f} V.",
        f"- Effective candidate Rm={float(candidate['Rm_ohm']):.0f} Ohm: minimum={float(candidate['V_min_V']):.3f} V, frequency={float(candidate['frequency_MHz']):.2f} MHz.",
        f"- Candidate full-pulse temperature range={float(candidate['T_pulse_min_K']):.2f}--{float(candidate['T_pulse_max_K']):.2f} K, inside the fitted specimen R(T) range.",
        "",
        "The 150 Ohm value is not claimed as the intrinsic metallic resistance measured by the R(T) fit. It is an effective switched-state candidate that could represent unresolved contact resistance, source/measurement impedance, or a different device state. The comparison isolates why C alone cannot raise the valley.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/generate_nonzero_valley_examples.py",
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    payload = {
        name: {**asdict(params), "resist_params": asdict(params.resist_params)}
        for name, params in configs.items()
    }
    (out_dir / "simulation_parameters.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "figures" / "current_drive" / "nonzero_valley_examples",
    )
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_params = CurrentDriveParams(
        dt_s=0.5e-9,
        t_end_s=40e-6,
        pulse_on_s=2e-6,
        pulse_off_s=38e-6,
        T0_K=325.0,
        T_init_K=325.0,
        C_F=145.34619293e-12,
        C_th_J_per_K=4.0 * 49.62776831e-12,
        S_e_W_per_K=0.20558726e-3,
        resist_params=YuanhangResistParams(),
        start_branch="insulator",
    )
    sample = _load_sample_resistance()
    lab_base = dict(
        dt_s=0.1e-9,
        t_end_s=1.4e-6,
        pulse_on_s=0.15e-6,
        pulse_off_s=1.25e-6,
        T0_K=325.0,
        T_init_K=325.0,
        C_F=22.7e-12,
        C_th_J_per_K=0.5e-12,
        S_e_W_per_K=0.08e-3,
        start_branch="insulator",
    )
    control_params = CurrentDriveParams(**lab_base, resist_params=sample)
    candidate_params = CurrentDriveParams(
        **lab_base,
        resist_params=replace(sample, Rm0=150.0, Rm_factor=1.0),
    )

    reference, reference_row = _run_case(
        name="yuanhang_centered_in_range",
        current_uA=600.0,
        params=reference_params,
        analysis_start_s=10e-6,
        analysis_stop_s=36e-6,
        interpretation="Published R(T), C, S_e, and T0; C_th quadrupled to keep the full pulse inside the calibrated R(T) range.",
    )
    control, control_row = _run_case(
        name="lab_shape_unmodified_Rm_control",
        current_uA=1000.0,
        params=control_params,
        analysis_start_s=0.3e-6,
        analysis_stop_s=1.15e-6,
        interpretation="Fitted specimen R(T) including Rm=18.3 Ohm; expected near-zero ideal-current valley.",
    )
    candidate, candidate_row = _run_case(
        name="lab_shape_effective_Rm_150_candidate",
        current_uA=1000.0,
        params=candidate_params,
        analysis_start_s=0.3e-6,
        analysis_stop_s=1.15e-6,
        interpretation="Exploratory effective switched resistance; not an intrinsic-Rm refit.",
    )
    rows = [reference_row, control_row, candidate_row]
    pd.DataFrame(rows).to_csv(out_dir / "simulation_metrics.csv", index=False)
    _plot_examples(
        reference=reference,
        control=control,
        candidate=candidate,
        rows=rows,
        out_path=out_dir / "current_input_voltage_output.png",
    )
    _write_report(
        out_dir,
        rows,
        {
            "yuanhang_centered_in_range": reference_params,
            "lab_shape_unmodified_Rm_control": control_params,
            "lab_shape_effective_Rm_150_candidate": candidate_params,
        },
    )
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
