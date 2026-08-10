"""Sweep electrical and thermal capacitance in the ideal-current VO2 model.

The defaults are centered on the values in Yuanhang Zhang's reference code:

* C = 145.34619293 pF
* C_th = 49.62776831 pJ/K (equivalently mW ns/K)
* S_e = 0.20558726 mW/K
* T0 = 325 K

One heatmap is written for each current.  Frequency is reported only for a
sustained, regular late-time oscillation; fixed points and unresolved
transients are saved as zero frequency.  The C=0 column is the algebraic limit
V=I R(T), not a small positive capacitance surrogate.

Usage:
    python scripts/sweep_current_capacitances.py
    python scripts/sweep_current_capacitances.py --currents-uA 400,500,600,700
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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

import neuristor.model as model
from neuristor.current_drive_sim import (
    CurrentDriveParams,
    current_drive_operating_estimates,
    simulate_current_step,
)
from neuristor.model import HysteresisArray, YuanhangResistParams


YUANHANG_C_PF = 145.34619293
YUANHANG_CTH_PJ_PER_K = 49.62776831
YUANHANG_SE_MW_PER_K = 0.20558726


def _parse_csv_floats(text: str) -> list[float]:
    values = [float(token.strip()) for token in str(text).split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated number")
    return values


def _grid_values() -> tuple[np.ndarray, np.ndarray]:
    c_pF = YUANHANG_C_PF * np.asarray([0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
    cth_pJ_K = YUANHANG_CTH_PJ_PER_K * np.asarray([0.5, 0.70710678, 1.0, 1.41421356, 2.0, 2.82842712, 4.0])
    return c_pF, cth_pJ_K


def _threshold_crossings(values: np.ndarray) -> np.ndarray:
    if values.size < 8 or not np.all(np.isfinite(values)):
        return np.asarray([], dtype=int)
    low, high = np.quantile(values, [0.1, 0.9])
    if high - low < 0.05:
        return np.asarray([], dtype=int)
    threshold = 0.5 * (low + high)
    return np.flatnonzero((values[:-1] < threshold) & (values[1:] >= threshold)) + 1


def _trace_metrics(t_s: np.ndarray, voltage: np.ndarray) -> dict[str, float]:
    voltage = np.asarray(voltage, dtype=float)
    crossings = _threshold_crossings(voltage)
    periods = np.diff(t_s[crossings]) if crossings.size >= 2 else np.asarray([], dtype=float)
    cycles = max(0, int(crossings.size - 1))
    period_mean = float(np.mean(periods)) if periods.size else float("nan")
    period_cv = float(np.std(periods) / period_mean) if periods.size and period_mean > 0.0 else float("inf")
    frequency_MHz = 1e-6 / period_mean if np.isfinite(period_mean) and period_mean > 0.0 else 0.0
    v_low, v_high = np.quantile(voltage, [0.02, 0.98])
    vpp = float(v_high - v_low)
    half = voltage.size // 2
    early_vpp = float(np.ptp(voltage[:half])) if half >= 2 else 0.0
    late_vpp = float(np.ptp(voltage[half:])) if voltage.size - half >= 2 else 0.0
    persistence = late_vpp / max(early_vpp, 1e-12)
    oscillatory = cycles >= 3 and vpp >= 0.05 and period_cv <= 0.20 and 0.65 <= persistence <= 1.35
    return {
        "frequency_MHz": float(frequency_MHz if oscillatory else 0.0),
        "raw_frequency_MHz": float(frequency_MHz),
        "oscillatory": float(1.0 if oscillatory else 0.0),
        "cycles": float(cycles),
        "period_cv": period_cv,
        "persistence_ratio": float(persistence),
        "V_pp_V": vpp,
        "V_min_V": float(np.min(voltage)),
        "V_max_V": float(np.max(voltage)),
    }


def _simulate_grid(
    *,
    current_uA: float,
    c_pF: np.ndarray,
    cth_pJ_K: np.ndarray,
    dt_ns: float,
    t_end_us: float,
    analysis_start_us: float,
    resist: YuanhangResistParams,
) -> pd.DataFrame:
    c_mesh, cth_mesh = np.meshgrid(c_pF, cth_pJ_K)
    c_F = np.asarray(c_mesh.reshape(-1) * 1e-12, dtype=np.float32)
    cth_J_K = np.asarray(cth_mesh.reshape(-1) * 1e-12, dtype=np.float32)
    n_points = c_F.size
    dt_s = np.float32(dt_ns * 1e-9)
    n_steps = int(round(t_end_us * 1e-6 / float(dt_s))) + 1
    analysis_start = int(round(analysis_start_us * 1e-6 / float(dt_s)))
    analysis_start = min(max(0, analysis_start), n_steps - 2)
    analysis_steps = n_steps - analysis_start

    voltage = np.zeros(n_points, dtype=np.float32)
    temperature = np.full(n_points, 325.0, dtype=np.float32)
    voltage_late = np.zeros((analysis_steps, n_points), dtype=np.float32)
    temp_min = np.full(n_points, np.inf, dtype=np.float32)
    temp_max = np.full(n_points, -np.inf, dtype=np.float32)
    max_step_K = np.zeros(n_points, dtype=np.float32)

    hyst = HysteresisArray(resist, size=n_points, start_branch="insulator", independent_anchors=True)
    hyst.initialize(temperature)

    I_A = np.float32(current_uA * 1e-6)
    S_e = np.float32(YUANHANG_SE_MW_PER_K * 1e-3)
    T0 = np.float32(325.0)
    algebraic = c_F == 0.0
    finite_c = ~algebraic
    thermal_alpha = -np.expm1(-(dt_s * S_e) / cth_J_K)
    late_idx = 0

    for step in range(n_steps):
        resistance, _ = hyst.evaluate(temperature)
        resistance = np.maximum(np.asarray(resistance, dtype=np.float32), np.float32(1e-12))
        voltage[algebraic] = I_A * resistance[algebraic]
        power = voltage * voltage / resistance

        if step >= analysis_start:
            voltage_late[late_idx, :] = voltage
            temp_min = np.minimum(temp_min, temperature)
            temp_max = np.maximum(temp_max, temperature)
            late_idx += 1
        if step == n_steps - 1:
            break

        voltage_next = voltage.copy()
        if np.any(finite_c):
            target = I_A * resistance[finite_c]
            tau = c_F[finite_c] * resistance[finite_c]
            alpha = -np.expm1(-dt_s / tau)
            voltage_next[finite_c] = voltage[finite_c] + alpha * (target - voltage[finite_c])
        voltage_next[algebraic] = I_A * resistance[algebraic]

        thermal_target = T0 + power / S_e
        temperature_next = temperature + thermal_alpha * (thermal_target - temperature)
        max_step_K = np.maximum(max_step_K, np.abs(temperature_next - temperature))
        voltage = voltage_next
        temperature = temperature_next.astype(np.float32, copy=False)

    t_late = (analysis_start + np.arange(analysis_steps, dtype=float)) * float(dt_s)
    rows: list[dict[str, float]] = []
    tau_th_s = cth_J_K.astype(float) / float(S_e)
    tau_el_metal_s = c_F.astype(float) * float(resist.Rm)
    for idx in range(n_points):
        metrics = _trace_metrics(t_late, voltage_late[:, idx])
        rows.append(
            {
                "I_uA": float(current_uA),
                "C_pF": float(c_F[idx] * 1e12),
                "C_th_pJ_per_K": float(cth_J_K[idx] * 1e12),
                "tau_el_metal_ns": float(tau_el_metal_s[idx] * 1e9),
                "tau_th_ns": float(tau_th_s[idx] * 1e9),
                "tau_el_over_tau_th": float(tau_el_metal_s[idx] / tau_th_s[idx]),
                "T_min_K": float(temp_min[idx]),
                "T_max_K": float(temp_max[idx]),
                "outside_resistance_temperature_range": float(
                    1.0
                    if temp_min[idx] < float(resist.T_min_K) or temp_max[idx] > float(resist.T_max_K)
                    else 0.0
                ),
                "max_temperature_step_K": float(max_step_K[idx]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _plot_heatmap(df: pd.DataFrame, *, current_uA: float, out_path: Path) -> None:
    c_values = np.sort(df["C_pF"].unique())
    cth_values = np.sort(df["C_th_pJ_per_K"].unique())
    freq = (
        df.pivot(index="C_th_pJ_per_K", columns="C_pF", values="frequency_MHz")
        .reindex(index=cth_values, columns=c_values)
        .to_numpy(dtype=float)
    )
    outside_range = (
        df.pivot(index="C_th_pJ_per_K", columns="C_pF", values="outside_resistance_temperature_range")
        .reindex(index=cth_values, columns=c_values)
        .to_numpy(dtype=float)
    )
    masked = np.ma.masked_where(freq <= 0.0, freq)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e7eb")

    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    image = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(c_values.size))
    ax.set_yticks(np.arange(cth_values.size))
    ax.set_xticklabels(["0\n(RC=0)" if value == 0.0 else f"{value:.1f}" for value in c_values])
    ax.set_yticklabels([f"{value:.1f}" for value in cth_values])
    ax.set_xlabel("Electrical capacitance C (pF)")
    ax.set_ylabel("Thermal capacitance C_th (pJ/K)")
    ax.set_title(f"Sustained oscillation frequency at I = {current_uA:.0f} uA")
    for row in range(cth_values.size):
        for col in range(c_values.size):
            suffix = "*" if outside_range[row, col] >= 0.5 else ""
            label = "--" if freq[row, col] <= 0.0 else f"{freq[row, col]:.2f}{suffix}"
            color = "#111827" if freq[row, col] <= 0.0 or freq[row, col] < np.nanmax(freq) * 0.65 else "white"
            ax.text(col, row, label, ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("Frequency (MHz)")
    ax.text(
        0.0,
        -0.16,
        "Gray/-- = no sustained regular oscillation. * = T leaves the calibrated R(T) range.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_panel(
    results: pd.DataFrame,
    *,
    currents_uA: list[float],
    out_path: Path,
) -> None:
    """Render all current slices with one shared frequency scale."""

    positive = results.loc[results["frequency_MHz"] > 0.0, "frequency_MHz"]
    vmax = float(positive.max()) if not positive.empty else 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e7eb")
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.4), sharex=True, sharey=True)
    image = None
    for ax, current_uA in zip(axes.flat, currents_uA):
        frame = results[np.isclose(results["I_uA"], current_uA)]
        c_values = np.sort(frame["C_pF"].unique())
        cth_values = np.sort(frame["C_th_pJ_per_K"].unique())
        freq = (
            frame.pivot(index="C_th_pJ_per_K", columns="C_pF", values="frequency_MHz")
            .reindex(index=cth_values, columns=c_values)
            .to_numpy(dtype=float)
        )
        outside = (
            frame.pivot(
                index="C_th_pJ_per_K",
                columns="C_pF",
                values="outside_resistance_temperature_range",
            )
            .reindex(index=cth_values, columns=c_values)
            .to_numpy(dtype=float)
        )
        image = ax.imshow(
            np.ma.masked_where(freq <= 0.0, freq),
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )
        outside_rows, outside_cols = np.where(outside >= 0.5)
        ax.scatter(outside_cols, outside_rows, marker="*", s=26, c="black", linewidths=0.2)
        nominal_col = int(np.argmin(np.abs(c_values - YUANHANG_C_PF)))
        nominal_row = int(np.argmin(np.abs(cth_values - YUANHANG_CTH_PJ_PER_K)))
        ax.scatter(
            [nominal_col],
            [nominal_row],
            marker="s",
            s=150,
            facecolors="none",
            edgecolors="white",
            linewidths=1.8,
        )
        ax.set_title(f"I = {current_uA:.0f} uA")
        ax.set_xticks(np.arange(c_values.size))
        ax.set_yticks(np.arange(cth_values.size))
        ax.set_xticklabels(["0" if value == 0.0 else f"{value:.0f}" for value in c_values])
        ax.set_yticklabels([f"{value:.0f}" for value in cth_values])
        ax.grid(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Electrical capacitance C (pF)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Thermal capacitance C_th (pJ/K)")
    fig.suptitle("Yuanhang-parameter ideal-current capacitance study", fontsize=16)
    fig.text(
        0.5,
        0.025,
        "Gray = no sustained regular cycle; black star = T leaves calibrated R(T) range; white square = Yuanhang C and C_th.",
        ha="center",
        fontsize=10,
        color="#374151",
    )
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.09, top=0.92, wspace=0.12, hspace=0.18)
    if image is not None:
        colorbar_ax = fig.add_axes([0.92, 0.16, 0.016, 0.68])
        cbar = fig.colorbar(image, cax=colorbar_ax)
        cbar.set_label("Sustained oscillation frequency (MHz)")
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_trace_comparison(
    *,
    current_uA: float,
    c_values_pF: list[float],
    dt_ns: float,
    out_path: Path,
    resist: YuanhangResistParams,
) -> None:
    fig, (ax_v, ax_t) = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for c_pF in c_values_pF:
        params = CurrentDriveParams(
            dt_s=dt_ns * 1e-9,
            t_end_s=12e-6,
            T0_K=325.0,
            T_init_K=325.0,
            C_F=c_pF * 1e-12,
            C_th_J_per_K=YUANHANG_CTH_PJ_PER_K * 1e-12,
            S_e_W_per_K=YUANHANG_SE_MW_PER_K * 1e-3,
            resist_params=resist,
            start_branch="insulator",
        )
        out = simulate_current_step(current_uA, params=params, seed=0)
        mask = out["t"] >= 4e-6
        label = "C=0 (V=IR)" if c_pF == 0.0 else f"C={c_pF:.1f} pF"
        ax_v.plot(out["t"][mask] * 1e6, out["V_vo2"][mask], linewidth=1.2, label=label)
        ax_t.plot(out["t"][mask] * 1e6, out["T"][mask], linewidth=1.2, label=label)
    floor = current_uA * 1e-6 * resist.Rm
    ax_v.axhline(floor, color="black", linestyle="--", linewidth=1.1, label=f"I Rm = {floor:.3f} V")
    ax_t.axhline(resist.T_max_K, color="black", linestyle=":", linewidth=1.1, label="R(T) clamp limit")
    ax_v.set_ylabel("VO2 voltage (V)")
    ax_t.set_ylabel("Temperature (K)")
    ax_t.set_xlabel("Time (us)")
    ax_v.set_title(f"Changing C changes timing, not the ideal-current metallic floor ({current_uA:.0f} uA)")
    ax_v.legend(ncols=3, fontsize=8)
    ax_t.legend(loc="upper left", fontsize=8)
    for ax in (ax_v, ax_t):
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def _write_report(
    *,
    out_dir: Path,
    results: pd.DataFrame,
    currents_uA: list[float],
    dt_ns: float,
    t_end_us: float,
    analysis_start_us: float,
    resist: YuanhangResistParams,
) -> None:
    base = CurrentDriveParams(
        C_F=YUANHANG_C_PF * 1e-12,
        C_th_J_per_K=YUANHANG_CTH_PJ_PER_K * 1e-12,
        S_e_W_per_K=YUANHANG_SE_MW_PER_K * 1e-3,
        T0_K=325.0,
        T_init_K=325.0,
        resist_params=resist,
    )
    estimates = current_drive_operating_estimates(base, I_uA=max(currents_uA))
    oscillatory = results[results["oscillatory"] >= 0.5]
    outside_range = results[results["outside_resistance_temperature_range"] >= 0.5]
    lines = [
        "# Current-drive capacitance study",
        "",
        "This sweep uses Yuanhang's resistance, thermal-conductance, and ambient-temperature values in the derived ideal-current circuit.",
        "The upstream reference circuit itself is voltage-driven through a 12 kOhm load; it is not an ideal-current circuit.",
        "",
        "## Baseline parameters",
        "",
        f"- Electrical capacitance: {YUANHANG_C_PF:.8g} pF",
        f"- Thermal capacitance: {YUANHANG_CTH_PJ_PER_K:.8g} pJ/K",
        f"- Environmental thermal conductance: {YUANHANG_SE_MW_PER_K:.8g} mW/K",
        "- Ambient temperature: 325 K",
        f"- Thermal time constant C_th/S_e: {estimates['tau_thermal_s'] * 1e9:.3f} ns",
        f"- Metallic electrical time constant Rm*C: {estimates['tau_metal_s'] * 1e9:.3f} ns",
        f"- Integration step: {dt_ns:g} ns; simulated to {t_end_us:g} us; analysis starts at {analysis_start_us:g} us",
        "",
        "## Main findings",
        "",
        f"- Yuanhang metallic resistance is {resist.Rm:.3f} Ohm, so the ideal-current voltage floor is I*Rm, not zero.",
        "- Lower C shortens the approach to that floor; it does not raise the floor.",
        (
            f"- The approximate RC=0 thermal-cycle bounds are {estimates['thermal_only_lower_current_uA']:.1f} to "
            f"{estimates['thermal_only_upper_current_uA']:.1f} uA. Because the lower bound exceeds the upper bound, "
            "the Yuanhang parameters do not support a thermal-only limit cycle in this approximation."
        ),
        f"- Sustained cells found in the finite-C grid: {len(oscillatory)} of {len(results)}.",
        f"- {len(outside_range)} cells leave the {resist.T_min_K:g}-{resist.T_max_K:g} K calibrated resistance range; they are marked with * and should be treated as extrapolations with saturated R(T).",
        "- Cells marked zero/gray either settle to a fixed point or fail the late-time regularity criteria; startup ringing is excluded.",
        "",
        "## Calibration equations",
        "",
        "- From a fixed-resistance electrical transient: C = tau_el/R.",
        "- From thermal recovery: C_th = S_e*tau_th.",
        "- At switching onset: S_e approximately equals P_switch/(T_switch-T0), with P=V^2/R (or I^2 R at current steady state).",
        "- In a single-device run, neighbor coupling is zero. Fit the environment/substrate path as S_e; reserve S_c for multi-device thermal coupling.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))
    payload = {
        "model": "derived ideal-current source with Yuanhang R(T)",
        "resistance_params": asdict(resist),
        "baseline": {
            "C_pF": YUANHANG_C_PF,
            "C_th_pJ_per_K": YUANHANG_CTH_PJ_PER_K,
            "S_e_mW_per_K": YUANHANG_SE_MW_PER_K,
            "T0_K": 325.0,
        },
        "currents_uA": currents_uA,
        "numerics": {"dt_ns": dt_ns, "t_end_us": t_end_us, "analysis_start_us": analysis_start_us},
        "operating_estimates": estimates,
    }
    (out_dir / "study_config.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--currents-uA", type=_parse_csv_floats, default=[300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
    parser.add_argument("--dt-ns", type=float, default=0.5)
    parser.add_argument("--t-end-us", type=float, default=30.0)
    parser.add_argument("--analysis-start-us", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "current_capacitance_study")
    args = parser.parse_args()
    if args.dt_ns <= 0.0 or args.t_end_us <= 0.0:
        parser.error("dt and duration must be positive")
    if not 0.0 <= args.analysis_start_us < args.t_end_us:
        parser.error("analysis start must lie inside the simulation interval")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    c_pF, cth_pJ_K = _grid_values()
    resist = YuanhangResistParams()

    # NumPy float32 follows the same equations and makes the large vectorized
    # sweep practical. The upstream-fidelity unit test covers the Torch path.
    previous_backend = model._TORCH_HYSTERESIS_AVAILABLE
    model._TORCH_HYSTERESIS_AVAILABLE = False
    frames: list[pd.DataFrame] = []
    try:
        for current_uA in args.currents_uA:
            print(f"[sweep] I={current_uA:g} uA")
            frame = _simulate_grid(
                current_uA=current_uA,
                c_pF=c_pF,
                cth_pJ_K=cth_pJ_K,
                dt_ns=args.dt_ns,
                t_end_us=args.t_end_us,
                analysis_start_us=args.analysis_start_us,
                resist=resist,
            )
            frames.append(frame)
            _plot_heatmap(frame, current_uA=current_uA, out_path=out_dir / f"frequency_heatmap_{current_uA:g}uA.png")
    finally:
        model._TORCH_HYSTERESIS_AVAILABLE = previous_backend

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(out_dir / "capacitance_frequency_grid.csv", index=False)
    if len(args.currents_uA) == 6:
        _plot_heatmap_panel(
            results,
            currents_uA=[float(value) for value in args.currents_uA],
            out_path=out_dir / "frequency_heatmaps_all_currents.png",
        )
    _plot_trace_comparison(
        current_uA=600.0,
        c_values_pF=[0.0, 0.25 * YUANHANG_C_PF, YUANHANG_C_PF, 4.0 * YUANHANG_C_PF],
        dt_ns=args.dt_ns,
        out_path=out_dir / "capacitance_trace_comparison_600uA.png",
        resist=resist,
    )
    _write_report(
        out_dir=out_dir,
        results=results,
        currents_uA=[float(value) for value in args.currents_uA],
        dt_ns=float(args.dt_ns),
        t_end_us=float(args.t_end_us),
        analysis_start_us=float(args.analysis_start_us),
        resist=resist,
    )
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
