#!/usr/bin/env python3
"""
Generate figures used by SCIENTIFIC_MODEL_DEMONSTRATION.tex.

Outputs:
  outputs/paper_figures/model_lineage_flow.png
  outputs/paper_figures/table_extract.png
  outputs/paper_figures/fit_transform_pipeline.png
  outputs/paper_figures/fit_overlay_with_residuals.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data/experimental/100425_chip1_gap3.tsv"
PRESET_PATH = REPO_ROOT / "presets/resistance_100425_chip1_gap3.json"
OUT_DIR = REPO_ROOT / "outputs/paper_figures"


def _load_experimental_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", engine="python", skip_blank_lines=True)
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) > 0 and str(df.iloc[0, 0]).strip().upper() in {"K", "TEMP", "TEMPERATURE"}:
        df = df.iloc[1:].reset_index(drop=True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Temperature", "Resistance"]).reset_index(drop=True)
    return df


def _estimate_seed_triplet(T: np.ndarray, R: np.ndarray) -> tuple[float, float, float]:
    q_low = np.quantile(T, 0.22)
    low_mask = T <= q_low
    if np.sum(low_mask) < 8:
        idx = np.argsort(T)[: max(8, min(24, T.size))]
        low_mask = np.zeros_like(T, dtype=bool)
        low_mask[idx] = True

    T_low = T[low_mask]
    R_low = R[low_mask]
    x = 1.0 / np.maximum(T_low, 1e-9)

    r_min = float(np.min(R))
    rm_grid = np.linspace(max(0.05 * r_min, 1e-6), max(2.2 * r_min, r_min + 1.0), 280)

    best_err = float("inf")
    best = (1.0, 1.0, 0.0)
    for rm in rm_grid:
        y = R_low - rm
        if np.any(y <= 0.0):
            continue
        logy = np.log(y)
        slope, intercept = np.polyfit(x, logy, 1)
        fit = slope * x + intercept
        err = float(np.sqrt(np.mean((fit - logy) ** 2)))
        if err < best_err:
            best_err = err
            best = (float(np.exp(intercept)), float(slope), float(rm))
    return best


def _make_lineage_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str = "#f4f7fb") -> None:
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#2c3e50",
            facecolor=fc,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=14, linewidth=1.4, color="#2c3e50")
        ax.add_patch(a)

    box(0.4, 2.35, 2.7, 1.2, "Paper A (2024)\nElectrical + Thermal ODEs\nVoltage-driven neuristor")
    box(3.6, 2.35, 2.7, 1.2, "Paper B (2002)\nHysteresis law\nMajor/minor loops")
    box(6.8, 2.35, 2.9, 1.2, "model.py\nR(T,H) + Euler solver\nSingle + array")
    box(9.9, 2.35, 1.7, 1.2, "current_drive_sim.py\nCurrent-input module\nEuler/EM")

    box(2.0, 0.55, 3.5, 1.2, "Experimental table\n100425_chip1_gap3.tsv\n(T, R, time, ...)", fc="#fff8e8")
    box(
        6.1,
        0.55,
        3.8,
        1.2,
        "resistance_custom_analysis.py\nFit R0,Ea,Rm,w,Tc,beta,gamma\nSave specimen preset JSON",
        fc="#fff8e8",
    )

    arrow(3.1, 2.95, 3.6, 2.95)
    arrow(6.3, 2.95, 6.8, 2.95)
    arrow(9.7, 2.95, 9.9, 2.95)
    arrow(3.75, 1.75, 6.1, 1.15)
    arrow(8.0, 1.75, 8.0, 2.35)

    ax.set_title("Equation and Data Lineage Used in This Project", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _make_table_extract_figure(df: pd.DataFrame, out_path: Path) -> None:
    T = df["Temperature"].to_numpy(float)
    R = df["Resistance"].to_numpy(float)
    dT = np.diff(T, prepend=T[0])
    branch = np.where(dT > 0, "heating", np.where(dT < 0, "cooling", "flat"))

    extract = pd.DataFrame(
        {
            "idx": np.arange(len(df)),
            "T [K]": T,
            "R [Ohm]": R,
            "dT [K]": dT,
            "branch": branch,
        }
    )
    pick_idx = [0, 1, 2, 60, 120, 180, 240, len(df) - 1]
    tab = extract.iloc[pick_idx].copy()
    for c in ("T [K]", "R [Ohm]", "dT [K]"):
        tab[c] = tab[c].map(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.axis("off")
    tbl = ax.table(cellText=tab.values, colLabels=tab.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    ax.set_title("Extract of Experimental Input Table (100425_chip1_gap3.tsv)", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _make_transform_pipeline_figure(T: np.ndarray, R: np.ndarray, out_path: Path) -> tuple[float, float, float]:
    dT = np.diff(T, prepend=T[0])
    r0_est, ea_est, rm_est = _estimate_seed_triplet(T, R)
    g_seed = (R - rm_est) / np.maximum(r0_est * np.exp(ea_est / np.maximum(T, 1e-9)), 1e-12)
    g_seed = np.clip(g_seed, 0.0, 1.0)

    q_low = np.quantile(T, 0.22)
    low_mask = T <= q_low
    if np.sum(low_mask) < 8:
        idx = np.argsort(T)[: max(8, min(24, T.size))]
        low_mask = np.zeros_like(T, dtype=bool)
        low_mask[idx] = True
    T_low = T[low_mask]
    R_low = R[low_mask]
    x_all = 1.0 / np.maximum(T_low, 1e-9)
    y_all = np.log(np.maximum(R_low - rm_est, 1e-9))
    log_r0_est = np.log(max(r0_est, 1e-12))

    fig = plt.figure(figsize=(12, 6.6))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.semilogy(T[dT < 0], R[dT < 0], ".", alpha=0.65, label="Cooling points")
    ax1.semilogy(T[dT > 0], R[dT > 0], ".", alpha=0.65, label="Heating points")
    ax1.set_xlabel("Temperature [K]")
    ax1.set_ylabel("Resistance [Ohm]")
    ax1.set_title("Step A: Raw Measured R(T)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=9)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x_all, y_all, "o", ms=4, alpha=0.8, label="Low-T samples")
    xx = np.linspace(x_all.min(), x_all.max(), 200)
    ax2.plot(xx, ea_est * xx + log_r0_est, "-", lw=2, label="Linear fit")
    ax2.set_xlabel("1/T [1/K]")
    ax2.set_ylabel("ln(R - Rm_est)")
    ax2.set_title("Step B: Arrhenius Linearization for Seed (R0, Ea, Rm)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best", fontsize=9)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(T, g_seed, ".", ms=4, alpha=0.8)
    ax3.set_xlabel("Temperature [K]")
    ax3.set_ylabel("g_seed [0..1]")
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title("Step C: Convert Measured R(T) to Seed g(T)")
    ax3.grid(alpha=0.25)

    fig.suptitle(
        f"Seed-transform values used in fitting: R0_est={r0_est:.4g} Ohm, Ea_est={ea_est:.1f} K, Rm_est={rm_est:.3f} Ohm",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return r0_est, ea_est, rm_est


def _replay_hysteresis_prediction(T: np.ndarray, params_payload: dict, start_branch: str) -> np.ndarray:
    p = params_payload
    w = float(p["w"]) * float(p.get("width_factor", 1.0))
    Tc = float(p["Tc_K"])
    beta = float(p["beta"])
    gamma = float(p["gamma"])
    R0 = float(p["R0"])
    Ea = float(p["Ea_over_k"])
    Rm = float(p["Rm0"]) * float(p.get("Rm_factor", 1.0))
    rev_thr = float(p.get("reversal_threshold_K", 0.01))
    Tmin = float(p.get("T_min_K", np.min(T) - 1.0))
    Tmax = float(p.get("T_max_K", np.max(T) + 1.0))

    delta = 1.0 if str(start_branch).lower() == "insulator" else -1.0
    reversed_flag = 0.0

    T0 = float(np.clip(T[0], Tmin, Tmax))

    def g_major(t_val: float, d_val: float) -> float:
        return float(np.clip(0.5 + 0.5 * np.tanh(beta * (d_val * w / 2.0 + Tc - t_val)), 0.0, 1.0))

    gr = g_major(T0, delta)
    Tr = T0
    Tpr = delta * w / 2.0 + Tc - (1.0 / beta) * np.arctanh(np.clip(2.0 * gr - 1.0, -1.0 + 1e-12, 1.0 - 1e-12)) - Tr
    T_last = T0

    def pfun(x: float) -> float:
        return float(0.5 * (1.0 - np.sin(gamma * x)) * (1.0 + np.tanh(np.pi * np.pi - 2.0 * np.pi * x)))

    pred = np.zeros_like(T)
    for i, t_raw in enumerate(T):
        t_val = float(np.clip(t_raw, Tmin, Tmax))
        dtemp = t_val - T_last
        if abs(dtemp) > rev_thr:
            dnew = np.sign(dtemp)
            if dnew == 0.0:
                dnew = delta
            if dnew != delta:
                denom = Tpr if abs(Tpr) > 1e-9 else (1e-9 if Tpr >= 0 else -1e-9)
                Tp = Tpr * pfun((t_val - Tr) / denom) * reversed_flag
                g_now = float(np.clip(0.5 + 0.5 * np.tanh(beta * (delta * w / 2.0 + Tc - (t_val + Tp))), 0.0, 1.0))
                gr = g_now
                delta = dnew
                reversed_flag = 1.0
                Tr = t_val
                Tpr = (
                    delta * w / 2.0
                    + Tc
                    - (1.0 / beta) * np.arctanh(np.clip(2.0 * gr - 1.0, -1.0 + 1e-12, 1.0 - 1e-12))
                    - Tr
                )
        T_last = t_val

        denom = Tpr if abs(Tpr) > 1e-9 else (1e-9 if Tpr >= 0 else -1e-9)
        Tp = Tpr * pfun((t_val - Tr) / denom) * reversed_flag
        g_val = float(np.clip(0.5 + 0.5 * np.tanh(beta * (delta * w / 2.0 + Tc - (t_val + Tp))), 0.0, 1.0))
        pred[i] = R0 * np.exp(Ea / max(t_val, 1e-9)) * g_val + Rm

    return pred


def _make_fit_overlay_figure(T: np.ndarray, R: np.ndarray, preset_payload: dict, out_path: Path) -> None:
    dT = np.diff(T, prepend=T[0])
    pred = _replay_hysteresis_prediction(
        T=T,
        params_payload=preset_payload["resist_params"],
        start_branch=str(preset_payload.get("start_branch", "insulator")),
    )
    res_log = np.log10(np.maximum(pred, 1e-12)) - np.log10(np.maximum(R, 1e-12))

    fig = plt.figure(figsize=(12, 6.8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    ax_top.semilogy(T[dT < 0], R[dT < 0], ".", alpha=0.6, label="Measured cooling")
    ax_top.semilogy(T[dT > 0], R[dT > 0], ".", alpha=0.6, label="Measured heating")
    ax_top.semilogy(T, pred, "-", lw=2.0, label="Fitted model replay")
    ax_top.set_ylabel("Resistance [Ohm]")
    ax_top.set_title("Measured vs Fitted Hysteresis Model (Specimen 100425_chip1_gap3)")
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc="best", fontsize=9)

    ax_bot.plot(T, res_log, ".", ms=3, alpha=0.8)
    ax_bot.axhline(0.0, color="k", lw=1)
    ax_bot.set_xlabel("Temperature [K]")
    ax_bot.set_ylabel("log10 residual")
    ax_bot.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_experimental_data(DATA_PATH)
    T = df["Temperature"].to_numpy(float)
    R = df["Resistance"].to_numpy(float)

    _make_lineage_figure(OUT_DIR / "model_lineage_flow.png")
    _make_table_extract_figure(df, OUT_DIR / "table_extract.png")
    r0_est, ea_est, rm_est = _make_transform_pipeline_figure(T, R, OUT_DIR / "fit_transform_pipeline.png")

    preset = json.loads(PRESET_PATH.read_text())
    _make_fit_overlay_figure(T, R, preset, OUT_DIR / "fit_overlay_with_residuals.png")

    print("Wrote figures to:", OUT_DIR)
    print(f"Data rows: {len(df)}")
    print(f"Seed estimates used in transformation figure: R0={r0_est:.6g}, Ea={ea_est:.6g}, Rm={rm_est:.6g}")


if __name__ == "__main__":
    main()

