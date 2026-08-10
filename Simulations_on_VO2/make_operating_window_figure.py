from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "public_jobs"
    / "20260707_145140_paper_frequency_f881d2"
    / "current_sweep_summary.csv"
)
OUTPUT = Path(__file__).resolve().parent / "assets" / "operating_window_response.png"


def main() -> None:
    data = pd.read_csv(SUMMARY)
    oscillatory = data[data["oscillatory"] == 1.0].copy()
    oscillatory["V_pp_V"] = oscillatory["V_pp_active_mV"] / 1000.0

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": "#2f3440",
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), sharex=True)
    colors = ["#7434db", "#2878c7", "#15945f"]
    columns = ["dominant_freq_MHz", "V_pp_V", "cycle_energy_pJ"]
    titles = ["Oscillation frequency", "Voltage amplitude", "Energy per cycle"]
    ylabels = ["Frequency (MHz)", r"Late-window $V_{pp}$ (V)", "Energy (pJ)"]

    for ax, color, column, title, ylabel in zip(
        axes, colors, columns, titles, ylabels, strict=True
    ):
        ax.axvspan(1250, 1350, color="#7b8494", alpha=0.12, lw=0)
        ax.axvspan(1350, 2150, color="#7434db", alpha=0.08, lw=0)
        ax.axvspan(2150, 2250, color="#ef4444", alpha=0.10, lw=0)
        ax.axvline(1350, color="#7434db", lw=1.1, ls="--", alpha=0.65)
        ax.axvline(2150, color="#7434db", lw=1.1, ls="--", alpha=0.65)
        ax.plot(
            oscillatory["I_target_uA"],
            oscillatory[column],
            color=color,
            marker="o",
            ms=6.5,
            lw=2.4,
        )
        ax.set_title(title, pad=9, weight="semibold")
        ax.set_xlabel(r"Imposed current ($\mu$A)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1250, 2250)
        ax.set_xticks([1300, 1500, 1700, 1900, 2100, 2200])
        ax.grid(True, color="#c9ced6", alpha=0.45, lw=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylim(32, 64)
    axes[1].set_ylim(1.82, 2.20)
    axes[2].set_ylim(33, 57)

    axes[0].annotate(
        "35.7",
        (1400, oscillatory.iloc[0]["dominant_freq_MHz"]),
        xytext=(7, 8),
        textcoords="offset points",
        color=colors[0],
        weight="bold",
    )
    axes[0].annotate(
        "60.7 MHz",
        (2100, oscillatory.iloc[-1]["dominant_freq_MHz"]),
        xytext=(-52, -19),
        textcoords="offset points",
        color=colors[0],
        weight="bold",
    )
    axes[2].annotate(
        "53.9",
        (1400, oscillatory.iloc[0]["cycle_energy_pJ"]),
        xytext=(7, 7),
        textcoords="offset points",
        color=colors[2],
        weight="bold",
    )
    axes[2].annotate(
        "36.3 pJ",
        (2100, oscillatory.iloc[-1]["cycle_energy_pJ"]),
        xytext=(-50, 8),
        textcoords="offset points",
        color=colors[2],
        weight="bold",
    )

    fig.suptitle(
        "Quantitative response inside the sustained oscillatory domain",
        fontsize=17,
        weight="semibold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.005,
        "Shading: insulating equilibrium | sustained oscillations | metallic lock",
        ha="center",
        color="#555b66",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96), w_pad=2.4)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
