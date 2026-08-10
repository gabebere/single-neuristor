from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuristor.current_results_digitizer import digitize_directory, traces_to_dataframe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Digitize the lab current-result PNGs into trace CSVs.")
    parser.add_argument("--image-dir", default="data/Current Results")
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    return parser


def _plot_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes[0, 0]
    ax.plot(summary_df["frame_index"], summary_df["plateau_current_visible_uA"], "o-", label="Visible green plateau")
    ax.plot(summary_df["frame_index"], summary_df["current_inferred_uA"], "-", label="Inferred sweep")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Current (uA)")
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(summary_df["current_inferred_uA"], summary_df["v_plateau_mean_mV"], "o-")
    ax.set_xlabel("Inferred current (uA)")
    ax.set_ylabel("Plateau mean V_out (mV)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot(summary_df["current_inferred_uA"], summary_df["v_plateau_vpp_mV"], "o-")
    ax.set_xlabel("Inferred current (uA)")
    ax.set_ylabel("Plateau Vpp (mV)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(summary_df["current_inferred_uA"], summary_df["v_turnoff_min_300_450_mV"], "o-")
    ax.set_xlabel("Inferred current (uA)")
    ax.set_ylabel("Turn-off minimum (mV)")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    image_dir = Path(args.image_dir)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"digitized_current_results_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds, traces, summary_df = digitize_directory(image_dir)
    trace_df = traces_to_dataframe(traces, summary_df)

    summary_path = out_dir / "digitized_summary.csv"
    trace_path = out_dir / "digitized_traces.csv"
    summary_df.to_csv(summary_path, index=False)
    trace_df.to_csv(trace_path, index=False)
    _plot_summary(summary_df, out_dir / "digitized_summary.png")

    print(f"Plot bounds: left={bounds.x_left_axis}, right={bounds.x_right_axis}, top={bounds.y_top_axis}, bottom={bounds.y_bottom_axis}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote traces: {trace_path}")
    first = summary_df.iloc[0]
    last = summary_df.iloc[-1]
    print(
        "Inferred current sweep: "
        f"frame 0 -> {float(first['current_inferred_uA']):.3f} uA, "
        f"frame {int(last['frame_index'])} -> {float(last['current_inferred_uA']):.3f} uA"
    )


if __name__ == "__main__":
    main()
