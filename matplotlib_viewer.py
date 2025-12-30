#!/usr/bin/env python3
"""Open matplotlib plots for a sweep2d CSV locally."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plots


def _build_sweep_results(df: pd.DataFrame, x_label: str, y_label: str) -> dict:
    if x_label not in df.columns or y_label not in df.columns:
        raise ValueError(f"CSV must contain columns: {x_label}, {y_label}, freq_MHz")
    pivot = df.pivot(index=y_label, columns=x_label, values="freq_MHz")
    x_vals = pivot.columns.values.astype(float)
    y_vals = pivot.index.values.astype(float)
    freq = pivot.values.astype(float)
    return {"x_values": x_vals, "y_values": y_vals, "freq_MHz": freq}


def main() -> None:
    parser = argparse.ArgumentParser(description="Open matplotlib plots for a 2D sweep CSV.")
    parser.add_argument("--csv", required=True, help="Path to sweep2d_frequency.csv")
    parser.add_argument("--x-label", required=True, help="X-axis label (CSV column)")
    parser.add_argument("--y-label", required=True, help="Y-axis label (CSV column)")
    parser.add_argument("--title", default="", help="Optional title prefix")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    sweep = _build_sweep_results(df, args.x_label, args.y_label)
    plots.plot_frequency_2d(sweep, args.x_label, args.y_label, title_prefix=args.title or None)
    plt.show()


if __name__ == "__main__":
    main()
