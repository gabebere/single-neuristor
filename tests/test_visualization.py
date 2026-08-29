from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from neuristor.visualization import (
    animate_current_resistance_temperature,
    plot_lab_detection_window_trace,
    plot_lab_oscillation_bracket,
    plot_resistance_temperature_trajectory,
)


def _trace() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_us": [0.0, 1.0, 2.0, 3.0, 4.0],
            "current_uA": [0.0, 600.0, 600.0, 600.0, 0.0],
            "voltage_V": [0.0, 1.0, 4.0, 1.2, 0.0],
            "temperature_K": [325.0, 330.0, 345.0, 335.0, 325.0],
            "resistance_ohm": [16_000.0, 12_000.0, 1_400.0, 5_000.0, 16_000.0],
        }
    )


def test_resistance_temperature_trajectory_writes_png(tmp_path: Path) -> None:
    path = plot_resistance_temperature_trajectory(_trace(), tmp_path / "trajectory.png")
    assert path.is_file()
    with Image.open(path) as image:
        assert image.format == "PNG"


def test_current_rt_animation_writes_multiframe_gif(tmp_path: Path) -> None:
    path = animate_current_resistance_temperature(
        _trace(), tmp_path / "trajectory.gif", frame_count=4, duration_s=0.4
    )
    assert path.is_file()
    with Image.open(path) as image:
        assert image.format == "GIF"
        assert image.n_frames == 4


def test_lab_detection_window_and_bracket_plots_write_png(tmp_path: Path) -> None:
    trace = pd.DataFrame(
        {
            "time_ns": [-50.0, 0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0],
            "input_current_uA": [5.0, 20.0, 100.0, 101.0, 100.0, 101.0, 100.0, 5.0],
            "output_voltage_mV": [-4.0, 10.0, 220.0, 160.0, 225.0, 155.0, 220.0, -4.0],
        }
    )
    summary = pd.Series(
        {
                "nominal_drive_mV": 300.0,
                "current_baseline_uA": 5.0,
                "current_step_uA": 95.0,
                "voltage_baseline_mV": -4.0,
            "oscillation_detected": True,
            "oscillation_frequency_MHz": 41.7,
            "oscillation_peak_count": 8.0,
            "oscillation_period_cv": 0.0203,
        }
    )
    pre_onset = summary.copy()
    pre_onset["nominal_drive_mV"] = 250.0
    pre_onset["oscillation_detected"] = False
    pre_onset["oscillation_frequency_MHz"] = float("nan")
    pre_onset["oscillation_peak_count"] = 1.0
    pre_onset["oscillation_period_cv"] = float("nan")
    outputs = [
        plot_lab_detection_window_trace(trace, summary, tmp_path / "onset.png"),
        plot_lab_detection_window_trace(trace, pre_onset, tmp_path / "pre_onset.png"),
        plot_lab_oscillation_bracket(
            trace,
            pre_onset,
            trace,
            summary,
            tmp_path / "bracket.png",
        ),
    ]
    for path in outputs:
        assert path.is_file()
        with Image.open(path) as image:
            assert image.format == "PNG"
