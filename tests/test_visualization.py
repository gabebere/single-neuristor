from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from neuristor.visualization import (
    animate_current_resistance_temperature,
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
