from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuristor.resistance_custom_analysis import fit_major_loop_resistance_params
from neuristor.workflows import run_resistance_fit


EXPECTED = {
    "R0": 0.8,
    "Ea_over_k": 2500.0,
    "Rm": 18.0,
    "w": 7.0,
    "Tc_K": 333.5,
    "beta": 0.30,
}


def _synthetic_major_loop() -> pd.DataFrame:
    cooling = np.linspace(365.0, 300.0, 70)
    heating = np.linspace(300.0, 365.0, 70)[1:]
    temperature = np.concatenate([cooling, heating])
    direction = np.concatenate([-np.ones_like(cooling), np.ones_like(heating)])
    fraction = 0.5 + 0.5 * np.tanh(
        EXPECTED["beta"] * (direction * EXPECTED["w"] / 2.0 + EXPECTED["Tc_K"] - temperature)
    )
    resistance = (
        EXPECTED["R0"] * np.exp(EXPECTED["Ea_over_k"] / temperature) * fraction + EXPECTED["Rm"]
    )
    return pd.DataFrame({"Temperature": temperature, "Resistance": resistance})


def test_major_loop_fit_recovers_synthetic_parameters() -> None:
    frame = _synthetic_major_loop()

    result, prediction, bootstrap = fit_major_loop_resistance_params(
        frame,
        seed=7,
        bootstrap_samples=8,
        bootstrap_block_size=4,
    )

    np.testing.assert_allclose(prediction, frame["Resistance"].to_numpy(dtype=float), rtol=2e-5)
    assert result.rmse_log10 < 1e-5
    assert result.r2_log10 > 0.999999
    assert result.start_branch == "metal"
    assert result.params.Rm == pytest.approx(EXPECTED["Rm"], rel=2e-4)
    assert result.params.w == pytest.approx(EXPECTED["w"], rel=2e-4)
    assert result.params.Tc_K == pytest.approx(EXPECTED["Tc_K"], rel=2e-4)
    assert len(bootstrap) == 8


def test_major_loop_workflow_archives_uncertainty_and_residuals(tmp_path: Path) -> None:
    data_path = tmp_path / "synthetic.tsv"
    _synthetic_major_loop().to_csv(data_path, sep="\t", index=False)

    bundle = run_resistance_fit(
        data_path,
        name="Synthetic major loop",
        method="major-loop",
        bootstrap_samples=6,
        seed=11,
        output_root=tmp_path / "runs",
        command="test fit",
    )

    assert (bundle.root / "parameter_summary.csv").is_file()
    assert (bundle.root / "parameter_bootstrap.csv").is_file()
    assert (bundle.root / "figures" / "resistance_fit.png").is_file()
    measured = pd.read_csv(bundle.root / "measured.csv")
    assert "log10_residual" in measured
    manifest = json.loads((bundle.root / "run.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["summary"]["fit_method"] == "major-loop-log-least-squares"
