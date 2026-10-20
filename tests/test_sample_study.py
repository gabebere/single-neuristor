"""Unit declarations, bounded workload, standalone media and the unified CLI."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.config import ConfigError, load_toml
from neuristor.map_animations import animation_controls_html, write_standalone_map
from neuristor.sample_study import load_study_rt, plan_sample_study
from neuristor.artifact_catalog import build_artifact_catalog
from neuristor.runs import RunBundle

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "experiments/studies/sample_parameter_map.toml"


def test_rt_import_converts_explicit_units_and_preserves_loop(tmp_path):
    temperature = np.r_[np.linspace(20, 100, 12), np.linspace(99, 20, 12)]
    resistance = np.linspace(1, 2, len(temperature))
    file = tmp_path / "curve.csv"
    pd.DataFrame(dict(temp=temperature, resistance=resistance)).to_csv(file, index=False)
    frame = load_study_rt(
        file,
        dict(temperature_column="temp", resistance_column="resistance", temperature_unit="C", resistance_unit="kohm"),
    )
    np.testing.assert_allclose(frame.Temperature, temperature + 273.15)
    np.testing.assert_allclose(frame.Resistance, resistance * 1000)
    assert frame.Temperature.iloc[12] < frame.Temperature.iloc[11]
    with pytest.raises(ConfigError, match="missing declared columns"):
        load_study_rt(file, {})
    with pytest.raises(ConfigError, match="temperature_unit"):
        load_study_rt(file, {"temperature_unit": "guess"})


def test_units_row_conflicts_and_missing_values_are_not_silent(tmp_path):
    source = ROOT / "data/experimental/100425_chip1_gap3.tsv"
    with pytest.raises(ConfigError, match="conflicts"):
        load_study_rt(source, {"temperature_unit": "C"})
    frame = load_study_rt(source, {})
    frame.loc[1, "Resistance"] = np.nan
    file = tmp_path / "curve.csv"
    frame.to_csv(file, index=False)
    with pytest.raises(ConfigError, match="finite numeric"):
        load_study_rt(file, {})


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("maps", "verification_dt_ns", [0.05, 0.025], "successively smaller"),
        ("maps", "maximum_frequency_MHz", float("nan"), "finite"),
        ("maps", "max_points", 1, "budget|exceeds"),
        ("animations", "dt_ns", 0.02, "half"),
        ("animations", "frames", 1, "integer"),
        ("input", "off_us", 0.3, "constant"),
        ("thermal", "noise_W_sqrt_s", 1e-10, "noise"),
    ],
)
def test_plan_rejects_unsafe_or_ignored_options(section, key, value, match):
    config = load_toml(RECIPE)
    config[section][key] = value
    with pytest.raises(ConfigError, match=match):
        plan_sample_study(config)


def test_plan_command_is_readonly_and_reports_grid(tmp_path):
    result = CliRunner().invoke(app, ["study", "plan", "--config", str(RECIPE)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["grid_points"] == 27
    config = load_toml(RECIPE)
    config["resistance"]["parameters"]["R0"] = 1
    with pytest.raises(ConfigError, match="not fitted constants"):
        plan_sample_study(config)


def test_standalone_catalog_key_order_embeds_exact_gif_bytes(tmp_path):
    import base64

    (tmp_path / "GIFs").mkdir()
    file = tmp_path / "GIFs/point_00001.gif"
    Image.new("RGB", (2, 2), "red").save(file)
    # Deliberately unsorted viewer keys and sorted on-disk JSON.
    catalog = {"1": dict(status="pass", gif="GIFs/point_00001.gif", caption="Point", detail="checked")}
    (tmp_path / "gif_catalog.json").write_text(json.dumps(catalog, sort_keys=True))
    (tmp_path / "parameter_maps.html").write_text("<html><body>" + animation_controls_html(catalog) + "</body></html>")
    target = tmp_path / "standalone.html"
    summary = write_standalone_map(tmp_path, target, introduction="<h1>Fit</h1>")
    html = target.read_text()
    encoded = re.search(r"data:image/gif;base64,([A-Za-z0-9+/=]+)", html).group(1)
    assert base64.b64decode(encoded) == file.read_bytes()
    assert summary["embedded_gifs"] == 1 and "<h1>Fit</h1>" in html
    assert "GIFs/point_00001.gif" not in html
    assert not list(tmp_path.glob("*.zip"))
    bad = copy.deepcopy(catalog)
    bad["1"]["gif"] = "../outside.gif"
    (tmp_path / "gif_catalog.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="outside"):
        write_standalone_map(tmp_path, target)


def test_sample_study_cli_completes_empty_domain_without_zip(tmp_path):
    # Deliberately weak input yields no sustained cycles. This exercises the
    # real fitter/solver and verifies an honest empty map, not mock physics.
    arguments = [
        "study",
        "run",
        "--config",
        str(RECIPE),
        "--output-root",
        str(tmp_path / "runs"),
        "--set",
        "input.amplitude_uA=1",
        "--set",
        "animations.frames=2",
        "--set",
        "maps.axes.C_th_pJ_per_K.values=[0.047873236]",
        "--set",
        "maps.axes.S_e_mW_per_K.values=[0.0036751265]",
        "--set",
        "maps.axes.C_pF.values=[16.4]",
    ]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output + repr(result.exception)
    root = next((tmp_path / "runs").iterdir())
    manifest = json.loads((root / "run.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["summary"]["embedded_gifs"] == 0
    assert manifest["summary"]["coverage_complete"] is True
    assert manifest["summary"]["dynamic_calibration_adopted"] is False
    assert (root / "parameter_maps.html").is_file()
    assert not list(root.rglob("*.zip"))
    assert all((root / entry["path"]).is_file() for entry in manifest["artifacts"])
    assert len(list((root / "stages").glob("*/run.json"))) == 3


def test_catalog_lists_evidence_without_mutating_or_copying_it(tmp_path):
    # The registry accepts a scratch project-like root; no Git mutation.
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='scratch'\n")
    bundle = RunBundle.create(
        name="A <map>",
        model="current",
        kind="study",
        config={},
        output_root=tmp_path / "runs",
        command="test",
        project_root=tmp_path,
    )
    bundle.write_text("START_HERE.html", "<html>map</html>", label="Viewer")
    bundle.complete()
    original = (bundle.root / "run.json").read_bytes()
    summary = build_artifact_catalog(tmp_path)
    assert summary["run_records"] == 1
    catalog = (tmp_path / "artifacts/index.html").read_text()
    assert "A &lt;map&gt;" in catalog and "Sample studies" in catalog
    assert "successful fit" not in catalog
    assert (bundle.root / "run.json").read_bytes() == original
    assert set(path.name for path in (tmp_path / "artifacts").iterdir()) == {
        "index.html", "runs.csv", "files.csv", "source_inventory.csv"
    }
    assert pd.read_csv(tmp_path / "artifacts/source_inventory.csv").empty
