"""Stable heatmap displays and non-mutating historical viewer revisions."""

from dataclasses import asdict
import itertools
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.current_drive_sim import CurrentDriveParams
from neuristor.analysis.physical_maps import AXES, heatmap_display_ranges
from neuristor.orchestration.map_viewers import refresh_map_viewer, refresh_existing_maps
from neuristor.orchestration.artifact_catalog import build_artifact_catalog
from neuristor.runs import RunBundle, RunRegistry


def plot_payloads(html):
    """Read actual offline Plotly data/layout, including every slider action."""
    decoder = json.JSONDecoder()
    result = {}
    for match in re.finditer(r'Plotly.newPlot\(\s*"(map-[^"]+)"\s*,\s*', html):
        rest = html[match.end():]
        data, used = decoder.raw_decode(rest)
        layout, _ = decoder.raw_decode(rest[used:].lstrip()[1:].lstrip())
        result[match[1]] = (data, layout)
    return result


@pytest.fixture
def source(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='map-viewer-test'\n")
    axes = [np.array([1., 2.]), np.array([.003, .006]), np.array([1., 3.])]
    config = dict(maps=dict(maximum_frequency_MHz=110., fixed_values=dict(zip(AXES, [1., .003, 1.]))),
                  input=dict(amplitude_uA=500.))
    run = RunBundle.create(name="Synthetic map", model="current", kind="analysis", config=config,
                           output_root=tmp_path / "runs/local", command="test", project_root=tmp_path)
    rows = [dict(zip(AXES, values), point_index=i, included=i != 7,
                 frequency_MHz=(20. if values[0] == 1 else 70.) + values[2] - 1 if i != 7 else 1000.,
                 extrema_vpp_mV=100. + i * 10 if i != 7 else 10000., temperature_max_K=350.)
            for i, values in enumerate(itertools.product(*axes))]
    pd.DataFrame(rows).to_csv(run.add_artifact("grid.csv", label="Synthetic grid"), index=False)
    pd.DataFrame(columns=["point_index"]).to_csv(run.add_artifact("verification.csv", label="Synthetic checks"), index=False)
    run.write_json("axis_values.json", dict(zip(AXES, [a.tolist() for a in axes])), label="Axes")
    run.write_json("physics_parameters.json", asdict(CurrentDriveParams()), label="Physics")
    run.write_text("GIFs/point_00000.gif", "GIF89a", label="Synthetic GIF signature")
    run.write_text("Figures/point_00000.png", "synthetic", label="Synthetic media")
    catalog = {"0": dict(point_index=0, gif="GIFs/point_00000.gif", png="Figures/point_00000.png", caption="Test")}
    run.write_json("gif_catalog.json", catalog, label="Media")
    run.write_text("parameter_maps.html", "<html><body>historical viewer</body></html>", label="Old viewer")
    run.complete()
    return tmp_path, run


def test_metric_ranges_ignore_excluded_values_and_handle_empty_singleton():
    axes = [np.array([1.])] * 3
    table = pd.DataFrame(dict(included=[True, False], frequency_MHz=[20., 1000.], extrema_vpp_mV=[100., 9999.]))
    ranges = heatmap_display_ranges(table, axes)
    assert ranges["color_limits"]["frequency_MHz"]["low"] < 20 < ranges["color_limits"]["frequency_MHz"]["high"]
    assert ranges["color_limits"]["frequency_MHz"]["high"] < 21
    assert all(a < b for a, b in ranges["log_axis_ranges"].values())
    table.included = False
    empty = heatmap_display_ranges(table, axes)
    assert all(not item["has_values"] for item in empty["color_limits"].values())


def test_refresh_cli_preserves_source_and_stabilizes_every_slider(source, monkeypatch):
    project, original = source
    before = {p.relative_to(original.root): p.read_bytes() for p in original.root.rglob("*") if p.is_file()}
    def forbidden(*args, **kwargs):
        pytest.fail("Viewer refresh must not run a simulation")
    monkeypatch.setattr("neuristor.analysis.physical_maps.evaluate_map", forbidden)
    result = CliRunner().invoke(app, ["analyze", "refresh-maps", "--source", str(original.root),
                                      "--output-root", str(project / "runs/local")])
    assert result.exit_code == 0, result.output + repr(result.exception)
    viewer = next(r for r in RunRegistry(project).discover() if r.id != original.id)
    payloads = plot_payloads((viewer.root / "parameter_maps.html").read_text())
    assert len(payloads) == 4
    assert viewer.summary["resimulated"] is False
    assert viewer.summary["retained_points"] == 7
    for key in ("map-slice-0", "map-slice-1", "map-slice-2"):
        data, layout = payloads[key]
        assert [d["zmin"] for d in data] == [20., 100.]
        assert [d["zmax"] for d in data] == [72., 160.]
        for step in layout["sliders"][0]["steps"]:
            trace_update, layout_update = step["args"]
            assert trace_update["zmin"] == [20., 100.]
            assert trace_update["zmax"] == [72., 160.]
            assert trace_update["zauto"] == [False, False]
            for axis in ("xaxis", "xaxis2", "yaxis", "yaxis2"):
                assert layout[axis]["autorange"] is False
                assert layout_update[axis + ".autorange"] is False
                assert layout_update[axis + ".range"] == layout[axis]["range"]
            ids = np.asarray(trace_update["customdata"][0])
            z = np.asarray(trace_update["z"][0], float)
            assert set(ids.ravel()) <= {-1, *range(7)}
            assert np.isfinite(z).sum() == (ids >= 0).sum()
    for name in ("grid.csv", "verification.csv", "physics_parameters.json", "axis_values.json"):
        assert (viewer.root / name).read_bytes() == before[Path(name)]
    assert before == {p.relative_to(original.root): p.read_bytes() for p in original.root.rglob("*") if p.is_file()}
    assert "../" + original.id + "/GIFs/point_00000.gif" in (viewer.root / "parameter_maps.html").read_text().replace(r"\u002f", "/")
    registry = RunRegistry(project)
    assert "links to this run's GIFs" in registry.trash_reason(registry.get(original.id))
    build_artifact_catalog(project)
    assert "Updated fixed-scale viewer" in (project / "runs/catalog/index.html").read_text()
    assert refresh_existing_maps(project_root=project, output_root=project / "runs/local") == []
    registry.trash(viewer.id)
    assert registry.trash_reason(registry.get(original.id)) is None


def test_refreshed_study_preserves_standalone_format_without_copying_gifs(source):
    project, media = source
    parent = RunBundle.create(name="Synthetic sample", model="current", kind="study", config={},
                              output_root=project / "runs/local", command="test", project_root=project)
    import shutil
    shutil.copytree(media.root, parent.root / "stages/grid")
    parent.write_json("stages.json", dict(grid="stages/grid", media="stages/grid"), label="Stages")
    parent.write_text("parameter_maps.html", "old standalone export", label="Old HTML")
    parent.complete()
    viewer = refresh_map_viewer(parent.root, output_root=project / "runs/local")
    assert viewer.manifest["summary"]["standalone"]
    assert viewer.manifest["summary"]["embedded_gifs"] == 1
    html = (viewer.root / "parameter_maps.html").read_text()
    assert "data:image/gif;base64,R0lGODlh" in html
    assert not (viewer.root / "GIFs").exists()
    assert not (viewer.root / "embedded.html").exists()
    assert len(plot_payloads(html)) == 4


def test_refresh_refuses_failed_source(source):
    project, original = source
    original.fail(RuntimeError("synthetic"))
    with pytest.raises(ValueError, match="completed"):
        refresh_map_viewer(original.root, output_root=project / "runs/local")
