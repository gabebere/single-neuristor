"""Input-only reconstruction, immutable stage resume, and portable viewer bindings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest
from typer.testing import CliRunner

from neuristor.analysis.alternating_amplitudes import cycle_table, summarize_cycles
from neuristor.cli import app
from neuristor.config import ConfigError, load_toml
from neuristor.interfaces.presentation_server import (
    rebuilt_presentation,
    create_presentation_server,
    check_presentation_files,
)
from neuristor.orchestration import presentation_reproduction as reproduction
from neuristor.runs import RunBundle

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "experiments/reproduce_presentation.toml"


def test_full_plan_preserves_recorded_workloads_without_result_inputs():
    plan = reproduction.plan_presentation_reproduction(load_toml(RECIPE))
    assert plan["recipes"]["map_grid"]["grid_points"] == 12167
    assert sum(v["grid_points"] for k, v in plan["recipes"].items() if k in reproduction.MICA_NAMES) == 2616
    assert len(plan["stages"]) == 18
    assert all(not p.startswith(("runs/", "private/")) for p in plan["source_input_hashes"])
    assert {"numpy", "torch"}.issubset(plan["environment"])
    cfg = load_toml(RECIPE)
    assert cfg["reproduction"]["static_fit"]["bootstrap_samples"] == 1000
    for key in ("frequency_search", "amplitude_search"):
        child = load_toml(RECIPE.parent / cfg["reproduction"]["recipes"][key])
        assert child["steady"]["maximum_evaluations"] == 256
        assert child["steady"]["verification_dt_ns"] == [0.025, 0.0125, 0.00625]
    result = CliRunner().invoke(app, ["reproduce-presentation", "--plan"])
    assert result.exit_code == 0, result.output
    assert "Several hours" in result.output


def test_plan_rejects_derived_inputs_before_any_work():
    config = load_toml(RECIPE)
    config["reproduction"]["inputs"]["specimen_data"] = "../runs/published"
    with pytest.raises(ConfigError, match="source input"):
        reproduction.plan_presentation_reproduction(config)


def test_every_child_recipe_binds_to_newly_computed_dependencies(tmp_path):
    config = load_toml(RECIPE)
    parent = RunBundle(tmp_path, {}, [])
    stages = {key: tmp_path / key for key in ("specimen_fit", "mica_fit")}
    for key in config["reproduction"]["recipes"]:
        child = reproduction._child_config(key, config, parent, stages)
        declaration = json.dumps(child)
        assert "runs/published" not in declaration, key
        assert "public_jobs" not in declaration, key


def test_individual_primary_cycle_detector_distinguishes_alternation():
    t = np.arange(0.0, 500.0, 0.05)
    phase = (t % 25.0) / 25.0
    pulse = np.exp(-(((phase - 0.35) / 0.09) ** 2))
    amplitudes = np.where((t // 25).astype(int) % 2, 350.0, 240.0)
    fixed = cycle_table(t, 250.0 * pulse, [0.0, 500.0])
    alternating = cycle_table(t, amplitudes * pulse, [0.0, 500.0])
    assert not summarize_cycles(fixed)["strong_alternation"]
    assert summarize_cycles(alternating)["strong_alternation"]


def test_mica_independent_checks_use_the_supported_solver_api(tmp_path, monkeypatch):
    from neuristor import workflows
    from neuristor.current_drive_sim import CurrentDriveParams

    config = load_toml(RECIPE)
    config["reproduction"]["profile"] = "smoke"
    monkeypatch.setattr(reproduction, "_child_config", lambda *args: {})
    monkeypatch.setattr(workflows, "_current_params_from_config", lambda cfg: (CurrentDriveParams(),))
    bundle = reproduction._mica_checks(config, None, {}, tmp_path)
    assert bundle.manifest["status"] == "completed"
    assert bundle.manifest["summary"]["integrations"] == 3
    assert "minimum_proximity_slope" in (bundle.root / "checks.csv").read_text()


def test_alternating_comparison_exports_its_gallery_dependencies(tmp_path):
    from types import SimpleNamespace
    import pandas as pd
    from neuristor.orchestration.alternating_amplitudes import _write_comparison

    bundle = RunBundle.create(
        name="Comparison", model="current", kind="analysis", config={}, output_root=tmp_path, command="test"
    )
    dataset = SimpleNamespace(
        nominal_drives_mV=np.array([100.0, 200.0]),
        time_ns=np.array([0.0, 1.0, 2.0]),
        voltage_mV=np.array([[0.0, 0.0], [10.0, 20.0], [0.0, 0.0]]),
    )
    for label in dataset.nominal_drives_mV:
        path = bundle.path(f"dt_0.025ns/current_{label:g}.npz")
        np.savez(
            path,
            time_ns=[0.0, 0.5, 1.5, 2.0],
            simulated_voltage_mV=[0.0, 5.0, 5.0, 0.0],
            resistance_ohm=[100.0, 200.0, 200.0, 100.0],
        )
    _write_comparison(bundle, dataset, [80.0, 160.0], 0.025)
    samples = pd.read_csv(bundle.root / "traces.csv")
    assert samples.source_mV.nunique() == 2
    assert samples[samples.source_mV == 200.0].measured_voltage_mV.tolist() == [0.0, 10.0, 10.0, 0.0]
    assert (bundle.root / "comparison.html").is_file()
    assert '<script src="https://cdn.plot.ly' not in (bundle.root / "comparison.html").read_text()


def _synthetic_plan(config):
    return dict(
        fingerprint="same-source-and-inputs", profile=config["reproduction"]["profile"], time_allowance="Test only"
    )


@pytest.mark.parametrize("tamper", [None, "stage", "inputs", "source"])
@pytest.mark.parametrize("profile", ["full", "smoke"])
def test_resume_reuses_only_intact_stages_and_never_activates_partial_work(tmp_path, monkeypatch, tamper, profile):
    # Only stage orchestration is synthetic. Real numerical integration is tested
    # by the solver suites and the separate empty-results end-to-end run.
    project = tmp_path
    (project / "src").mkdir()
    (project / "pyproject.toml").write_text('[project]\nname="reproduction-test"\n')
    source = project / "input.csv"
    source.write_text("value\n1\n")
    cfg = dict(
        name="Synthetic reproduction",
        reproduction=dict(
            profile=profile,
            inputs=dict(
                specimen_data=str(source),
                mica_data=str(source),
                search_seeds=str(source),
                alternating_parameters=str(source),
                device_photo=str(source),
            ),
        ),
    )
    monkeypatch.setattr(reproduction, "plan_presentation_reproduction", _synthetic_plan)
    monkeypatch.setattr(reproduction, "STAGES", ("first", "second"))

    def assemble(parent, config, stages):
        parent.write_text("START_HERE.html", "fresh results", label="Synthetic viewer")
        parent.write_json(
            "presentation_manifest.json",
            dict(
                schema_version=1,
                profile=profile,
                entry="START_HERE.html",
                routes={k: "START_HERE.html" for k in reproduction.ROUTES},
            ),
            label="Synthetic links",
        )

    monkeypatch.setattr(reproduction, "_assemble", assemble)
    calls = []

    def execute(key, config, parent, stages, output):
        calls.append(key)
        if key == "second" and calls.count("second") == 1:
            raise RuntimeError("simulated interruption")
        child = RunBundle.create(
            name=key,
            model="current",
            kind="analysis",
            config={},
            output_root=output,
            command="test",
            project_root=project,
        )
        child.write_text("data.csv", "value\n2\n", label="Synthetic result")
        child.complete()
        return child

    monkeypatch.setattr(reproduction, "_execute_stage", execute)
    with pytest.raises(RuntimeError, match="interruption"):
        reproduction.run_presentation_reproduction(cfg, project_root=project, output_root=project / "runs/local")
    parent = next((project / "runs/local").glob("*/run.json")).parent
    assert not (project / "runs/local/presentation.json").exists()
    if tamper:
        if tamper == "stage":
            next((parent / "stages").rglob("data.csv")).write_text("changed result")
        elif tamper == "inputs":
            (parent / "inputs/specimen_data.tsv").write_text("changed input")
        else:
            monkeypatch.setattr(
                reproduction,
                "plan_presentation_reproduction",
                lambda cfg: dict(_synthetic_plan(cfg), fingerprint="changed"),
            )
        with pytest.raises(ValueError, match="changed"):
            reproduction.run_presentation_reproduction(cfg, project_root=project, resume=parent)
        assert calls == ["first", "second"]
        assert not (project / "runs/local/presentation.json").exists()
        return
    resumed = reproduction.run_presentation_reproduction(cfg, project_root=project, resume=parent)
    assert calls == ["first", "second", "second"]
    assert resumed.manifest["status"] == "completed"
    assert "error" not in resumed.manifest
    assert json.loads((parent / "status.json").read_text())["state"] == "completed"
    assert (project / "runs/local/presentation.json").exists() == (profile == "full")
    if profile == "full":
        assert rebuilt_presentation(project)[0] == parent


def _install_fake_reproduction(project, *, profile="full", status="completed"):
    location = project / "runs/local/reproduced"
    location.mkdir(parents=True)
    routes = {route: "fresh/" + route[1:] + ".html" for route in reproduction.ROUTES}
    for relative in routes.values():
        path = location / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("freshly generated viewer")
    (location / "START_HERE.html").write_text("fresh result entrypoint")
    (location / "run.json").write_text(json.dumps(dict(kind="presentation-reproduction", status=status)))
    payload = dict(schema_version=1, profile=profile, routes=routes, entry="START_HERE.html")
    manifest = location / "presentation_manifest.json"
    manifest.write_text(json.dumps(payload))
    pointer = project / "runs/local/presentation.json"
    pointer.write_text(
        json.dumps(
            dict(
                schema_version=1,
                bundle="runs/local/reproduced",
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
        )
    )
    return location, payload, pointer


def test_fresh_presentation_routes_do_not_require_published_bundles(tmp_path):
    location, manifest, _ = _install_fake_reproduction(tmp_path)
    pdf = tmp_path / "docs/presentation/main.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    assert not (tmp_path / "runs/published").exists()
    assert check_presentation_files(tmp_path) == pdf
    with create_presentation_server(tmp_path, port=0) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            for route in reproduction.ROUTES:
                with urlopen(base + route) as response:
                    assert response.read() == b"freshly generated viewer"
            with urlopen(base + "/reproduction") as response:
                assert response.read() == b"fresh result entrypoint"
            private = tmp_path / "private"
            private.mkdir()
            (private / "secret").write_text("private")
            (location / "leak").symlink_to(private / "secret")
            with pytest.raises(HTTPError) as error:
                urlopen(base + "/runs/local/reproduced/leak")
            assert error.value.code == 403
        finally:
            server.shutdown()
            worker.join()


@pytest.mark.parametrize("profile,status", [("smoke", "completed"), ("full", "running"), ("full", "failed")])
def test_unfinished_or_smoke_runs_cannot_replace_presentation(tmp_path, profile, status):
    _install_fake_reproduction(tmp_path, profile=profile, status=status)
    with pytest.raises(ValueError, match="incomplete"):
        rebuilt_presentation(tmp_path)


def test_tampered_or_escaping_viewer_manifest_is_rejected(tmp_path):
    location, manifest, pointer = _install_fake_reproduction(tmp_path)
    manifest["routes"]["/loss-gifs"] = "../../../../private/secret.html"
    path = location / "presentation_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed"):
        rebuilt_presentation(tmp_path)
    metadata = json.loads(pointer.read_text())
    metadata["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    pointer.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="escapes"):
        rebuilt_presentation(tmp_path)


@pytest.mark.parametrize("payload", [[], {}, {"schema_version": 1, "bundle": 17}])
def test_malformed_reproduction_pointer_has_actionable_error(tmp_path, payload):
    _, _, pointer = _install_fake_reproduction(tmp_path)
    pointer.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid"):
        rebuilt_presentation(tmp_path)
