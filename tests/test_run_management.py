"""Recoverable run lifecycle and authenticated local-manager HTTP behavior.

All mutations use synthetic runs in temporary projects. Never use a real
laboratory bundle to test deletion, restoration or unsafe-path rejection.
"""

from __future__ import annotations

import json
import re
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.interfaces.run_manager import create_manager
from neuristor.orchestration.artifact_catalog import build_artifact_catalog
from neuristor.runs import RunBundle, RunRegistry


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='management-test'\n")
    return tmp_path


def bundle(project, *, complete=True):
    run = RunBundle.create(name="Synthetic <run>", model="current", kind="study", config={},
                           output_root=project / "runs/local", command="test", project_root=project)
    run.write_text("stages/child/data.csv", "measurement,value\n1,2", label="Synthetic child")
    if complete:
        run.complete()
    return run


def test_delete_and_restore_preserve_manifest_child_data_and_catalog(project):
    run = bundle(project)
    original = {p.relative_to(run.root): p.read_bytes() for p in run.root.rglob("*") if p.is_file()}
    registry = RunRegistry(project)
    metadata = registry.trash(run.id)
    assert not run.root.exists()
    assert registry.discover() == []
    assert registry.trashed() == [metadata]
    build_artifact_catalog(project)
    assert run.id not in (project / "runs/catalog/files.csv").read_text()
    registry.restore(metadata["trash_id"])
    assert registry.trashed() == []
    assert {p.relative_to(run.root): p.read_bytes() for p in run.root.rglob("*") if p.is_file()} == original
    assert registry.get(run.id).status == "completed"


def test_published_and_running_runs_are_protected(project):
    registry = RunRegistry(project)
    running = bundle(project, complete=False)
    with pytest.raises(ValueError, match="Running"):
        registry.trash(running.id)
    public = bundle(project)
    published = registry.publish(public.id)
    with pytest.raises(ValueError, match="Published"):
        registry.trash(public.id)
    assert public.root.exists() and published.root.exists() and running.root.exists()


def test_failed_runs_are_recoverable_and_restore_never_overwrites(project):
    run = bundle(project, complete=False)
    run.fail(ValueError("Synthetic failure"))
    registry = RunRegistry(project)
    metadata = registry.trash(run.id)
    run.root.mkdir()
    with pytest.raises(FileExistsError, match="nothing was overwritten"):
        registry.restore(metadata["trash_id"])
    assert len(registry.trashed()) == 1


def test_recipe_references_and_symlink_lanes_block_deletion(project, tmp_path):
    run = bundle(project)
    recipes = project / "experiments"
    recipes.mkdir()
    recipe = recipes / "depends.toml"
    recipe.write_text(f'source = "../runs/local/{run.id}/data.csv"')
    with pytest.raises(ValueError, match="recipe references"):
        RunRegistry(project).trash(run.id)
    recipe.unlink()
    # Move the synthetic folder aside; an alias must not grant deletion rights.
    outside = project / "other"
    run.root.rename(outside)
    run.root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        RunRegistry(project).trash(run.id)
    assert (outside / "run.json").exists()


@pytest.mark.parametrize("identifier", ["../other", "/", "", ".trash", ".."])
def test_invalid_restore_id_cannot_escape_trash(project, identifier):
    with pytest.raises(ValueError, match="Invalid trash ID"):
        RunRegistry(project).restore(identifier)


def test_cli_requires_confirmation_and_can_restore(project, monkeypatch):
    monkeypatch.chdir(project)
    run = bundle(project)
    runner = CliRunner()
    declined = runner.invoke(app, ["runs", "delete", run.id], input="n\n")
    assert declined.exit_code != 0 and run.root.exists()
    approved = runner.invoke(app, ["runs", "delete", run.id], input="y\n")
    assert approved.exit_code == 0, approved.output
    trash_id = RunRegistry(project).trashed()[0]["trash_id"]
    restored = runner.invoke(app, ["runs", "restore", trash_id])
    assert restored.exit_code == 0 and run.root.exists()


def test_static_catalog_actions_are_disabled_and_published_protected(project):
    local = bundle(project)
    public = bundle(project)
    RunRegistry(project).publish(public.id)
    build_artifact_catalog(project)
    page = (project / "runs/catalog/index.html").read_text()
    assert f'button disabled data-run-id="{local.id}"' in page
    assert f'data-run-id="{public.id}"' not in page
    assert '"token":' not in page
    assert "Protected" in page and "neuristor runs manage" in page


def test_manager_http_confirmation_auth_delete_restore_and_containment(project):
    run = bundle(project)
    server = create_manager(project)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"

    def request(path, *, data=None, auth=True, foreign=False, confirmed=True):
        headers = {}
        if data is not None:
            headers = {"Content-Type": "application/json", "Origin": "https://other.invalid" if foreign else origin}
            if auth:
                headers["X-Neuristor-Token"] = token
            data = json.dumps({**data, "confirm": confirmed}).encode()
        with urlopen(Request(origin + path, data=data, headers=headers), timeout=5) as response:
            return response.read().decode()

    try:
        page = request("/runs/catalog/index.html")
        token = json.loads(re.search(r'enableRunManager\((\{[^\n]+\})\);', page)[1])["token"]
        for options in ({"auth": False}, {"foreign": True}):
            with pytest.raises(HTTPError) as error:
                request("/api/trash", data={"run_id": run.id}, **options)
            assert error.value.code == 403
            assert run.root.exists()
        with pytest.raises(HTTPError) as error:
            request("/api/trash", data={"run_id": run.id}, confirmed=False)
        assert error.value.code == 400 and run.root.exists()
        deleted = json.loads(request("/api/trash", data={"run_id": run.id}))
        assert not run.root.exists()
        assert json.loads(request("/api/trash"))[0]["trash_id"] == deleted["trash_id"]
        request("/api/restore", data={"trash_id": deleted["trash_id"]})
        assert run.root.exists()
        for path in ("/.git/config", "/runs/.trash/", "/../../outside"):
            with pytest.raises(HTTPError) as error:
                request(path)
            assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            with urlopen(Request(origin + "/runs/catalog/index.html", headers={"Host": "other.invalid"}), timeout=5):
                pass
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
