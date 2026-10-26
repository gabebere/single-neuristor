"""Relocated evidence remains readable without modifying frozen records."""

import json
import pytest

from neuristor.layout import resolve_recorded_path
from neuristor.runs import RunBundle, RunRegistry


def project_at(root):
    (root / "src").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='layout-test'\n")
    return root


def test_former_output_recipe_joins_new_lane_and_publishes(tmp_path):
    project = project_at(tmp_path)
    bundle = RunBundle.create(name="Old recipe", model="current", kind="simulation", config={},
                              output_root="runs", command="test", project_root=project)
    bundle.write_text("report.md", "preserved evidence", label="Report")
    bundle.complete()
    assert bundle.root.parent == project / "runs/local"
    published = RunRegistry(project).publish(bundle.id)
    assert published.root == project / "runs/published" / bundle.id
    assert (published.root / "report.md").read_bytes() == (bundle.root / "report.md").read_bytes()
    assert RunRegistry(project).get(bundle.id).storage == "public"
    with pytest.raises(ValueError, match="Published"):
        RunRegistry(project).trash(bundle.id)


def test_frozen_legacy_paths_resolve_without_rewriting_manifest(tmp_path):
    project = project_at(tmp_path)
    legacy = project / "runs/legacy/old-job"
    legacy.mkdir(parents=True)
    data = legacy / "data.csv"
    data.write_bytes(b"time_ns,current_uA\n0,500\n")
    manifest = legacy / "job.json"
    manifest.write_text(json.dumps({"id": "old-job", "status": "completed", "outputs": [
        {"path": "jobs/old-job/data.csv"}, {"path": str(project / "jobs/old-job/data.csv")},
    ]}))
    original = manifest.read_bytes()
    record = RunRegistry(project).get("old-job")
    assert record.artifact_paths() == [data, data]
    assert manifest.read_bytes() == original
    assert not (project / "jobs").exists()


def test_frozen_viewer_dependency_still_protects_relocated_local_run(tmp_path):
    project = project_at(tmp_path)
    source = RunBundle.create(name="Media", model="current", kind="simulation", config={},
                              output_root="runs/local", command="test", project_root=project)
    source.complete()
    viewer = RunBundle.create(name="Viewer", model="current", kind="analysis", config={},
                              output_root="runs/local", command="test", project_root=project)
    viewer.complete(summary={"media_dependencies": [str(project / "runs" / source.id)]})
    with pytest.raises(ValueError, match="GIFs"):
        RunRegistry(project).trash(source.id)


def test_old_trash_sidecar_restores_to_consolidated_lane(tmp_path):
    project = project_at(tmp_path)
    bundle = RunBundle.create(name="Recover", model="current", kind="simulation", config={},
                              output_root="runs/local", command="test", project_root=project)
    bundle.complete()
    original = (bundle.root / "run.json").read_bytes()
    registry = RunRegistry(project)
    metadata = registry.trash(bundle.id)
    sidecar = project / "runs/local/.trash" / metadata["trash_id"] / "deletion.json"
    metadata["original_path"] = f"runs/{bundle.id}"
    sidecar.write_text(json.dumps(metadata))
    registry.restore(metadata["trash_id"])
    assert (bundle.root / "run.json").read_bytes() == original


def test_resolution_never_substitutes_unknown_or_external_evidence(tmp_path):
    project = project_at(tmp_path)
    target = project / "runs/published/known/data.csv"
    target.parent.mkdir(parents=True)
    target.write_text("numerical evidence")
    assert resolve_recorded_path(project / "public_jobs/known/data.csv", project) == target
    outside = project.parent / "another-project/public_jobs/known/data.csv"
    assert resolve_recorded_path(outside, project) == outside
    missing = project / "unknown/known/data.csv"
    assert resolve_recorded_path(missing, project) == missing


def test_symlink_at_consolidated_parent_cannot_grant_deletion(tmp_path):
    project = project_at(tmp_path)
    bundle = RunBundle.create(name="Retain", model="current", kind="simulation", config={},
                              output_root="runs/local", command="test", project_root=project)
    bundle.complete()
    (project / "runs").rename(project / "external")
    (project / "runs").symlink_to(project / "external", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        RunRegistry(project).trash(bundle.id)
    assert (project / "external/local" / bundle.id / "run.json").is_file()
