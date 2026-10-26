"""Grouped source, import compatibility, paper navigation, and frozen provenance."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from neuristor.provenance import package_source_files, snapshot_package
from neuristor.runs import RunBundle
from neuristor.validation import validate_reproduction_index

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "analysis": "experimental_waveforms extrema metrics lab_estimates model_validation oscillation_audit lab_replay physical_maps",
    "fitting": "resistance_custom_analysis parameter_inference joint_inference steady_inference current_domain_search",
    "plotting": "visualization replay_plots map_animations",
    "orchestration": "map_workflows sample_study artifact_catalog",
    "interfaces": "dashboard playground",
    "legacy": "plots sample_library",
}


@pytest.mark.parametrize("group,module", [(g, m) for g, modules in GROUPS.items() for m in modules.split()])
def test_old_import_is_the_actual_grouped_module(group, module):
    old = importlib.import_module(f"neuristor.{module}")
    canonical = importlib.import_module(f"neuristor.{group}.{module}")
    assert old is canonical


def test_alias_preserves_private_names_and_monkeypatches(monkeypatch):
    old = importlib.import_module("neuristor.map_animations")
    canonical = importlib.import_module("neuristor.plotting.map_animations")
    sentinel = object()
    monkeypatch.setattr(old, "_structure_test_marker", sentinel, raising=False)
    assert canonical._structure_test_marker is sentinel
    assert old.replay_animation_batch.__globals__ is canonical.replay_animation_batch.__globals__


def test_snapshot_has_complete_nested_sources_and_valid_hashes(tmp_path):
    bundle = RunBundle.create(name="Source snapshot test", model="none", kind="analysis", config={}, output_root=tmp_path, command="test")
    hashes = snapshot_package(bundle)
    assert "source/neuristor/analysis/extrema.py" in hashes
    assert "source/neuristor/plotting/map_animations.py" in hashes
    assert len(hashes) == len(package_source_files())
    for relative, expected in hashes.items():
        assert hashlib.sha256(bundle.path(relative).read_bytes()).hexdigest() == expected
    bundle.complete()
    paths = {a["path"] for a in json.loads((bundle.root / "run.json").read_text())["artifacts"]}
    assert set(hashes).issubset(paths)
    # -I removes the editable checkout/PYTHONPATH. The only project source on
    # this child interpreter's path is the newly frozen recursive package.
    result = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(bundle.root / 'source')!r}); "
         "import neuristor.cli; import neuristor.extrema as a; import neuristor.analysis.extrema as b; assert a is b"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_project_navigation_targets_exist():
    from urllib.parse import unquote

    for page in (ROOT / "README.md", ROOT / "docs/README.md", ROOT / "runs/index.html"):
        source = page.read_text()
        targets = re.findall(r"\]\(([^)]+)\)", source) + re.findall(r'href="([^\"]+)"', source)
        for relative in targets:
            if relative.startswith(("http:", "https:", "#")):
                continue
            target = (page.parent / unquote(relative.split("#", 1)[0])).resolve()
            # The local catalog is explicitly generated after checkout.
            if target.is_relative_to(ROOT / "runs/catalog"):
                continue
            assert target.exists(), (page.name, relative)


def test_paper_evidence_index_has_valid_paths_and_recipe_commands():
    count, errors = validate_reproduction_index(ROOT)
    assert count == 5
    assert errors == []


def test_paper_index_rejects_invalid_links_and_recipe_commands(tmp_path):
    (tmp_path / "docs/reproduction").mkdir(parents=True)
    (tmp_path / "docs/reproduction/reproduction.toml").write_text('''schema_version = 1
[[evidence]]
id = "invalid"
claim_scope = "Navigation test, not a scientific claim"
inputs = ["missing.csv", "../outside.csv"]
recipe = "missing.toml"
command = "neuristor study run --config different.toml"
outputs = ["../escape.csv"]
''')
    count, errors = validate_reproduction_index(tmp_path)
    assert count == 1
    assert any("declared recipe" in e for e in errors)
    assert any("out-of-project" in e for e in errors)
    assert any("relative bundle" in e for e in errors)
