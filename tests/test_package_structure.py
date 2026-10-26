"""Canonical source imports, paper navigation, and frozen provenance."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

from neuristor.provenance import package_source_files, snapshot_package
from neuristor.runs import RunBundle
from neuristor.validation import validate_reproduction_index

ROOT = Path(__file__).resolve().parents[1]
def test_owned_modules_import_from_the_canonical_source_tree():
    package = ROOT / "src/neuristor"
    for source in package_source_files():
        relative = source.relative_to(package).with_suffix("")
        if relative.name == "__init__":
            continue
        module = importlib.import_module("neuristor." + ".".join(relative.parts))
        assert Path(module.__file__).resolve() == source.resolve()


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
        [sys.executable, "-I", "-c", f"""import sys, importlib, importlib.abc, pkgutil
sys.path.insert(0, {str(bundle.root / 'source')!r})
class WithoutStreamlit(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'streamlit' or fullname.startswith('streamlit.'):
            raise ModuleNotFoundError('Streamlit is not installed', name=fullname)
sys.meta_path.insert(0, WithoutStreamlit())
import neuristor
for module in pkgutil.walk_packages(neuristor.__path__, 'neuristor.'):
    importlib.import_module(module.name)
from neuristor.cli import app
from typer.testing import CliRunner
result = CliRunner().invoke(app, ['--help'])
assert result.exit_code == 0, result.output
assert 'dashboard' not in result.output and 'playground' not in result.output
assert not any(name == 'streamlit' or name.startswith('streamlit.') for name in sys.modules)
"""],
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
