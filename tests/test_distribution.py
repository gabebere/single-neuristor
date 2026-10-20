"""Keep private paper PDFs out of staged source and portable release packages."""

import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
import zipfile

from neuristor.analysis.distribution import manifest_case_errors, validate_distribution_file, validate_publication_index
from neuristor.validation import validate_recipe_inputs
from neuristor.runs import RunBundle, resolve_archive_case


def test_unapproved_pdf_even_with_an_innocent_extension(tmp_path):
    file = tmp_path / "measurement.dat"
    file.write_bytes(b"%PDF-1.7 not our authored document")
    assert "not an approved" in validate_distribution_file(file, set())[0]
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    assert validate_distribution_file(file, {digest}) == []


def test_zip_checks_nested_pdf_and_paths(tmp_path):
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("paper.pdf", b"%PDF-1.7 private")
    file = tmp_path / "results.zip"
    with zipfile.ZipFile(file, "w") as archive:
        archive.writestr("nested.zip", nested.getvalue())
        archive.writestr("../outside.csv", b"numbers")
        archive.writestr("private/note.txt", b"restricted")
    errors = validate_distribution_file(file, set())
    assert len(errors) == 3
    assert any("nested.zip!paper.pdf" in error for error in errors)


def test_tar_checks_links_and_pdf_without_extracting(tmp_path):
    file = tmp_path / "results.tar.gz"
    with tarfile.open(file, "w:gz") as archive:
        info = tarfile.TarInfo("measurements.csv")
        data = b"time,current\n0,1\n"
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo("paper.pdf")
        data = b"%PDF-1.7 forbidden"
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo("private_link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/some/private/paper.pdf"
        archive.addfile(info)
    errors = validate_distribution_file(file, set())
    assert len(errors) == 2
    assert not (tmp_path / "measurements.csv").exists()


def test_clean_zip_and_approved_authored_pdf(tmp_path):
    data = b"%PDF-1.7 our report"
    file = tmp_path / "results.zip"
    with zipfile.ZipFile(file, "w") as archive:
        archive.writestr("report.pdf", data)
        archive.writestr("results.csv", b"frequency,amplitude\n1,2")
    assert validate_distribution_file(file, {hashlib.sha256(data).hexdigest()}) == []


def test_publication_checks_index_not_working_copy(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/distribution.toml").write_text("schema_version = 1\n")
    git("add", "docs/distribution.toml")
    file = tmp_path / "reference.pdf"
    file.write_bytes(b"%PDF-1.7 unapproved")
    git("add", "reference.pdf")
    file.write_bytes(b"not the staged bytes")
    assert "reference.pdf" in validate_publication_index(tmp_path)[0]
    # An unstaged allowlist cannot approve a PDF in the staged release.
    digest = hashlib.sha256(b"%PDF-1.7 unapproved").hexdigest()
    (tmp_path / "docs/distribution.toml").write_text(
        f'schema_version = 1\n[[authored_pdf]]\nsha256 = "{digest}"\n'
    )
    assert "reference.pdf" in validate_publication_index(tmp_path)[0]
    git("rm", "--cached", "-f", "reference.pdf")
    (tmp_path / "private").mkdir()
    (tmp_path / "private/note.md").write_text("unapproved")
    git("add", "private/note.md")
    assert "private material" in validate_publication_index(tmp_path)[0]


def test_publication_rejects_private_or_absolute_symlink(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/distribution.toml").write_text("schema_version = 1\n")
    (tmp_path / "reference").symlink_to("private/paper.pdf")
    (tmp_path / "absolute").symlink_to("/outside")
    subprocess.run(["git", "add", "reference", "absolute", "docs/distribution.toml"], cwd=tmp_path, check=True, capture_output=True)
    assert len(validate_publication_index(tmp_path)) == 2


def test_recipe_validator_catches_unavailable_local_history(tmp_path):
    recipe = tmp_path / "recipe.toml"
    config = {"steady": {"seed_table": "./ignored-history/parameters.csv"},
              "resistance": {"preset": "yuanhang"}, "output": {"root": "./new-output"}}
    errors = validate_recipe_inputs(config, recipe)
    assert len(errors) == 1 and "steady.seed_table" in errors[0]
    (tmp_path / "ignored-history").mkdir()
    (tmp_path / "ignored-history/parameters.csv").write_text("candidate\ncontrol\n")
    assert validate_recipe_inputs(config, recipe) == []


def test_publication_catches_case_sensitive_bundle_paths():
    manifest = {"artifacts": [{"path": "Figures/point.png"}]}
    assert len(manifest_case_errors("public_jobs/id/run.json", manifest, {"public_jobs/id/figures/point.png"})) == 1
    assert manifest_case_errors("public_jobs/id/run.json", manifest, {"public_jobs/id/Figures/point.png"}) == []


def test_legacy_case_resolution_keeps_missing_files_missing(tmp_path):
    (tmp_path / "Figures").mkdir()
    actual = tmp_path / "Figures/point.png"
    actual.write_bytes(b"a historical image")
    assert resolve_archive_case(tmp_path / "figures/point.png").read_bytes() == actual.read_bytes()
    assert not resolve_archive_case(tmp_path / "figures/genuinely_missing.png").exists()


def test_new_bundle_uses_the_requested_media_directory_spelling(tmp_path):
    bundle = RunBundle.create(name="Case portability", model="none", kind="analysis",
                              config={}, output_root=tmp_path, command="test")
    assert not (bundle.root / "figures").exists()
    bundle.path("Figures/point.png").write_bytes(b"generated image")
    assert "Figures" in {p.name for p in bundle.root.iterdir()}
