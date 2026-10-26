"""Structural publication checks; never upload files or infer legal permission.

Experimental numerical data and our generated results are approved for sharing.
Third-party PDFs are not. The exact authored-PDF allowlist also catches copies
under another filename and PDF members inside a release archive. This is a
disclosure guard, not an adversarial file-format scanner or a history purge.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import zipfile

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def approved_pdf_hashes(root: Path) -> set[str]:
    """Load explicit byte identities of this project's authored documents."""
    return _pdf_hashes((root / "docs/distribution.toml").read_text())


def _pdf_hashes(text: str) -> set[str]:
    payload = tomllib.loads(text)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported distribution policy schema")
    return {entry["sha256"] for entry in payload.get("authored_pdf", [])}


def _private_name(name: str) -> bool:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return "private" in parts or ".git" in parts or ".." in parts or name.startswith(("/", "~"))


def _check_payload(name: str, stream, allowed: set[str], *, depth: int = 0) -> list[str]:
    """Inspect a stream, including nested ZIPs, without extracting any files."""
    if _private_name(name):
        return [f"{name}: private, absolute, or escaping distribution path"]
    prefix = stream.read(8)
    suffix = PurePosixPath(name).suffix.lower()
    if suffix == ".pdf" or prefix.startswith(b"%PDF-"):
        digest = hashlib.sha256(prefix)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return [] if digest.hexdigest() in allowed else [f"{name}: PDF is not an approved authored document"]
    if suffix == ".zip" or prefix.startswith(b"PK\x03\x04"):
        if depth >= 3:
            return [f"{name}: nested archive requires manual review"]
        # Prevent an unchecked memory spike in nested package inspection.
        tail = stream.read(256 * 1024 * 1024 + 1)
        if len(tail) > 256 * 1024 * 1024:
            return [f"{name}: nested ZIP exceeds inspection budget; distribute separately"]
        with zipfile.ZipFile(io.BytesIO(prefix + tail)) as archive:
            return _check_zip(archive, allowed, parent=name, depth=depth + 1)
    if suffix in {".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar"}:
        return [f"{name}: nested/opaque compressed file requires separate inspection"]
    return []


def _check_zip(archive: zipfile.ZipFile, allowed: set[str], *, parent: str, depth: int) -> list[str]:
    errors = []
    for entry in archive.infolist():
        if entry.is_dir():
            continue
        label = f"{parent}!{entry.filename}"
        if (entry.external_attr >> 16) & 0o170000 == 0o120000:
            errors.append(f"{label}: archive symlink requires manual review")
        else:
            with archive.open(entry) as stream:
                # Check the member name separately: the label contains its parent.
                if _private_name(entry.filename):
                    errors.append(f"{label}: private or escaping archive member")
                else:
                    errors.extend(_check_payload(label, stream, allowed, depth=depth))
    return errors


def validate_distribution_file(path: Path, allowed: set[str]) -> list[str]:
    """Check one proposed release file, streaming top-level ZIP/tar members.

    PDF hashes are checked in contents as well as extensions. Symbolic/hard links
    in release archives are refused: portable packages should contain actual
    approved files, not machine-specific or private targets.
    """
    try:
        if path.is_symlink():
            return [f"{path.name}: distribution symlink is not permitted"]
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                return _check_zip(archive, allowed, parent=path.name, depth=0)
        if path.name.endswith((".tar", ".tar.gz", ".tgz")):
            errors = []
            with tarfile.open(path, mode="r|*") as archive:
                for member in archive:
                    if member.isdir():
                        continue
                    if not member.isfile():
                        errors.append(f"{member.name}: archive link/special file requires review")
                        continue
                    with archive.extractfile(member) as stream:
                        errors.extend(_check_payload(member.name, stream, allowed))
            return errors
        with path.open("rb") as stream:
            return _check_payload(path.name, stream, allowed)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile, RuntimeError) as exc:
        return [f"{path.name}: distribution could not be inspected: {exc}"]


def validate_publication_index(root: Path) -> list[str]:
    """Check exact staged Git blobs, not potentially different working files.

    Does not inspect historical commits or upload anything. Run after staging;
    a clean index and an archive check remain separate from scientific review.
    """
    policy = subprocess.run(
        ["git", "show", ":docs/distribution.toml"], cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    allowed = _pdf_hashes(policy)
    listing = subprocess.run(["git", "ls-files", "--stage", "-z"], cwd=root, check=True, capture_output=True).stdout
    entries = []
    for row in listing.split(b"\0"):
        if row:
            meta, name = row.split(b"\t", 1)
            mode, oid, stage = meta.decode().split()
            entries.append((mode, oid, stage, name.decode()))
    sizes = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objectsize)"], cwd=root,
        input="".join(f"{oid}\n" for _, oid, _, _ in entries), text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    errors = []
    tracked_paths = {entry[3] for entry in entries}
    for (mode, oid, stage, name), size_line in zip(entries, sizes):
        size = int(size_line.split()[1])
        if stage != "0" or mode == "160000":
            errors.append(f"{name}: unresolved index entry or unchecked submodule")
        elif _private_name(name):
            errors.append(f"{name}: private material is staged")
        elif size >= 100 * 1024 * 1024:
            errors.append(f"{name}: exceeds ordinary Git file budget; use a reviewed Release asset")
        elif mode == "120000":
            target = subprocess.run(["git", "cat-file", "blob", oid], cwd=root, check=True, capture_output=True).stdout.decode()
            resolved = (root / name).parent.joinpath(target).resolve()
            if not resolved.is_relative_to(root.resolve()) or "private" in resolved.relative_to(root.resolve()).parts:
                errors.append(f"{name}: staged symlink reaches outside public project content")
            if name.lower().endswith(".pdf"):
                errors.append(f"{name}: PDF symlinks cannot be publicly staged")
        elif name.lower().endswith((".pdf", ".zip", ".tar", ".gz", ".tgz")):
            data = subprocess.run(["git", "cat-file", "blob", oid], cwd=root, check=True, capture_output=True).stdout
            errors.extend(_check_payload(name, io.BytesIO(data), allowed))
        if name.startswith("runs/published/") and name.endswith("/run.json") and mode == "100644":
            data = subprocess.run(["git", "cat-file", "blob", oid], cwd=root, check=True, capture_output=True).stdout
            errors.extend(manifest_case_errors(name, json.loads(data), tracked_paths))
    return errors


def manifest_case_errors(name: str, manifest: dict, tracked_paths: set[str]) -> list[str]:
    """Catch macOS-only path spellings before a bundle is cloned on Linux.

    Check only case mismatches, not release assets absent from source Git. The
    ordinary repository validator separately enforces complete local bundles.
    Original manifests and numerical bytes are never rewritten by this check.
    """
    folded = {path.casefold(): path for path in tracked_paths}
    errors = []
    # This frozen parent recorded the macOS directory spelling, while its child
    # declared Figures/. Both refer to the same verified PNG. Do not rewrite an
    # old manifest to fix packaging; RunRegistry resolves this unique legacy case.
    legacy_parent = "runs/published/20261007_083908_sample-r-t-to-screened-map-and-embedded-gifs_ec2d1e/run.json"
    legacy_artifact = ("stages/20261007_083920_sample-r-t-to-screened-map-and-embedded-gifs-poi_56b6f3/"
                       "figures/point_00025.png")
    for artifact in manifest.get("artifacts", []):
        target = str(PurePosixPath(name).parent / artifact["path"])
        if target not in tracked_paths and target.casefold() in folded:
            if name == legacy_parent and artifact["path"] == legacy_artifact:
                continue
            errors.append(f"{name}: artifact spelling {target} differs from staged {folded[target.casefold()]}")
    return errors
