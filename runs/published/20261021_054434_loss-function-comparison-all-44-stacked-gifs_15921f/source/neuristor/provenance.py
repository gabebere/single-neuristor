"""Freeze the complete installed source tree, including purpose-specific packages.

Run evidence must survive later code reorganizations. Recursive relative paths
keep snapshots importable and prevent silently omitting nested implementations.
This module records source; it never declares experimental agreement.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def package_source_files() -> list[Path]:
    """Return deterministic Python sources from this installed package."""
    return sorted(Path(__file__).parent.rglob("*.py"))


def snapshot_package(bundle, *, prefix: str = "source/neuristor") -> dict[str, str]:
    """Copy all package sources into a run and return relative SHA-256 hashes.

    Only Python source is copied: caches, datasets and generated media belong
    elsewhere in the bundle. Registering every file keeps manifest paths valid.
    """
    root = Path(__file__).parent
    hashes = {}
    for source in package_source_files():
        relative = f"{prefix}/{source.relative_to(root).as_posix()}"
        shutil.copy2(source, bundle.add_artifact(relative, label=f"Frozen source {source.relative_to(root)}"))
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes
