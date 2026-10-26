"""Repository locations and read-only resolution of pre-consolidation paths.

Frozen research records retain their original paths and hashes. Resolve their
known former repository locations at read time instead of editing the evidence.
No scientific calculation or filesystem mutation belongs in this module.
"""

from __future__ import annotations

from pathlib import Path


LOCAL_RUNS = Path("runs/local")
PUBLISHED_RUNS = Path("runs/published")
LEGACY_RUNS = Path("runs/legacy")
CATALOG = Path("runs/catalog")

FORMER_LOCATIONS = {
    "public_jobs": PUBLISHED_RUNS,
    "jobs": LEGACY_RUNS,
    "outputs": Path("runs/scratch"),
    "output": Path("runs/scratch/documents"),
    "tmp": Path("runs/scratch/tmp"),
    "presets": Path("data/parameters"),
    "new_samples": Path("data/experimental/additional_samples"),
    "references": Path("archive/references"),
    "legacy_scripts": Path("archive/scripts"),
    "Simulations_on_VO2": Path("docs/presentation"),
    "paper": Path("docs/reproduction"),
    "environments": Path("docs/environment"),
    "artifacts": CATALOG,
    "results": Path("runs"),
}


def resolve_recorded_path(path: str | Path, project_root: Path) -> Path:
    """Resolve a known former location within this checkout without rewriting it.

    Existing paths win, including explicitly chosen output directories. Unknown
    and out-of-project paths stay missing; matching a basename must never silently
    substitute unrelated evidence. Old flat ``runs/ID`` paths map to local runs.
    """
    root = project_root.resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.exists():
        return candidate
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return candidate
    if not relative.parts or ".." in relative.parts:
        return candidate
    first, *tail = relative.parts
    if first in FORMER_LOCATIONS:
        return root / FORMER_LOCATIONS[first] / Path(*tail)
    if first == "runs" and tail and tail[0] not in {"local", "published", "legacy", "scratch", "catalog"}:
        return root / LOCAL_RUNS / Path(*tail)
    return candidate
