"""Standard run bundles and discovery of current plus historical jobs.

Every new command writes the same small contract.  ``run.json`` is the index;
all paths inside it are relative to the run directory, making bundles portable
across machines and GitHub clones.  The registry also understands historical
``public_jobs/*/job.json`` records so the new dashboard does not discard years
of provenance.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


RUN_SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    """Convert NumPy-like values and non-finite floats to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest parent containing ``pyproject.toml`` and ``src``."""

    cursor = Path(start or Path.cwd()).expanduser().resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for candidate in (cursor, *cursor.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    return cursor


def _slug(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return normalized[:48] or "run"


def _git_provenance(project_root: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(command("status", "--porcelain")),
    }


@dataclass
class RunBundle:
    """Writable, self-describing output directory for one CLI workflow."""

    root: Path
    manifest: dict[str, Any]
    _artifacts: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        name: str,
        model: str,
        kind: str,
        config: Mapping[str, Any],
        output_root: str | Path,
        command: str,
        project_root: str | Path | None = None,
    ) -> "RunBundle":
        project = find_project_root(project_root)
        output = Path(output_root).expanduser()
        if not output.is_absolute():
            output = project / output
        output.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc)
        run_id = f"{timestamp:%Y%m%d_%H%M%S}_{_slug(name)}_{uuid.uuid4().hex[:6]}"
        root = output / run_id
        (root / "figures").mkdir(parents=True, exist_ok=False)
        manifest: dict[str, Any] = {
            "schema_version": RUN_SCHEMA_VERSION,
            "id": run_id,
            "name": name,
            "model": model,
            "kind": kind,
            "status": "running",
            "created_at": timestamp.isoformat(),
            "updated_at": timestamp.isoformat(),
            "command": command,
            "storage": "local",
            "provenance": _git_provenance(project),
            "artifacts": [],
        }
        bundle = cls(root=root, manifest=manifest)
        bundle.write_json("resolved_config.json", dict(config), label="Resolved configuration")
        bundle._save_manifest()
        return bundle

    @property
    def id(self) -> str:
        return str(self.manifest["id"])

    def path(self, relative: str | Path) -> Path:
        target = (self.root / relative).resolve()
        try:
            target.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError(f"Artifact path escapes run directory: {relative}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def add_artifact(self, relative: str | Path, *, label: str, media_type: str | None = None) -> Path:
        rel = Path(relative).as_posix()
        entry = {"path": rel, "label": label}
        if media_type:
            entry["media_type"] = media_type
        if entry not in self._artifacts:
            self._artifacts.append(entry)
        return self.path(relative)

    def write_json(self, relative: str | Path, payload: Any, *, label: str) -> Path:
        path = self.add_artifact(relative, label=label, media_type="application/json")
        path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")
        return path

    def write_text(self, relative: str | Path, text: str, *, label: str, media_type: str = "text/markdown") -> Path:
        path = self.add_artifact(relative, label=label, media_type=media_type)
        path.write_text(text.rstrip() + "\n")
        return path

    def register_file(self, relative: str | Path, *, label: str, media_type: str | None = None) -> Path:
        path = self.add_artifact(relative, label=label, media_type=media_type)
        if not path.exists():
            raise FileNotFoundError(f"Cannot register missing artifact: {path}")
        return path

    def complete(self, *, summary: Mapping[str, Any] | None = None) -> None:
        self.manifest["status"] = "completed"
        self.manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        if summary is not None:
            self.manifest["summary"] = dict(summary)
        self._save_manifest()

    def fail(self, error: BaseException) -> None:
        self.manifest["status"] = "failed"
        self.manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.manifest["error"] = f"{type(error).__name__}: {error}"
        self._save_manifest()

    def _save_manifest(self) -> None:
        self.manifest["artifacts"] = list(self._artifacts)
        (self.root / "run.json").write_text(
            json.dumps(_json_safe(self.manifest), indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
        )


@dataclass(frozen=True)
class RunRecord:
    """Read-only normalized view of a new run or historical Streamlit job."""

    id: str
    name: str
    model: str
    kind: str
    status: str
    created_at: str
    root: Path
    storage: str
    manifest: Mapping[str, Any]
    legacy: bool = False

    @property
    def summary(self) -> Mapping[str, Any]:
        value = self.manifest.get("summary", {})
        return value if isinstance(value, Mapping) else {}

    def files(self) -> list[Path]:
        return sorted(path for path in self.root.rglob("*") if path.is_file() and path.name != ".DS_Store")

    def artifact_paths(self) -> list[Path]:
        entries = self.manifest.get("outputs" if self.legacy else "artifacts", [])
        paths: list[Path] = []
        declared = False
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, Mapping) or not entry.get("path"):
                    continue
                declared = True
                path = Path(str(entry["path"]))
                if not path.is_absolute():
                    if self.legacy and path.parts and path.parts[0] in {"public_jobs", "jobs", "runs"}:
                        path = find_project_root(self.root) / path
                    else:
                        path = self.root / path
                paths.append(path.resolve())
        return paths if declared else self.files()


class RunRegistry:
    """Discover, inspect, and publish standard runs plus historical jobs."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = find_project_root(project_root)

    def discover(self, *, include_private_legacy: bool = True) -> list[RunRecord]:
        records: list[RunRecord] = []
        records.extend(self._new_records(self.project_root / "runs", storage="local"))
        records.extend(self._new_records(self.project_root / "public_jobs", storage="public"))
        records.extend(self._legacy_records(self.project_root / "public_jobs", storage="public"))
        if include_private_legacy:
            records.extend(self._legacy_records(self.project_root / "jobs", storage="legacy-local"))
        # Publishing intentionally leaves the exploratory source in ``runs``.
        # Prefer the reviewed public copy so the archive does not show one run twice.
        priority = {"public": 0, "local": 1, "legacy-local": 2}
        ordered = sorted(records, key=lambda record: (priority.get(record.storage, 9), record.id))
        unique: dict[str, RunRecord] = {}
        for record in ordered:
            unique.setdefault(record.id, record)
        return sorted(unique.values(), key=lambda record: record.created_at, reverse=True)

    def get(self, run_id: str) -> RunRecord:
        matches = [record for record in self.discover() if record.id == run_id]
        if not matches:
            raise KeyError(f"Unknown run: {run_id}")
        return matches[0]

    def publish(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        if record.storage == "public":
            return record
        destination = self.project_root / "public_jobs" / record.id
        if destination.exists():
            raise FileExistsError(f"Public destination already exists: {destination}")
        shutil.copytree(record.root, destination)
        manifest_path = destination / "run.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            manifest["storage"] = "public"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest = json.loads((destination / "run.json").read_text())
        return RunRecord(
            id=str(manifest.get("id", destination.name)),
            name=str(manifest.get("name", destination.name)),
            model=str(manifest.get("model", "unknown")),
            kind=str(manifest.get("kind", "unknown")),
            status=str(manifest.get("status", "unknown")),
            created_at=str(manifest.get("created_at", "")),
            root=destination,
            storage="public",
            manifest=manifest,
        )

    def _new_records(self, root: Path, *, storage: str) -> Iterable[RunRecord]:
        if not root.is_dir():
            return []
        records: list[RunRecord] = []
        for manifest_path in root.glob("*/run.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
                records.append(
                    RunRecord(
                        id=str(manifest.get("id", manifest_path.parent.name)),
                        name=str(manifest.get("name", manifest_path.parent.name)),
                        model=str(manifest.get("model", "unknown")),
                        kind=str(manifest.get("kind", "unknown")),
                        status=str(manifest.get("status", "unknown")),
                        created_at=str(manifest.get("created_at", "")),
                        root=manifest_path.parent,
                        storage=str(manifest.get("storage", storage)),
                        manifest=manifest,
                    )
                )
            except (OSError, json.JSONDecodeError):
                continue
        return records

    def _legacy_records(self, root: Path, *, storage: str) -> Iterable[RunRecord]:
        if not root.is_dir():
            return []
        records: list[RunRecord] = []
        for manifest_path in root.glob("*/job.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
                records.append(
                    RunRecord(
                        id=str(manifest.get("id", manifest_path.parent.name)),
                        name=str(
                            manifest.get("name")
                            or manifest.get("params", {}).get("job_name")
                            or manifest_path.parent.name
                        ),
                        model=str(
                            manifest.get("source_model") or manifest.get("params", {}).get("source_model") or "unknown"
                        ),
                        kind=str(manifest.get("type", "legacy")),
                        status=str(manifest.get("status", "unknown")),
                        created_at=str(manifest.get("created_at", "")),
                        root=manifest_path.parent,
                        storage=storage,
                        manifest=manifest,
                        legacy=True,
                    )
                )
            except (OSError, json.JSONDecodeError):
                continue
        return records
