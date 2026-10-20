"""Repository and archive validation used by ``neuristor validate`` and tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import ConfigError, load_toml
from .runs import RunRegistry, find_project_root


@dataclass
class ValidationReport:
    """Structured validation result suitable for terminal and CI output."""

    checked_configs: int = 0
    checked_runs: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_repository(project_root: str | Path | None = None) -> ValidationReport:
    """Validate every experiment TOML and every discoverable run manifest."""

    root = find_project_root(project_root)
    report = ValidationReport()
    experiment_root = root / "experiments"
    for path in sorted(experiment_root.rglob("*.toml")) if experiment_root.is_dir() else []:
        try:
            config = load_toml(path)
            if config.get("kind") == "sweep":
                base_path = Path(str(config["sweep"]["base_config"]))
                if not base_path.is_absolute():
                    base_path = path.parent / base_path
                load_toml(base_path)
            report.checked_configs += 1
        except (ConfigError, OSError, ValueError) as exc:
            report.errors.append(f"{path.relative_to(root)}: {exc}")

    registry = RunRegistry(root)
    seen: dict[tuple[str, str], Path] = {}
    for record in registry.discover():
        report.checked_runs += 1
        key = (record.id, record.storage)
        if key in seen:
            report.warnings.append(f"Duplicate run id {record.id!r} in storage lane {record.storage!r}")
        seen[key] = record.root
        manifest_name = "job.json" if record.legacy else "run.json"
        try:
            json.loads((record.root / manifest_name).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            report.errors.append(f"{record.root.relative_to(root)}: invalid {manifest_name}: {exc}")
        missing = [path for path in record.artifact_paths() if not path.exists()]
        if missing:
            report.errors.append(f"{record.id}: {len(missing)} declared artifact(s) are missing")
    return report
