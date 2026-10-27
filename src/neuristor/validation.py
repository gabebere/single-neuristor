"""Repository and archive validation used by ``neuristor validate`` and tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import shlex

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .config import ConfigError, load_toml
from .runs import RunRegistry, find_project_root


@dataclass
class ValidationReport:
    """Structured validation result suitable for terminal and CI output."""

    checked_configs: int = 0
    checked_runs: int = 0
    checked_evidence: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_repository(project_root: str | Path | None = None) -> ValidationReport:
    """Validate every experiment TOML and every discoverable run manifest."""

    root = find_project_root(project_root)
    report = ValidationReport()
    manifest = root / "docs/reproduction/reproduction.toml"
    if manifest.exists():
        try:
            report.checked_evidence, errors = validate_reproduction_index(root)
            report.errors.extend(errors)
        except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            report.errors.append(f"docs/reproduction/reproduction.toml: {exc}")
    experiment_root = root / "experiments"
    for path in sorted(experiment_root.rglob("*.toml")) if experiment_root.is_dir() else []:
        try:
            config = load_toml(path)
            report.errors.extend(validate_recipe_inputs(config, path))
            if 'profile_search' in config:
                from .orchestration.filament_profile import plan_filament_profile

                plan_filament_profile(config)
            if config.get("kind") == "reproduction":
                from .orchestration.presentation_reproduction import plan_presentation_reproduction

                plan_presentation_reproduction(config)
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


def validate_recipe_inputs(config: dict, recipe: Path) -> list[str]:
    """Catch missing declared relative inputs before an expensive workflow starts.

    This is a file-availability check, not a fit or a permission decision. Output
    destinations and internal loader metadata need not exist yet. Built-in preset
    names contain no relative path and are intentionally not treated as files.
    """
    errors = []

    def visit(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.startswith("_") or key == "output":
                    continue
                visit(item, f"{prefix}.{key}" if prefix else key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{prefix}[{index}]")
        elif isinstance(value, str) and value.startswith(("../", "./")):
            target = recipe.parent / value
            if not target.exists():
                errors.append(f"{recipe.name}: missing input {prefix} = {value}")

    visit(config)
    return errors


def validate_reproduction_index(root: Path) -> tuple[int, list[str]]:
    """Check evidence links and recipe commands without executing any experiment.

    This validates the navigation/reproduction contract, not scientific fit
    quality. Final figure assignment and calibration acceptance remain review
    decisions. Evidence paths must stay inside the project.
    """
    root = root.resolve()
    payload = tomllib.loads((root / "docs/reproduction/reproduction.toml").read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported paper evidence schema")
    records = payload.get("evidence", [])
    errors = []
    seen = set()
    for record in records:
        identifier = record.get("id", "<missing>")
        if identifier in seen or identifier == "<missing>":
            errors.append(f"paper evidence: duplicate or missing ID {identifier}")
        seen.add(identifier)
        if not record.get("claim_scope") or not record.get("outputs"):
            errors.append(f"{identifier}: missing scope or outputs")
        recipe = record.get("recipe")
        command = shlex.split(record.get("command", ""))
        if not command or command[0] != "neuristor":
            errors.append(f"{identifier}: command must use the supported neuristor CLI")
        if recipe and not any(a == "--config" and b == recipe for a, b in zip(command, command[1:])):
            errors.append(f"{identifier}: command does not refer to its declared recipe")
        paths = [*record.get("inputs", []), *([recipe] if recipe else [])]
        if record.get("reference"):
            paths.append(record["reference"])
        for relative in paths:
            target = (root / relative).resolve()
            if Path(relative).is_absolute() or not target.is_relative_to(root) or not target.exists():
                errors.append(f"{identifier}: missing or out-of-project path {relative}")
        for relative in record.get("outputs", []):
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                errors.append(f"{identifier}: output must be a relative bundle path: {relative}")
        reference = record.get("reference")
        if reference and (root / reference).is_dir():
            for relative in record.get("outputs", []):
                if not (root / reference / relative).is_file():
                    errors.append(f"{identifier}: reference is missing output {relative}")
            stage_outputs = record.get("stage_outputs", {})
            if stage_outputs:
                stages = json.loads((root / reference / "stages.json").read_text())
                for stage, names in stage_outputs.items():
                    relative_stage = stages.get(stage, "")
                    if not relative_stage or not (root / reference / relative_stage).resolve().is_relative_to(root):
                        errors.append(f"{identifier}: missing or invalid stage {stage}")
                        continue
                    for name in names:
                        target = (root / reference / relative_stage / name).resolve()
                        if not target.is_relative_to(root) or not target.is_file():
                            errors.append(f"{identifier}: reference is missing stage output {stage}/{name}")
    return len(records), errors
