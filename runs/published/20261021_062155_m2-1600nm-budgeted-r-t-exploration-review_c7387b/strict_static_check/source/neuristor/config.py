"""Experiment configuration loading, validation, and command-line overrides.

TOML files are the durable user-facing description of an experiment.  This
module intentionally returns ordinary dictionaries: they serialize cleanly,
are easy for humans and agents to inspect, and keep the numerical dataclasses
in the physics modules independent from file-format concerns.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]


SCHEMA_VERSION = 1
SUPPORTED_MODELS = frozenset({"current", "voltage"})
SUPPORTED_KINDS = frozenset({"simulation", "sweep"})


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid or incomplete."""


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load and validate an experiment TOML file.

    The private ``_source`` key records where relative paths should resolve. It
    is removed before the configuration is written to a public run bundle.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"Experiment config does not exist: {source}")
    with source.open("rb") as handle:
        config = tomllib.load(handle)
    config["_source"] = str(source)
    validate_config(config)
    return config


def resolved_copy(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy without private loader metadata."""

    return {key: copy.deepcopy(value) for key, value in config.items() if not str(key).startswith("_")}


def source_directory(config: Mapping[str, Any]) -> Path:
    """Directory used to resolve paths referenced by a configuration."""

    source = config.get("_source")
    return Path(str(source)).resolve().parent if source else Path.cwd().resolve()


def deep_get(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    """Read ``a.b.c`` from a nested mapping."""

    value: Any = mapping
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigError(f"Unknown configuration path: {dotted_path}")
        value = value[key]
    return value


def deep_set(mapping: MutableMapping[str, Any], dotted_path: str, value: Any, *, create: bool = False) -> None:
    """Set ``a.b.c`` in a nested mapping with typo protection by default."""

    keys = dotted_path.split(".")
    cursor: MutableMapping[str, Any] = mapping
    for key in keys[:-1]:
        child = cursor.get(key)
        if child is None and create:
            child = {}
            cursor[key] = child
        if not isinstance(child, MutableMapping):
            raise ConfigError(f"Configuration path is not a table: {dotted_path}")
        cursor = child
    if not create and keys[-1] not in cursor:
        raise ConfigError(f"Unknown configuration path: {dotted_path}")
    cursor[keys[-1]] = value


def parse_override_value(text: str) -> Any:
    """Parse one CLI override value using TOML scalar syntax.

    Examples: ``600`` -> int, ``0.5`` -> float, ``true`` -> bool. Bare words
    are treated as strings so ``--set resistance.preset=yuanhang`` stays easy.
    """

    try:
        return tomllib.loads(f"value = {text}\n")["value"]
    except Exception:
        return text


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply repeated ``path=value`` overrides and revalidate the result."""

    updated = copy.deepcopy(dict(config))
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override must have path=value form: {item!r}")
        path, raw_value = item.split("=", 1)
        deep_set(updated, path.strip(), parse_override_value(raw_value.strip()))
    validate_config(updated)
    return updated


def _require_table(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Missing TOML table: [{key}]")
    return value


def _positive(table: Mapping[str, Any], key: str, table_name: str, *, allow_zero: bool = False) -> float:
    if key not in table:
        raise ConfigError(f"Missing required value: {table_name}.{key}")
    try:
        value = float(table[key])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{table_name}.{key} must be numeric") from exc
    invalid = value < 0.0 if allow_zero else value <= 0.0
    if invalid:
        relation = ">= 0" if allow_zero else "> 0"
        raise ConfigError(f"{table_name}.{key} must be {relation}")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the stable cross-workflow schema and core physical units."""

    version = int(config.get("schema_version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise ConfigError(f"Unsupported schema_version={version}; expected {SCHEMA_VERSION}")
    if not str(config.get("name", "")).strip():
        raise ConfigError("Experiment config requires a non-empty 'name'")
    kind = str(config.get("kind", "simulation")).lower()
    model = str(config.get("model", "")).lower()
    if kind not in SUPPORTED_KINDS:
        raise ConfigError(f"Unsupported kind={kind!r}; choose from {sorted(SUPPORTED_KINDS)}")
    if model not in SUPPORTED_MODELS:
        raise ConfigError(f"Unsupported model={model!r}; choose from {sorted(SUPPORTED_MODELS)}")

    if kind == "sweep":
        sweep = _require_table(config, "sweep")
        if not sweep.get("base_config"):
            raise ConfigError("Sweep configs require sweep.base_config")
        axes = sweep.get("axes")
        if not isinstance(axes, list) or not axes:
            raise ConfigError("Sweep configs require one or more [[sweep.axes]] tables")
        paths: list[str] = []
        for axis in axes:
            if not isinstance(axis, Mapping) or not str(axis.get("path", "")).strip():
                raise ConfigError("Every sweep axis requires a dotted 'path'")
            paths.append(str(axis["path"]))
            values = axis.get("values")
            has_range = all(key in axis for key in ("start", "stop", "step"))
            if not (isinstance(values, list) and values) and not has_range:
                raise ConfigError("Every sweep axis requires values=[...] or start/stop/step")
        if len(paths) != len(set(paths)):
            raise ConfigError("Sweep axis paths must be unique")
        return

    time = _require_table(config, "time")
    input_table = _require_table(config, "input")
    electrical = _require_table(config, "electrical")
    thermal = _require_table(config, "thermal")
    _positive(time, "dt_ns", "time")
    _positive(time, "duration_us", "time")
    _positive(thermal, "C_th_pJ_per_K", "thermal")
    _positive(thermal, "S_e_mW_per_K", "thermal", allow_zero=True)
    if model == "current":
        _positive(input_table, "amplitude_uA", "input", allow_zero=True)
        _positive(electrical, "C_pF", "electrical", allow_zero=True)
    else:
        _positive(input_table, "amplitude_V", "input", allow_zero=True)
        _positive(electrical, "C_pF", "electrical")
        _positive(electrical, "R_series_kohm", "electrical")


def config_as_pretty_json(config: Mapping[str, Any]) -> str:
    """Human-readable resolved configuration for CLI display and reports."""

    return json.dumps(resolved_copy(config), indent=2, sort_keys=True)
