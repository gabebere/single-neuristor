# The simulation library

Most users should edit [experiment recipes](../../experiments/README.md), not
these files. Developers can navigate by responsibility:

| Location | What it does |
|---|---|
| `model.py` | Original voltage-source circuit and the shared resistance/hysteresis model. Scientific source of truth. |
| `current_drive_sim.py` | Ideal-current extension, including the exact zero-capacitance limit. Scientific source of truth. |
| `analysis/` | Import measurements, identify settled cycles, estimate parameters, and screen simulation outputs. |
| `fitting/` | Static R(T) fits and recorded dynamic parameter-search methods. |
| `plotting/` | Figures, voltage/R(T) animations, and standalone HTML media embedding. |
| `orchestration/` | Coordinate existing calculations into study/map bundles and build the artifact catalog. |
| `interfaces/` | Streamlit archive browser/replay interface and local artifact manager with recoverable Delete/Restore. |
| `legacy/` | Older plotting/sample helpers retained for historical compatibility, not the new study path. |
| `workflows.py` | Stable public workflow API; unit conversion and run writing. |
| `config.py`, `runs.py`, `validation.py`, `provenance.py` | Recipes, run manifests, repository checks, and complete source snapshots. |
| `cli.py` | Terminal arguments and messages; no independent physics. |

The small old-name files, such as `map_animations.py`, are compatibility imports.
They contain no separate implementation. They intentionally refer to the same
module object as the grouped implementation, so earlier imports and tests work.
New code should import the grouped path; edit that implementation, not its bridge.

Dependency direction: interfaces → workflows/orchestration → analysis/fitting
and plotting → physics. Physics must never import an interface or workflow.
The core files remain at the authoritative paths named in [AGENTS.md](../../AGENTS.md).

[Detailed module responsibilities](../../docs/CODEBASE_MAP.md) ·
[Source inventory](../../artifacts/source_inventory.csv) ·
[Contribution rules](../../CONTRIBUTING.md)
