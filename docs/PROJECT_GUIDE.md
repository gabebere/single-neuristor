# Project guide for a new collaborator

This is one VO₂ simulation project, not a collection of independent models.
Start at [the project entry point](../START_HERE.html), then read the short
[research handoff](RESEARCH_HANDOFF.md). You do not need previous chat history.

## Four things we built

1. **Model:** Zhang's voltage circuit and inherited hysteretic resistance, plus
   a derived ideal-current formulation. Both use the same resistance memory.
2. **Analysis:** resistance fitting, conditional parameter estimates, measured
   current replay, settled-cycle features and parameter searches.
3. **Exploration:** screened 3D maps, six adjustable 2D slices, synchronized
   input/voltage/R(T) GIFs, and adaptive oscillation-boundary diagnostics.
4. **Reproducibility:** editable recipes, standard run bundles, frozen numerical
   sources, reports, tests, and a searchable artifact catalog.

The software supports these tasks; it has not solved the quantitative TIA fit.

## Where files belong

| Location | What it means / when to open it |
|---|---|
| `src/neuristor/` | Active software. Physics, analysis, fitting, plotting, orchestration and interfaces are separated. |
| `experiments/` | User-editable TOML inputs; start here to change a study, not in Python. |
| `tests/` | Model fidelity, convergence, command, artifact and safety checks. |
| `data/` | Canonical numerical measurements. Originals are not edited. |
| `presets/` | Resistance fits and saved parameters; a fit is not a dynamic validation. |
| `docs/` | Current handoff, methods, reports, presentations and detailed history. |
| `paper/` | Claim/input/recipe/output reproduction map, not another dataset. |
| `public_jobs/` | Complete locally preserved evidence bundles, named by immutable run ID. |
| `runs/` | Exploratory trials, failed attempts and software checks; ignored by Git. |
| `artifacts/` | Rebuildable HTML/CSV catalog pointing to the actual files. No duplicate media. |
| `references/` | Attributed upstream source and bibliography; third-party PDFs remain local only. |
| `archive/` | Historical guides and unique notes; not the active implementation. |
| `private/` | Unapproved/local reference material; never upload this folder. |
| `environments/` | Recorded environment and reproduction guidance. |

Older `jobs/`, `outputs/`, `output/`, `new_samples/`, `Simulations_on_VO2/` and
`legacy_scripts/` locations are documented compatibility/history locations, not
alternative active workflows. Existing paths are intentionally retained where
frozen recipes, LaTeX or immutable bundles depend on them. Do not break working
reproduction merely to reduce the number of directory names.

## Read an unfamiliar run in this order

1. `report.md`: question, method, outcome and caveats.
2. `resolved_config.json`: actual values and units, not today's recipe defaults.
3. `metrics.json` and numerical tables: evidence behind the summary.
4. `run.json`: status, command, code provenance and relative artifact paths.
5. HTML/figures/GIFs: visual interpretation of those numerical results.

Old-style jobs use `job.json`; the library understands both. **Completed** means
file generation completed. **Failed** can still contain useful partial evidence.
**Superseded** means read the replacement, not erase the historical result.

## Start a study

Follow [SAMPLE_STUDY](SAMPLE_STUDY.md): declare the R(T) path, column names/units,
current, ambient temperature and search axes in one recipe. Plan first, run
second. Outputs go in a new run directory. No reference PDF is needed at runtime.

For implementation changes, follow [AGENTS](../AGENTS.md) and run the tests,
repository validator and affected recipe. Do not edit completed evidence bundles.

## Current work versus history

The [short handoff](RESEARCH_HANDOFF.md) states what is true now and what is next.
The [research history](RESEARCH_HISTORY.md) preserves detailed checkpoints.
The [archive chronology](ARCHIVE_INDEX.md) identifies current/superseded evidence.
The [data policy](DATA_POLICY.md) determines what may leave this machine.
