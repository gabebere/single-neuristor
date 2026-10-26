# Layout and recovery

The [root README](../README.md) is the collaborator entry point.
[CODEBASE_MAP.md](CODEBASE_MAP.md) describes the maintained implementation.

## Working tree

| Folder | Purpose |
|---|---|
| `src/neuristor/` | Physics, analysis, fitting, plotting, orchestration and interfaces |
| `tests/` | Scientific and software regression checks |
| `experiments/` | Editable recipes |
| `data/` | Original numerical measurements and saved parameter sets |
| `runs/` | Local trials, published evidence, scratch output and the generated catalog |
| `docs/` | Maintained instructions, scientific records, methodology and current slides |

`private/` is local and ignored. Third-party PDFs are never redistributed.

All active imports use their canonical implementation group, such as
`neuristor.analysis.extrema`. The old import wrappers have been removed.
Physics, public workflow and scientific CLI entrypoints retain their established
paths. The Streamlit dashboard/playground and their launch commands are retired;
use `neuristor runs manage` for browsing and TOML recipes for simulation/replay.

## What was removed on 26 October 2026

- Pre-CLI one-off scripts and the unused vendored upstream source tree.
- The old voltage plotting frontend and dormant current-domain search backend.
- The old sample interface's unused storage and create/delete/rename helpers.
  Its numerical fit replay survives in `fitting/saved_fit_replay.py` because a
  scientific regression test still verifies the six recorded measured fits.
- Twenty-two old-name import wrappers, with all active callers updated.
- The Streamlit dashboard/playground, root app launcher, final dashboard wrapper,
  UI-only tests, launch commands and direct package dependency.
- Old report/supplement assemblies, unused backup slides and images, and guides
  superseded by the README, codebase map and current CLI.

Original measurements, parameters, scientific notes, supported commands and
published run bundles remain. Frozen source snapshots in result bundles are
part of the evidence and remain unchanged. Unique ignored local research is
retained because GitHub does not back it up.

## Recovery points

| Checkpoint | Contents | Backup status |
|---|---|---|
| `12be9a9a729e03b164464fad0daab7c1b9d4f4c4` | Original scripts, upstream source and historical report assemblies before the folder reorganization | Verified on GitHub `main` on 26 October |
| `b2ee2ff` | Full versioned state before reorganizing folders | Local commit |
| `080ce19` / `cleanup-before-pruning-20261026` | Exact versioned state immediately before removing unused files | Local commit and annotated tag |
| `d5df40b` | Exact versioned state before removing Streamlit | Local commit |
| `v0.1.0-working-baseline` | Pre-refactor implementation | Tag verified on GitHub |

[Browse the verified GitHub snapshot](https://github.com/gabebere/single-neuristor/tree/12be9a9a729e03b164464fad0daab7c1b9d4f4c4).
The newer local checkpoints have not been pushed. No Git history was rewritten.

To inspect a removed file without restoring obsolete code into the working tree:

```bash
git show cleanup-before-pruning-20261026:archive/scripts/manual.py
git show cleanup-before-pruning-20261026:src/neuristor/legacy/plots.py
```

Before the first reorganization, `public_jobs/` was the published lane, flat
`runs/` held local trials, and `jobs/` held legacy jobs. Those lanes now live
inside `runs/` as `published/`, `local/`, and `legacy/`. `outputs/`, `output/`,
`tmp/`, caches and temporary packages moved into ignored `runs/scratch/`.
The old `results/` front page is `runs/index.html`; `artifacts/` is `runs/catalog/`.
`presets/` became `data/parameters/`; `new_samples/` became
`data/experimental/additional_samples/`. The current slide source moved from
`Simulations_on_VO2/` to `docs/presentation/`; `paper/` became `docs/reproduction/`.

Historical path strings in dated notes describe those original trees. Maintained
links lead to current files or verified GitHub history. Read-only path resolution
in `layout.py` supports preserved run manifests without rewriting them. The
`runs/public_jobs` symlink preserves relative GIF links in one frozen viewer;
it is a navigation alias, not another collection of results.

## Cleanup verification

The pre-Streamlit-removal suite passed 184 tests. Removed alias-only tests were replaced by a
canonical-package import check; numerical fidelity and convergence checks remain.
Repository validation checks all 47 recipes and five reproduction records.

Exact current-valley, voltage-control and 27-point sample-study recipes complete.
Their numerical traces, static-fit summary, grid, smaller-step verification and
animation checks match the pre-pruning runs byte for byte. Both authoritative
physics files are unchanged, as are the calculation bodies retained for
saved-fit replay. All four real presentation routes and the original GIF
dependency load through the restricted local server.

Measured data, saved parameters and complete published bundles are checked
against the pre-pruning Git trees. Build checks use clean sources, so deleted
wrappers cannot survive accidentally in a cached package.

Temporary verification bundles and build workspaces created by this chat were
removed after checking their outputs. The original 162 distinct research records
remain discoverable; unique local trials, failed runs and reviewed evidence
were not pruned. Pruning reduced the maintained package from 83 to 58 Python modules; removing
Streamlit reduces it further to 55. Scientific calculations have one authoritative
implementation path. Frozen historical run sources retain their original contents.

Streamlit removal verification: all 183 remaining tests pass, including isolated
imports of every package module with Streamlit blocked. The clean wheel contains
neither the removed interfaces nor a Streamlit dependency; sample planning works
from that wheel. Validation still finds all 47 recipes, 162 research records and
five reproduction entries. Physics, measured data and published bundles remain
unchanged. The removed UI test was specific to the retired application.
