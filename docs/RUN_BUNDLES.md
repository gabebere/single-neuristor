# Run-bundle contract

Run schema version `1` is a portable directory. It can be copied between `runs/local/` and
`runs/published/` without rewriting internal paths.

## Required metadata

`run.json` contains:

- immutable run ID and creation timestamp;
- name, model, workflow kind, and completion status;
- exact terminal command;
- Git commit, branch, and dirty-worktree flag;
- relative artifact paths with labels and media types;
- a small scalar summary for archive filtering.

`resolved_config.json` contains all scientific inputs after CLI overrides and omits
private loader metadata.

## Expected evidence

Simulation bundles contain `traces.csv`, `metrics.json`, `report.md`, and an overview
figure. Sweep bundles contain `sweep.csv`, metrics, a report, and an automatic figure
for up to three axes. Fit and lab-analysis bundles use workflow-specific CSV names but
preserve the same manifest, metrics, report, and figure conventions.

Non-finite values are serialized as JSON `null`, not nonstandard `NaN` tokens.

## Storage lanes

- `runs/local/`: local, ignored, exploratory.
- `runs/published/`: tracked, reviewed evidence.
- `runs/legacy/`: ignored historical Streamlit jobs.

`neuristor runs publish RUN_ID` copies a local bundle to the public lane and changes
the copied manifest's storage label. It refuses to overwrite an existing public ID.

Completed public bundles are immutable. Corrections are new runs with new IDs and a
report explaining what changed.

## Recoverable local deletion

`neuristor runs manage` provides confirmed Delete/Restore controls for eligible
local trials. `RunRegistry.trash` moves the intact folder to
`runs/local/.trash/TRASH_ID/bundle`, with an administrative `deletion.json` beside it.
This sidecar is not a new scientific run manifest or another metadata database.
Active discovery and inventory exclude trashed data; restore refuses overwrites.
Published, running, legacy, symlinked and recipe-referenced runs are protected.
See [RUN_MANAGEMENT.md](RUN_MANAGEMENT.md); no permanent erase is implemented.
