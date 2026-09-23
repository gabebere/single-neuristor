# Run-bundle contract

Run schema version `1` is a portable directory. It can be copied between `runs/` and
`public_jobs/` without rewriting internal paths.

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

- `runs/`: local, ignored, exploratory.
- `public_jobs/`: tracked, reviewed evidence.
- `jobs/`: ignored historical Streamlit jobs.

`neuristor runs publish RUN_ID` copies a local bundle to the public lane and changes
the copied manifest's storage label. It refuses to overwrite an existing public ID.

Completed public bundles are immutable. Corrections are new runs with new IDs and a
report explaining what changed.
