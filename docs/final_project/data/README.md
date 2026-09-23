# Data and reviewed results

This directory is an organized view of the project's canonical evidence. Its entries
are relative links rather than copied files, so there is only one authoritative copy of
each raw measurement and immutable run bundle.

- `resistance/` contains the raw R(T) measurement and its reviewed fit.
- `experiment_runs/` contains the professor-supplied oscilloscope exports and their
  reviewed normalized sweep.
- `parameter_estimates/` contains the current environmental-conductance and thermal-
  capacitance analyses.
- `reviewed_runs/` exposes the complete tracked `public_jobs/` archive, including
  superseded and historical runs. Use `docs/ARCHIVE_INDEX.md` to distinguish them.

Do not edit files through a reviewed-run link. Re-run and publish a new bundle instead.
