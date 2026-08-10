# Public Jobs

This folder is intentionally not ignored by Git.

Use the Streamlit **Save in public history** toggle when a run should
be shareable with collaborators through GitHub. Ordinary exploratory jobs should
stay in `jobs/`, which is ignored and local-only.

Before committing files here, check that the job is small enough to review and
that its `job.json` clearly describes the sample, source model, parameters, and
outputs.

## Archived runs

- `20260707_145140_paper_frequency_f881d2`: fitted-specimen paper-frequency analog,
  including current sweep, spectra, energy cycle, pump-probe recovery, and animations.
- `20260707_145140_sample_scale_89ffb4`: fitted-specimen sample-scale control and
  operating-window sweep.
- `20260707_154737_390uA_valley_ff1d22`: explicit 390 uA nonzero-valley trial using
  a raised metallic resistance; it is a diagnostic trial, not a fitted solution.
- `20260709_115815_published_table_current_3c2df4`: Yuanhang published-table values
  translated to the ideal-current circuit, including deterministic/noisy controls and
  timestep checks.

These are immutable historical run records from July 2026. They predate the August
2026 endpoint-fidelity and exact-substep corrections, so use them for provenance and
presentation history. New quantitative claims should be regenerated with the current
solver. Public `job.json` paths are repository-relative so the History tab works after
cloning the repository.
