# Working safely in this repository

This file is the short operational map for humans and AI agents. Read the root
`README.md`, this file, and the relevant experiment recipe before changing code.

## Source-of-truth hierarchy

1. `src/neuristor/model.py` is authoritative for the Yuanhang voltage circuit and
   hysteresis implementation.
2. `src/neuristor/current_drive_sim.py` is authoritative for the supported ideal
   current-source extension.
3. `src/neuristor/workflows.py` translates documented units and writes artifacts. It
   must call the physics modules rather than duplicate their equations.
4. `src/neuristor/cli.py` and `src/neuristor/dashboard.py` are presentation layers.
   They must not contain independent scientific logic.
5. `experiments/*.toml` are the versioned user-facing inputs.

Dependencies should point downward through that list, never from physics into CLI or
Streamlit code.

## Non-negotiable physical invariants

- The current-source model is ideal: `C dV/dt = I - V/R(T)`.
- A series resistor does not change an imposed ideal current and is not part of that
  reduced model.
- `C_F = 0` is the exact algebraic limit `V = I R(T)`.
- The steady metallic voltage floor is approximately `I Rm`; capacitance changes the
  approach time, not that floor.
- Preserve the float32-faithful hysteresis behavior unless a dedicated upstream
  fidelity audit proves a change is required.
- Keep units explicit at file and CLI boundaries. The physics modules use SI unless
  their dataclass field name states otherwise.

## How to add a capability

1. Add or extend a pure function in the relevant physics or analysis module.
2. Add orchestration in `workflows.py` that returns a standard `RunBundle`.
3. Expose it through one `neuristor` subcommand.
4. Add a reusable TOML recipe when the workflow is configuration-driven.
5. Add tests at the lowest useful layer and one command/bundle smoke test.
6. Document the command in the README or linked format specification.

Do not add a new top-level one-off Python script. Pre-refactor scripts live in
`legacy_scripts/` only as historical source and can be recovered exactly from the
`v0.1.0-working-baseline` tag.

## Run-bundle contract

New workflows write a directory containing `run.json`, `resolved_config.json`,
`metrics.json`, a human-readable `report.md`, numerical tables, and figures when
appropriate. All manifest artifact paths are relative to the bundle root. Exploratory
runs belong in ignored `runs/`; reviewed evidence is copied with
`neuristor runs publish RUN_ID` into tracked `public_jobs/`.

Never edit a completed public bundle in place. Re-run the experiment and publish a new
bundle so provenance remains inspectable.

## Required checks

```bash
pytest -q
neuristor validate
```

Also run the exact recipe affected by a workflow change. For physics changes, compare
at least two smaller timesteps and add or update a convergence assertion. Do not
weaken tolerances merely to make a changed trajectory pass.

## Repository hygiene

- Keep the root small; implementation belongs in `src/neuristor/`.
- Preserve unrelated user changes in a dirty worktree.
- Prefer descriptive names with units, such as `C_th_pJ_per_K` and `duration_us`.
- Public functions and non-obvious numerical choices need docstrings or comments that
  explain intent and physical meaning, not line-by-line syntax.
- Generated scratch output stays in `runs/` or `outputs/`, never beside source files.
- Update `docs/ARCHIVE_INDEX.md` when adding reviewed scientific evidence.
