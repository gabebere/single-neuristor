# Legacy scripts

These are the pre-refactor one-off CLIs and figure generators. They remain here for
historical provenance and to explain older committed artifacts, but they are not the
active interface and receive no new features.

Use the installable `neuristor` command and versioned recipes under `experiments/` for
new work. The exact pre-refactor repository is preserved by the Git tag
`v0.1.0-working-baseline`.

Migration map:

| Historical script purpose | Current interface |
|---|---|
| `manual.py`, `run_current_drive.py` | `neuristor simulate ...` and experiment TOML |
| `sweep_current_capacitances.py` | `neuristor sweep run --config experiments/sweeps/current_capacitance_map.toml` |
| former screenshot-digitization scripts | removed; use numerical exports with `neuristor analyze lab --data ...` |
| `fit_resistance.py` | `neuristor fit resistance` |
| `search_sample_current_domain.py` | parameter sweep recipes and dashboard filtering |
| fidelity/check scripts | `pytest -q`, with these sources retained for deeper historical audits |

The large presentation- and paper-specific generators describe fixed historical
deliverables already stored under `docs/`, `Simulations_on_VO2/`, and `public_jobs/`.
