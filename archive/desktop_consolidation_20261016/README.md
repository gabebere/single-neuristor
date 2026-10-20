# Desktop archive, consolidated into the project

16 October 2026. Use [the browser guide](index.html) or the
[main project directory](../../START_HERE.html).

The Desktop `VO2 Research Archive` was a second copy of results already in this
repository. Its content is now accounted for in one project directory:

- **914 files** already existed byte-for-byte inside the project. They were not
  copied again. Their originals remain in `public_jobs/`, `runs/`, and existing
  document/data/output folders.
- **12 unique files**, totaling **25,149 bytes**, are retained in `original_notes/`
  with their contents unchanged. These are historical notes, indexes, a launcher
  snapshot, and an old replay recipe—not new measurements or fitted parameters.
- **Six symbolic links** already pointed into the project. Their original targets
  are recorded; their targets were not moved or deleted.
- **19 run-folder references** below lead to existing reports/viewers.

[inventory.json](inventory.json) records every Desktop content path, SHA-256,
size, repository destination and action. Before moving the Desktop folder, each
of the 926 content files was compared in full with its retained repository copy.
Finder `.DS_Store` files are excluded from scientific/content comparison and
remain only with the recoverable original folder.

## Find the results

These are historical trials and diagnostic parameter changes. They do not become
accepted calibrations through filing. Read the [current scientific handoff](../../docs/RESEARCH_HANDOFF.md)
before comparing fits. The three-point map preview is not the complete GIF map;
use the [artifact library](../../artifacts/index.html) for newer maps and viewers.

| Former Desktop grouping | Existing report or viewer |
|---|---|
| 00_Current_trial/20261003_relative_squared_96b944/GIF Results | [4e79ac](../../public_jobs/20261003_183956_simple-loss-diagnostic-1-all-current-gif-replay-_4e79ac/START_HERE.html) |
| 00_Current_trial/20261003_relative_squared_96b944/Search Results | [96b944](../../public_jobs/20261003_183555_simple-relative-squared-extrema-fitting-pilot_96b944/report.md) |
| 01_Fitting_trials/20261002_scaled_log_extrema_32d91d | [32d91d](../../public_jobs/20261002_145243_meeting-extrema-only-physically-screened-fitting_32d91d/report.md) |
| 01_Fitting_trials/VO2 Background Search/Frozen Results | [f426de](../../runs/20260922_175152_background-settled-oscillations-for-original-lab_f426de/report.md) |
| 01_Fitting_trials/VO2 Monotone Proximity Search/Frozen Results | [ffdabf](../../runs/20260922_210712_monotone-proximity-constrained-settled-oscillati_ffdabf/report.md) |
| 01_Fitting_trials/VO2 Results — Expanded Fit — 2026-09-22/Optimization — expanded search | [694804](../../public_jobs/20260921_210646_expanded-persistence-aware-joint-inference_694804/report.md) |
| 01_Fitting_trials/VO2 Results — Expanded Fit — 2026-09-22/Optimization — previous joint search | [6b5d8f](../../public_jobs/20260921_130032_budgeted-joint-resistance-and-waveform-inference_6b5d8f/report.md) |
| 01_Fitting_trials/VO2 Results — Expanded Fit — 2026-09-22/Previous interface replay — 0.025 ns | [eef073](../../runs/20260921_215248_interactive-specimen-laboratory-replay_eef073/comparison.html) |
| 01_Fitting_trials/VO2 Results — Expanded Fit — 2026-09-22/Scope replay | [fe6a6a](../../runs/20260922_032944_expanded-specimen-synchronized-oscilloscope-expo_fe6a6a/scope_viewer.html) |
| 01_Fitting_trials/VO2 Tanh Proximity Pilot/Results | [11a3f7](../../runs/20260923_061836_tanh-proximity-bounded-pilot-labels-300-to-700_11a3f7/report.md) |
| 01_Fitting_trials/VO2 monotone best search_1 — GIFs by current | [7f6312](../../runs/20260923_060307_saved-replay-stacked-current-voltage-hysteresis-_7f6312/START_HERE.html) |
| 01_Fitting_trials/VO2 steady best search_1 — GIFs by current | [bfa3de](../../runs/20260922_204656_saved-replay-stacked-current-voltage-hysteresis-_bfa3de/START_HERE.html) |
| 02_Parameter_diagnostics/VO2 500 uA — Cth 0.072 — T0 298 K/fine_timestep_check | [11b59d](../../runs/20260923_150626_500-ua-with-cth-0-072-pj-per-k-at-298-k_11b59d/report.md) |
| 02_Parameter_diagnostics/VO2 500 uA — Cth 0.072 — T0 298 K/main_run | [f8d474](../../runs/20260923_150423_500-ua-with-cth-0-072-pj-per-k-at-298-k_f8d474/report.md) |
| 02_Parameter_diagnostics/VO2 500 uA — T0 298 K — Cth 1e-4 | [635615](../../runs/20260923_145310_500-ua-with-cth-1e-4-pj-per-k-at-298-k_635615/report.md) |
| 02_Parameter_diagnostics/VO2 eef073 — T0 324.4 K | [9c4563](../../runs/20260923_074403_eef073-replay-with-ambient-t0-324-4-k_9c4563/comparison.html) |
| 02_Parameter_diagnostics/VO2 eef073 — T0 324.4 K — Hysteresis GIFs | [847061](../../runs/20260923_074617_saved-replay-stacked-current-voltage-hysteresis-_847061/START_HERE.html) |
| 03_Historical_references/VO2 eef073 — GIFs by current (A) | [20a254](../../runs/20260922_053840_saved-replay-stacked-current-voltage-hysteresis-_20a254/START_HERE.html) |
| 04_Parameter_maps/20261007_Clickable_map_preview_3_points | [f34881](../../public_jobs/20261007_061058_portable-screened-parameter-map-with-point-gifs-_f34881/parameter_maps.html) |

## Unique historical notes

Original folder names are retained only for provenance. Older claims about jobs
being “running,” the “current trial,” Desktop locations, and portable relative
paths describe the original delivery date, not today's project state.

The old redirect launcher is stored as `START HERE.html.txt` so it can be read
without opening a broken redirect. Its bytes are unchanged. Use the canonical
scope viewer linked above. The saved `reproduce_scope.toml` documents the old
Desktop layout; for the maintained project command use
[experiments/current/specimen_scope_export.toml](../../experiments/current/specimen_scope_export.toml).
Its source paths/export controls differ from the archived delivery recipe.

- [00_Current_trial/20261003_relative_squared_96b944/REVIEW.md](original_notes/00_Current_trial/20261003_relative_squared_96b944/REVIEW.md)
- [01_Fitting_trials/VO2 Background Search/README.txt](original_notes/01_Fitting_trials/VO2%20Background%20Search/README.txt)
- [01_Fitting_trials/VO2 Monotone Proximity Search/README.txt](original_notes/01_Fitting_trials/VO2%20Monotone%20Proximity%20Search/README.txt)
- [01_Fitting_trials/VO2 Results — Expanded Fit — 2026-09-22/README.txt](original_notes/01_Fitting_trials/VO2%20Results%20%E2%80%94%20Expanded%20Fit%20%E2%80%94%202026-09-22/README.txt)
- [01_Fitting_trials/VO2 Results — Expanded Fit — 2026-09-22/START HERE.html](original_notes/01_Fitting_trials/VO2%20Results%20%E2%80%94%20Expanded%20Fit%20%E2%80%94%202026-09-22/START%20HERE.html.txt)
- [01_Fitting_trials/VO2 Results — Expanded Fit — 2026-09-22/reproduce_scope.toml](original_notes/01_Fitting_trials/VO2%20Results%20%E2%80%94%20Expanded%20Fit%20%E2%80%94%202026-09-22/reproduce_scope.toml)
- [01_Fitting_trials/VO2 monotone best search_1 — GIFs by current/SEARCH_SOURCE.txt](original_notes/01_Fitting_trials/VO2%20monotone%20best%20search_1%20%E2%80%94%20GIFs%20by%20current/SEARCH_SOURCE.txt)
- [01_Fitting_trials/VO2 steady best search_1 — GIFs by current/SEARCH_SOURCE.txt](original_notes/01_Fitting_trials/VO2%20steady%20best%20search_1%20%E2%80%94%20GIFs%20by%20current/SEARCH_SOURCE.txt)
- [03_Historical_references/VO2 eef073 — GIFs by current (A)/README.txt](original_notes/03_Historical_references/VO2%20eef073%20%E2%80%94%20GIFs%20by%20current%20(A)/README.txt)
- [04_Parameter_maps/README.md](original_notes/04_Parameter_maps/README.md)
- [ARCHIVE_INVENTORY.json](original_notes/ARCHIVE_INVENTORY.json)
- [START_HERE.md](original_notes/START_HERE.md)

## Recovery and storage

The original Desktop directory is moved intact to recoverable macOS Trash only
after all comparisons succeed. Recovery location:

`/Users/gabrielberezovsky/.Trash/VO2 Research Archive — consolidated 2026-10-16`

To recover the exact Desktop layout, drag that folder from Trash to Desktop and
restore its original name, provided no replacement folder would be overwritten.
Nothing is permanently erased and Trash is not emptied. This does **not** free
disk space until you separately choose to empty Trash.

Canonical results, raw measurements, source, numerical parameters, and completed
public bundles are unchanged. No simulations were run for filing; no Git commit,
push, or external backup is implied. This folder contains notes and a provenance
map, not another copy of the animation archives or a new database.

## Verification

All 926 content paths and six symbolic links were accounted for before cleanup;
all 12 imported files retain their original hashes and contents. All 47 local
links in the new browser guide and main project directory resolve. Repository
validation passed (32 recipes, 132 run records, five paper evidence entries),
and the full suite passed: 163 tests with four existing arctanh warnings.
