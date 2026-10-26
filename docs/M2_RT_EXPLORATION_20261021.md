# M2 1600 nm R–T exploration — 21 October 2026

[Open the interactive review](../runs/published/20261021_062155_m2-1600nm-budgeted-r-t-exploration-review_c7387b/parameter_maps.html)
or [read the full report](../runs/published/20261021_062155_m2-1600nm-budgeted-r-t-exploration-review_c7387b/report.md).
This is a conditional diagnostic with **zero retained points**, not a calibration.

## Expanded stage: 1,843 additional grid entries

[Open the larger interactive maps](../runs/published/20261021_064103_m2-1600nm-expanded-r-t-exploration-review_226099/parameter_maps.html)
or [read the expanded report](../runs/published/20261021_064103_m2-1600nm-expanded-r-t-exploration-review_226099/report.md).
The original review is unchanged. The expansion adds a wider 8³ grid and an
11³ grid around onset: **2,616 sampled grid entries across both stages**, including
overlapping settings. It reuses the exact prior approximate R–T fit, 500 µA,
298 K and gamma; it does not relax the measured temperature domain.

| New grid | Entries | Cth (pJ/K) | Se (mW/K) | C (pF) | Sustained detections | Retained |
|---|---:|---|---|---|---:|---:|
| Expanded | 512 | 0.001–50 | 0.0001–3 | 0.0001–500 | 35 | 0 |
| Dense onset | 1,331 | 0.5–4 | 0.10–0.22 | 0.1–1.2 | 373 | 0 |

The broad grid uses dt=0.0125 ns over 2.4 µs; the dense grid uses 0.00625 ns
over 1.2 µs. Their two finer verification steps are preserved in the recipes.
All 408 sustained detections fail the full-trajectory measured-temperature
screen. Only coarse-eligible points receive full two-step verification; no
rejected grid cell is promoted to a verified oscillator.

An independent startup check selects 12 low-Tmax sustained settings from the
previous onset grid and an interim new-grid checkpoint. At 0.00078125 and
0.000390625 ns, all 24 integrations already exceed 370.1795 K within 150 ns;
the lowest observed maximum is **372.67096 K**. This checks early temperature
exclusions, not steady-state feature convergence or the entire expanded grid.
Exact selection, configuration, source and numerical checks are archived.

Reproduce with `neuristor analyze physical-maps --config` and the two
`experiments/current/m2_1600nm_{expanded,dense_onset}_20261021.toml` recipes.
The archive contains portable copies with a local static preset. Animations
remain a separate optional stage. **196 tests passed**, repository validation
passed, portable preset/domain checks passed, links and artifacts resolve, and
the new diagnostic PNG was visually checked. Browser interaction remains untested.
No new specimen calibration is adopted.

## Original budgeted stage

The user supplied `m2_8.9_1stL_1600nm_A_RT`. Its unmodified numerical contents
are preserved as `data/experimental/m2_8.9_1stL_1600nm_A_RT.tsv` (374 rows;
temperature K, resistance ohm, acquisition time seconds). The file covers
289.98–370.1795 K and one cooling/heating reversal. SHA-256:
`fdda8d2293404566fb16154223da131b1a44f15975469f1cdfd8622948112e03`.

The existing major-loop fit gives log10 RMSE **0.08065244**, about a 1.204
typical multiplicative error, with structured transition residuals. The initial
strict study stopped at the normal 0.05 acceptance gate. Subsequent recipes
explicitly permit 0.10 **only for exploratory plots**; they do not improve or
validate the static law. The failed strict run is preserved in the review.

All grids retain the previous map's 500 µA ideal imposed current, 298 K ambient
and initial temperature, zero initial voltage, insulating initial branch and
assumed gamma=0.956269682. Cth, Se and C remain search coordinates. The full
trajectory must stay in the measured temperature domain; the 110 MHz operating
screen, persistence detector and 5% two-step feature criterion remain unchanged.

| Grid | Entries | Base dt (ns) | Duration (µs) | Sustained detections | Retained |
|---|---:|---:|---:|---:|---:|
| Broad | 216 | 0.05 | 0.6 | 54 | 0 |
| Lower heating | 216 | 0.05 | 2.0 | 10 | 0 |
| Transition | 216 | 0.00625 | 1.2 | 68 | 0 |
| Onset | 125 | 0.003125 | 0.6 | 50 | 0 |

Every sustained detection exceeded the measured temperature domain. The closest
onset candidate reached 373.059 K. A separate 36-point pilot at 0.0015625 ns
also found no eligible oscillator; that does not finely verify every excluded
cell. The 773 entries include overlapping settings across grids. The original
screened maps are empty. A diagnostic companion displays computed f/Vpp with
exclusions explicit, alongside six slider-controlled slices per grid. Those
values are **not validated device predictions**. Blank cells and search edges
do not prove physical absence of oscillations. No GIFs are generated when no
point is retained.

Reproduce with the four `experiments/studies/m2_1600nm_*_20261021.toml`
recipes through `neuristor study run --config FILE`. Portable copies in the
review resolve to its preserved raw input. The review contains all four
completed studies, their frozen sources, a strict-failure record, the finer
pilot configuration and tables, and the exact saved-data assembly script.
Archival publication is local copying, not a GitHub upload.

Checks: **194 tests passed**, repository validation passed, all four portable
recipe plans passed, all registered artifacts and HTML links resolve, input
bytes match the attachment, and the diagnostic PNG was visually inspected.
The 5.51 MB HTML embeds Plotly and the R–T fit image. Browser interaction was
not verified because the available browser blocks local file URLs.

No solver or hysteresis implementation was changed. The next useful question
is whether the inherited static law adequately represents this sample before
interpreting a denser dynamic parameter search.
