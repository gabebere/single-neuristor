# Changelog

## Unreleased — bounded overnight filament profile

- Add `neuristor analyze filament-profile` with a frozen-source, checkpointed
  eight-hour recipe, two thermal-bound families, fixed-f forward/reverse
  continuations, adaptive midpoints and paired mean-aware refinements.
- Preserve optimizer-excluded current checks, all-current physical diagnostics,
  two timestep halvings, long controls and explicit failed acceptance gates.
- Add live progress, named parameter tables and final evidence plots; guard
  resume with source/input/environment hashes, exclusive locking and cached
  evaluation caps. Physics and default fitting objectives remain unchanged.
- Verify 223 tests and all 50 recipes. Overnight scientific outcomes are pending;
  no calibration is adopted.

## Unreleased — experimental filament and cold surroundings

- Add an optional normalized parallel-conductance model with separately
  accounted branch currents and Joule heat; preserve the exact original f=1 limit.
- Add a versioned presentation-data experiment comparing bounded baseline and
  filament fits, conditional fraction scans and three-timestep verification.
- Export original inputs, complete source, numerical checks, native branch
  histories, figures and an offline interactive comparison in a new local bundle.
- Complete 249 evaluations per arm on the presentation inputs. The added-parameter
  search returns f=1 without an all-target fit; explicit fractional cases fail
  temperature/cycle or convergence checks. Preserve reviewed evidence and limits;
  no calibration is adopted.

## Unreleased — optional presentation reproduction

- Add `neuristor reproduce-presentation`: rerun all declared presentation stages
  from measured inputs without reading archived outputs or Release assets.
- Save stage checksums and source/input/environment fingerprints for safe resume.
- Regenerate numerical tables, figures, GIFs, HTML and editable slide result
  tables; connect completed full results to the existing presentation buttons.
- Document the optional setup step and the several-hour compute allowance.

## Unreleased — 26 October repository cleanup

- Consolidate jobs and outputs under `runs/`, with one collaborator README.
- Remove unused archived scripts, plotting/search backends, old report assemblies
  and unused slide assets; keep recovery points in Git history.
- Replace 22 old-name Python import wrappers with canonical grouped imports.
  Python callers should use e.g. `neuristor.analysis.extrema` and
  `neuristor.plotting.map_animations`. Scientific CLI commands,
  `neuristor.workflows` and authoritative physics paths remain available.
- Remove the Streamlit dashboard and playground, their launch commands and
  package dependency. Browse results with `neuristor runs manage`; replay
  measurements with `neuristor analyze replay-lab --config FILE.toml`.
- Retain saved-fit numerical replay and its regression checks while removing
  the unused sample-management interface code. Preserve measurements, parameters
  and completed result bundles.

## 20 October 2026 — canonical research-library checkpoint

- Consolidate the previously local software, recipes and reviewed evidence into
  GitHub `main`; close superseded PR #3 and merge PR #4.
- Add concise current handoff, detailed preserved history, clear project/results
  guides and the public-data/private-reference-PDF policy.
- Publish large-map media and complementary provenance in a dated same-repository
  Release, with verified hashes and successful full-bundle reconstruction.
- Repair a hidden ignored-run seed dependency, pandas 3 plot-only array ownership,
  and portable archive path casing. No authoritative physics or accepted sample
  calibration changes. Missing files and numerical tolerances are not weakened.
- Pass 174 tests locally/fresh checkout and independent Linux Python 3.10/3.13
  verification, normal/publication validation and actual control/study workflows.
- Preserve a full private local recovery snapshot. Do not claim ignored scratch
  history is entirely backed up in public Git or old public PDF copies are purged.

The entries below describe earlier **local-only** checkpoints, not their current
publication state. Software version remains 0.2.0; the dated research tag is
`research-library-2026-10-20`, not a final fitted-model/paper release.

## Unreleased — recoverable run management

- Add loopback artifact-manager Delete/Restore controls and matching terminal commands.
- Move eligible local trials into recoverable project trash; no permanent erase.
- Protect published/running/legacy/symlinked/recipe-referenced runs and reject restore overwrites.
- Require same-origin session-token authorization and confirmation for browser mutations.
- Verify with synthetic temporary bundles only; no real run was deleted. Full suite: 146 tests passed.

## Unreleased — library organization

- Group analysis, fitting, plotting, orchestration, interfaces, and legacy helpers.
- Preserve authoritative model paths and old imports through module aliases.
- Freeze recursively complete package sources in study/map/search bundles.
- Add plain-language folder guides, project navigation, paper-evidence index,
  source inventory, citation metadata, and release preparation guidance.
- Preserve all prior measurements, output locations, failed trials, and public bundles.

No scientific calibration or accepted dynamic fit was claimed by those earlier
local reorganizations. The 20 October entry records the subsequent public
checkpoint; see Git history and the archive chronology for research changes.
