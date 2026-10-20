# Library reorganization — 7 October 2026

## Scope

Make the supported software, publication evidence, and historical material easy
to distinguish. No new physical calibration or parameter-search result was
introduced. Existing dirty-worktree edits were preserved rather than reset.

## What moved

Twenty-three implementation modules now live in purpose-specific subpackages:
`analysis/` (8), `fitting/` (5), `plotting/` (3), `orchestration/` (3),
`interfaces/` (2), and `legacy/` (2). Old module names are small compatibility
bridges to the same module object, preserving private imports and monkeypatches.
The Streamlit replay implementation now has an explicit `main()` so its widgets
render on every rerun, rather than only on a cached import.

`model.py`, `current_drive_sim.py`, the public `workflows.py` API, and CLI command
names remain at their authoritative paths. No existing measured file, completed
public bundle, manuscript source, historical script, or Desktop archive was moved
or deleted. Old document/output locations remain to protect provenance and links.

## Navigation and release foundations

- `START_HERE.html` / `.md`: non-coder navigation.
- Folder READMEs and `docs/CODEBASE_MAP.md`: exact responsibilities.
- `artifacts/source_inventory.csv`: owned Python files, role, size, first-line purpose;
  generated with the existing `neuristor runs index` command, not a new database.
- `paper/reproduction.toml`: five supporting evidence IDs, inputs, commands,
  output contracts and references. Final paper figure assignment remains open.
- `CITATION.cff`, upstream MIT notice, contribution and release guidance.
- A recorded local environment and configured GitHub checks. These are not a
  complete dependency lock or an already executed remote CI result.

Dependencies now have one authoritative declaration in `pyproject.toml`;
`requirements.txt` delegates to it. The wheel carries folder guides and upstream
license notice. Its version remains 0.2.0 until a reviewed release.

## Provenance corrections

Source snapshots now include nested package implementations recursively. New
simulation/fit bundles also preserve package hashes; resistance fits copy their
original input and hash. Numerical reuse preserves the original generating
source separately from the current rendering source, avoiding mixed versions.
Historical frozen snapshots remain unchanged.

## Verification

- Full suite: **134 passed**, with four pre-existing `arctanh` warnings.
- Targeted structure/UI/study verification passed after correcting a cached-import
  Streamlit rerun regression and updating the new catalog-output contract.
- Repository validation checks experiment recipes, manifests, artifact paths,
  and the five paper evidence entries.
- Wheel built and installed into a separate local target; isolated-path imports
  confirmed grouped implementations and compatibility aliases. Dependencies came
  from the existing interpreter: this is not a fresh dependency-isolated install.
- Actual current and voltage recipes both completed:
  `20261007_104254_current-step-with-a-nonzero-metallic-voltage-val_3def6d` and
  `20261007_104306_yuanhang-voltage-driven-relaxation-oscillator_f2a33c`.
- End-to-end sample study completed:
  `20261007_103609_sample-r-t-to-screened-map-and-embedded-gifs_896056`.
  Its summary metrics, grid.csv, verification.csv, animation_checks.csv,
  gif_catalog.json, original GIF bytes, and standalone HTML are exactly equal to
  the preceding reviewed `ec2d1e` demonstration. The example still uses 27
  combinations and retains one oscillator at 23.2153 MHz; it is not a new fit.
- Eighteen moved analysis/fitting/plotting/legacy modules have identical parsed
  source after normalizing import locations and the two relocated root-path
  constants, compared with the pre-reorganization frozen `ec2d1e` source.
- Authoritative physics checksums remain unchanged from the start of this work:
  `model.py`: `1662a1720d694cc20caabf19ef5eb6ced582872ee45d05c26f4868cc4b8369df`;
  `current_drive_sim.py`: `27d640b988c2d4dca11fab70a90a8bd5412766afdda56b9e09e2a5ecee27d2d3`.

## Still to do before a paper release

Resolve scientific claims/channel interpretation, select final figures, review
data redistribution and authorship, freeze a complete release environment, and
perform a fresh-machine reproduction. Review/commit/push changes before claiming
a GitHub backup; choose an external archive/DOI separately with the team.
No Git commit, push, DOI deposit, bulk ZIP, or historical deletion occurred here.
