# Repository layout

Reorganized on 26 October 2026. The [root README](../README.md) is the single
collaborator entry point. Scientific conclusions remain in
[RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md); implementation responsibilities are
in [CODEBASE_MAP.md](CODEBASE_MAP.md).

## Former locations

| Before | Now |
|---|---|
| `public_jobs/` | `runs/published/` |
| Flat `runs/RUN_ID/` | `runs/local/RUN_ID/` |
| `jobs/` | `runs/legacy/` |
| `outputs/` | `runs/scratch/` |
| `output/` | `runs/scratch/documents/` |
| `tmp/` | `runs/scratch/tmp/` |
| Root `build/`, `__pycache__/` | `runs/scratch/build/`, `runs/scratch/python_cache/` |
| `presentation_package_*` | `runs/scratch/packages/` |
| `artifacts/` | `runs/catalog/` |
| `results/index.html`, `results/releases.toml` | `runs/index.html`, `runs/releases.toml` |
| `Simulations_on_VO2/` | `docs/presentation/` |
| `paper/` | `docs/reproduction/` |
| `environments/` | `docs/environment/` |
| `presets/` | `data/parameters/` |
| `new_samples/` | `data/experimental/additional_samples/` |
| `references/` | `archive/references/` |
| `legacy_scripts/` | `archive/scripts/` |
| `START_HERE.md`, `START_HERE.html`, `Amir_readme.md`, `docs/PROJECT_GUIDE.md` | Root `README.md` |
| `src/README.md`, source-folder READMEs | `docs/CODEBASE_MAP.md` |
| `tests/README.md` | `CONTRIBUTING.md` |
| Separate jobs/catalog/results READMEs | `runs/README.md` |

Folder-specific documents describing unique data formats, scientific methods,
upstream licensing or presentation sources remain beside those files. They are
linked from the main guide instead of repeating the onboarding instructions.
The active Python package keeps its established responsibility groups, public
imports and authoritative physics paths.

## Evidence preservation

Checkpoint `b2ee2ff` captures the complete versioned state before any moves.
Ignored local trials, old jobs and scratch outputs were moved intact, not added
to Git. No history was rewritten and no remote upload was performed.

Published bundle IDs and file contents are unchanged, including numerical data,
reports, manifests, resolved recipes, source snapshots and source hashes. Their
parent directory changed. Historical absolute paths are interpreted at read time
by `neuristor.layout.resolve_recorded_path`; they are not rewritten in evidence.
Existing trash sidecars with former `runs/ID` locations remain restorable into
`runs/local/`. Active recipes, interfaces, indexes and document links use the new
locations. Saved parameter values and measured numerical data are preserved.

`runs/public_jobs` is a relative compatibility symlink to `published`. A frozen
historical viewer refers to `../../public_jobs/...`; this one alias keeps its
file-based GIF links working without editing the viewer. It is not another storage
lane. The presentation server also resolves the former `/public_jobs/` URL prefix
and applies its existing resolved-path allowlist to every request.

Local trials and published results remain distinct inside one `runs/` folder:
merging them into a single undifferentiated lane would lose publication and
recoverable-delete protections. Catalog generation reads these lanes directly,
excludes trash and frozen sources from its file inventory, and does not follow
the compatibility link as another copy of the evidence.

To orient a collaborator, start at the root README. To reconstruct a historical
path, use the table above. To continue a study, use the current versioned recipe
and generate a new bundle rather than editing a completed one.

## Verification of this migration

- Original checkpoint: 201 passing tests and repository validation.
- Reorganized checkout: 207 passing tests, including six relocation regressions.
- Repository and staged publication checks pass: 47 recipes, five reproduction
  entries, and the preserved archive plus three new local verification runs.
- All 4,816 versioned published-bundle paths and Git blob identities match the
  checkpoint after changing only their root prefix. Original Git letter case is
  retained, including three historical `Figures/` folders.
- Exact current-valley, original-voltage and 27-point sample-study recipes
  complete under `runs/local/`; the sample exports its embedded-GIF HTML.
- All four real presentation routes and the historical map's GIF dependency
  load through the restricted loopback server. Guide links and document
  symlink targets resolve locally.
- A wheel built from clean staged sources imports successfully and contains
  the new layout module without the removed source-folder guides.

No numerical fitting or simulation equations changed. Verification runs remain
local and ignored; they are software checks rather than new reviewed evidence.
