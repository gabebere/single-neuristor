# Canonical repository consolidation — 20 October 2026

## Scope

One canonical repository: `https://github.com/gabebere/single-neuristor`.
No new scientific fit or parameter estimate is introduced. Existing model paths,
CLI workflows, completed run IDs and evidence contents are preserved.

Gabriel approved public experimental numerical data and our generated results,
but prohibited third-party paper/manuscript PDFs. See [DATA_POLICY](DATA_POLICY.md).

## Recovery checkpoint

A full local tar snapshot was created **before cleanup** in the sibling folder
`../single-neuristor-backups/20261020_pre_consolidation/project.tar`.
Its SHA-256 is
`a94723a4717d63ad43d67163a8aac897588bab2ca57bb3a53a558eab6688c33f`.
The snapshot includes Git history and tracked, untracked and ignored research
evidence. Rebuildable `.venv`, build/cache directories and bytecode were excluded.
The Git daemon's live socket was not archived; it is not project data.
macOS AppleDouble resource metadata in the tar is not an extra research file.
At the initial cleanup checkpoint, all 18,349 original content files were checked against the snapshot by SHA-256:
none were missing, and only the 17 intentional documentation/CLI/catalog edits
made by this cleanup differed. Canonical data, completed evidence and the two
authoritative physics files matched exactly. New files are separately versioned.
Backup permissions are restricted to this account. **Do not upload this snapshot:**
it contains the reference PDFs and historical Git objects.

## Changes

- Short current `RESEARCH_HANDOFF.md`, with the previous 1,033-line record
  preserved as `RESEARCH_HISTORY.md` at the same directory depth so links work.
- Plain-language `PROJECT_GUIDE.md`, current navigation, disclosure policy and
  exact authored-PDF hash allowlist.
- Third-party reference PDF bytes moved to ignored `private/references/papers/`.
  Local shortcuts retain the old paths; PDF links and copies are removed from
  the Git index. Bibliographic links replace distributable copies.
- The old screenshots ZIP is also local-only; no permission is inferred for it.
- An exact staged-blob publication check and a streaming release-file check are
  available through `neuristor validate`. These never upload or purge history.
- Completed bundles remain unchanged. The large 1,258-GIF map is kept locally
  and designated for a same-repository Release, not ordinary Git.
- Rebuildable local catalogs are ignored rather than committed as another
  database or a misleading complete list of files available in a fresh clone.

## Deliberately not done

No force-push, history rewrite, repository visibility change, deletion of another
repository, or claim of a satisfactory dynamic sample calibration. Root paths
used by frozen recipes and LaTeX remain supported; cosmetic renaming does not
justify breaking reproduction. A public commit cannot back up ignored local
trials; the full local snapshot serves that separate purpose.

## Verification and GitHub state

Local verification: **169 tests passed**, with four existing `arctanh` warnings.
`neuristor validate --publication` passed after staging. Both actual control
recipes completed (`660a0c` current, `f2ed4f` voltage), and the full 27-point
sample-study/GIF recipe completed as `19f494`. The 589,752,448-byte historical
map ZIP passed the streaming distribution inspection; it contains no unapproved
PDF. Three fixed-scale display revisions were copied into the reviewed lane;
their original numerical/source history remains intact.

The first fresh-checkout run exposed one hidden dependency: four fitting recipes
pointed to an ignored local run's `parameters.csv`. The exact table is now
versioned under `presets/search_seeds/` with its source ID/hash, and those recipes
refer to it. No seed values changed. Repository validation now also catches
missing declared relative recipe inputs; a regression test covers this case.
Fresh-checkout and remote verification are being repeated after this correction.

The corrected source passed **170 tests in a fresh checkout**, repository/
publication validation and the actual study-plan command. This reused the local
interpreter dependencies; it was not a new machine/environment installation.
Independent Linux CI then found a pandas 3 Copy-on-Write plotting issue. The
display matrix now explicitly owns a writable copy; a regression test exercises
read-only shared-array behavior without changing physics or numerical tables.
See [pandas' explanation](https://pandas.pydata.org/docs/user_guide/copy_on_write.html#read-only-numpy-arrays).

The publication audit also found 26 PNG index paths whose `figures/` spelling
disagreed with the original manifests' `Figures/`. Only the Git packaging path
spellings were corrected to the already-declared names. Original local evidence
bytes and manifest contents were not edited. A new gate checks this independently
of whether the local filesystem is case-sensitive. These compatibility fixes
are verified separately below; the prior failed CI runs remain inspectable.

One immutable parent/child pair originally recorded conflicting case spellings
for the same PNG. Rather than rewriting either manifest or weakening numerical
checks, runtime archive discovery now permits only an unambiguous case fallback;
the publication gate records that exact historical exception. New bundles no
longer precreate an empty lowercase media directory, preventing the macOS origin
of this mismatch. Truly missing/ambiguous files still fail validation.

The large-map download has two complementary assets: the easy-to-share HTML/GIF
ZIP (589,752,448 bytes), and complete source/numerical/PNG provenance without GIF
duplication (108,319,199 bytes). Their hashes and scopes are in
`results/releases.toml`. All 1,263 ZIP members were checked against the complete
original bundle and were identical. The provenance export only normalizes
physical path case to the original declared spelling; original bundles remain
unchanged. Both assets passed the disclosure inspection.
At the starting checkpoint, `main` was `715032f` (10 August), and the research
branch `e2dd84e` (23 September), with newer local work uncommitted. PR #3's old
scientific description needs correction, not an unreviewed merge.
