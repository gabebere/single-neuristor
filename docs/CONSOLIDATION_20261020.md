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
All 18,349 original content files were checked against the snapshot by SHA-256:
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
At the starting checkpoint, `main` was `715032f` (10 August), and the research
branch `e2dd84e` (23 September), with newer local work uncommitted. PR #3's old
scientific description needs correction, not an unreviewed merge.
