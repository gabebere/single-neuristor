# Paper/software release checklist

The current organization supports review. It is not yet an externally archived
paper release, and the quantitative sample/TIA match remains unresolved.

- [ ] Choose the final scientific claims and map each paper figure/table to a
  data source, recipe, numerical table, immutable bundle, and verification result.
- [ ] Confirm the measured channel is the modelled device voltage or document
  the measurement/circuit transfer relation.
- [ ] Clearly separate assumed parameters, measured values, fitted estimates,
  exploratory candidates, and rejected diagnostics.
- [x] Record Gabriel's 20 October approval for experimental numerical data and
  our generated results in DATA_POLICY.md.
- [ ] Check new material against that scope. Never redistribute third-party
  reference PDFs/manuscripts; review authorship/citation metadata separately.
- [ ] Keep upstream copyright/license notices and cite the primary papers.
- [ ] Freeze a complete resolved environment for the chosen release platform;
  perform fresh-machine reproduction. A local package-install smoke test is not
  the same as a clean dependency resolution.
- [ ] Pass `pytest -q`, `neuristor validate`, exact paper recipes, and numerical
  convergence checks. Inspect scientific outputs, not only process exits.
- [ ] Review the Git diff and included/untracked files. Publish only selected
  evidence; exclude caches, scratch trials, local environments, and unnecessary
  bulk duplicated media from the software release.
- [ ] After staging, run `neuristor validate --publication`; this checks staged
  bytes, not different working copies. It does not erase historical exposure.
- [ ] Run `neuristor validate --distribution FILE` on each release asset. This
  streams archive members, rejects private/link paths and unapproved PDF bytes.
  Still review opaque/embedded content manually; this is not legal certification.
- [ ] Make a reviewed versioned commit/tag, push it, and verify remote retrieval.
- [ ] Separately archive any large data/results deposit with checksums and links;
  choose an external repository/DOI with the team. No placeholder DOI is used.
- [ ] Update `CITATION.cff`, the reproduction index, and archive chronology after
  the release identity is known.

None of these external steps are implied by `neuristor runs publish`. It only
copies a completed local bundle to `runs/published/`. This refactor does not push
GitHub changes or alter completed historical bundles.
