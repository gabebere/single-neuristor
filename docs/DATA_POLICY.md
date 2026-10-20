# Data, reference papers, and publication

## Decision recorded on 20 October 2026

Gabriel approved making this project's experimental numerical data and our own
generated results public in the definitive `gabebere/single-neuristor` repository.
This is publication permission, not a new blanket license for third-party work.

**Do not redistribute third-party paper/manuscript PDFs**, including the
collaborators' manuscript PDF, even where a paper is publicly downloadable.
Cite its DOI/arXiv/publisher page instead. The prohibition applies to Git,
GitHub release assets, ZIPs/tars, and symlinks.

| Material | Treatment |
|---|---|
| Our source, tests, recipes and documentation | Versioned in the canonical repository; retain attribution. |
| Experimental numerical data and our derived results | Approved for public inclusion; keep units, hashes and limitations. |
| Our authored reports/presentations and generated figures | May be included; do not bundle copies of reference PDFs. |
| Third-party paper/manuscript PDFs | Local `private/` storage only; use bibliographic links publicly. |
| Upstream implementation | Keep its original license and source attribution. |
| New collaborator material with unclear permission | Private by default until explicitly approved. |
| Environments, caches and scratch output | Local, not source deliverables. |

MIT covers this project's software, not automatically measurements, publications,
or publisher material. Existing reference licensing does not override Gabriel's
instruction not to distribute the PDFs.

## Storage is not approval

`public_jobs/` is a historical name for locally preserved evidence. Its name and
`neuristor runs publish` do not authorize GitHub publication. Similarly, a
standalone HTML is technically portable but its complete payload still needs
review before sharing. Do not embed private reference documents.

Large approved result packages belong in versioned Releases of the **same**
repository, linked from the project index. Do not place unapproved material in
a public release: release assets inherit repository visibility.

## Protect local references

`private/` is Git-ignored. Old reference paths may remain local compatibility
symlinks, also excluded from Git. `.gitignore` does not remove an already-tracked
file: verify the actual staged/committed tree, not just ignore patterns.

Private here means **not intentionally distributed by this project**; it is not
encryption and does not prevent an unrelated computer-backup/sync service from
copying local files. Keep private backups under restricted local permissions.

## Historical exposure

Before this policy, `main` already contained the Almeida and Zhang paper PDFs;
the research branch also contained the Gildor manuscript PDF and reference links.
The cleanup excludes them from the new current tree. Previous public commits,
PR diffs, forks and caches may retain copies. No deletion or ignore rule can
promise their removal. Rewriting/deleting remote history requires a separate,
explicitly authorized remediation plan; do not force-push during housekeeping.
