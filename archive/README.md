# History: preserve evidence without making duplicate archives

This folder is the guide to historical material, not another copy of gigabytes
of GIFs. The complete searchable record is the [artifact library](../runs/catalog/index.html).

The [consolidated Desktop guide](desktop_consolidation_20261016/index.html) links
the former `VO2 Research Archive` folders to their canonical project results.
Only 12 unique historical notes/index/recipe files were imported; 914 duplicate
files were verified against existing project files instead of copied again.
The old Desktop archive was moved intact to recoverable macOS Trash.
See [the filing audit and recovery record](desktop_consolidation_20261016/README.md).

| Historical material | Canonical location |
|---|---|
| Reviewed scientific chronology | [docs/ARCHIVE_INDEX.md](../docs/ARCHIVE_INDEX.md) |
| Local trials, including failed exporters | `runs/local/` and the library's status column |
| Reviewed/preserved bundles | [runs/published/](../runs/README.md) |
| Pre-refactor scripts and their replacements | [archive/scripts/README.md](scripts/README.md) |
| Earlier package plotting/sample helpers | [src/neuristor/legacy/](../docs/CODEBASE_MAP.md#historical-compatibility-helpers) |
| Earlier reports, GIFs, and presentation output | `docs/`, `runs/scratch/documents/`, `runs/scratch/`, `docs/presentation/` |
| Original model and source papers | [archive/references/README.md](references/README.md) |
| File-by-file source inventory | [runs/catalog/source_inventory.csv](../runs/catalog/source_inventory.csv) |
| All produced files and sizes | [runs/catalog/files.csv](../runs/catalog/files.csv) |

## Interpret history correctly

- **Completed**: outputs were generated. This says nothing about fit quality.
- **Failed**: preserve the failure report and any partial numerical evidence.
- **Superseded**: use the newer viewer/method named in the chronology; do not erase its predecessor.
- **Reviewed evidence**: read its assumptions and limitations before adopting it.

Storage lanes and generation status are recorded facts. Scientific acceptance
must come from the report/handoff, not a filename or an automatically assigned
label. The first standalone-export failure remains discoverable for provenance.

Historical bundle bytes remain intact after relocation; active links use the
[consolidated layout](../docs/REPOSITORY_LAYOUT.md). Empty caches and ignored
scratch folders are not software deliverables.
No new ZIPs or second metadata database are needed.

Rebuild navigation with `neuristor runs index`. Recover the original working
baseline from the local Git tag `v0.1.0-working-baseline`; verify its remote
availability before describing it as an externally backed-up release.
