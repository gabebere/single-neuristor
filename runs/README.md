# Jobs and results

This is the single home for generated work. Start with [selected results](index.html),
or run `neuristor runs index` and open [the searchable catalog](catalog/index.html).
Return to the [project guide](../README.md) for setup and commands.

| Location | Meaning |
|---|---|
| `local/` | New exploratory `run.json` bundles; ignored by Git |
| `published/` | Reviewed immutable bundles; tracked except large Release assets |
| `legacy/` | Original pre-refactor `job.json` jobs; ignored by Git |
| `scratch/` | Historical loose outputs, document renders, temporary builds and packages; ignored |
| `catalog/` | Generated navigation HTML, runs/files/source CSVs; ignored |
| `index.html`, `releases.toml` | Selected evidence and large-download metadata |

Read a bundle's report before its plots. Completed means generation completed,
not that a fit succeeded. Publishing preserves evidence locally; committing and
pushing are separate operations. Never edit a completed published bundle.
[Scientific handoff](../docs/RESEARCH_HANDOFF.md).

`neuristor runs manage` enables confirmed, recoverable delete/restore for eligible
local bundles. Trash stays under `local/.trash/`. Published, running and referenced
runs are protected. [Management guide](../docs/RUN_MANAGEMENT.md).

New sample studies embed GIFs in one shareable `parameter_maps.html`. Historical
maps may need adjacent `GIFs/`; keep their full bundle together. The complete
1,258-GIF collection and provenance package are linked from [selected results](index.html)
and the same repository's Release. Extract them into the matching
`published/20261007_074702_portable-screened-parameter-map-with-point-gifs_eaf6ab/`
folder, preserving the GIFs subfolder. The provenance archive also expects the
original ZIP in that folder. Run `neuristor validate` after reconstruction.

Former `public_jobs/`, root-level local `runs/`, `jobs/`, `outputs/`, `output/`,
`tmp/`, `artifacts/`, and `results/` are consolidated here.
[Exact move map and preservation details](../docs/REPOSITORY_LAYOUT.md).
