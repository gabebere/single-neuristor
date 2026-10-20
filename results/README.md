# Results: one front door

Open [index.html](index.html) for selected evidence and the large-map download.
For every local trial, run `neuristor runs index` and open
`artifacts/index.html`. That generated catalog is intentionally not a second
versioned database.

Software/data/reports and compact reviewed bundles live in this repository.
The complete 1,258-GIF historical viewer is a versioned asset in the **same**
repository's Releases. Its old HTML is not standalone; keep the included GIF
folder beside it. New sample-study HTMLs embed their GIFs.

## Scientific labels

- **Control:** tests the implementation, not the experimental sample fit.
- **Static fit:** fits measured R(T), not dynamic oscillations or gamma.
- **Conditional exploration:** holds assumptions; does not calibrate a sample.
- **Diagnostic / rejected fit:** evidence of disagreement, not adopted parameters.
- **Display revision:** same numerics with a corrected viewer, not new physics.

Completed bundles stay at their original `public_jobs/RUN_ID` paths; renaming
them would break frozen provenance. `runs/` preserves local exploratory and
software trials. The full pre-cleanup local backup also preserves ignored work;
it is not a public release and contains restricted reference PDFs.

## Download the complete animation collection

Download `VO2_parameter_map_with_GIFs.zip` from the linked GitHub Release,
extract it into an empty directory, and open `parameter_maps.html`. This
original corrected viewer works without Python or a server.

To use the updated fixed-scale viewer from a source checkout, extract the ZIP
into `public_jobs/20261007_074702_portable-screened-parameter-map-with-point-gifs_eaf6ab/`
inside that checkout, preserving the `GIFs/` subfolder. Then open the updated
viewer linked from this page's index. This restores the media dependency; it
does not reconstruct all original numerical/source provenance. The complete
local bundle and private full recovery snapshot preserve those separately.

For the complete reviewed bundle, the same Release also supplies
`VO2_parameter_map_provenance.tar.gz`: frozen source, resolved inputs, checks,
original manifest and PNGs, without duplicating the GIFs. Save the original ZIP
itself inside the canonical `public_jobs/...eaf6ab/` folder, extract its contents
there, then extract the provenance tar into that same folder. This restores all
declared artifacts; run `neuristor validate` afterward. No private paper PDF is
included in either asset. The old intermediate packaging mirror is unnecessary.

See [the concise handoff](../docs/RESEARCH_HANDOFF.md),
[chronology](../docs/ARCHIVE_INDEX.md), [release metadata](releases.toml), and
[data policy](../docs/DATA_POLICY.md).
