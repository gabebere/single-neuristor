# Current presentation

`main.tex` is the editable 22-slide Beamer deck; `main.pdf` is the existing
reviewed export. For viewing and portable setup, follow the
[project instructions](../../README.md#view-the-presentation-amir-and-other-collaborators).
The `.command` launcher starts the same restricted local viewer service.

## Sources

| Source | Content |
|---|---|
| `main.tex` | Model equations, specimen resistance fit and the presentation structure |
| `loss_function_story.tex` | Parameter-to-trace and frequency/amplitude loss definitions |
| `loss_comparison_results.tex` | Matched search results and representative trajectories |
| `parameter_exploration.tex` | Parameter maps and common capacitance scaling |
| `mica_specimen.tex` | M2 1600 nm static fit and conditional parameter exploration |
| `process_flow.tex` | End-to-end data, model and inference workflow |
| `alternating_amplitudes.tex` | Recovered shallow/deep resets, timescales and qualifications |
| `assets/` | Images used by these sources, with provenance records |

Old backup slide fragments, their unused images, July animation copies and the
one-off figure generator were removed. They are recoverable from
`cleanup-before-pruning-20261026`; their numerical evidence remains in the
original result bundles. [Recovery details](../REPOSITORY_LAYOUT.md).

## Scientific interpretation

The slides summarize diagnostic searches rather than an accepted sample
calibration. Input labels use measured baseline-corrected current, not mV filenames.
The six major-loop parameters constrain static R(T); gamma, Se, C and Cth are
unknown dynamic constants. No slide or screenshot is a numerical data source.

Detailed methods and limits:

- [Frequency-only versus frequency/amplitude fitting](../LOSS_COMPARISON_20261020.md).
- [44 measured-input loss animations](../LOSS_ANIMATIONS_20261021.md).
- [Controlled parameter trends and scaling](../PARAMETER_EXPLORATION_TRENDS_20261020.md).
- [M2 1600 nm exploration](../M2_RT_EXPLORATION_20261021.md).
- [Alternating amplitudes and timestep limitations](../ALTERNATING_AMPLITUDES_20261025.md).
- [Current scientific handoff](../RESEARCH_HANDOFF.md).

## Viewer dependencies

The PDF buttons use `http://127.0.0.1:8767/`. Keep the complete project with its
supplied `runs/published/` bundles when copying it to another computer. The
launcher checks these dependencies before opening the PDF:

- Loss gallery `20261021_054434_…_15921f`.
- Fixed-scale map `20261007_123036_…_4325d1`.
- That map's original media bundle `20261007_074702_…_eaf6ab`.
- Completed `*m2-1600nm*review*` maps.
- Completed `*fixed-alternating-amplitude-classic-gifs*` galleries.

The server serves these reviewed bundles and their media only. It resolves
historical figure-name case differences when unambiguous. The original viewer
links and GIF bytes remain unchanged. Private papers are not needed.

## Build

Compile from this directory with Tectonic or a TeX installation, putting temporary
output under `runs/scratch/`. Keep the referenced published figure bundles present.
Inspect the resulting slides before replacing `main.pdf` and update its approved
hash in `../distribution.toml`. Folder cleanup does not require regenerating the
existing PDF: all current TeX inputs and referenced images are retained.
