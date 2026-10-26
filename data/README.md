# Data: measurements, not model outputs

`experimental/` contains the canonical numerical measurements used by supported
recipes. Treat originals as read-only; put transformed tables inside a run bundle
and record units and the source hash.

- `experimental/100425_chip1_gap3.tsv`: measured resistance versus temperature,
  with acquisition order needed to distinguish heating and cooling.
- `experimental/m2_8.9_1stL_1600nm_A_RT.tsv`: untouched user-supplied M2 1600 nm
  resistance loop; 374 measurements in K and ohm with acquisition time in seconds.
  [Exploration and provenance](../docs/M2_RT_EXPLORATION_20261021.md).
- `experimental/tia_current_sweep/`: professor-provided current/voltage exports,
  original files, units/provenance notes, and SHA-256 checksums. Files marked
  `converted` were supplied for the paper's figure 7; other originals for figure 6.

The mV in an acquisition filename is a nominal drive label, not proof that the
sample current has that value in µA. Use the measured current channel.

Historical sample-library imports also live under `presets/samples/data/` and
`new_samples/`; these remain separate from the canonical TIA dataset.
Gabriel approved public release of this project's experimental numerical data
and our generated results on 20 October 2026. Preserve provenance and do not
extend that permission to new unapproved inputs or third-party manuscripts.
See [the disclosure policy](../docs/DATA_POLICY.md).
