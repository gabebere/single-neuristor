# Paper reproduction: inputs → recipe → outputs

This is the publication-oriented evidence index. It references existing canonical
files rather than copying figures or creating a second archive. It is not a claim
that sample fitting is finished, and final paper figure numbers are not assigned.

| Evidence ID | What it establishes | Reproduce from the repository root |
|---|---|---|
| `yuanhang_voltage_control` | Original-circuit mechanism control with declared thermal-capacitance choice. | `neuristor simulate voltage --config experiments/voltage/yuanhang_oscillator.toml` |
| `ideal_current_valley` | Ideal-current oscillations can have a nonzero metallic valley; approximately I Rm. | `neuristor simulate current --config experiments/current/nonzero_voltage_valley.toml` |
| `sample_static_resistance` | A major-loop R(T) fit; gamma remains assumed. | `neuristor fit resistance --data data/experimental/100425_chip1_gap3.tsv --method major-loop --bootstrap-samples 0 --seed 42` |
| `sample_map_demonstration` | Full fit → screened map → GIF → standalone HTML software demonstration, not dynamic sample calibration. | `neuristor study run --config experiments/studies/sample_parameter_map.toml` |
| `frozen_lab_comparison` | A recorded forward comparison and sensitivity study, including disagreement with measured oscillations. | `neuristor analyze model-validation --config experiments/current/specimen_model_validation.toml` |

The machine-readable [reproduction.toml](reproduction.toml) records inputs,
commands, expected output filenames, scientific scope, and available evidence
links. Structural tests verify its paths and recipe declarations; they do not
assert that every historical figure is reproduced or that the model matches data.
The existing [reviewed chronology](../ARCHIVE_INDEX.md) is the full historical
evidence map. Use [the artifact library](../../runs/catalog/index.html) for all trials.

## Recreate the complete presentation dependency set

```bash
neuristor reproduce-presentation --plan
neuristor reproduce-presentation
neuristor present
```

The plan is read-only. The rebuild uses `experiments/reproduce_presentation.toml`
and the original numerical measurements, source templates, optimizer seed vectors
and declared historical alternating-reset parameters. It reads no existing run
bundle, rendered result, Release asset or private reference PDF. The frozen
alternating-reset vector is an explicit setting for that conditional experiment;
it is replayed, rather than replaced by a new optimization.

The full recipe executes 18 checkpointed stages:

1. Specimen major-loop fit with 1,000 bootstrap samples.
2. Frequency-only and frequency/amplitude searches with the original budgets,
   followed by all-current checks at three timesteps, comparison tables and 44 GIFs.
3. The original 12,167-point parameter grid and GIFs for every retained point.
4. The extended controlled scaling/trend checks.
5. Mica static fitting, four initial and two expanded grids (2,616 entries),
   the 36-point pilot, 24 finer startup checks, and diagnostic interactive maps.
6. Fixed-vector alternating-amplitude replays at three timesteps, 44 GIFs, and
   the central longer constant-current control at all three timesteps.

Every stage delegates to the authoritative physics and existing numerical
analysis. The original screens and convergence settings remain in the linked
recipes. Excluded or unconverged results stay labeled. No new specimen
calibration, complete physical boundary or period-doubling claim is implied.

Allow several hours and potentially overnight; this is compute- and
storage-intensive. A short developer smoke profile exercises the same stage
connections with smaller searches/grids and fewer animation frames; it is not
a full scientific reproduction and cannot activate the presentation links.

The output is a standard parent RunBundle with relative child-stage paths,
actual environment versions, source/input hashes, numerical evidence, figures,
GIFs, HTML, and `presentation_manifest.json`. `START_HERE.html` shows the freshly
computed results. The original PDF remains immutable; editable TeX with fresh
figure links and result tables is emitted under `presentation/` for inspection
and optional compilation. Original experimental device photographs remain inputs.

Only after a full run completes does an atomic local pointer connect
`neuristor present` and its four viewer routes to the new outputs. An interrupted
or smoke run never replaces the active presentation. To resume, rerun with
`--resume PATH_TO_THE_INTERRUPTED_BUNDLE`. Completed stages are reused only if
their content checksums, source, input data, recipe and environment still agree.
A failed partial stage is rerun. Repeating without `--resume` starts a fresh experiment.

The original fitting searches have wall-clock limits as well as fixed seeds and
an evaluation cap. Hardware and package differences can therefore change the
selected parameters; inspect the new comparison tables instead of assuming
byte-identical optimization results. The environment record is descriptive, not
a portable dependency lock. This command verifies reproducibility of the declared
workflow, not the scientific correctness of a quantitative specimen fit.

## Review a reproduced result

1. Read the assumptions in the recipe and the generated `resolved_config.json`.
2. Inspect numerical CSVs and `metrics.json` before interpreting plots.
3. Read `report.md`, including exclusions, uncertainty, and unresolved mismatch.
4. For the sample-study recipe, use `stages.json` to locate `verification.csv`
   in its grid stage and `animation_checks.csv` in its media stage. A blank
   coarse-screened cell is not proof of no oscillator.
5. Preserve the new bundle; do not overwrite a historical reference bundle.

Commands generate unique ignored `runs/local/` folders. No ZIP is required for new
standalone maps. The tiny map example uses 27 parameter combinations and no
bootstrap uncertainty calculation; it is a workflow demonstration, not an
exhaustive parameter search. Older manuscript tables may use a different
bootstrap count; reproduce those from their recorded run/configuration.

## Before publishing

Finalize which evidence supports each claim and figure, confirm collaborators'
data permissions, review attribution/authorship, and record an environment for
the release. [Release checklist](../RELEASE_CHECKLIST.md).
The local tests, wheel-install check, and recipe reruns do not substitute for a
dependency-isolated fresh-machine reproduction or an externally archived release.
