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
