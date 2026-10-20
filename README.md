# Single VO₂ Neuristor

Fit a measured resistance loop, explore an ideal-current model, and share a
clickable parameter map with synchronized current/voltage/R(T) animations.

The project now has one end-to-end command for a new sample. Historical analyses
remain available, but are not prerequisites for using the program.

This is the definitive repository for the VO₂ project. New collaborators should
read the [plain-language project guide](docs/PROJECT_GUIDE.md) and
[short research handoff](docs/RESEARCH_HANDOFF.md); the complete dated record is
preserved separately in [RESEARCH_HISTORY.md](docs/RESEARCH_HISTORY.md).

**Sharing:** experimental numerical data and our generated results are approved
for publication. Third-party paper/manuscript PDFs stay in local ignored
`private/` storage, not Git or Releases. See [DATA_POLICY.md](docs/DATA_POLICY.md).

## Start here

Open [START_HERE.html](START_HERE.html) for a browser-friendly project directory,
or [START_HERE.md](START_HERE.md) for a plain-language guide.

1. [Run a new sample study](docs/SAMPLE_STUDY.md): inputs, units, bounds, outputs.
2. [Selected results](results/index.html): labeled evidence, maps and large downloads.
3. [Paper reproduction](paper/README.md): measured inputs → recipes → supporting outputs.
4. [Research handoff](docs/RESEARCH_HANDOFF.md): what is supported and what remains unresolved.
5. [Source guide](src/neuristor/README.md): folders, responsibilities, and compatibility imports.

**Scientific status:** the current solver and static R(T) fit are tested. A
physically supported quantitative match to the laboratory TIA oscillations is
still unresolved. A retained point in a parameter map is a conditional model
result, not an experimentally validated sample calibration.

## Install

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate with `.venv\Scripts\Activate.ps1`.

## One configuration, one study

Copy [the commented example recipe](experiments/studies/sample_parameter_map.toml).
Edit the resistance-data path and column names/units, current, ambient temperature,
initial conditions, parameter axes and time window. Do not edit solver code.

```bash
neuristor study plan --config experiments/studies/sample_parameter_map.toml
neuristor study run  --config experiments/studies/sample_parameter_map.toml
```

The example starts with 27 combinations. The plan checks acquisition order,
units, frequency ceiling, grid size and timestep settings before doing any fit.
For a different current, use an explicit override:

```bash
neuristor study run --config experiments/studies/sample_parameter_map.toml \
  --set input.amplitude_uA=600
```

The workflow:

- Fits six major-loop resistance parameters to the supplied numerical R(T) curve.
- Samples Cth, Se and C at the configured constant ideal current.
- Screens settled voltage extrema, temperature, proximity response and frequency.
- Rechecks survivors at two smaller timesteps.
- Generates finer-step GIFs for every retained point, with explicit resolution warnings.
- Embeds all GIFs and the fitted R(T) image into **one standalone HTML file**.

Share `parameter_maps.html` alone. No ZIP, server or internet is needed. Large
grids create large HTML files; check the size before emailing. The complete study
folder separately preserves raw measurements, numerical checks and frozen source.
See [the detailed output contract](docs/SAMPLE_STUDY.md#output-folder).

Major-loop data do not identify gamma or the dynamic thermal/electrical constants.
Those remain declared assumptions or search axes. The default 110 **MHz** cutoff
is an operating screen, not a universal physical law. The default temperature
screen uses the measured R(T) range, without extrapolation.

## Explore oscillation boundaries

```bash
neuristor analyze boundary-maps --config experiments/current/adaptive_boundary_maps.toml
```

This two-stage search expands a coarse log grid before refining the active domain
and verifying both sides of transition brackets at two smaller timesteps. Slow
transients extend their observation window; unresolved or excluded points remain
visible. Search limits and temperature/frequency screens are never labeled as
physical oscillation boundaries.

The deterministic model has an open common-capacitance scale: multiplying C and
Cth together preserves trajectory shape and changes its timescale. The search
therefore explores C/Cth and Se, then reconstructs a three-parameter view with
its open scale directions labeled. See [method and output guide](docs/BOUNDARY_SEARCH.md).

## Find and preserve evidence

```bash
neuristor runs index
neuristor runs list
neuristor runs show RUN_ID
neuristor runs publish RUN_ID
neuristor runs manage
```

Open [artifacts/index.html](artifacts/index.html). Its CSV inventory also covers old
loose figures, data, PDFs and presentations. The hub links canonical files instead
of making duplicate archives. Rebuild it after new runs or publication.

Files formerly delivered on Desktop are linked in the
[consolidated Desktop guide](archive/desktop_consolidation_20261016/index.html).
Their unique notes are preserved under `archive/`; existing data and GIF bundles
stay in their canonical locations without another copy.

Exploratory runs go to ignored `runs/`. Publishing copies a completed bundle to
`public_jobs/`; it does **not** commit or push to GitHub. Old bundles are immutable.
Generation status does not establish fit quality: always read the report.

To remove unwanted local trials, run `neuristor runs manage` and open its printed
local URL. **Delete…** asks for confirmation and moves the complete run into
recoverable `runs/.trash/`; **Restore** brings it back without overwriting files.
Published evidence, running jobs and recipe-referenced runs are protected. The
exported file-based catalog remains read-only. Trash does not free disk space.
Terminal equivalents: `neuristor runs delete RUN_ID`, `neuristor runs trash`, and
`neuristor runs restore TRASH_ID`. [Management guide](docs/RUN_MANAGEMENT.md).

## Other supported commands

| Task | Command |
|---|---|
| Single current simulation | `neuristor simulate current --config FILE.toml` |
| Original voltage circuit | `neuristor simulate voltage --config FILE.toml` |
| Standalone R(T) fit | `neuristor fit resistance --data FILE.tsv --method major-loop` |
| Laboratory replay | `neuristor analyze replay-lab --config FILE.toml` |
| Existing screened-map workflow | `neuristor analyze physical-maps --config FILE.toml` |
| Attach GIFs to an existing map | `neuristor analyze map-animations --config FILE.toml` |
| Update historical map displays without resimulation | `neuristor analyze refresh-maps --all-existing` |
| Frequency/amplitude fitting | `neuristor analyze fit-steady --config FILE.toml` |
| Parameter sweep | `neuristor sweep run --config FILE.toml` |
| Read-only archive dashboard | `neuristor dashboard` |

Use `neuristor --help` and the relevant command's `--help`.
[Map display guide](docs/MAP_DISPLAY.md): heatmap axes and color scales now stay
fixed across slider positions. The artifact hub links historical maps to their
updated viewers while preserving the original exports.
[Full workflow reference](docs/WORKFLOW_REFERENCE.md) preserves the earlier detailed
command guide and historical interpretations. [Archive chronology](docs/ARCHIVE_INDEX.md)
and [research handoff](docs/RESEARCH_HANDOFF.md) take precedence over older fit claims.

## Repository responsibilities

| Folder/file | Responsibility |
|---|---|
| `src/neuristor/` | Authoritative physics plus separate analysis, fitting, plotting, orchestration and interface folders |
| `experiments/` | Editable, versioned TOML recipes with explicit units |
| `tests/` | Numerical fidelity, convergence, workflows and artifact tests |
| `data/experimental/` | Canonical measured numerical inputs |
| `presets/` | Archived resistance fits and parameter sets |
| `runs/` | Ignored exploratory study/run bundles |
| `public_jobs/` | Reviewed immutable bundles; Git-trackable, not necessarily pushed |
| `artifacts/` | Lightweight navigation hub; no duplicate media/database |
| `results/` | Selected-result front door and same-repository large-artifact downloads |
| `private/` | Local third-party reference PDFs and unapproved material; never distributed |
| `paper/` | Reproduction index linking claims to inputs, commands and outputs |
| `archive/` | Guide to historical trials, superseded methods and legacy files |
| `environments/` | Recorded reference environment and release reproducibility guidance |
| `docs/` | Methods, handoff, manuscript and historical interpretations |
| `output/`, `outputs/`, `Simulations_on_VO2/` | Preserved legacy document/output locations |
| `legacy_scripts/` | Historical scripts, not the active API |
| `references/` | Source papers and unmodified original implementation |

The ideal current model is `C dV/dt = I − V/R(T)`. It uses the same hysteresis
implementation as the Yuanhang voltage model. Its steady metallic voltage floor
is approximately `I Rm`; capacitance changes the approach time, not that floor.
No hidden series resistor changes an imposed ideal current. `C=0` is supported by
the solver exactly; logarithmic three-axis maps currently require positive C.

## Development and verification

Read [AGENTS.md](AGENTS.md) before changing code. Unit conversion stays in the
workflow layer; interfaces do not independently implement equations.

```bash
pytest -q
neuristor validate
# After staging the intended release changes:
neuristor validate --publication
# Before uploading a large result package:
neuristor validate --distribution PATH_TO_PACKAGE
```

Run the affected recipe as well. Physics changes additionally require two smaller
timesteps and a convergence assertion. The new sample pipeline reuses the existing
physics: it does not replace the resistance law or invent smooth R(T) paths.

The pre-refactor baseline is recoverable at the Git tag `v0.1.0-working-baseline`.
See [CONTRIBUTING.md](CONTRIBUTING.md), [citation metadata](CITATION.cff), and
[release checklist](docs/RELEASE_CHECKLIST.md). The canonical software, approved
inputs, documentation and compact reviewed evidence are on GitHub `main`;
[large map assets](https://github.com/gabebere/single-neuristor/releases/tag/research-library-2026-10-20)
are in the same repository's dated Release. Ignored exploratory history is
preserved in the private full local recovery backup, not claimed as a complete
public archive. This is not a DOI deposit or completed quantitative calibration.
The [20 October consolidation record](docs/CONSOLIDATION_20261020.md) distinguishes
the local backup, current-tree cleanup, and verified remote publication status.
Original Zhang code:
[references/yuanhangzhang98-collective_dynamics_neuristor-217d4f0](references/yuanhangzhang98-collective_dynamics_neuristor-217d4f0/).

Gabriel Berezovsky · supervised by Amir Gildor and Prof. Yoav Kalcheim.
