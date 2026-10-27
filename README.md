# Single VO₂ Neuristor

Simulate a VO₂ neuristor, fit measured resistance loops, and explore current-driven
oscillations with interactive maps and synchronized voltage/R(T) animations.
Gabriel Berezovsky · supervised by Amir Gildor and Prof. Yoav Kalcheim.

**Start with this page.** It contains project navigation, installation, and Amir’s presentation setup. The [research handoff](docs/RESEARCH_HANDOFF.md)
states the current conclusions and next unresolved question.

The solver and static R(T) fitting are tested. A physically supported quantitative
match to the laboratory TIA oscillations remains unresolved. A completed run means
its outputs were generated; it does not establish experimental agreement.

## Find what you need

| I want to… | Open or run |
|---|---|
| View the slides and their interactive examples | [Presentation setup below](#view-the-presentation-amir-and-other-collaborators) |
| Browse selected maps, animations and reports | [Results library](runs/index.html) |
| Search every job on this computer | `neuristor runs index`, then [searchable catalog](runs/catalog/index.html) |
| Run a new measured sample | [Sample-study guide](docs/SAMPLE_STUDY.md) and [example recipe](experiments/studies/sample_parameter_map.toml) |
| Continue the research | [Current handoff](docs/RESEARCH_HANDOFF.md) |
| Find measurements or saved parameters | [Data guide](data/README.md) |
| Understand or change the code | [Codebase map](docs/CODEBASE_MAP.md), [contributing](CONTRIBUTING.md), [AGENTS.md](AGENTS.md) |
| Read methods, reports or presentations | [Document index](docs/README.md) |
| Reproduce supporting evidence | [Reproduction guide](docs/reproduction/README.md) |
| Understand previous approaches | [Research history](docs/RESEARCH_HISTORY.md) and [reviewed chronology](docs/ARCHIVE_INDEX.md) |

## View the presentation: Amir and other collaborators

The presentation’s interactive examples can use supplied reviewed bundles or
results recreated locally with the optional step below. For a supplied copy,
keep `docs/presentation/`, `src/`, and `runs/published/` together; older viewers
need their adjacent GIFs. In a Terminal opened inside the project, run:

```bash
python3 --version
python3 src/neuristor/interfaces/presentation_server.py --present
```

Python **3.9+** is enough for this viewer. It opens the existing PDF and starts
all linked interactive viewers locally. No extra Python packages, LaTeX, or
internet connection are needed. Run it again after restarting the computer.
On macOS you can also double-click
[Start presentation viewers.command](docs/presentation/Start%20presentation%20viewers.command).
If the project is installed, `neuristor present` does the same thing.

## Install to run simulations

Use Python **3.10+**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows activate with `.venv\Scripts\Activate.ps1`. For future sessions,
activate the environment again before running `neuristor`.

## Optional final setup step: recreate the presentation results

**Optional, but recommended to demonstrate reproducibility.** After installing
the project, run:

```bash
neuristor reproduce-presentation
neuristor present
```

The first command reruns the presentation’s resistance fitting, both dynamic
parameter searches, parameter grids, scaling checks, Mica studies, and animated
current/voltage/R(T) replays. It creates the numerical tables, figures, GIFs and
interactive HTML locally, then connects the presentation buttons to those new
results. You do not need our saved simulation bundles or large GIF downloads.
The original laboratory measurements and declared model/optimizer settings are
necessary inputs and are included in the repository; the program does not
recreate physical measurements.

**Compute-intensive:** plan for a few hours, potentially overnight on slower
computers. Expect sustained CPU use, significant memory use and several GB of
output. This is a planning allowance, not a measured guarantee. The command
prints stage progress and saves completed stages. If interrupted, run
`neuristor reproduce-presentation --resume "PATH_PRINTED_BY_THE_COMMAND"`.

For a fast preview of the work without running it, use
`neuristor reproduce-presentation --plan`. The versioned recipe is
[reproduce_presentation.toml](experiments/reproduce_presentation.toml).

Fresh results, editable slide sources and all dependencies stay together in a
new ignored `runs/local/` bundle, with `START_HERE.html` as its entry point.
The supplied PDF remains the historical presentation; the recreated HTML and
slide tables show the new results. Optimizer time budgets and software/hardware
can affect the selected candidates, so inspect the new report when comparing
numbers. Original results are preserved. [Scope and reproduction details](docs/reproduction/README.md).

## Run a new sample

Copy the [commented TOML recipe](experiments/studies/sample_parameter_map.toml).
Edit the measured-data path, column names and units, current, ambient temperature,
parameter axes, and time window in the recipe; keep measured originals unchanged.

```bash
neuristor study plan --config experiments/studies/sample_parameter_map.toml
neuristor study run --config experiments/studies/sample_parameter_map.toml
```

The example tries 27 combinations. It fits the static R(T) loop, screens dynamic
parameters, checks survivors at two smaller timesteps, and exports
`parameter_maps.html` with all GIFs embedded. That HTML can be shared alone.
The full run folder preserves the inputs, numerical checks, report and frozen code.
Use `--set input.amplitude_uA=600` for an explicit current override.
[Input/output contract and interpretation](docs/SAMPLE_STUDY.md).

Major-loop data do not identify gamma or the dynamic thermal/electrical constants.
The 110 MHz cutoff is an operating screen, not a universal physical limit.

For the optional hot-filament/cold-surroundings experiment on the presentation
measurements, see [the declared model and comparison protocol](docs/FILAMENT_SURROUNDINGS_20261027.md).

## One home for jobs and results

All generated work belongs under [runs/](runs/README.md):

| Location | Contents |
|---|---|
| `runs/local/` | New exploratory bundles; ignored by Git |
| `runs/published/` | Reviewed preserved bundles, with unchanged run IDs and contents |
| `runs/legacy/` | Older `job.json` jobs; local and ignored |
| `runs/scratch/` | Loose historical outputs, render workspaces and temporary packages; ignored |
| `runs/catalog/` | Rebuildable searchable HTML and CSV inventories; ignored |
| `runs/index.html` | Selected results and same-repository Release downloads |

```bash
neuristor runs index
neuristor runs list
neuristor runs show RUN_ID
neuristor runs publish RUN_ID
neuristor runs manage
```

Publish copies a bundle into `runs/published/`; it does not commit or push it.
Read an unfamiliar run in this order: `report.md`, `resolved_config.json`,
`metrics.json` and tables, `run.json`, then figures. Legacy jobs use `job.json`.
Never edit a completed preserved bundle; create a new run.

The local manager can move eligible local runs into recoverable trash and restore
them. Published, running and recipe-referenced runs are protected.
[Management guide](docs/RUN_MANAGEMENT.md).

## Folder map

| Folder | Purpose |
|---|---|
| `src/neuristor/` | Physics, analysis, fitting, plotting, workflows and interfaces |
| `tests/` | Numerical, workflow, command and artifact verification |
| `experiments/` | Editable versioned experiment recipes |
| `data/` | Measured inputs and saved parameter sets in `data/parameters/` |
| `runs/` | All jobs, results, generated outputs and their catalog |
| `docs/` | Methods, handoff, history, presentations, reproduction and environment records |
| `private/` | Local reference PDFs and unapproved material; never distributed |

[Migration record](docs/REPOSITORY_LAYOUT.md) maps every former folder to its new
home and records recovery points for retired files. Active imports use the
purpose-specific source groups; the authoritative physics remains in `model.py` and `current_drive_sim.py`.

## Common commands

| Task | Command |
|---|---|
| Current-source simulation | `neuristor simulate current --config FILE.toml` |
| Original voltage circuit | `neuristor simulate voltage --config FILE.toml` |
| Static resistance fit | `neuristor fit resistance --data FILE.tsv --method major-loop` |
| Recreate presentation results | `neuristor reproduce-presentation` |
| Laboratory replay | `neuristor analyze replay-lab --config FILE.toml` |
| Screened maps | `neuristor analyze physical-maps --config FILE.toml` |
| Adaptive boundary search | `neuristor analyze boundary-maps --config FILE.toml` |
| Map animations | `neuristor analyze map-animations --config FILE.toml` |
| Update historical map displays | `neuristor analyze refresh-maps --all-existing` |
| Parameter-trend checks | `neuristor analyze validate-trends --config FILE.toml` |
| Frequency/amplitude fitting | `neuristor analyze fit-steady --config FILE.toml` |
| Parameter sweep | `neuristor sweep run --config FILE.toml` |
| Local run library | `neuristor runs manage` |

Use `neuristor --help` and each command's `--help`.
[Boundary search](docs/BOUNDARY_SEARCH.md) · [Map displays](docs/MAP_DISPLAY.md).

## Develop and verify

Read [AGENTS.md](AGENTS.md), [the codebase map](docs/CODEBASE_MAP.md), and the relevant
recipe before changing implementation. Units are converted at workflow boundaries;
interfaces reuse the physics. The ideal-current model is
`C dV/dt = I − V/R(T)`, with exact `C=0` behavior and metallic floor near `I Rm`.

```bash
pytest -q
neuristor validate
neuristor validate --publication  # inspect the staged publication payload
```

Run the affected recipe too. Physics changes require two smaller timesteps and a
convergence assertion. [Contribution rules](CONTRIBUTING.md).

Experimental numerical data and our own generated results are approved for
sharing. Third-party paper/manuscript PDFs remain in ignored `private/` storage.
[Data policy](docs/DATA_POLICY.md) · [Release checklist](docs/RELEASE_CHECKLIST.md) ·
[Citation](CITATION.cff) · [Third-party attribution](THIRD_PARTY_NOTICES.md).
The pre-refactor baseline remains at tag `v0.1.0-working-baseline`; the checkpoint
before this reorganization is `b2ee2ff`. Neither a local commit nor a completed
catalog is a remote backup or a scientific calibration.
