# Start here

This folder contains the VO₂ model, measured inputs, experiment recipes, and the
record of our attempts to match laboratory measurements. You do not need to read
Python to browse results or prepare a new sample study.

| I want to… | Open this |
|---|---|
| Understand what was built and where everything belongs | [Project guide](docs/PROJECT_GUIDE.md) |
| Continue in a new chat without the old conversation | [Short research handoff](docs/RESEARCH_HANDOFF.md) |
| Know what may be shared publicly | [Data and reference-PDF policy](docs/DATA_POLICY.md) |
| Browse selected simulations, maps, animations and downloads | [Results](results/index.html) |
| Browse every trial on this computer | Run `neuristor runs index`, then open [the local catalog](artifacts/index.html) |
| Remove and restore unwanted local runs | [Local-manager guide](docs/RUN_MANAGEMENT.md) |
| Run my own measured R(T) sample | [Sample-study guide](docs/SAMPLE_STUDY.md) and [editable recipe](experiments/studies/sample_parameter_map.toml) |
| Find the original measurements | [Data guide](data/README.md) |
| Read the report or methodology | [Document guide](docs/README.md) |
| Reproduce supporting evidence for a paper | [Paper guide](paper/README.md) |
| Understand the code | [Source guide](src/neuristor/README.md) and [module map](docs/CODEBASE_MAP.md) |
| Understand previous approaches and failures | [History guide](archive/README.md) |
| Find the project files formerly on Desktop | [Consolidated Desktop guide](archive/desktop_consolidation_20261016/index.html) |
| Know which conclusions are still unresolved | [Research handoff](docs/RESEARCH_HANDOFF.md) |

For a browser-friendly version, open [START_HERE.html](START_HERE.html).

## What counts as a result?

`runs/` contains local trials. `public_jobs/` contains locally published,
preserved bundles. A **completed** run only means the program finished writing
its outputs. It does not mean its parameters reproduce the experiment.

The static resistance fit and numerical implementation have verification tests.
A physically supported quantitative match to the TIA oscillations remains open.
The 110 MHz cutoff is an operating screen, not a universal material limit.

## What should I edit?

Edit a TOML recipe in `experiments/`, not a solver file. Keep original measured
files unchanged. New work writes a new run folder; it never overwrites a published
one. Generated scratch folders are not part of the supported source library.

Publishing a bundle locally is not a GitHub push. Experimental numerical data
and our generated results are approved for sharing; third-party reference PDFs
are not. See [the disclosure policy](docs/DATA_POLICY.md),
[consolidation status](docs/CONSOLIDATION_20261020.md) and
[release checklist](docs/RELEASE_CHECKLIST.md).
