# Local trials

Supported commands write a new dated run bundle here. Open the project
[artifact library](../artifacts/index.html) to browse them. Study stages are
nested inside their parent study, with locations recorded in `stages.json`.

Run outputs are ignored by Git. This guide is the exception. A local folder is
not an external backup. To preserve reviewed evidence, use
`neuristor runs publish RUN_ID`; then separately review/commit/push the chosen
bundle. Keep failures and partial evidence until their provenance is reviewed.

`refactor_verification/` contains ignored packaging smoke-test products. It is
not a reviewed simulation or an additional supported package installation.
