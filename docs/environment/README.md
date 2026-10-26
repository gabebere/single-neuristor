# Environments and reproducibility

Install from `pyproject.toml`; it is the authoritative dependency declaration.
For development use `python -m pip install -e '.[dev]'`.

`reference_environment.json` records the interpreter/platform and directly used
package versions on the local machine during this reorganization. It is a
historical environment **record**, not a portable, dependency-complete lockfile.
It includes Streamlit, which was subsequently removed from the maintained
package; installing the current project does not require it. Do not treat
versions from one macOS/Python environment as a validated Linux installation.

The GitHub workflow installs in a fresh hosted environment and runs tests,
repository validation, and the read-only sample plan. It is configured locally
but has not run on GitHub until these changes are reviewed and pushed.

A paper release still needs a fully resolved environment/lock for its chosen
platform plus fresh-machine reproduction of the selected evidence. Record exact
versions and checksums with the release, not an unsupported claim that all past
trials used the same dependencies.
