# Legacy helpers

`plots.py` and `sample_library.py` support older saved figures/sample presets.
They are retained because historical sources and imports depend on them, but
they are not the new sample-study interface. No new workflow should depend on
these helpers without a documented reason.

Historical one-off command scripts are outside the package in
[legacy_scripts/](../../../legacy_scripts/README.md). The actual scientific model
has not been copied into this folder.
