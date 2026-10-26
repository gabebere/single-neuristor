# Codebase map

The [root README](../README.md) is the collaborator guide. This page consolidates
all former source-folder READMEs into one implementation reference.
[Repository layout and former paths](REPOSITORY_LAYOUT.md).

## Read the active workflow first

```text
experiments/studies/*.toml
  → cli.py + config.py
  → workflows.run_sample_study
  → orchestration/sample_study.py
      → fitting/resistance_custom_analysis.py: measured major-loop fit
      → orchestration/map_workflows.py: grid → screening → smaller-step checks
          → analysis/physical_maps.py + analysis/extrema.py
          → current_drive_sim.py → model.py
          → plotting/map_animations.py + plotting/replay_plots.py
  → runs.py: parent bundle and child stages
  → orchestration/artifact_catalog.py: searchable links to the evidence
```

CLI/local viewers → workflows/orchestration → analysis/fitting/plotting → physics.
Physics never imports an interface. No CLI or HTML viewer implements
an independent scientific model. Units are explicit in recipes and converted at
workflow boundaries; physics uses SI unless the dataclass field says otherwise.

## Stable core and shared infrastructure

| Module | Responsibility |
|---|---|
| [model.py](../src/neuristor/model.py) | Authoritative voltage circuit, parameter dataclasses and shared float32-faithful hysteresis memory. |
| [current_drive_sim.py](../src/neuristor/current_drive_sim.py) | Authoritative ideal-current model, exact C=0 limit and scalar/vectorized integrators. |
| [workflows.py](../src/neuristor/workflows.py) | Stable public workflow API: translates units, coordinates calculations and writes RunBundles. |
| [config.py](../src/neuristor/config.py) | Recipe loading, validation, source-relative paths and dotted overrides. |
| [runs.py](../src/neuristor/runs.py) | Manifests, discovery, publication and protected recoverable trash/restore. |
| [layout.py](../src/neuristor/layout.py) | Canonical storage lanes and read-only resolution of former recorded paths. |
| [provenance.py](../src/neuristor/provenance.py) | Recursive source snapshots and hashes, including grouped implementations. |
| [validation.py](../src/neuristor/validation.py) | Recipe, manifest, artifact and reproduction-index checks. |
| [cli.py](../src/neuristor/cli.py) | Terminal parsing and messages; delegates scientific work. |

The current-source equation is `C dV/dt = I − V/R(T,H)`; `C=0` gives `V=IR(T,H)`.
A series resistor does not change an ideal imposed current. Capacitance changes
the approach time, not the steady metallic floor near `I Rm`. Preserve hysteresis
fidelity and convergence assertions. Passing numerical checks does not establish
the unresolved quantitative TIA fit.

## Numerical analysis

Import measurements, extract cycle features, estimate conditional parameters, and screen or audit simulations. Historical detectors are distinct from the current settled-extrema analysis; use the method declared by the recipe.

| Module | Purpose |
|---|---|
| [boundary_search.py](../src/neuristor/analysis/boundary_search.py) | Adaptive exploration of the deterministic ideal-current oscillatory domain. |
| [distribution.py](../src/neuristor/analysis/distribution.py) | Structural publication checks; never upload files or infer legal permission. |
| [experimental_waveforms.py](../src/neuristor/analysis/experimental_waveforms.py) | Load and summarize the professor-supplied TIA current-sweep waveforms. |
| [extrema.py](../src/neuristor/analysis/extrema.py) | Phase-independent oscillation measurements from settled voltage extrema. |
| [frequency_sensitivity.py](../src/neuristor/analysis/frequency_sensitivity.py) | Per-input frequency derivatives from saved parameter perturbation experiments. |
| [lab_estimates.py](../src/neuristor/analysis/lab_estimates.py) | Estimate environmental thermal conductance from numerical TIA waveforms. |
| [lab_replay.py](../src/neuristor/analysis/lab_replay.py) | All-current manual comparisons using measured input histories and shared physics. |
| [loss_comparison.py](../src/neuristor/analysis/loss_comparison.py) | Common feature summaries for completed, matched steady-search experiments. |
| [metrics.py](../src/neuristor/analysis/metrics.py) | Shared numerical summaries used by experiment workflows. |
| [model_validation.py](../src/neuristor/analysis/model_validation.py) | Blind validation of the specimen current-drive model against lab waveforms. |
| [optimizer_diagnostics.py](../src/neuristor/analysis/optimizer_diagnostics.py) | Finite-difference evidence for a sampled, potentially discontinuous loss. |
| [oscillation_audit.py](../src/neuristor/analysis/oscillation_audit.py) | Window-by-window waveform evidence and controlled parameter maps. |
| [physical_maps.py](../src/neuristor/analysis/physical_maps.py) | Screened Cth/C/Se maps measured with steady-state voltage extrema. |
| [trend_validation.py](../src/neuristor/analysis/trend_validation.py) | New controlled tests of capacitance scaling and conditional parameter trends. |

## Fitting and parameter search

Fit static resistance or search dynamic parameters using the recorded objective. Static major-loop data do not identify gamma or dynamic constants. Historical fitting methods remain available to explain archived runs.

| Module | Purpose |
|---|---|
| [joint_inference.py](../src/neuristor/fitting/joint_inference.py) | Budgeted joint static/dynamic inference using the authoritative simulator. |
| [optimizer_audit.py](../src/neuristor/fitting/optimizer_audit.py) | Cached normalized-coordinate evaluations and bounded gradient score trials. |
| [parameter_inference.py](../src/neuristor/fitting/parameter_inference.py) | Global inverse fitting of the current-drive model to measured waveforms. |
| [resistance_custom_analysis.py](../src/neuristor/fitting/resistance_custom_analysis.py) | Fit specimen-specific VO2 resistance/hysteresis parameters from experimental R(T) data. |
| [steady_inference.py](../src/neuristor/fitting/steady_inference.py) | Phase-independent late-window inference with bounded independent restarts. |

## Workflow coordination

Combine existing numerical functions into reproducible studies and evidence bundles. These modules own stages, reuse checks, display-only revisions and catalog generation, not independent circuit equations.

| Module | Purpose |
|---|---|
| [artifact_catalog.py](../src/neuristor/orchestration/artifact_catalog.py) | A lightweight human-readable catalog, not another database or artifact copy. |
| [boundary_workflows.py](../src/neuristor/orchestration/boundary_workflows.py) | Coarse exploration followed by fine boundary verification in one run bundle. |
| [loss_comparison.py](../src/neuristor/orchestration/loss_comparison.py) | Bundle a comparison of two matched completed steady-search runs. |
| [loss_scopes.py](../src/neuristor/orchestration/loss_scopes.py) | Render both saved loss candidates without repeating their optimization. |
| [map_viewers.py](../src/neuristor/orchestration/map_viewers.py) | Presentation-only revisions of completed maps, without new trajectories. |
| [map_workflows.py](../src/neuristor/orchestration/map_workflows.py) | Screened map and point-media orchestration behind the public workflows API. |
| [optimizer_audit.py](../src/neuristor/orchestration/optimizer_audit.py) | Reproducible local derivative, timestep and L-BFGS-B experiments. |
| [optimizer_review.py](../src/neuristor/orchestration/optimizer_review.py) | Review completed derivative and expanded-search experiments on a common basis. |
| [sample_study.py](../src/neuristor/orchestration/sample_study.py) | One sample, one recipe, one portable evidence folder. |
| [trend_validation.py](../src/neuristor/orchestration/trend_validation.py) | Controlled fresh integrations for the documented full-map tendencies. |

## Figures and animations

Render saved or simulated numerical trajectories. Playback speed is separate from integration timestep. Do not change physical states or invent R(T) paths to improve a display.

| Module | Purpose |
|---|---|
| [boundary_maps.py](../src/neuristor/plotting/boundary_maps.py) | Display sampled classifications and boundary brackets without interpolating them. |
| [frequency_sensitivity.py](../src/neuristor/plotting/frequency_sensitivity.py) | Direct f_i(theta) profiles from saved simulations, using measured current names. |
| [loss_comparison.py](../src/neuristor/plotting/loss_comparison.py) | Presentation-ready plots from saved verification samples, without simulation. |
| [map_animations.py](../src/neuristor/plotting/map_animations.py) | Portable point-to-GIF drill-downs, using authoritative current-drive physics. |
| [optimizer_diagnostics.py](../src/neuristor/plotting/optimizer_diagnostics.py) | Plots of saved parameter perturbations with raw scientific eligibility visible. |
| [replay_plots.py](../src/neuristor/plotting/replay_plots.py) | Voltage/resistance comparison graphics and lightweight per-current animations. |
| [trend_validation.py](../src/neuristor/plotting/trend_validation.py) | Display controlled simulations without filling unresolved parameter points. |
| [visualization.py](../src/neuristor/plotting/visualization.py) | Small, consistent plotting surface for run bundles and exported viewers. |

## User interfaces

The local HTTP interfaces serve existing reports, figures and interactive HTML.
The run manager enforces lifecycle rules through RunRegistry; the portable
presentation server uses only the standard library.

| Module | Purpose |
|---|---|
| [presentation_server.py](../src/neuristor/interfaces/presentation_server.py) | Read-only loopback delivery for the presentation's local HTML viewers. |
| [run_manager.py](../src/neuristor/interfaces/run_manager.py) | Loopback-only artifact library with explicit, recoverable run deletion. |

## Retired code and saved-fit verification

Unused one-off scripts, the old voltage plotting frontend, and the dormant
current-domain search have been removed. Use current workflows and recipes.
[Recovery points](REPOSITORY_LAYOUT.md) retain their history without placing
archived code beside active implementation.

[fitting/saved_fit_replay.py](../src/neuristor/fitting/saved_fit_replay.py) retains
only the numerical detector needed to reproduce stored resistance-fit metrics.
It has no sample create/delete/rename interface and does not replace the solver.
The corresponding regression test still reproduces all six measured sample fits.

Active code imports the grouped modules directly, for example
`neuristor.analysis.extrema` and `neuristor.plotting.map_animations`.
Streamlit applications and their launch commands have been retired. Public
physics and workflow paths and scientific CLI commands remain available. Frozen sources in result bundles
retain their own original modules and are never refactored in place.

## Extend and verify

Add a pure function at the lowest useful layer, coordinate it through workflows,
expose one CLI command, and add a versioned recipe when configuration-driven.
Return a standard RunBundle containing config, metrics, report, numerical tables
and relative artifact paths. See [AGENTS.md](../AGENTS.md) and
[CONTRIBUTING.md](../CONTRIBUTING.md) for required tests and recipe checks.

`neuristor runs index` generates `runs/catalog/source_inventory.csv` with every
owned Python file's role, size and purpose without executing historical scripts.
