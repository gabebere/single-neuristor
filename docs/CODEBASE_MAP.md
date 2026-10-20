# Codebase map

Start with the sample-study recipe, not historical scripts. The stable Python
entry points remain in `neuristor.workflows`; the terminal interface is `neuristor`.
Focused study/map orchestration is isolated without changing scientific equations.

## Active path

```text
experiments/studies/*.toml → config.py + cli.py → workflows.run_sample_study
    → orchestration/sample_study.py: declared-unit input, stage coordination, sharing
    → workflows.run_resistance_fit → fitting/resistance_custom_analysis.py
    → orchestration/map_workflows.py: screening → verification → point GIFs
        → analysis/physical_maps.py + analysis/extrema.py: coordinates, eligibility, cycle features
        → current_drive_sim.py → model.py: authoritative physics/hysteresis
        → plotting/map_animations.py + plotting/replay_plots.py: real trajectories and GIFs
    → runs.py: one parent bundle with immutable child stages
    → orchestration/artifact_catalog.py: links to canonical evidence, not another database
```

No independent model is implemented in the CLI, HTML or dashboard. Historical
spectral/peak-count detectors are not interchangeable with current extrema
analysis; each research recipe declares its method.

## Physics and numerical truth

| Module | Responsibility |
|---|---|
| `model.py` | Original voltage-circuit model, parameter dataclasses and shared float32-faithful hysteresis memory. |
| `current_drive_sim.py` | Ideal-current extension, exact C=0 limit, serial/vectorized integration. Every new map/replay uses it. |

These modules know nothing about CLI/Streamlit or study-folder paths. Never
change physical state to improve a plot or weaken a convergence test.

## Numerical analysis and inference

| Module | Responsibility |
|---|---|
| `fitting/resistance_custom_analysis.py` | Major-loop and historical stateful R(T) fitting, uncertainty resampling and diagnostics. |
| `analysis/experimental_waveforms.py` | Numerical TIA waveform import; nominal drive labels remain distinct from measured current. |
| `analysis/extrema.py` | Settled complete-cycle maxima/minima, frequency, Vpp and persistence. |
| `analysis/physical_maps.py` | Exact Cth/Se/C sampling, physical screening, two-step verification, slider slices and rendering. |
| `analysis/metrics.py` | Shared single-run summaries; basic historical detector is not the current extrema-fit objective. |
| `analysis/lab_estimates.py` | Conditional conductance/electrical and thermal timing estimates from lab records. |
| `analysis/model_validation.py` | Frozen specimen comparison and capacitance sensitivity against measured waveforms. |
| `fitting/parameter_inference.py` | Earlier shared-waveform inference and inverse/reconstruction helpers; historical objectives preserved. |
| `fitting/joint_inference.py` | Budgeted joint static-resistance/dynamic-feature fits with recorded constraints. |
| `fitting/steady_inference.py` | Late-window searches, configured feature/loss definitions and physical eligibility gates. |
| `analysis/oscillation_audit.py` | Windowed persistence and controlled diagnostics; distinguishes transient peaks from sustained cycles. |
| `analysis/lab_replay.py` | Measured-current forward replay and major-branch guides with one shared parameter vector. |
| `fitting/current_domain_search.py` | Earlier exploratory domain search, not part of the new sample pipeline. |
| `legacy/sample_library.py` | Compatibility for older saved samples and historical fit-display detector; not another time-domain solver. |

Historical paths remain when needed to explain archived results. New work should
reuse the smallest relevant analysis function rather than add another script.

## Orchestration, artifacts and interfaces

| Module | Responsibility |
|---|---|
| `config.py` | TOML loading/core checks, relative-path origin and typo-safe dotted overrides. |
| `workflows.py` | Stable public API, unit translation, simulation/fitting/replay/audit orchestration; delegates map/study stages. |
| `orchestration/sample_study.py` | Strict CSV/TSV unit normalization, major-loop/workload checks, fit → grid → GIF coordination, standalone HTML. |
| `orchestration/map_workflows.py` | Screened grid/media bundles, checkpoints, reuse provenance and historical packaging compatibility. |
| `runs.py` | Relative-path manifests, status/provenance, current/legacy discovery and non-destructive publication. |
| `orchestration/artifact_catalog.py` | HTML chronology and CSV inventory; never duplicates, deletes or scientifically endorses evidence. |
| `validation.py` | Recipe/manifest/artifact checks for `neuristor validate`. |
| `cli.py` | Command parsing and output/error messages; no independent scientific equations. |
| `interfaces/dashboard.py` | Read-only archive browsing/comparison. |
| `interfaces/playground.py` | Editable lab-replay UI; simulations remain in workflows/physics. |
| `interfaces/run_manager.py` | Loopback artifact library, authenticated Delete/Restore requests; lifecycle protections stay in runs.py. |
| `__init__.py` | Package metadata and compatibility exports. |

## Plotting and media

| Module | Responsibility |
|---|---|
| `plotting/visualization.py` | Simulation, fit, lab-estimate, inference and diagnostic graphics. |
| `plotting/replay_plots.py` | Lab overlays and synchronized current/voltage/R(T) GIF writer; playback separate from integration. |
| `plotting/map_animations.py` | Point refinement, resolution checks, explicit curve breaks, point controls and embedding original GIF bytes. |
| `legacy/plots.py` | Historical voltage-sweep graphics; retained for compatibility, unused by the new pipeline. |

Pre-refactor scripts stay in `legacy_scripts/`. Papers and upstream source stay
in `references/`. Old document outputs keep their original locations and appear
in the `artifacts/` catalog. The former long README is preserved in
`docs/WORKFLOW_REFERENCE.md`; current handoff conclusions take precedence.

## Safe extension checklist

1. Put scientific calculations in pure physics/analysis functions.
2. Expose orchestration through workflows; focused modules must reuse unit
   translation and equations, not fork them.
3. Use a versioned TOML and existing CLI instead of a one-off script.
4. Write a standard RunBundle with numerical evidence and a readable report.
5. Test the lowest relevant layer, one command/bundle case and the exact recipe.
6. Physics changes need two smaller timesteps without weaker tolerances.
7. Publish new reviewed evidence, never edit completed public bundles.
8. Update the handoff/chronology and refresh `neuristor runs index`.

The new pipeline automates evidence. It does not resolve the sample/TIA mismatch
or retroactively turn earlier optimization attempts into accepted calibrations.
