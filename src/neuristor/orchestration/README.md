# Orchestration: one coherent output folder per job

- `sample_study.py`: measured R(T) → static fit → screened grid → verified point GIFs → standalone HTML.
- `map_workflows.py`: reusable map and animation stages, checkpoints, and source/media provenance.
- `boundary_workflows.py`: expanding coarse exploration followed by fine transition verification, with an open-scale 3D reconstruction.
- `map_viewers.py`: display-only revisions of historical maps, preserving numerical data and reusing media.
- `artifact_catalog.py`: rebuild the searchable HTML/CSV library from existing files, without duplicating them.
- `loss_comparison.py`: compare two completed matched steady fits using saved voltage samples and measured-current labels.
- `loss_scopes.py`: replay both retained loss candidates, verify saved-voltage agreement, and render every input as stacked current/voltage/R(T) GIFs.
- `optimizer_audit.py`: coordinate perturbations, finite-difference checks, gradient trials and all-current timestep verification in a new bundle.

- `optimizer_review.py`: join completed optimizer experiments with common metrics, frequency profiles and provenance.

Call these through `neuristor` or the stable `neuristor.workflows` API.
Recipes hold units and assumptions; physics modules hold equations. A completed
bundle is immutable. A new exporter or analysis creates a new bundle.

`trend_validation.py` writes planned cases, new integration checkpoints, clock comparisons and controlled parameter plots for `analyze validate-trends`.
