# Orchestration: one coherent output folder per job

- `sample_study.py`: measured R(T) → static fit → screened grid → verified point GIFs → standalone HTML.
- `map_workflows.py`: reusable map and animation stages, checkpoints, and source/media provenance.
- `boundary_workflows.py`: expanding coarse exploration followed by fine transition verification, with an open-scale 3D reconstruction.
- `map_viewers.py`: display-only revisions of historical maps, preserving numerical data and reusing media.
- `artifact_catalog.py`: rebuild the searchable HTML/CSV library from existing files, without duplicating them.

Call these through `neuristor` or the stable `neuristor.workflows` API.
Recipes hold units and assumptions; physics modules hold equations. A completed
bundle is immutable. A new exporter or analysis creates a new bundle.
