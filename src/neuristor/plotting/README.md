# Plotting: show calculations faithfully

- `visualization.py`: simulation, fitting, and laboratory-diagnostic figures.
- `replay_plots.py`: measured/simulated overlays and synchronized current/voltage/R(T) animations.
- `map_animations.py`: finer-step point replay, resolution checks, animation controls, and HTML embedding.
- `boundary_maps.py`: sampled coarse/fine classifications, verified transition brackets and an explicitly open scale reconstruction.
- `loss_comparison.py`: common frequency/amplitude plots and three representative voltage overlays from verified saved samples.
- `optimizer_diagnostics.py`: saved-coordinate score profiles with timestep differences labeled separately from scientific losses.

Integration timestep controls numerical resolution. Frame count controls playback.
`replay_plots.write_scope_gif` can keep measured channels on their acquisition
grid while drawing simulated voltage and R(T) from every native solver endpoint.
Do not interpolate nonexistent R(T) branches to make an animation look smoother.
The solver and its physical state belong outside this folder.

`frequency_sensitivity.py` plots saved per-input frequency profiles across parameter perturbations and solver timesteps.

`trend_validation.py` plots only verified oscillation features and keeps fixed versus scaled timestep clock tests distinct.
