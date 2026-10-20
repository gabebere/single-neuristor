# Plotting: show calculations faithfully

- `visualization.py`: simulation, fitting, and laboratory-diagnostic figures.
- `replay_plots.py`: measured/simulated overlays and synchronized current/voltage/R(T) animations.
- `map_animations.py`: finer-step point replay, resolution checks, animation controls, and HTML embedding.
- `boundary_maps.py`: sampled coarse/fine classifications, verified transition brackets and an explicitly open scale reconstruction.

Integration timestep controls numerical resolution. Frame count controls playback.
Do not interpolate nonexistent R(T) branches to make an animation look smoother.
The solver and its physical state belong outside this folder.
