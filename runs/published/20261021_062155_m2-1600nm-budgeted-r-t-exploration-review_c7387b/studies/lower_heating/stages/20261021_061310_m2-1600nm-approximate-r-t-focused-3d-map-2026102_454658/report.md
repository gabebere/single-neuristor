# Portable map animations

Original map: 20261021_061258_m2-1600nm-approximate-r-t-focused-3d-map-2026102_bdc07e. 0/0 retained points have animations.
Complete requested generation.
Right-click a hovered point and choose View simulation GIF, or select it with a
left click and use the View button. Heatmap point IDs follow the active slider.
Keep this folder viewer parameter_maps.html beside GIFs/. No server or internet
is needed. This stage HTML alone does NOT include the GIF bytes.
No ZIP is generated. The parent sample study embeds the GIFs in its standalone HTML.

Input is constant 500 µA; ambient/initial temperature 298 K.
No matched experiment is available for these conditions, so no laboratory voltage
is overlaid. Three panels show input current, simulated voltage and simulated R(T).
GIFs reveal a few cycles from the steady-state window, not the startup transient.
All displayed R(T) samples come from the supported solver: no interpolation or
projection onto major branches. Simulation dt starts at 0.00625 ns,
half the old finest map step, and halves up to 2
times if frequency/Vpp change exceeds 5%
or adjacent steady samples change by >0.25 K or >0.08 in ln(R). These last two
are visual resolution criteria, not physical laws. Residual jumps are broken in
the plot instead of connected by invented straight chords. Genuine minor loops
need not coincide with either major branch. Checks are in animation_checks.csv.
Old grid and verification files are byte-preserved; new replay failures remain
diagnostic warnings and do not silently alter the historical map membership.
Frames=20 controls playback only, not integration resolution.
All source code and resolved original physics are frozen in this bundle.

Elapsed 1.2 seconds; projected full coverage 0.00 hours
(small batches can substantially overestimate simulation cost because of batching).
