# M2 1600nm approximate R-T exploratory 3D map 20261021

374 measured R(T) rows; log10 RMSE 0.080652. 0 / 216 grid points retained; 0 GIFs. Constant 500 µA, frequency ceiling 110 MHz.

## What was fitted and what was assumed

The measured major loop was fitted for R0, Ea/kB, metallic resistance, Tc,
width and beta in log10 resistance. Gamma is not inferred from a major loop.
Its simulation value is 0.95627; proximity function is
yuanhang. Electrical/thermal capacitance,
environmental conductance, ambient temperature and initial conditions are recipe
assumptions or sampled axes, not estimates extracted from this R(T) fit.

The selected resistance domain is measured:
289.98–370.18 K.
The grid uses late complete-cycle voltage maxima/minima for frequency and Vpp,
and separately screens persistence, temperature, sampled proximity slope and the
configured frequency ceiling. Survivors must pass two finer timestep checks.
Coarse rejected points are not all refined: blank cells are not a proof of no cycles.
Heatmap axes and colors stay fixed across slider positions, using retained-point metric ranges.

GIFs use finer integration and refinement checks. 0
of 0 replays passed refinement/eligibility; 0
were visually resolved. Unresolved R(T) jumps are broken instead of inventing
smooth paths. Replay failures are warned in the viewer; they do not silently
change membership of the already completed numerical grid.

This is NOT a validation against experimental voltage/current traces or a fitted
TIA circuit. The cutoff is an operational choice, not a universal material law.
An empty screened domain is a valid outcome. No automatic laboratory overlay is
fabricated for an unmatched constant-current experiment.

## Review in order

1. [Resistance fit](stages/20261021_061157_m2-1600nm-approximate-r-t-exploratory-3d-map-202_360212/report.md), its graph and parameter CSV.
2. [Grid methodology and exclusions](stages/20261021_061157_m2-1600nm-approximate-r-t-exploratory-3d-map-202_08d115/report.md), grid.csv and verification.csv.
3. [Clickable map and all GIFs](parameter_maps.html), animation_checks.csv and gif_catalog.json.

Open parameter_maps.html; share that one file. All 0 GIFs
and the fitted R(T) image are embedded. No ZIP is generated. The standalone HTML
is 5.19 MB; large grids may exceed email limits.
The standalone file includes the exact recipe, resolved resistance and input hash,
but the complete raw/numerical/source archive remains in the study folder.
Original measurements, hashes, resolved recipes and frozen source are included.
