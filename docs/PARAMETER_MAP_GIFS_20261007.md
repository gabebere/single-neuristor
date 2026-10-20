# Portable point-to-animation map — 7 October 2026

## User-facing workflow

The screened map is extended into a portable ZIP: extract it, then open
`parameter_maps.html` alongside `GIFs/`. Right-click a hovered retained point
and choose **View simulation GIF**. Alternatively left-click a point and use
the View button; this avoids relying on right-click availability on a trackpad.
A dialog plays the GIF and offers a direct download. Slider changes clear stale
selection, and heatmap cells carry stable point IDs for their new slice.
Frequency/amplitude colors still normalize to the selected visible slice.

Sending the HTML alone no longer sends the animation collection. The ZIP
requires no Python, local web server, internet, or connections to the sender.
The original self-contained numerical map remains unchanged in its public bundle.

## Numerical and physical meaning

The source map explores Cth, Se and C at **500 µA ideal constant input**,
T0=Tinit=298 K, with the specimen's fixed fitted static resistance parameters.
No measured voltage matches those operating conditions; no experimental trace
is overlaid. Three panels show imposed current, simulated voltage and simulated
R(T) over major-branch guides. Genuine minor loops need not lie on those guides.
These are conditional model simulations, not newly fitted specimen values.

Animations use the authoritative batched current solver with independent
hysteresis memories. Initial replay dt=0.00625 ns, half the old finest map step.
Frequency/Vpp are compared to the previous step and physical eligibility is
rechecked over the full history. dt halves at most twice when feature changes
exceed 5%, or adjacent steady samples change by more than 0.25 K or 0.08 in ln R.
The latter criteria measure visual resolution, not material laws. Repeated
failure or a genuine discontinuity is labeled explicitly; adjacent large jumps
receive a plot break rather than an invented straight chord. Every integrated
R(T) endpoint is preserved. There is no spline, smoothing, interpolation or
projection onto the major branches. Float32 upstream hysteresis is unchanged.

The GIF shows at least 80 ns or approximately three periods within 300–600 ns,
whichever is longer, capped at the window end. It does not display startup.
32 animation frames control playback, not numerical resolution; all resolved
samples up to each frame time are plotted. A finer endpoint motion would require
more frames but does not change the R(T) path already drawn within each frame.

## Provenance / commands

Recipe: `experiments/current/physical_parameter_map_gifs.toml`.

```bash
# Evenly spaced retained points: runtime and resolution benchmark only
neuristor analyze map-animations --config experiments/current/physical_parameter_map_gifs.toml --limit 3
# Full retained domain
neuristor analyze map-animations --config experiments/current/physical_parameter_map_gifs.toml
```

Each creates a new bundle. The numerical grid and verification CSVs are copied
byte-for-byte from the completed source. Source configuration, physics parameters,
hashes, original sampled axes, frozen Python source, new refinement checks,
point-to-GIF catalog, PNGs and media are preserved. If a new replay fails, its
warning is shown without silently changing old map membership. This distinction
matters when examining points close to the 110 MHz screening ceiling.

The 3-point benchmark bundle is
`runs/20261007_061058_portable-screened-parameter-map-with-point-gifs-_f34881`.
Its ZIP is approximately 3.5 MB. All three new replays pass eligibility and
feature convergence and meet the visual resolution thresholds. Two require
dt=0.003125 ns; the third passes at 0.00625 ns. Wall time approximately 42 s.
That small-batch rate cannot be extrapolated linearly to a large batched solver.

The full job started at 09:12 Israel time on 7 October 2026:
`runs/20261007_061243_portable-screened-parameter-map-with-point-gifs_81ae34`.
Read `status.json` rather than assuming completion; two rendering workers are
used and the integrator remains a local process. No ChatGPT calls are made by
this numerical job. Failure/completion are captured in the run manifest.

Tests cover endpoint preservation, gap insertion, timestep halving, stable
slice IDs, stale-selection clearing and the portable ZIP contract. The popup
event logic is exercised with a mocked DOM and real JavaScript runtime; this
is not a claim of end-to-end browser or right-click hit-testing verification.
Readiness/results will be recorded after the full generation completes.

Delivery checkpoint: 204/1,258 GIFs generated after approximately 6.3 minutes;
current throughput projects about 39 minutes total (estimate, not a deadline).
93 tests passed and repository validation passed. Four historical arctanh
warnings remain. Completed benchmark published locally as
`public_jobs/20261007_061058_portable-screened-parameter-map-with-point-gifs-_f34881`
and copied to Desktop `VO2 Research Archive/04_Parameter_maps/`.
No external upload or GitHub commit/push was performed. Full job remains local.

## Completed generation and viewer correction

Full generation `81ae34` completed: all 1,258 GIFs, approximately 562 MB ZIP.
1,256 pass the new eligibility/convergence checks; 1,174 meet visual resolution
thresholds. The 84 remaining trajectories draw unresolved jumps with breaks.
GIFs are available for every retained source-map point, including diagnostic
warnings. All GIFs retain the original scientific limitations.

The user then reported “No animation available” for every selected point.
Inspection found that Plotly serialized initial customdata into binary typed-array
objects, whereas the event handler expected ordinary rows/scalars. The corrected
renderer writes ordinary JSON lists explicitly, and the event handler handles
typed-array rows as well as absent event customdata, using trace/pointNumber
fallbacks (including heatmap row/column indices). Original mocked-DOM tests used
only ordinary event arrays and did not expose this representation mismatch.

Corrected complete viewer bundle:
`runs/20261007_074702_portable-screened-parameter-map-with-point-gifs_eaf6ab`.
The identical corrected bundle is also published locally under `public_jobs/`.
No simulation was rerun. All 1,258 GIFs and PNGs are byte-preserved, with SHA-256
media provenance. The old completed bundle remains unchanged. Original simulation
source is preserved separately from the corrected viewer source.
All 1,258 3D point IDs and all 69 slider slices (3,774 retained slice cells)
resolve to existing catalog entries and GIF files. ZIP integrity passed.
Runtime event tests include typed rows, missing metadata and heatmap fallback.
These are serialization/runtime tests, not end-to-end browser hit-testing.

For an already extracted original ZIP, replace only `parameter_maps.html` with
the corrected file; keep the existing `GIFs/` folder alongside it. This avoids
another 562 MB download. New recipients should receive the corrected full ZIP.

Reproduce the viewer-only rebuild:

```bash
neuristor analyze map-animations --config experiments/current/physical_parameter_map_gifs.toml \
  --reuse-media runs/20261007_061243_portable-screened-parameter-map-with-point-gifs_81ae34
```

Reuse requires a completed bundle, identical source grid, identical physics and
identical animation recipe. It creates a new bundle rather than modifying a
completed one. Rebuild runtime is not animation-generation runtime.
