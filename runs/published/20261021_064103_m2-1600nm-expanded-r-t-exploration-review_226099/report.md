# M2 1600 nm: expanded R–T exploration

Additional search requested 21 October 2026 with a 15-minute budget. Two new
coarse grids add 1843 sampled entries to the earlier 773 (2,616 total,
including overlaps). Fixed ideal current 500 µA, ambient and initial temperature
298 K, zero initial voltage, insulating branch and gamma=0.956269682. No physics
or static-fit parameter was changed.

## Results

| study       |   points |   sustained |   temperature_valid |   retained |   sustained_temperature_excluded |   dt_ns |   duration_us |
|:------------|---------:|------------:|--------------------:|-----------:|---------------------------------:|--------:|--------------:|
| Expanded    |      512 |          35 |                 337 |          0 |                               35 | 0.0125  |           2.4 |
| Dense onset |     1331 |         373 |                 894 |          0 |                              373 | 0.00625 |           1.2 |

0 new verified retained points. Diagnostic plots include
excluded computed oscillations, marked as such; they are not validated device
predictions. The full trajectory must remain inside the measured R–T interval
289.98–370.1795 K. This finite search does not prove absence of an in-domain
oscillator. Only coarse-eligible points receive the workflow's two-step check;
other rejected cells are not individually certified at the finest timestep.

An additional startup-only check replays 12 nearby sustained settings at
0.00078125 and 0.000390625 ns for the first 150 ns. Selection uses the lowest
coarse Tmax among the earlier onset grid and an interim dense-grid checkpoint;
the exact selection is saved. 24/24 integrations
already exceed the measured range within that startup window; the lowest
observed maximum is 372.670959 K. This verifies
those early temperature exclusions without asserting steady-state feature or
waveform convergence. It is not a full fine replay of all 1,843 points.

## Search ranges and numerical settings

The 8×8×8 broad grid expands Cth to 0.001–50 pJ/K, Se to
0.0001–3 mW/K, and C to 0.0001–500 pF. Base dt=0.0125 ns, duration
2.4 µs, late window 1.2–2.4 µs, verification steps 0.00625 and 0.003125 ns.
The 11×11×11 onset grid samples Cth=0.5–4 pJ/K, Se=0.10–0.22 mW/K,
C=0.1–1.2 pF. Base dt=0.00625 ns, duration 1.2 µs, late window
0.6–1.2 µs, verification steps 0.003125 and 0.0015625 ns.
All retain the operational 110 MHz screen, existing persistence detector,
sampled proximity check and 5% feature-convergence threshold. Search bounds
and calibration exclusions are not physical oscillation boundaries.

## Inherited static-fit limitation

The same 374-row R–T fit has log10 RMSE 0.08065244 (typical multiplicative
factor 1.20407); it failed the normal 0.05 gate. The first study explicitly
allowed 0.10 for diagnostic maps. This expansion reuses its exact fit without
claiming improved static agreement. Gamma, electrical/thermal capacitance and
thermal conductance remain assumptions or axes, not measured estimates.
Input SHA-256: fdda8d2293404566fb16154223da131b1a44f15975469f1cdfd8622948112e03.

## Reproduction and checks

Run `neuristor analyze physical-maps --config` on the two portable TOML recipes
in `recipes/`; the static preset is preserved in this bundle. The complete new
map bundles, all flags, two-step verification tables (where applicable), frozen
source, physics parameters, raw measurement and inherited fit are included.
Animations are a separate optional stage, not an unbounded cost in this search.
The exact saved-evidence rendering script is preserved as `build_review.py`.
The standalone HTML embeds Plotly and its R–T image. Original screened maps
remain unchanged and are linked beside each diagnostic grid.
Browser interaction was not verified; numerical artifacts, links and the static
PNG were checked. Software validation logs are in `checks/`. The previous
773-entry study remains immutable at 20261021_062155_m2-1600nm-budgeted-r-t-exploration-review_c7387b. No new calibration is adopted.
