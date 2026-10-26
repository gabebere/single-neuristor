# M2 1600 nm: budgeted R–T exploration

Completed 21 October 2026. Same ideal-current physics, 500 µA, 298 K, initial V=0,
insulating branch and assumed gamma=0.956269682 as the earlier map. The supplied
file contains 374 acquisition-ordered measurements in K and ohm, 289.98–370.1795 K.
Its exact raw bytes are preserved in every study and their hash is in metrics.json.

## Result

773 sampled grid entries in four deliberately coarse grids;
0 verified retained points. These are sampled entries, not a claim of
that many distinct physical settings. The map viewer shows computed sustained
oscillators with their exclusions visible; original screened viewers remain
unchanged. All sustained results in the first three grids exceeded the measured
temperature domain. This is not proof that no in-domain oscillator exists.
Coarse rejected points do not automatically receive the two-step recheck.

| study         |   points |   sustained |   temperature_valid |   retained |   sustained_temperature_excluded |    dt_ns |   duration_us |
|:--------------|---------:|------------:|--------------------:|-----------:|---------------------------------:|---------:|--------------:|
| Broad         |      216 |          54 |                  32 |          0 |                               54 | 0.05     |           0.6 |
| Lower heating |      216 |          10 |                 174 |          0 |                               10 | 0.05     |           2   |
| Transition    |      216 |          68 |                 108 |          0 |                               68 | 0.00625  |           1.2 |
| Onset         |      125 |          50 |                  60 |          0 |                               50 | 0.003125 |           0.6 |

## Static model limitation

The six-parameter inherited law gives log10 RMSE 0.08065244, R²
0.99359124, typical multiplicative discrepancy
1.20407 and worst multiplicative discrepancy
1.76373. Structured transition
residuals remain. The strict initial run stopped at its 0.05 gate; exploratory
recipes explicitly permit 0.10. This is a recorded exception, not an improved fit.
No equations, physics tolerances or convergence criteria were modified.
Gamma, C, Cth and Se are not identified from the measured major loop.

## Checks and interpretation

All displayed data are saved numerical outputs. Frequency uses late voltage peak
periods; Vpp uses complete-cycle extrema and the existing persistence detector.
Original screens retain the measured temperature domain, nonnegative sampled
proximity slope, operational 110 MHz ceiling and 5% feature changes at two smaller
steps. A separate 36-point pilot at 0.0015625 ns (32× finer than the first grid)
still found no eligible oscillator; this does not finely verify every excluded cell.
Finite-window stable/no-cycle cells and search edges are not physical boundaries.
The HTML includes all diagnostic maps and the fit image with no network dependencies.
Browser interaction was not verified because the available browser blocks local
file URLs. Numerical tables, generated figures, HTML structure and artifact paths
were checked independently. Software checks: 194 tests passed; repository validation
passed. No adopted specimen calibration or experimental waveform agreement.

## Reproduce

Run `neuristor study run --config` with each of the four portable recipes in
`recipes/`; their data path points to the preserved raw input in this bundle.
The source repository also contains recipes under `experiments/studies/` and
the identical raw input under `data/experimental/`. The strict initial failure
is preserved separately in `strict_static_check/`. The finer 36-point pilot's
complete physics, axes, timestep and feature window are in `fine_pilot_config.json`.
Each `studies/` directory is an immutable completed bundle with frozen source,
resolved settings, raw data, fits, grids, and manifests. This review's exact
assembly script is `build_review.py`; it renders saved data without changing physics.
