# Meeting implementation: steady extrema and screened parameter maps

The voltage objective now compares steady-state amplitude and frequency extracted
from maxima/minima, without matching phase or individual voltage samples. Previous
fitting recipes remain reproducible; the new recipe explicitly selects `extrema`.
The independent static R(T) constraint is retained.

## Measurement method

1. Use the declared late window: 150–250 ns for the measured finite pulses;
   300–600 ns for the new constant-current maps.
2. Locate prominent maxima/minima with 0.5 ns smoothing. Read their voltages from
   the original samples. Reject any false raw maximum that lies below its minimum.
3. For each complete cycle between adjacent maxima, take the mean of the two
   maxima minus the intervening minimum. The median cycle value is Vpp, not half
   amplitude. Frequency is `1000 / median peak-to-peak period_ns`, in MHz.
4. Require at least two complete cycles, cycles in both halves, half-window
   amplitude ratio 0.75–1.333, and period coefficient of variation <=0.2.
   Flat signals and decaying transients cannot qualify through a nominal frequency.

These operational gates define the tested steady regime; a finite window does not
prove asymptotic stability. Measured exports are sampled at 1 ns, so timing and
frequency precision remain limited by that sampling and the number of cycles.

The dynamic loss averages, equally across the nine target records,

```
[(f_pred/f_meas - 1) / 0.025]^2
+ [log((Vpp_pred + 0.1 mV)/(Vpp_meas + 0.1 mV)) / log(1.075)]^2.
```

A missing acceptable persistent oscillator adds 10,000. The map/fitting frequency
ceiling is 110 MHz, as requested in the meeting. It is an operating selection,
not an established universal physical cutoff for VO2.

## Parameter maps

The [adjustable slice viewer](../runs/published/20261002_152003_focused-dense-physically-screened-steady-state-e_c1772e/parameter_maps.html)
adds three independent sliders, each selecting any sampled value for one locked
parameter. Both frequency and Vpp heatmaps update together. Each colorbar uses
only retained values in that slice, retaining the original MHz/mV units;
screened-out points stay blank and empty slices are explicitly labeled.
Colors therefore cannot be compared directly across slices: compare their
colorbar values instead. The original 3D map remains available above the sliders.
Numerical CSVs are byte-identical to the preceding reviewed map; only rendering
changed. The earlier common-color-scale HTML below remains preserved as history.
All 69 sampled slices were checked for exact retained-point counts and local color
limits. Grid indexing tolerates CSV floating-point roundoff without interpolating
points. The initial slider rendering `2bffd0` is superseded by `c1772e`, which fixes
that indexing issue. Automated browser preview was unavailable for local file URLs;
the controls and slice data were verified from the generated artifact and tests.

Both maps use **500 µA constant ideal current and T0=Tinit=298 K**, the fitted
sample's static resistance preset and its fixed Yuanhang gamma. They are
conditional model explorations. The separate laboratory fitting uses measured
input waveforms and allows ambient temperature within its declared bounds.

The original `runs/local/3d_domain_exploration.npy` has only binary oscillation labels.
It contains neither frequency nor temperature trajectories and cannot support
the requested filters. No denser version of that same 3D grid was found.

- [Broad 9,261-point map](../runs/published/20261002_135437_dense-physically-screened-steady-state-extrema-p_9ac3ad/report.md):
  retains all original 15×15×15 grid coordinates and adds six useful intermediate
  values per axis. It has 66 retained oscillators after verification.
- [Focused interactive 3D and heatmaps](../runs/published/20261002_145342_focused-dense-physically-screened-steady-state-e_b9e298/parameter_maps.html):
  23×23×23 = **12,167** simulated points, **1,258** retained. Maximum retained
  frequency is **109.890 MHz**, maximum trajectory temperature **372.188 K**.
  [Static heatmaps](../runs/published/20261002_145342_focused-dense-physically-screened-steady-state-e_b9e298/heatmaps.png)
  use common color limits across panels.

Screening additionally requires the **entire** temperature trajectory within
the configured resistance implementation's 284.8545–375.1625 K domain,
nonnegative sampled effective-temperature/proximity slope, and frequency/amplitude
changes <=5% under both refinements 0.05→0.025→0.0125 ns. This retains operationally
screened trajectories, not independently measured C/Cth/Se values. The measured
static R(T) data themselves span 289.8545–370.1625 K; the implementation has 5 K
padding, so passing its domain check is not identical to staying within measured
temperatures. In particular the 372.188 K maximum involves slight extrapolation.

The 66 MHz/2494 K diagnostic fails the temperature check despite satisfying
the 110 MHz ceiling. Frequency alone would have retained it incorrectly.

The focused grid explores Cth=0.02–0.5 pJ/K, Se=0.0025–0.012 mW/K,
and C=0.8–20 pF, with extra exact slice/reference values including C=0.39 pF.
The fixed slices are Cth=0.047873236 pJ/K, Se=0.0036751265 mW/K, and C=16.4 pF.
The resulting frequency/amplitude pairs make six heatmaps. The large electrical
capacitance slice is diagnostic and does not replace the experimental 0.39 pF
timing bound. The retained bands are narrow; at fixed Se, increasing C generally
requires increased Cth along the displayed oscillatory band.

Blank cells preserve omitted/nonoscillating candidates in `grid.csv`. Coarse
rejected points were not all rerun at finer steps, so blank cells do not prove
the absence of oscillations. The finer checks cover every displayed oscillator.
No interpolation is used to manufacture more 3D points. The focused HTML reuses
the completed numerical bundle with byte-identical CSVs and explicit provenance.

## Laboratory fitting pilot

The bounded pilot uses nine original source-record labels 300–700, corresponding
to measured plateau currents approximately **228–533 µA**. These labels remain
record identifiers rather than imposed physical current values. It searches one
shared parameter vector with the static R(T) error <=0.05, gamma=0.60–0.98,
sampled proximity monotonicity and temperature-domain checks. Search and fine
verification use measured current prehistory. The complete 22-record comparison
is repeated at 0.025, 0.0125 and 0.00625 ns.

The [completed 79-candidate pilot](../runs/published/20261002_145243_meeting-extrema-only-physically-screened-fitting_32d91d/report.md)
found **zero acceptable target-frequency matches**.
Its best available vector largely produced tiny high-frequency
numerical features rather than the measured oscillations; these fail the 110 MHz
gate. The old lower-loss eef073 reference violates the proximity check and is
not eligible for adoption. The pilot therefore does not resolve the model/experiment
discrepancy, nor establish that a satisfactory fit cannot exist. The raw-maxima
check was strengthened during development to reject smoothing-created extrema
whose raw maxima lie below their raw minima. The final run includes that check;
earlier development pilots are not the reviewed result.

The new measurements recover experimental frequency approximately 41.7–58.8 MHz
and Vpp 13.8–101.8 mV across these nine records. At the finest timestep, the
best available searched vector has no accepted predicted frequencies and no
near-target traces. The 100% frequency error in `summary.csv` is the prescribed
missing-frequency penalty, not a measured error on a valid frequency estimate.

Verification: **83 tests passed**, including scalar/batched trajectory agreement,
switching-feature convergence at two smaller timesteps, false-cycle rejection,
extrema phase/DC/startup invariance and numerical-reuse provenance. Repository
validation passed. Reviewed bundles are copied into `runs/published/`; this task did
not make a Git commit or push to GitHub.

## Reproduction

```bash
neuristor analyze physical-maps --config experiments/current/physical_parameter_maps.toml
neuristor analyze physical-maps --config experiments/current/physical_parameter_maps_focused.toml
neuristor analyze fit-steady --config experiments/current/specimen_steady_extrema.toml
```

To regenerate graphics only, add `--reuse-numerics COMPLETED_MAP_DIRECTORY` to
the matching map command. Source bundles remain intact. No physics equation or
resistance law was changed; parameter batching calls the existing electrical,
thermal and float32 hysteresis updates with independent memory per grid column.
