# All-input loss-function GIFs — 21 October 2026

The two retained objectives each have 22 animations: **44 GIFs total**. Later
optimizer follow-up runs are excluded, as requested. Open the
[all-current viewer](../runs/published/20261021_054434_loss-function-comparison-all-44-stacked-gifs_15921f/START_HERE.html).
The folder also contains one full-trace PNG per GIF and compressed numerical samples.

Each animation has measured input current at the top, measured output and simulated
device voltage in the middle, and the simulated resistance–temperature history at
the bottom. Heating/cooling major branches are model guides. Titles and filenames
use measured 50–250 ns mean current, never the acquisition filename's mV number.

The original parameter vectors and 0.00625 ns verification timestep are retained.
Optimization is not repeated. A fixed-parameter replay supplies the previously
unsaved native R/T histories. Every complete acquisition-grid voltage trace must
match its saved parent within 0.000001 mV before animation rendering begins.
The original pre-pulse measured current history remains part of the integration;
the animation shows −30 through 250 ns. Both measured channels retain their
acquisition grid; the simulated path retains every native integration endpoint.

R–T jumps exceeding 0.25 K or 0.08 in ln R per native step are broken, rather than
connected by artificial chords. These are visual-resolution criteria, not a new
physical acceptance gate. The index records jump counts and whether each complete
trajectory leaves the measured R(T) temperature domain. R and T are simulated;
no apparent resistance is inferred from the measured channels. Axes fit each
record, so compare their labels when assessing amplitudes across GIFs.

The 96 animation frames control playback only. Simulated and measured voltages
are neither phase aligned nor rescaled. All records, including controls and poor
matches, are included. These remain diagnostic candidates; the
[loss-comparison limitations](LOSS_COMPARISON_20261020.md) still apply.

Reproduce locally:

```bash
neuristor analyze loss-scopes \
  --config experiments/current/specimen_frequency_only_20261020.toml \
  --frequency-run runs/published/20261020_155947_controlled-frequency-only-specimen-fit_543408 \
  --combined-run runs/published/20261020_165107_controlled-frequency-amplitude-specimen-fit_5ad999 \
  --frames 96
```

The original frequency-only recipe supplies fixed settings and resolves input
paths. The exporter rejects differing fixed settings, incomplete parents, an
ambiguous selected vector, swapped objectives, or changed replay voltages.
The new bundle records parent hashes and parameter vectors without altering
the completed parents.

Verification: all 44 GIFs contain 96 frames at 850 × 1000 pixels, with a companion
full-trace PNG and native-sample NPZ. Maximum voltage difference from the saved
parent verification is 5.69 × 10⁻¹⁴ mV (floating-point serialization precision).
The complete test suite passes: 194 tests. Repository validation and unchanged
parent-file hash checks also pass. The 19-slide presentation is unchanged.
