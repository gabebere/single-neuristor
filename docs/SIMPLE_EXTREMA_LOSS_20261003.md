# Equal-weight extrema trial — 3 October 2026

## Method

New opt-in recipe: `experiments/current/specimen_steady_relative.toml`.
Historical recipes and completed bundles are unchanged.

For each oscillating experimental record, measure frequency from the median
adjacent-maximum period and Vpp from the median complete peak–trough cycle.
Use only 150–250 ns, retaining the measured input prehistory during simulation.
The scientific objective is

\[
L = \frac{1}{N}\sum_i\left[
(f_i^{sim}/f_i^{exp}-1)^2 + (A_i^{sim}/A_i^{exp}-1)^2\right].
\]

A is peak-to-peak voltage, not half-amplitude. Equal percentage frequency and
amplitude errors contribute equally. There is no phase/DC/waveform point error,
logarithm, amplitude offset, or tolerance scaling. Static R(T) is only a hard
RMSE ceiling of 0.05 in log10 R, not another weighted objective term.

Eligibility is separate: sustained complete cycles on all nine target records,
frequency <=110 MHz, temperatures within the configured implementation domain,
and nonnegative sampled proximity slope throughout each trajectory. The cutoff
and domain are operational screens, not proof of physical validity. Initial
sampling and final verification use the original Yuanhang proximity kernel.

The numerical optimizer needs finite scores even where frequency is undefined.
Rejected candidates use a feasibility sentinel (1e6 + 1000 times the number of
failed targets, with a bounded diagnostic tie-breaker). It is NOT the scientific
loss. Missing frequencies/errors remain null. Rejected candidates cannot enter
accepted elites. If no eligible candidate is found, the best physically screened
diagnostic attempt is verified and animated under an explicit diagnostic label.
This means an output GIF need not represent an accepted fit.

Same bounds, source labels, measured currents, window and two-start 180-second
pilot budget as the previous extrema trial; seed is unchanged. A time budget
does not imply an identical number of evaluations. Verification uses 0.025,
0.0125 and 0.00625 ns on all 22 records. No physical equation changed.

Reproduce:

```bash
neuristor analyze fit-steady --config experiments/current/specimen_steady_relative.toml
```

## Desktop archive

**Location update, 16 October 2026:** use the
[consolidated project guide](../archive/desktop_consolidation_20261016/index.html).
The following describes the original delivery; duplicate Desktop copies are now
in recoverable Trash, with unique notes preserved inside this repository.

`/Users/gabrielberezovsky/Desktop/VO2 Research Archive/START_HERE.md` categorizes
18 original Desktop items into fitting trials, parameter diagnostics and historical
references. Files were moved without deleting duplicates or altering completed
bundles. Linked old Background/Monotone results were additionally copied into
real Frozen Results folders. Old launcher text is historical, not a running-job
status. This local organization does not imply a GitHub push.

Recommended comparison: use one common extrema measurement definition and window,
then inspect physical checks and finest-step errors before the GIFs. Do not rank
different historical objectives by their numerical scores.

## Results

Search bundle: `public_jobs/20261003_183555_simple-relative-squared-extrema-fitting-pilot_96b944`.
85 unique candidates, 181.55 seconds searching, 228.17 seconds including all nine
verification batches (three parameter vectors × three timesteps × 22 records).
No candidate was eligible on all nine targets. `diagnostic_1` is the best
physically screened failed attempt, not an optimizer winner accepted as a fit.

At 0.00625 ns it sustains 8/9 target traces; 5/9 also meet the <=110 MHz
frequency screen. Static RMSE is 0.036628 and minimum sampled proximity slope
is +0.24467. All 22 simulated histories stay in the configured temperature domain.
No target meets the combined 2% frequency / 5% amplitude accuracy goal.

| Measured current (µA, rounded) | Measured / simulated f (MHz) | Measured / simulated Vpp (mV) |
|---|---|---|
| 267 | 43.48 / 62.50 | 101.83 / 379.90 |
| 381 | 51.28 / 90.91 | 45.76 / 401.98 |
| 533 | 58.82 / 125.00 | 13.77 / 355.86 |

The lowest target (~228 µA) has no simulated complete cycles. The three highest
target frequencies exceed the selected ceiling. Average amplitude relative error
over all nine targets is 973.4%; frequency error is 78.9% over only eight defined
frequencies. That partial frequency statistic is NOT a nine-trace fit score.
The bundle's `mean_dynamic_loss` on a rejected candidate averages only defined
diagnostic cycle losses; it is NOT the proposed accepted all-target objective.
For missing frequencies, individual loss/error cells are blank, not fabricated.
Classification changes from 6/9 frequency-valid at 0.025 ns to 5/9 at both smaller
steps. Extrema are located on the experimental sampling grid; reported feature
quantization and the cutoff should not be mistaken for precise physical bounds.

Parameters: C=3.744741 pF, Se=0.004846582 mW/K, Cth=0.015325537 pJ/K,
T0=317.456572 K, gamma=0.755334; six major-loop values remain at the static fit.
The finest-step all-current GIF replay uses
`experiments/current/specimen_relative_diagnostic_replay.toml`. GIFs progressively
show imposed measured current, measured/simulated voltage and simulated R(T)
over static major-branch guides. R and T are simulated, not experimental channels.

Reviewed GIF bundle:
`public_jobs/20261003_183956_simple-loss-diagnostic-1-all-current-gif-replay-_4e79ac`.
22 GIFs / 22 PNGs, 48 readable frames each; replay/search voltage difference
<=0.0093 mV. Full replay temperature range 317.46–364.07 K. Its historical
persistence summary is separate from the new extrema gate; do not substitute
its recovery count for the 5/9 screened target result above.
Desktop delivery: `VO2 Research Archive/00_Current_trial/20261003_relative_squared_96b944`.

The simpler objective is implemented, but this bounded search is unsuccessful.
The old reference has a smaller diagnostic loss but fails proximity; the static
reference does not oscillate. This is precisely why eligibility is separate.
Do not claim a global optimum or attribute differences solely to loss: the
time-limited search visits a different sequence of candidates. A future search
can use this candidate as a seed, but its large amplitudes remain unexplained.
