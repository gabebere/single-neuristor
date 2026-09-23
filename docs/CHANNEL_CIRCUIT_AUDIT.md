# Channel and circuit audit for the VO₂ current sweep

Reviewed 18 September 2026. This audits the supplied workbooks, converted exports,
manuscript, existing bundles, and current code, then replays selected measured
currents with one controlled conductance change. It adopts no new specimen
calibration and does not modify raw data or completed public bundles.

## What the files establish

The three Figure 6 workbooks have the same seven primary columns; the 500 mV
workbook adds `Output Power [μW]`. Their cells contain values, not spreadsheet
formulas or instrument metadata. Direct numerical checks give:

| Field | Observed conversion or meaning | Physical interpretation still needed |
|---|---|---|
| `Time(s)` and first `Time [ns]` | `t_ns = 10^9 × t_s`, with 1 ns spacing | The shared sample index does not establish channel skew. |
| Second `Time [ns]` | First time column +150 ns in every row | Presentation shift, not a physical delay. |
| `CH1V` → `Output Voltage [mV]` | Approximately `1000 × CH1V`; maximum rounding residual 0.00334 mV | Exact probe nodes, sign convention, loading, bandwidth and transfer function are absent. |
| `CH2V` → `Input Current [μA]` | Approximately `794.33 × CH2V`; maximum rounding residual 0.00335 μA across the three books | The calibration circuit and whether this is feedback-element current are absent. `1/(794.33 μA/V) ≈ 1.259 kΩ` is only an equivalent conversion factor, not an identified shunt. |
| `Output Power [μW]` | Numerically `V_out[mV] × I_in[μA] / 1000` | It is VO₂ Joule power only if the voltage is across VO₂ and the current traverses its resistive path, after allowing for displacement current. |

For the 100, 500 and 1000 mV labels, the corresponding converted CSV rows match
the workbooks to less than 5 × 10⁻⁶ μA and 4 × 10⁻⁶ mV; time
matches to numerical precision. The converted-file labels are source settings,
not measured currents. The files do not include a conversion formula, schematic,
scope setup, or calibration record. Pre-pulse currents and voltages vary by record;
the existing analyses subtract each record's pre-pulse median. The output plateau
described below is also visible before that subtraction.

Across all 22 records, those pre-pulse medians run from 3.87 to 38.23 µA
and −10.32 to +19.52 mV. Several records have positive raw current and
negative raw output before the pulse, so raw `V/I` cannot be read literally
as a passive device resistance. Per-record subtraction removes the observed
electrical baseline, but whether some of that current physically preheats
VO₂ is unknown. If the upper-end 40 µA leakage actually traversed a cold
2.57 kΩ device, the drop would be about 103 mV, much larger than the
corresponding recorded pre-pulse output, and its simple `I²R/S_e` steady
rise would be about 1.1 K;
that scale matters near onset, though it is smaller than the roughly 5 K
high-current inverse discrepancy. The model initializes each replay at T₀
on the insulating branch, so leakage and recovery history need a circuit
measurement before being treated as known initial conditions.

One controlled replay treated each record's *raw* current, including its
pre-pulse offset, as actual device current for the full 296 ns prehistory.
After subtracting the simulated pre-pulse voltage, the predicted late
incremental output at about 606 µA fell to 110 mV versus 181 mV measured;
at about 908 µA it fell to 16 mV versus 173 mV measured. For the latter
record the model also predicts about 89 mV pre-pulse device voltage versus
19.5 mV recorded raw output. Full assignment of the CH2 baseline to a cold
VO₂ branch under the adopted model therefore worsens the mismatch. Some
physical leakage or initial-state effect remains possible; this replay does
not determine the true current split.

A separate forward sensitivity test varied the model's starting temperature
from 285 to 375 K while retaining the measured, baseline-corrected 296 ns
prehistory and every frozen specimen parameter. Across all 22 traces, the
largest change in a 50–250 ns predicted mean was 0.144 mV; the mean-voltage
RMSE remained about 44.82 mV and no oscillations appeared. Starting the
hysteresis state on the metallic instead of insulating major branch produced
the same maximum 0.144 mV change and no oscillations. This is expected from
the adopted 13.03 ns thermal time: 296 ns spans about 22.7 thermal times.
Thus an arbitrary *thermal* initial temperature or major-branch label cannot
repair this frozen replay when its observed prehistory is retained. A real
long-lived filament/phase state outside the implemented temperature-memory
law remains possible and should be tested with recovery-delay measurements.

The [experimental manuscript](https://arxiv.org/pdf/2604.04594) says a function
generator pulse is converted to current by an LT1228 transconductance amplifier,
VO₂ replaces the TIA feedback resistor, and a common-base protection stage precedes
the TIA (Results around Fig. 6 and Methods B). It describes `Vout` as the output
voltage and uses `Vout*Iin` for power. The manuscript description and supplied
exports do not establish synchronized measurements of both VO₂ terminals,
the CH2 current-monitor transfer, summing-node voltage, current split, or calibrated
channel delay for these records. Thus the available evidence does not prove that
`Output Voltage` is exactly the drop across VO₂ or that `Input Current` is exactly
the current through the modeled parallel VO₂–C branch. The measured source current
is relatively flat during the oscillations; that observation does not establish
feedback-path current in an active circuit.

### Energy-per-period definition

Figure 7's approximately 1.2–1.9 pJ values can be reproduced without a device
model. After the same −200 to −50 ns per-record baseline subtraction, direct
integration of `Input Current × Output Voltage` between adjacent late voltage
peaks gives 1.159–1.754 pJ across the 11 records that pass the persistence audit.
These direct cycle integrals agree with `mean power / frequency`, the archived
calculation, to 1.30% median and 2.76% maximum relative difference.

This agreement validates the numerical calculation of the published
**output-derived total energy per period**. It does not establish that the product
is heat deposited only in VO₂. It also does not measure only the oscillatory
excursion: integrating power after subtracting each cycle's minimum gives
0.044–0.313 pJ, 2.5–27.0% of the total (9.3% median). Future comparisons should
state which definition is used and should reserve “device switching energy” for
simultaneous, calibrated device-terminal voltage and device-current measurements.

## A cross-current signature stronger than the single 607 μA point

Using the unaltered CSVs, subtract each channel's median over −200 to −50 ns and
average over 150–250 ns. Since the late current is nearly constant, the ratio of
means agrees with the archived mean instantaneous effective resistance:

| Source label (mV) | Late current (μA) | Late output (mV) | V/I (Ω) |
|---:|---:|---:|---:|
| 250 | 190.1 | 319.5 | 1680.5 |
| 300 | 228.7 | 232.7 | 1017.4 |
| 350 | 267.3 | 189.9 | 710.7 |
| 400 | 303.6 | 188.0 | 619.1 |
| 500 | 381.4 | 186.3 | 488.4 |
| 800 | 606.6 | 180.9 | 298.2 |
| 1000 | 758.4 | 174.9 | 230.6 |
| 1200 | 907.9 | 172.9 | 190.4 |

From about 303 to 908 μA, the late output remains within approximately 173–188 mV
while V/I falls by more than a factor of three. This could reflect a driven
voltage-clamping/filament state or the active circuit's transfer function; the data
alone do not assign it to either. A constant positive series resistance predicts a
positive `I × R_s` voltage slope when the device is metallic and does not explain
this nearly flat or gently falling output branch.
At the highest measured setting, apparent 190 Ω is still about 10.4 times
the low-current fitted 18.2 Ω metallic resistance. Thus the statement that
the device is completely metallic at high current and the identification
`V_out/I_in = R_m` cannot both hold for these records without another
voltage drop or a changed driven resistance state.

The plateau survives the baseline choice: before correction, these 15 records
span only 184–192 mV in late output. Corrected late input current is almost
linear in the source-setting label across all 22 records
(0.7577 μA per mV, R² = 0.99998, maximum residual 2.65 μA). The recorded
current channel shows no plateau or clipping up to about 908 μA. That does not
prove the same current flows through VO₂ inside the active feedback circuit.

Shifting the 800 mV record's current channel by −20 to +20 ns relative to
voltage keeps its 150–250 ns effective `V/I` within 298.20–298.28 Ω
and mean `V×I` within 109.69–109.72 μW. An unknown
channel delay matters for edge timing, phase and capacitance, but an ordinary
nanosecond skew alone does not remove this late mean discrepancy.

At 606.7 μA, the conditional reconstruction reports 298.2 Ω and 344.25 K, while
the Yuanhang law replayed on that temperature history gives 29.7 Ω (26.4–33.8 Ω
over the four tested γ values). The measured-output product is about 110 μW.
The major heating branch at 344.25 K gives about 34 Ω; to obtain 298 Ω on that
branch requires approximately 339.27 K. A *quasi-steady* balance at that point
would require S_e ≈ 4.41 μW/K rather than 3.68 μW/K. This 5 K thermal
shift explains why a tenfold resistance gap need not imply a tenfold parameter
error near a steep transition. The record oscillates, so this calculation is a
diagnostic scale, not a new estimate of S_e.

The frozen *forward* prediction gives the clearest scale comparison. For the
800 mV source label (about 606 μA), it predicts 153.3 mV mean output and
339.6–339.7 K, with about 93 μW modeled Joule power. Its implied resistance
is about 253 Ω. The measured mean is 181.6 mV and its output-current product
about 110 μW. Applying the adopted S_e to the roughly 17 μW power difference
shifts the thermal balance by about 4.7 K. The conditional inverse replay
then evaluates the steep fitted R(T) at roughly 344.3 K, yielding about
30 Ω. Thus the tenfold *inverse* resistance gap is amplified from a roughly
28 mV forward mean-voltage error through the assumed power-to-temperature
map; it is not evidence by itself for a tenfold error in the forward voltage.
The forward prediction still lacks the observed sustained voltage cycles.

A direct one-factor replay used the same measured current histories, resistance
preset, initial temperature, `C = 0.39 pF`, `C_th = 0.047873236 pJ/K`, and
`dt = 0.05 ns`; only `S_e` changed. Values below are baseline-corrected means
over 50–250 ns, in mV:

| Measured current (µA) | Measured | Forward at 3.675 µW/K | Forward at 4.413 µW/K |
|---:|---:|---:|---:|
| 189.6 | 318.9 | 319.5 | 339.5 |
| 228.2 | 235.1 | 315.1 | 352.5 |
| 380.9 | 187.0 | 225.2 | 265.5 |
| 606.3 | 181.6 | 153.3 | 181.7 |
| 756.6 | 175.7 | 127.2 | 150.6 |
| 906.8 | 173.4 | 109.1 | 129.0 |

At 606 µA, the higher conductance removes the *mean* error to about 0.1 mV,
yet its predicted full 50–250 ns voltage span is only 1.8 mV. It worsens
the 228 µA mean error from 80 to 117 mV and leaves a 44 mV deficit near
907 µA. That value therefore cannot be promoted as a shared calibration.

For another diagnostic, invert the *major heating branch* at each measured late
mean `V/I`, then calculate the conductance that would balance `I×V` at that
temperature. This imposes a direct channel map, a uniform static resistance law,
and approximate steady state; it is not an independent estimate. The apparent
`S_e` is 3.68 µW/K near 190 µA, 2.47 near 304 µA, 4.41 near 607 µA,
5.22 near 758 µA and 6.09 near 908 µA. Strong oscillations make the
midrange inversion especially approximate, but the high-current variation
shows why a constant conductance fitted at one record does not close the
whole sweep under those assumptions.

This inconsistency does not depend on adopting the old T₀ or S_e estimates.
Under the same quasi-steady major-branch interpretation, the 190→606 µA
pair changes apparent power by 49.0 µW and inferred temperature by 8.38 K,
requiring slope `ΔP/ΔT = 5.85 µW/K`. The 606→908 µA pair changes power by
47.2 µW but temperature by only 0.93 K, requiring 51.0 µW/K.
Fitting that latter pair alone extrapolates to `T₀ ≈ 337.1 K`, above the
190 µA record's inferred 330.9 K. In 1000 *paired* static R(T) bootstrap
draws, the high-pair/low-pair slope ratio remains 7.4–10.7 (95% range).
Thus allowing a different constant S_e and T₀ does not make these late
points mutually consistent within the direct-channel, static-branch,
near-steady model. The bootstrap only samples that fit's uncertainty;
driven filament states, circuit transfer, and thermal nonuniformity remain
unconstrained alternatives.

This cross-current signature is insensitive to reasonable choices of averaging
windows. Across four pre-pulse median windows (−250 to −150, −200 to −50,
−150 to −25, and −100 to −20 ns) crossed with four output windows (50–150,
100–200, 150–250, and 175–250 ns), the 800 mV record's apparent resistance
ranges from 296.1 to 301.4 Ω. Repeating the same major-branch inversion for
the 250, 800, and 1200 mV records gives a high-pair/low-pair `ΔP/ΔT` slope
ratio of 8.49–9.24 across the 16 choices. This tests window and baseline
sensitivity only; all 16 calculations retain the same unverified channel map,
static R(T) transfer, and approximate steady-state premise.

A fixed *relative* channel gain can change that slope ratio, because it changes
which part of the steep R(T) curve is used as a thermometer. For example,
artificially multiplying every late `V/I` by 0.10046 makes the inferred
temperatures at 190, 607, and 908 µA approximately 340.44, 344.73, and
348.86 K and equalizes the two apparent power–temperature slopes. This is a
three-point construction, not a gain estimate: it corresponds to the actual
device voltage being about one tenth of reported output relative to the actual
device current, or an equivalent current-scale error. Extrapolating those three
points gives `T₀ ≈ 335.12 K`, versus the adopted 314.4 K. Even under that
constructed balance, the 38 µA control inverts to 339.79 K but its power
predicts only 335.42 K. A large, state-dependent transfer, a different driven
R(T), or nonuniform/dynamic heating could still matter; a constant gain cannot
be inferred or validated from these three selected points.

The conductance conflict also survives choosing records away from the strongest
midrange oscillations. Using the 50→250 mV low-current pair and 900→1200 mV
high-current pair, the same late major-branch inversion gives apparent
`ΔP/ΔT` slopes of 4.64 and 55.31 µW/K, respectively (ratio 11.93).
Across the 16 baseline/output-window choices, that ratio is 11.52–12.78.
Across all 1000 paired draws in the
[archived static R(T) bootstrap](../public_jobs/20260816_125905_sample-r-t-major-loop-hysteresis-fit_0849a9/parameter_bootstrap.csv), its
2.5–97.5% range is 10.73–13.14. The high-current records retain small
periodic output, so they are only approximately steady; the bootstrap still
does not account for a changed driven R(T), spatial temperature, or an active
channel transfer. This stronger separation cannot calibrate `S_e` on its own.

If both channels truly bound the modeled VO₂–C branch, capacitance cannot remove
the average-power discrepancy over a repeated cycle. Integrating the electrical
balance gives `mean(I×V) − mean(V²/R) = C×[V(t₂)² − V(t₁)²]/[2×(t₂−t₁)]`.
The right side vanishes for equal cycle endpoints. The late mean power is
therefore largely a channel/circuit or thermal/constitutive question, even if
the instantaneous capacitive split is uncertain.

The discrepancy has opposite signs elsewhere: at 228.7 μA the measured effective
resistance is about 1018 Ω while replay gives 1763 Ω. A single shift in γ,
temperature, or uniform voltage/current scale is therefore an inadequate
explanation without checking the entire sweep. Above 700 μA, measured effective
resistance remains roughly 190–341 Ω while the conditional replay approaches the
18.2 Ω metallic floor. The gap is a branch-wide issue, not an isolated waveform.

There is a particularly simple onset ordering check. From the 250 to 300 mV
source labels, the corrected late output/current ratio falls from 1680.5 to
1017.4 Ω while their product falls from 60.73 to 53.23 µW. At the 350 mV
label, both fall further, to 710.7 Ω and 50.76 µW. Without baseline
subtraction, the raw output-current products also fall, from 62.01 to
53.36 to 50.93 µW. A common positive thermal conductance, ambient
temperature, and *monotone quasi-steady heating
branch* would instead require lower resistance to accompany higher heating
power. All 16 baseline/output-window combinations above preserve both
decreases; for 250→350 mV the power drop is 8.5–11.0 µW. Multiplying both
channels by constant positive gains preserves the ordering. The 300 and
350 mV records oscillate, however, so their mean `V/I` need not equal
`R(mean T)` on the static major branch. This is a falsification of the
combined quasi-steady static interpretation across onset, not proof of a
particular new device state or circuit error.

To reverse the 250→350 mV apparent-power ordering while holding the reported
350 mV current fixed would require at least 9.97 µW more device power there,
equivalent to about 37 mV more device voltage. That is a useful resolution
target for a two-terminal measurement; it is not an offset to insert into the
simulation, because the true current path and oscillatory state remain unknown.

As a check on forcing, fitting a sinusoid at the measured late voltage frequency
near the 800 mV label gives about 6.2 mV peak-to-peak in output voltage and only
0.08 μA peak-to-peak in recorded input current. At a fixed 298 Ω, approximately
21 μA current ripple would be needed for that voltage component. Ordinary ripple
in the *recorded* current channel therefore cannot account for the voltage cycle;
unmeasured current inside the TIA remains possible. The 1 ns samples resolve the
approximately 63 MHz component, but they do not establish analog bandwidth or
probe phase calibration.

The high-current boundary is threshold dependent. At the 1100 and 1200 mV
labels (about 835 and 907 µA measured), a sine fit to 150–250 ns gives
roughly 5.7 and 6.0 mV peak-to-peak at 68.6 and 68.8 MHz. Same-frequency
pre-pulse fits are about 0.05 and 0.12 mV; recorded input-current components
are only about 0.14 and 0.09 µA peak-to-peak. These records fail the existing
four-window 6 mV persistence rule, so they are not evidence of sustained
VO₂ cycling. Their late periodic output still merits a dummy-resistor and
longer-pulse control before treating the high-current branch as featureless.

Forcing the old 344.3 K replay to match by subtracting the full roughly
163 mV voltage difference is not self-consistent: the remaining device power
would be about 11 μW, implying a steady temperature near 317.4 K, where the
fitted heating branch is insulating. A smaller correction, about 28 mV at
607 μA, would bring the measured mean to the forward fixed point *if* the
actual circuit supported that map. But the frozen forward mean is roughly
80–100 mV **above** measurement at 228–267 μA and about 28–64 mV
**below** it from 606 to 907 μA. A single fixed output offset cannot repair
the full sweep, and any circuit correction must be propagated through power,
temperature and oscillation dynamics.

A broader simple readout map also fails in sample. Across all 22 frozen
mean predictions, fitting `V_meas = a + b V_model` lowers RMSE from
44.8 to 29.3 mV but still misses the 190 µA control by 62.7 mV.
Allowing an additional linear current-feedthrough term lowers RMSE only to
27.0 mV and still misses that control by 60.6 mV. The frozen forward
predictions differ by only 4.4 mV between 190 and 228 µA, while measured
means drop by 83.8 mV. These are deliberately permissive *in-sample*
linear maps, so their failure rules out a simple fixed gain/offset/series
term; a nonlinear, state-dependent circuit transfer remains possible.

The turn-off edge gives another direct boundary test. In 9 of 22 records,
raw output voltage is below −10 mV while raw current is above 50 μA
between 270 and 400 ns; this remains true after per-record baseline
correction. For example, at the 1000 mV label and 313 ns, raw output is
−64.75 mV with raw input current 104.11 μA (corrected values −81.49 mV
and 71.17 μA). In that record the corrected output first crosses zero
near 304 ns while corrected current is still about 345 μA; current remains
positive up to the crossing. Starting from positive voltage, the modeled passive
parallel R–C branch with positive imposed current and positive R cannot
cross below zero: at V = 0 its derivative is I/C > 0 for `C > 0`, and
the exact `C = 0` limit gives `V = I R > 0`. These samples therefore
require a channel transfer, unmeasured current path/reversal, or another
active-circuit effect if the measured current is truly the branch current.
Turn-off behavior alone does not determine the cause of the late output plateau.
The edge sign argument is sensitive to unmeasured channel alignment. At the
first corrected output zero crossing in these nine records, corrected input
current ranges from about 188 to 362 µA on the recorded time axis. A synthetic
current advance of about 9–15 ns brings that current below 50 µA in every
record; advancing it by 15 ns removes the specific simultaneous
`V < −10 mV, I > 50 µA` flag from all nine. It takes roughly 18–25 ns
to reach 10 µA at those crossings, and baseline noise prevents a reliable
zero-current timing for several traces. Neither advance is an actual measured
skew. Thus the raw edge is a targeted timing and circuit-calibration test,
not standalone proof of an active current reversal or negative resistance.

## Model and inference audit

The implemented ideal-source equations and boundary units agree with the documented
model: `C dV/dt = I − V/R(T,H)` and
`C_th dT/dt = V²/R − S_e(T−T₀)`. The exact `C = 0` limit and the
float32-faithful hysteresis path are explicit in the source. The historical
mechanism control and timestep studies support numerical implementation, while
the frozen specimen prediction fails to recover any of the 11 measured oscillatory
records. A passing peak-count label in later fits did not guarantee persistence;
the 607 μA anchored prediction decayed by about two orders of magnitude across
the audited windows. These findings point to model assumptions and/or the
measurement map rather than a simple unit conversion or integrator error.

The experimental manuscript provides a specific reason to test an additional
state only after circuit calibration: under separate voltage-pulse measurements,
it reports roughly 30 ns switching rise, resistance creep toward a settled value
for about 300 ns, and small pump–probe memory at 100–200 ns delay. It interprets
these as filament growth and relaxation. Those tests used a different PCB and
drive mode, so they support a *candidate mechanism*, not a fitted kinetic law
for this current sweep. A single measured relaxation state or spatial filament
model would be more interpretable than freely refitting all lumped parameters.

One diagnostic implementation error was found and corrected during this audit:
the timestep reporter clamped thermal capacitance to 10⁻¹² J/K although the
specimen value is 4.79 × 10⁻¹⁴ J/K, understating its estimated per-step
thermal jump by about 20.9-fold. A separate domain-search reporting denominator
had a similar unit-insensitive floor. The physical integrators, archived waveform
fits, and inverse replay did not use these diagnostic denominators. A regression
check now evaluates the physical step-jump ratio directly. The diagnostic from
`current_drive_numerics_report` also feeds automatic timestep reduction in the
standalone sweep-GIF helper, so a future direct use of that helper can now
choose a smaller step; the supported frozen validation path supplies its
own explicit timestep.

The thermal-capacitance estimate is highly sensitive to that unknown alignment.
Recomputing the *existing* 100/150/200 mV edge fit (15–35 ns, same R(T),
S_e and smoothing) after replacing recorded current by
`I_shift(t) = I_recorded(t + shift)` gives:

| Current shift (ns) | C = 0: fitted C_th (pJ/K), RMSE (K) | C = 0.39 pF: fitted C_th (pJ/K), RMSE (K) |
|---:|---:|---:|
| −2 | 0.06108, 1.827 | 0.06117, 2.263 |
| 0 | 0.04716, 0.806 | 0.04787, 1.145 |
| +2 | 0.03668, 0.675 | 0.03792, 0.635 |

At the adopted `C = 0.39 pF`, just +2 ns changes the point estimate by
−21% and improves the residual from 1.145 to 0.635 K. This is a synthetic
skew sensitivity calculation, not evidence that the actual channels are 2 ns
misaligned. The reported capacitance interval did not sample channel delay,
so its precision cannot be read as an independent hardware measurement.
For context, fitting a constant `V = R I_shift` relation only over the first
−5 to 15 ns of the 100–250 mV rising edges favors an *effective* current
advance of about 1.3–2.1 ns. That edge fit mixes probe delay, circuit
transfer, capacitance, and early heating, so it cannot calibrate true skew.

| Quantity | What the present evidence supports | Main entanglement |
|---|---|---|
| Static major R(T) | Good log-space description of this 2 μA heating/cooling sweep, with typical 8.8% and maximum 33.6% multiplicative fit errors | Transfer to the same mounted TIA device and driven filament state is unverified; R₀ and activation energy covary. |
| Minor-loop γ | No sample-specific estimate | A single major loop contains no minor-loop information; γ trades against thermal/circuit dynamics in voltage fits. |
| Electrical C | No resolved positive value from the source-limited edge | The 0.39 pF number uses an assumed 1 ns lag bound, cold R(T), and uncalibrated channel skew; it is conditional, not a hard physical upper bound. |
| S_e | Conditional estimate from one settled pre-onset record | Requires V/I to be device resistance, static thermometry to transfer, and the assumed ambient temperature; its bootstrap interval omits those systematics. |
| C_th | Conditional early-heating fit and time scale | Shares R(T), S_e, C, channel alignment and power assumptions; 15–35 ns overlaps the 26–27 ns source rise, and an uncalibrated ±2 ns current shift moves the point fit by −21% to +28%. |
| Voltage mean, oscillation amplitude/frequency/persistence | Directly measurable at the output with stated windows and bandwidth | They do not by themselves identify internal temperature, phase fraction, branch current or capacitance. |

In the nonswitching linear limit, the heating trace principally exposes a time
constant `τ = C_th/S_e` and a response amplitude proportional to `P/S_e`.
The latter becomes an `S_e` estimate only after measured channel product `P`
and static `R(T)` thermometry are accepted. An unknown channel transfer changes
both the apparent power and the inferred temperature. Electrical `C` and
channel delay both affect edge phase. A major R(T) loop does not excite the
minor-loop parameter γ. These dependencies explain why resampling the same
trace can give precise conditional intervals without identifying all of the
physical parameters.

The model's thermal-only transition-center diagnostic gives approximately 330 μA
to heat into the transition and 254 μA to cool back with adopted parameters, so it
has no simple `C = 0` oscillation window. Raising S_e to the 607 μA illustrative
value raises these thresholds further, away from the observed onset near 228 μA.
One thermal adjustment cannot be inferred from the high-current point alone.

## Three prioritized discriminating tests

1. **Calibrate the circuit boundary.** Obtain the Figure 6 board schematic and
   probe/gain records, then measure both VO₂ terminal voltages, the TIA output,
   the current entering the feedback element and the CH2 monitor simultaneously
   at about 190, 267, 607, and 908 μA. The 190/267 µA pair directly tests
   the onset power-ordering reversal above. Include a known resistor in the
   feedback position to measure the transfer function, sign, baseline, loading
   and channel skew;
   record actual VO₂ leakage and recovery before the pulse.
   A roughly 300 Ω dummy feedback resistor is especially informative: an
   ideal TIA would give output magnitudes near 57 mV at 190 μA and 272 mV
   at 908 μA, instead of the recorded nearly flat output.
   Check whether the dummy also shows a negative turn-off undershoot.
   Compare the inferred device voltage/current/power with the exported columns.
   A state-dependent output or current-path difference would directly invalidate
   the present thermal inversion; agreement would make intrinsic dynamics the
   leading target. At 607 μA, the frozen model's roughly 153 mV device
   prediction and the recorded 181 mV output give a useful quantitative check.
2. **Repeat resistance and heating checks on the *same mounted device*.** Confirm
   its low-current major and minor R(T) loops and actual ambient temperature.
   Add at least one independent temperature or calibrated thermal-response measure
   under drive, if feasible. Compare the observed 607 μA operating state with the
   roughly 339 K versus 344 K alternatives above. If device temperature is near
   344 K yet device resistance is near 300 Ω, quasi-static lumped R(T) fails;
   if it is near 339 K, revisit heat-loss/power assumptions before changing γ.
3. **Test memory and long-lived dynamics with a small pulse matrix.** Record
   approximately 190, 228, 267, 607, 682, and 908 µA for 300 ns and
   approximately 1 µs pulses, at recovery delays around 100 and
   500 ns. Evaluate late-window mean, robust and periodic amplitude, frequency,
   current–voltage phase and window-to-window retention. If calibrated electrical
   channels are consistent but resistance depends on pulse history beyond the
   thermal response, fit one measured phase/filament relaxation law and validate
   it on held-out currents. If a state decays after 250 ns, do not label it
   sustained from the present finite record.

Accept a new shared model only after it predicts the low stable control, onset,
midrange and high-current output means, late amplitudes, frequencies and
persistence at converged timesteps. Keep parameter intervals conditional on the
measurement map and the model used to obtain them; no broad eight-parameter fit is
needed before these three checks.
