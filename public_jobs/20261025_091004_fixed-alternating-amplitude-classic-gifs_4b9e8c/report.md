# Fixed historical parameters and classic current-sweep GIFs

The exact 7 September relaxed diagnostic vector is frozen in parameters.json.
All 22 original baseline-corrected current inputs are replayed without optimization
at dt=[0.025, 0.0125, 0.00625] ns. The 44 GIFs show the original 0.025 ns and finer 0.00625 ns
replay, each with current at top, simulated voltage in the middle, and native
R(T) below. There is no measured-voltage overlay. Heating/cooling guides use the
same resistance law. The window is −30 to 280 ns. Playback frames do not alter
integration resolution. Unresolved R(T) jumps have breaks, never invented chords.
The complete native arrays at all three timesteps are preserved.

Original 0.025 ns sampled voltages reproduce the source within
1.523e-11 mV maximum difference. Current labels
give the median input over 150–250 ns and retain the original source filename label
separately. The same vector drives all currents. Its C and gamma exceed the
scope of the historical constrained calibration. No calibration is adopted.

## Time constants and a working explanation

Passive thermal cooling scale: tau_th=Cth/Se=11.90870638 ns.
Instantaneous frozen-resistance electrical scale: tau_e(t)=C R(T,H).
Along the 400 mV-source replay in 50–280 ns, tau_e spans
0.133365–11.984629 ns; the fully metallic limit is
0.124510 ns. The thermal scale is not an independent fitted
parameter, and Joule heating/hysteresis mean it is not the full coupled-system
relaxation time or a predicted oscillation period.

Individual cycles alternate between shallow heating to about 338 K (minimum R
about 120 ohm) and deeper heating to about 348 K (minimum R about 20 ohm).
RC charging in the high-resistance state takes a time comparable to thermal
recovery. The much shorter electrical discharge during a deep low-R reset and
the different hysteresis reversal states can leave different charge/thermal
histories for the next cycle. This is a plausible electrothermal-memory
explanation, not a demonstrated causal bifurcation mechanism. Time-scale
comparison alone cannot establish that mechanism.

The earlier fixed-vector audit shows that short-window alternation persists
at two finer timesteps but a longer 2 us constant-current replay becomes irregular
at the finest step. Stable long-time period doubling and chaos remain unproven.
See original audit 20261025_082439_documented-alternating-amplitude-historical-simu_1072ef. No physical solver or memory rule was changed.

The repository recipe is experiments/current/alternating_amplitude_gifs_20261025.toml.
The exact media-generation script and package source are preserved. parameters.json
contains all resistance and current-model settings in SI and user units.
