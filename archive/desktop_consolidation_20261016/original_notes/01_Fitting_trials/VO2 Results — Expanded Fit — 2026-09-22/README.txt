VO2 RESULTS — latest expanded fit with stronger R(T) constraint

START HERE: open START HERE.html, then select a current below the animation.
Scope replay/scope/ contains 22 synchronized GIFs and 22 full-trace PNGs.
Each GIF progressively draws measured/simulated voltage and the measured imposed
current, while the simulated resistance-temperature trajectory evolves on the
heating and cooling major branches. Playback is slowed down, not real time.

Parameter choice: static_preserving, expanded search 694804.
Same parameters as the most recent interface result eef073; the new export uses
the finest checked timestep (0.00625 ns) instead of 0.025 ns. This recovers 7 of
11 sustained oscillators, versus 8 at the coarser step. The mean-voltage RMSE is
about 44.6 mV. This is a promising compromise, not an unqualified best fit:
the older anchored fit recovers 9/11, and the relaxed fit improves mean voltage
but worsens static R(T). Oscillation amplitudes still disagree substantially.

Scope replay/traces.csv: all 22 records, time, measured input current, measured
voltage, simulated voltage, simulated R and T on the experimental time grid.
Scope replay/major_branches.csv: heating and cooling model curves.
Scope replay/summary.csv and metrics.json: per-current and aggregate errors.
Scope replay/recipe.toml: exact original recipe and provenance.
Scope replay/comparison.html: offline interactive voltage/resistance slider.
reproduce_scope.toml: portable recipe pointing at the included raw data;
run with `neuristor analyze replay-lab --config reproduce_scope.toml` in an
environment with the neuristor package installed. Source snapshots are retained
inside the run bundles. Completed bundle contents have not been modified.

Both measured channels are pre-pulse baseline corrected. Measured current is
imposed total input current, not separately measured VO2 conduction current.
Temperature and resistance are model predictions, not experimental traces.
Major branches are reference guides; a driven minor loop need not follow them.
The relation of exported experimental voltage to device voltage remains conditional.

Also included: both joint-search bundles (parameters, verification at three
steps, numerical results and figures), the previous displayed replay, measured
input data, analysis notes and the updated presentation. The Desktop copy is
independent of the browser/server and can be kept or shared as a folder.
