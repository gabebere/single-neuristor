# Budgeted joint R(T)/waveform inference

One shared vector fits all static measurements and nine configured representative current records. Other currents are excluded from optimization, but were seen in earlier research and are not pristine blind data. The eleven free quantities include all six major-loop parameters, gamma, ambient temperature, electrical capacitance and two thermal quantities.

Mixed log/linear unit-cube coordinates reduce scale disparities. Rs at 315 K replaces the correlated R0 coordinate; thermal time replaces Cth. Both searches share a feasible Latin-hypercube/perturbed historical-seed population and cached evaluations. Differential evolution is followed by a strictly capped Powell refinement; no new optimizer dependency is required.

Waveform loss = mean-voltage term + 3 × windowed periodic-amplitude term + robust-amplitude term + 0.5 × gated-frequency term. Mean scale is 20 mV, frequency scale 10 MHz, amplitudes use log ratios with a 3 mV floor. Static loss = weight × (log10 R RMSE / 0.05)^2. Weights, bounds, static rejection ceiling and budgets are in resolved_config.json. These are engineering preferences, not a noise likelihood or confidence intervals.

Budget: 240 objective calls; 214 unique candidates; 201 search simulations; 152.6 search seconds.

Inspect summary.csv by split and timestep; parameters.csv records static-fit degradation. Frequency error is reported only where both measured and predicted persistence pass; the denominator is explicit. All final candidates are checked on all currents at every configured timestep. Neither early termination nor a lower objective proves global optimality or a physical calibration. Source snapshots and hashes preserve the dirty-worktree implementation.
