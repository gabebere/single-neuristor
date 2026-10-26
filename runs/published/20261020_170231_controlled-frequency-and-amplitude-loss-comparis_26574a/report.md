# Frequency-only versus frequency-and-amplitude fitting

Both searches use the same measured currents, fixed R(T) and T0, bounds,
starting vectors, seed, evaluation cap, and verification protocol. Only the
amplitude-error weight differs. A finite budget is not a global optimum.

The unknowns are gamma, C, Se and Cth. Internally Cth is parameterized by Cth/Se.
The broad search bounds are diagnostic choices, not measured specimen intervals.
All selected target records were used for optimization; the other records are controls.
Control-oscillation counts use the simulated complete-cycle gate, not a new
experimental reclassification. Gate outcomes must not be confused with
complete-waveform agreement.

Scientific objective values are present only for all-target eligible candidates.
Partial frequency MAPE is labeled with the number of defined frequencies.
The two *_MSE_defined columns cross-evaluate both losses on defined frequencies
only; these partial diagnostic values are not accepted all-target objectives.
Amplitude means peak-to-peak voltage. Means, controls and timestep changes remain
out-of-objective diagnostics. Sampled proximity and temperature screens do not
prove physical validity. The voltage traces are compared without phase alignment.

Low, middle and high representative target currents were selected by current
rank, before inspecting residuals. Titles use baseline-corrected measured mean
current over 50–250 ns; acquisition mV labels appear only in provenance tables.
No new simulation, smoothing, or filtering was performed for this comparison.

See summary.csv for both objective evaluations on both parameter vectors,
convergence.csv for each current at both refinements, and parameters.csv.

frequency_only: diagnostic_1; 250 unique candidates; accepted search candidate: False.

frequency_amplitude: diagnostic_1; 250 unique candidates; accepted search candidate: False.

Measured target records at 606.3 µA themselves fail the selected retention gate, despite measurable frequency/Vpp.
