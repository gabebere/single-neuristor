# Presets: saved parameters and older sample records

`resistance_100425_chip1_gap3.json` is a saved resistance fit used by historical
sample recipes. `samples/` holds older sample-library records and imported curves.
These are not new raw measurements or a guarantee of a successful dynamic fit.

`alternating_amplitude_relaxed_20260907.json` freezes the recovered historical
alternating-amplitude diagnostic. It includes the exact resistance law, full
current-model settings in SI, user units and derived time constants. It is not
an adopted calibration. Use `experiments/current/alternating_amplitude_gifs_20261025.toml`
for the corresponding operating settings and animation-study recipe.

For a new sample, provide its measured R(T) file to the
[sample-study recipe](../../experiments/studies/sample_parameter_map.toml). The fitted
parameters and assumptions are then recorded in that study's output bundle.
Do not silently change a preset used by reviewed evidence.
