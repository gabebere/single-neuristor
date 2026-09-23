# Professor-supplied TIA current sweep

These files are the untouched numerical sources supplied by Prof. Yoav Kalcheim on
2026-08-17 for the VO2 transimpedance-amplifier measurements in Amir Gildor's
manuscript.

- The 22 files ending in `_converted.csv` were used to generate Figure 7.
- `100mv02.xlsx`, `500mv0-power-oscillations.xlsx`, and `1000mv02.xlsx` were used to
  generate the three Figure 6 panels.
- Each converted CSV contains 1000 rows with no header: relative time in ns, input
  current in uA, and output voltage in mV.
- The workbooks retain the original channel voltages, converted quantities, and two
  presentation time columns. The second `Time [ns]` column is the first plus 150 ns;
  it moves the pulse into the 0--600 ns Figure 6 display and is not a physical delay.

The source ZIP was `/Users/gabrielberezovsky/Downloads/Data.zip` with SHA-256
`ddc9bb4be993f34875f6d025b2b479df447a56c044fd1814707ba104b58303b5`.
AppleDouble `__MACOSX` metadata from the ZIP is intentionally not archived.
Run `shasum -a 256 -c SHA256SUMS` in this directory to verify every preserved
measurement file.

Active analysis reads the numerical CSVs through
`src/neuristor/experimental_waveforms.py`; no plot or screenshot digitization is
used.
