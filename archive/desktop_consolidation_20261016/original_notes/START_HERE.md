# VO₂ research archive

Organized 3 October 2026. Original Desktop files were moved, not deleted or
rewritten. Duplicate ZIPs and renderings remain as received. This is a local
archive, not confirmation of a GitHub backup. Historical links to the former
Desktop paths may need updating; repository run IDs are unchanged.

## Recommended review order

**New:** `04_Parameter_maps/` holds the completed three-point clickable-GIF
preview. Its README distinguishes the benchmark from the full generation job.

1. **00_Current_trial** — the new equal-weight relative-squared frequency/Vpp
   objective. Read the review before watching the GIFs. An attractive waveform
   is not proof of physical validity or a successful fit.
2. **01_Fitting_trials** — previous optimization methods, in chronological order:
   expanded joint fit → unconstrained settled fit → monotone-proximity fit →
   tanh pilot → the new extrema trials. Compare all candidates using one common
   measurement window and extrema definition, not their original scores.
3. **02_Parameter_diagnostics** — controlled parameter changes, not fitted models.
4. **03_Historical_references** — the visually preferred eef073 reference and
   parameter comparison spreadsheet; retained for comparison, not adopted as truth.

## What the old folders mean

| Folder / group | Question tested | Interpretation |
|---|---|---|
| Expanded Fit, 22 September | Joint static resistance and dynamic fitting | Improves some averages, but amplitudes and persistence remain inadequate. |
| Background Search / steady best search_1 | Fit settled oscillations without phase/DC error | Best frequency error about 14.9%, amplitude error about 109.5%; diagnosed nonmonotone proximity. Not physically accepted. |
| Monotone Proximity Search / monotone best | Can admissible minor-loop behavior preserve the fit? | Constraint removes the problematic cooling hook; no satisfactory fit. |
| Tanh Proximity Pilot | Could another proximity kernel sustain oscillations? | 9/9 sustained target records, but about 647% amplitude error. A mechanism lead, not a calibration. Gamma here means tanh k, not the sine-kernel gamma. |
| 500 µA, Cth 1e-4, T0 298 K | Reproduce an approximately 67 MHz example | About 66.13 MHz, but temperature reaches about 2494 K. Nonphysical diagnostic; do not select it by frequency alone. |
| 500 µA, Cth 0.072 | Increase thermal capacity at the same nominal setting | One-parameter diagnostic. Inspect its saved temperature trace rather than assuming it is a fit. |
| eef073, T0 324.4 K | Ambient-temperature sensitivity | Saved replay recovers 0/11 measured oscillators under its historical gate. Not a calibration. Different GIF folders are different renderings, not independent fits. |
| eef073 GIFs by current | Visual reference | Original preferred replay; gamma violates the later proximity screen. Preserve, but do not adopt. |

The archive includes **18 original top-level items**: seven fitting-trial items,
nine diagnostic items, and two reference items. Original names are preserved.
`ARCHIVE_INVENTORY.json` records file counts, sizes, aggregate SHA-256 checksums
and symbolic-link targets for each historical item. The added 2 October extrema
bundle is listed separately from the 18 original Desktop items.

## Evidence and portability

The Background and Monotone launcher folders originally contained symbolic links
to repository results. **Frozen Results** now contains real copies of those run
bundles. Use those copies for archival review; the original Results/status/log
links remain untouched. Their launch instructions saying “running” are historical,
not current process status. Other original folders may also contain local paths;
inspect provenance before relocating the archive to another computer.

Each run should be reviewed in this order: report → parameters → finest-step
summary / physical checks → figures → GIFs. Record the objective, target records,
actual measured currents, measurement window, timestep, static R(T) error,
persistence, frequency error, Vpp error, and rejection reason. Lower historical
scores cannot be compared across different objectives or measurement definitions.

Scientific handoff and canonical evidence:
`/Users/gabrielberezovsky/Documents/Projects/single-neuristor/docs/RESEARCH_HANDOFF.md`.

For future trials use `YYYYMMDD_method_runID`, one immutable folder per run,
with a short review and a status of **accepted / diagnostic / superseded**.
No quantitatively satisfactory physically supported specimen fit is established.
