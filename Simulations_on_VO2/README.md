# VO2 model, specimen and loss-function presentation

`main.tex` is the editable Beamer source; `main.pdf` is the compiled presentation.
The talk retains the reviewed first ten slides and continues with the fitting
story: how parameters produce a voltage trace, how frequency and peak-to-peak
voltage enter the two losses, and the paired controlled searches.

## Sequence

| Slides | Content |
|---|---|
| 1–2 | Title and experimental target |
| 3–7 | Electrical/thermal equations, material law, proximity and reversal math |
| 8 | From voltage drive to ideal current drive |
| 9 | Measured specimen R(T) fit and six fitted resistance parameters |
| 10 | Fitted values, adopted ambient temperature and four unknown specimen parameters |
| 11 | Parameters → simulator → complete-cycle frequency and Vpp, with Vpp defined |
| 12 | Relative errors and the frequency-only / frequency-plus-amplitude losses |
| 13 | Results across the measured input currents, with a button opening all 44 GIFs |
| 14 | Selected parameter values |
| 15–16 | Three low/middle/high-current examples for each search |
| 17 | Why we explored C, Cth and Se; example heatmaps and interactive-viewer button |
| 18 | Sparse controlled follow-up: shared C/Cth scale, inverse frequency and preserved voltage swing |
| 19 | Mica introduction, measured/fitted R(T), static parameters and adopted 298 K |
| 20 | Mica expanded search, diagnostic heatmaps and latest-completed-review button |
| 21 | End-to-end process chart: measured inputs, static fit, simulation/loss loop and candidate dynamic parameters |
| 22 | Alternating shallow/deep resets, thermal and electrical time constants, working mechanism and 44-GIF viewer |

`loss_function_story.tex` holds the editable methods/history continuation.
`loss_comparison_results.tex` holds tables authored from the new comparison bundle
and references its saved-sample figures. Complete methods, formulas, bounds,
measured-current definition and parent IDs are in
[`docs/LOSS_COMPARISON_20261020.md`](../docs/LOSS_COMPARISON_20261020.md).
Historical loss-function slides are omitted. Only the two objectives used in the
retained comparison are presented; earlier trials remain in the research archive.
Slide 13 includes an **Open all 44 GIFs** button targeting the local HTML viewer
in bundle `20261021_054434_loss-function-comparison-all-44-stacked-gifs_15921f`.
Both HTML buttons now use ordinary HTTP links on `127.0.0.1:8767`, replacing
local file URLs and PDF Launch actions that failed in Preview. No upload or
system startup agent is involved. The viewer service is read-only, listens only
on this computer, and serves only the original reviewed viewers, their
GIF dependency bundle, and completed published Mica reviews.

**One-command presentation (from the project folder):**

```bash
python3 src/neuristor/interfaces/presentation_server.py --present
```

This opens the existing PDF and starts all four HTML viewers in the background.
It works with Python 3.9 or newer using only the standard library: no package
installation, simulation dependencies, LaTeX, or Internet connection is needed.
Paths are resolved relative to the copied project, not Gabriel's home directory.
On macOS, double-clicking `Start presentation viewers.command` does the same thing.
With the full neuristor package already installed, `neuristor present` is shorter.

For a colleague's computer, preserve these relative folders:

- `Simulations_on_VO2/main.pdf` and optionally the `.command` launcher.
- `src/neuristor/interfaces/presentation_server.py`.
- `public_jobs/20261021_054434_loss-function-comparison-all-44-stacked-gifs_15921f/`.
- `public_jobs/20261007_123036_fixed-scale-viewer-portable-screened-parameter-m_4325d1/`.
- `public_jobs/20261007_074702_portable-screened-parameter-map-with-point-gifs_eaf6ab/` (original GIF dependency).
- The completed `public_jobs/*m2-1600nm*review*/` folders for Mica.
- `public_jobs/20261025_091004_fixed-alternating-amplitude-classic-gifs_4b9e8c/`
  for the final slide's 44 current / simulated-voltage / R(T) GIFs.

Only our presentation, viewer code and reviewed generated results are needed;
private/reference papers and the research environment are unnecessary. A PDF-only
copy cannot provide the linked web pages. The launcher reports missing viewer
files before opening the deck. Historical Figures/figures spelling differences
are resolved when unambiguous for case-sensitive computers.

The launcher reuses this checkout's running service or starts it in the background.
After restarting the computer, run the command again. It installs no startup agent.
Logs are in `outputs/presentation_viewers.log`. Server-only command, for an
installed research environment: `neuristor analyze serve-presentation --background`.

The heatmap button on slide 17 opens fixed-scale viewer `4325d1`, whose GIFs
remain in original bundle `eaf6ab`. Both redirects preserve the original relative
media URLs. The heatmap example is the first row of saved `4325d1/heatmaps.png`,
cropped without changing its data or colors.

Verification: the diagnostic PDF's HTTP connection link was clicked in Preview;
Preview displayed an Open in Safari confirmation, and Safari displayed the live
service identity. Both production PDF URI destinations are verified. The original
HTML viewers were not interacted with by browser automation: earlier local-file
browser access was blocked, so no complete heatmap/GIF browser interaction is
claimed. Synthetic HTTP tests cover both redirects, media delivery, read-only
methods and rejection of private paths, symlinks and foreign Host headers.

Older conditional parameter-estimation derivations and obsolete backup slides
remain excluded. Source/result files from prior revisions are preserved in Git
history and ignored local `outputs/presentation_revision/` snapshots.

## Parameters and experimental inputs

The last slide uses the separately preserved historical diagnostic
`presets/alternating_amplitude_relaxed_20260907.json`. The same vector drives
22 measured-current inputs at two displayed timesteps (44 GIFs), with native
numerical checks at three timesteps. Its passive thermal scale is
`Cth/Se = 11.9087 ns`. Along the central displayed replay, `C R(T,H)` spans
`0.1334–11.9846 ns`; the fully metallic limit is `0.1245 ns`. Shallow and deep
cycles follow different native R(T) paths. RC recharge, thermal recovery and
hysteresis memory provide a working explanation, but the longer constant-current
state remains timestep-sensitive. No stable period doubling or calibration is
claimed. The final slide button uses `/alternating-gifs` on the same loopback
viewer service. [Detailed recovered finding](../docs/ALTERNATING_AMPLITUDES_20261025.md).

The six major-loop parameters are fitted to 264 supplied numerical measurements
from sample `100425 chip1 gap3`, bundle `0849a9`.
Ambient 314.4 K is adopted from the midpoint of 41.1 and 41.4 °C reported on pages
3 and 5 of Gildor et al.,
[*Harnessing the VO2 Phase Transition for Automatic Gain Control in
Transimpedance Amplifiers*, arXiv:2604.04594v1](https://arxiv.org/abs/2604.04594v1).
The paper is cited, not bundled.

Gamma, Se, C and Cth remain unknown specimen constants. Values selected in the
new searches are diagnostic attempts, not established physical calibrations.
Both new recipes fix the six major-loop constants and T0; optional noise is zero.
Cth/Se is an internal search coordinate, not a fifth independent unknown.
Input currents and initial conditions are operating conditions. Implementation
scale factors are absorbed into effective Rm and w; deadbands/bounds are numerical
settings. No scientific model equation was changed.

Plots and example titles use measured baseline-corrected input current averaged
over 50–250 ns. Filenames containing mV are acquisition identifiers only. Actual
simulation input is the entire measured current waveform including prehistory.
Examples are selected by current rank, before inspecting errors: approximately
228.3, 418.9 and 606.3 µA. Each appears for both objectives without phase alignment.

## Historical evidence

All bundles are under `public_jobs/`; shortened suffixes printed on slides are
unique. Full ancestry is in `docs/RESEARCH_HISTORY.md` and the method note above.

- `0849a9`: measured R(T) fit.
- `8f12d6`: original shared waveform/feature search.
- `6b5d8f`: joint static/dynamic search.
- `694804`: expanded persistence-aware search.
- `32d91d`: scaled extrema pilot.
- `96b944`: squared relative extrema pilot and common starting vectors.

Scientific status remains in `docs/RESEARCH_HANDOFF.md`. No numerical source is a
slide or screenshot; plots reuse saved samples and metrics from reviewed bundles.

## Build and review

Compile `main.tex` with Tectonic or the LaTeX compile skill, directing temporary
files to `outputs/`. Inspect the new slides and tables, then copy the verified PDF
to `main.pdf`. Expected page count: 21. Compile after the comparison result source
has been generated and its figure bundle published locally.

The optimizer follow-up and older loss formulations are excluded at the user's
request; their numerical evidence remains archived. The deck now has 21 slides.
The full parameter/GIF explorer is documented separately:
[controlled trends and scaling](../docs/PARAMETER_EXPLORATION_TRENDS_20261020.md).

## Mica (M2, 1600 nm)

Slides 19–20 introduce the second specimen studied by Amir. The user explicitly
adopted **298 K** on 21 October, matching both parameter explorations; this is
not the earlier specimen replay's adopted 314.4 K. Current is fixed at 500 µA,
gamma at 0.956269682, and C/Cth/Se remain exploration axes.

The R(T) plot uses all 374 saved measured/model samples from initial review
`c7387b`, without refitting. Its six parameters are copied from that review's
resistance preset. The approximate fit's log10 RMSE is 0.08065244; structured
transition errors and the failed normal 0.05 cutoff are stated.

The heatmap uses the exact 121-point Cth=1.5 pJ/K slice of the **Dense onset**
grid in expanded review `226099`. Only computed sustained-cycle f/Vpp values
are colored; white crosses mark temperature-range failures. Gray cells are
missing sustained-cycle values, not physical nonoscillation boundaries.
The two expanded grids add 1,843 entries to the original 773 (2,616 including
overlaps). All accepted-point counts remain zero. No new simulation or fit
was performed for the presentation.

The `/mica-map` button resolves to the newest **completed, locally published**
M2 1600 nm review with a `parameter_maps.html` file. In-progress runs are ignored;
the route updates without editing the PDF when another review is published.
The local server was restarted to load this route. Its redirect points to
`20261021_064103_m2-1600nm-expanded-r-t-exploration-review_226099` at this revision.
Synthetic HTTP tests verify the completed-only update behavior. Live redirect
headers and PDF annotations are checked; full HTML interaction is not claimed.

## Sparse controlled follow-up (first specimen)

Slide 18 reuses completed trend-validation bundle `3599fb`. Three reference
vectors are tested with shared capacitance multipliers 0.5, 1 and 2, at constant
500 µA, 298 K, fixed Se for each vector and gamma=0.956269682. The deterministic
model predicts V_s(t)=V(t/s), T_s(t)=T(t/s), f_s=f/s and unchanged physical Vpp.
The figure plots the six co-scaled-timestep comparisons and their three centers,
normalized to their own saved reference cases. All 12 final scale checks pass
2%; the figure does not claim to show all 12. The small (<0.9%) extracted Vpp
differences arise from the fixed-time feature operator. This is timing/path
scaling, not proof of pointwise convergence at fixed physical timestep.

Figure sources and hashes are in
`assets/controlled_capacitance_scaling_provenance.json`; plotted points are saved
under `outputs/presentation_sparse_exploration/plotted_scaling_points.csv`.
No simulation or fit was repeated. Methods and wider conditional trends:
[`FRESH_PARAMETER_TRENDS_20261020`](../docs/FRESH_PARAMETER_TRENDS_20261020.md).

## Process chart

The final slide (`process_flow.tex`) summarizes the workflow requested by Amir:
measured R(T), current and voltage are separate inputs; least squares minimizes
log-resistance RMSE to fit the static law; measured current and the specimen law
feed the electrothermal simulator; a frequency or frequency-plus-Vpp loss compares
the simulated and measured response; Cth/C/Se are updated and the simulation is
repeated. Passing candidates still need validation across measured inputs.
Gamma must be specified independently of the static R(T) fit, and earlier
frequency/amplitude trials also searched it. The asterisk records the separate
Cth/C/Se domain exploration. This is a workflow summary, not a new calibration.
