# VO2 fitting journey: eight-slide LaTeX presentation

`VO2_Fitting_Journey.tex` is the editable 16:9 Beamer presentation;
`VO2_Fitting_Journey.pdf` is its reviewed PDF snapshot. It summarizes the research
through 14 September 2026 without running new simulations or changing the report.

The narrative separates the target, intervention, result and remaining limitation.
Historical peak-count labels are explicitly distinguished from the later persistence
audit. No diagnostic parameter vector is presented as a new physical calibration.

## Slide-to-evidence map

All run directories below are relative to the repository's `public_jobs/` folder.
Short suffixes printed on slides resolve to the full IDs here.

| Slide | Question and evidence |
|---|---|
| 1 | Target: one vector across 22 measured current inputs, including 11 experimental oscillators. See frozen comparison below. |
| 2 | Mechanism control: `20260817_100102_current-step-with-a-nonzero-metallic-voltage-val_6765e0`; sample R(T): `20260816_125905_sample-r-t-major-loop-hysteresis-fit_0849a9`; conductance: `20260817_153807_environmental-thermal-conductance-estimate_761640`; thermal capacitance: `20260828_112314_thermal-capacitance-estimate-with-conservative-0_aa2469`. Electrical bound and ambient interpretation: manuscript Sections 3–6. |
| 3 | Frozen comparison: `20260829_100718_specimen-model-prediction-versus-measured-curren_eefab7`; original constrained/relaxed inference: `20260829_105704_global-specimen-parameter-inference-from-all-cur_8f12d6`. Objective values are compared only within this objective definition. |
| 4 | Oscillation-priority: `20260829_160147_oscillation-priority-global-specimen-parameter-i_ac1c5e`; amplitude-priority: `20260829_210212_amplitude-tuned-oscillation-priority-specimen-in_85526c`; physics-anchored: `20260907_125315_physics-anchored-specimen-waveform-inference_717797`. Use reports, trace metrics and manuscript Section 10. The first final evaluation is 0.05 ns; the latter two are 0.025 ns. |
| 5 | Corrected persistence interpretation: `20260914_073330_sustained-oscillation-audit-and-three-current-ma_26ff13`. The plotted samples use the finest 0.00625 ns replay; quoted first/last raw voltage spans use the archived 0.025 ns window audit, explicitly labeled. |
| 6 | Same audit bundle: 126 conditional grid points, candidate rankings, all-current verification and timestep checks. The amplitude comparison is the robust 5–95% voltage span in 150–250 ns, not the historical amplitude MAE on slide 4. |
| 7 | Conditional inverse test: `20260914_074731_conditional-hysteresis-reconstruction-from-measu_fa4b66`, central case, 0.25 ns replay, means over 150–250 ns. Currents are late preprocessed means and differ slightly from the earlier run labels. |
| 8 | Synthesis of these diagnostics. Circuit/readout and phase-dynamics explanations are hypotheses; this presentation does not establish a unique cause of mismatch. |

The model's original publication is Zhang et al., *Collective dynamics and long-range
order in thermal neuristor networks*, Nature Communications 15, 6986 (2024),
doi:10.1038/s41467-024-51254-4. Experimental context is Gildor et al., *Harnessing
the VO2 Phase Transition for Automatic Gain Control in Transimpedance Amplifiers*,
arXiv:2604.04594v1 (2026). Numerical experimental samples come from the supplied
exports, not digitized paper images.

## Plot provenance

Slide 4 recomputes amplitude MAE consistently on the **same 11 measured oscillatory
records**: select `fit_mode == relaxed` and `measured_oscillation == True` from each
bundle's `trace_metrics.csv`, then average
`abs(predicted_vpp_mV - measured_vpp_mV)`. Values are 289.6297, 81.1798 and
212.5128 mV. The older manuscript values 280.1 and 88.5 mV were restricted to
jointly detected experimental/model oscillators, so their subsets differ; those
numbers are not used in this presentation's common-subset comparison.

`data/current_606uA.csv` is a lossless selection/pivot of the audit's
`candidate_traces.csv` for source-setting label 800 mV, `dt_ns == 0.00625`,
and `time_ns >= 50`. The label identifies the record; the actual measured current
is about 606 µA. Columns retain each candidate prediction and one copy of the
identical measured voltage. No filtering, smoothing or resimulation was added.

- Source SHA-256: `c7d8145a793bdec5e9480c69b610b30a16398bf0c6cafe8a7637c1ae9459a92b`
- Presentation CSV SHA-256: `23e92f21f6a090b2c28286f88cc87bac9dde59cdd03a043d93d08feecf6213e1`

## Build

From this directory with Tectonic available:

```bash
tectonic -X compile --keep-logs --outdir ../../../../output/pdf VO2_Fitting_Journey.tex
```

Alternatively use the installed LaTeX compile skill with this file's absolute path.
The document requires Beamer, fontspec, TeX Gyre Heros and PGFPlots; Tectonic may
download missing packages on the first build. It has exactly eight frames, no
overlays and no extra bibliography or appendix slides. Source references are kept
in the slide footers and this README to preserve readability.

After editing, confirm eight PDF pages, inspect all rendered pages for overflow,
and refresh the PDF snapshot in this folder. The full research manuscript remains
at `../main.tex` and `../Simulations_for_VO2_AGC.pdf`.
