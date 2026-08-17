# Public run archive

This directory is intentionally tracked by Git. It contains reviewed evidence that is
discoverable in the archive dashboard after cloning the repository.

New workflows write exploratory bundles under ignored `runs/`. Review a bundle's
report, metrics, configuration, and size before publishing it:

```bash
neuristor runs show RUN_ID
neuristor runs publish RUN_ID
git add public_jobs/RUN_ID
git commit -m "Archive <description>"
git push
```

Publishing copies a bundle; it never overwrites an existing ID and does not silently
commit or push. Completed public bundles are immutable. Corrections should be new runs
whose reports explain the changed assumption or implementation.

## Historical records

Current `run.json` evidence, generated from clean commit `af4638c`:

- `20260810_170502_current-step-with-a-nonzero-metallic-voltage-val_b05d39`:
  600 uA Yuanhang-centered oscillation with a 0.906 V measured valley and 0.772 V
  metallic fixed-point floor.
- `20260810_170512_yuanhang-voltage-driven-relaxation-oscillator_b997e9`:
  upstream-style voltage-source oscillator, including a validity warning where its
  temperature slightly exceeds the configured R(T) range.

Numerical laboratory evidence generated from clean commit `bba2fd8`:

- `20260817_134254_measured-laboratory-current-sweep_a45254`: 22 professor-supplied
  current/voltage traces, numerical summary, and the measured 41.7--62.5 MHz
  operating window.
- `20260817_134254_lab-current-trace-parameter-estimates_4223e0`: 19.8 pF median
  cold-edge capacitance estimate, ambient/conductance and thermal-capacitance
  scenarios, and effective-resistance/voltage-floor comparisons.
Historical `job.json` evidence:

- `20260707_145140_paper_frequency_f881d2`: fitted-specimen paper-frequency analog,
  including current sweep, spectra, energy cycle, pump-probe recovery, and animations.
- `20260707_145140_sample_scale_89ffb4`: fitted-specimen sample-scale control and
  operating-window sweep.
- `20260707_154737_390uA_valley_ff1d22`: explicit 390 uA nonzero-valley trial using
  a raised metallic resistance; it is a diagnostic trial, not a fitted solution.
- `20260709_115815_published_table_current_3c2df4`: Yuanhang published-table values
  translated to the ideal-current circuit, including deterministic/noisy controls and
  timestep checks.

Those `job.json` records predate the August 2026 endpoint-fidelity and exact-substep
corrections. They remain for provenance, not new quantitative claims. The dashboard
normalizes them alongside current `run.json` bundles.

The former screenshot-derived laboratory bundle was removed after the original
numerical oscilloscope exports became available. Git history preserves it, but it is
not current evidence.
