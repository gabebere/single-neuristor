# Professor paper simulation package

Generated with the hardcoded Yuanhang ideal-current-source model.

## What was run

- Paper-frequency analog: one VO2 domain, ideal current source, fitted sample R(T), `C = 4 pF`, `C_th = 1 pJ/K`, `S_e = 0.10 mW/K`, 300 ns current pulse from 150 ns to 450 ns.
- Validated sample-scale control: same equations and R(T), `C = 25.930953 pF`, `C_th = 5 pJ/K`, `S_e = 0.10 mW/K`, same pulse timing.
- No dynamic/quasistatic/multidomain modes were used.

## Main result

- Paper-frequency analog oscillatory window: 1400-2100 uA, 35.7-60.7 MHz.
- Paper-frequency analog full-cycle energy: 36.3-53.9 pJ. This is larger than the paper's reported 1-2 pJ scale, so the current package supports the oscillatory mechanism and frequency phenomenology, not an absolute energy calibration.
- Validated sample-scale oscillatory window: 1600-2800 uA, 7.1-14.3 MHz.

The paper-frequency analog matches the paper's frequency scale, but not its absolute current scale.
That means the clean presentation is a normalized/phenomenological comparison: same low-current, oscillatory, and high-current regimes, with the same hidden R(T) relaxation mechanism.

## App history jobs

- Paper-frequency analog job: `20260707_145140_paper_frequency_f881d2`
- Validated sample-scale job: `20260707_145140_sample_scale_89ffb4`

Open Streamlit History, filter to Public, and open those jobs.
