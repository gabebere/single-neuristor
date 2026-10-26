# Expanded search and interactive laboratory

The expanded fit improves some errors but does not resolve the discrepancy.
Ambient temperature **was already free**, bounded at 308–322 K. The new search
keeps these physically interpretable bounds and expands optimizer coverage:
24 candidates, seven differential-evolution generations and at most 36 Powell
calls per static-weight setting, seeded with both previous joint candidates and
three historical vectors. It uses 0.025 ns during search (previously 0.05 ns).
There were 456 calls, 400 distinct candidates and 399 simulated search candidates;
search time was 672.6 seconds. The shared cache avoids duplicate simulations.

The loss adds a smooth persistence deficit weighted by 12 and a missing-frequency
penalty of 1, retaining the existing mean/amplitude/frequency and static terms.
It penalizes insufficient window amplitude, retention and coherence on measured
oscillators. The smooth frequency gate retains a small regularization cost even
for a self-match; this is an engineering objective, not a likelihood.
Nine currents train the model; thirteen are excluded from this optimization but
were inspected in earlier research. All seven candidate vectors were replayed
on all 22 currents at 0.025, 0.0125 and 0.00625 ns.

| Candidate | R(T) log10 RMSE | Late mean voltage RMSE, all 22 (mV) | Sustained recovered / 11 | Amplitude MAE on same 11 (mV) |
|---|---:|---:|---:|---:|
| Original specimen | 0.0366 | 45.3 | 0 | 44.3 |
| Prior anchored | 0.0426 | 50.9 | 9 | 140.1 |
| Prior joint, stronger static weight | 0.0495 | 50.5 | 4 | 62.0 |
| Expanded, stronger static weight | 0.0489 | 44.6 | 7 | 127.9 |
| Expanded, relaxed static weight | 0.1473 | 39.8 | 6 | 130.6 |

Values use the finest step. No listed candidate predicts false oscillations at
that step. Amplitude is late periodic peak-to-peak amplitude; mean-voltage error
is not point-by-point waveform RMSE. The expanded static-constrained candidate
recovers eight oscillators at the two coarser steps but seven at the finest;
its classification is therefore not fully converged. The relaxed candidate
recovers six at all three steps. Lower mean error trades against amplitude,
frequency, static fit and oscillation-window errors; no new calibration is adopted.

The stronger-static result has T0=313.09 K, C=4.650 pF, Se=4.134 µW/K and
Cth=0.04302 pJ/K. The relaxed result has T0=315.00 K and C=12.334 pF.
Neither pushes T0 to its bounds, so widening ambient bounds is not the evident
next remedy. Better next searches should retain several nondominated candidates,
refine near oscillation boundaries and compare measurement/thermal model variants
only after circuit calibration. Adding unconstrained parameters can conceal a
wrong observation model. Same-device static R(T) does not establish uniform
heating during a driven filamentary transition.

Evidence: [expanded bundle](../runs/published/20260921_210646_expanded-persistence-aware-joint-inference_694804),
particularly `parameters.csv`, `summary.csv`, `verification.csv` and its source
snapshot. The recipe is `experiments/current/specimen_joint_expanded.toml`.
Raw scores should not be compared to the previous loss: the bundle re-evaluates
historical candidates with the new loss for a fair comparison.

## Replay and reproduction

The Streamlit controls described in the original September workflow were retired
on 26 October 2026. Edit the resistance, electrical, thermal and initial values
in `experiments/current/specimen_lab_replay.toml`, or pass explicit `--set`
overrides to the CLI. The measured input histories remain fixed experimental
inputs; the exported HTML current slider selects a saved record.

Each replay writes a new bundle with traces, metrics, standalone HTML, GIFs and
a reproducible recipe. Simulated R(t) is not an experimental V/I estimate.
Repeat promising values at smaller timesteps. Browse completed runs with
`neuristor runs manage`.

CLI reproduction:

```bash
neuristor analyze replay-lab --config experiments/current/specimen_lab_replay.toml
neuristor analyze fit-joint --config experiments/current/specimen_joint_expanded.toml
```
