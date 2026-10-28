# Reviewed material-model pilot — 28 October 2026

**The three optional mechanisms did not produce a validated fit in this bounded pilot.**
No selected candidate reproduces the decreasing-amplitude trend while retaining all three
pilot target oscillations, and none passes the complete all-current acceptance criteria.
No calibration is adopted. This sparse comparison does not rule out spatial localization,
finite phase dynamics or a different thermal/geometry mapping.

## Coverage and final comparisons

All 119 screen cases completed: 87 declared
grid points and 32 shared one-coordinate probes. Seven retained
cases (four candidates plus three exact-reference-vector mechanism-off controls) replay all
22 inputs at four timesteps. Four 2 µs constant-current cases use three pilot currents and
the three finest timesteps. Elapsed numerical time: 14.12 minutes.

Final values use 0.003125 ns. Frequency MAPE uses the stated defined denominator; Vpp
uses all 11 targets, and mean-voltage RMSE all 22 records. The configured temperature
domain is 284.8545–375.1625 K. All rows fail full waveform convergence and shape acceptance.

| Model | Valid targets | Defined frequencies | Frequency MAPE | Vpp MAPE | Mean RMSE (mV) | Max T (K) |
|---|---:|---:|---:|---:|---:|---:|
| Original model | 7/11 | 7/11 | 32.83% | 559.1% | 37.97 | 363.52 |
| Warm parallel regions | 10/11 | 10/11 | 15.06% | 1639.0% | 64.02 | 415.79 |
| Series VO₂ bottleneck | 4/11 | 4/11 | 21.96% | 522.0% | 45.80 | 367.55 |
| Finite switching time | 9/11 | 9/11 | 36.67% | 1087.9% | 43.47 | 394.65 |


The series bottleneck has smaller Vpp errors partly because it loses oscillations:
at the pilot currents, predicted Vpp is 0, 312.6 and 0 mV versus measured 101.8, 34.8
and 9.6 mV. It retains only four of 11 target records. This is quenching, not a fit.
Warm parallel regions retain 10 targets and frequency MAPE ~15.1%, but their swings
remain too large and temperatures exceed calibration. A 6 ns finite phase response
reduces some swings but slows the oscillations, retains nine targets and also exceeds
the temperature domain. These are conditional trade-offs, not accepted improvements.

The pilot's baseline is a different bounded selection from the previous overnight
f=1 diagnostic (10/11 targets; frequency MAPE 16.59%; Vpp MAPE 1437.28%; mean RMSE
35.52 mV). Screening here prioritizes physical trajectories on five records at
0.025 ns before cycles and feature accuracy; the selected baseline sacrifices cycles.
Do not replace the earlier retained reference with this seven-target pilot selection
or interpret the reduced Vpp error as evidence of a better overall fit. Coarse exclusions
may change under refinement; the pilot is not a complete scan of fine-step feasibility.

## Physical model and comparison limits

The original ideal-current solver is unchanged. The spatial variants separately test
warm parallel paths and a series VO₂ bottleneck/access path, with correct branch-current,
internal-voltage and heat allocation. Uniform-temperature resistances recover the same
measured R(T). Two passive thermal nodes exchange equal/opposite heat. Finite phase lag
is a separate phenomenological state relaxing toward the original hysteresis target;
no separate latent-heat model is added. The zero-lag limit calls the exact original solver.

The actual weights tested are 0.25, 0.5, 0.75, secondary capacity ratios 10/100 and
coupling ratios 0/1; no continuous geometry optimization was performed. Phase times
are 0.2, 0.6, 2, 6 ns. The retained lag reaches the upper trial value; this does not
identify a phase time. Static R(T) and ambient are fixed. Reference-budget partitions,
secondary thermal ratios and local constitutive-law transfer are diagnostic hypotheses.
No parameter is fitted separately for each current. All records and historical anchors
were seen before, so there is no blind validation claim. See original_report.md for
the full equations, feature score and long-control protocol.

## Integrity and independent checks

Every original artifact hash (520) and source-file hash passes verification.
All 20 saved finest-step native histories reconstruct the saved measured-grid voltage
features. Joule-heat partitions agree with V²/R within maximum relative error
2.38e-07; saved parallel currents and series
voltage/current identities pass independently. Both regional temperatures are included
in physical-domain checks. Independent analytic RC/thermal/phase controls and 116
pre-change batched-waveform arrays verify the optional implementation and preservation
of existing output. Software checks pass 246 tests and validation of 51 recipes.

## Next step

This pilot supplies no robust reason to commit to one of these material extensions.
Obtain the actual TIA circuit/probe-node and conversion/feedback-current information
before another large fit. The previous fixed-gain/offset audit already failed to
explain the sweep; a circuit hypothesis must represent actual feedback/load dynamics.
If continuing with current data, use narrowly justified geometry or minor-loop/phase
measurements to constrain an extension rather than adding freely varying f(t).
Finer-step checks and all-current controls remain mandatory.

## Preservation

Original metrics/configuration, input data, full executable source, all 119 case
tables/specifications, all seven replay/ablation tables, long-control tables, figures
and offline viewer are byte-preserved separately from this review. The active manifest
contains only files actually included. Twenty bulky native NPZ arrays stay in the
full local numerical bundle; their hashes, sizes and exclusion are recorded in
full_local_inventory.json. Those omitted bytes are not backed up to GitHub by this
compact review. Source numerical commit: 0e39b5ee587ddf18e4abd9fc654b123427d31986.
The source and review bundles are distinct, and no existing public bundle was edited.
