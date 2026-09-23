# Cooling-hook diagnosis — 22 September 2026

The cooling hook is a reproducible property of the fitted minor-loop law. It is not a GIF artifact, reversed branch sign, or resistance-unit conversion error. The unrestricted gamma search admitted nonmonotone effective-temperature mappings. This is a missing admissibility constraint in the inference setup, not evidence that the local implementation mistranscribed Yuanhang's code. No physics or saved fit was changed in this diagnosis.

## Evidence and scope

Selected fit: `search_1` in `runs/20260922_175152_background-settled-oscillations-for-original-lab_f426de`. Replay: `runs/20260922_204643_steady-oscillation-search-best-search-1-finest-t_ac96b8`, timestep 0.00625 ns. Detailed inspection uses original label 500 (measured plateau 380.749 µA). One settled cycle spans 157.26–178.34 ns, with temperature 331.288–339.699 K.

The internal direction correctly switches between +1 and −1. The cooling path contains an approximately 4.97 Ω decrease in R as T falls through 338.239–337.900 K. Replaying the identical fine-grid temperature history through the archived upstream VO2 class, with the fitted parameters, agrees within 0.0000611 Ω across the full history. Temperatures stay within both implementations' clamp intervals. Five existing hysteresis/reversal tests pass. A separate double-precision derivative check confirms the hook without a time integrator.

Diagnostic files: `outputs/cooling_audit/`: `500_label_internal_state.csv`, `analytic_derivative.csv`, `diagnosis.json`, `gamma_slope_scan.csv`, `finite_difference_check.json`, `heating_cooling_check.png`, `proximity_slope.png`. These are exploratory diagnostics, not a published run bundle. Scripts there reproduce the inspection using existing physics; no new workflow was added.

## Mathematical cause

Write A = Tpr and E = Ea/kB (the repository's Ea_over_k), with the reversal anchor fixed between events:

```
x = (T − Tr)/(A + ε), ε = 10^-6 K
P(x) = 1/2 [1 − sin(γx)] [1 + tanh(π² − 2πx)]
Teff = T + A P(x)
g = 1/2 + 1/2 tanh{β[δw/2 + Tc − Teff]}
R = R0 exp(E/T) g + Rm
```

Here g is the insulating contribution in this implementation: increasing g raises R. The exact slope between reversals is

```
q = dTeff/dT = 1 + A/(A + ε) P′(x)
P′(x) = −γ/2 cos(γx)[1 + tanh(z)] − π[1 − sin(γx)]sech²(z)
z = π² − 2πx
dg/dT = −2βg(1−g)q
dR/dT = R0 exp(E/T)[−2βg(1−g)q − E g/T²]
```

Positive q gives the ordinary phase-response direction. Negative q reverses that contribution; the full R derivative must still include the activated-resistance term. In the observed hook the full derivative is positive, so decreasing T decreases R.

At 158.98125 ns: T = 338.07434 K, δ = −1, Tr = 339.68677 K, A = −1.031555 K, x = 1.56310, g = 0.0198529. The calculated q is −1.6893 and dR/dT = +24.3875 Ω/K. Double-precision symmetric differences with ΔT = 0.01, 0.001 and 0.0001 K yield +24.3395, +24.3873 and +24.3878 Ω/K. Thus the effect does not disappear with smaller differentiation steps or higher precision.

The sharp tanh cutoff is centered at x = π/2. At that point, ignoring ε,

```
q = 1 − γ/2 cos(γπ/2) − π[1 − sin(γπ/2)].
```

With γ near 1, the sine factor nearly vanishes at the cutoff. With the new γ = 0.102034, it does not; the proximity correction falls too steeply and Teff reverses locally. The earlier favorite γ = 0.181050 is affected too. A numerical scan over x ∈ [0,8] gives minimum q of −1.69155, −1.35040 and +0.04373 for γ = 0.102034, 0.181050 and the original reference 0.956270, respectively. This scan concerns the kernel, not a new self-consistent dynamical fit.

Small gamma also gives q near 1−γ immediately after reversal. This allows a steep return trajectory and helps explain why the two sides resemble one another. Narrow minor loops alone are not a bug; the local backward phase response is the specific issue established here.

## Why the optimization did not reject it

The recipe permitted gamma from 0.03 to 4. Gamma does not enter an unreversed major branch, so preserving major-loop R(T) fit error cannot constrain it. The loss assessed frequency, amplitude, retention and major-loop fit, but did not screen minor-loop derivative sign or loop ordering. The optimizer found a lower-loss parameter set within those stated bounds; that does not establish its physical admissibility. I should have included this screening before interpreting the broad search as physical parameter estimation.

This identifies an explanation for the hook, not proof that the hook causes the whole voltage discrepancy. The discrepancy also involves mean voltage, circuit/readout uncertainty and oscillation amplitude. The new fit remains an exploratory numerical fit and must not be adopted as a physical calibration.

## Corrective direction

1. Add an explicit minor-loop admissibility screen before future optimization. Check q (with the actual epsilon factor), full R slope, and controlled heating/cooling paths, including nested reversals, rather than trusting static R(T) alone.
2. As a bounded control, use the reference gamma or restrict it to a screened range and repeat the same objective. A dense positive-x scan suggests approximately 0.543 ≤ gamma ≤ 1 as a candidate range, not an experimental confidence interval or proof covering all reversal states. Require margins and actual-state checks, including small A and negative x, before encoding bounds.
3. Refit shared parameters under that restriction and compare the same nine targets and timestep checks. Changing gamma in isolation will change the oscillations; do not claim the prior accuracy survives.
4. If admissible LLP cannot fit the data, evaluate a separately versioned monotone memory law and measure same-device minor loops. Do not clip R or force an instantaneous jump to the cooling major branch: both would silently replace the model.

No new large optimization was launched, and no GIF or completed bundle was overwritten.

## Primary sources

Zhang et al., *Collective dynamics and long-range order in thermal neuristor networks*, Nature Communications 15, 6986 (2024), Methods Eq. (3): https://doi.org/10.1038/s41467-024-51254-4 . Accessible author-hosted PDF: https://escholarship.org/content/qt0tw9628s/qt0tw9628s_noSplash_252a39ee6f42c1bbc1d23df15e2b599e.pdf . The sine/tanh proximity law is given there. The derivative analysis and parameter admissibility diagnosis above are our own calculations, not claims made by the paper.

Archived executable reference: `references/yuanhangzhang98-collective_dynamics_neuristor-217d4f0/model.py`, functions P, Tpr_func, g and reversal. Its R output is kΩ; the comparison explicitly converts it to Ω.
