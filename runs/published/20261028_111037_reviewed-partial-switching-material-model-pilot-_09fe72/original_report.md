# Partial-switching material-model pilot

This finite pilot completed 119 screening cases and 7 full-current verification cases.
The prespecified target currents are [266.63381756349264, 418.8568677552722, 606.2799905984709] µA; controls
are [189.78481489725564, 907.0417701985936] µA. Historical records and seeds were already seen;
there is no claim of blind validation. No calibration is automatically adopted.

## Models and physical accounting

- Baseline calls the unchanged authoritative ideal-current model.
- Warm parallel regions use G=w/Rfit(Tprimary,Hprimary)+(1-w)/Rfit(Tsecondary,Hsecondary).
  Each receives its own V²G Joule heat. Both temperatures evolve with bath cooling and mutual exchange.
- A series VO₂ bottleneck uses R=w Rfit(Tprimary,Hprimary)+(1-w) Rfit(Tsecondary,Hsecondary).
  The regions carry the same device-branch current and receive Ibranch²Rregion heat. Their voltage drops
  sum to the measured device voltage. This introduces no external series resistor or current change.
- Finite switching time relaxes the semiconducting weight toward the existing hysteresis target:
  tau dx/dt=g_hyst(T,H)-x; R=Rm+R0 exp(Ea_over_k/T)x. Anchors follow the original target.
  Tau=0 delegates exactly to the original model. No spatial mechanism is combined with phase lag.

Both two-region topologies recover the measured uniform-temperature R(T). The reference Cth and Se
budgets are split by w and 1-w, then secondary capacity/cooling ratios introduce asymmetry.
These budget relations are diagnostic assumptions, not measured active volumes. Positive mutual
conductance transfers K(Tprimary-Tsecondary), cancelling between nodes. Each frozen-power thermal
step uses the exact two-node matrix exponential; electrical and phase substeps are exact with frozen
coefficients. Coupling between subproblems still requires timestep refinement.

## Search, objective and checks

The declared grid uses shared historical anchors. Four one-coordinate reference parameters
(C, Se, tau_th and gamma) then receive the same two local probes in every arm. Static R(T) and T0
remain fixed. Cases/ contains exact specifications, realized inputs via the verification results,
features and diagnostics. No parameter is fitted separately for each current.

Ranking prioritizes physical temperatures/proximity, valid target cycles and excessive control
oscillations before a finite diagnostic feature score. That score combines relative frequency/Vpp,
mean-voltage errors normalized by 20.0 mV and mean peak/trough squared errors at
the same scale. Missing cycles do not receive an accepted scientific loss. Selected controls are
flagged for sustained amplitude above max(10.0 mV,
2.0 times their measured Vpp).

Winners and exact-vector mechanism-off ablations replay all 22 original inputs with full measured
prehistory at [0.025, 0.0125, 0.00625, 0.003125] ns. The original full-current observable/physical gates apply,
plus target peak/trough RMSE ≤ 20.0 mV. Long controls last
2000.0 ns at the three pilot mean currents and three finest timesteps.
Native finest histories at the five pilot/control records retain branch currents, powers and both
temperatures where present. Both temperatures and hysteresis slopes are screened.

## Results

See comparison.csv and the offline viewer. Fully successful candidates: [].
Pilot or conditional improvement must not be called an adopted calibration. A lower diagnostic score
is distinct from full target, quiet-control, temperature, shape, timestep and long-control acceptance.
Zero-amplitude quenched trajectories are never an oscillatory fit.

## Limits and next decision

This is a sparse mechanism pilot with a small shared refinement, not an exhaustive parameter search.
The real TIA circuit boundary and readout transfer remain unresolved. Only a mechanism that improves
amplitude, mean and cycle shape across currents, survives finer steps and keeps controls physical
merits a larger follow-up. Per-arm selections can differ in nuisance coordinates; matched baseline
ablations expose that confounding. No free f(t), per-current scaling, gain or time shift is introduced.
Original input files, complete source, execution recipe, all case tables and hashes are preserved.
