# Hysteresis Implementation Audit

Last audited: 2026-06-20

This document is the provenance record for hysteresis branch handling in this
repository. Read it before changing reversal detection, fitting an R(T) path,
or promoting an oscillatory domain as a physical result.

## Source of truth

The upstream implementation is vendored at:

`references/yuanhangzhang98-collective_dynamics_neuristor-217d4f0/model.py`

The local implementation is:

`src/neuristor/model.py::HysteresisArray`

The ideal-current integration is:

`src/neuristor/current_drive_sim.py`

The electrical and thermal current-source equations were not changed during
the reversal-detector work:

```text
C dV/dt = I_in - V/R(T,H)
C_th dT/dt = V^2/R(T,H) - S_e (T-T0)
```

Only the discrete hysteresis state update was changed.

## Branch convention

`delta = +1` is the insulating/heating major branch. Its transition is shifted
to higher temperature. `delta = -1` is the metallic/cooling major branch. Its
transition is shifted to lower temperature.

`start_branch` is an initial-state hint, not a permanent constraint. Once the
temperature moves far enough in the opposite direction, the detector changes
`delta`. A measured R(T) file may store `start_branch="metal"` because its first
samples are cooling, while a pulse simulation of a cold device may correctly
start on `insulator` using the same fitted resistance parameters.

## Supported reversal implementation

The runtime has one hysteresis implementation: the faithful Yuanhang Zhang
accumulated-displacement detector. There is no reversal-mode setting in
`CurrentDriveParams`, `HysteresisArray`, or the Streamlit interface.

With the Torch backend available, the implementation is a faithful float32
port of Yuanhang Zhang's code. `T_last` is an accepted anchor. Sub-threshold motion does not update it. Once accumulated
displacement exceeds 0.01 K, direction is evaluated and `T_last` is advanced.
If direction changed, the sampled detection point is stored as the reversal
point.

This is not a single-timestep threshold. It is an accumulated-displacement
deadband. The local float32 audit matches the vendored upstream trace to within
0.001 ohm on the standard major/minor-loop path. The NumPy-only fallback keeps
the same equations but is not bit-faithful near saturated minor loops because
NumPy and Torch float32 transcendental arithmetic can diverge there.

## Removed implementations

An early local port incorrectly updated `T_last` on every call, including
sub-threshold calls. The 0.01 K condition therefore became a one-timestep
condition and changed when `dt` changed. Several saved apparent oscillation
jobs were generated with this bug.

The later turning-point extension tracked the furthest temperature reached on the active
branch and stored that extremum as the reversal point after a 0.01 K excursion.
It was timestep robust, but it was not Yuanhang's algorithm. Because minor-loop
state is initialized at the extremum rather than the sampled detection point,
minor-loop geometry can differ strongly from upstream on aggressive reversal
paths (up to about 70% resistance difference in the audit path).

Both alternate implementations were removed from executable code on
2026-06-21. Their names remain only in historical artifact metadata and this
audit. Loading an old job discards its obsolete mode field and uses the sole
reference implementation; existing rendered outputs are not rewritten.

A 2026-06-20 reference-mode resweep found sustained late-window oscillations
through 2.47 mA and a fixed point at 2.48 mA for the validated circuit/thermal
parameter set. The earlier 2.747 mA upper boundary is specific to
`turning_point`. The lower edge remains near 1.27--1.30 mA, depending on the
minimum number of cycles required in the finite analysis window.

The active implementation does not clip reversal fraction `g_r` before
evaluating `arctanh(2g_r-1)`. This matches upstream float32 arithmetic,
including its possible infinities at exactly saturated endpoints.

## Integration ordering

At each current-drive Euler step:

1. Evaluate R and phase fraction at the current temperature.
2. Compute electrical and thermal derivatives.
3. Advance V and T.
4. Update hysteresis direction using the new temperature.

The next step evaluates resistance with the updated branch state. Calling the
detector again at the same temperature does not create another reversal.

## Performance implementation

Torch is used for the float32 transcendental operations needed to match the
upstream trace. NumPy owns the ODE state and integration arrays. The current
sweep runner batches deterministic quasistatic amplitudes into one vectorized
hysteresis evaluation per timestep. Each batch column has an independent
accepted-temperature anchor, making its output exactly equal to running the
same amplitudes serially. A regression test compares every voltage,
temperature, resistance, phase, current, and power sample.

Stochastic, dynamic-phase, and multidomain configurations deliberately use the
serial integrator because batching could otherwise change random streams or
state semantics. Streamlit reuses the traces produced for GIF rendering when
building CSV and FFT outputs, so a sweep is never simulated twice.

## Historical artifact provenance

- Jobs/presets labeled `legacy pre-correction result` were generated with the
  `ported_per_step` behavior.
- The saved `Validated old-model quasistatic oscillatory domain` and the GIF
  campaign created in June 2026 were generated with the `turning_point`
  extension. Their electrical/thermal equations and fitted R(T) parameters are
  original, but their reversal geometry is not the exact Yuanhang algorithm.
- New runs always use the reference implementation and no longer store a mode
  selector.

## Verification commands

```bash
python scripts/audit_model_fidelity.py
python -m unittest discover -s tests -v
```

The audit must show near-zero float32 difference from the upstream reference.
The tests verify accumulated sub-threshold motion, detection-point geometry,
a known upstream resistance trace, and oscillator timestep convergence.

## Rules for future work

1. Do not reintroduce a selectable reversal algorithm without a separate model
   class, upstream justification, and new validation.
2. Treat alternate mode names in old metadata as provenance only.
3. Report `start_branch` as an initial condition.
4. Validate oscillations in a late window that excludes startup overshoot.
5. Repeat accepted oscillations at smaller `dt` and longer duration.
6. When comparing R(T) geometry, plot the actual simulated `(T,R)` path, not
   only the two major branches.
