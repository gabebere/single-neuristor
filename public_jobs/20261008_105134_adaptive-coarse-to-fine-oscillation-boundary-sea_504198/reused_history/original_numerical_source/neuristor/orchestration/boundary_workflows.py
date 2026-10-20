"""Coarse exploration followed by fine boundary verification in one run bundle."""
import copy
import dataclasses

import pandas as pd

from neuristor.config import resolved_copy, validate_config
from neuristor.runs import RunBundle
from neuristor.analysis.boundary_search import (
    BoundaryEvaluator, reconstruct_capacitance_scales, search_boundary_domain, validate_boundary_settings,
)
from neuristor.plotting.boundary_maps import render_boundary_maps


def run_boundary_maps(config, *, output_root=None, command='neuristor analyze boundary-maps'):
    """Explore the normalized domain, verify transitions, and label every open limit."""
    validate_config(config)
    from neuristor.workflows import _current_params_from_config
    params, current_uA, *_ = _current_params_from_config(config)
    settings = copy.deepcopy(config['boundary'])
    validate_boundary_settings(settings)
    if config['model'] != 'current':
        raise ValueError('Boundary maps require the ideal-current model')
    bundle = RunBundle.create(name=config['name'], model='current', kind='analysis',
        config=resolved_copy(config), output_root=output_root or config.get('output', {}).get('root', 'runs'),
        command=command)
    print(f'RUN_DIRECTORY={bundle.root}', flush=True)
    try:
        checkpoint_path = bundle.add_artifact('verification_checkpoint.csv', label='Every integration and horizon check')
        def checkpoint(rows):
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        def progress(state):
            bundle.write_json('status.json', state, label='Coarse/fine progress')
            print(f"{state['stage']}: {state.get('points', 0)} points", flush=True)
        evaluator = BoundaryEvaluator(params, current_uA, settings, checkpoint=checkpoint)
        coarse, fine, boundaries, metrics = search_boundary_domain(settings, evaluator, progress=progress)
        scaled = reconstruct_capacitance_scales(fine, settings)
        for name, table, label in [('coarse_grid.csv', coarse, 'Expanded coarse exploration'),
            ('fine_grid.csv', fine, 'All fine point classifications'),
            ('boundaries.csv', boundaries, 'Verified transition brackets and unresolved refinements'),
            ('scaled_oscillators.csv', scaled, 'Open common-scale predictions')]:
            table.to_csv(bundle.add_artifact(name, label=label), index=False)
        metrics.update(current_uA=current_uA, ambient_K=params.T0_K,
            initial_temperature_K=params.T_init_K, reference_C_th_pJ_per_K=settings['reference_C_th_pJ_per_K'],
            integration_attempts=len(evaluator.verification), transition_brackets=len(boundaries),
            tolerance_reached_brackets=int(boundaries.tolerance_reached.sum()),
            temperature_domain_K=[params.resist_params.T_min_K, params.resist_params.T_max_K])
        bundle.write_json('metrics.json', metrics, label='Coverage, caps and verified classifications')
        bundle.write_json('physics_parameters.json', dataclasses.asdict(params), label='Fixed model parameters in SI')
        render_boundary_maps(coarse, fine, boundaries, scaled, metrics, bundle.root)
        bundle.register_file('parameter_maps.html', label='Offline coarse/fine and open-scale 3D viewer')
        bundle.register_file('boundary_maps.png', label='Classifications across both stages')
        bundle.write_text('report.md', f'''# Adaptive oscillation-domain search

Input: {current_uA:g} µA constant ideal current. Ambient {params.T0_K:g} K,
initial temperature {params.T_init_K:g} K, branch {params.start_branch}.
Resistance domain: {metrics['temperature_domain_K']} K. Static R(T) and gamma are fixed.

## Why the three-parameter domain need not close

The deterministic equations are C dV/dt = I - V/R(T,H) and
Cth dT/dt = V²/R(T,H) - Se(T-T0). Hysteresis depends on the temperature path.
Replacing both C and Cth with s times their values gives the same state path at
time t/s. Thus Vpp and temperature extrema stay fixed and frequency divides by s.
We search C/Cth (pF K/pJ) and Se (mW/K) at reference Cth
{settings['reference_C_th_pJ_per_K']:g} pJ/K. The common scale direction is open
by model symmetry, not a missing closed upper surface. This reduction applies
only to deterministic constant current and the declared initial state/resistance law.
The scaled 3D samples are predictions from this identity, not extra simulations.
Their display range and the {settings['maximum_frequency_MHz']:g} MHz operational
selection never enter the oscillation-boundary classification.

## Two stages

The coarse log grid expands by its original log spacing where oscillators or
unresolved dynamics touch an outer face. Old points are cached. Hard axis limits,
expansion limits and point budgets remain explicit, never physical boundaries.
The fine grid covers the active coarse envelope plus neighboring cells. Every
fine point, including the nonoscillating side, is rerun at
{[settings['fine_dt_ns'], *settings['verification_dt_ns']]} ns. Only matching
classifications and <= {100*settings['convergence_fraction']:g}% frequency/Vpp
changes qualify. Disagreements get up to {settings.get('maximum_timestep_refinements', 0)}
additional step halvings, comparing the last three steps. Each fine row records
its actual verification_steps_ns. Neighboring verified oscillators/equilibria define log-space
brackets, then midpoint refinement narrows them. Censored midpoints stop a bracket.

Each normalized replay starts from the same state. The horizon starts at
{settings['initial_duration_ns']:g} ns, extends up to {settings['maximum_duration_ns']:g} ns,
and requires agreement at two horizons and adjacent late windows. Slow or decaying
transients remain unresolved. Nonoscillation requires small voltage/temperature
spans and small current/heat-balance residuals, not merely absent peaks.
Thresholds: voltage span <= {settings['equilibrium_voltage_span_mV']:g} mV,
temperature span <= {settings['equilibrium_temperature_span_K']:g} K,
relative balance residuals <= {settings['equilibrium_relative_tolerance']:g}.
These are approximate equilibrium criteria, not rigorous asymptotic stability.

## Results and scope

Coarse points: {len(coarse)}. Fine points: {len(fine)}.
Fine classifications: {metrics['classifications']}.
Transition brackets: {len(boundaries)}, with {metrics['tolerance_reached_brackets']}
meeting the requested log-width tolerance. Budget exhausted: {metrics['budget_exhausted']}.
Inspect metrics.json for every outer-face status, and boundaries.csv for both sides.

No complete global physical boundary is claimed. Finite grids can miss narrow
or disconnected islands, calibration/proximity exclusions censor a domain rather
than prove nonoscillation, and a finite horizon can remain inconclusive.
All limits and rejected points remain in the numerical tables and viewer.
No experimental calibration or parameter uncertainty is inferred.
''', label='Method, scale identity, results and remaining open limits')
        from neuristor.provenance import snapshot_package
        bundle.write_json('source_hashes.json', snapshot_package(bundle), label='Frozen numerical and rendering source')
        bundle.write_json('status.json', dict(stage='completed', **metrics), label='Final coverage status')
        bundle.complete(summary=metrics)
    except BaseException as exc:
        bundle.fail(exc)
        raise
    return bundle
