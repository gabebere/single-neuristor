"""Coarse exploration followed by fine boundary verification in one run bundle."""
import copy
import dataclasses
import ast
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from neuristor.config import resolved_copy, validate_config
from neuristor.runs import RunBundle
from neuristor.analysis.boundary_search import (
    BoundaryEvaluator, reconstruct_capacitance_scales, search_boundary_domain, validate_boundary_settings,
)
from neuristor.plotting.boundary_maps import render_boundary_maps


def load_boundary_cache(source, params, current_uA, settings):
    """Reuse point evaluations only when solver/classifier semantics match exactly.

    Geometry, display ranges and point budgets may change; numerical controls may
    not. Compare the frozen evaluator/classifier AST without executing old source,
    and byte hashes for physics and extrema modules. Resolved SI parameters and
    imposed current must also agree. Original verification/source remain evidence.
    """
    source = Path(source).resolve()
    if json.loads((source / 'run.json').read_text())['status'] != 'completed':
        raise ValueError('Only completed boundary evaluations can be reused')
    old = json.loads((source / 'resolved_config.json').read_text())['boundary']
    geometry = {'axes', 'seed_points', 'coarse_count', 'fine_count', 'maximum_expansions',
                'max_coarse_points', 'max_fine_points', 'boundary_iterations', 'boundary_log10_tolerance',
                'display_C_th_pJ_per_K', 'maximum_frequency_MHz', 'batch_size'}
    keys = (set(old) | set(settings)) - geometry
    if any(old.get(k) != settings.get(k) for k in keys):
        raise ValueError('Reused boundary numerical settings must match exactly')
    if json.loads((source / 'physics_parameters.json').read_text()) != dataclasses.asdict(params):
        raise ValueError('Reused boundary physical parameters must match exactly')
    if json.loads((source / 'metrics.json').read_text())['current_uA'] != current_uA:
        raise ValueError('Reused boundary imposed current must match exactly')
    from neuristor.analysis import boundary_search
    package = Path(boundary_search.__file__).resolve().parents[1]
    for name in ('model.py', 'current_drive_sim.py', 'analysis/extrema.py'):
        if (source / 'source/neuristor' / name).read_bytes() != (package / name).read_bytes():
            raise ValueError(f'Reused boundary solver source differs: {name}')
    def definitions(path):
        tree = ast.parse(path.read_text())
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                and n.name in ('BoundaryEvaluator', 'classify_trajectory')}
    if definitions(source / 'source/neuristor/analysis/boundary_search.py') != definitions(Path(boundary_search.__file__)):
        raise ValueError('Reused boundary evaluator/classifier source differs')
    tables = {name: pd.read_csv(source / name, float_precision='round_trip')
              for name in ('coarse_grid.csv', 'fine_grid.csv')}
    cache = {}
    for table in tables.values():
        for _, series in table.iterrows():
            row = series.to_dict()
            if isinstance(row.get('verification_steps_ns'), str):
                row['verification_steps_ns'] = ast.literal_eval(row['verification_steps_ns'])
            point = (float(row['C_over_Cth_pF_K_per_pJ']), float(row['S_e_mW_per_K']))
            cache[(row['stage'], point)] = row
    return source, cache


def run_boundary_maps(config, *, output_root=None, command='neuristor analyze boundary-maps', reuse_exploration=None):
    """Explore the normalized domain, verify transitions, and label every open limit."""
    validate_config(config)
    from neuristor.workflows import _current_params_from_config
    params, current_uA, *_ = _current_params_from_config(config)
    settings = copy.deepcopy(config['boundary'])
    validate_boundary_settings(settings)
    if config['model'] != 'current':
        raise ValueError('Boundary maps require the ideal-current model')
    reused = load_boundary_cache(reuse_exploration, params, current_uA, settings) if reuse_exploration else None
    bundle = RunBundle.create(name=config['name'], model='current', kind='analysis',
        config=resolved_copy(config), output_root=output_root or config.get('output', {}).get('root', 'runs/local'),
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
        reused_keys, used_keys = set(), set()
        if reused:
            source, cache = reused
            evaluator.cache.update(cache)
            reused_keys = set(cache)
            # A cache-only run still has a checkpoint table with a stable schema.
            pd.DataFrame(columns=['stage', 'dt_ns', 'duration_ns', 'classification']).to_csv(checkpoint_path, index=False)
            source_hashes = {}
            for name in ('coarse_grid.csv', 'fine_grid.csv', 'verification_checkpoint.csv'):
                relative = f'original_evaluations/{name}'
                shutil.copy2(source / name, bundle.add_artifact(relative, label='Original reused evaluation evidence'))
                source_hashes[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
            # Reusing an already-reused exploration must preserve its earlier
            # checks too; otherwise a portable bundle loses verification ancestry.
            for name in ('original_evaluations', 'original_numerical_source'):
                if (source / name).is_dir():
                    target = bundle.path(f'reused_history/{name}')
                    shutil.copytree(source / name, target)
                    for file in target.rglob('*'):
                        if file.is_file():
                            bundle.register_file(file.relative_to(bundle.root).as_posix(), label='Earlier reused evidence')
            if (source / 'reused_history').is_dir():
                target = bundle.path('reused_history/ancestors')
                shutil.copytree(source / 'reused_history', target)
                for file in target.rglob('*'):
                    if file.is_file():
                        bundle.register_file(file.relative_to(bundle.root).as_posix(), label='Evaluation ancestry')
            if (source / 'numerical_provenance.json').is_file():
                shutil.copy2(source / 'numerical_provenance.json', bundle.add_artifact(
                    'reused_history/parent_provenance.json', label='Parent reuse provenance'))
            shutil.copytree(source / 'source', bundle.path('original_numerical_source'))
            for frozen in bundle.path('original_numerical_source').rglob('*.py'):
                bundle.register_file(frozen.relative_to(bundle.root).as_posix(), label='Original evaluation source')
            bundle.write_json('numerical_provenance.json', dict(source_bundle=str(source),
                table_sha256=source_hashes, cached_evaluations=len(cache),
                checks='Matching SI parameters, current, numerical settings, solver bytes, evaluator/classifier AST'),
                label='Checked reusable evaluations and original evidence')
        def evaluate(points, stage):
            for point in points:
                key = (stage, tuple(map(float, point)))
                if key in reused_keys:
                    used_keys.add(key)
            return evaluator(points, stage)
        coarse, fine, boundaries, metrics = search_boundary_domain(settings, evaluate, progress=progress)
        scaled = reconstruct_capacitance_scales(fine, settings)
        for name, table, label in [('coarse_grid.csv', coarse, 'Expanded coarse exploration'),
            ('fine_grid.csv', fine, 'All fine point classifications'),
            ('boundaries.csv', boundaries, 'Verified transition brackets and unresolved refinements'),
            ('scaled_oscillators.csv', scaled, 'Open common-scale predictions')]:
            table.to_csv(bundle.add_artifact(name, label=label), index=False)
        metrics.update(current_uA=current_uA, ambient_K=params.T0_K,
            initial_temperature_K=params.T_init_K, reference_C_th_pJ_per_K=settings['reference_C_th_pJ_per_K'],
            integration_attempts=len(evaluator.verification), transition_brackets=len(boundaries),
            reused_evaluations=len(used_keys),
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
fine edge is inspected again, and extends outward if finer integration reveals
oscillatory or unresolved points there. The envelope is not frozen to the coarse
pass's potentially incomplete classification. Every
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
Reused point evaluations: {len(used_keys)}. New integration/horizon checks:
{len(evaluator.verification)}. Reused records have their original tables and frozen
source retained separately, with byte hashes and numerical compatibility checks.
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
