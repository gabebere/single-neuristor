"""Bundle a comparison of two matched completed steady-search runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from neuristor.analysis.loss_comparison import representative_records, timestep_comparison, verification_summary
from neuristor.plotting.loss_comparison import plot_current_features, plot_representative_voltage
from neuristor.runs import RunBundle


def compare_steady_runs(frequency_run, combined_run, *, output_root=None, command='neuristor analyze compare-steady'):
    """Reuse saved all-current verification; never rerun or alter either parent."""
    sources = {'frequency_only': Path(frequency_run).resolve(),
               'frequency_amplitude': Path(combined_run).resolve()}
    parents = {}
    for mode, source in sources.items():
        manifest = json.loads((source/'run.json').read_text())
        metrics = json.loads((source/'metrics.json').read_text())
        if manifest['status'] != 'completed' or not metrics['verification_complete']:
            raise ValueError('Both parent runs must have completed all verification batches')
        config = json.loads((source/'resolved_config.json').read_text())
        settings = config['steady']
        if settings.get('loss_function') != 'relative_squared' or settings.get('frequency_weight') != 1.:
            raise ValueError('Comparison requires relative-squared frequency objectives')
        expected = 0. if mode == 'frequency_only' else 1.
        if settings.get('amplitude_weight') != expected:
            raise ValueError('Supply frequency-only first and equal-weight frequency/amplitude second')
        search = json.loads((source/'search_result.json').read_text())
        params = pd.read_csv(source/'parameters.csv')
        names = [n for n in params.candidate if n.startswith(('search_', 'diagnostic_'))]
        if len(names) != 1:
            raise ValueError('Comparison requires exactly one newly searched verified candidate per parent')
        verification = pd.read_csv(source/'verification.csv')
        selected = verification[verification.candidate == names[0]].copy()
        parents[mode] = dict(config=config, settings=settings, metrics=metrics, search=search,
                             candidate=names[0], table=selected, parameters=params[params.candidate == names[0]])
    def protocol(parent):
        result = dict(parent['settings'])
        result.pop('amplitude_weight')
        return result
    if protocol(parents['frequency_only']) != protocol(parents['frequency_amplitude']):
        raise ValueError('Parent protocols differ beyond the amplitude weight')
    def fixed_config(parent):
        return {k:v for k,v in parent['config'].items() if k not in ('name','steady')}
    if fixed_config(parents['frequency_only']) != fixed_config(parents['frequency_amplitude']):
        raise ValueError('Parent fixed simulation configurations differ')
    config = dict(sources={n:str(p) for n,p in sources.items()}, protocol=protocol(parents['frequency_only']))
    bundle = RunBundle.create(name='Controlled frequency and amplitude loss comparison',
        model='steady-loss-comparison', kind='analysis', config=config,
        output_root=output_root or 'runs', command=command)
    try:
        from neuristor.provenance import snapshot_package
        snapshot_package(bundle)
        summary, features, reps, convergence, parameters, provenance = [], [], [], [], [], {}
        reference = None
        for mode, parent in parents.items():
            source = sources[mode]
            table = parent['table']
            fine = table[table.dt_ns == table.dt_ns.min()].sort_values('source_label_mV')
            evidence = fine[['source_label_mV','measured_current_A','measured_frequency_MHz','measured_extrema_vpp_mV','target']].reset_index(drop=True)
            if reference is not None:
                pd.testing.assert_frame_equal(reference, evidence)
            reference = evidence
            common = verification_summary(table).assign(mode=mode, candidate=parent['candidate'])
            common['unique_candidates'] = parent['metrics']['unique_candidates']
            common['search_elapsed_s'] = parent['metrics']['search_elapsed_s']
            summary.append(common)
            features.append(fine.assign(mode=mode))
            chosen = representative_records(table)
            reps.append(chosen.assign(mode=mode))
            convergence.append(timestep_comparison(table).assign(mode=mode))
            parameters.append(parent['parameters'].assign(mode=mode))
            traces = pd.read_csv(source/'verification_traces.csv')
            traces = traces[traces.candidate == parent['candidate']]
            plot_representative_voltage(traces, chosen, mode,
                bundle.add_artifact(f'figures/{mode}_representatives.png', label=f'{mode}: low/middle/high measured-current voltage'))
            shown = traces[(traces.dt_ns == fine.dt_ns.iloc[0])
                           & traces.source_label_mV.isin(chosen.source_label_mV)]
            shown.to_csv(bundle.add_artifact(f'{mode}_representative_traces.csv', label='Exact saved samples for representative plots'), index=False)
            provenance[mode] = dict(source=str(source), candidate=parent['candidate'],
                hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                        for name in ['run.json','resolved_config.json','verification.csv','verification_traces.csv','parameters.csv','search_result.json']})
        summaries = pd.concat(summary, ignore_index=True)
        comparisons = pd.concat(features, ignore_index=True)
        representative = pd.concat(reps, ignore_index=True)
        for name, frame in [('summary.csv',summaries), ('finest_features.csv',comparisons),
                            ('representative_metrics.csv',representative), ('convergence.csv',pd.concat(convergence)),
                            ('parameters.csv',pd.concat(parameters))]:
            frame.to_csv(bundle.add_artifact(name,label=name.removesuffix('.csv').replace('_',' ')), index=False)
        plot_current_features(comparisons, bundle.add_artifact('figures/features_vs_measured_current.png',label='Common frequency/amplitude comparison'))
        bundle.write_json('numerical_provenance.json',provenance,label='Immutable parent artifacts and hashes')
        bundle.write_json('metrics.json',dict(parent_runs={n:str(p) for n,p in sources.items()},
            target_count=int(reference.target.sum()), control_count=int((~reference.target).sum()),
            representative_current_uA=(representative[representative['mode']=='frequency_only'].measured_current_A*1e6).tolist()),label='Comparison scope')
        text = '''# Frequency-only versus frequency-and-amplitude fitting

Both searches use the same measured currents, fixed R(T) and T0, bounds,
starting vectors, seed, evaluation cap, and verification protocol. Only the
amplitude-error weight differs. A finite budget is not a global optimum.

The unknowns are gamma, C, Se and Cth. Internally Cth is parameterized by Cth/Se.
The broad search bounds are diagnostic choices, not measured specimen intervals.
All selected target records were used for optimization; the other records are controls.
Control-oscillation counts use the simulated complete-cycle gate, not a new
experimental reclassification. Gate outcomes must not be confused with
complete-waveform agreement.

Scientific objective values are present only for all-target eligible candidates.
Partial frequency MAPE is labeled with the number of defined frequencies.
The two *_MSE_defined columns cross-evaluate both losses on defined frequencies
only; these partial diagnostic values are not accepted all-target objectives.
Amplitude means peak-to-peak voltage. Means, controls and timestep changes remain
out-of-objective diagnostics. Sampled proximity and temperature screens do not
prove physical validity. The voltage traces are compared without phase alignment.

Low, middle and high representative target currents were selected by current
rank, before inspecting residuals. Titles use baseline-corrected measured mean
current over 50–250 ns; acquisition mV labels appear only in provenance tables.
No new simulation, smoothing, or filtering was performed for this comparison.

See summary.csv for both objective evaluations on both parameter vectors,
convergence.csv for each current at both refinements, and parameters.csv.
'''
        for mode, parent in parents.items():
            text += f"\n{mode}: {parent['candidate']}; {parent['metrics']['unique_candidates']} unique candidates; accepted search candidate: {parent['search']['best'] is not None}.\n"
        measured_gates = parents['frequency_only']['table']
        if 'measured_sustained' in measured_gates:
            failed = measured_gates[(measured_gates.dt_ns == measured_gates.dt_ns.min())
                & measured_gates.target & ~measured_gates.measured_sustained]
            if len(failed):
                currents = ', '.join(f'{value*1e6:.1f}' for value in failed.measured_current_A)
                text += f'\nMeasured target records at {currents} µA themselves fail the selected retention gate, despite measurable frequency/Vpp.\n'
        bundle.write_text('report.md',text,label='Comparison methods, scope and limitations')
        bundle.complete(summary=dict(parent_count=2,target_count=int(reference.target.sum())))
    except Exception as exc:
        bundle.fail(exc)
        raise
    return bundle
