"""Phase-independent late-window inference with bounded independent restarts.

Only features inside the selected settled window enter the dynamic objective.
The full measured prehistory still drives the authoritative current-source model.
"""
from dataclasses import replace
from time import monotonic

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import detrend, find_peaks, savgol_filter
from scipy.stats import qmc

from neuristor.current_drive_sim import simulate_current_waveforms
from neuristor.fitting.joint_inference import NAMES, LOG, encode, decode, natural_vector, joint_model, static_rmse
from neuristor.analysis.oscillation_audit import dominant_frequency, _harmonic_fit


def settled_features(time_ns, voltage_mV, window_ns=(150., 250.), *, method="spectral"):
    """Extract settled amplitude, frequency and decay, excluding DC and startup.

    Robust Vpp is the 95th minus 5th percentile after removing a linear trend.
    Fundamental Vpp and frequency follow the existing spectral/harmonic audit.
    Half-window envelope ratios distinguish sustained cycles from decaying rings.
    The finite time window limits frequency resolution; no precision is inferred
    from the optimizer's tolerances or spectral interpolation.
    """
    if method == "extrema":
        from neuristor.analysis.extrema import voltage_extrema
        f = voltage_extrema(time_ns, voltage_mV, window_ns)
        # Retain established table columns so archived comparison plots work.
        return dict(**f, feature_method="extrema", robust_vpp_mV=f['extrema_vpp_mV'],
            periodic_vpp_mV=f['extrema_vpp_mV'], coherence=1.0 if f['sustained'] else 0.0,
            peaks_in_each_half=f['retention'] is not None,
            peak_frequency_MHz=f['frequency_MHz'], peak_period_cv=f['period_cv'])
    if method != "spectral":
        raise ValueError("Unknown settled feature method")
    t, y = np.asarray(time_ns, float), np.asarray(voltage_mV, float)
    mask = (t >= window_ns[0]) & (t <= window_ns[1])
    if mask.sum() < 32 or not np.isfinite(y).all():
        raise ValueError('Settled features require at least 32 finite samples')
    t, y = t[mask], y[mask]
    v = detrend(y)
    f = dominant_frequency(t, v, (10., 200.))
    amp, coherence = _harmonic_fit(t, v, f) if np.isfinite(f) else (0., 0.)
    robust = float(np.quantile(v, .95)-np.quantile(v, .05))
    halves = np.array_split(v, 2)
    half_amp = [float(np.quantile(h, .95)-np.quantile(h, .05)) for h in halves]
    retention = (half_amp[1]+1e-6)/(half_amp[0]+1e-6)
    smooth = savgol_filter(v, 5, 2)
    peaks, _ = find_peaks(smooth, prominence=max(2., .15*np.ptp(smooth)), distance=2)
    periods = np.diff(t[peaks])
    return dict(frequency_MHz=float(f) if np.isfinite(f) else None,
        periodic_vpp_mV=float(amp), robust_vpp_mV=robust, coherence=float(coherence),
        retention=float(retention), mean_mV=float(y.mean()),
        peak_count=int(len(peaks)),
        peaks_in_each_half=bool(np.any(peaks<len(v)//2) and np.any(peaks>=len(v)//2)),
        peak_frequency_MHz=float(1000/np.median(periods)) if len(periods) >= 2 else None,
        peak_period_cv=float(np.std(periods)/np.mean(periods)) if len(periods) >= 2 else None)


def settled_loss(measured, predicted, settings):
    """Equal weight per trace, relative frequency/amplitude loss with no DC loss."""
    if measured.get('feature_method') == 'extrema':
        from neuristor.analysis.extrema import extrema_loss
        return extrema_loss(measured, predicted, settings)
    if measured['frequency_MHz'] is None or measured['robust_vpp_mV'] <= 0:
        raise ValueError('Optimization targets must contain measured oscillations')
    frequency_valid = (predicted['frequency_MHz'] is not None
        and predicted['periodic_vpp_mV'] >= max(2., .2*measured['periodic_vpp_mV'])
        and predicted['coherence'] >= .2)
    sustained = bool(frequency_valid and predicted['peak_count'] >= 3
        and predicted['peaks_in_each_half'] and .75 <= predicted['retention'] <= 1/.75
        and predicted['peak_period_cv'] is not None and predicted['peak_period_cv'] <= .2)
    if settings.get('require_sustained', False):
        frequency_valid = frequency_valid and sustained
    ferr = predicted['frequency_MHz']/measured['frequency_MHz']-1 if frequency_valid else 1.
    aratio = (predicted['robust_vpp_mV']+.1)/(measured['robust_vpp_mV']+.1)
    pratio = (predicted['periodic_vpp_mV']+.1)/(measured['periodic_vpp_mV']+.1)
    decay = np.log(max(predicted['retention'], 1e-6)/max(measured['retention'], 1e-6))
    loss = ((ferr/settings['frequency_tolerance_fraction'])**2
            + (np.log(aratio)/np.log1p(settings['amplitude_tolerance_fraction']))**2
            + .5*(np.log(pratio)/np.log1p(settings['amplitude_tolerance_fraction']))**2
            + .5*(decay/.15)**2)
    if settings.get('require_sustained', False) and not sustained:
        loss += 10000.
    return dict(loss=float(loss), frequency_relative_error=float(ferr),
        sustained=sustained,
        robust_amplitude_relative_error=float(aratio-1),
        periodic_amplitude_relative_error=float(pratio-1), frequency_valid=bool(frequency_valid),
        near_target=bool(frequency_valid and abs(ferr)<=.02 and abs(aratio-1)<=.05
                         and abs(pratio-1)<=.05 and abs(decay)<=np.log(1.2)))


def evaluate_settled(dataset, values, base, indices, settings, dt_ns, measured=None):
    """Simulate with real prehistory; return only settled-window objective features."""
    window = settings['window_ns']
    p = joint_model(values, base, dt_ns)
    p = replace(p, t_pre_s=max(0, -dataset.time_ns[0]*1e-9), t_end_s=window[1]*1e-9)
    outputs = simulate_current_waveforms(dataset.current_uA[:, indices], p,
        waveform_time_s=dataset.time_ns*1e-9,
        audit_proximity=bool(settings.get('require_monotone_proximity', False)))
    keep = dataset.time_ns <= window[1]
    time = dataset.time_ns[keep]
    columns, rows = [], []
    outside = False
    for j, (i, o) in enumerate(zip(indices, outputs)):
        if not np.isfinite(o['V_vo2']).all() or not np.isfinite(o['T']).all():
            raise ValueError('Non-finite simulation')
        outside |= bool(o['T'].min()<p.resist_params.T_min_K or o['T'].max()>p.resist_params.T_max_K)
        voltage = np.interp(time*1e-9, o['t'].astype(float), o['V_vo2'].astype(float))*1e3
        method = settings.get('feature_method', 'spectral')
        pred = settled_features(time, voltage, window, method=method)
        m = measured[j] if measured is not None else settled_features(dataset.time_ns, dataset.voltage_mV[:,i],window, method=method)
        row = dict(source_label_mV=float(dataset.nominal_drives_mV[i]),
                   measured_current_A=float(np.mean(dataset.current_uA[(dataset.time_ns>=50)&(dataset.time_ns<=250),i]))*1e-6,
                   **{f'measured_{k}':v for k,v in m.items()},
                   **{f'predicted_{k}':v for k,v in pred.items()})
        if settings.get('require_monotone_proximity', False):
            slope = float(o['minimum_proximity_slope'])
            row.update(minimum_proximity_slope=slope if np.isfinite(slope) else None,
                       proximity_admissible=bool(np.isfinite(slope) and slope >= 0))
        # Non-target / nonoscillating control traces are still reported, without
        # assigning them a fictitious measured target frequency.
        if m['frequency_MHz'] is not None and m['robust_vpp_mV'] > 0:
            row.update(settled_loss(m,pred,settings))
        rows.append(row)
        columns.append(voltage)
    return rows, np.column_stack(columns), outside


class StageExpired(Exception):
    """Internal graceful deadline/stop signal, checked before each evaluation."""


def relative_candidate_score(rows, outside_domain=False):
    """Return the unchanged finite search score and separate scientific loss.

    Missing/invalid cycles are ranked by their count before the bounded partial
    error tie-breaker. The sentinel is useful to an optimizer but is never a
    derivative or value of the accepted all-target scientific objective.
    """
    invalid = sum(not row['frequency_valid'] for row in rows)
    errors = [row['loss'] for row in rows if row['loss'] is not None]
    dynamic = float(np.mean(errors)) if errors else None
    score = (1e6 + 1000*invalid + min(dynamic or 0., 999.)) if invalid else dynamic
    admissible = all(row.get('proximity_admissible', True) for row in rows)
    if outside_domain:
        score += 1e6
    if not admissible:
        score += 1e6
    return dict(score=float(score), dynamic_loss=dynamic,
                invalid_count=int(invalid), proximity_admissible=bool(admissible),
                eligible=bool(not invalid and not outside_domain and admissible),
                objective_loss=dynamic if not invalid and not outside_domain and admissible else None)


def measured_current_targets(dataset, current_range_uA):
    """Select records by baseline-corrected mean current over 50–250 ns.

    Acquisition filenames remain provenance only; their nominal mV labels are
    never interpreted as physical input-current magnitudes.
    """
    low, high = map(float, current_range_uA)
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError('Measured-current limits must be finite and ordered')
    window = (dataset.time_ns >= 50) & (dataset.time_ns <= 250)
    if not window.any():
        raise ValueError('Measured-current selection requires 50–250 ns samples')
    currents = np.mean(dataset.current_uA[window], axis=0)
    indices = np.flatnonzero((currents >= low) & (currents <= high))
    if not 0 < len(indices) < len(currents):
        raise ValueError('Select at least one target and retain at least one control')
    return indices


def search_settled(dataset, rt, base, settings, seed_vectors, bootstrap,
                   checkpoint, stop_requested=lambda: False):
    """Time-boxed DE/Powell restarts with feasible correlated static seeds.

    First restarts hold all six major-loop parameters at the static fit; later
    restarts jointly vary them under a hard R(T) RMSE ceiling. Most initial
    dynamic coordinates are independently stratified, rather than perturbations
    of one winner. Cache keys include all eleven normalized physical coordinates.
    The callback receives every evaluated candidate for durable checkpoints.
    """
    bounds=np.asarray([settings['bounds'][n] for n in NAMES],float)
    if bounds.shape!=(11,2) or np.any(bounds[:,1]<=bounds[:,0]) or np.any(bounds[LOG]<=0):
        raise ValueError('Invalid parameter bounds')
    if int(settings['population'])<5 or int(settings['restarts'])<1:
        raise ValueError('Require population >=5 and at least one restart')
    frozen=natural_vector(base)
    static_indices=np.array([4,5,6,8,9,10])
    fixed_names = settings.get('fixed_parameters', [])
    if any(n not in NAMES for n in fixed_names):
        raise ValueError('Unknown fixed search parameter')
    always_fixed = {NAMES.index(n) for n in fixed_names}
    evaluation_limit = int(settings.get('maximum_evaluations', 2**63-1))
    if evaluation_limit < 1:
        raise ValueError('maximum_evaluations must be positive')
    measured=[settled_features(dataset.time_ns,dataset.voltage_mV[:,i],settings['window_ns'],
        method=settings.get('feature_method', 'spectral')) for i in dataset.train_indices]
    start=monotonic(); cache={}; history_count=0; elites=[]; restart_results=[]
    maximum=float(settings['maximum_static_rmse_log10'])
    end=start+float(settings['search_seconds'])
    best=None
    diagnostic_best=None
    simple = settings.get('loss_function') == 'relative_squared'
    for restart in range(int(settings['restarts'])):
        if monotonic()>=end or stop_requested() or history_count >= evaluation_limit: break
        fixed=restart<int(settings['fixed_resistance_restarts'])
        active=np.array([i for i in range(11)
                         if i not in always_fixed and (not fixed or i not in static_indices)])
        if not len(active):
            raise ValueError('At least one search parameter must remain free')
        rng=np.random.default_rng(int(settings['seed'])+restart*104729)
        stage_end=min(end,start+(restart+1)*settings['search_seconds']/settings['restarts'])
        de_end=monotonic()+.8*max(0,stage_end-monotonic())
        frozen_unit=encode(frozen,bounds)
        def expand(short):
            u=frozen_unit.copy();u[active]=short
            return u
        local_best=[float('inf'),None]
        deadline=[de_end]
        def objective(short):
            nonlocal history_count,best,diagnostic_best
            if monotonic()>=deadline[0] or stop_requested() or history_count >= evaluation_limit:
                raise StageExpired()
            u=expand(short); key=tuple(np.round(u,12))
            if key not in cache:
                v=decode(u,bounds); re=static_rmse(v,rt)
                record=dict(restart=restart,static_fixed=fixed,values=v.tolist(),static_rmse_log10=float(re),
                            simulated=False,outside_domain=False,proximity_admissible=True)
                if re>maximum:
                    score=1e6+1e4*(re/maximum)**2
                else:
                    try:
                        rows,_,outside=evaluate_settled(dataset,v,base,dataset.train_indices,
                            settings,settings['search_dt_ns'],measured)
                        invalid = sum(not r['frequency_valid'] for r in rows)
                        if simple:
                            # Feasibility takes priority over accuracy. This finite
                            # solver sentinel is NOT part of the scientific loss.
                            scoring = relative_candidate_score(rows, outside)
                            dynamic = scoring['dynamic_loss']
                            score = scoring['score']
                        else:
                            score=float(np.mean([r['loss'] for r in rows]))+settings['static_weight']*(re/.05)**2
                        if not np.isfinite(score):
                            raise ValueError('Non-finite objective')
                        if outside and not simple: score+=1e6
                        admissible=all(r.get('proximity_admissible', True) for r in rows)
                        if not admissible and not simple: score+=1e6
                        record.update(simulated=True,outside_domain=outside,per_current=rows,
                                      proximity_admissible=admissible,
                                      oscillations_eligible=invalid == 0,
                                      dynamic_loss=dynamic if simple else None)
                    except (FloatingPointError,OverflowError,ValueError) as exc:
                        score=1e9;record.update(error=str(exc))
                record.update(score=float(score),elapsed_s=monotonic()-start)
                cache[key]=record
                if record['simulated'] and not record['outside_domain'] and record['proximity_admissible']:
                    if diagnostic_best is None or score < diagnostic_best['score']:
                        diagnostic_best=record
                    if not simple or record['oscillations_eligible']:
                        elites.append(record);elites.sort(key=lambda r:r['score']);del elites[40:]
                        if best is None or score<best['score']: best=record
                history_count+=1
                checkpoint(record,dict(unique_candidates=history_count,elapsed_s=monotonic()-start,
                    restart=restart+1,restarts=settings['restarts'],best=best,elites=elites[:12]))
            score=cache[key]['score']
            if score<local_best[0]:local_best[:]=[score,np.array(short).copy()]
            return score
        population=[]
        # Include a small known seed set; the majority of the population remains
        # independent in dynamic coordinates, preserving basin diversity.
        for v in seed_vectors[:int(settings.get('initial_seed_count', 3))]:
            population.append(np.clip(encode(v,bounds)[active],0,1))
        if best is not None:population.append(np.clip(encode(best['values'],bounds)[active],0,1))
        draws=qmc.LatinHypercube(11,seed=int(settings['seed'])+restart).random(int(settings['population'])*30)
        for u in draws:
            if len(population)>=int(settings['population']):break
            v=decode(u,bounds)
            if not fixed:
                if rng.random()<.65:
                    sample=bootstrap.iloc[int(rng.integers(len(bootstrap)))]
                    v[static_indices]=[sample.Tc_K,sample.w_K,sample.beta_per_K,
                        sample.R0_ohm*np.exp(sample.Ea_over_k_K/315),sample.Ea_over_k_K,sample.Rm_ohm]
                else:
                    v[static_indices]=np.asarray(seed_vectors[int(rng.integers(len(seed_vectors)))])[static_indices]
            else:v[static_indices]=frozen[static_indices]
            if np.any(v<bounds[:,0]) or np.any(v>bounds[:,1]) or static_rmse(v,rt)>maximum:continue
            population.append(encode(v,bounds)[active])
        if len(population)!=int(settings['population']):raise ValueError('Insufficient feasible population')
        try:
            differential_evolution(objective,[(0.,1.)]*len(active),init=np.asarray(population),
                maxiter=int(settings.get('maximum_generations',100000)),seed=int(settings['seed'])+restart,
                polish=False,mutation=(.35,1.0),recombination=.8,tol=0,atol=0,updating='immediate')
        except StageExpired:pass
        deadline[0]=stage_end
        if local_best[1] is not None and not stop_requested():
            try:
                minimize(objective,local_best[1],method='Powell',bounds=[(0.,1.)]*len(active),
                    options={'maxfev':int(settings['local_evaluations']),'xtol':.0005,'ftol':1e-5})
            except StageExpired:pass
        if local_best[1] is not None:
            record=cache[tuple(np.round(expand(local_best[1]),12))]
            restart_results.append(dict(record,restart=restart,static_fixed=fixed))
    return dict(best=best,diagnostic_best=diagnostic_best,elites=elites,restarts=restart_results,
                unique_candidates=history_count,elapsed_s=monotonic()-start)
