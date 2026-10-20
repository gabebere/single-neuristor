"""Budgeted joint static/dynamic inference using the authoritative simulator.

Search coordinates separate thermal time from conductance and Arrhenius slope
from resistance at 315 K. The score is a diagnostic engineering objective, not
a likelihood; validation records are excluded from every optimizer evaluation.
"""
from dataclasses import replace
from time import monotonic

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

from .current_drive_sim import simulate_current_waveforms
from .oscillation_audit import audit_voltage
from .resistance_custom_analysis import _major_loop_prediction, _temperature_branch_directions


NAMES = ("C_pF", "S_e_mW_per_K", "tau_th_ns", "T0_K", "Tc_K", "w_K",
         "beta_per_K", "gamma", "Rs_315K_ohm", "Ea_over_k_K", "Rm_ohm")
LOG = np.array([True, True, True, False, False, False, True, True, True, False, True])


def natural_vector(params):
    """Represent a physical model in the eleven joint-search coordinates."""
    r = params.resist_params
    return np.array([params.C_F*1e12, params.S_e_W_per_K*1e3,
                     params.C_th_J_per_K/params.S_e_W_per_K*1e9,
                     params.T0_K, r.Tc_K, r.w, r.beta, r.gamma,
                     r.R0*np.exp(r.Ea_over_k/315.0), r.Ea_over_k, r.Rm])


def joint_model(values, base, dt_ns):
    """Map search units to SI, deriving Cth and the Arrhenius prefactor."""
    c, se, tau, t0, tc, w, beta, gamma, rs, ea, rm = values
    rp = replace(base.resist_params, R0=rs*np.exp(-ea/315.0), Ea_over_k=ea,
                 Rm0=rm, Rm_factor=1.0, Tc_K=tc, w=w, beta=beta, gamma=gamma)
    return replace(base, C_F=c*1e-12, S_e_W_per_K=se*1e-3,
                   C_th_J_per_K=se*tau*1e-12, T0_K=t0, T_init_K=t0,
                   dt_s=dt_ns*1e-9, resist_params=rp)


def rt_prediction(values, rt):
    """Use the existing major-loop fitter's resistance law on measured branches."""
    _, _, _, _, tc, w, beta, _, rs, ea, rm = values
    temp = rt.Temperature.to_numpy(float)
    x = np.array([np.log10(rs)-ea/315.0/np.log(10), ea, np.log10(rm), w, tc, np.log10(beta)])
    return _major_loop_prediction(temp, _temperature_branch_directions(temp), x)


def static_rmse(values, rt):
    """Log10 resistance RMSE; all measured static points have equal weight."""
    return float(np.sqrt(np.mean(np.log10(rt_prediction(values, rt)/rt.Resistance.to_numpy(float))**2)))


def encode(values, bounds):
    """Normalize mixed log/linear physical coordinates to a unit cube."""
    lo, hi = np.asarray(bounds, float).T.copy()
    v = np.array(values, float, copy=True)
    lo[LOG], hi[LOG], v[LOG] = np.log(lo[LOG]), np.log(hi[LOG]), np.log(v[LOG])
    return (v-lo)/(hi-lo)


def decode(unit, bounds):
    """Undo unit-cube scaling without changing parameter units."""
    lo, hi = np.asarray(bounds, float).T.copy()
    lo[LOG], hi[LOG] = np.log(lo[LOG]), np.log(hi[LOG])
    v = lo + np.asarray(unit)*(hi-lo)
    v[LOG] = np.exp(v[LOG])
    return v


def voltage_features(time, voltage):
    """Keep per-window amplitudes so damped startup cannot masquerade as cycles."""
    summary, windows = audit_voltage(time, voltage)
    return summary, windows


def feature_loss(measured, predicted):
    """Continuous phase-tolerant mismatch with explicitly scaled feature terms.

    Mean voltage uses 20 mV; windowed amplitude uses log ratios with a 3 mV
    floor. Frequency is gated smoothly by predicted amplitude and coherence,
    and assessed only for experimentally persistent traces. The amplitude term
    still penalizes missing oscillations when the frequency gate closes.
    """
    ms, mw = measured
    ps, pw = predicted
    mean = np.mean(((pw.mean_mV.to_numpy()-mw.mean_mV.to_numpy())/20.0)**2)
    amp = np.mean(np.log((pw.periodic_vpp_mV.to_numpy()+3)/(mw.periodic_vpp_mV.to_numpy()+3))**2)
    robust = np.mean(np.log((pw.robust_vpp_mV.to_numpy()+3)/(mw.robust_vpp_mV.to_numpy()+3))**2)
    frequency = 0.0
    if ms["sustained"] and np.isfinite(ps["late_frequency_MHz"]):
        gate = ps["late_periodic_vpp_mV"]/(ps["late_periodic_vpp_mV"]+3.0)*ps["late_coherence"]
        frequency = gate*((ps["late_frequency_MHz"]-ms["late_frequency_MHz"])/10.0)**2
    return dict(mean=float(mean), amplitude=float(amp), robust=float(robust), frequency=float(frequency))


def predict_features(dataset, params, indices):
    """Preserve full input prehistory and crop only the unused post-250 ns tail."""
    local = replace(params, t_pre_s=max(0, -dataset.time_ns[0]*1e-9), t_end_s=250e-9)
    outputs = simulate_current_waveforms(dataset.current_uA[:, indices], local,
                                        waveform_time_s=dataset.time_ns*1e-9)
    keep = dataset.time_ns <= 250.0
    time = dataset.time_ns[keep]
    voltages = np.column_stack([np.interp(time*1e-9, o["t"].astype(float), o["V_vo2"])*1e3 for o in outputs])
    features = [voltage_features(time, v) for v in voltages.T]
    outside = any(np.min(o["T"]) < params.resist_params.T_min_K or
                  np.max(o["T"]) > params.resist_params.T_max_K for o in outputs)
    return features, voltages, bool(outside)


def fit_joint(dataset, rt, base, settings, seed_models, progress=None):
    """Run two bounded DE/Powell searches, sharing cached simulation evaluations.

    A cheap static-data screen rejects log10 RMSE above the declared ceiling.
    Both modes start from the same feasible population and report budget-limited
    termination. No simulation backend or physical equations are replaced.
    """
    bounds = np.array([settings["bounds"][name] for name in NAMES], float)
    if bounds.shape != (len(NAMES), 2) or np.any(bounds[:, 0] >= bounds[:, 1]) or np.any(bounds[LOG] <= 0):
        raise ValueError("Joint bounds must increase, with positive log coordinates")
    popsize = int(settings["population"])
    if popsize < 5:
        raise ValueError("Joint population must contain at least five candidates")
    rng = np.random.default_rng(int(settings["seed"]))
    max_rt = float(settings["maximum_static_rmse_log10"])
    measured = [voltage_features(dataset.time_ns, dataset.voltage_mV[:, i]) for i in dataset.train_indices]
    population = [np.clip(encode(natural_vector(p), bounds), 0, 1) for p in seed_models]
    seed_count = len(population)
    # Preserve good known regimes, perturb them, then add genuinely broad points.
    while len(population) < popsize//2:
        candidate = np.clip(population[len(population) % seed_count] + rng.normal(0, .035, len(NAMES)), 0, 1)
        if static_rmse(decode(candidate, bounds), rt) <= max_rt:
            population.append(candidate)
    for candidate in qmc.LatinHypercube(len(NAMES), seed=int(settings["seed"])).random(2048):
        if len(population) >= popsize:
            break
        if static_rmse(decode(candidate, bounds), rt) <= max_rt:
            population.append(candidate)
    if len(population) != popsize:
        raise ValueError("Could not construct the requested feasible starting population")
    cache, history, fits = {}, [], {}
    start = monotonic()

    def components(unit):
        key = tuple(np.round(unit, 12))
        if key not in cache:
            values = decode(unit, bounds)
            error = static_rmse(values, rt)
            terms = dict(mean=0., amplitude=0., robust=0., frequency=0.)
            screened = error > max_rt
            outside = False
            if not screened:
                features, _, outside = predict_features(dataset, joint_model(values, base, settings["search_dt_ns"]), dataset.train_indices)
                errors = [feature_loss(m, p) for m, p in zip(measured, features)]
                terms = {name: float(np.mean([e[name] for e in errors])) for name in terms}
            waveform = terms["mean"] + 3*terms["amplitude"] + terms["robust"] + .5*terms["frequency"]
            cache[key] = dict(**terms, waveform=waveform, static_rmse_log10=error,
                              screened=screened, outside_domain=outside,
                              **dict(zip(NAMES, values)))
        return cache[key]

    for mode, rt_weight in settings["static_weights"].items():
        best = [float("inf"), None]
        def objective(unit):
            terms = components(unit)
            loss = terms["waveform"] + float(rt_weight)*(terms["static_rmse_log10"]/.05)**2
            if terms["screened"] or terms["outside_domain"]:
                loss += 1e4
            history.append(dict(mode=mode, evaluation=len(history)+1, elapsed_s=monotonic()-start,
                                loss=loss, **terms))
            if loss < best[0]:
                best[:] = [loss, np.array(unit).copy()]
            if progress and len(history) % 12 == 0:
                progress(mode, len(history), best[0], len(cache))
            return loss
        de = differential_evolution(objective, [(0, 1)]*len(NAMES), init=np.array(population),
                                    maxiter=int(settings["generations"]), seed=int(settings["seed"]),
                                    polish=False, updating="immediate", mutation=(.3, .8), recombination=.7,
                                    tol=0.0, atol=0.0)
        local = minimize(objective, best[1], method="Powell", bounds=[(0, 1)]*len(NAMES),
                         options={"maxfev": int(settings["local_evaluations"]), "xtol": .003, "ftol": .001})
        fits[mode] = dict(values=decode(best[1], bounds), loss=best[0],
                          differential_evolution_message=str(de.message), local_message=str(local.message))
    return fits, pd.DataFrame(history), dict(unique_candidates=len(cache),
            simulations=sum(not c["screened"] for c in cache.values()), elapsed_s=monotonic()-start)
