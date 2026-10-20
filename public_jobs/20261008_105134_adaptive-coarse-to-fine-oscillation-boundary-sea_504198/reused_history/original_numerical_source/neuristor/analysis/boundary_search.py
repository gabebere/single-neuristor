"""Adaptive exploration of the deterministic ideal-current oscillatory domain.

At fixed current, ambient, resistance law and initial state, multiplying C and
Cth by s replaces t by t/s. Explore C/Cth and Se once, rather than mistaking
a finite common-capacitance plotting range for a physical upper boundary.
All boundary statements are finite-resolution, initial-state-conditional brackets.
"""
from __future__ import annotations

from dataclasses import replace
import itertools

import numpy as np
import pandas as pd

from neuristor.analysis.extrema import voltage_extrema
from neuristor.current_drive_sim import simulate_current_parameter_grid


AXES = ("C_over_Cth_pF_K_per_pJ", "S_e_mW_per_K")
UNKNOWN = {"time_unresolved", "numerical_unresolved", "step_budget"}
CLASSES = ("oscillatory", "nonoscillatory", "calibration_censored", "proximity_censored",
           "time_unresolved", "numerical_unresolved", "step_budget")


def validate_boundary_settings(settings):
    """Reject impossible budgets and unsafe time/search definitions before running."""
    s = settings
    for key in ("reference_C_th_pJ_per_K", "initial_duration_ns", "maximum_duration_ns",
                "coarse_dt_ns", "fine_dt_ns", "convergence_fraction", "equilibrium_relative_tolerance",
                "equilibrium_voltage_span_mV", "equilibrium_temperature_span_K",
                "boundary_log10_tolerance", "maximum_frequency_MHz"):
        if not np.isfinite(float(s[key])) or float(s[key]) <= 0:
            raise ValueError(f"boundary.{key} must be finite and positive")
    if s['maximum_duration_ns'] < s['initial_duration_ns'] * 2:
        raise ValueError("maximum_duration_ns must allow at least two observation horizons")
    steps = [s['fine_dt_ns'], *s['verification_dt_ns']]
    if len(steps) != 3 or any(not np.isfinite(v) or v <= 0 for v in steps) or any(
            b >= a for a, b in zip(steps[:-1], steps[1:])):
        raise ValueError("Require two successively smaller verification_dt_ns")
    if s['coarse_dt_ns'] < s['fine_dt_ns'] or s['initial_duration_ns'] / s['coarse_dt_ns'] < 64:
        raise ValueError("Coarse timestep must allow >=64 samples and be >= fine_dt_ns")
    for key in ('coarse_count', 'fine_count', 'batch_size', 'max_coarse_points', 'max_fine_points',
                'max_steps_per_simulation', 'maximum_expansions', 'boundary_iterations'):
        if int(s[key]) != s[key] or s[key] < (3 if key in ('coarse_count', 'fine_count') else 1):
            raise ValueError(f"boundary.{key} must be a positive integer (counts >=3)")
    refinements = s.get('maximum_timestep_refinements', 0)
    if int(refinements) != refinements or refinements < 0:
        raise ValueError('maximum_timestep_refinements must be a nonnegative integer')
    if s['max_coarse_points'] < s['coarse_count'] ** 2 or s['max_fine_points'] < s['fine_count'] ** 2:
        raise ValueError("Point budgets must cover the initial coarse and fine grids")
    for name in AXES:
        a = s['axes'][name]
        v = [a[k] for k in ('hard_low', 'low', 'high', 'hard_high')]
        if not np.isfinite(v).all() or not 0 < v[0] <= v[1] < v[2] <= v[3]:
            raise ValueError(f"Invalid ordered positive search limits for {name}")
    for point in s.get('seed_points', []):
        if len(point) != 2 or not np.isfinite(point).all() or any(
            not s['axes'][a]['low'] <= point[i] <= s['axes'][a]['high'] for i, a in enumerate(AXES)):
            raise ValueError('seed_points must be pairs inside the initial search rectangle')
    initial_axes = [np.unique(np.r_[np.geomspace(s['axes'][a]['low'], s['axes'][a]['high'], s['coarse_count']),
                    [p[i] for p in s.get('seed_points', [])]]) for i, a in enumerate(AXES)]
    if np.prod([len(a) for a in initial_axes]) > s['max_coarse_points']:
        raise ValueError('Coarse budget must cover the grid including seed coordinates')
    scales = np.asarray(s['display_C_th_pJ_per_K'], float)
    if not len(scales) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("display_C_th_pJ_per_K must contain positive finite values")


def classify_trajectory(out, params, settings):
    """Separate persistent cycles, approximate equilibria and censored trajectories.

    Two adjacent late windows must agree. Absence of detected peaks alone never
    establishes nonoscillation: small V/T spans and small circuit/thermal balance
    residuals must also hold in both windows. Thresholds define a numerical
    approximate equilibrium, not a mathematical proof of asymptotic stability.
    """
    t = np.asarray(out['t'], float) * 1e9
    v = np.asarray(out['V_vo2'], float) * 1e3
    temp, r = np.asarray(out['T'], float), np.asarray(out['R'], float)
    row = dict(classification='numerical_unresolved', frequency_MHz=None, extrema_vpp_mV=None,
               duration_ns=float(t[-1]), temperature_min_K=float(np.nanmin(temp)),
               temperature_max_K=float(np.nanmax(temp)), minimum_proximity_slope=float(out['minimum_proximity_slope']))
    if not all(np.isfinite(a).all() for a in (t, v, temp, r)):
        return row
    if temp.min() < params.resist_params.T_min_K or temp.max() > params.resist_params.T_max_K:
        return dict(row, classification='calibration_censored')
    if not np.isfinite(row['minimum_proximity_slope']) or row['minimum_proximity_slope'] < 0:
        return dict(row, classification='proximity_censored')
    end = t[-1]
    windows = [(end * .5, end * .75), (end * .75, end)]
    features = [voltage_extrema(t, v, w) for w in windows]
    a, b = features
    row.update(frequency_MHz=b['frequency_MHz'], extrema_vpp_mV=b['extrema_vpp_mV'],
               cycle_count=b['cycle_count'])
    if a['sustained'] and b['sustained']:
        agreement = all(abs(a[k] / b[k] - 1) <= settings['convergence_fraction']
                        for k in ('frequency_MHz', 'extrema_vpp_mV'))
        if agreement:
            return dict(row, classification='oscillatory')
    equilibrium = []
    for lo, hi in windows:
        mask = (t >= lo) & (t <= hi)
        vv, tt, rr = v[mask] * 1e-3, temp[mask], r[mask]
        ii = np.asarray(out['I_in'], float)[mask]
        electric = np.max(np.abs(ii - vv / rr)) / max(np.max(np.abs(ii)), 1e-15)
        joule, cooling = vv ** 2 / rr, params.S_e_W_per_K * (tt - params.T0_K)
        thermal = np.max(np.abs(joule - cooling)) / max(np.max(np.abs(joule)), np.max(np.abs(cooling)), 1e-15)
        equilibrium.append(np.ptp(vv) * 1e3 <= settings['equilibrium_voltage_span_mV']
                           and np.ptp(tt) <= settings['equilibrium_temperature_span_K']
                           and electric <= settings['equilibrium_relative_tolerance']
                           and thermal <= settings['equilibrium_relative_tolerance'])
    return dict(row, classification='nonoscillatory' if all(equilibrium) else 'time_unresolved')


class BoundaryEvaluator:
    """Batch the existing physics, extending horizons and verifying every fine point.

    A fresh replay from the same initial state at each horizon preserves hysteresis
    history. All fine classifications, including rejected/nonoscillating neighbors,
    use two smaller timesteps. Nothing is inferred from an expired time budget.
    """
    def __init__(self, params, current_uA, settings, checkpoint=lambda rows: None):
        if params.sigma_W_sqrt_s != 0 or params.pulse_off_s is not None or params.pulse_on_s != 0 or params.t_pre_s != 0:
            raise ValueError("Boundary scaling requires deterministic constant current from t=0 without prehistory")
        self.params, self.current_uA, self.settings = params, current_uA, settings
        self.checkpoint, self.cache, self.verification = checkpoint, {}, []

    def _at_step(self, points, dt_ns, stage):
        s = self.settings
        pending = list(points)
        results, previous = {}, {}
        duration = s['initial_duration_ns']
        while pending:
            if duration / dt_ns + 2 > s['max_steps_per_simulation']:
                for p in pending:
                    results[p] = dict(previous.get(p, {}), classification='step_budget', duration_ns=duration,
                        frequency_MHz=None, extrema_vpp_mV=None, dt_ns=dt_ns, stage=stage,
                        **dict(zip(AXES, p)))
                    self.verification.append(results[p].copy())
                self.checkpoint(self.verification)
                break
            again = []
            for start in range(0, len(pending), s['batch_size']):
                part = pending[start:start + s['batch_size']]
                p = replace(self.params, dt_s=dt_ns * 1e-9, t_end_s=duration * 1e-9,
                            C_th_J_per_K=s['reference_C_th_pJ_per_K'] * 1e-12)
                outputs = simulate_current_parameter_grid(self.current_uA, p,
                    C_F=np.array([x[0] for x in part]) * s['reference_C_th_pJ_per_K'] * 1e-12,
                    C_th_J_per_K=np.full(len(part), p.C_th_J_per_K),
                    S_e_W_per_K=np.array([x[1] for x in part]) * 1e-3)
                for point, out in zip(part, outputs):
                    actual = replace(p, S_e_W_per_K=point[1] * 1e-3)
                    row = classify_trajectory(out, actual, s)
                    row.update(**dict(zip(AXES, point)), dt_ns=dt_ns, stage=stage)
                    self.verification.append(row.copy())
                    old = previous.get(point)
                    stable = old is not None and old['classification'] == row['classification']
                    if stable and row['classification'] == 'oscillatory':
                        stable = all(abs(old[k] / row[k] - 1) <= s['convergence_fraction']
                                     for k in ('frequency_MHz', 'extrema_vpp_mV'))
                    censored = row['classification'] in ('calibration_censored', 'proximity_censored', 'numerical_unresolved')
                    if censored or (stable and row['classification'] in ('oscillatory', 'nonoscillatory')):
                        results[point] = row
                    elif duration >= s['maximum_duration_ns']:
                        results[point] = dict(row, classification='time_unresolved')
                    else:
                        previous[point] = row
                        again.append(point)
                self.checkpoint(self.verification)
            pending = again
            duration = min(duration * 2, s['maximum_duration_ns'])
        return results

    def __call__(self, points, stage):
        points = list(dict.fromkeys(tuple(map(float, p)) for p in points))
        new = [p for p in points if (stage, p) not in self.cache]
        if new:
            steps = [self.settings['coarse_dt_ns']] if stage == 'coarse' else [
                self.settings['fine_dt_ns'], *self.settings['verification_dt_ns']]
            runs = [self._at_step(new, dt, stage) for dt in steps]
            histories = {p: [run[p] for run in runs] for p in new}
            def agrees(history):
                last = history[-3:]
                agree = all(r['classification'] == last[-1]['classification'] for r in last)
                if agree and last[-1]['classification'] == 'oscillatory':
                    agree = all(abs(a[k] / b[k] - 1) <= self.settings['convergence_fraction']
                                for a, b in zip(last[:-1], last[1:])
                                for k in ('frequency_MHz', 'extrema_vpp_mV'))
                return agree
            if stage != 'coarse':
                for refinement in range(self.settings.get('maximum_timestep_refinements', 0)):
                    pending = [p for p in new if not agrees(histories[p])]
                    if not pending:
                        break
                    extra = self._at_step(pending, steps[-1] / 2 ** (refinement + 1), stage)
                    for p in pending:
                        histories[p].append(extra[p])
            for p in new:
                row = dict(histories[p][-1], **dict(zip(AXES, p)), stage=stage)
                if stage != 'coarse':
                    agree = agrees(histories[p])
                    if not agree and row['classification'] != 'step_budget':
                        row['classification'] = 'numerical_unresolved'
                    row['timestep_verified'] = agree and row['classification'] in ('oscillatory', 'nonoscillatory')
                    row['verification_steps_ns'] = [r['dt_ns'] for r in histories[p][-3:]]
                self.cache[(stage, p)] = row
        return [self.cache[(stage, p)] for p in points]


def face_status(table, axes):
    """Label sampled outer faces without turning caps or censoring into boundaries."""
    faces = []
    for i, name in enumerate(AXES):
        for side, value in [('low', axes[i][0]), ('high', axes[i][-1])]:
            part = table[np.isclose(table[name], value, rtol=1e-10, atol=0)]
            classes = set(part.classification) if len(part) else set()
            if 'oscillatory' in classes:
                status = 'oscillatory_open'
            elif not classes or classes & UNKNOWN:
                status = 'unresolved_open'
            elif classes - {'nonoscillatory'}:
                status = 'censored_open'
            else:
                status = 'sampled_nonoscillatory'
            faces.append(dict(axis=name, side=side, value=float(value), status=status))
    return faces


def search_boundary_domain(settings, evaluate, *, progress=lambda state: None):
    """Expand a sparse log grid, then refine its active domain and transition edges.

    Preserve spacing during expansion, cache old coordinates and inspect every
    outer face. Fine grids include a coarse-neighbor margin around oscillatory
    points; unknown cells remain visible. Adjacent oscillatory/equilibrium pairs
    form brackets, bisected in log space. Censored/unknown midpoints stop a bracket
    rather than certify it. Disconnected islands smaller than grid spacing can
    remain undiscovered; no finite search promises all bifurcations.
    """
    validate_boundary_settings(settings)
    s = settings
    axes = [np.geomspace(s['axes'][a]['low'], s['axes'][a]['high'], s['coarse_count']) for a in AXES]
    spacing = [np.log10(a[1] / a[0]) for a in axes]
    axes = [np.unique(np.r_[a, [p[i] for p in s.get('seed_points', [])]]) for i, a in enumerate(axes)]
    coarse = {}
    rounds = []
    budget_exhausted = False
    for iteration in range(s['maximum_expansions'] + 1):
        new = [p for p in itertools.product(*axes) if p not in coarse]
        if len(coarse) + len(new) > s['max_coarse_points']:
            budget_exhausted = True
            break
        for row in evaluate(new, 'coarse'):
            coarse[tuple(row[a] for a in AXES)] = row
        table = pd.DataFrame(coarse.values())
        faces = face_status(table, axes)
        rounds.append(dict(iteration=iteration, points=len(coarse), faces=faces))
        progress(dict(stage='coarse', rounds=rounds, points=len(coarse)))
        next_axes = [a.copy() for a in axes]
        for face in faces:
            if face['status'] not in ('oscillatory_open', 'unresolved_open'):
                continue
            i = AXES.index(face['axis']); limits = s['axes'][face['axis']]
            old = axes[i][0 if face['side'] == 'low' else -1]
            value = max(limits['hard_low'], old / 10 ** spacing[i]) if face['side'] == 'low' else min(
                limits['hard_high'], old * 10 ** spacing[i])
            next_axes[i] = np.unique(np.r_[next_axes[i], value])
        if iteration == s['maximum_expansions'] or all(np.array_equal(a, b) for a, b in zip(axes, next_axes)):
            break
        # Only committed/evaluated axes are returned if the next shell exceeds budget.
        if np.prod([len(a) for a in next_axes]) > s['max_coarse_points']:
            budget_exhausted = True
            break
        axes = next_axes
    table = pd.DataFrame(coarse.values())
    active = table[table.classification.eq('oscillatory') | table.classification.isin(UNKNOWN)]
    fine_axes = []
    for name, a in zip(AXES, axes):
        if len(active):
            left = max(0, np.searchsorted(a, active[name].min()) - 1)
            right = min(len(a) - 1, np.searchsorted(a, active[name].max()) + 1)
        else:
            left, right = 0, len(a) - 1
        seeds = [p[len(fine_axes)] for p in s.get('seed_points', []) if a[left] <= p[len(fine_axes)] <= a[right]]
        fine_axes.append(np.unique(np.r_[np.geomspace(a[left], a[right], s['fine_count']), seeds]))
    if np.prod([len(a) for a in fine_axes]) > s['max_fine_points']:
        raise ValueError('Fine budget must cover the envelope grid including seed coordinates')
    fine = {}
    for row in evaluate(list(itertools.product(*fine_axes)), 'fine'):
        fine[tuple(row[a] for a in AXES)] = row
    progress(dict(stage='fine', points=len(fine), coarse_points=len(coarse)))
    # Bracket transitions along BOTH axes, verifying the nonoscillating side too.
    brackets = []
    for dimension in range(2):
        other = 1 - dimension
        for fixed in fine_axes[other]:
            for left, right in zip(fine_axes[dimension][:-1], fine_axes[dimension][1:]):
                p, q = [0., 0.], [0., 0.]
                p[other] = q[other] = fixed; p[dimension], q[dimension] = left, right
                p, q = tuple(p), tuple(q)
                if {fine[p]['classification'], fine[q]['classification']} == {'oscillatory', 'nonoscillatory'}:
                    brackets.append(dict(axis=AXES[dimension], left=p, right=q, status='bracketed'))
    for iteration in range(s['boundary_iterations']):
        proposals = []
        for bracket in brackets:
            d = AXES.index(bracket['axis']); p, q = bracket['left'], bracket['right']
            if bracket['status'] != 'bracketed' or np.log10(q[d] / p[d]) <= s['boundary_log10_tolerance']:
                continue
            mid = list(p); mid[d] = np.sqrt(p[d] * q[d]); mid = tuple(mid)
            proposals.append((bracket, mid))
        new = list(dict.fromkeys(m for _, m in proposals if m not in fine))
        if len(fine) + len(new) > s['max_fine_points']:
            budget_exhausted = True
            break
        if not proposals:
            break
        for row in evaluate(new, 'fine'):
            fine[tuple(row[a] for a in AXES)] = row
        for bracket, mid in proposals:
            cls = fine[mid]['classification']
            if cls not in ('oscillatory', 'nonoscillatory'):
                bracket['status'] = cls
            elif cls == fine[bracket['left']]['classification']:
                bracket['left'] = mid
            else:
                bracket['right'] = mid
        progress(dict(stage='boundary_refinement', iteration=iteration, points=len(fine)))
    boundary_rows = []
    for b in brackets:
        d = AXES.index(b['axis']); p, q = b['left'], b['right']
        width = float(np.log10(q[d] / p[d]))
        boundary_rows.append(dict(axis=b['axis'], fixed_axis=AXES[1-d], fixed_value=p[1-d],
            low=p[d], high=q[d], log10_width=width, low_class=fine[p]['classification'],
            high_class=fine[q]['classification'], status=b['status'],
            tolerance_reached=b['status'] == 'bracketed' and width <= s['boundary_log10_tolerance']))
    fine_table = pd.DataFrame(fine.values())
    summary = dict(coarse_points=len(table), fine_points=len(fine_table), rounds=rounds,
                   budget_exhausted=budget_exhausted, coarse_faces=face_status(table, axes),
                   fine_faces=face_status(fine_table, fine_axes), common_capacitance_scale='open_by_model_symmetry',
                   complete_physical_boundary=False,
                   classifications=fine_table.classification.value_counts().to_dict())
    return table, fine_table, pd.DataFrame(boundary_rows, columns=[
        'axis', 'fixed_axis', 'fixed_value', 'low', 'high', 'log10_width', 'low_class', 'high_class',
        'status', 'tolerance_reached']), summary


def reconstruct_capacitance_scales(fine, settings):
    """Extrude verified oscillators along exact scale rays, labeling display limits.

    These are scaled predictions, not extra integrations. Vpp and temperature
    stay fixed, f scales as 1/Cth. The optional frequency screen labels points
    without erasing the underlying dynamical domain.
    """
    rows = []
    for _, p in fine[fine.classification.eq('oscillatory')].iterrows():
        for cth in settings['display_C_th_pJ_per_K']:
            scale = cth / settings['reference_C_th_pJ_per_K']
            f = p.frequency_MHz / scale
            rows.append(dict(C_th_pJ_per_K=cth, S_e_mW_per_K=p.S_e_mW_per_K,
                C_pF=p[AXES[0]] * cth, frequency_MHz=f, extrema_vpp_mV=p.extrema_vpp_mV,
                reference_ratio=p[AXES[0]], scale=scale,
                reference_duration_ns=p.duration_ns, equivalent_duration_ns=p.duration_ns * scale,
                frequency_screen_pass=f <= settings['maximum_frequency_MHz'],
                provenance='deterministic_time_scaling', scale_boundary='display_only_open'))
    return pd.DataFrame(rows, columns=['C_th_pJ_per_K', 'S_e_mW_per_K', 'C_pF', 'frequency_MHz',
        'extrema_vpp_mV', 'reference_ratio', 'scale', 'reference_duration_ns', 'equivalent_duration_ns',
        'frequency_screen_pass', 'provenance', 'scale_boundary'])
