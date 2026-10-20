"""Screened Cth/C/Se maps measured with steady-state voltage extrema.

The frequency ceiling is a user-selected operating filter, not a fundamental
VO2 material limit. Temperature, proximity and timestep checks are separate.
"""
from __future__ import annotations

from dataclasses import replace
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .current_drive_sim import simulate_current_parameter_grid
from .extrema import voltage_extrema


AXES = ("C_th_pJ_per_K", "S_e_mW_per_K", "C_pF")
AXIS_LABELS = ("Thermal capacitance Cth (pJ/K)", "Environmental conductance Se (mW/K)",
               "Electrical capacitance C (pF)")


def grid_values(settings):
    """Include the old log grid plus exact extra values for denser, reproducible slices."""
    axes = []
    for name in AXES:
        spec = settings['axes'][name]
        if 'values' in spec:
            values = np.asarray(spec['values'], float)
        else:
            values = np.geomspace(spec['low'], spec['high'], int(spec['count']))
            values = np.r_[values, spec.get('extra_values', [])]
        values = np.unique(values)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Map axes must be positive finite values")
        axes.append(values)
    return axes


def screen_trajectory(out, params, settings):
    """Separate sustained cycles, temperature validity and the frequency ceiling."""
    f = voltage_extrema(out['t'].astype(float) * 1e9, out['V_vo2'].astype(float) * 1e3,
                        settings['window_ns'])
    temperature = np.asarray(out['T'], float)
    finite = bool(np.isfinite(temperature).all())
    temperature_valid = bool(finite and temperature.min() >= params.resist_params.T_min_K
        and temperature.max() <= params.resist_params.T_max_K)
    slope = float(out['minimum_proximity_slope'])
    proximity_valid = bool(np.isfinite(slope) and slope >= 0)
    frequency_valid = bool(f['frequency_MHz'] is not None and
        0 < f['frequency_MHz'] <= settings['maximum_frequency_MHz'])
    eligible = bool(f['sustained'] and temperature_valid and proximity_valid and frequency_valid)
    return dict(frequency_MHz=f['frequency_MHz'], extrema_vpp_mV=f['extrema_vpp_mV'],
        peak_mV=f['peak_mV'], trough_mV=f['trough_mV'], retention=f['retention'],
        period_cv=f['period_cv'], cycle_count=f['cycle_count'], sustained=f['sustained'],
        temperature_min_K=float(temperature.min()), temperature_max_K=float(temperature.max()),
        temperature_valid=temperature_valid, minimum_proximity_slope=slope,
        proximity_valid=proximity_valid, frequency_valid=frequency_valid, eligible=eligible)


def evaluate_map(params, current_uA, settings, *, checkpoint=lambda rows: None):
    """Screen the Cartesian grid, then re-integrate survivors at two finer steps.

    Only survivors converging in frequency and amplitude appear in the 3D view.
    Coarse rejected points are not certified absent oscillators: a parameter map
    remains a conditional numerical exploration, not a complete bifurcation proof.
    """
    axes = grid_values(settings)
    combinations = np.asarray(list(itertools.product(*axes)))
    if len(combinations) > int(settings.get('max_points', 30000)):
        raise ValueError("Map exceeds configured point budget")
    steps = [float(params.dt_s * 1e9), *map(float, settings['verification_dt_ns'])]
    if len(steps) < 3 or any(b >= a for a, b in zip(steps[:-1], steps[1:])):
        raise ValueError("Require two successively smaller verification timesteps")
    rows = []
    chunk_size = int(settings.get('batch_size', 384))
    for start in range(0, len(combinations), chunk_size):
        chunk = combinations[start:start + chunk_size]
        outputs = simulate_current_parameter_grid(current_uA, params,
            C_F=chunk[:, 2] * 1e-12, C_th_J_per_K=chunk[:, 0] * 1e-12,
            S_e_W_per_K=chunk[:, 1] * 1e-3)
        for offset, (values, out) in enumerate(zip(chunk, outputs)):
            rows.append(dict(point_index=start + offset, **dict(zip(AXES, values)),
                current_uA=float(current_uA), **screen_trajectory(out, params, settings)))
        checkpoint(rows)
    coarse = pd.DataFrame(rows)
    verification = []
    survivors = coarse[coarse.eligible].copy()
    for dt_ns in steps[1:]:
        finer = replace(params, dt_s=dt_ns * 1e-9)
        for start in range(0, len(survivors), chunk_size):
            chunk = survivors.iloc[start:start + chunk_size]
            outputs = simulate_current_parameter_grid(current_uA, finer,
                C_F=chunk.C_pF.to_numpy() * 1e-12,
                C_th_J_per_K=chunk.C_th_pJ_per_K.to_numpy() * 1e-12,
                S_e_W_per_K=chunk.S_e_mW_per_K.to_numpy() * 1e-3)
            for (_, original), out in zip(chunk.iterrows(), outputs):
                verification.append(dict(point_index=int(original.point_index), dt_ns=dt_ns,
                    **screen_trajectory(out, finer, settings)))
    feature_columns = [key for key in coarse.columns if key not in (*AXES, 'point_index', 'current_uA')]
    verified = pd.DataFrame(verification, columns=['point_index', 'dt_ns', *feature_columns])
    result = coarse.copy()
    result['coarse_eligible'] = result.eligible
    result['converged'] = False
    result['included'] = False
    result['frequency_step_change_fraction'] = np.nan
    result['amplitude_step_change_fraction'] = np.nan
    for index in survivors.index:
        part = verified[verified.point_index == result.loc[index, 'point_index']].sort_values('dt_ns', ascending=False)
        previous = coarse.loc[index]
        frequency_changes, amplitude_changes = [], []
        for _, fine in part.iterrows():
            for column, changes in [('frequency_MHz', frequency_changes), ('extrema_vpp_mV', amplitude_changes)]:
                old, new = float(previous[column]), float(fine[column])
                changes.append(abs(new - old) / max(abs(new), 1e-9) if np.isfinite(old) and np.isfinite(new) else np.inf)
            previous = fine
        fchange, achange = max(frequency_changes), max(amplitude_changes)
        converged = bool(fchange <= settings['convergence_fraction'] and achange <= settings['convergence_fraction'])
        finest = part.iloc[-1]
        for column in finest.index:
            if column not in ('point_index', 'dt_ns'):
                result.loc[index, column] = finest[column]
        result.loc[index, 'frequency_step_change_fraction'] = fchange
        result.loc[index, 'amplitude_step_change_fraction'] = achange
        result.loc[index, 'converged'] = converged
        result.loc[index, 'included'] = bool(converged and part.eligible.all())
    return result, verified, axes


def slice_heatmap_values(table, axes, fixed, chosen):
    """Select one sampled plane and independently scale each metric to its visible data.

    Screened-out cells stay blank. A single-valued plane gets a tiny symmetric
    color-range padding; an empty plane has no meaningful color scale.
    """
    others = [name for name in AXES if name != fixed]
    sub = table[np.isclose(table[fixed], chosen, rtol=1e-9, atol=0)].copy()
    metrics = []
    for metric in ('frequency_MHz', 'extrema_vpp_mV'):
        sub['shown'] = np.where(sub.included, sub[metric], np.nan)
        pivot = sub.pivot(index=others[1], columns=others[0], values='shown').reindex(
            index=axes[AXES.index(others[1])], columns=axes[AXES.index(others[0])])
        z = pivot.to_numpy(float)
        finite = z[np.isfinite(z)]
        low, high = (float(finite.min()), float(finite.max())) if len(finite) else (0., 1.)
        if high == low:
            padding = max(abs(low) * 1e-6, 1e-6)
            low, high = low - padding, high + padding
        metrics.append(dict(z=z, low=low, high=high, has_values=bool(len(finite))))
    return dict(x=axes[AXES.index(others[0])], y=axes[AXES.index(others[1])],
        others=others, metrics=metrics, count=int(sub.included.sum()))


def render_physical_maps(table, axes, settings, destination: Path):
    """Write offline interactive 3D and six fixed-parameter heatmaps, plus PNGs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    keep = table[table.included].copy()
    limits = dict(frequency_MHz=float(settings['maximum_frequency_MHz']),
        extrema_vpp_mV=max(1.0, float(keep.extrema_vpp_mV.max())) if len(keep) else 1.0)
    custom = keep[[*AXES, 'frequency_MHz', 'extrema_vpp_mV', 'temperature_max_K']].to_numpy()
    fig = go.Figure()
    for column, label, visible in [('frequency_MHz', 'Frequency (MHz)', True),
                                  ('extrema_vpp_mV', 'Peak-to-peak voltage (mV)', False)]:
        fig.add_trace(go.Scatter3d(x=np.log10(keep[AXES[0]]), y=np.log10(keep[AXES[1]]),
            z=np.log10(keep[AXES[2]]), mode='markers', visible=visible, customdata=custom,
            marker=dict(size=4, color=keep[column], colorscale='Viridis', cmin=0, cmax=limits[column],
                        colorbar=dict(title=label)),
            hovertemplate='Cth=%{customdata[0]:.5g} pJ/K<br>Se=%{customdata[1]:.5g} mW/K<br>'
                'C=%{customdata[2]:.5g} pF<br>f=%{customdata[3]:.3f} MHz<br>'
                'Vpp=%{customdata[4]:.3f} mV<br>Tmax=%{customdata[5]:.2f} K<extra></extra>'))
    scene = {}
    for key, values, label in zip(('xaxis', 'yaxis', 'zaxis'), axes, AXIS_LABELS):
        ticks = np.geomspace(values.min(), values.max(), 6)
        scene[key] = dict(title=label, tickvals=np.log10(ticks), ticktext=[f'{x:.3g}' for x in ticks],
                          range=[np.log10(values.min()), np.log10(values.max())])
    fig.update_layout(title=f"Screened oscillation domain: {len(keep)} / {len(table)} grid points",
        scene=scene, height=760, margin=dict(l=20, r=30, b=20, t=75),
        updatemenus=[dict(buttons=[dict(label='Frequency', method='update', args=[{'visible': [True, False]}]),
            dict(label='Amplitude', method='update', args=[{'visible': [False, True]}])])])
    html_parts = [pio.to_html(fig, full_html=False, include_plotlyjs=True)]
    titles = []
    symbols = ('Cth', 'Se', 'C')
    units = ('pJ/K', 'mW/K', 'pF')
    for axis, symbol, unit in zip(AXES, symbols, units):
        text = f"Fixed {symbol} = {settings['fixed_values'][axis]:.5g} {unit}"
        titles.extend([f"{text}: frequency", f"{text}: amplitude"])
    static, static_axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
    for row_index, fixed in enumerate(AXES):
        fixed_index = AXES.index(fixed)
        chosen = float(settings['fixed_values'][fixed])
        if not np.any(np.isclose(axes[fixed_index], chosen, rtol=1e-9, atol=0)):
            raise ValueError(f"Fixed slice {fixed}={chosen} is not an actual grid value")
        initial = slice_heatmap_values(table, axes, fixed, chosen)
        others = initial['others']
        heat = make_subplots(rows=1, cols=2, subplot_titles=('Frequency (MHz)', 'Peak-to-peak voltage (mV)'),
            horizontal_spacing=.24)
        for column_index, (metric, label) in enumerate([('frequency_MHz', 'Frequency (MHz)'),
                                                       ('extrema_vpp_mV', 'Vpp (mV)')]):
            values = initial['metrics'][column_index]
            z = values['z']
            heat.add_trace(go.Heatmap(x=initial['x'], y=initial['y'], z=z,
                colorscale='Viridis', zmin=values['low'], zmax=values['high'],
                showscale=values['has_values'],
                colorbar=dict(title=label, len=.85, x=.40 if column_index == 0 else 1.0), hoverongaps=False,
                hovertemplate=f'{others[0]}=%{{x:.5g}}<br>{others[1]}=%{{y:.5g}}<br>{label}=%{{z:.4g}}<extra></extra>'),
                row=1, col=column_index+1)
            heat.update_xaxes(type='log', title_text=AXIS_LABELS[AXES.index(others[0])], row=1, col=column_index+1)
            heat.update_yaxes(type='log', title_text=AXIS_LABELS[AXES.index(others[1])], row=1, col=column_index+1)
            ax = static_axes[row_index, column_index]
            mesh = ax.pcolormesh(initial['x'], initial['y'], np.ma.masked_invalid(z),
                shading='nearest', cmap='viridis', vmin=values['low'], vmax=values['high'])
            ax.set(xscale='log', yscale='log', xlabel=AXIS_LABELS[AXES.index(others[0])],
                   ylabel=AXIS_LABELS[AXES.index(others[1])], title=titles[row_index*2+column_index])
            ax.set_facecolor('#eeeeee')
            static.colorbar(mesh, ax=ax, label=label)
        def slice_title(value, count):
            suffix = f'{count} retained oscillators' if count else 'No retained oscillators in this slice'
            return f'Locked {symbols[row_index]} = {value:.6g} {units[row_index]} · {suffix}'
        steps = []
        for value in axes[fixed_index]:
            plane = slice_heatmap_values(table, axes, fixed, value)
            steps.append(dict(label=f'{value:.5g}', method='update', args=[
                dict(z=[m['z'].tolist() for m in plane['metrics']],
                    zmin=[m['low'] for m in plane['metrics']], zmax=[m['high'] for m in plane['metrics']],
                    showscale=[m['has_values'] for m in plane['metrics']]),
                {'title.text': slice_title(value, plane['count'])}]))
        active = int(np.flatnonzero(np.isclose(axes[fixed_index], chosen, rtol=1e-9, atol=0))[0])
        heat.update_layout(title=slice_title(chosen, initial['count']), height=630,
            plot_bgcolor='#eeeeee', margin=dict(l=95, r=115, t=100, b=145),
            sliders=[dict(active=active, steps=steps, pad=dict(t=55),
                currentvalue=dict(prefix=f'Locked {symbols[row_index]} ({units[row_index]}): '),
                transition=dict(duration=0))])
        html_parts.append(pio.to_html(heat, full_html=False, include_plotlyjs=False))
    static.suptitle(f"{settings['current_uA']:g} µA, T0={settings['ambient_K']:g} K • screened steady-state extrema", fontsize=16)
    static.savefig(destination / 'heatmaps.png', dpi=145)
    plt.close(static)
    note = (f"<p>{len(table):,} parameter combinations; {len(keep):,} retained. "
        f"Frequency ≤ {settings['maximum_frequency_MHz']:g} MHz, temperature inside the resistance domain, "
        "nonnegative proximity slope, sustained late cycles and timestep convergence are required. "
        "Gray/blank cells contain no retained result; see grid.csv for the reason. "
        "Each slider selects an actual sampled parameter value and updates both heatmaps. "
        "Heatmap colorbars rescale independently to the visible values in that slice, in MHz and mV. "
        "Colors are therefore not directly comparable between different slices. "
        "These are conditional model maps, not measured physical parameter bounds.</p>")
    page = '<!doctype html><html><head><meta charset="utf-8"><title>VO2 screened parameter maps</title>'
    page += '<style>body{font-family:Arial,sans-serif;max-width:1300px;margin:30px auto;color:#203040}p{line-height:1.6}</style></head><body>'
    page += '<h1>Steady-state VO₂ parameter maps</h1>' + note + ''.join(html_parts) + '</body></html>'
    (destination / 'parameter_maps.html').write_text(page)
