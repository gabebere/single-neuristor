"""Display sampled classifications and boundary brackets without interpolating them."""
from pathlib import Path
import html
import json

import numpy as np

from neuristor.analysis.boundary_search import AXES, CLASSES


COLORS = dict(zip(CLASSES, ['#008C95', '#637487', '#D46732', '#AE55A6', '#D9AD28', '#D64B4B', '#111111']))


def render_boundary_maps(coarse, fine, boundaries, scaled, summary, destination):
    """Write an offline viewer and a static classification plot with explicit caps."""
    import plotly.graph_objects as go
    import plotly.io as pio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(destination)
    figures = []
    for label, table in [('Coarse exploration', coarse), ('Fine verification', fine)]:
        fig = go.Figure()
        for cls in CLASSES:
            part = table[table.classification.eq(cls)]
            fig.add_trace(go.Scatter(x=part[AXES[0]], y=part[AXES[1]], mode='markers', name=cls,
                marker=dict(color=COLORS[cls], size=7),
                customdata=part[['frequency_MHz', 'extrema_vpp_mV', 'duration_ns']].to_numpy().tolist(),
                hovertemplate='C/Cth=%{x:.5g} pF K/pJ<br>Se=%{y:.5g} mW/K<br>'
                              'Reference f=%{customdata[0]} MHz<br>Vpp=%{customdata[1]} mV<br>'
                              'Reference horizon=%{customdata[2]} ns<extra>%{fullData.name}</extra>'))
        if label.startswith('Fine'):
            for _, b in boundaries[boundaries.status.eq('bracketed')].iterrows():
                x, y = ([b.low, b.high], [b.fixed_value] * 2) if b.axis == AXES[0] else (
                    [b.fixed_value] * 2, [b.low, b.high])
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#152D42', width=3),
                    showlegend=False, hovertemplate='Verified oscillatory/equilibrium bracket<extra></extra>'))
        fig.update_layout(title=label, xaxis=dict(type='log', title='C/Cth (pF K/pJ)'),
                          yaxis=dict(type='log', title='Se (mW/K)'), height=650, template='plotly_white')
        figures.append(pio.to_html(fig, full_html=False, include_plotlyjs=True if not figures else False))
    fig = go.Figure()
    color_low = float(scaled.frequency_MHz.min()) if len(scaled) else 0.
    color_high = float(scaled.frequency_MHz.max()) if len(scaled) else 1.
    for passed, name, symbol in [(True, 'Oscillatory, within frequency screen', 'circle'),
                                 (False, 'Oscillatory, above chosen frequency screen', 'diamond')]:
        part = scaled[scaled.frequency_screen_pass.eq(passed)]
        fig.add_trace(go.Scatter3d(x=part.C_th_pJ_per_K, y=part.S_e_mW_per_K, z=part.C_pF,
            mode='markers', name=name, marker=dict(size=3, color=part.frequency_MHz,
            colorscale='Viridis', cmin=color_low, cmax=color_high, symbol=symbol,
            showscale=passed, colorbar=dict(title='MHz')),
            customdata=part[['frequency_MHz', 'extrema_vpp_mV', 'equivalent_duration_ns']].to_numpy().tolist(),
            hovertemplate='Cth=%{x:.5g} pJ/K<br>Se=%{y:.5g} mW/K<br>C=%{z:.5g} pF<br>'
                'f=%{customdata[0]:.3g} MHz<br>Vpp=%{customdata[1]:.3g} mV<br>'
                'Equivalent horizon=%{customdata[2]:.3g} ns<br>Scaled prediction, open scale ray<extra></extra>'))
    fig.update_layout(title='Three-parameter reconstruction: capacitance scale edges are display limits',
        scene=dict(xaxis=dict(type='log', title='Cth (pJ/K)'), yaxis=dict(type='log', title='Se (mW/K)'),
                   zaxis=dict(type='log', title='C (pF)')), height=750, template='plotly_white')
    figures.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))
    intro = ('<h1>Adaptive oscillation-domain exploration</h1><p>Coarse expansion precedes fine verification. '
        'A plotted edge can be a verified transition bracket, a calibration exclusion, unresolved time/step behavior, '
        'or an open search limit. The 110 MHz-style selection is separate from oscillation classification.</p>'
        '<p><b>Common capacitance scale is open:</b> (C, Cth) multiplied by s preserves voltage/temperature paths '
        'and Vpp, and divides frequency by s in the deterministic constant-current model. '
        'The 3D display samples that symmetry, without claiming finite upper/lower scale boundaries.</p>'
        '<p>Brackets apply to this initial state, current and resistance law. Finite sampling can miss narrow or '
        'disconnected domains. Approximate equilibria do not constitute a global bifurcation proof.</p>')
    (root / 'parameter_maps.html').write_text('<!doctype html><meta charset="utf-8"><title>Boundary search</title>'
        + intro + ''.join(figures) + '<h2>Limits and classifications</h2><pre>'
        + html.escape(json.dumps(summary, indent=2)) + '</pre>')
    fig, panels = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, (label, table) in zip(panels, [('Coarse expansion', coarse), ('Fine: timestep checks', fine)]):
        for cls in CLASSES:
            p = table[table.classification.eq(cls)]
            if len(p):
                ax.scatter(p[AXES[0]], p[AXES[1]], s=12, color=COLORS[cls], label=cls)
        ax.set(xscale='log', yscale='log', xlabel='C/Cth (pF K/pJ)', ylabel='Se (mW/K)', title=label)
        ax.legend(fontsize=7)
    for _, b in boundaries[boundaries.status.eq('bracketed')].iterrows():
        x, y = ([b.low, b.high], [b.fixed_value] * 2) if b.axis == AXES[0] else ([b.fixed_value] * 2, [b.low, b.high])
        panels[1].plot(x, y, color='#152D42', linewidth=1.2)
    fig.suptitle('Sampled domain, with unresolved and censored points retained')
    fig.savefig(root / 'boundary_maps.png', dpi=180)
    plt.close(fig)
