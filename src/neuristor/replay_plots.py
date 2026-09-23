"""Voltage/resistance comparison graphics and lightweight per-current animations."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def comparison_figure(traces, *, all_currents=False):
    """Return a shared-time plot; exported all-current versions include a slider."""
    groups = list(traces.groupby("source_mV", sort=True))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.1,
                        subplot_titles=("Measured and simulated voltage", "Simulated device resistance"))
    for index, (_, frame) in enumerate(groups):
        for column, label, color, row in (("measured_voltage_mV", "Measured voltage", "#202c39", 1),
                ("simulated_voltage_mV", "Simulated voltage", "#6846df", 1),
                ("resistance_ohm", "Simulated R", "#13977f", 2)):
            fig.add_trace(go.Scatter(x=frame.time_ns, y=frame[column], name=label,
                line=dict(color=color, width=1.6), visible=index == 0), row=row, col=1)
    fig.update_yaxes(title_text="Voltage (mV)", row=1, col=1)
    fig.update_yaxes(title_text="Resistance (Ω)", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Time relative to pulse (ns)", range=[-30, 350], row=2, col=1)
    title = lambda f: f"{f.current_step_uA.iloc[0]:.1f} µA · source setting {f.source_mV.iloc[0]:g} mV"
    fig.update_layout(height=660, title=title(groups[0][1]), template="plotly_white",
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color="#202c39"),
                      margin=dict(l=65, r=25, t=85, b=45), legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(gridcolor="#dddddd", zerolinecolor="#aaaaaa")
    fig.update_yaxes(gridcolor="#dddddd", zerolinecolor="#aaaaaa")
    if all_currents:
        steps = [dict(method="update", label=f"{f.current_step_uA.iloc[0]:.1f}",
                      args=[{"visible": [j//3 == i for j in range(3*len(groups))]}, {"title": title(f)}])
                 for i, (_, f) in enumerate(groups)]
        fig.update_layout(sliders=[dict(steps=steps, currentvalue={"prefix": "Current (µA): "}, pad={"t": 50})], height=740)
    return fig


def write_current_gif(frame, path, *, frames=24):
    """Animate a time cursor over both complete traces without resimulating physics."""
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    shown = frame[(frame.time_ns >= -30) & (frame.time_ns <= 350)]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    fig.subplots_adjust(left=.13, right=.97, top=.89, bottom=.12, hspace=.23)
    axes[0].plot(shown.time_ns, shown.measured_voltage_mV, color="#202c39", label="Measured")
    axes[0].plot(shown.time_ns, shown.simulated_voltage_mV, color="#6846df", label="Simulated")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("Voltage (mV)")
    axes[1].semilogy(shown.time_ns, shown.resistance_ohm, color="#13977f")
    axes[1].set(ylabel="Simulated R (Ω)", xlabel="Time (ns)", xlim=(-30, 350))
    fig.suptitle(f"{frame.current_step_uA.iloc[0]:.1f} µA · voltage and resistance")
    fig.canvas.draw()
    base = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
    height = base.height
    boxes = [ax.get_window_extent().bounds for ax in axes]
    images = []
    for time in np.linspace(0, 300, frames):
        im = base.copy()
        draw = ImageDraw.Draw(im)
        for x, y, w, h in boxes:
            xpos = x+(time+30)/380*w
            draw.line([(xpos, height-y-h), (xpos, height-y)], fill="#ed7934", width=2)
        images.append(im)
    images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)
    plt.close(fig)


def write_scope_gif(frame, branches, path, *, frames=96, snapshot_path=None, stacked_amperes=False, original_label_uA=None):
    """Reveal synchronized V(t), imposed I(t), and simulated R(T) up to each time.

    Major branches are static guides. The colored trajectory and its endpoint
    reveal only the elapsed simulated history; the curves do not predict future
    data ahead of the scope sweep. Use cached backgrounds to avoid redrawing axes.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    if frames < 2:
        raise ValueError('Scope animation needs at least two frames')
    shown = frame[(frame.time_ns >= -30) & (frame.time_ns <= 350)].reset_index(drop=True)
    if stacked_amperes:
        shown['input_current_A'] = shown.input_current_uA * 1e-6
        current_column = 'input_current_A'
        fig = plt.figure(figsize=(8.5, 10), dpi=100, facecolor='white')
        grid = fig.add_gridspec(3, 1, left=.13, right=.97, bottom=.105, top=.90,
                               height_ratios=[1, 1, 1.35], hspace=.42)
        current = fig.add_subplot(grid[0, 0])
        voltage = fig.add_subplot(grid[1, 0], sharex=current)
        rt = fig.add_subplot(grid[2, 0])
        current.set(ylabel='Input current (A)', xlabel='Time relative to pulse (ns)')
        current.ticklabel_format(axis='y', style='plain', useOffset=False)
        voltage.set(xlabel='Time relative to pulse (ns)')
    else:
        current_column = 'input_current_uA'
        fig = plt.figure(figsize=(11, 6), dpi=100, facecolor='white')
        grid = fig.add_gridspec(2, 2, left=.075, right=.975, bottom=.12, top=.84,
                               width_ratios=[1.18, 1], hspace=.27, wspace=.3)
        voltage = fig.add_subplot(grid[0, 0])
        current = fig.add_subplot(grid[1, 0], sharex=voltage)
        rt = fig.add_subplot(grid[:, 1])
        current.set(xlabel='Time relative to pulse (ns)', ylabel='Imposed current (µA)')
    voltage.set(ylabel='Voltage (mV)', xlim=(-30, 350))
    rt.set(xlabel='Simulated temperature (K)', ylabel='Simulated resistance (Ω)', yscale='log')
    rt.plot(branches.temperature_K, branches.heating_ohm, color='#e76f51', lw=1.7, label='Heating major branch')
    rt.plot(branches.temperature_K, branches.cooling_ohm, color='#348dc1', lw=1.7, label='Cooling major branch')
    temp_lo = min(shown.temperature_K.min()-2, branches.temperature_K.min())
    temp_hi = max(shown.temperature_K.max()+2, branches.temperature_K.max())
    rt.set_xlim(temp_lo, temp_hi)
    rt.set_ylim(min(branches.cooling_ohm.min(), shown.resistance_ohm.min())*.8,
                max(branches.heating_ohm.max(), shown.resistance_ohm.max())*1.2)
    lines = [voltage.plot([], [], color='#263238', lw=1.6, label='Measured output')[0],
             voltage.plot([], [], color='#7145d6', lw=1.6, label='Simulated device')[0],
             current.plot([], [], color='#b47900', lw=1.6, label='Measured input history')[0],
             rt.plot([], [], color='#148477', lw=1.7, label='Driven trajectory')[0]]
    dot, = rt.plot([], [], 'o', color='#148477', ms=6)
    for ax, cols in [(voltage, ['measured_voltage_mV', 'simulated_voltage_mV']),
                     (current, [current_column])]:
        vals = shown[cols].to_numpy()
        low, high = np.min(vals), np.max(vals)
        pad = max((high-low)*.1, 1e-7 if stacked_amperes and ax is current else 1)
        ax.set_ylim(low-pad, high+pad)
    for ax in (voltage, current, rt):
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='upper right')
    cursors = [ax.axvline(-30, color='#999999', lw=.8) for ax in (voltage, current)]
    if stacked_amperes:
        title = (f'{original_label_uA:g} µA' if original_label_uA is not None
                 else f'Input current: {frame.current_step_uA.iloc[0]*1e-6:.9f} A')
        fig.suptitle(title, fontsize=16, y=.982)
        if original_label_uA is not None:
            fig.text(.13, .953, f'Original numeric trace label · measured plateau: {frame.current_step_uA.iloc[0]:.3f} µA', fontsize=9, color='#555555')
        clock = fig.text(.13, .932, '', fontsize=11)
        fig.text(.13, .017, 'Progressive playback of saved traces (slowed down).\nR and T are simulated; heating/cooling branches are model guides.', fontsize=9, color='#555555')
    else:
        fig.suptitle(f'{frame.current_step_uA.iloc[0]:.1f} µA  |  voltage, current and hysteresis', fontsize=15, y=.975)
        clock = fig.text(.075, .88, '', fontsize=11)
        fig.text(.075, .025, 'R(T) and temperature are simulated; branches are model guides. Both measured channels are baseline corrected.', fontsize=9, color='#555555')
    artists = [*lines, dot, *cursors, clock]
    for artist in artists:
        artist.set_animated(True)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    images = []
    for time in np.linspace(shown.time_ns.iloc[0], shown.time_ns.iloc[-1], frames):
        part = shown[shown.time_ns <= time]
        for line, x, y in zip(lines, ['time_ns']*3+['temperature_K'],
                              ['measured_voltage_mV', 'simulated_voltage_mV', current_column, 'resistance_ohm']):
            line.set_data(part[x], part[y])
        dot.set_data(part.temperature_K.iloc[-1:], part.resistance_ohm.iloc[-1:])
        for cursor in cursors:
            cursor.set_xdata([time, time])
        clock.set_text(f'Elapsed pulse time: {time:6.1f} ns')
        fig.canvas.restore_region(background)
        for artist in artists:
            fig.draw_artist(artist)
        fig.canvas.blit(fig.bbox)
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
    if snapshot_path is not None:
        images[-1].save(snapshot_path)
    durations = [70]*len(images)
    durations[-1] = 1600
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0)
    plt.close(fig)


def scope_gallery(rows):
    """Offline current selector for the synchronized GIFs and full-trace figures."""
    import json
    records = json.dumps(rows, ensure_ascii=False)
    return '''<!doctype html><meta charset="utf-8"><title>VO₂ oscilloscope and hysteresis</title>
<style>body{font:17px system-ui;background:#f3f5fa;color:#243044;max-width:1150px;margin:30px auto;padding:0 20px}img{width:100%;border-radius:12px}input{width:100%}button{padding:8px 18px}a{color:#6345bd}</style>
<h1>VO₂ · oscilloscope and hysteresis</h1><p>One shared parameter set drives all measured current records.
Voltage and current reveal together; the green R(T) path follows the same time.</p>
<h2 id="title"></h2><img id="animation"><p><input id="current" type="range" min="0" step="1"></p>
<button id="previous">Previous current</button> <button id="next">Next current</button>
<p><a id="gif">Download GIF</a> · <a id="still">Full-trace figure</a> · <a href="comparison.html">Interactive voltage / resistance comparison</a></p>
<p>This replay is exploratory; inspect metrics.json for numerical errors and recovered oscillations. R and T are simulated, not measured.
Playback is slowed down and sampled for viewing. All numerical data are in traces.csv.</p>
<script>const runs='''+records+''';const slider=document.getElementById('current');slider.max=runs.length-1;
function show(){const r=runs[Number(slider.value)];document.getElementById('title').textContent=r.label;
document.getElementById('animation').src=r.gif;document.getElementById('gif').href=r.gif;document.getElementById('still').href=r.png;}
slider.oninput=show;document.getElementById('previous').onclick=()=>{slider.value=Math.max(0,Number(slider.value)-1);show()};
document.getElementById('next').onclick=()=>{slider.value=Math.min(runs.length-1,Number(slider.value)+1);show()};show();</script>'''
