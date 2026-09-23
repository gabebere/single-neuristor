"""Voltage/resistance comparison graphics and lightweight per-current animations."""
import io

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
                      margin=dict(l=65, r=25, t=85, b=45), legend=dict(orientation="h", y=1.12))
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
