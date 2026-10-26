"""Presentation-ready plots from saved verification samples, without simulation."""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_representative_voltage(traces, representatives, mode, path):
    """Show the same three current records with actual measured-current titles."""
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.5), constrained_layout=True)
    color = '#2864dc' if mode == 'frequency_only' else '#ce632b'
    for ax, row in zip(axes, representatives.itertuples()):
        frame = traces[(traces.source_label_mV == row.source_label_mV)
                       & (traces.dt_ns == row.dt_ns) & traces.time_ns.between(100, 250)]
        ax.plot(frame.time_ns, frame.measured_mV, color='#172b40', linewidth=1.4, label='Measured')
        ax.plot(frame.time_ns, frame.predicted_mV, color=color, linewidth=1.25, label='Simulated')
        ax.axvspan(150, 250, color='#b7c8df', alpha=.14)
        ax.set(title=f'{row.measured_current_A*1e6:.1f} µA', xlabel='Time (ns)', ylabel='Voltage (mV)', xlim=(100, 250))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_current_features(features, path):
    """Compare frequencies and amplitudes across measured currents for both losses."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True)
    for mode, frame in features.groupby('mode'):
        frame = frame[frame.target.astype(bool)].sort_values('measured_current_A')
        label = 'Frequency only' if mode == 'frequency_only' else 'Frequency + amplitude'
        color = '#2864dc' if mode == 'frequency_only' else '#ce632b'
        axes[0].plot(frame.measured_current_A*1e6, frame.predicted_frequency_MHz, '.-', color=color, label=label)
        axes[1].plot(frame.measured_current_A*1e6, frame.predicted_extrema_vpp_mV, '.-', color=color, label=label)
    axes[0].plot(frame.measured_current_A*1e6, frame.measured_frequency_MHz, 'ko--', label='Measured', markersize=4)
    axes[1].plot(frame.measured_current_A*1e6, frame.measured_extrema_vpp_mV, 'ko--', label='Measured', markersize=4)
    for ax, ylabel in zip(axes, ['Frequency (MHz)', 'Peak-to-peak voltage (mV)']):
        ax.set(xlabel='Measured input current (µA)', ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)
