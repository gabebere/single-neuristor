"""Phase-independent oscillation measurements from settled voltage extrema."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def voltage_extrema(time_ns, voltage_mV, window_ns=(150.0, 250.0), *, prominence_mV=2.0):
    """Measure complete peak-to-trough cycles inside an explicitly selected window.

    Amplitude is median peak-to-peak voltage (not half amplitude). Frequency is
    1000 divided by the median adjacent-maximum period in ns. A short 0.5 ns
    smoothing window locates extrema only; reported voltages use original data.
    Complete cycles exclude partial cycles at window edges. Persistence requires
    cycles in both halves, stable amplitudes, and regular periods. A late window
    by itself does not establish that an arbitrarily slow transient has settled.
    """
    t, y = np.asarray(time_ns, float), np.asarray(voltage_mV, float)
    if t.ndim != 1 or y.shape != t.shape or not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError("Time and voltage must be matching finite one-dimensional arrays")
    if np.any(np.diff(t) <= 0) or window_ns[0] >= window_ns[1]:
        raise ValueError("Time must increase and the measurement window must be ordered")
    mask = (t >= window_ns[0]) & (t <= window_ns[1])
    t, y = t[mask], y[mask]
    if len(t) < 9:
        raise ValueError("Extrema measurements require at least nine window samples")
    dt = float(np.median(np.diff(t)))
    width = max(5, int(round(0.5 / dt)) | 1)
    width = min(width, len(y) if len(y) % 2 else len(y) - 1)
    smooth = savgol_filter(y, width, 2)
    prominence = max(float(prominence_mV), 0.1 * float(np.ptp(smooth)))
    peaks, _ = find_peaks(smooth, prominence=prominence, distance=max(1, round(1.0 / dt)))
    troughs, _ = find_peaks(-smooth, prominence=prominence, distance=max(1, round(1.0 / dt)))
    cycles = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        inside = troughs[(troughs > left) & (troughs < right)]
        if len(inside) != 1:
            continue
        bottom = int(inside[0])
        top = 0.5 * (y[left] + y[right])
        # Smoothing may create a false extremum on a drifting or quantized
        # signal. A raw maximum must actually exceed its paired raw minimum.
        if top <= y[bottom]:
            continue
        cycles.append(dict(peak_time_ns=float(t[left]), next_peak_time_ns=float(t[right]),
            trough_time_ns=float(t[bottom]), peak_mV=float(top), trough_mV=float(y[bottom]),
            vpp_mV=float(top - y[bottom]), period_ns=float(t[right] - t[left]),
            midpoint_ns=float(0.5 * (t[left] + t[right]))))
    periods = np.array([c["period_ns"] for c in cycles])
    amps = np.array([c["vpp_mV"] for c in cycles])
    middle = 0.5 * (window_ns[0] + window_ns[1])
    early = [c["vpp_mV"] for c in cycles if c["midpoint_ns"] < middle]
    late = [c["vpp_mV"] for c in cycles if c["midpoint_ns"] >= middle]
    retention = float(np.median(late) / np.median(early)) if early and late and np.median(early) > 0 else None
    period_cv = float(np.std(periods) / np.mean(periods)) if len(periods) >= 2 else None
    sustained = bool(len(cycles) >= 2 and early and late and retention is not None
        and 0.75 <= retention <= 1 / 0.75 and period_cv is not None and period_cv <= 0.2)
    return dict(frequency_MHz=float(1000 / np.median(periods)) if len(periods) >= 2 else None,
        extrema_vpp_mV=float(np.median(amps)) if len(amps) else 0.0,
        peak_mV=float(np.median([c["peak_mV"] for c in cycles])) if cycles else None,
        trough_mV=float(np.median([c["trough_mV"] for c in cycles])) if cycles else None,
        mean_mV=float(np.mean(y)), retention=retention, period_cv=period_cv,
        peak_count=len(peaks), cycle_count=len(cycles), sustained=sustained,
        peak_times_ns=t[peaks].tolist(), trough_times_ns=t[troughs].tolist(), cycles=cycles,
        window_start_ns=float(window_ns[0]), window_stop_ns=float(window_ns[1]))


def extrema_loss(measured, predicted, settings):
    """Compare only settled extrema frequency/amplitude, with a missing-cycle penalty."""
    if settings.get('loss_function', 'scaled_log') == 'relative_squared':
        return relative_squared_extrema_loss(measured, predicted, settings)
    if measured['frequency_MHz'] is None or measured['extrema_vpp_mV'] <= 0:
        raise ValueError("Optimization targets must have complete measured cycles")
    sustained = bool(predicted['sustained'])
    valid = bool(sustained and predicted['frequency_MHz'] is not None
                 and predicted['frequency_MHz'] <= settings.get('maximum_frequency_MHz', float('inf')))
    ferr = predicted['frequency_MHz'] / measured['frequency_MHz'] - 1 if valid else 1.0
    # A smoothing artifact on a monotone or strongly drifting candidate can
    # locate a raw "peak" below a raw "trough". Such a candidate has no usable
    # positive oscillation amplitude and must receive a finite missing-cycle loss.
    amplitude = max(float(predicted['extrema_vpp_mV']), 0.0)
    valid = valid and amplitude > 0.0
    ferr = predicted['frequency_MHz'] / measured['frequency_MHz'] - 1 if valid else 1.0
    aratio = (amplitude + 0.1) / (measured['extrema_vpp_mV'] + 0.1)
    loss = (ferr / settings['frequency_tolerance_fraction']) ** 2
    loss += (np.log(aratio) / np.log1p(settings['amplitude_tolerance_fraction'])) ** 2
    if not valid:
        loss += 10000.0
    return dict(loss=float(loss), sustained=sustained, frequency_valid=valid,
        frequency_relative_error=float(ferr), robust_amplitude_relative_error=float(aratio - 1),
        periodic_amplitude_relative_error=float(aratio - 1),
        near_target=bool(valid and abs(ferr) <= 0.02 and abs(aratio - 1) <= 0.05))


def relative_squared_extrema_loss(measured, predicted, settings):
    """Equal relative frequency/Vpp errors, with eligibility reported separately.

    An absent frequency has no meaningful relative error: its diagnostic error
    is null and its loss is null. Search orchestration handles rejection; neither
    an epsilon, a logarithm, nor an arbitrary penalty enters this scientific loss.
    """
    fw = float(settings.get('frequency_weight', 1.0))
    aw = float(settings.get('amplitude_weight', 1.0))
    if not np.isfinite([fw, aw]).all() or fw < 0 or aw < 0 or fw + aw <= 0:
        raise ValueError('Relative loss weights must be finite, nonnegative and not both zero')
    fm, am = measured['frequency_MHz'], measured['extrema_vpp_mV']
    if fm is None or fm <= 0 or am <= 0:
        raise ValueError('Targets require positive frequency and amplitude')
    fp, ap = predicted['frequency_MHz'], float(predicted['extrema_vpp_mV'])
    ferr = float(fp / fm - 1) if fp is not None and np.isfinite(fp) and fp > 0 else None
    aerr = float(ap / am - 1)
    valid = bool(predicted['sustained'] and ferr is not None and ap > 0
                 and fp <= settings.get('maximum_frequency_MHz', float('inf')))
    return dict(loss=float(fw*ferr**2 + aw*aerr**2) if ferr is not None else None,
        sustained=bool(predicted['sustained']), frequency_valid=valid,
        frequency_relative_error=ferr, robust_amplitude_relative_error=aerr,
        periodic_amplitude_relative_error=aerr,
        near_target=bool(valid and abs(ferr) <= .02 and abs(aerr) <= .05))
