from __future__ import annotations

import numpy as np

from neuristor.parameter_inference import _sustained_oscillation_loss


def test_sustain_loss_rejects_a_turn_on_transient() -> None:
    time_ns = np.arange(50.0, 251.0, 1.0)
    measured = 40.0 * np.sin(2.0 * np.pi * 50.0 * time_ns * 1e-3)
    sustained = measured.copy()
    transient = measured.copy()
    transient[time_ns >= 100.0] = 0.0
    flat = np.zeros_like(measured)

    sustained_loss, _, sustained_amplitude = _sustained_oscillation_loss(
        time_ns, measured, sustained, 50.0
    )
    transient_loss, measured_amplitude, transient_amplitude = _sustained_oscillation_loss(
        time_ns, measured, transient, 50.0
    )
    flat_loss, _, flat_amplitude = _sustained_oscillation_loss(time_ns, measured, flat, 50.0)

    assert sustained_loss < 1e-12
    assert transient_loss > 1.0
    assert flat_loss > transient_loss
    assert abs(measured_amplitude - 40.0) < 1e-9
    assert sustained_amplitude > transient_amplitude > flat_amplitude
