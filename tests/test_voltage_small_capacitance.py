"""Independent RC/Joule-heating reference at specimen-scale SI capacities."""

import numpy as np
import pytest

from neuristor.model import YuanhangCircuitParams, YuanhangResistParams, simulate_yuanhang


def test_sub_picofarad_and_sub_picojoule_capacities_match_analytic_solution():
    # Remove R(T) feedback to isolate both unit conversions and integration.
    circuit = YuanhangCircuitParams(
        R_series_kohm=1.0, C_par_pF=0.5, Cth_mW_ns_per_K=0.05,
        Sth_mW_per_K=0.004, T_base_K=314.4,
    )
    resistance = YuanhangResistParams(R0=0.0, Rm0=1000.0, Rm_factor=1.0)
    errors = []
    for dt_ns in (0.005, 0.0025, 0.00125):
        out = simulate_yuanhang(
            1.0, t_end=2e-9, dt=dt_ns * 1e-9,
            resist_params=resistance, circuit_params=circuit,
            init={"T_K": 314.4, "Vn": 0.0},
        )
        t = np.asarray(out["time_s"])
        # R_series || R_device = 500 ohm, V_in = 1 V.
        b = 1.0 / (500.0 * 0.5e-12)
        a = 0.004e-3 / 0.05e-12
        voltage = 0.5 * (-np.expm1(-b * t))
        # Exact convolution of V(t)^2/R with the thermal cooling kernel.
        temperature = 314.4 + (0.5**2 / 1000.0 / 0.05e-12) * (
            -np.expm1(-a * t) / a
            - 2.0 * (np.exp(-b * t) - np.exp(-a * t)) / (a - b)
            + (np.exp(-2.0 * b * t) - np.exp(-a * t)) / (a - 2.0 * b)
        )
        errors.append([
            np.max(np.abs(np.asarray(out["V_node"]) - voltage)),
            np.max(np.abs(np.asarray(out["T_K"]) - temperature)),
        ])
    errors = np.asarray(errors)
    assert np.all(errors[1:] < 0.8 * errors[:-1]), errors
    assert errors[-1, 0] < 0.0005
    assert errors[-1, 1] < 0.005


@pytest.mark.parametrize("field", ["C_par_pF", "Cth_mW_ns_per_K"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_voltage_capacitances_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match="finite and positive"):
        simulate_yuanhang(1.0, circuit_params=YuanhangCircuitParams(**{field: value}))
