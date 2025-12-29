"""
Quick test script to run a single neuristor simulation and display time traces:
V_vo2 & V_load, T_vo2, I_vo2, and P_vo2 vs time.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

import plots as analysis
from model import YuanhangCircuitParams, YuanhangResistParams, simulate_yuanhang


def main() -> None:
    resist = YuanhangResistParams()
    circuit = YuanhangCircuitParams()

    sim = simulate_yuanhang(
        Vin=14.5,
        t_end=300e-6,
        dt=10e-9,
        resist_params=resist,
        circuit_params=circuit,
    )

    analysis.plot_time_traces(
        sim,
        R_series_kohm=circuit.R_series_kohm,
        title="Time traces (V_vo2/V_load, T_vo2, I_vo2, P_vo2)",
    )
    plt.show()


if __name__ == "__main__":
    main()
