from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step
from neuristor.model import YuanhangResistParams


JOB_ID = "20260614_125407_article_audit_816bf1"
JOB_DIR = ROOT / "jobs" / JOB_ID
CURRENTS_UA = [400.0, 600.0, 800.0, 1100.0, 1500.0, 1800.0, 2000.0]
DT_NS = 0.025
T_END_NS = 5_000.0
TRACE_MAX_POINTS = 10_000


def _turn_count(values: np.ndarray) -> float:
    d = np.diff(values)
    return float(np.sum((d[:-1] * d[1:]) < 0.0)) if d.size >= 2 else 0.0


def _spectrum(current_uA: float, out: dict[str, np.ndarray], late: np.ndarray) -> pd.DataFrame:
    t = out["t"][late]
    v = out["V_vo2"][late]
    centered = v - float(np.mean(v))
    windowed = centered * np.hanning(centered.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freq_mhz = np.fft.rfftfreq(centered.size, d=float(np.median(np.diff(t)))) * 1e-6
    gain_linear = spectrum / max(abs(current_uA) * 1e-6 * 50.0, 1e-18)
    mask = (freq_mhz >= 1.0) & (freq_mhz <= 1_000.0)
    return pd.DataFrame(
        {
            "I_target_uA": current_uA,
            "input_power_dBm": 10.0
            * math.log10(max((current_uA * 1e-6) ** 2 * 50.0 / 1e-3, 1e-30)),
            "freq_MHz": freq_mhz[mask],
            "gain_linear": gain_linear[mask],
            "gain_dB": 20.0 * np.log10(np.maximum(gain_linear[mask], 1e-12)),
        }
    )


def main() -> None:
    job_path = JOB_DIR / "job.json"
    job = json.loads(job_path.read_text())
    current_params = dict(job["params"]["current_params"])
    current_params["resist_params"] = YuanhangResistParams(**current_params["resist_params"])
    current_params["dt_s"] = DT_NS * 1e-9
    current_params["t_end_s"] = T_END_NS * 1e-9
    current_params.pop("hysteresis_reversal_mode", None)
    params = CurrentDriveParams(**current_params)

    traces: list[pd.DataFrame] = []
    summaries: list[dict[str, float]] = []
    spectra: list[pd.DataFrame] = []
    for index, current_uA in enumerate(CURRENTS_UA, start=1):
        out = simulate_current_step(current_uA, params=params, seed=0)
        late = out["t"] >= 0.60 * out["t"][-1]
        voltage_mV = out["V_vo2"][late] * 1e3
        summaries.append(
            {
                "I_target_uA": current_uA,
                "I_avg_uA": float(np.mean(out["I_in"][late]) * 1e6),
                "V_avg_mV": float(np.mean(voltage_mV)),
                "V_std_mV": float(np.std(voltage_mV)),
                "V_pp_mV": float(np.ptp(voltage_mV)),
                "turn_count": _turn_count(voltage_mV),
                "input_power_dBm": 10.0
                * math.log10(max((current_uA * 1e-6) ** 2 * 50.0 / 1e-3, 1e-30)),
            }
        )
        stride = max(1, int(math.ceil(out["t"].size / TRACE_MAX_POINTS)))
        traces.append(
            pd.DataFrame(
                {
                    "I_target_uA": current_uA,
                    "time_ns": out["t"][::stride] * 1e9,
                    "I_in_uA": out["I_in"][::stride] * 1e6,
                    "V_vo2_mV": out["V_vo2"][::stride] * 1e3,
                    "T_K": out["T"][::stride],
                    "R_ohm": out["R"][::stride],
                    "P_W": out["P"][::stride],
                    "g_eq": out["g_eq"][::stride],
                    "g_dynamic": out["g_dyn"][::stride],
                }
            )
        )
        spectra.append(_spectrum(current_uA, out, late))
        print(f"{index}/{len(CURRENTS_UA)} generated {current_uA:g} uA", flush=True)

    traces_path = JOB_DIR / "current_sweep_traces.csv"
    summary_path = JOB_DIR / "current_sweep_summary.csv"
    spectra_path = JOB_DIR / "current_sweep_spectra.csv"
    pd.concat(traces, ignore_index=True).to_csv(traces_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.concat(spectra, ignore_index=True).to_csv(spectra_path, index=False)

    retained = [
        output
        for output in job["outputs"]
        if not output["path"].endswith(
            ("current_domain_scan_detail.csv", "current_domain_scan_summary.csv")
        )
    ]
    slider_outputs = [
        {"label": "Current sweep traces CSV", "path": str(traces_path)},
        {"label": "Current sweep summary CSV", "path": str(summary_path)},
        {"label": "Current sweep spectra CSV", "path": str(spectra_path)},
    ]
    job["outputs"] = slider_outputs + retained
    job["type"] = "current_sweep"
    job["params"]["type"] = "current_sweep"
    job["params"]["I_start_uA"] = int(min(CURRENTS_UA))
    job["params"]["I_stop_uA"] = int(max(CURRENTS_UA))
    job["params"]["I_step_uA"] = 0
    job["params"]["displayed_currents_uA"] = CURRENTS_UA
    job["params"]["slider_reference_dt_ns"] = DT_NS
    job["params"]["presentation"] = (
        "Original current-sweep History representation with a current trace slider. "
        "The convergence audit remains available in the supporting downloads."
    )
    job_path.write_text(json.dumps(job, indent=2))

    with (JOB_DIR / "log.txt").open("a") as log:
        log.write("[presentation] converted History view from timestep-domain map to current-sweep slider\n")
        log.write(f"[presentation] slider currents={CURRENTS_UA}, reference dt={DT_NS} ns\n")
        log.write("[presentation] retained convergence audit as supporting downloads\n")
    print(json.dumps({"job_id": JOB_ID, "type": job["type"], "currents_uA": CURRENTS_UA}, indent=2))


if __name__ == "__main__":
    main()
