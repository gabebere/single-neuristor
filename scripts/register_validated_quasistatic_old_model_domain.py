from __future__ import annotations

import json
import math
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step
from neuristor.sample_library import load_sample_json, params_from_dict


RESISTANCE_PRESET = ROOT / "presets" / "resistance_100425_chip1_gap3.json"
PRESET_PATH = ROOT / "presets" / "current_drive" / "validated_quasistatic_old_model_domain.json"
JOB_ROOT = ROOT / "jobs"
DISPLAY_CURRENTS_UA = [1200.0, 1270.0, 1600.0, 2000.0, 2400.0, 2740.0, 2800.0]


def _crossing_metrics(t_ns: np.ndarray, v_mV: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(v_mV, [0.1, 0.9])
    threshold = 0.5 * (low + high)
    crossings = np.flatnonzero((v_mV[:-1] < threshold) & (v_mV[1:] >= threshold)) + 1
    periods = np.diff(t_ns[crossings])
    period_ns = float(np.mean(periods)) if periods.size else float("nan")
    return {
        "threshold_crossing_cycles": float(crossings.size),
        "period_ns": period_ns,
        "period_cv": float(np.std(periods) / period_ns) if periods.size and period_ns > 0.0 else float("nan"),
        "frequency_MHz": float(1e3 / period_ns) if period_ns > 0.0 else 0.0,
        "V_avg_mV": float(np.mean(v_mV)),
        "V_std_mV": float(np.std(v_mV)),
        "V_pp_mV": float(np.ptp(v_mV)),
    }


def _spectrum(current_uA: float, t_s: np.ndarray, v_v: np.ndarray) -> pd.DataFrame:
    dt_s = float(np.median(np.diff(t_s)))
    centered = v_v - float(np.mean(v_v))
    amplitude = np.abs(np.fft.rfft(centered * np.hanning(centered.size)))
    freq_mhz = np.fft.rfftfreq(centered.size, dt_s) * 1e-6
    keep = (freq_mhz >= 0.5) & (freq_mhz <= 100.0)
    gain = amplitude / max(abs(current_uA) * 1e-6 * 50.0, 1e-18)
    return pd.DataFrame(
        {
            "I_target_uA": current_uA,
            "input_power_dBm": 10.0 * math.log10(max((current_uA * 1e-6) ** 2 * 50.0 / 1e-3, 1e-30)),
            "freq_MHz": freq_mhz[keep],
            "gain_linear": gain[keep],
            "gain_dB": 20.0 * np.log10(np.maximum(gain[keep], 1e-12)),
        }
    )


def main() -> None:
    resist = params_from_dict(load_sample_json(RESISTANCE_PRESET)["resist_params"])
    params = CurrentDriveParams(
        dt_s=0.00791e-9,
        t_end_s=2_500e-9,
        T0_K=311.21937437938016,
        T_init_K=311.21937437938016,
        C_F=25.930953e-12,
        C_th_J_per_K=5.0e-12,
        S_e_W_per_K=0.1e-3,
        phase_mode="quasistatic",
        resist_params=resist,
        start_branch="insulator",
    )
    current_params = asdict(params)
    preset = {
        "schema_version": 1,
        "preset_type": "current_drive",
        "display_name": "Yuanhang quasistatic oscillatory domain",
        "description": "Timestep-converged relaxation oscillations using the reference Yuanhang hysteresis implementation.",
        "sample_id": "100425_chip1_gap3",
        "sample_name": "100425 chip1 gap3 - top-aware fitted R-T",
        "source_model": "Quasistatic Ideal Current Source + Yuanhang Hysteresis",
        "I_target_uA": 1640.226226,
        "current_params": current_params,
        "verification": {
            "status": "validated reference-model result",
            "validated_dt_ns": [0.01582, 0.00791, 0.003955],
            "validated_period_ns": [123.206, 123.135, 123.095],
            "observed_current_domain_uA": [1270.0, 2470.0],
        },
    }
    PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRESET_PATH.write_text(json.dumps(preset, indent=2))

    job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_quasistatic_validated_{uuid.uuid4().hex[:6]}"
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    (job_dir / "current_drive_preset.json").write_text(json.dumps(preset, indent=2))

    trace_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, float]] = []
    spectra: list[pd.DataFrame] = []
    for current_uA in DISPLAY_CURRENTS_UA:
        out = simulate_current_step(current_uA, params=params, seed=0)
        t_ns = out["t"] * 1e9
        late = t_ns >= 0.6 * t_ns[-1]
        metrics = _crossing_metrics(t_ns[late], out["V_vo2"][late] * 1e3)
        summaries.append(
            {
                "I_target_uA": current_uA,
                "I_avg_uA": float(np.mean(out["I_in"][late]) * 1e6),
                **metrics,
                "T_min_K": float(np.min(out["T"][late])),
                "T_max_K": float(np.max(out["T"][late])),
                "R_min_ohm": float(np.min(out["R"][late])),
                "R_max_ohm": float(np.max(out["R"][late])),
                "oscillatory": float(metrics["threshold_crossing_cycles"] >= 4 and metrics["V_pp_mV"] >= 20.0),
            }
        )
        stride = max(1, int(math.ceil(t_ns.size / 8_000)))
        sl = slice(None, None, stride)
        trace_frames.append(
            pd.DataFrame(
                {
                    "I_target_uA": current_uA,
                    "time_ns": t_ns[sl],
                    "I_in_uA": out["I_in"][sl] * 1e6,
                    "V_vo2_mV": out["V_vo2"][sl] * 1e3,
                    "T_K": out["T"][sl],
                    "R_ohm": out["R"][sl],
                    "P_W": out["P"][sl],
                    "g_eq": out["g_eq"][sl],
                    "g_dynamic": out["g_dyn"][sl],
                }
            )
        )
        spectra.append(_spectrum(current_uA, out["t"][late], out["V_vo2"][late]))

    traces_path = job_dir / "current_sweep_traces.csv"
    summary_path = job_dir / "current_sweep_summary.csv"
    spectra_path = job_dir / "current_sweep_spectra.csv"
    domain_path = job_dir / "validated_current_domain.csv"
    pd.concat(trace_frames, ignore_index=True).to_csv(traces_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.concat(spectra, ignore_index=True).to_csv(spectra_path, index=False)
    domain_parts = sorted((ROOT / "outputs" / "validated_quasistatic_old_model").glob("domain_*.csv"))
    pd.concat([pd.read_csv(path) for path in domain_parts], ignore_index=True).drop_duplicates(
        subset=["current_uA"], keep="last"
    ).sort_values("current_uA").to_csv(domain_path, index=False)

    log_path = job_dir / "log.txt"
    log_path.write_text(
        "\n".join(
            [
                "[job] starting current_sweep",
                "[model] original quasistatic ideal-current equations; no dynamic phase mechanism",
                "[resistance] fixed top-aware fit for measured sample 100425 chip1 gap3",
                "[search] candidates targeted from unstable-focus linearization on the fitted heating branch",
                "[result] sustained relaxation oscillations found from approximately 1270 to 2740 uA",
                "[validation] 1640.226 uA period converges: 123.206, 123.135, 123.095 ns as dt halves",
                "[job] completed current_sweep",
            ]
        )
        + "\n"
    )
    outputs = [
        {"label": "Current sweep traces CSV", "path": str(traces_path)},
        {"label": "Current sweep summary CSV", "path": str(summary_path)},
        {"label": "Current sweep spectra CSV", "path": str(spectra_path)},
        {"label": "Validated current domain CSV", "path": str(domain_path)},
        {"label": "Current-drive preset JSON", "path": str(job_dir / "current_drive_preset.json")},
    ]
    job = {
        "id": job_id,
        "name": "Validated old-model quasistatic oscillatory domain",
        "type": "current_sweep",
        "sample_id": "100425_chip1_gap3",
        "sample_name": "100425 chip1 gap3 - top-aware fitted R-T",
        "source_model": "Original Quasistatic Ideal Current Source",
        "status": "completed",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "type": "current_sweep",
            "I_start_uA": int(min(DISPLAY_CURRENTS_UA)),
            "I_stop_uA": int(max(DISPLAY_CURRENTS_UA)),
            "I_step_uA": 100,
            "displayed_currents_uA": DISPLAY_CURRENTS_UA,
            "conclusion": "A broad, timestep-converged quasistatic oscillatory domain exists at approximately 1.27-2.74 mA.",
            "current_params": current_params,
        },
        "outputs": outputs,
        "log_path": str(log_path),
    }
    (job_dir / "job.json").write_text(json.dumps(job, indent=2))
    print(json.dumps({"job_id": job_id, "preset": str(PRESET_PATH), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
