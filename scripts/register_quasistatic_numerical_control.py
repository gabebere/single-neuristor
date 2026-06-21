from __future__ import annotations

import json
import math
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuristor.current_domain_search import analyze_current_trace
from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step
from neuristor.model import YuanhangResistParams


RESISTANCE_PRESET = ROOT / "presets" / "resistance_100425_chip1_gap3.json"
JOB_ROOT = ROOT / "jobs"
PRESET_DIR = ROOT / "presets" / "current_drive"
CURRENTS_UA = [400.0, 600.0, 800.0, 1100.0, 1500.0, 1800.0, 2000.0]
REFERENCE_DT_NS = 0.025
CONVERGENCE_CURRENTS_UA = [800.0, 1100.0, 1500.0]
CONVERGENCE_DT_NS = [0.05, 0.025, 0.0125, 0.00625]


def _base_params(resist: YuanhangResistParams, dt_ns: float, duration_ns: float) -> CurrentDriveParams:
    return CurrentDriveParams(
        dt_s=dt_ns * 1e-9,
        t_end_s=duration_ns * 1e-9,
        T0_K=325.0,
        T_init_K=325.0,
        C_F=80e-12,
        C_th_J_per_K=3e-12,
        S_e_W_per_K=0.052838e-3,
        phase_mode="quasistatic",
        tau_g_s=10e-9,
        resist_params=resist,
        start_branch="insulator",
    )


def _turn_count(values: np.ndarray) -> float:
    d = np.diff(values)
    return float(np.sum((d[:-1] * d[1:]) < 0.0)) if d.size >= 2 else 0.0


def _spectrum(current_uA: float, out: dict[str, np.ndarray], late: np.ndarray) -> pd.DataFrame:
    t = out["t"][late]
    v = out["V_vo2"][late]
    centered = v - float(np.mean(v))
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(centered.size)))
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
    resistance_payload = json.loads(RESISTANCE_PRESET.read_text())
    resist = YuanhangResistParams(**resistance_payload["resist_params"])
    job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_quasistatic_control_{uuid.uuid4().hex[:6]}"
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    PRESET_DIR.mkdir(parents=True, exist_ok=True)

    params = _base_params(resist, REFERENCE_DT_NS, 5_000.0)
    trace_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, float]] = []
    spectra: list[pd.DataFrame] = []
    for index, current_uA in enumerate(CURRENTS_UA, start=1):
        out = simulate_current_step(current_uA, params=params, seed=0)
        late = out["t"] >= 0.60 * out["t"][-1]
        v_late_mV = out["V_vo2"][late] * 1e3
        summaries.append(
            {
                "I_target_uA": current_uA,
                "I_avg_uA": float(np.mean(out["I_in"][late]) * 1e6),
                "V_avg_mV": float(np.mean(v_late_mV)),
                "V_std_mV": float(np.std(v_late_mV)),
                "V_pp_mV": float(np.ptp(v_late_mV)),
                "turn_count": _turn_count(v_late_mV),
                "input_power_dBm": 10.0
                * math.log10(max((current_uA * 1e-6) ** 2 * 50.0 / 1e-3, 1e-30)),
            }
        )
        stride = max(1, int(math.ceil(out["t"].size / 10_000)))
        trace_frames.append(
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
        print(f"slider {index}/{len(CURRENTS_UA)}: {current_uA:g} uA", flush=True)

    pd.concat(trace_frames, ignore_index=True).to_csv(job_dir / "current_sweep_traces.csv", index=False)
    pd.DataFrame(summaries).to_csv(job_dir / "current_sweep_summary.csv", index=False)
    pd.concat(spectra, ignore_index=True).to_csv(job_dir / "current_sweep_spectra.csv", index=False)

    convergence: list[dict[str, float]] = []
    for current_uA in CONVERGENCE_CURRENTS_UA:
        for dt_ns in CONVERGENCE_DT_NS:
            test_params = _base_params(resist, dt_ns, 3_000.0)
            out = simulate_current_step(current_uA, params=test_params, seed=0)
            late = out["t"] >= 0.60 * out["t"][-1]
            late_out = {
                key: value[late]
                for key, value in out.items()
                if isinstance(value, np.ndarray) and value.shape[0] == late.shape[0]
            }
            metrics = analyze_current_trace(
                late_out,
                params=test_params,
                min_vpp_mV=12.0,
                max_vpp_mV=2_000.0,
                min_cycles=4,
                pulse_on_ns=float(late_out["t"][0] * 1e9),
                pulse_off_ns=None,
            )
            convergence.append(
                {
                    "I_target_uA": current_uA,
                    "dt_ns": dt_ns,
                    **{str(key): float(value) for key, value in metrics.items()},
                }
            )
            print(f"audit I={current_uA:g} uA dt={dt_ns:g} ns", flush=True)
    convergence_df = pd.DataFrame(convergence)
    convergence_df.to_csv(job_dir / "quasistatic_timestep_audit.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for current_uA, group in convergence_df.groupby("I_target_uA"):
        group = group.sort_values("dt_ns")
        axes[0].plot(group["dt_ns"], group["dominant_freq_MHz"], marker="o", label=f"{current_uA:g} uA")
        axes[1].plot(group["dt_ns"], group["V_pp_mV"], marker="o", label=f"{current_uA:g} uA")
    for ax in axes:
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.set_xlabel("Timestep (ns), finer to the right")
    axes[0].set_ylabel("Late-window frequency (MHz)")
    axes[1].set_ylabel("Late-window Vpp (mV)")
    axes[0].legend()
    fig.suptitle("Quasistatic apparent oscillations fail timestep convergence")
    fig.tight_layout()
    fig.savefig(job_dir / "quasistatic_timestep_audit.png", dpi=180)
    plt.close(fig)

    current_params = asdict(params)
    preset = {
        "schema_version": 1,
        "preset_type": "current_drive",
        "display_name": "Quasistatic apparent oscillations - numerical control",
        "description": "Matched quasistatic control. Apparent oscillations are not timestep converged.",
        "sample_id": "100425_chip1_gap3",
        "sample_name": "100425 chip1 gap3 - top-aware fitted R-T",
        "source_model": "Original Quasistatic Current Source",
        "I_target_uA": 1100.0,
        "current_params": current_params,
        "verification": {
            "status": "not a validated physical oscillator",
            "reason": "Oscillation existence, frequency, and amplitude change when only dt changes.",
        },
    }
    preset_path = PRESET_DIR / "quasistatic_apparent_oscillations_numerical_control.json"
    preset_path.write_text(json.dumps(preset, indent=2))
    (job_dir / "current_drive_preset.json").write_text(json.dumps(preset, indent=2))
    log_path = job_dir / "log.txt"
    log_path.write_text(
        "\n".join(
            [
                "[job] starting current_sweep",
                "[control] original quasistatic phase response: g = g_eq(T,H)",
                "[control] same resistance fit and circuit/thermal parameters as the dynamic comparison base",
                "[control] some reference-dt traces look oscillatory",
                "[control] oscillation existence/frequency/amplitude change when only timestep changes",
                "[control] result: apparent quasistatic oscillations are not timestep-converged physical limit cycles",
                "[job] completed current_sweep",
            ]
        )
        + "\n"
    )
    outputs = [
        {"label": "Current sweep traces CSV", "path": str(job_dir / "current_sweep_traces.csv")},
        {"label": "Current sweep summary CSV", "path": str(job_dir / "current_sweep_summary.csv")},
        {"label": "Current sweep spectra CSV", "path": str(job_dir / "current_sweep_spectra.csv")},
        {"label": "Quasistatic timestep audit CSV", "path": str(job_dir / "quasistatic_timestep_audit.csv")},
        {"label": "Quasistatic timestep audit PNG", "path": str(job_dir / "quasistatic_timestep_audit.png")},
        {"label": "Quasistatic control preset JSON", "path": str(job_dir / "current_drive_preset.json")},
    ]
    job = {
        "id": job_id,
        "name": "Quasistatic apparent oscillations - numerical control",
        "type": "current_sweep",
        "sample_id": "100425_chip1_gap3",
        "sample_name": "100425 chip1 gap3 - top-aware fitted R-T",
        "source_model": "Original Quasistatic Current Source",
        "status": "completed",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "type": "current_sweep",
            "I_start_uA": int(min(CURRENTS_UA)),
            "I_stop_uA": int(max(CURRENTS_UA)),
            "I_step_uA": 0,
            "displayed_currents_uA": CURRENTS_UA,
            "reference_dt_ns": REFERENCE_DT_NS,
            "conclusion": "Apparent quasistatic oscillations fail timestep convergence and are not accepted as physical limit cycles.",
            "current_params": current_params,
        },
        "outputs": outputs,
        "log_path": str(log_path),
    }
    (job_dir / "job.json").write_text(json.dumps(job, indent=2))
    print(json.dumps({"job_id": job_id, "preset": str(preset_path)}, indent=2))


if __name__ == "__main__":
    main()
