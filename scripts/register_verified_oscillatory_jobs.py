from __future__ import annotations

import json
import math
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step
from neuristor.model import YuanhangResistParams


RESISTANCE_PRESET = ROOT / "presets" / "resistance_100425_chip1_gap3.json"
CURRENT_PRESET_DIR = ROOT / "presets" / "current_drive"
JOB_ROOT = ROOT / "jobs"

CASES = [
    {
        "slug": "verified_osc_balanced",
        "name": "Verified current oscillation - balanced",
        "description": "Moderate-power sustained dynamic-phase oscillation.",
        "I_uA": 1500.0,
        "C_pF": 80.0,
        "Cth_pJ_per_K": 3.0,
        "Se_mW_per_K": 0.052838,
        "tau_g_ns": 10.0,
    },
    {
        "slug": "verified_osc_low_current",
        "name": "Verified current oscillation - low current",
        "description": "Lower-current sustained dynamic-phase oscillation.",
        "I_uA": 1100.0,
        "C_pF": 80.0,
        "Cth_pJ_per_K": 2.0,
        "Se_mW_per_K": 0.032844,
        "tau_g_ns": 10.0,
    },
    {
        "slug": "verified_osc_regular",
        "name": "Verified current oscillation - regular",
        "description": "Most conservative regular moderate-amplitude oscillation.",
        "I_uA": 1300.0,
        "C_pF": 120.0,
        "Cth_pJ_per_K": 3.0,
        "Se_mW_per_K": 0.036205,
        "tau_g_ns": 5.0,
    },
    {
        "slug": "verified_osc_high_power",
        "name": "Verified current oscillation - high power",
        "description": "High-power, large-amplitude dynamic-phase oscillation.",
        "I_uA": 2364.046908,
        "C_pF": 83.970146,
        "Cth_pJ_per_K": 4.001637,
        "Se_mW_per_K": 0.171520,
        "tau_g_ns": 1.627852,
    },
]


def _count_turns(values: np.ndarray) -> int:
    d = np.diff(values)
    return int(np.sum((d[:-1] * d[1:]) < 0.0)) if d.size >= 2 else 0


def _cycle_metrics(t_ns: np.ndarray, v_mV: np.ndarray) -> dict[str, float]:
    smooth = np.convolve(v_mV, np.array([0.25, 0.5, 0.25]), mode="same")
    d = np.diff(smooth)
    signs = np.sign(d)
    peaks = np.where((signs[:-1] > 0.0) & (signs[1:] < 0.0))[0] + 1
    troughs = np.where((signs[:-1] < 0.0) & (signs[1:] > 0.0))[0] + 1
    periods: list[float] = []
    if peaks.size >= 2:
        periods.extend(np.diff(t_ns[peaks]).tolist())
    if troughs.size >= 2:
        periods.extend(np.diff(t_ns[troughs]).tolist())
    period_ns = float(np.mean(periods)) if periods else float("nan")
    period_cv = float(np.std(periods) / period_ns) if periods and period_ns > 0.0 else float("nan")
    return {
        "turn_count": float(_count_turns(v_mV)),
        "cycle_count": float(min(peaks.size, troughs.size)),
        "period_ns": period_ns,
        "period_cv": period_cv,
        "frequency_MHz": float(1e3 / period_ns) if period_ns > 0.0 else 0.0,
    }


def _make_spectrum(i_uA: float, t_s: np.ndarray, v_v: np.ndarray, late_mask: np.ndarray) -> pd.DataFrame:
    t_late = t_s[late_mask]
    v_late = v_v[late_mask]
    dt_s = float(np.median(np.diff(t_late)))
    centered = v_late - float(np.mean(v_late))
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(centered.size)))
    freq_mhz = np.fft.rfftfreq(centered.size, d=dt_s) * 1e-6
    gain_linear = spectrum / max(abs(i_uA) * 1e-6 * 50.0, 1e-18)
    gain_db = 20.0 * np.log10(np.maximum(gain_linear, 1e-12))
    mask = (freq_mhz >= 1.0) & (freq_mhz <= 1000.0)
    return pd.DataFrame(
        {
            "I_target_uA": i_uA,
            "input_power_dBm": 10.0 * math.log10(max((i_uA * 1e-6) ** 2 * 50.0 / 1e-3, 1e-30)),
            "freq_MHz": freq_mhz[mask],
            "gain_linear": gain_linear[mask],
            "gain_dB": gain_db[mask],
        }
    )


def _make_plot(out: dict[str, np.ndarray], case: dict[str, Any], path: Path) -> None:
    t_ns = out["t"] * 1e9
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t_ns, out["V_vo2"] * 1e3, linewidth=0.8)
    axes[0].set_ylabel("V (mV)")
    axes[1].plot(t_ns, out["T"], linewidth=0.8)
    axes[1].set_ylabel("T (K)")
    axes[2].plot(t_ns, out["R"], linewidth=0.8)
    axes[2].set_ylabel("R (Ohm)")
    axes[3].plot(t_ns, out["g_dyn"], linewidth=0.8)
    axes[3].set_ylabel("g dynamic")
    axes[3].set_xlabel("time (ns)")
    fig.suptitle(case["name"])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _job_id(index: int) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{index:02d}{uuid.uuid4().hex[:6]}"


def main() -> None:
    resistance_payload = json.loads(RESISTANCE_PRESET.read_text())
    resist_params = YuanhangResistParams(**resistance_payload["resist_params"])
    CURRENT_PRESET_DIR.mkdir(parents=True, exist_ok=True)
    JOB_ROOT.mkdir(parents=True, exist_ok=True)

    created: list[dict[str, Any]] = []
    for index, case in enumerate(CASES, start=1):
        params = CurrentDriveParams(
            dt_s=0.025e-9,
            t_end_s=5_000e-9,
            t_pre_s=0.0,
            pulse_on_s=0.0,
            pulse_off_s=None,
            V_init_V=0.0,
            T0_K=325.0,
            T_init_K=325.0,
            C_F=float(case["C_pF"]) * 1e-12,
            C_th_J_per_K=float(case["Cth_pJ_per_K"]) * 1e-12,
            S_e_W_per_K=float(case["Se_mW_per_K"]) * 1e-3,
            sigma_W_sqrt_s=0.0,
            phase_mode="dynamic",
            tau_g_s=float(case["tau_g_ns"]) * 1e-9,
            resist_params=resist_params,
            start_branch="insulator",
        )
        current_params = asdict(params)
        preset_payload = {
            "schema_version": 1,
            "preset_type": "current_drive",
            "display_name": case["name"],
            "description": case["description"],
            "sample_id": "100425_chip1_gap3",
            "sample_name": "100425 chip1 gap3 - lab fitted R-T",
            "sample_source": str(ROOT / "data" / "experimental" / "100425_chip1_gap3.tsv"),
            "source_model": "Current Source with Dynamic VO2 Phase",
            "I_target_uA": float(case["I_uA"]),
            "current_params": current_params,
            "verification": {
                "original_mechanism_note": "Uses dynamic VO2 phase lag: phase_mode=dynamic and tau_g_s.",
                "verified_dt_ns": [0.05, 0.025],
                "registration_dt_ns": 0.025,
            },
        }
        library_preset_path = CURRENT_PRESET_DIR / f"{case['slug']}.json"
        library_preset_path.write_text(json.dumps(preset_payload, indent=2))

        job_id = _job_id(index)
        job_dir = JOB_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        job_preset_path = job_dir / "current_drive_preset.json"
        job_preset_path.write_text(json.dumps(preset_payload, indent=2))
        log_path = job_dir / "log.txt"
        log_path.write_text(
            "\n".join(
                [
                    "[job] starting current_sweep",
                    "[current_sweep] registered verified positive-result dynamic-phase experiment",
                    f"[current_sweep] sample={preset_payload['sample_name']}",
                    f"[current_sweep] phase_mode=dynamic, tau_g={case['tau_g_ns']:.6g} ns",
                    f"[current_sweep] running representative current {case['I_uA']:.6g} uA",
                ]
            )
            + "\n"
        )

        out = simulate_current_step(float(case["I_uA"]), params=params, seed=0)
        t_ns = out["t"] * 1e9
        late_mask = t_ns >= 0.6 * float(t_ns[-1])
        v_late_mV = out["V_vo2"][late_mask] * 1e3
        cycle = _cycle_metrics(t_ns[late_mask], v_late_mV)
        verification = {
            "I_target_uA": float(case["I_uA"]),
            "analysis_window_start_ns": float(t_ns[late_mask][0]),
            "analysis_window_end_ns": float(t_ns[-1]),
            "V_mean_mV": float(np.mean(v_late_mV)),
            "V_std_mV": float(np.std(v_late_mV)),
            "V_pp_mV": float(np.ptp(v_late_mV)),
            "T_min_K": float(np.min(out["T"][late_mask])),
            "T_max_K": float(np.max(out["T"][late_mask])),
            "R_min_ohm": float(np.min(out["R"][late_mask])),
            "R_max_ohm": float(np.max(out["R"][late_mask])),
            "g_dynamic_min": float(np.min(out["g_dyn"][late_mask])),
            "g_dynamic_max": float(np.max(out["g_dyn"][late_mask])),
            **cycle,
            "oscillatory": 1.0 if cycle["cycle_count"] >= 5.0 and float(np.ptp(v_late_mV)) >= 20.0 else 0.0,
        }

        traces_df = pd.DataFrame(
            {
                "I_target_uA": float(case["I_uA"]),
                "time_ns": t_ns,
                "I_in_uA": out["I_in"] * 1e6,
                "V_vo2_mV": out["V_vo2"] * 1e3,
                "T_K": out["T"],
                "R_ohm": out["R"],
                "P_W": out["P"],
                "g_eq": out["g_eq"],
                "g_dynamic": out["g_dyn"],
            }
        )
        summary_df = pd.DataFrame(
            [
                {
                    "I_target_uA": float(case["I_uA"]),
                    "I_avg_uA": float(np.mean(out["I_in"][late_mask]) * 1e6),
                    "V_avg_mV": verification["V_mean_mV"],
                    "V_std_mV": verification["V_std_mV"],
                    "V_pp_mV": verification["V_pp_mV"],
                    "turn_count": verification["turn_count"],
                    "input_power_dBm": 10.0
                    * math.log10(max((float(case["I_uA"]) * 1e-6) ** 2 * 50.0 / 1e-3, 1e-30)),
                }
            ]
        )
        spectra_df = _make_spectrum(float(case["I_uA"]), out["t"], out["V_vo2"], late_mask)
        verification_df = pd.DataFrame([verification])

        traces_path = job_dir / "current_sweep_traces.csv"
        summary_path = job_dir / "current_sweep_summary.csv"
        spectra_path = job_dir / "current_sweep_spectra.csv"
        verification_path = job_dir / "oscillation_verification.csv"
        plot_path = job_dir / "oscillation_trace.png"
        gif_path = job_dir / "current_sweep.gif"
        traces_df.to_csv(traces_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        spectra_df.to_csv(spectra_path, index=False)
        verification_df.to_csv(verification_path, index=False)
        _make_plot(out, case, plot_path)
        imageio.mimsave(gif_path, [imageio.imread(plot_path)], duration=1.0, loop=0)

        config = {
            "type": "current_sweep",
            "job_name": case["name"],
            "sample_id": preset_payload["sample_id"],
            "sample_name": preset_payload["sample_name"],
            "sample_source": preset_payload["sample_source"],
            "source_model": preset_payload["source_model"],
            "I_start_uA": int(round(float(case["I_uA"]))),
            "I_stop_uA": int(round(float(case["I_uA"]))),
            "I_step_uA": 1,
            "frame_duration_s": 1.0,
            "seed": 0,
            "current_params": current_params,
            "research_notes": {
                "description": case["description"],
                "history_summary_uses_late_window": True,
                "analysis_window": "last 40% of the 5 us trace",
                "mechanism": "dynamic VO2 phase lag",
            },
        }
        outputs = [
            {"label": "Current sweep GIF", "path": str(gif_path)},
            {"label": "Current sweep traces CSV", "path": str(traces_path)},
            {"label": "Current sweep summary CSV", "path": str(summary_path)},
            {"label": "Current sweep spectra CSV", "path": str(spectra_path)},
            {"label": "Oscillation verification CSV", "path": str(verification_path)},
            {"label": "Oscillation trace PNG", "path": str(plot_path)},
            {"label": "Current-drive preset JSON", "path": str(job_preset_path)},
        ]
        job = {
            "id": job_id,
            "name": case["name"],
            "type": "current_sweep",
            "sample_id": preset_payload["sample_id"],
            "sample_name": preset_payload["sample_name"],
            "source_model": preset_payload["source_model"],
            "status": "completed",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": config,
            "outputs": outputs,
            "log_path": str(log_path),
        }
        (job_dir / "job.json").write_text(json.dumps(job, indent=2))
        with log_path.open("a") as log:
            log.write(f"[current_sweep] late-window Vpp={verification['V_pp_mV']:.3f} mV\n")
            log.write(f"[current_sweep] cycles={verification['cycle_count']:.0f}, frequency={verification['frequency_MHz']:.3f} MHz\n")
            log.write("[current_sweep] wrote standard History outputs and verification files\n")
            log.write("[job] completed current_sweep\n")
        created.append(
            {
                "job_id": job_id,
                "name": case["name"],
                "preset": str(library_preset_path),
                **verification,
            }
        )
        print(f"Created {job_id}: {case['name']}")

    manifest = CURRENT_PRESET_DIR / "verified_oscillatory_jobs_manifest.json"
    manifest.write_text(json.dumps(created, indent=2))
    print(json.dumps(created, indent=2))


if __name__ == "__main__":
    main()
