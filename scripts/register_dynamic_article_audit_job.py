from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "outputs" / "dynamic_article_comparison"
JOB_ROOT = ROOT / "jobs"
PRESET_DIR = ROOT / "presets" / "current_drive"
RESISTANCE_PRESET = ROOT / "presets" / "resistance_100425_chip1_gap3.json"


def main() -> None:
    verification = pd.read_csv(SOURCE_DIR / "verification_results.csv")
    pd.read_csv(SOURCE_DIR / "sweep_results.csv")
    job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_article_audit_{uuid.uuid4().hex[:6]}"
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    PRESET_DIR.mkdir(parents=True, exist_ok=True)

    detail = verification.copy()
    detail["scan_value"] = detail["dt_ns"]
    detail["oscillatory"] = detail["verified_sustained"]
    detail["turn_count"] = detail["n_cycles"]
    detail.to_csv(job_dir / "current_domain_scan_detail.csv", index=False)
    summary_rows = []
    for dt_ns, group in detail.groupby("scan_value"):
        accepted = group[group["oscillatory"] > 0.5]
        best = accepted.sort_values("turn_count", ascending=False).head(1)
        summary_rows.append(
            {
                "scan_value": float(dt_ns),
                "osc_fraction": float(group["oscillatory"].mean()),
                "best_current_uA": float(best["I_target_uA"].iloc[0]) if not best.empty else 0.0,
                "best_turn_count": float(group["turn_count"].max()),
                "n_points": int(len(group)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(job_dir / "current_domain_scan_summary.csv", index=False)

    for name in ("sweep_results.csv", "sweep_frequency_summary.png", "summary.json"):
        shutil.copy2(SOURCE_DIR / name, job_dir / name)
    verification.to_csv(job_dir / "timestep_convergence_audit.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, group in verification.groupby("label"):
        if "oscillatory" not in label:
            continue
        group = group.sort_values("dt_ns")
        axes[0].plot(group["dt_ns"], group["dominant_freq_MHz"], marker="o", label=label)
        axes[1].plot(group["dt_ns"], group["V_pp_mV"], marker="o", label=label)
    for ax in axes:
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.set_xlabel("Timestep (ns), finer to the right")
    axes[0].set_ylabel("Late-window frequency (MHz)")
    axes[1].set_ylabel("Late-window voltage Vpp (mV)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Dynamic-phase current-source timestep convergence audit")
    fig.tight_layout()
    fig.savefig(job_dir / "timestep_convergence_audit.png", dpi=180)
    plt.close(fig)

    resist_payload = json.loads(RESISTANCE_PRESET.read_text())
    current_params = {
        "dt_s": 2.5e-11,
        "t_end_s": 5e-6,
        "t_pre_s": 0.0,
        "pulse_on_s": 0.0,
        "pulse_off_s": None,
        "V_init_V": 0.0,
        "T0_K": 325.0,
        "T_init_K": 325.0,
        "C_F": 80e-12,
        "R_out_ohm": 0.0,
        "C_th_J_per_K": 3e-12,
        "S_e_W_per_K": 0.052838e-3,
        "sigma_W_sqrt_s": 0.0,
        "thermal_mode": "single",
        "C_sub_J_per_K": 49.62776831e-12,
        "G_hot_sub_W_per_K": 0.20558726e-3,
        "T_sub_init_K": None,
        "phase_mode": "dynamic",
        "tau_g_s": 10e-9,
        "domain_count": 1,
        "domain_temperature_span_K": 0.0,
        "domain_coupling_W_per_K": 0.0,
        "resist_params": resist_payload["resist_params"],
        "start_branch": "insulator",
    }
    preset = {
        "schema_version": 1,
        "preset_type": "current_drive",
        "display_name": "Dynamic article comparison base - convergence audit",
        "description": "Exploratory base point. Not validated as timestep-converged physical prediction.",
        "sample_id": "100425_chip1_gap3",
        "sample_name": "100425 chip1 gap3 - top-aware fitted R-T",
        "source_model": "Current Source with Dynamic VO2 Phase",
        "I_target_uA": 1500.0,
        "current_params": current_params,
        "verification": {
            "status": "not timestep converged",
            "warning": "Frequency and amplitude change sharply at dt=0.0125 ns because hysteresis reversal detection uses a per-step 0.01 K threshold.",
        },
    }
    preset_path = PRESET_DIR / "dynamic_article_comparison_base_convergence_audit.json"
    preset_path.write_text(json.dumps(preset, indent=2))
    (job_dir / "current_drive_preset.json").write_text(json.dumps(preset, indent=2))

    log_path = job_dir / "log.txt"
    log_path.write_text(
        "\n".join(
            [
                "[job] starting current_domain_scan",
                "[audit] swept current, C, C_th, S_e, and tau_g using the unchanged dynamic-phase model",
                "[audit] broad 200-600 uA / 40-60 MHz paper-target searches found no sustained cycles",
                "[audit] apparent 0.8-1.8 mA oscillations were tested at dt=0.05, 0.025, and 0.0125 ns",
                "[audit] central/high waveforms changed sharply at the finest timestep",
                "[audit] result is not promoted as a timestep-converged physical prediction",
                "[job] completed current_domain_scan",
            ]
        )
        + "\n"
    )
    outputs = []
    labels = {
        "current_domain_scan_detail.csv": "Timestep audit detail CSV",
        "current_domain_scan_summary.csv": "Timestep audit summary CSV",
        "timestep_convergence_audit.csv": "Convergence results CSV",
        "timestep_convergence_audit.png": "Convergence audit PNG",
        "sweep_results.csv": "Variable sweep results CSV",
        "sweep_frequency_summary.png": "Variable sweep frequency PNG",
        "summary.json": "Sweep summary JSON",
        "current_drive_preset.json": "Exploratory current-drive preset JSON",
    }
    for name, label in labels.items():
        outputs.append({"label": label, "path": str(job_dir / name)})
    job = {
        "id": job_id,
        "name": "Dynamic article comparison - convergence audit",
        "type": "current_domain_scan",
        "sample_id": "100425_chip1_gap3",
        "sample_name": "100425 chip1 gap3 - top-aware fitted R-T",
        "source_model": "Current Source with Dynamic VO2 Phase (working extension)",
        "status": "completed",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "type": "current_domain_scan",
            "scan_param_key": "dt_ns",
            "scan_param_label": "Timestep convergence",
            "source_model": "Current Source with Dynamic VO2 Phase (working extension)",
            "paper_target_current_uA": [200.0, 600.0],
            "paper_target_frequency_MHz": [40.0, 60.0],
            "conclusion": "No paper-like domain found; apparent higher-current domains are not timestep converged.",
            "current_params": current_params,
        },
        "outputs": outputs,
        "log_path": str(log_path),
    }
    (job_dir / "job.json").write_text(json.dumps(job, indent=2))
    print(json.dumps({"job_id": job_id, "preset": str(preset_path)}, indent=2))


if __name__ == "__main__":
    main()
