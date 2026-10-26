"""Reproduce the bounded September 21 audit; run from the repository root.

This is a run-owned investigation script, not an alternate physics model.
Major-branch linearization freezes hysteresis memory and is not a bifurcation
calculation for the complete nonsmooth hysteretic system.
"""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from neuristor.config import load_toml, resolved_copy
from neuristor.current_drive_sim import simulate_current_waveforms
from neuristor.experimental_waveforms import load_converted_sweep
from neuristor.model import HysteresisArray
from neuristor.model_validation import _baseline_corrected_trace
from neuristor.oscillation_audit import audit_voltage
from neuristor.runs import RunBundle
from neuristor.workflows import _current_params_from_config

config = load_toml("experiments/current/specimen_model_validation.toml")
p, *_ = _current_params_from_config(config)
frames, _ = load_converted_sweep("data/experimental/tia_current_sweep")
groups = list(frames.groupby("nominal_drive_mV", sort=True))
time, _, _ = _baseline_corrected_trace(groups[0][1])
inputs = np.column_stack([_baseline_corrected_trace(f)[1] for _, f in groups])
late = (time >= 150) & (time <= 250)
bundle = RunBundle.create(
    name="Independent units stability and specimen replay audit",
    model="current-audit", kind="analysis", output_root="runs",
    command="python PATH_TO_THIS_BUNDLE/reproduce.py (from repository root)",
    config={"recipe": resolved_copy(config), "parameters_SI": asdict(p),
            "dt_ns": [0.05, 0.025, 0.0125], "late_window_ns": [150, 250],
            "derivative_halfwidths_K": [0.01, 0.02, 0.04],
            "user_confirmation": "R(T) measured on same device; schematic unavailable",
            "stability_scope": "Fresh heating/cooling major branches, frozen memory"},
)
bundle.write_text("reproduce.py", Path(__file__).read_text(), label="Exact investigation script", media_type="text/x-python")
paths = list(Path("src/neuristor").glob("*.py")) + [Path(config["_source"]), Path("presets/resistance_100425_chip1_gap3.json")]
paths += list(Path("data/experimental/tia_current_sweep").glob("*_converted.csv"))
bundle.write_json("source_hashes.json", {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}, label="Input and source SHA256")
bundle.write_text("working_tree.patch", subprocess.check_output(["git", "diff"], text=True), label="Pre-existing and current working tree changes", media_type="text/plain")

replays = []
for dt_ns in (0.05, 0.025, 0.0125):
    local = replace(p, dt_s=dt_ns*1e-9, t_pre_s=-time[0]*1e-9, t_end_s=time[-1]*1e-9)
    outputs = simulate_current_waveforms(inputs, local, waveform_time_s=time*1e-9)
    for (label, frame), out in zip(groups, outputs):
        t, i, v = _baseline_corrected_trace(frame)
        prediction = np.interp(time*1e-9, out["t"], out["V_vo2"])*1000
        measured, _ = audit_voltage(time, v)
        predicted, _ = audit_voltage(time, prediction)
        replays.append({"source_mV": label, "dt_ns": dt_ns,
                        "current_uA": float(i[late].mean()),
                        **{f"measured_{k}": value for k, value in measured.items()},
                        **{f"predicted_{k}": value for k, value in predicted.items()},
                        "temperature_min_K": float(out["T"].min()),
                        "temperature_max_K": float(out["T"].max())})
    print("Finished all 22 currents at dt_ns =", dt_ns, flush=True)
replays = pd.DataFrame(replays)
replays.to_csv(bundle.add_artifact("replay.csv", label="All-current refinement and persistence", media_type="text/csv"), index=False)

def major_resistance(T, branch):
    h = HysteresisArray(p.resist_params, size=1, start_branch=branch)
    h.initialize(np.array([T]))
    return float(h.evaluate(np.array([T]))[0][0])

rows = []
for branch in ("insulator", "metal"):
    for current_uA in replays.current_uA.unique():
        I = current_uA*1e-6
        T = brentq(lambda T: I**2*major_resistance(T, branch)-p.S_e_W_per_K*(T-p.T0_K), p.T0_K, p.resist_params.T_max_K)
        R = major_resistance(T, branch)
        for h in (0.01, 0.02, 0.04):
            derivative = (major_resistance(T+h, branch)-major_resistance(T-h, branch))/(2*h)
            driving = -I**2*derivative-p.S_e_W_per_K
            Ccrit = p.C_th_J_per_K/(R*driving) if driving > 0 else np.nan
            jacobian = np.array([[-1/(p.C_F*R), I*derivative/(p.C_F*R)],
                                  [2*I/p.C_th_J_per_K, driving/p.C_th_J_per_K]])
            rows.append(dict(branch=branch, current_uA=current_uA, T_K=T, R_ohm=R,
                             derivative_halfwidth_K=h, dR_dT_ohm_per_K=derivative,
                             Ccrit_pF=Ccrit*1e12,
                             maximum_real_eigenvalue_per_s=max(np.linalg.eigvals(jacobian).real)))
stability = pd.DataFrame(rows)
stability.to_csv(bundle.add_artifact("major_branch_stability.csv", label="Conditional local stability", media_type="text/csv"), index=False)
fine = replays[replays.dt_ns == 0.0125]
mid = replays[replays.dt_ns == 0.025]
metrics = {
    "measured_persistent_count": int(fine.measured_sustained.sum()),
    "predicted_persistent_count": int(fine.predicted_sustained.sum()),
    "late_mean_voltage_RMSE_mV": float(np.sqrt(np.mean((fine.predicted_late_mean_mV-fine.measured_late_mean_mV)**2))),
    "maximum_late_mean_change_025_to_0125_mV": float(np.max(np.abs(mid.predicted_late_mean_mV.to_numpy()-fine.predicted_late_mean_mV.to_numpy()))),
    "minimum_major_branch_Ccrit_pF": float(stability.Ccrit_pF.min()),
    "all_major_branch_equilibria_locally_stable_at_039pF": bool((stability.maximum_real_eigenvalue_per_s < 0).all()),
    "conditional_active_volume_um3_at_cv_3e6": p.C_th_J_per_K/3e6*1e18,
    "conditional_width_um_at_gap_02um_thickness_015um": p.C_th_J_per_K/3e6*1e18/(0.2*0.15),
}
bundle.write_json("metrics.json", metrics, label="Audit summary")
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
axes[0].plot(fine.current_uA, fine.measured_late_mean_mV, "o-", label="Measured output")
axes[0].plot(fine.current_uA, fine.predicted_late_mean_mV, "s-", label="Frozen model, 0.0125 ns")
axes[0].set(xlabel="Late input current (µA)", ylabel="Late mean voltage (mV)", title="Same parameters across all 22 records")
axes[0].legend()
for branch, group in stability[stability.derivative_halfwidth_K == 0.02].groupby("branch"):
    axes[1].plot(group.current_uA, group.Ccrit_pF, "o-", label=f"{branch} major branch")
axes[1].axhline(0.39, color="black", linestyle="--", label="Adopted C = 0.39 pF")
axes[1].set(xlabel="Current (µA)", ylabel="Local trace-zero capacitance (pF)", title="Frozen-memory diagnostic; not a limit-cycle boundary", ylim=(0, 20))
axes[1].legend(fontsize=8)
fig.savefig(bundle.add_artifact("figures/audit.png", label="Replay and conditional stability", media_type="image/png"), dpi=180)
plt.close(fig)
bundle.write_text("report.md", "# Independent discrepancy audit\n\nAll 22 measured inputs were replayed at three timesteps. No fitted parameters were changed.\n\n" + "\n".join(f"- {k}: {v}" for k,v in metrics.items()) + "\n\nThe stability table linearizes each fresh major branch with hysteresis memory frozen. It is neither a full hysteretic bifurcation proof nor a calibrated replacement capacitance. The active-volume calculation assumes all fitted heat capacity belongs to VO2 with volumetric heat capacity 3 MJ/m³/K; it is a geometric plausibility check only. Source hashes and the dirty-tree patch preserve the examined state. See docs/DISCREPANCY_AUDIT_20260921.md for derivation, literature, bug scope, and experiments.", label="Audit report")
bundle.complete(summary=metrics)
print(bundle.root, flush=True)
print(metrics, flush=True)
