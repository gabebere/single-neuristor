from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REF_MODEL = ROOT / "references" / "yuanhangzhang98-collective_dynamics_neuristor-217d4f0" / "model.py"
SPECIMEN_PRESET = ROOT / "presets" / "resistance_100425_chip1_gap3.json"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_step
from neuristor.model import HysteresisArray, YuanhangResistParams


@dataclass
class HysteresisComparison:
    label: str
    max_abs_diff_ohm: float
    max_rel_diff: float
    worst_step: int
    worst_temperature_K: float
    local_resistance_ohm: float
    reference_resistance_ohm: float


def _load_reference_module():
    spec = importlib.util.spec_from_file_location("yuanhang_reference_model", REF_MODEL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load reference model from {REF_MODEL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_hysteresis_trace(ref_module, temperatures_K: Iterable[float], dtype: torch.dtype) -> np.ndarray:
    vo2 = ref_module.VO2(1)
    vo2.delta = vo2.delta.to(dtype)
    vo2.reversed = vo2.reversed.to(dtype)
    vo2.initialize(torch.tensor(324.9, dtype=dtype))
    out = []
    for temp in temperatures_K:
        t = torch.tensor([float(temp)], dtype=dtype)
        vo2.reversal(t)
        out.append(float(vo2.R(t)[0]) * 1000.0)
    return np.asarray(out, dtype=float)


def _local_hysteresis_trace(temperatures_K: Iterable[float]) -> np.ndarray:
    h = HysteresisArray(
        YuanhangResistParams(),
        size=1,
        start_branch="insulator",
    )
    h.initialize(np.asarray([324.9], dtype=float))
    out = []
    for temp in temperatures_K:
        r, _ = h.evaluate(np.asarray([float(temp)], dtype=float))
        out.append(float(r[0]))
    return np.asarray(out, dtype=float)


def compare_hysteresis(ref_module) -> list[HysteresisComparison]:
    temps = np.asarray([324.9, 325.0, 360.0, 325.0, 360.0, 330.0, 360.0, 335.0, 360.0], dtype=float)
    results: list[HysteresisComparison] = []
    local = _local_hysteresis_trace(temps)
    for dtype_label, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        reference = _reference_hysteresis_trace(ref_module, temps, dtype)
        diff = np.abs(local - reference)
        rel = diff / np.maximum(np.abs(reference), 1e-12)
        worst = int(np.argmax(diff))
        results.append(
            HysteresisComparison(
                label=f"reference_{dtype_label}",
                max_abs_diff_ohm=float(np.max(diff)),
                max_rel_diff=float(np.max(rel)),
                worst_step=worst,
                worst_temperature_K=float(temps[worst]),
                local_resistance_ohm=float(local[worst]),
                reference_resistance_ohm=float(reference[worst]),
            )
        )
    return results


def _paper_current_params(resist_params: YuanhangResistParams, start_branch: str) -> CurrentDriveParams:
    return CurrentDriveParams(
        dt_s=10e-9,
        t_end_s=600e-9,
        t_pre_s=0.0,
        pulse_on_s=0.0,
        pulse_off_s=None,
        V_init_V=0.0,
        T0_K=298.0,
        T_init_K=297.9,
        C_F=145.34619293e-12,
        C_th_J_per_K=49.62776831e-12,
        S_e_W_per_K=0.20558726e-3,
        sigma_W_sqrt_s=1.0e-6,
        resist_params=resist_params,
        start_branch=start_branch,
    )


def _experiment_style_params(
    resist_params: YuanhangResistParams,
    *,
    start_branch: str,
    T0_K: float,
    T_init_K: float,
    dt_s: float,
) -> CurrentDriveParams:
    return CurrentDriveParams(
        dt_s=dt_s,
        t_end_s=600e-9,
        t_pre_s=200e-9,
        pulse_on_s=0.0,
        pulse_off_s=300e-9,
        V_init_V=0.0,
        T0_K=T0_K,
        T_init_K=T_init_K,
        C_F=145.34619293e-12,
        C_th_J_per_K=49.62776831e-12,
        S_e_W_per_K=0.20558726e-3,
        sigma_W_sqrt_s=0.0,
        resist_params=resist_params,
        start_branch=start_branch,
    )


def _numerics_report(params: CurrentDriveParams) -> dict[str, float]:
    h = HysteresisArray(params.resist_params, size=1, start_branch=params.start_branch)
    T0 = np.asarray([float(params.T_init_K)], dtype=float)
    h.initialize(T0)
    R_init = float(h.evaluate(T0)[0][0])
    R_fast = float(params.resist_params.Rm)
    tau_init_ns = float(params.C_F * R_init * 1e9)
    tau_fast_ns = float(params.C_F * R_fast * 1e9)
    return {
        "R_init_ohm": R_init,
        "R_fast_ohm": R_fast,
        "tau_init_ns": tau_init_ns,
        "tau_fast_ns": tau_fast_ns,
        "dt_ns": float(params.dt_s * 1e9),
        "dt_over_tau_init": float(params.dt_s * 1e9 / max(tau_init_ns, 1e-12)),
        "dt_over_tau_fast": float(params.dt_s * 1e9 / max(tau_fast_ns, 1e-12)),
        "Tc_minus_T0_K": float(params.resist_params.Tc_K - params.T0_K),
        "Tc_minus_Tinit_K": float(params.resist_params.Tc_K - params.T_init_K),
    }


def _plateau_metrics(out: dict[str, np.ndarray]) -> dict[str, float]:
    t_ns = out["t"] * 1e9
    v_mV = out["V_vo2"] * 1e3
    plateau = (t_ns >= 100.0) & (t_ns <= 250.0)
    turnoff = (t_ns >= 300.0) & (t_ns <= 420.0)
    if not np.any(plateau):
        plateau = t_ns >= 0.0
    if not np.any(turnoff):
        turnoff = t_ns >= 0.0
    return {
        "plateau_mean_mV": float(np.mean(v_mV[plateau])),
        "plateau_vpp_mV": float(np.ptp(v_mV[plateau])) if np.sum(plateau) >= 2 else 0.0,
        "vmax_mV": float(np.max(v_mV)),
        "turnoff_min_mV": float(np.min(v_mV[turnoff])),
    }


def _current_family_summary(name: str, params: CurrentDriveParams, currents_uA: Iterable[int]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for idx, current in enumerate(currents_uA):
        out = simulate_current_step(float(current), params=params, seed=idx + 1)
        metrics = _plateau_metrics(out)
        metrics["I_target_uA"] = float(current)
        rows.append(metrics)
    return rows


def _print_hysteresis_report(results: list[HysteresisComparison]) -> None:
    print("== Hysteresis Fidelity ==")
    print("Temperature path: 324.9 -> 325 -> 360 -> 325 -> 360 -> 330 -> 360 -> 335 -> 360 K")
    for result in results:
        print(
            f"- Local vs {result.label}: max |dR| = {result.max_abs_diff_ohm:.3f} ohm, "
            f"max relative diff = {100.0 * result.max_rel_diff:.2f}% "
            f"(worst step {result.worst_step}, T={result.worst_temperature_K:.1f} K, "
            f"local={result.local_resistance_ohm:.3f} ohm, ref={result.reference_resistance_ohm:.3f} ohm)"
        )
    print()


def _print_numerics_report(title: str, report: dict[str, float], *, start_branch: str, pulse_off_s: float | None) -> None:
    print(f"== {title} ==")
    print(f"- start_branch = {start_branch}")
    print(f"- pulse_off_s = {pulse_off_s}")
    print(
        f"- R_init = {report['R_init_ohm']:.3f} ohm, R_fast = {report['R_fast_ohm']:.3f} ohm, "
        f"tau_init = {report['tau_init_ns']:.3f} ns, tau_fast = {report['tau_fast_ns']:.3f} ns"
    )
    print(
        f"- dt = {report['dt_ns']:.4f} ns, dt/tau_init = {report['dt_over_tau_init']:.4f}, "
        f"dt/tau_fast = {report['dt_over_tau_fast']:.4f}"
    )
    print(
        f"- Tc - T0 = {report['Tc_minus_T0_K']:.3f} K, "
        f"Tc - T_init = {report['Tc_minus_Tinit_K']:.3f} K"
    )
    print()


def _print_current_family(name: str, rows: list[dict[str, float]]) -> None:
    print(f"== {name} ==")
    print("I_uA | plateau_mean_mV | plateau_vpp_mV | vmax_mV | turnoff_min_mV")
    for row in rows:
        print(
            f"{int(row['I_target_uA']):4d} | "
            f"{row['plateau_mean_mV']:15.3f} | "
            f"{row['plateau_vpp_mV']:14.3f} | "
            f"{row['vmax_mV']:7.3f} | "
            f"{row['turnoff_min_mV']:14.3f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit model fidelity and likely current-drive mismatch sources.")
    parser.add_argument(
        "--specimen-preset",
        type=Path,
        default=SPECIMEN_PRESET,
        help="Specimen RT preset JSON to audit.",
    )
    args = parser.parse_args()

    ref_module = _load_reference_module()
    hysteresis_results = compare_hysteresis(ref_module)

    payload = json.loads(args.specimen_preset.read_text())
    specimen_resist = YuanhangResistParams(**payload["resist_params"])
    specimen_start_branch = str(payload.get("start_branch", "insulator")).strip().lower()
    if specimen_start_branch not in {"insulator", "metal"}:
        specimen_start_branch = "insulator"

    paper_params = _paper_current_params(YuanhangResistParams(), "insulator")
    specimen_paper_defaults = _paper_current_params(specimen_resist, specimen_start_branch)
    specimen_exp_style = _experiment_style_params(
        specimen_resist,
        start_branch="insulator",
        T0_K=325.0,
        T_init_K=324.9,
        dt_s=0.1e-9,
    )

    _print_hysteresis_report(hysteresis_results)
    _print_numerics_report(
        "Paper Current Preset",
        _numerics_report(paper_params),
        start_branch=paper_params.start_branch,
        pulse_off_s=paper_params.pulse_off_s,
    )
    _print_numerics_report(
        "Specimen RT + Paper Current Defaults",
        _numerics_report(specimen_paper_defaults),
        start_branch=specimen_paper_defaults.start_branch,
        pulse_off_s=specimen_paper_defaults.pulse_off_s,
    )
    _print_numerics_report(
        "Specimen RT + Experiment-Style Pulse (325 K, dt=0.1 ns)",
        _numerics_report(specimen_exp_style),
        start_branch=specimen_exp_style.start_branch,
        pulse_off_s=specimen_exp_style.pulse_off_s,
    )

    currents = [50, 300, 600, 1000, 1500, 2000]
    _print_current_family(
        "Specimen RT + Paper Current Defaults + 300 ns pulse",
        _current_family_summary(
            "specimen_rt_paper_defaults",
            _experiment_style_params(
                specimen_resist,
                start_branch=specimen_start_branch,
                T0_K=298.0,
                T_init_K=297.9,
                dt_s=10e-9,
            ),
            currents,
        ),
    )
    _print_current_family(
        "Specimen RT + Experiment-Style Pulse (325 K, dt=0.1 ns)",
        _current_family_summary("specimen_rt_experiment_style", specimen_exp_style, currents),
    )

    print("== Interpreting This Audit ==")
    print("- local vs reference_float32 should be nearly exact; this is the only supported hysteresis implementation.")
    print("- A large dt/tau_fast means the coupled current-drive transition is under-resolved; the exponential RC substep remains stable.")
    print("- A large positive Tc - T0 means the device starts far below transition and will look RC-like unless pulse/thermal parameters compensate.")
    print("- If plateau_mean_mV keeps scaling with current instead of flattening near ~200 mV, the reduced ideal-current model and/or dynamic parameter set is not reproducing the lab pulse family.")


if __name__ == "__main__":
    main()
