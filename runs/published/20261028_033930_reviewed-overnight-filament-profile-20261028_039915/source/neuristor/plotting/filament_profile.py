"""Live and final views of a fixed-f search, keeping failed gates visible."""

from __future__ import annotations

import html
import json

import pandas as pd

from neuristor.fitting.joint_inference import NAMES


def _frames(state):
    searches = []
    checks = []
    for entry in state["completed"].values():
        result = entry["result"]
        if result["kind"] == "search":
            winner = result["winner"]
            searches.append(
                dict(
                    stage=result["stage"],
                    family=result["family"],
                    fraction=result["fraction"],
                    objective=result["objective_label"],
                    fit_records=len(result["training"]),
                    evaluations=result["unique_candidates"],
                    eligible_training=result["eligible_training"],
                    optimizer_score=winner["score"],
                    **dict(zip(NAMES, result["values"])),
                )
            )
        elif result["kind"] == "verification":
            candidate = result["candidate"]
            a = result["acceptance"]
            checks.append(
                dict(
                    stage=result["stage"],
                    family=candidate["family"],
                    fraction=candidate["fraction"],
                    objective=candidate["objective_label"],
                    fit_records=len(candidate["training"]),
                    passing_targets=a["passing_targets"],
                    defined_frequencies=a["defined_frequencies"],
                    frequency_MAPE_percent=100 * a["frequency_MAPE_defined"]
                    if a["frequency_MAPE_defined"] is not None
                    else None,
                    amplitude_MAPE_percent=100 * a["amplitude_MAPE_all"],
                    mean_voltage_RMSE_mV=a["mean_voltage_RMSE_mV"],
                    common_primary_loss=result["all_target"]["objective_loss"],
                    partial_primary_loss=result["all_target"]["dynamic_loss"],
                    maximum_temperature_K=result["all_target"]["maximum_temperature_K"],
                    all_observable_and_numerical_gates=a["success"],
                    failed_gates=", ".join(a["failed_gates"]),
                    initially_excluded_passing=result["optimizer_excluded"]["screened_targets"],
                    **a["gates"],
                )
            )
    return pd.DataFrame(searches), pd.DataFrame(checks)


def render_profile_progress(bundle, state, status):
    """Refresh one local HTML entry point; produce static scientific figures at completion."""
    searches, checks = _frames(state)
    if len(searches):
        searches.to_csv(
            bundle.add_artifact("search_parameters.csv", label="Named parameters and training scores"), index=False
        )
    if len(checks):
        checks.to_csv(
            bundle.add_artifact(
                "verified_candidates.csv", label="All-target errors and failed physical/numerical gates"
            ),
            index=False,
        )
    running = status["state"] == "running"
    active = "".join(
        "<li>" + html.escape(t["stage"]) + ": " + html.escape(json.dumps(t["progress"])) + "</li>"
        for t in status.get("active", [])
    )
    links = "".join(
        f'<li><a href="stages/{html.escape(row["stage"])}/verification.csv">{html.escape(row["stage"])}</a> · '
        f'<a href="stages/{html.escape(row["stage"])}/convergence.csv">timestep checks</a> · '
        f'<a href="stages/{html.escape(row["stage"])}/traces.csv">waveforms</a></li>'
        for row in checks.to_dict("records")
    )
    columns = [
        "stage",
        "family",
        "fraction",
        "objective",
        "fit_records",
        "passing_targets",
        "defined_frequencies",
        "frequency_MAPE_percent",
        "amplitude_MAPE_percent",
        "mean_voltage_RMSE_mV",
        "failed_gates",
    ]
    table = (
        checks[columns].to_html(index=False, na_rep="undefined", float_format=lambda x: f"{x:.4g}")
        if len(checks)
        else "<p>No verified candidate yet.</p>"
    )
    figure = ""
    if not running and len(checks):
        _profile_figure(bundle, checks)
        _waveform_figure(bundle, state)
        figure = '<img src="figures/profile.png" alt="Verified profile errors and valid-target counts">'
        if (bundle.root / "figures/final_waveforms.png").exists():
            figure += '<h2>Final candidates across measured currents</h2><img src="figures/final_waveforms.png" alt="Measured and simulated final waveforms">'
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8">
    {'<meta http-equiv="refresh" content="60">' if running else ""}
    <title>Overnight filament profile</title><style>
    body{{font:16px system-ui;margin:2rem;color:#152332;line-height:1.5}} table{{border-collapse:collapse;font-size:13px}}
    td,th{{padding:.45rem;border:1px solid #cdd5dc;text-align:left}}th{{background:#edf2f6}} img{{max-width:100%}}
    .scroll{{overflow:auto}} a{{color:#11629a}} .note{{background:#fff3d4;padding:1rem}}
    </style><h1>Overnight filament profile</h1>
    <p><strong>{html.escape(status["state"])}</strong> · {html.escape(status.get("phase", ""))}<br>
    Updated: {html.escape(status.get("updated_utc", ""))}<br>Deadline: {html.escape(status.get("deadline_utc", "see plan.json"))}</p>
    <p class="note">A completed search is not an accepted physical fit. Missing cycles remain undefined.
    Frequency errors use only defined frequencies; the defined count is shown beside them.
    Mean-voltage error covers all 22 records. No calibration is automatically adopted.</p>
    <p>The first profile fits 8 targets and excludes 3 from optimization. Those 3 were seen historically,
    so this is not blind validation. Final refinements use all 11. Both thermal families remain diagnostic assumptions.</p>
    <p><a href="report.md">Report</a> · <a href="plan.json">Prespecified plan</a> · <a href="current_split.csv">Current split</a> ·
    <a href="metrics.json">Final metrics</a> · <a href="status.json">Status</a></p>
    <h2>Active stages</h2><ul>{active or "<li>None</li>"}</ul>
    <p>{len(searches)} completed searches; {len(checks)} verified candidates.</p>
    {figure}<h2>Verified candidates</h2><div class="scroll">{table}</div><h2>Numerical evidence</h2><ul>{links}</ul>
    <p><a href="search_parameters.csv">All search parameters</a> · <a href="verified_candidates.csv">Complete verification table</a></p></html>"""
    temporary = bundle.root / "START_HERE.html.tmp"
    temporary.write_text(content)
    temporary.replace(bundle.root / "START_HERE.html")


def _profile_figure(bundle, checks):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    fields = [
        ("frequency_MAPE_percent", "Frequency MAPE (%) on defined targets"),
        ("amplitude_MAPE_percent", "Vpp MAPE (%) on all targets"),
        ("mean_voltage_RMSE_mV", "Mean-voltage RMSE (mV), all 22"),
        ("passing_targets", "Valid targets / 11"),
    ]
    for family, group in checks[checks.stage.str.startswith("profile_check_")].groupby("family"):
        group = group.sort_values("fraction")
        for ax, (field, label) in zip(axes.flat, fields):
            ax.plot(group.fraction, group[field], "o-", label=family)
    for ax, (field, label) in zip(axes.flat, fields):
        ax.set(xscale="log", xlabel="Fixed filament conductance weight f", ylabel=label)
        ax.grid(alpha=0.2)
        if field == "passing_targets":
            ax.set_ylim(-0.5, 11.5)
        else:
            ax.set_yscale("symlog", linthresh=1)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    fig.suptitle("Profile diagnostics: failed temperature/cycle/convergence gates remain in the table")
    fig.savefig(bundle.add_artifact("figures/profile.png", label="Common fine-step profile diagnostics"), dpi=160)
    plt.close(fig)


def _waveform_figure(bundle, state):
    import matplotlib.pyplot as plt

    finals = [
        e["result"]
        for e in state["completed"].values()
        if e["result"]["kind"] == "verification" and e["result"]["stage"].startswith("final_")
    ]
    if not finals:
        return
    # Report all final candidates at low/mid/high target currents, not just a best-looking trace.
    split = pd.read_csv(bundle.root / "current_split.csv")
    labels = split[split.group != "control"].sort_values("current_uA")
    labels = labels.iloc[[0, len(labels) // 2, -1]].source_label_mV.to_list()
    fig, axes = plt.subplots(len(finals), 3, figsize=(14, 2.3 * len(finals)), squeeze=False, layout="constrained")
    for row, result in enumerate(finals):
        traces = pd.read_csv(bundle.root / "stages" / result["stage"] / "traces.csv")
        for col, label in enumerate(labels):
            trace = traces[(traces.source_label_mV == label) & (traces.time_ns >= 150)]
            ax = axes[row, col]
            ax.plot(trace.time_ns, trace.measured_voltage_mV, color="black", lw=1, label="measured")
            ax.plot(trace.time_ns, trace.simulated_voltage_mV, lw=1, label="simulated")
            ax.set_title(f"{result['candidate']['stage']} | {trace.current_uA.iloc[0]:.1f} µA", fontsize=9)
            ax.set(xlabel="Time (ns)", ylabel="Voltage (mV)")
            ax.grid(alpha=0.2)
            if row == col == 0:
                ax.legend(fontsize=8)
    fig.savefig(
        bundle.add_artifact(
            "figures/final_waveforms.png", label="All final candidates at low/mid/high measured currents"
        ),
        dpi=140,
    )
    plt.close(fig)


def profile_report(bundle, state, metrics):
    """Generate the final methods/outcomes record without interpreting nonidentification as absence."""
    config = json.loads((bundle.root / "execution_config.json").read_text())
    profile = config["profile_search"]
    successes = metrics["successful_candidates"]
    outcome = (
        ("Candidates passing every prespecified gate, including long controls: " + ", ".join(successes))
        if successes
        else "No candidate passed every prespecified observable, physical, timestep and long-control gate."
    )
    return f"""# Overnight fixed-f filament profile

{outcome} **No calibration is adopted.** An unsuccessful bounded search does not exclude a filament or identify a unique fraction.

## Method

The authoritative ideal-current solver is unchanged. Its optional model uses
`G_eff = f/R_fit(T_filament,H_filament) + (1-f)/R_fit(T0,H0)`, with only
`f V²/R_fit(T_filament,H_filament)` warming the hot node. The cold branch is an ambient reservoir.
The six fitted R(T) constants and T0 are fixed. Fits vary C, Se, tau=Cth/Se and gamma.

Declared fractions: {profile["fractions"]}. Inherited bounds preserve the previous presentation intervals.
The second family multiplies the Se interval by f and keeps the tau interval, allowing a proportionally
smaller heated volume. That scaling is an explicit diagnostic hypothesis, not a measured heat-flow law.
Starts include the prior baseline, both presentation vectors, direct and coupled thermal starts,
Latin-hypercube populations, neighboring fractions, reverse continuations and training-selected geometric midpoints.
All clamps are recorded. Circuit capacitance is never geometrically scaled.

Initial fits use 8 targets; 3 are optimizer-excluded by current rank. All 22 measured inputs are replayed.
These records and historical seeds have been seen before: this is not blind validation.
Final paired frequency/Vpp and mean-aware refinements use all 11 targets. Their selection uses full-current checks,
so initial exclusions cannot be presented as independent test evidence after refinement.
Primary loss is equal-weight squared relative frequency and robust-Vpp error. Mean-aware loss adds
`((mean_sim - mean_meas)/{profile["mean_voltage_scale_mV"]} mV)^2` with unit weight.
Finite rejection scores rank invalid candidates; they are not accepted scientific losses.

Search step: {profile["search_dt_ns"]} ns; profile checks: {profile["verification_dt_ns"]} ns.
Refinement step: {profile["refinement_dt_ns"]} ns; final checks: {profile["final_dt_ns"]} ns.
The two smaller steps must each pass convergence. Long constant-current controls last
{profile["constant_current_duration_ns"]} ns, at three measured sustained-target currents selected before fitting.
Native finest-step final histories preserve the electrical split and heating channels.

## Prespecified acceptance

All 11 target cycles/frequencies must be valid; all 22 records must remain in the configured temperature
domain with admissible proximity. Frequency MAPE ≤ {100 * profile["success"]["frequency_MAPE"]:g}% and worst error ≤
{100 * profile["success"]["maximum_frequency_error"]:g}%; Vpp MAPE ≤ {100 * profile["success"]["amplitude_MAPE"]:g}% and worst error ≤
{100 * profile["success"]["maximum_amplitude_error"]:g}%. All-record mean-voltage RMSE ≤
{profile["success"]["mean_voltage_RMSE_mV"]:g} mV. Quiet controls must not acquire false large oscillations.
Both timestep halvings require feature changes ≤ {100 * profile["success"]["convergence_fraction"]:g}% and
waveform RMS difference ≤ {profile["success"]["waveform_rms_tolerance_mV"]:g} mV on all targets.
Long controls must remain sustained, physical and feature-converged at all three steps.

## Execution and evidence

Elapsed: {metrics["elapsed_s"] / 3600:.3f} hours; wall allowance: {profile["wall_hours"]} hours;
{metrics["total_unique_candidate_evaluations"]} unique evaluations across {metrics["search_stages"]} search stages.
Completed profile points: {metrics["completed_profile_points"]}; final candidates verified:
{metrics["final_candidates_verified"]}; long-control candidate cases: {metrics["constant_control_cases"]}.
Coverage complete: {metrics["coverage_complete"]}. Budget-skipped stages: {metrics["skipped_budget"]}.
Resume attempts: {state["attempts"]}. Resume preserves cached evaluations and the original wall deadline;
optimizer trajectories may differ after interruption. Completed stage and frozen-input hashes are checked.

Open [START_HERE.html](START_HERE.html), [verified_candidates.csv](verified_candidates.csv),
[search_parameters.csv](search_parameters.csv), [current_split.csv](current_split.csv) and the stage subdirectories.
`frozen_hashes.json`, `source_hashes.json`, `environment.json`, copied inputs and executable source establish provenance.

## Interpretation limits

The circuit channel/current conversion, readout transfer and geometric thermal mapping remain unresolved.
A successful conditional fit would warrant follow-up measurements, not automatic physical identification.
Frequency MAPE on defined records must always be read beside its denominator; missing cycles are never scored as zero error.
Failures, temperature excursions and coarse-only oscillations remain visible in the tables.
"""
