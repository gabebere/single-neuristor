"""Offline material-model comparisons, including mechanism-off controls."""

from __future__ import annotations

import html
import json

import pandas as pd


LABELS = {
    "baseline": "Original model",
    "warm_parallel": "Warm parallel regions",
    "series_bottleneck": "Series VO₂ bottleneck",
    "phase_lag": "Finite switching time",
}


def render_model_pilot(bundle, screened, verified, controls, context):
    """Write complete tables and views; never hide failed cycles or domain checks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    p = context["config"]["model_pilot"]
    root = bundle.root
    screening = []
    for result in screened:
        screening.append(
            dict(
                id=result["id"],
                mechanism=result["specification"]["kind"],
                anchor=result["anchor"],
                stage=result["stage"],
                simulated=result["simulated"],
                **result["specification"],
                **result["metrics"],
            )
        )
    pd.DataFrame(screening).to_csv(
        bundle.add_artifact("screening.csv", label="Every declared and refined case, including failures"), index=False
    )
    summaries = []
    for result in verified:
        a = result["acceptance"]
        kind = result["specification"]["kind"]
        control = next((r for r in controls if r["id"] == result["id"]), None)
        summaries.append(
            dict(
                id=result["id"],
                mechanism=kind,
                label=LABELS[kind],
                role=result["role"],
                passing_targets=a["passing_targets"],
                defined_frequencies=a["defined_frequencies"],
                frequency_MAPE_percent=100 * a["frequency_MAPE_defined"]
                if a["frequency_MAPE_defined"] is not None
                else None,
                amplitude_MAPE_percent=100 * a["amplitude_MAPE_all"],
                mean_voltage_RMSE_mV=a["mean_voltage_RMSE_mV"],
                peak_trough_RMSE_mV=result["shape_RMSE_mV"],
                maximum_temperature_K=result["maximum_temperature_K"],
                amplitude_decreases_across_pilot=result["amplitude_decreases_across_pilot"],
                physical=a["gates"]["all_record_physical"],
                converged=a["gates"]["numerically_converged"],
                short_gate_success=a["success"],
                long_gate_success=control["persistent_physical_and_converged"] if control else None,
                failed_gates=", ".join(a["failed_gates"]),
                paired_with=result.get("paired_with"),
            )
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(
        bundle.add_artifact("comparison.csv", label="All-current and matched-vector ablation results"), index=False
    )
    candidates = [r for r in verified if r["role"] == "candidate"]
    cached = {r["id"]: pd.read_csv(root / "verification" / r["id"] / "traces.csv") for r in candidates}
    fig, axes = plt.subplots(
        len(candidates),
        len(context["pilot"]),
        figsize=(4.2 * len(context["pilot"]), 2.6 * len(candidates)),
        squeeze=False,
        layout="constrained",
    )
    for i, result in enumerate(candidates):
        data = cached[result["id"]]
        for j, index in enumerate(context["pilot"]):
            label = context["dataset"].nominal_drives_mV[index]
            row = data[
                (data.source_label_mV == label) & data.time_ns.between(*context["config"]["steady"]["window_ns"])
            ]
            ax = axes[i, j]
            ax.plot(row.time_ns, row.measured_voltage_mV, color="#172c43", lw=1.6, label="Measured")
            ax.plot(row.time_ns, row.simulated_voltage_mV, color="#d95f02", lw=1.2, label="Simulated")
            ax.set_title(
                LABELS[result["specification"]["kind"]] + f" · {context['currents'][index]:.1f} µA", fontsize=10
            )
            ax.set(xlabel="Time (ns)", ylabel="Voltage (mV)")
            ax.grid(alpha=0.2)
            if i == j == 0:
                ax.legend(fontsize=9)
    fig.savefig(
        bundle.add_artifact(
            "figures/pilot_waveforms.png", label="All four mechanism winners at prespecified pilot currents"
        ),
        dpi=150,
    )
    plt.close(fig)
    viewer = go.Figure()
    buttons = []
    labels = context["dataset"].nominal_drives_mV
    initial = context["pilot"][len(context["pilot"]) // 2]
    width = 1 + len(candidates)
    colors = ["#172c43", "#777777", "#008e86", "#b15b00", "#8854a8"]
    for index, label in enumerate(labels):
        first = cached[candidates[0]["id"]]
        measured = first[
            (first.source_label_mV == label) & first.time_ns.between(*context["config"]["steady"]["window_ns"])
        ]
        viewer.add_trace(
            go.Scatter(
                x=measured.time_ns,
                y=measured.measured_voltage_mV,
                name="Measured",
                visible=index == initial,
                line=dict(color=colors[0], width=3),
            )
        )
        for offset, result in enumerate(candidates, 1):
            data = cached[result["id"]]
            row = data[
                (data.source_label_mV == label) & data.time_ns.between(*context["config"]["steady"]["window_ns"])
            ]
            viewer.add_trace(
                go.Scatter(
                    x=row.time_ns,
                    y=row.simulated_voltage_mV,
                    name=LABELS[result["specification"]["kind"]],
                    visible=index == initial,
                    line=dict(color=colors[offset], width=1.7),
                )
            )
        buttons.append(
            dict(
                label=f"{context['currents'][index]:.1f} µA",
                method="update",
                args=[
                    dict(visible=[i // width == index for i in range(len(labels) * width)]),
                    dict(
                        title=f"Measured and simulated · {context['currents'][index]:.1f} µA",
                        yaxis=dict(title="Voltage (mV)", autorange=True),
                    ),
                ],
            )
        )
    viewer.update_layout(
        template="plotly_white",
        height=560,
        title=f"Measured and simulated · {context['currents'][initial]:.1f} µA",
        updatemenus=[dict(buttons=buttons, active=initial, x=0, y=1.16)],
        margin=dict(t=110, b=60),
        xaxis=dict(title="Time (ns)", range=context["config"]["steady"]["window_ns"]),
        yaxis_title="Voltage (mV)",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    table = summary.drop(columns=["paired_with"]).to_html(
        index=False, na_rep="undefined", float_format=lambda x: f"{x:.5g}"
    )
    links = []
    for result in verified:
        identifier = result["id"]
        links.append(
            f'<li>{html.escape(identifier)}: <a href="verification/{identifier}/verification.csv">all timestep features</a> · '
            f'<a href="verification/{identifier}/convergence.csv">convergence</a> · <a href="verification/{identifier}/traces.csv">all 22 waveforms</a></li>'
        )
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Material-model pilot</title><style>body{font:16px system-ui;line-height:1.6;color:#172c43;max-width:1250px;margin:2rem auto;padding:0 1rem}
    .scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccd5df;padding:.4rem}th{background:#edf3f7}img{max-width:100%}
    .note{padding:1rem;background:#fff4db}a{color:#08688e}</style><h1>Partial-switching material-model pilot</h1>
    <p class="note">Optional hypotheses were tested separately. No calibration is automatically adopted.
    This is a fixed-grid pilot using historically seen records and shared parameter vectors, not blind validation or a global fit.</p>
    <p><a href="report.md">Methods and interpretation</a> · <a href="comparison.csv">Comparison table</a> · <a href="screening.csv">Every screened case</a> · <a href="metrics.json">Coverage and success</a></p>
    <h2>All 22 measured inputs</h2><p>Select a current. Measured voltage and the four retained mechanism candidates share the same axes and original time samples.</p>"""
    page += viewer.to_html(full_html=False, include_plotlyjs=True)
    page += '<h2>Prespecified pilot currents</h2><img src="figures/pilot_waveforms.png" alt="Four mechanism candidates compared at three pilot currents">'
    page += (
        '<h2>Full-current checks and mechanism-off ablations</h2><p>Frequency MAPE uses defined frequencies, Vpp uses all 11 targets, and mean RMSE all 22 records. Undefined cycles are failures. Matched baseline rows turn off only the mechanism while keeping its selected reference vector unchanged.</p><div class="scroll">'
        + table
        + "</div>"
    )
    page += "<h2>Numerical evidence</h2><ul>" + "".join(links) + "</ul></html>"
    bundle.write_text("START_HERE.html", page, label="Offline all-current model comparison", media_type="text/html")
    report = f"""# Partial-switching material-model pilot

This finite pilot completed {len(screened)} screening cases and {len(verified)} full-current verification cases.
The prespecified target currents are {context["currents"][context["pilot"]].tolist()} µA; controls
are {context["currents"][context["controls"]].tolist()} µA. Historical records and seeds were already seen;
there is no claim of blind validation. No calibration is automatically adopted.

## Models and physical accounting

- Baseline calls the unchanged authoritative ideal-current model.
- Warm parallel regions use G=w/Rfit(Tprimary,Hprimary)+(1-w)/Rfit(Tsecondary,Hsecondary).
  Each receives its own V²G Joule heat. Both temperatures evolve with bath cooling and mutual exchange.
- A series VO₂ bottleneck uses R=w Rfit(Tprimary,Hprimary)+(1-w) Rfit(Tsecondary,Hsecondary).
  The regions carry the same device-branch current and receive Ibranch²Rregion heat. Their voltage drops
  sum to the measured device voltage. This introduces no external series resistor or current change.
- Finite switching time relaxes the semiconducting weight toward the existing hysteresis target:
  tau dx/dt=g_hyst(T,H)-x; R=Rm+R0 exp(Ea_over_k/T)x. Anchors follow the original target.
  Tau=0 delegates exactly to the original model. No spatial mechanism is combined with phase lag.

Both two-region topologies recover the measured uniform-temperature R(T). The reference Cth and Se
budgets are split by w and 1-w, then secondary capacity/cooling ratios introduce asymmetry.
These budget relations are diagnostic assumptions, not measured active volumes. Positive mutual
conductance transfers K(Tprimary-Tsecondary), cancelling between nodes. Each frozen-power thermal
step uses the exact two-node matrix exponential; electrical and phase substeps are exact with frozen
coefficients. Coupling between subproblems still requires timestep refinement.

## Search, objective and checks

The declared grid uses shared historical anchors. Four one-coordinate reference parameters
(C, Se, tau_th and gamma) then receive the same two local probes in every arm. Static R(T) and T0
remain fixed. Cases/ contains exact specifications, realized inputs via the verification results,
features and diagnostics. No parameter is fitted separately for each current.

Ranking prioritizes physical temperatures/proximity, valid target cycles and excessive control
oscillations before a finite diagnostic feature score. That score combines relative frequency/Vpp,
mean-voltage errors normalized by {p["voltage_scale_mV"]} mV and mean peak/trough squared errors at
the same scale. Missing cycles do not receive an accepted scientific loss. Selected controls are
flagged for sustained amplitude above max({p["control_amplitude_floor_mV"]} mV,
{p["control_amplitude_ratio"]} times their measured Vpp).

Winners and exact-vector mechanism-off ablations replay all 22 original inputs with full measured
prehistory at {p["verification_dt_ns"]} ns. The original full-current observable/physical gates apply,
plus target peak/trough RMSE ≤ {p["success"]["shape_RMSE_mV"]} mV. Long controls last
{p["constant_duration_ns"]} ns at the three pilot mean currents and three finest timesteps.
Native finest histories at the five pilot/control records retain branch currents, powers and both
temperatures where present. Both temperatures and hysteresis slopes are screened.

## Results

See comparison.csv and the offline viewer. Fully successful candidates: {json.loads((root / "metrics.json").read_text())["successful_candidates"]}.
Pilot or conditional improvement must not be called an adopted calibration. A lower diagnostic score
is distinct from full target, quiet-control, temperature, shape, timestep and long-control acceptance.
Zero-amplitude quenched trajectories are never an oscillatory fit.

## Limits and next decision

This is a sparse mechanism pilot with a small shared refinement, not an exhaustive parameter search.
The real TIA circuit boundary and readout transfer remain unresolved. Only a mechanism that improves
amplitude, mean and cycle shape across currents, survives finer steps and keeps controls physical
merits a larger follow-up. Per-arm selections can differ in nuisance coordinates; matched baseline
ablations expose that confounding. No free f(t), per-current scaling, gain or time shift is introduced.
Original input files, complete source, execution recipe, all case tables and hashes are preserved.
"""
    bundle.write_text("report.md", report, label="Models, grid, acceptance and limits")
