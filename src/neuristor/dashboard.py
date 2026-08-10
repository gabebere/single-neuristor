"""Archive-first Streamlit dashboard for completed scientific run bundles.

The dashboard deliberately does not launch simulations.  Experiments are
created through the versionable TOML + CLI workflow; this interface focuses on
finding, understanding, comparing, and downloading the resulting evidence.
"""

from __future__ import annotations

import json
import html
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import plotly.express as px
import streamlit as st

from .runs import RunRecord, RunRegistry, find_project_root


_CSS = """
<style>
    :root {
        --archive-ink: #162033;
        --archive-muted: #657084;
        --archive-border: #dfe5ee;
        --archive-blue: #315efb;
        --archive-panel: #f7f9fc;
    }
    .stApp { background: linear-gradient(180deg, #f8faff 0, #ffffff 340px); color: var(--archive-ink); }
    .block-container { max-width: 1480px; padding-top: 2.1rem; padding-bottom: 5rem; }
    [data-testid="stSidebar"] { border-right: 1px solid var(--archive-border); background: #fbfcff; }
    h1, h2, h3 { letter-spacing: -0.025em; color: var(--archive-ink); }
    .archive-kicker { color: var(--archive-blue); font-size: .76rem; font-weight: 750; letter-spacing: .11em; text-transform: uppercase; }
    .archive-subtitle { color: var(--archive-muted); max-width: 820px; font-size: 1.02rem; line-height: 1.55; margin: -.5rem 0 1.4rem; }
    .archive-card { border: 1px solid var(--archive-border); border-radius: 16px; padding: 1rem 1.1rem; background: rgba(255,255,255,.86); box-shadow: 0 8px 28px rgba(22,32,51,.045); }
    .archive-label { color: var(--archive-muted); font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .archive-value { color: var(--archive-ink); font-size: 1.35rem; font-weight: 720; margin-top: .2rem; overflow-wrap: anywhere; }
    .archive-pill { display: inline-block; padding: .24rem .55rem; margin: 0 .3rem .3rem 0; border-radius: 999px; background: #edf2ff; color: #2848b8; font-size: .78rem; font-weight: 650; }
    .archive-path { color: var(--archive-muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .78rem; overflow-wrap: anywhere; }
    div[data-testid="stMetric"] { border: 1px solid var(--archive-border); background: #fff; padding: .85rem 1rem; border-radius: 14px; }
    div[data-testid="stMetricLabel"] { color: var(--archive-muted); }
    div[data-testid="stDataFrame"] { border: 1px solid var(--archive-border); border-radius: 12px; overflow: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--archive-border); }
    .stTabs [data-baseweb="tab"] { border-radius: 9px 9px 0 0; padding: .55rem .9rem; }
</style>
"""


def _stamp(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


@st.cache_data(show_spinner=False)
def _read_csv(path_text: str, _mtime: int) -> pd.DataFrame:
    return pd.read_csv(path_text)


@st.cache_data(show_spinner=False)
def _read_json(path_text: str, _mtime: int) -> Any:
    return json.loads(Path(path_text).read_text())


@st.cache_data(show_spinner=False)
def _read_text(path_text: str, _mtime: int) -> str:
    return Path(path_text).read_text(errors="replace")


def _matching_files(record: RunRecord, suffixes: Iterable[str]) -> list[Path]:
    wanted = {suffix.lower() for suffix in suffixes}
    return [path for path in record.files() if path.suffix.lower() in wanted]


def _first_existing(record: RunRecord, names: Iterable[str]) -> Path | None:
    for name in names:
        path = record.root / name
        if path.is_file():
            return path
    return None


def _metrics(record: RunRecord) -> dict[str, Any]:
    metrics = dict(record.summary)
    path = _first_existing(record, ("metrics.json", "published_table_report.json"))
    if path:
        try:
            loaded = _read_json(str(path), _stamp(path))
            if isinstance(loaded, Mapping):
                metrics.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    return {str(key): value for key, value in metrics.items() if not isinstance(value, (dict, list))}


def _short_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def _filter_records(records: list[RunRecord]) -> list[RunRecord]:
    st.sidebar.markdown("### Find runs")
    query = st.sidebar.text_input("Search", placeholder="name, ID, model…")
    storage_options = sorted({record.storage for record in records})
    model_options = sorted({record.model for record in records})
    kind_options = sorted({record.kind for record in records})
    status_options = sorted({record.status for record in records})
    storage = st.sidebar.multiselect("Storage", storage_options, default=storage_options)
    model = st.sidebar.multiselect("Model", model_options, default=model_options)
    kind = st.sidebar.multiselect("Workflow", kind_options, default=kind_options)
    status = st.sidebar.multiselect("Status", status_options, default=status_options)
    needle = query.casefold().strip()
    return [
        record
        for record in records
        if record.storage in storage
        and record.model in model
        and record.kind in kind
        and record.status in status
        and (not needle or needle in " ".join((record.id, record.name, record.model, record.kind)).casefold())
    ]


def _archive_overview(all_records: list[RunRecord], visible_records: list[RunRecord]) -> None:
    completed = sum(record.status == "completed" for record in all_records)
    public = sum(record.storage == "public" for record in all_records)
    models = len({record.model for record in all_records})
    columns = st.columns(4)
    columns[0].metric("Visible runs", len(visible_records), help="Runs matching the sidebar filters")
    columns[1].metric("Completed", completed)
    columns[2].metric("Public evidence", public)
    columns[3].metric("Models represented", models)


def _record_table(records: list[RunRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "created": record.created_at,
                "name": record.name,
                "model": record.model,
                "workflow": record.kind,
                "status": record.status,
                "storage": record.storage,
                "id": record.id,
            }
            for record in records
        ]
    )


def _record_header(record: RunRecord) -> None:
    st.markdown('<p class="archive-kicker">Selected run</p>', unsafe_allow_html=True)
    st.header(record.name)
    pills = "".join(
        f'<span class="archive-pill">{html.escape(str(value))}</span>'
        for value in (
            record.model,
            record.kind,
            record.status,
            record.storage,
            "legacy" if record.legacy else "bundle v1",
        )
    )
    st.markdown(pills, unsafe_allow_html=True)
    st.markdown(f'<div class="archive-path">{html.escape(str(record.root))}</div>', unsafe_allow_html=True)


def _render_overview(record: RunRecord) -> None:
    metrics = _metrics(record)
    preferred = [
        "frequency_MHz",
        "oscillatory",
        "voltage_min_V",
        "voltage_max_V",
        "temperature_max_K",
        "metallic_voltage_floor_V",
        "points",
        "rmse_log10",
    ]
    shown = [(key, metrics[key]) for key in preferred if key in metrics][:4]
    if not shown:
        shown = list(metrics.items())[:4]
    if shown:
        for column, (key, value) in zip(st.columns(len(shown)), shown):
            column.metric(key.replace("_", " "), _short_number(value))

    report = _first_existing(record, ("report.md", "professor_paper_simulation_report.md", "log.txt"))
    if report:
        st.markdown("### Run report")
        content = _read_text(str(report), _stamp(report))
        if report.suffix.lower() == ".md":
            st.markdown(content)
        else:
            st.code(content[-20_000:], language="text")
    else:
        st.info(
            "This historical run has no Markdown report. Its manifest and artifacts remain available in the other tabs."
        )

    images = _matching_files(record, {".png", ".jpg", ".jpeg", ".gif"})
    if images:
        st.markdown("### Evidence gallery")
        for row_start in range(0, min(len(images), 12), 3):
            columns = st.columns(3)
            for column, path in zip(columns, images[row_start : row_start + 3]):
                with column:
                    st.image(str(path), caption=str(path.relative_to(record.root)), width="stretch")
        if len(images) > 12:
            st.caption(f"Showing 12 of {len(images)} images. Every file is available in the Files tab.")


def _default_column(columns: list[str], candidates: Iterable[str], fallback: int = 0) -> int:
    lowered = {column.lower(): index for index, column in enumerate(columns)}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return min(fallback, max(len(columns) - 1, 0))


def _render_data(record: RunRecord) -> None:
    csv_files = _matching_files(record, {".csv"})
    if not csv_files:
        st.info("No tabular artifacts were found for this run.")
        return
    labels = [str(path.relative_to(record.root)) for path in csv_files]
    selected_label = st.selectbox("Dataset", labels)
    path = csv_files[labels.index(selected_label)]
    if path.stat().st_size > 150_000_000:
        st.warning("This CSV exceeds 150 MB. Download it from Files instead of loading it into the browser.")
        return
    try:
        frame = _read_csv(str(path), _stamp(path))
    except Exception as exc:
        st.error(f"Could not read {selected_label}: {exc}")
        return
    st.caption(f"{len(frame):,} rows × {len(frame.columns)} columns")
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    if len(numeric) >= 2:
        controls = st.columns([1, 1, 1])
        x_index = _default_column(numeric, ("time_us", "time_ns", "frame_index", "point_index"))
        x = controls[0].selectbox("X axis", numeric, index=x_index)
        remaining = [column for column in numeric if column != x]
        y_index = _default_column(
            remaining,
            ("voltage_V", "v_out_mV", "frequency_MHz", "temperature_K", "current_inferred_uA"),
        )
        y = controls[1].selectbox("Y axis", remaining, index=y_index)
        group_candidates = ["frame_index", "current_inferred_uA", "current_uA", "seed", "point_index"]
        groups = [column for column in group_candidates if column in frame.columns and column not in {x, y}]
        color = controls[2].selectbox("Color/group", ["None", *groups])
        plotted = frame
        if len(plotted) > 100_000:
            step = max(len(plotted) // 100_000, 1)
            plotted = plotted.iloc[::step]
            st.caption(
                f"Interactive plot downsampled to {len(plotted):,} points; the table and download preserve all rows."
            )
        figure = px.line(plotted, x=x, y=y, color=None if color == "None" else color, render_mode="webgl")
        figure.update_layout(height=510, margin=dict(l=20, r=20, t=35, b=20), legend_title_text=color)
        st.plotly_chart(figure, width="stretch")
    st.dataframe(frame.head(5_000), width="stretch", hide_index=True)
    if len(frame) > 5_000:
        st.caption("Table preview is limited to 5,000 rows.")


def _render_metrics(record: RunRecord) -> None:
    metrics = _metrics(record)
    if not metrics:
        st.info("No normalized metrics are available for this historical run.")
        return
    frame = pd.DataFrame([{"metric": key, "value": _short_number(value)} for key, value in sorted(metrics.items())])
    st.dataframe(frame, width="stretch", hide_index=True)
    numeric = {
        key: value for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if numeric:
        chart = pd.DataFrame([{"metric": key, "value": value} for key, value in numeric.items()])
        st.caption(
            "Metrics use different units; the chart is for quick relative inspection, not dimensional comparison."
        )
        st.bar_chart(chart.set_index("metric"))


def _render_files(record: RunRecord) -> None:
    files = record.files()
    if not files:
        st.info("This run directory is empty.")
        return
    rows = []
    for path in files:
        rows.append(
            {
                "file": str(path.relative_to(record.root)),
                "type": path.suffix.lower() or "file",
                "size_kB": round(path.stat().st_size / 1024.0, 2),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    selected = st.selectbox("Download artifact", [row["file"] for row in rows])
    path = record.root / selected
    st.download_button(
        "Download selected file",
        data=path.read_bytes(),
        file_name=path.name,
        mime="application/octet-stream",
        width="content",
    )


def _render_config(record: RunRecord) -> None:
    config_path = _first_existing(record, ("resolved_config.json",))
    if config_path:
        st.markdown("### Resolved experiment")
        st.json(_read_json(str(config_path), _stamp(config_path)), expanded=True)
    st.markdown("### Manifest and provenance")
    st.json(dict(record.manifest), expanded=True)
    command = record.manifest.get("command")
    if command:
        st.markdown("### Reproduction command")
        st.code(str(command), language="bash")


def _render_compare(records: list[RunRecord], selected: RunRecord) -> None:
    if len(records) < 2:
        st.info("At least two visible runs are required for comparison.")
        return
    by_id = {record.id: record for record in records}
    default = [selected.id]
    for record in records:
        if record.id != selected.id and record.model == selected.model:
            default.append(record.id)
            break
    chosen = st.multiselect(
        "Runs to compare",
        options=list(by_id),
        default=default,
        max_selections=8,
        format_func=lambda run_id: f"{by_id[run_id].name} · {run_id}",
    )
    if not chosen:
        return
    metric_rows = []
    for run_id in chosen:
        record = by_id[run_id]
        row = {"run": record.name, "id": record.id, "model": record.model}
        row.update(_metrics(record))
        metric_rows.append(row)
    comparison = pd.DataFrame(metric_rows)
    st.dataframe(comparison, width="stretch", hide_index=True)

    trace_records: list[tuple[RunRecord, Path]] = []
    for run_id in chosen:
        record = by_id[run_id]
        path = _first_existing(record, ("traces.csv", "current_sweep_traces.csv"))
        if path and path.stat().st_size <= 150_000_000:
            trace_records.append((record, path))
    if len(trace_records) < 2:
        st.caption("Two compatible trace files are needed for an overlay.")
        return
    common_numeric: set[str] | None = None
    loaded: list[tuple[RunRecord, pd.DataFrame]] = []
    for record, path in trace_records:
        frame = _read_csv(str(path), _stamp(path))
        numeric = {column for column in frame if pd.api.types.is_numeric_dtype(frame[column])}
        common_numeric = numeric if common_numeric is None else common_numeric & numeric
        loaded.append((record, frame))
    columns = sorted(common_numeric or [])
    if len(columns) < 2:
        st.caption("The selected runs do not share two numeric trace columns.")
        return
    controls = st.columns(2)
    x = controls[0].selectbox(
        "Overlay X axis", columns, index=_default_column(columns, ("time_us", "time_ns")), key="compare_x"
    )
    y_options = [column for column in columns if column != x]
    y = controls[1].selectbox(
        "Overlay Y axis",
        y_options,
        index=_default_column(y_options, ("voltage_V", "v_out_mV", "temperature_K")),
        key="compare_y",
    )
    pieces = []
    for record, frame in loaded:
        piece = frame[[x, y]].dropna().copy()
        if len(piece) > 40_000:
            piece = piece.iloc[:: max(len(piece) // 40_000, 1)]
        piece["run"] = record.name
        pieces.append(piece)
    overlay = pd.concat(pieces, ignore_index=True)
    figure = px.line(overlay, x=x, y=y, color="run", render_mode="webgl")
    figure.update_layout(height=540, margin=dict(l=20, r=20, t=35, b=20))
    st.plotly_chart(figure, width="stretch")


def main() -> None:
    """Render the complete Streamlit archive application."""

    st.set_page_config(page_title="Neuristor Archive", page_icon="◉", layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown('<p class="archive-kicker">VO2 neuristor research</p>', unsafe_allow_html=True)
    st.title("Simulation archive")
    st.markdown(
        '<p class="archive-subtitle">A reproducible view of current-source and voltage-source simulations, '
        "parameter sweeps, specimen fits, and digitized laboratory evidence. Experiments are created in the terminal; "
        "this dashboard is the durable reading and comparison surface.</p>",
        unsafe_allow_html=True,
    )

    project_root = find_project_root()
    registry = RunRegistry(project_root)
    all_records = registry.discover()
    if not all_records:
        st.info("No runs are archived yet. Start with a recipe from `experiments/` using the `neuristor` command.")
        st.code("neuristor simulate current --config experiments/current/nonzero_voltage_valley.toml", language="bash")
        return
    visible = _filter_records(all_records)
    _archive_overview(all_records, visible)
    st.sidebar.divider()
    st.sidebar.caption("Create runs in the terminal")
    st.sidebar.code("neuristor --help", language="bash")
    st.sidebar.caption(f"Repository\n{project_root}")
    if not visible:
        st.warning("No runs match the current filters.")
        return

    with st.expander(f"Archive index · {len(visible)} matching runs", expanded=False):
        st.dataframe(_record_table(visible), width="stretch", hide_index=True)
    labels = {record.id: f"{record.name} · {record.id}" for record in visible}
    selected_id = st.selectbox("Open run", options=list(labels), format_func=labels.get, label_visibility="collapsed")
    selected = next(record for record in visible if record.id == selected_id)
    st.divider()
    _record_header(selected)

    overview, data, metrics, compare, files, config = st.tabs(
        ["Overview", "Data explorer", "Metrics", "Compare", "Files", "Configuration"]
    )
    with overview:
        _render_overview(selected)
    with data:
        _render_data(selected)
    with metrics:
        _render_metrics(selected)
    with compare:
        _render_compare(visible, selected)
    with files:
        _render_files(selected)
    with config:
        _render_config(selected)
