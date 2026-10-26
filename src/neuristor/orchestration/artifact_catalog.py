"""A lightweight human-readable catalog, not another database or artifact copy.

Existing run manifests remain the source of truth. This module rebuilds a small
HTML/CSV navigation index and a file inventory. It never rewrites completed
bundles, deletes old trials, or interprets 'completed' as a successful fit.
"""

from __future__ import annotations

import csv
import ast
import html
import os
from pathlib import Path
from urllib.parse import quote

from neuristor.runs import RunRegistry, find_project_root
from neuristor.layout import CATALOG


# The exported file remains read-only. The loopback manager enables these
# controls in memory and supplies a per-session token; it is never persisted.
MANAGER_SCRIPT = r"""
let runManager = null;
async function managerAction(endpoint, payload) {
  const response = await fetch(endpoint, {method:'POST',
    headers:{'Content-Type':'application/json','X-Neuristor-Token':runManager.token},
    body:JSON.stringify({...payload,confirm:true})});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Action failed');
  return result;
}
async function enableRunManager(config) {
  runManager = config;
  document.getElementById('manager-note').textContent =
    'Local manager active. Delete moves completed/failed local runs to recoverable trash. Published and running runs are protected.';
  document.querySelectorAll('button[data-run-id]').forEach(button => {
    button.disabled = false;
    button.title = 'Move this local run to recoverable trash';
    button.addEventListener('click', async () => {
      if (!confirm('Move “'+button.dataset.runName+'” to trash?\n\nRun: '+button.dataset.runId+
        '\nThe whole run folder, including child stages, will move. You can restore it below.')) return;
      button.disabled = true;
      try {
        const result = await managerAction('/api/trash', {run_id:button.dataset.runId});
        if (result.index_warning) alert('Moved to trash, but index refresh failed: '+result.index_warning);
        location.reload();
      } catch (error) { alert(error.message); button.disabled = false; }
    });
  });
  document.getElementById('trash-panel').hidden = false;
  try {
    const response = await fetch('/api/trash');
    if (!response.ok) throw new Error('Cannot load trash');
    const records = await response.json();
    const list = document.getElementById('trash-list');
    if (!records.length) list.textContent = 'Trash is empty.';
    for (const record of records) {
      const item = document.createElement('li');
      const label = document.createElement('span');
      label.textContent = record.name+' · '+record.run_id+' · deleted '+record.deleted_at.slice(0,10)+' ';
      const restore = document.createElement('button');
      restore.textContent = 'Restore';
      restore.onclick = async () => {
        if (!confirm('Restore “'+record.name+'” to its original run folder?')) return;
        restore.disabled = true;
        try { await managerAction('/api/restore',{trash_id:record.trash_id}); location.reload(); }
        catch (error) { alert(error.message); restore.disabled = false; }
      };
      item.append(label,restore); list.append(item);
    }
  } catch (error) { document.getElementById('trash-list').textContent = error.message; }
}
"""


# These are viewer corrections/coverage facts, not inferred scientific verdicts.
VIEWER_NOTES = {
    "20261007_061243_portable-screened-parameter-map-with-point-gifs_81ae34": "Superseded viewer: GIFs complete, click metadata corrected in eaf6ab.",
    "20261007_074702_portable-screened-parameter-map-with-point-gifs_eaf6ab": "Corrected complete 1,258-GIF viewer; conditional 500 µA, 298 K map, not sample calibration.",
    "20261007_061058_portable-screened-parameter-map-with-point-gifs-_f34881": "Three-point benchmark only; not full GIF coverage.",
    "20261002_151711_focused-dense-physically-screened-steady-state-e_2bffd0": "Superseded slider indexing; corrected in c1772e and eaf6ab.",
    "20261007_080825_sample-r-t-to-screened-map-and-embedded-gifs_485278": "Failed first standalone export; numerical stages preserved. Corrected pipeline was rerun successfully.",
}


def build_source_inventory(root: Path, output: Path) -> int:
    """Inventory owned Python files without importing scripts or running studies.

    This is a generated navigation aid, not a second scientific database. The
    first docstring line provides intent. Retired scripts live in Git history
    and are not mixed into the active source inventory.
    """
    rows = []
    for folder in ("src", "tests"):
        for path in sorted((root / folder).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            text = path.read_text()
            doc = ast.get_docstring(ast.parse(text)) or "See module source and folder guide."
            if folder == "src":
                category = "compatibility import" if '"""Compatibility import for' in text else "supported library"
            elif folder == "tests":
                category = "verification"
            else:
                category = "historical script (not active API)"
            rows.append([relative, category, len(text.splitlines()), doc.splitlines()[0]])
    with (output / "source_inventory.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["relative_path", "role", "lines", "purpose"])
        writer.writerows(rows)
    return len(rows)


def _category(record) -> str:
    """Group workflow responsibility, without assigning physical credibility."""
    name = record.name.lower()
    if record.legacy:
        return "Historical jobs"
    if record.kind == "study":
        return "Sample studies"
    if "map" in name and (record.root / "parameter_maps.html").exists():
        return "Parameter maps"
    if record.kind == "fit" or any(word in name for word in ("inference", "fit", "search")):
        return "Fitting trials"
    if record.kind in {"simulation", "sweep"}:
        return "Simulations and sweeps"
    return "Measurements and diagnostics"


def build_artifact_catalog(project_root: str | Path | None = None, output_directory: str | Path | None = None) -> dict:
    """Rebuild the navigation hub from canonical manifests and produced files.

    Public/local duplicates are resolved by RunRegistry, which prefers public.
    The full file inventory includes both copies to expose physical storage.
    Links are relative to the catalog, so moving the whole repository preserves
    them. The catalog itself is not a standalone shareable research artifact.
    """
    root = find_project_root(project_root)
    output = Path(output_directory).expanduser().resolve() if output_directory else root / CATALOG
    output.mkdir(parents=True, exist_ok=True)
    registry = RunRegistry(root)
    records = registry.discover()
    replacements = {}
    for record in records:
        summary = record.manifest.get("summary", {})
        if record.status == "completed" and summary.get("display_version") and summary.get("source_run_id"):
            replacements.setdefault(summary["source_run_id"], record.root / "parameter_maps.html")
    source_count = build_source_inventory(root, output)

    def link(path: Path, label: str) -> str:
        target = quote(Path(os.path.relpath(path, output)).as_posix(), safe="/.")
        return f'<a href="{target}">{html.escape(label)}</a>'

    rows, cards = [], []
    for record in records:
        manifest = record.root / ("job.json" if record.legacy else "run.json")
        viewers = [
            record.root / name
            for name in (
                "START_HERE.html",
                "parameter_maps.html",
                "scope_viewer.html",
                "report.md",
                "run.json",
                "job.json",
            )
        ]
        entry = next((path for path in viewers if path.is_file()), manifest)
        note = VIEWER_NOTES.get(record.id, "")
        row = dict(
            id=record.id,
            created_at=record.created_at,
            category=_category(record),
            name=record.name,
            generation_status=record.status,
            storage=record.storage,
            bundle_path=record.root.relative_to(root).as_posix(),
            entry_path=entry.relative_to(root).as_posix(),
            note=note,
        )
        rows.append(row)
        report = record.root / "report.md"
        extra = " · " + link(report, "Report") if report.is_file() and entry != report else ""
        if record.id in replacements:
            extra += " · " + link(replacements[record.id], "Updated fixed-scale viewer")
        protected = registry.trash_reason(record)
        if protected:
            action = f'<span title="{html.escape(protected, quote=True)}">Protected</span>'
        else:
            action = (f'<button disabled data-run-id="{html.escape(record.id, quote=True)}" '
                      f'data-run-name="{html.escape(record.name, quote=True)}" '
                      'title="Start neuristor runs manage to enable deletion">Delete…</button>')
        cards.append(
            f"<tr><td>{html.escape(record.created_at[:10])}</td>"
            f"<td>{html.escape(row['category'])}</td><td>{link(entry, record.name)}{extra}"
            f"<br><small>{html.escape(record.id)}</small><p>{html.escape(note)}</p></td>"
            f"<td>{html.escape(record.status)}<br>{html.escape(record.storage)}</td>"
            f"<td>{link(record.root, 'Folder')}</td><td>{action}</td></tr>"
        )
    fields = [
        "id",
        "created_at",
        "category",
        "name",
        "generation_status",
        "storage",
        "bundle_path",
        "entry_path",
        "note",
    ]
    with (output / "runs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # One inventory for the old loose outputs as well as current bundles. No
    # copying or rehashing gigabytes of media is needed to make them findable.
    produced_roots = (
        "runs/local",
        "runs/published",
        "runs/legacy",
        "runs/scratch",
        "docs",
        "data",
    )
    extensions = {".html", ".pdf", ".gif", ".png", ".csv", ".tsv", ".json", ".jsonl", ".tex", ".zip", ".md"}
    count = 0
    with (output / "files.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["relative_path", "file_type", "bytes", "storage_root"])
        for folder in produced_roots:
            for path in sorted((root / folder).rglob("*")):
                if (path.is_file() and path.suffix.lower() in extensions
                        and "source" not in path.parts and ".trash" not in path.parts):
                    writer.writerow(
                        [path.relative_to(root).as_posix(), path.suffix.lower()[1:], path.stat().st_size, folder]
                    )
                    count += 1
    document_links = []
    for folder in ("docs", "runs/scratch/documents/pdf"):
        for path in sorted((root / folder).rglob("*.pdf")):
            document_links.append(f"<li>{link(path, path.relative_to(root).as_posix())}</li>")
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>VO2 research artifacts</title>
<style>body{{font:16px system-ui;color:#18283a;background:#f8fafc;margin:35px auto;padding:0 25px;max-width:1350px}}
table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}
a{{color:#1755a0}}small{{font-size:11px}}input{{padding:12px;width:90%;margin-bottom:20px}}p{{margin:8px 0}}
th{{position:sticky;top:0;background:#e8eef6}}.note{{border-left:4px solid #e7b651;padding:15px;background:white}}
button{{cursor:pointer;padding:7px 12px;border:1px solid #bdcadd;border-radius:6px;background:#f5f8fc;color:#183450}}
button[data-run-id]{{color:#9d2836}}button:disabled{{cursor:default;opacity:.5}}
#trash-list li{{margin:12px 0}}</style>
<h1>VO₂ research artifacts</h1><p>{len(records)} distinct run/job records · {count:,} indexed files · {source_count} owned Python files.</p>
<p class="note">“Completed” means file generation completed, not that a physical fit succeeded.
Read the report and research handoff before adopting any parameters. Published means locally archived,
not necessarily committed or pushed to GitHub. Older bundles remain immutable.</p>
<p>{link(root / "docs/RESEARCH_HANDOFF.md", "Scientific handoff")} ·
{link(root / "README.md", "Project guide")} ·
{link(root / "docs/reproduction/README.md", "Paper reproduction guide")} ·
{link(root / "docs/ARCHIVE_INDEX.md", "Reviewed chronology")} ·
{link(root / "docs/SAMPLE_STUDY.md", "Run a new sample study")} ·
<a href="runs.csv">Run catalog CSV</a> · <a href="files.csv">Complete file inventory CSV</a> ·
<a href="source_inventory.csv">Source inventory CSV</a></p>
<h2>Reports and presentations</h2><details><summary>Browse all generated PDFs</summary><ul>{"".join(document_links)}</ul></details>
<h2>Run history</h2><input id="filter" type="search" placeholder="Filter by sample, date, workflow, run ID or status" aria-label="Filter run history">
<p id="manager-note">To enable Delete and Restore, run <code>neuristor runs manage</code>
and open the local URL it prints. This exported HTML cannot delete filesystem folders.
Deletion is recoverable and does not free disk space until trash is separately removed.</p>
<details id="trash-panel" hidden><summary>Trash — restore deleted runs</summary><ul id="trash-list"></ul></details>
<table><thead><tr><th>Date (UTC)</th><th>Category</th><th>Evidence</th><th>Generation / storage</th><th>Files</th><th>Actions</th></tr></thead>
<tbody>{"".join(cards)}</tbody></table><script>
document.getElementById('filter').addEventListener('input', e => {{
const query=e.target.value.toLowerCase();
for(const row of document.querySelectorAll('tbody tr'))row.hidden=!row.textContent.toLowerCase().includes(query);
}});
{MANAGER_SCRIPT}
</script></html>"""
    (output / "index.html").write_text(page)
    return dict(run_records=len(records), indexed_files=count, source_files=source_count, index=str(output / "index.html"))
