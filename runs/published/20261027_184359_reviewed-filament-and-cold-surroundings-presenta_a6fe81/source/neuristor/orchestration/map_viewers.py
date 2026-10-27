"""Presentation-only revisions of completed maps, without new trajectories.

Old bundles are immutable. New viewers copy saved numerical tables and refer to
existing GIFs rather than duplicating dense media archives. Small sample-study
viewers retain their standalone, embedded-GIF sharing format.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.parse import quote

import numpy as np
import pandas as pd

from neuristor.analysis.physical_maps import AXES, heatmap_display_ranges, render_physical_maps
from neuristor.provenance import snapshot_package
from neuristor.runs import RunBundle, RunRegistry, find_project_root


DISPLAY_VERSION = "fixed-heatmap-scales-v1"


def _stage(root, stages, name):
    """Resolve only a recorded child bundle within its completed parent."""
    path = (root / stages[name]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Study stage path escapes its parent")
    return path


def refresh_map_viewer(source, *, output_root=None, embed_gifs=None,
                       command="neuristor analyze refresh-maps"):
    """Render saved grid data with stable scales; never evaluate the solver.

    Folder-based media viewers retain relative links to original GIFs. Their
    dependencies are recorded explicitly and must be retained when sharing.
    ``embed_gifs=True`` makes a self-contained HTML instead; study parents use
    that format by default, as in the original study export.
    """
    source = Path(source).resolve()
    original = json.loads((source / "run.json").read_text())
    if original.get("status") != "completed" or not (source / "parameter_maps.html").is_file():
        raise ValueError("Source must be a completed parameter map or sample study")
    if original.get("summary", {}).get("display_version"):
        raise ValueError("Choose the original simulation bundle, not a viewer revision")
    stages_file = source / "stages.json"
    if stages_file.is_file():
        stages = json.loads(stages_file.read_text())
        numerical, media = _stage(source, stages, "grid"), _stage(source, stages, "media")
    else:
        numerical = media = source
    config = json.loads((numerical / "resolved_config.json").read_text())
    physics = json.loads((numerical / "physics_parameters.json").read_text())
    settings = copy.deepcopy(config["maps"])
    settings.update(current_uA=float(config["input"]["amplitude_uA"]), ambient_K=physics["T0_K"])
    saved_axes = json.loads((numerical / "axis_values.json").read_text())
    axes = [np.asarray(saved_axes[name], float) for name in AXES]
    table = pd.read_csv(numerical / "grid.csv")
    catalog_path = media / "gif_catalog.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.is_file() else None
    standalone = bool(stages_file.is_file() if embed_gifs is None else embed_gifs)
    bundle = RunBundle.create(
        name=f"Fixed-scale viewer: {original['name']}", model=original["model"], kind="analysis",
        config={"viewer_refresh": {"source": str(source), "source_run_id": original["id"],
                                   "display_version": DISPLAY_VERSION, "embed_gifs": standalone},
                "original_numerical_config": config},
        output_root=output_root or "runs/local", command=command,
    )
    print(f"RUN_DIRECTORY={bundle.root}", flush=True)
    try:
        hashes = {}
        for name in ("grid.csv", "verification.csv", "physics_parameters.json", "axis_values.json"):
            path = numerical / name
            shutil.copy2(path, bundle.add_artifact(name, label=f"Unchanged original {name}"))
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rendered_catalog = copy.deepcopy(catalog)
        dependencies = []
        if catalog is not None:
            for record in rendered_catalog.values():
                for field in ("gif", "png"):
                    if field not in record:
                        continue
                    target = (media / record[field]).resolve()
                    if not target.is_relative_to(media) or not target.is_file():
                        raise ValueError(f"Missing or out-of-bundle {field} media")
                    if not standalone:
                        record[field] = quote(os.path.relpath(target, bundle.root), safe="/.")
            if not standalone and catalog:
                dependencies.append(str(source))
        render_physical_maps(table, axes, settings, bundle.root, gif_catalog=rendered_catalog)
        if standalone and catalog is not None:
            from neuristor.plotting.map_animations import write_standalone_map

            # The refreshed catalog still uses the original relative paths.
            # Embed from that original root, not from copied or re-run media.
            export = write_standalone_map(media, bundle.path("embedded.html"),
                                          viewer_path=bundle.root / "parameter_maps.html")
            bundle.path("embedded.html").replace(bundle.path("parameter_maps.html"))
        else:
            export = dict(standalone=not bool(catalog), embedded_gifs=0)
        summary = dict(source_run_id=original["id"], display_version=DISPLAY_VERSION,
                       resimulated=False, grid_points=len(table), retained_points=int(table.included.sum()),
                       catalogued_gifs=len(catalog or {}), media_dependencies=dependencies, **export)
        bundle.register_file("parameter_maps.html", label="Fixed-scale map viewer", media_type="text/html")
        bundle.register_file("heatmaps.png", label="Fixed-scale slice overview")
        bundle.write_json("display_scaling.json", heatmap_display_ranges(table, axes),
                          label="Shared retained-point color limits and fixed log-axis ranges")
        bundle.write_json("source_provenance.json",
                          dict(source=str(source), numerical_source=str(numerical), media_source=str(media),
                               resimulated=False, hashes=hashes,
                               original_catalog_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest()
                               if catalog is not None else None), label="Unchanged source evidence")
        bundle.write_json("source_hashes.json", snapshot_package(bundle), label="Frozen viewer source hashes")
        bundle.write_json("metrics.json", summary, label="Presentation-only revision summary")
        sharing = ("The map HTML is standalone, with its GIFs embedded." if summary["standalone"] else
                   "This local viewer links to the original GIFs. Keep its recorded media dependency; "
                   "do not email this HTML alone. Use --embed-gifs for a standalone export (potentially large).")
        bundle.write_text("report.md", f"""# Fixed-scale viewer

Original run: `{original['id']}`.

This is a display revision, not a new simulation or fit. Saved numerical tables
are byte-identical copies, with SHA-256 hashes in source_provenance.json. Original
GIF files are reused unchanged. Earlier completed bundles are not modified.

Both heatmap parameter axes and metric color limits remain constant as each
locked-parameter slider moves. Color limits use finite retained-point values
over every slider position, not rejected points or a forced 0–110 MHz range.
Gray cells remain excluded. Empty maps have no meaningful color bar; a constant
metric receives tiny display padding. Frequency and amplitude keep separate units.

{sharing}

Original conditions, caveats and scientific interpretation remain those of the
source report. This revision does not improve experimental agreement.
""", label="Display change and sharing instructions")
        bundle.complete(summary=summary)
    except BaseException as error:
        bundle.fail(error)
        raise
    return bundle


def refresh_existing_maps(*, project_root=None, output_root=None,
                          command="neuristor analyze refresh-maps --all-existing"):
    """Create one current viewer revision per completed historical map.

    Existing current revisions are skipped, so repeating the command does not
    create duplicate archives. Failed studies and viewer revisions are excluded.
    """
    records = RunRegistry(find_project_root(project_root)).discover()
    revised = {r.manifest.get("summary", {}).get("source_run_id") for r in records
               if r.status == "completed" and r.manifest.get("summary", {}).get("display_version") == DISPLAY_VERSION}
    sources = [r.root for r in records if r.status == "completed" and not r.legacy
               and r.id not in revised and not r.manifest.get("summary", {}).get("display_version")
               and (r.root / "parameter_maps.html").is_file()]
    return [refresh_map_viewer(source, output_root=output_root, command=command) for source in sources]
