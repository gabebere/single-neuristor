"""Portable point-to-GIF drill-downs, using authoritative current-drive physics.

The map remains immutable. Animation replays have their own refinement checks;
no cosmetic interpolation invents a smooth hysteresis trajectory.
"""

from __future__ import annotations

from dataclasses import replace
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuristor.current_drive_sim import simulate_current_parameter_grid
from neuristor.analysis.physical_maps import screen_trajectory


def resolved_rt_path(time_ns, temperature_K, resistance_ohm, *, max_dT_K=0.25, max_dlogR=0.08):
    """Keep every integrated sample, breaking large jumps instead of inventing chords.

    The thresholds are visual resolution criteria, not physical constraints.
    Breaks remain breaks even after refinement if the model has a discontinuity.
    Genuine continuous minor loops are preserved, not projected onto major branches.
    """
    t, temp, r = map(lambda v: np.asarray(v, float), (time_ns, temperature_K, resistance_ohm))
    if t.ndim != 1 or temp.shape != t.shape or r.shape != t.shape or len(t) < 2:
        raise ValueError("Matching trajectory arrays with at least two samples required")
    if not np.isfinite(np.r_[t, temp, r]).all() or np.any(r <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("Finite ordered times and positive resistance required")
    jumps = (np.abs(np.diff(temp)) > max_dT_K) | (np.abs(np.diff(np.log(r))) > max_dlogR)
    before = np.flatnonzero(jumps) + 1
    # Insert NaNs between samples; never discard an endpoint or connect across it.
    return (
        np.insert(t, before, t[before]),
        np.insert(temp, before, np.nan),
        np.insert(r, before, np.nan),
        int(jumps.sum()),
    )


def animation_controls_html(catalog):
    """Embed local relative media links and right-click/keyboard point controls.

    No fetch, server or external dependency: the extracted HTML/GIF folder works
    under file://. Heatmap customdata follows the active slider, so point IDs do
    not depend on trace index, rounded coordinates or stale slice selections.
    """
    data = json.dumps(catalog, allow_nan=False).replace("<", "\\u003c")
    return """
<style>
#point-tools{position:sticky;bottom:0;padding:12px;background:#fff;border:1px solid #bbb;z-index:10}
#point-menu{display:none;position:fixed;padding:10px;background:white;box-shadow:0 2px 12px #888;z-index:100}
#gif-dialog{width:min(880px,90vw);max-height:90vh;overflow:auto}#gif-dialog img{width:100%;height:auto}
#gif-dialog button{float:right}button,a{cursor:pointer}
</style>
<div id="point-tools"><span id="point-label">Select a retained point, or right-click a point to view its simulation.</span>
 <button id="point-open" disabled>View simulation GIF</button></div>
<div id="point-menu"><button id="point-menu-open">View simulation GIF</button></div>
<dialog id="gif-dialog"><button id="gif-close">Close</button><h2 id="gif-title"></h2>
<p id="gif-detail"></p><img id="gif-image" alt="Synchronized current, voltage and resistance-temperature animation">
<p><a id="gif-download" download>Download GIF</a></p></dialog>
<script>
(() => {
 const catalog=CATALOG, plots=['map-3d','map-slice-0','map-slice-1','map-slice-2'];
 const menu=document.getElementById('point-menu'), dialog=document.getElementById('gif-dialog');
 const label=document.getElementById('point-label'), open=document.getElementById('point-open');
 let selected=null;
 function select(id){
   selected=catalog[String(id)]||null;
   open.disabled=!selected || !selected.gif;
   label.textContent=selected ? selected.caption+' · '+selected.status : 'No animation available for this point.';
 }
 function show(){
   menu.style.display='none'; if(!selected || !selected.gif)return;
   document.getElementById('gif-title').textContent=selected.caption;
   document.getElementById('gif-detail').textContent=selected.detail;
   document.getElementById('gif-image').src=selected.gif;
   document.getElementById('gif-download').href=selected.gif;
   if(!dialog.open)dialog.showModal();
 }
 for(const id of plots){
   const plot=document.getElementById(id);let hovered=null;
   const pointId=p=>{
     let value=p.customdata;
     // Plotly versions can omit event customdata or deliver typed arrays.
     // Fall back to the clicked trace and its current (slider-updated) cell.
     if(value===undefined || value===null){
       const source=p.data?.customdata || plot.data?.[p.curveNumber]?.customdata;
       const index=p.pointNumber ?? p.pointIndex;
       if(Array.isArray(index) || ArrayBuffer.isView(index))
         value=source?.[index[0]]?.[index[1]];
       else value=source?.[index];
     }
     if(Array.isArray(value) || ArrayBuffer.isView(value))value=value[6];
     const number=Number(value);
     return value!==null && value!==undefined && Number.isInteger(number) ? number : null;
   };
   plot.on('plotly_hover',e=>{hovered=pointId(e.points[0]);});
   plot.on('plotly_unhover',()=>{hovered=null;});
   plot.on('plotly_click',e=>select(pointId(e.points[0])));
   plot.on('plotly_sliderchange',()=>{hovered=null;selected=null;open.disabled=true;menu.style.display='none';label.textContent='Select a point in the new slice.';});
   plot.addEventListener('contextmenu',e=>{
     if(hovered===null || hovered===undefined || Number(hovered)<0)return;
     e.preventDefault();select(hovered);if(open.disabled)return;
     menu.style.display='block';menu.style.left=Math.min(e.clientX,window.innerWidth-190)+'px';
     menu.style.top=Math.min(e.clientY,window.innerHeight-65)+'px';
   },true);
 }
 open.addEventListener('click',show);document.getElementById('point-menu-open').addEventListener('click',show);
 document.getElementById('gif-close').addEventListener('click',()=>dialog.close());
 document.addEventListener('pointerdown',e=>{if(!menu.contains(e.target))menu.style.display='none';});
 document.addEventListener('keydown',e=>{if(e.key==='Escape')menu.style.display='none';});
})();
</script>""".replace("CATALOG", data)


def replay_animation_batch(points, params, current_uA, map_settings, settings):
    """Re-integrate one batch, halving dt for under-resolved or changed points.

    Compare frequency/Vpp against the previous timestep (the saved map initially).
    Each point's full thermal/proximity eligibility is also rechecked. Limit
    refinement: irreducible jumps are documented and drawn with breaks.
    """
    result = {}
    previous = {
        int(r.point_index): dict(frequency_MHz=r.frequency_MHz, extrema_vpp_mV=r.extrema_vpp_mV)
        for _, r in points.iterrows()
    }
    pending = points.copy()
    dt = float(settings["dt_ns"])
    for level in range(int(settings["maximum_refinements"]) + 1):
        outputs = simulate_current_parameter_grid(
            current_uA,
            replace(params, dt_s=dt * 1e-9),
            C_F=pending.C_pF.to_numpy() * 1e-12,
            C_th_J_per_K=pending.C_th_pJ_per_K.to_numpy() * 1e-12,
            S_e_W_per_K=pending.S_e_mW_per_K.to_numpy() * 1e-3,
        )
        retry = []
        for (_, point), out in zip(pending.iterrows(), outputs):
            key = int(point.point_index)
            features = screen_trajectory(out, params, map_settings)
            changes = []
            for name in ("frequency_MHz", "extrema_vpp_mV"):
                a, b = previous[key][name], features[name]
                changes.append(abs(b - a) / max(abs(b), 1e-12) if a is not None and b is not None else float("inf"))
            display = (out["t"] * 1e9 >= map_settings["window_ns"][0]) & (
                out["t"] * 1e9 <= map_settings["window_ns"][1]
            )
            temp, r = out["T"][display].astype(float), out["R"][display].astype(float)
            max_dT = float(np.max(np.abs(np.diff(temp))))
            max_dlogR = float(np.max(np.abs(np.diff(np.log(r)))))
            resolved = max_dT <= 0.25 and max_dlogR <= 0.08
            converged = max(changes) <= map_settings["convergence_fraction"]
            record = dict(
                point_index=key,
                dt_ns=dt,
                refinement_level=level,
                converged=converged,
                visually_resolved=resolved,
                maximum_dT_K=max_dT,
                maximum_dlogR=max_dlogR,
                frequency_change_fraction=changes[0] if np.isfinite(changes[0]) else None,
                amplitude_change_fraction=changes[1] if np.isfinite(changes[1]) else None,
                **features,
            )
            result[key] = (point, out, record)
            if level < settings["maximum_refinements"] and (not converged or not resolved):
                retry.append(key)
            previous[key] = features
        if not retry:
            break
        pending = pending[pending.point_index.isin(retry)]
        dt /= 2
    return list(result.values())


def write_standalone_map(source: Path, destination: Path, *, introduction: str = "", viewer_path: Path | None = None) -> dict:
    """Embed every catalogued GIF so one HTML file can be emailed on its own.

    The relative-folder viewer and original GIFs are preserved. Embedding uses
    the same point IDs/control code, not a second simulation or plotting path.
    Base64 expands media by roughly one third; large grids may exceed email
    attachment limits. No ZIP, server, external fetch or CDN is introduced.
    Media are streamed one GIF at a time so a dense map does not require several
    full copies of its embedded animations in Python memory during export.
    ``viewer_path`` may supply a freshly rendered viewer without changing the
    completed media bundle. Its catalog must still match the original media.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    catalog = json.loads((source / "gif_catalog.json").read_text())
    if destination == source / "parameter_maps.html":
        raise ValueError("Standalone export must not overwrite its source viewer")
    files = []
    media_bytes = 0
    for key, record in catalog.items():
        file = (source / record["gif"]).resolve()
        if not file.is_relative_to(source) or file.suffix.lower() != ".gif":
            raise ValueError("GIF catalog contains a path outside its bundle")
        with file.open("rb") as handle:
            signature = handle.read(6)
        if signature not in (b"GIF87a", b"GIF89a"):
            raise ValueError(f"Invalid GIF: {file.name}")
        media_bytes += file.stat().st_size
        files.append((key, record, file))
    original = Path(viewer_path or source / "parameter_maps.html").read_text()
    marker = " const catalog="
    if original.count(marker) != 1:
        raise ValueError("Expected exactly one point-media catalog")
    start = original.index(marker) + len(marker)
    parsed, length = json.JSONDecoder().raw_decode(original[start:])
    # The catalog file is written with sorted keys; the viewer may preserve
    # insertion order. Compare values, not their JSON spelling/order.
    if parsed != catalog:
        raise ValueError("Viewer and GIF catalog disagree")
    prefix = original[:start]
    if introduction:
        if "<body>" not in prefix:
            raise ValueError("Map HTML has no body")
        prefix = prefix.replace("<body>", "<body>" + introduction, 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        handle.write(prefix + "{")
        for index, (key, record, file) in enumerate(files):
            if index:
                handle.write(", ")
            embedded_record = dict(
                record, gif="data:image/gif;base64," + base64.b64encode(file.read_bytes()).decode("ascii")
            )
            handle.write(json.dumps(key) + ": " + json.dumps(embedded_record, allow_nan=False).replace("<", "\\u003c"))
        handle.write("}" + original[start + length :])
    return dict(embedded_gifs=len(files), gif_bytes=media_bytes, html_bytes=destination.stat().st_size, standalone=True)


def render_point_animation(point, out, record, branches, destination, settings, current_uA, ambient_K):
    """Render resolved samples in three panels; there is no matched lab trace here."""
    from neuristor.plotting.replay_plots import write_scope_gif

    destination = Path(destination)
    stem = f"point_{int(point.point_index):05d}"
    gif, png = f"GIFs/{stem}.gif", f"Figures/{stem}.png"
    t = out["t"].astype(float) * 1e9
    frame = pd.DataFrame(
        dict(
            time_ns=t,
            current_step_uA=current_uA,
            input_current_uA=out["I_in"].astype(float) * 1e6,
            simulated_voltage_mV=out["V_vo2"].astype(float) * 1e3,
            temperature_K=out["T"],
            resistance_ohm=out["R"],
        )
    )
    start = settings["window_ns"][0]
    f = record["frequency_MHz"] or float(point.frequency_MHz)
    stop = min(settings["window_ns"][1], start + max(80.0, 3000 / f))
    caption = f"Cth={point.C_th_pJ_per_K:.5g} pJ/K · Se={point.S_e_mW_per_K:.5g} mW/K · C={point.C_pF:.5g} pF"
    write_scope_gif(
        frame,
        branches,
        destination / gif,
        frames=int(settings["frames"]),
        snapshot_path=destination / png,
        stacked_amperes=True,
        time_window_ns=(start, stop),
        trajectory_breaks=True,
        parameter_caption=caption,
    )
    status = (
        "refinement checks pass" if record["converged"] and record["eligible"] else "diagnostic: refinement checks fail"
    )
    if not record["visually_resolved"]:
        status += "; unresolved jumps drawn with breaks"
    detail = (
        f"{current_uA:g} µA constant input; T0={ambient_K:g} K. No matching experimental trace. "
        f"dt={record['dt_ns']:.6g} ns; playback {start:g}–{stop:.2f} ns. {status}. "
        "Minor loops are model trajectories; major branches are guides. No smoothing or interpolation."
    )
    return dict(record, gif=gif, png=png, caption=caption, detail=detail, status=status)
