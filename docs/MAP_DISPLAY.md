# Comparable parameter-map slices

New maps keep the two physical parameter axes and both color scales fixed when
a locked-variable slider moves. Frequency (MHz) and peak-to-peak voltage (mV)
have separate color ranges. Each range uses the min/max of finite retained points
over all slider positions in that map, not rejected values or a forced 0–110 MHz
range. Every slider family visits the same full retained domain, so their metric
limits agree. Different runs can still have different limits.

Blank/gray cells remain excluded, without interpolating new results. An entirely
empty map has no meaningful color bar. Constant-valued maps receive tiny display
padding. Logarithmic parameter axes include fixed half-cell edge padding; a
singleton axis receives a small fixed span. Slider changes also restore these
ranges if the viewer was manually zoomed.

## Update historical viewers without rerunning physics

```bash
neuristor analyze refresh-maps --all-existing
neuristor runs index
```

This creates a current display revision for each completed map or sample study.
Repeating it skips sources that already have a current revision. Historical
bundles, including their HTML, stay untouched; the artifact hub links each old
record to its **Updated fixed-scale viewer**. Failed studies are not converted
into successful results.

For just one source:

```bash
neuristor analyze refresh-maps --source runs/published/YOUR_RUN_ID
```

Revisions copy the original numerical tables byte-for-byte and record their
SHA-256 hashes. They use saved axes, never a resampled grid, and do not run the
solver. Original GIFs are unchanged. Fixed scaling does not improve agreement
with experimental data or revise the physical screening assumptions.

Dense GIF-map revisions use relative links to the original media to avoid copying
large archives. Their HTML is **not independently shareable**: keep the recorded
source folder. Local source runs with a live linked viewer are protected from
deletion until that viewer is removed. Sample-study revisions default to embedded
GIFs and remain standalone. To explicitly embed a dense map's media:

```bash
neuristor analyze refresh-maps --source runs/published/YOUR_RUN_ID --embed-gifs
```

Embedding can produce a very large HTML (base64 adds roughly one third to GIF
size). No ZIP is created automatically. Existing shareable ZIPs remain historical
exports with their original scales; they are not silently rewritten.

Implementation: `analysis/physical_maps.py` computes display ranges and renders
all initial slices/slider steps consistently. `orchestration/map_viewers.py`
handles evidence reuse. `tests/test_map_viewers.py` checks every emitted slider
action, excluded outliers, constant/empty maps, point IDs, source immutability,
standalone exports, dependency protection, and repeat-command idempotence.

## Verification on 7 October 2026

- Full suite: 150 tests passed (four existing arctanh warnings).
- Repository validation: 31 recipes, 126 run records, five paper evidence entries.
- Historical refresh: 13 viewers, 651 slider steps checked for identical axis
  ranges and color limits; all 2,519 non-embedded GIF references resolve.
- Fresh sample-study recipe `9f299d`: original numerical tables, GIF and PNG
  match the preceding `896056` run byte-for-byte. Only presentation changed.
- Browser check: switching Cth from 0.047873236 to 0.072477966 pJ/K changes the
  retained plane from 33 to 64 oscillators while keeping axes and color scales.

The main dense-map limits are 8.26489–109.89004 MHz and 429.37822–963.33757 mV.
Those are display ranges over its retained data, not newly inferred parameters.
