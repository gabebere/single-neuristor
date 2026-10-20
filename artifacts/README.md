# One navigation hub, canonical evidence

Open [index.html](index.html) for categorized runs, reports and presentations.
Search by date, sample, run ID, workflow or generation status. The generated
[runs.csv](runs.csv) lists distinct run/job records; [files.csv](files.csv)
inventories numerical data, HTMLs, PDFs, figures, GIFs and original archives.

Refresh after producing or publishing results:

```bash
neuristor runs index
```

This folder is an index, not another database or a duplicate storage tree.
Canonical measurements stay in `data/experimental/`; complete new study evidence
stays together in its run bundle. Reviewed older evidence stays in `public_jobs/`.
Loose older document exports are indexed in their existing locations. Public/local
duplicates appear once in the run catalog (public copy preferred), while the file
inventory lists both physical copies so storage is transparent.

## Categories

- **Sample studies:** the unified fit → map → GIF workflow, one parent folder.
- **Parameter maps:** conditional screened maps and media/viewer extensions.
- **Fitting trials:** historical optimization attempts, successful or otherwise.
- **Measurements and diagnostics:** parameter estimates, audits and lab replays.
- **Simulations and sweeps:** solver demonstrations and sensitivity experiments.
- **Historical jobs:** preserved pre-refactor `job.json` format.

Categories describe a workflow, not scientific confidence. “Completed” means the
software wrote its outputs; “public” means locally archived, not necessarily
pushed to GitHub. Read each report and `docs/RESEARCH_HANDOFF.md` before adopting
parameters. Known superseded viewers and partial GIF benchmarks have explicit notes.

The corrected complete historical 1,258-GIF map is the `eaf6ab` bundle. Its HTML
still needs its adjacent GIFs/ folder: it predates standalone embedding. New sample
studies export one HTML with all animations embedded and do not create ZIPs.

The existing Desktop `VO2 Research Archive` is preserved. Its categorized
`START_HERE.md` and inventory remain the guide to those historical copies; this
project hub links canonical repository evidence, avoiding yet another copy.

No historical evidence was deleted or moved during this cleanup. The previous
long README is preserved in `docs/WORKFLOW_REFERENCE.md`; source one-offs remain
historical under `legacy_scripts/`, never part of the new pipeline.
## Navigation and inventories

Open `index.html` for the searchable research library. `runs.csv` records distinct
run statuses; `files.csv` lists generated files and sizes; `source_inventory.csv`
explains each owned Python file and distinguishes compatibility/historical code.
Rebuild these generated navigation files with `neuristor runs index`.
The [project directory](../START_HERE.html) is the first stop for non-coders.

## Delete and restore

Run `neuristor runs manage` and open its local URL to enable the new **Delete…**
and **Restore** controls. Confirmed deletion moves a local completed/failed run
to recoverable project trash; it does not permanently erase it or free space.
Published/running/recipe-referenced runs are protected. The ordinary exported
HTML is still read-only. [Details and terminal commands](../docs/RUN_MANAGEMENT.md).
