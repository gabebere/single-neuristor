# Delete and restore local runs

Start the local manager from the repository root:

```bash
neuristor runs manage
```

Open the URL it prints. Keep the terminal running while using the manager;
Ctrl+C stops it. The server listens only on this computer, on a free loopback
port. There is no external deployment or additional Python dependency.

## Browser controls

Each eligible local row has **Delete…**. Confirmation displays the run name and
ID and explains that its entire folder—including nested study stages—will move.
After confirmation the run disappears from active history. Open **Trash — restore
deleted runs** and click **Restore** to bring it back to its original location.

Opening `runs/catalog/index.html` directly as a file still supports browsing, but
its delete buttons are disabled. Browsers do not grant ordinary HTML files
filesystem deletion authority. Use the local manager URL for file operations.

## Protections and meaning

- Only direct, non-symlink folders in `runs/local/` with `completed` or `failed` status
  can be moved. Unknown-status/live jobs, published bundles, and legacy jobs are protected.
- A run referenced by an experiment TOML is protected until that recipe is updated.
- A run supplying GIFs to a live fixed-scale viewer is protected until that viewer is removed.
- When a public copy exists, its run ID remains protected; the manager does not
  silently delete or alter either public evidence or its local duplicate.
- Trash preserves files and manifests byte-for-byte. A separate `deletion.json`
  records the original location and restore ID. Restoring never overwrites an existing folder.
- Trashed folders are excluded from active run discovery and the live file inventory.
- Trash is project-local `runs/local/.trash/`, **not** the macOS Trash. It remains ignored
  by Git and consumes disk space. No permanent erase or automatic expiry is implemented.
- Moving a run can make old handwritten links to its local path unavailable;
  restoring repairs those paths. Review any needed evidence before deleting.

Browser mutations require same-origin requests, a session token, and explicit
confirmation. Tokens are injected into the served page only; exported HTML
does not contain them. Hidden project paths and external symlink targets are not served.

## Terminal equivalents

```bash
neuristor runs delete RUN_ID         # prompts for confirmation
neuristor runs delete RUN_ID --yes   # explicit non-interactive confirmation
neuristor runs trash                 # lists restore IDs
neuristor runs restore TRASH_ID      # never overwrites an existing folder
```

No real run was deleted while implementing this feature. Tests exercise deletion,
restoration, cancelled confirmations, protected runs, path containment and HTTP
authentication using synthetic temporary projects only. The scientific model and
previous reviewed bundles are unchanged.
