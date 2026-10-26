# Interfaces: controls and browsing

`dashboard.py` browses the run archive. `playground.py` provides lab-replay
controls. They delegate calculations to workflows; neither owns scientific
equations. The primary reproducible interface is the `neuristor` terminal command.

Launch with `neuristor dashboard` or `neuristor playground`. See command help for
options. The old root import paths are compatibility bridges.

`run_manager.py` serves the artifact library on loopback with confirmed,
recoverable Delete/Restore controls. Run lifecycle rules remain in `runs.py`.
Launch with `neuristor runs manage`; exported HTML stays read-only. No published
evidence is mutable through the manager. [Guide](../../../docs/RUN_MANAGEMENT.md).

`presentation_server.py` provides read-only loopback HTTP links for the two
presentation viewers and their reviewed GIF dependency bundle. Run
`neuristor analyze serve-presentation --background`, or the launcher in
`Simulations_on_VO2/`. No startup agent is installed; restart the service after
a reboot. Repository/private paths, directory listings and write methods are
unavailable. This interface does not change any scientific result or public bundle.
The Mica button (`/mica-map`) selects the newest completed published M2 1600 nm
review; in-progress expansions are not served through that shortcut.

To present from a copied project without installing scientific dependencies, use
`python3 src/neuristor/interfaces/presentation_server.py --present` (Python 3.9+).
It checks for the PDF/viewer folders, starts the server, and opens the system PDF
viewer. `neuristor present` exposes the same action in an installed environment.
See the [presentation guide](../../../Simulations_on_VO2/README.md) for the folders
needed on a colleague's computer. No absolute user path or Anaconda executable is
required by the portable launcher.
