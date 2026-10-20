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
