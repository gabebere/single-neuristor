# Verification tests

Run `pytest -q` from the repository root. Tests cover the physics implementation,
zero-capacitance behavior, hysteresis fidelity, timestep convergence, feature
extraction, fitting helpers, interfaces, and output-bundle contracts.

`test_package_structure.py` checks compatibility imports, recursive source
snapshots, navigation targets, and the paper-evidence recipe index.
`test_run_management.py` tests confirmed trash/restore and manager HTTP protections
using only synthetic temporary bundles; real research data is never deleted by tests.

Passing tests demonstrates the tested numerical/software properties. It does not
establish agreement with every laboratory trace. Also run `neuristor validate`
and the exact recipe affected by a change; see [AGENTS.md](../AGENTS.md).
