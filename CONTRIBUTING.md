# Contributing

Read [AGENTS.md](AGENTS.md), [the source guide](src/neuristor/README.md), and the
recipe relevant to your change. Scientific continuation also needs
[the research handoff](docs/RESEARCH_HANDOFF.md).

Use the existing physics, pure analysis functions, public workflows, and TOML
recipes. Put new implementation in the purpose-specific package, not an old-name
compatibility bridge or a top-level one-off script. Comments should explain units,
assumptions, and non-obvious numerical choices.

```bash
python -m pip install -e '.[dev]'
pytest -q
neuristor validate
neuristor validate --publication  # after staging: inspect exact Git payload
```

Run the affected recipe. Physics changes require two smaller timesteps and a
convergence assertion; never loosen a tolerance just to accept changed behavior.
New bundles must retain relative artifact paths and enough input/source evidence
to explain a dirty worktree. Preserve failed runs and do not edit public bundles.

Update documentation when a command, output contract, or scientific conclusion
changes. Do not infer experimental fit quality from a successful program exit.

Follow [the disclosure policy](docs/DATA_POLICY.md): approved numerical data and
our results may be public; third-party PDFs must not be included, even inside
release packages. Inspect each asset with `neuristor validate --distribution FILE`.
Update the authored-PDF hash allowlist only after reviewing the changed document.
