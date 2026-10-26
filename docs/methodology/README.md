# Methodology manuscript draft

This directory is a section-by-section methodology draft for the final paper.
`main.tex` is a lightweight compilation wrapper; substantive text lives in
`sections/` so each section can be reviewed before the next is written.

Current reviewed scope:

1. `sections/01_electrothermal_network.tex`: the voltage-driven electrothermal
   network equations from Zhang et al., their physical interpretation, and the
   discrete coupling convention implemented by the upstream and local code.

Intentionally deferred until Gabriel approves Section 1:

- the Almeida resistance and hysteresis law, including the exact proximity
  function preserved from Yuanhang's code;
- derivation of the ideal-current-source reduction;
- timestep update order and pseudocode;
- sample-specific parameter inference and fitting.

Scientific source-of-truth files used for Section 1:

- Zhang et al., DOI [10.1038/s41467-024-51254-4](https://doi.org/10.1038/s41467-024-51254-4); PDFs stay in ignored `private/`.
- [Original Zhang model in verified Git history](https://github.com/gabebere/single-neuristor/blob/12be9a9a729e03b164464fad0daab7c1b9d4f4c4/references/yuanhangzhang98-collective_dynamics_neuristor-217d4f0/model.py).
- `src/neuristor/model.py`
- `src/neuristor/current_drive_sim.py`
