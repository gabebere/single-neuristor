# Historical optimizer starting points

`f426de_parameters.csv` is an exact copy of `parameters.csv` from the completed
local run `20260922_175152_background-settled-oscillations-for-original-lab_f426de`.
It is versioned here because four supported search recipes use it as input;
depending on ignored `runs/` made those recipes fail in a fresh checkout.

SHA-256: `e3b81ee4da263a00cd97e1e86a5bee3a36b0817f3c52818068ec7a648f4515e1`.

These are **starting candidates, not accepted calibrations**. Some rows have
failed physical/dynamic checks; the recipe's eligibility gates still apply.
No values or solver equations were changed by copying the table. Original
resolved configurations and the historical bundle remain unchanged.

See [the dated research history](../../docs/RESEARCH_HISTORY.md) and
[current fitting outcome](../../docs/SIMPLE_EXTREMA_LOSS_20261003.md).

`controlled_loss_20261020.csv` copies the three vectors in reviewed bundle
`20261003_183555_simple-relative-squared-extrema-fitting-pilot_96b944/parameters.csv`.
The 20 October paired recipes use the same three dynamic starting vectors,
projecting the six static coordinates and T0 to the fixed specimen baseline.
These are optimizer starting points, not newly accepted physical calibrations.

`optimizer_audit_20261020.csv` combines the older common seeds with diagnostic_1
from `543408` (frequency only) and `5ad999` (frequency and amplitude). Both expanded
searches use the same five named starts, projected to fixed static values and T0.
These are warm starts for an optimizer-budget test, not new measured constants.
