"""Finish the separate human-reviewed bundle without changing numerical evidence."""
from pathlib import Path
import hashlib
import html
import json
import shutil

import numpy as np
import pandas as pd

from neuristor.runs import RunBundle

project=Path('/Users/gabrielberezovsky/Documents/Projects/single-neuristor')
root=Path((project/'runs/scratch/overnight_review_path.txt').read_text().strip())
manifest=json.loads((root/'run.json').read_text())
assert manifest['status']=='running'
bundle=RunBundle(root,manifest,list(manifest['artifacts']))
audit=json.loads((root/'review_metrics.json').read_text())
metrics=json.loads((root/'metrics.json').read_text())
final=pd.read_csv(root/'final_comparison.csv')
final['order']=final.stage.map({name:i for i,name in enumerate([
 'final_refine_baseline_primary','final_refine_baseline_mean','final_refine_inherited_primary',
 'final_refine_inherited_mean','final_refine_volume_scaled_primary','final_refine_volume_scaled_mean'])})
final=final.sort_values('order')
family={'inherited':'Inherited','volume_scaled':'Scaled thermal volume'}
short=[]
for row in final.itertuples():
 short.append(dict(Family=family[row.family],f=f'{row.fraction:.6g}',Objective='Frequency + Vpp' if row.objective=='frequency-plus-Vpp' else '+ mean voltage',
  Valid=f'{row.passing_targets}/11',Defined=f'{row.defined_frequencies}/11',
  Frequency_MAPE=f'{row.frequency_MAPE_percent:.2f}%',Vpp_MAPE=f'{row.amplitude_MAPE_percent:.1f}%',
  Mean_RMSE=f'{row.mean_voltage_RMSE_mV:.2f} mV',Outside_T=f'{row.outside_temperature_records}/22',
  Feature_pairs=f'{row.feature_halving_pairs_passing}/22',Waveform_pairs=f'{row.waveform_halving_pairs_passing}/22'))
comparison=pd.DataFrame(short)
headers=['Family','f','Objective','Valid','Defined','Frequency_MAPE','Vpp_MAPE','Mean_RMSE','Outside_T','Feature_pairs','Waveform_pairs']
mdtable='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
mdtable+='\n'.join('| '+' | '.join(str(row[h]) for h in headers)+' |' for row in short)
report=f'''# Reviewed overnight filament-fraction experiment — 28 October 2026

**The broader search did not recover a quantitative, converged match to the presentation data.**
No candidate passes every prespecified gate and no calibration is adopted.
This bounded result neither identifies f nor rules out current localization.

## Coverage and provenance

The frozen job ran for **{metrics['elapsed_s']/3600:.3f} hours**, from 21:35 on
27 October to approximately 04:39 on 28 October, Israel time. It completed all
76 stages, including 43 searches, 21 profile verifications, six final replays and
six long-control cases, before the original eight-hour deadline. Nothing was skipped.
There were **21,232 stage-unique candidate evaluations**; repeated starts across
stages are included in that total. The saved vectors contain 20,698 distinct
physical parameter/f combinations. No restart was required.

Nine declared fractions (1 to 0.01), two thermal-bound families, forward/reverse
continuations and four training-selected geometric midpoints were tested.
The six static R(T) constants and T0 stayed fixed; C, Se, tau=Cth/Se and gamma
were refitted. The scaled family multiplies the Se interval by f while leaving
tau bounds unchanged. That is a separate diagnostic thermal assumption.
The authoritative solver and its normalized parallel-conductance model were unchanged.

Initial optimization used eight currents with three excluded by rank; all 22
recordings were replayed. Those records were seen historically and informed old
seeds, so these are not blind validation data. Final fits and selection use all
11 targets. Paired primary and mean-aware attempts have the same bounds/budgets
and declared warm starts, with independent deterministic random populations;
differences between their outcomes are not a controlled causal estimate of the
mean term's effect. All fits stopped under finite search/evaluation allowances,
not a proof of optimizer convergence.

Numerical source: **8ae0081ba73b90474545ca8032843bba057a1c06**.
Original full local bundle: `20261027_193458_overnight-fixed-f-filament-profile-and-cross-cur_5e8844`.
All 94 frozen input/source files and all 76 completed-stage inventories pass
hash verification. Independently reconstructed native voltage features match
the saved final tables for all **132 finest-step histories**. Branch-current
and Joule-power identities agree to maximum relative error
**{audit['native_balance_maximum_relative_error']:.3g}**. This checks branch
accounting, not device-terminal interpretation or experimental validation.

## Matched final comparisons

All rows use the finest **0.003125 ns** replay. Frequency MAPE uses the displayed
number of defined frequencies, which can exceed the number passing sustained-cycle
checks. Vpp errors use all 11 targets; mean-voltage RMSE uses all 22 recordings.
Outside-T counts refer to the full configured 284.8545–375.1625 K domain.
Feature/waveform pair counts cover two timestep halvings on 11 targets (22 pairs).

{mdtable}

The primary f=1 baseline has the lowest frequency/Vpp/mean errors among these
six retained final candidates, yet still misses one target, has Vpp MAPE about
1,437%, violates one quiet-control criterion and passes none of the 22 target
waveform comparisons. Smaller fractions do not solve that mismatch in this
bounded search. At f=0.75 the retained primary and mean-aware vectors coincide;
four of 22 measured-input paths exceed the calibrated temperature domain,
reaching 379.82 K. Scaled-family f≈0.866025 avoids those short-replay excursions
and false quiet-control oscillations, but its amplitude and mean errors remain
larger than the f=1 baseline. None has an accepted all-target scientific loss.

Some very small-f profile candidates have Vpp error near 100% because their
oscillations disappear. That is not amplitude agreement. Failed/undefined cycles,
temperature/proximity exclusions and control errors remain visible in
`audited_verification.csv`; `original_verified_candidates.csv` is not substituted
for the unchanged original `verified_candidates.csv`.

## Timestep and long-time checks

Final replays use 0.0125, 0.00625 and 0.003125 ns. No final candidate passes the
full feature-and-waveform convergence gate. Feature-only comparisons pass 12–17
of 22 pairs; waveform comparisons pass 0–2 of 22. Phase-sensitive waveform
failure is a failed operational numerical criterion, not proof of a physical
instability or a unique diagnosis of the integrator.

All six 2 µs constant-current cases fail the joint persistence/physical/convergence
gate. The f=1 primary case oscillates at all three selected currents at all
three steps, with convergent features, but the high-current case (~569.55 µA)
leaves the temperature domain at every step. It passes two of the three
current-specific joint checks. The other five retain no sustained low-current
oscillation (~228.28 µA), pass the middle-current check and also leave the
temperature domain at the high current. Thus they pass one of three joint checks.
All long controls keep admissible proximity. The full current-by-current record
is in `long_control_audit.csv` and the unmodified constant-current tables.

## Reporting correction discovered during review

DataFrame conversion changed missing-cycle `None` losses to NaN. The original
post-processing helper consequently serialized partial diagnostic losses as null.
The optimizer's direct-row objectives and original acceptance gates were not
changed by that reporting path. The audit explicitly restores missing status
before averaging defined losses. **Both selected refinement seeds are identical
under the corrected ranking**, and every recomputed acceptance gate matches the
original. The maintained helper is fixed with a regression test (post-run commit
1322225); frozen code, optimizer records and completed evidence remain unchanged.
`audited_verification.csv` records corrected partial values and denominators;
accepted losses remain undefined. It also distinguishes all-record temperature
maxima from the original summary's target-only maximum.

## Recommended next step

Do not infer a specimen f from this negative fit or add an arbitrary f(t) simply
to reduce the loss. First reconcile the exported channels with device-terminal
voltage and imposed current: document probe nodes, conversion factors, gain,
offset and channel delay, ideally using a known-resistor control. In parallel,
perform a dedicated upstream-fidelity/timestep audit on a few fixed representative
vectors before interpreting another optimizer as calibrated physics. Preserve
the inherited float32 hysteresis unless that dedicated audit justifies a change.

If channel mapping is confirmed and independent evidence indicates surrounding
film warming, a two-temperature model with measured/constrained thermal coupling
is a more interpretable next hypothesis than a freely varying f. That extension
has not been implemented or tested here.

## Archive contents and reproducibility

This separate compact review preserves the original `metrics.json`,
`resolved_config.json`, inputs, executable source, plans, selected stage records,
all verification/convergence/constant-control tables, all six final sampled
waveform tables and final figures. `preserved_artifact_hashes.json` proves byte
identity. The active review manifest lists only files actually included.
The original manifest/state/report are retained under `original_*` names.

The approximately 964 MB full numerical bundle stays local and immutable.
Full optimizer JSONL histories, large native NPZ arrays, intermediate checkpoint
details and profile-only sampled waveforms are deliberately omitted here.
`full_local_inventory.json` gives sizes, hashes and an explicit inclusion flag.
Those omitted bytes are **not backed up on GitHub** by this compact review.
The versioned recipe and original measured numerical inputs support rerunning
the workflow; wall-clock allowances and hardware can select different optima,
so a new search is not promised to reproduce optimizer choices bit for bit.

Run `neuristor analyze filament-profile --config experiments/current/specimen_filament_overnight_20261027.toml`
from an installed checkout. Use the preserved source commit for the exact original
implementation. Copied execution configurations are original provenance, not
instructions to resume this completed review. No third-party PDFs are included.
'''
# A copied raw CSV keeps its original name; do not imply another original exists.
report=report.replace('`original_verified_candidates.csv` is not substituted\nfor the unchanged original `verified_candidates.csv`.',
    'the unchanged original summary remains in `verified_candidates.csv`.')
bundle.write_text('report.md',report,label='Reviewed methods, outcomes, correction and next step')
bundle.write_text('README.md','# Reviewed overnight filament search\n\nOpen [START_HERE.html](START_HERE.html) or [report.md](report.md).\n\nNo candidate passes every prespecified gate. Original metrics are preserved; independent review metrics are separate.',label='Review entry point')
profiles=pd.read_csv(root/'audited_verification.csv');profiles=profiles[profiles.stage.str.startswith('profile_check_')]
profilecols=['family','fraction','passing_targets','defined_frequencies','frequency_MAPE_percent','amplitude_MAPE_percent',
 'mean_voltage_RMSE_mV','outside_temperature_records','inadmissible_proximity_records']
controls=pd.read_csv(root/'long_control_audit.csv')
links=[]
for folder in sorted((root/'stages').iterdir()):
 if (folder/'verification.csv').is_file():
  link=f'<a href="stages/{folder.name}/verification.csv">features</a> · <a href="stages/{folder.name}/convergence.csv">timestep checks</a>'
  if (folder/'traces.csv').exists():
   link+=f' · <a href="stages/{folder.name}/traces.csv">all 22 sampled waveforms</a>'
  links.append(f'<li>{html.escape(folder.name)}: {link}</li>')
page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reviewed overnight filament search</title><style>
body{{font:16px system-ui;color:#172735;line-height:1.6;margin:2rem auto;max-width:1250px;padding:0 1rem}}
h1,h2{{line-height:1.2}}.result{{padding:1rem;border-left:5px solid #c48c28;background:#fff5dc}}
.scroll{{overflow:auto}} table{{border-collapse:collapse;font-size:13px}}td,th{{padding:.5rem;border:1px solid #ccd5df;text-align:left}}
th{{background:#edf3f7}}img{{max-width:100%;height:auto}} a{{color:#0d628f}}details{{margin:1rem 0}}
</style><h1>The overnight filament search is complete</h1>
<p class="result"><strong>No candidate recovered a quantitative, converged match.</strong>
The job completed 21,232 stage-unique evaluations in 7.06 hours, including both thermal assumptions,
all planned checks and long controls. No calibration is adopted. This does not rule out a filament.</p>
<p><a href="report.md">Full reviewed report</a> · <a href="final_comparison.csv">Final comparisons</a> ·
<a href="review_metrics.json">Integrity audit</a> · <a href="plan.json">Original plan</a> ·
<a href="full_local_inventory.json">Included and omitted evidence</a></p>
<h2>Six final comparisons</h2>
<p>Finest timestep: 0.003125 ns. <strong>Valid</strong> means sustained target checks passed;
<strong>Defined</strong> is the frequency-error denominator. Frequency errors use defined targets,
Vpp errors all 11 targets, and mean-voltage error all 22 records. Outside-T counts are out of 22.
Feature and waveform counts cover 22 comparisons (11 targets × two halvings).</p>
<div class="scroll">{comparison.to_html(index=False,escape=True)}</div>
<p>All six fail the joint long-control test. Smaller fractions and the mean-aware attempts did not solve
the discrepancy. Very small-f candidates can appear to reduce amplitude error by losing oscillations;
missing cycles remain failures, not zero error.</p>
<h2>What the final waveforms show</h2><p>Black: measured. Blue: simulated.
Every final candidate is shown at low/mid/high measured currents; all 22 waveforms are linked below.</p>
<img src="figures/final_waveforms.png" alt="Six final candidates compared with measured voltage at three currents">
<h2>Profile across fractions</h2><img src="figures/profile.png" alt="Frequency, amplitude, mean and valid-cycle diagnostics across filament fractions">
<details><summary>All 21 profile checks, including failed cases</summary><div class="scroll">
{profiles[profilecols].to_html(index=False,na_rep='undefined',float_format=lambda x:f'{x:.4g}')}</div></details>
<h2>Long-time checks</h2><p>At 2 µs, the high-current control (~569.55 µA) leaves the calibrated temperature
domain for every final candidate and timestep. Only the primary f=1 candidate retains sustained low-current
oscillations (~228.28 µA). None passes all three currents.</p>
<details><summary>Current-by-current long-control audit</summary><div class="scroll">{controls.to_html(index=False,na_rep='undefined',float_format=lambda x:f'{x:.5g}')}</div></details>
<h2>Recommended next step</h2><p>Confirm channel conversion, gain, offsets, delay and probe nodes with a
known-resistor control. Audit timestep/hysteresis fidelity on fixed representative vectors before another fit.
If independent evidence supports surrounding-film warming, consider a constrained two-temperature model.
An arbitrary varying fraction would add freedom without resolving the current uncertainty.</p>
<h2>Review and provenance</h2><p>All 94 frozen input/source files and 76 stage inventories pass hash checks.
All 132 native finest-step histories reproduce the saved final features and preserve branch-current/power
identities within 2.60×10⁻⁷ relative error. A post-processing NaN issue erased partial diagnostics in the original
summary. Corrected diagnostics preserve both selected refinement seeds and every acceptance decision;
original files remain unchanged. See the report for details.</p>
<p>Three currents were initially excluded from optimization but seen historically. Final fits use all 11 targets;
this is not blind validation. No accepted all-target loss or inferred physical area fraction is established.</p>
<p>The full ~964 MB numerical run remains local. This compact archive omits large native arrays and histories;
their hashes and exclusion flags are recorded. Those omitted bytes are not backed up on GitHub.</p>
<details><summary>Numerical tables and sampled waveforms</summary><ul>{''.join(links)}</ul></details>
<p><a href="audited_verification.csv">Audited profile/final table</a> · <a href="long_control_audit.csv">Long controls</a> ·
<a href="native_conservation_audit.csv">Native audit</a> · <a href="search_parameters.csv">Named parameters</a> ·
<a href="current_split.csv">Current split</a> · <a href="original_report.md">Original generated report</a></p></html>'''
bundle.write_text('START_HERE.html',page,label='Offline reviewed comparison and evidence',media_type='text/html')
shutil.copy2(__file__,bundle.add_artifact('review_finalize.py',label='Archival review/report generation source'))
for path in sorted(root.rglob('*')):
 if path.is_file() and path.name!='run.json':
  bundle.register_file(path.relative_to(root),label='Reviewed evidence '+path.relative_to(root).as_posix())
hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*')
 if p.is_file() and p.name not in ['run.json','review_artifact_hashes.json']}
bundle.write_json('review_artifact_hashes.json',hashes,label='Complete curated artifact hashes')
bundle.complete(summary=dict(metrics,reviewed=True,source_run_id='20261027_193458_overnight-fixed-f-filament-profile-and-cross-cur_5e8844',
 native_histories_audited=132,summary_correction_changes_selection=False))
print(root)
