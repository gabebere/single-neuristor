"""Review existing map/GIF evidence without rerunning or changing the solver.

Run from the repository root. All comparisons hold two coordinates fixed and
use adjacent positions on the ORIGINAL grid, never bridges across rejected cells.
"""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neuristor.runs import RunBundle

source = Path('public_jobs/20261007_074702_portable-screened-parameter-map-with-point-gifs_eaf6ab')
viewer = Path('public_jobs/20261007_123036_fixed-scale-viewer-portable-screened-parameter-m_4325d1')
axes = ['C_pF', 'C_th_pJ_per_K', 'S_e_mW_per_K']
grid = pd.read_csv(source/'grid.csv')
checks = pd.read_csv(source/'animation_checks.csv')
assert len(grid) == 12167 and grid.point_index.is_unique
assert grid.included.sum() == 1258 and checks.point_index.is_unique
assert set(checks.point_index) == set(grid.loc[grid.included, 'point_index'])
assert (source/'grid.csv').read_bytes() == (viewer/'grid.csv').read_bytes()
joined = grid[['point_index', *axes, 'included']].merge(checks, on='point_index', how='left', validate='one_to_one')
joined['keep'] = joined.included & joined.eligible.eq(True) & joined.converged.eq(True)
retained = joined[joined.keep].copy()
assert len(retained) == 1256

bundle = RunBundle.create(name='Saved full parameter exploration trends', model='current', kind='analysis',
    config={'source':str(source), 'viewer':str(viewer), 'resimulated':False,
            'comparison':'adjacent original grid positions, two other coordinates fixed',
            'relative_change_thresholds':[0.01,0.05]}, output_root='runs',
    command='python outputs/map_trends/review_saved_map.py')
try:
    shutil.copy2(__file__, bundle.add_artifact('analysis.py',label='Exact saved-data analysis and figure authoring'))
    hashes={}
    for name in ['grid.csv','verification.csv','animation_checks.csv','physics_parameters.json','axis_values.json']:
        shutil.copy2(source/name, bundle.add_artifact('inputs/'+name,label='Immutable input '+name))
        hashes[name]=hashlib.sha256((source/name).read_bytes()).hexdigest()
    bundle.write_json('source_provenance.json',{'source':str(source),'viewer':str(viewer),'sha256':hashes},label='Saved input provenance')
    pairs=[]
    for axis in axes:
        other=[x for x in axes if x!=axis]
        for _, group in joined.groupby(other, sort=True):
            group=group.sort_values(axis)
            for low,high in zip(group.iloc[:-1].itertuples(),group.iloc[1:].itertuples()):
                if not (low.keep and high.keep):continue
                assert all(getattr(low,k)==getattr(high,k) for k in other)
                pairs.append(dict(axis=axis,low_point=int(low.point_index),high_point=int(high.point_index),
                    low_value=getattr(low,axis),high_value=getattr(high,axis),
                    frequency_change=high.frequency_MHz/low.frequency_MHz-1,
                    amplitude_change=high.extrema_vpp_mV/low.extrema_vpp_mV-1))
    pairs=pd.DataFrame(pairs)
    pairs.to_csv(bundle.add_artifact('matched_neighbors.csv',label='Every valid adjacent one-coordinate comparison'),index=False)
    summaries=[]
    for axis,part in pairs.groupby('axis',sort=False):
        for metric in ['frequency','amplitude']:
            change=part[metric+'_change']
            for threshold in [.01,.05]:
                summaries.append(dict(axis=axis,metric=metric,threshold=threshold,pairs=len(part),
                    decreasing=int((change < -threshold).sum()),
                    within_threshold=int((change.abs()<=threshold).sum()),
                    increasing=int((change > threshold).sum()),median_change_percent=float(change.median()*100)))
    summary=pd.DataFrame(summaries)
    assert (summary.decreasing+summary.within_threshold+summary.increasing).eq(summary.pairs).all()
    summary.to_csv(bundle.add_artifact('neighbor_summary.csv',label='Descriptive direction counts at 1% and 5%'),index=False)

    # Choose examples by coverage, not favorable residuals or large effects.
    # Restrict illustrations to visually resolved GIFs; direction counts above
    # include all eligible, converged finer replays.
    illustrations=retained[retained.visually_resolved.eq(True)]
    groups=[]; examples=[]
    for axis in axes:
        other=[x for x in axes if x!=axis]
        candidates=list(illustrations.groupby(other,sort=True))
        _,part=max(candidates,key=lambda item:len(item[1]))
        part=part.sort_values(axis);groups.append(part)
        examples.append(part.iloc[[0,len(part)//2,-1]].assign(varied_axis=axis))
    examples=pd.concat(examples)
    examples.to_csv(bundle.add_artifact('representatives.csv',label='Coverage-selected low middle high examples'),index=False)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(13,6.1))
    labels=['Electrical capacitance C (pF)','Thermal capacitance Cth (pJ/K)','Thermal conductance Se (µW/K)']
    for col,(axis,part) in enumerate(zip(axes,groups)):
        # Leave gaps wherever a grid point failed a replay/visual check.
        line=part.set_index(axis).reindex(sorted(grid[axis].unique()))
        x=line.index.to_numpy()*(1000 if axis=='S_e_mW_per_K' else 1)
        for row,(metric,label) in enumerate([('frequency_MHz','Frequency (MHz)'),('extrema_vpp_mV','Vpp (mV)')]):
            ax=axs[row,col];ax.plot(x,line[metric],'-o',ms=4,color=['#17699a','#bd5b12'][row]);ax.set_xscale('log');ax.set_xlabel(labels[col]);ax.set_ylabel(label);ax.grid(alpha=.2)
        fixed=[k for k in axes if k!=axis]
        short={'C_pF':'C','C_th_pJ_per_K':'Cth','S_e_mW_per_K':'Se'}
        axs[0,col].set_title(', '.join(f'{short[k]}={part[k].iloc[0]*(1000 if k=="S_e_mW_per_K" else 1):.4g}' for k in fixed),fontsize=11)
    fig.suptitle('One parameter varies; the other two stay fixed\n500 µA constant current, 298 K, gamma = 0.95627',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(bundle.add_artifact('figures/controlled_trends.png',label='Controlled parameter slices from finer GIF replays'),dpi=180);plt.close(fig)

    # A grid-aligned constant C/Cth ray: pick the conductance with most resolved
    # samples, choosing the first ascending conductance on ties.
    scale=illustrations[np.isclose(illustrations.C_pF/illustrations.C_th_pJ_per_K,40)].copy()
    conductance=scale.groupby('S_e_mW_per_K',sort=True).size().idxmax()
    scale=scale[scale.S_e_mW_per_K==conductance].sort_values('C_pF')
    scale['scale_factor']=scale.C_pF/scale.C_pF.iloc[0]
    scale['expected_frequency_MHz']=scale.frequency_MHz.iloc[0]/scale.scale_factor
    scale.to_csv(bundle.add_artifact('common_scale.csv',label='Saved independent integrations along a common-capacitance ray'),index=False)
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    axs[0].plot(scale.scale_factor,scale.frequency_MHz,'o',label='Saved finer replays');axs[0].plot(scale.scale_factor,scale.expected_frequency_MHz,'--',label='f(1) / scale');axs[0].set_ylabel('Frequency (MHz)');axs[0].legend()
    axs[1].plot(scale.scale_factor,scale.extrema_vpp_mV,'o-');axs[1].set_ylabel('Vpp (mV)');axs[1].set_ylim(0,700)
    for ax in axs:ax.set_xscale('log');ax.set_xlabel('Common multiplier of C and Cth');ax.grid(alpha=.2)
    fig.suptitle(f'Same ratio C/Cth = 40 pF K/pJ; Se = {1000*conductance:.4g} µW/K')
    fig.tight_layout();fig.savefig(bundle.add_artifact('figures/common_scale.png',label='Frequency scales while amplitude stays approximately fixed'),dpi=180);plt.close(fig)
    for row in examples[examples.varied_axis=='C_pF'].itertuples():
        for column in ['gif','png']:
            original=source/getattr(row,column)
            assert original.is_file()
            shutil.copy2(original,bundle.add_artifact(f'examples/point_{row.point_index:05d}.'+column,label=f'C example {row.point_index} '+column))
    metrics=dict(grid_points=len(grid),original_retained=int(grid.included.sum()),
        finer_replays_retained=len(retained),visually_resolved=int(checks.visually_resolved.sum()),
        common_scale_points=len(scale),common_scale_max_frequency_discrepancy_fraction=float((scale.frequency_MHz/scale.expected_frequency_MHz-1).abs().max()),
        common_scale_amplitude_span_fraction=float((scale.extrema_vpp_mV.max()-scale.extrema_vpp_mV.min())/scale.extrema_vpp_mV.iloc[0]),
        resimulated=False)
    bundle.write_json('metrics.json',metrics,label='Saved-data scope and scaling checks')
    report='''# What can the full exploration tell us?

The original viewer covers 23³ = 12,167 combinations of C, Cth and Se. Gamma is
fixed at 0.956269682, current at 500 µA, T0 and initial temperature at 298 K,
and the specimen major-loop resistance fit is fixed. This is a three-parameter
conditional exploration, not a four-parameter specimen calibration. Source
screens used 284.8545–375.1625 K (a padded range) and a 110 MHz operating ceiling.
The stricter later boundary study must be consulted for boundary interpretation.

1,258 points passed the old map screens. All have GIFs. Finer GIF replays retain
1,256 eligible, converged points; 1,174 pass the visual-resolution checks.
The original map membership remains unchanged. Statistics use finer GIF features
on the 1,256 retained replays. Illustrations additionally require visual resolution.

## Tendencies with the other two parameters held fixed

Increasing C slows oscillations and usually raises Vpp or approaches an amplitude
plateau. Increasing Cth also slows oscillations but usually reduces Vpp or leaves
it nearly unchanged. Increasing Se mostly lowers frequency and raises Vpp in
this retained region, with exceptions. These are conditional local tendencies,
not universal monotonic laws or inferred sample parameter values.

matched_neighbors.csv compares adjacent positions of the original axis only,
holding both other parameters exactly fixed. Both endpoints must pass the finer
replay checks; missing cells are never bridged. The summary gives 1% and 5%
change bins. The 1% bins describe saved values and are not precision claims:
the original convergence threshold is 5%. Counts overlap and are correlated;
they are not independent statistical trials or confidence intervals.

The displayed one-dimensional examples maximize the number of visually resolved
points at fixed other parameters (ascending-coordinate tie break). Low, middle
and high examples use their current rank. They were not selected by error size.

## What helps separate the parameters?

C and Cth both tune frequency, so frequency alone cannot distinguish them well.
Their different amplitude tendencies offer additional information. More strongly,
at deterministic constant current multiplying BOTH capacitances by s preserves
the trajectory path and Vpp while replacing f by f/s. The saved ratio-40 ray
illustrates this identity with independent pre-existing integrations. Small
deviations reflect finite-step/window extraction; no new integration was run.
Thus amplitude/loop shape constrain the ratio and thermal conductance, while
frequency constrains the common time scale, conditional on fixed gamma and T0.
This does not establish unique identifiability from the measured data.

## Limits and next useful comparison

Blank cells mean excluded or unresolved under the source screens, not proven
nonoscillation. Finite map edges are not physical bounds. The later adaptive
boundary evidence (bundle 504198) retains unresolved/calibration-censored regions
and establishes the open common-capacitance direction analytically.

The retained map has hundreds of mV of Vpp at these conditions. It cannot be
used directly as agreement with the measured sweep at adopted T0=314.4 K.
The useful next experiment is a matched-condition ratio/Se exploration at that
ambient, with gamma explicitly fixed or swept, followed by a common-scale
frequency adjustment. That is a proposal, not a completed new simulation.

The original HTML and GIF collection are preserved. This bundle reads saved
measurements only, checks byte identity with the fixed-scale viewer, and records
input hashes and its exact analysis source. It contains no reference PDFs.
'''
    report+='\n## Metrics\n\n```json\n'+json.dumps(metrics,indent=2)+'\n```\n'
    bundle.write_text('report.md',report,label='Interpretation and limitations of the parameter trends')
    bundle.complete(summary=metrics)
    print(bundle.root)
    print(summary.to_string(index=False))
    print(json.dumps(metrics,indent=2))
except Exception as exc:
    bundle.fail(exc)
    raise
