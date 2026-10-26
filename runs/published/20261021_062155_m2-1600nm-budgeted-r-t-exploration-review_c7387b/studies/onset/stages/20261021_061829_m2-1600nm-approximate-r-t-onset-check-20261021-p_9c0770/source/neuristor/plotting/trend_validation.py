"""Display controlled simulations without filling unresolved parameter points."""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_trend_validation(summary,pairs,scaling,bundle):
    """Plot verified features and keep the two scaling protocols distinct."""
    colors=['#16749b','#bf6518','#748b27','#9550a0']
    labels={'C_pF':'C','C_th_pJ_per_K':'Cth','S_e_mW_per_K':'Se'}
    for scenario,table in summary[summary.family=='local'].groupby('scenario',sort=False):
        fig,axs=plt.subplots(2,3,figsize=(12,6),squeeze=False)
        for col,(axis,label) in enumerate(labels.items()):
            for j,(anchor,part) in enumerate(table[table.axis==axis].groupby('anchor',sort=False)):
                part=part.sort_values('factor')
                valid=(part.classification=='oscillatory')&part.timestep_verified&part.horizon_verified
                for row,metric in enumerate(['frequency_MHz','extrema_vpp_mV']):
                    axs[row,col].plot(part.factor,np.where(valid,part[metric],np.nan),'o-',label=anchor,color=colors[j%len(colors)])
            for row in range(2):
                axs[row,col].set_xlabel(f'Multiplier of {label} only');axs[row,col].grid(alpha=.2)
            axs[0,col].set_ylabel('Frequency (MHz)');axs[1,col].set_ylabel('Vpp (mV)')
            axs[0,col].legend(fontsize=8)
            part=table[table.axis==axis]
            count=int(((part.classification=='oscillatory')&part.timestep_verified&part.horizon_verified).sum())
            axs[0,col].set_title(f'{label}: {count}/{len(part)} verified oscillatory')
            if count==0:
                for row in range(2):axs[row,col].text(.5,.5,'No verified oscillations',ha='center',transform=axs[row,col].transAxes)
        condition=table.iloc[0]
        fig.suptitle(f'{scenario}: {condition.current_uA:g} µA, {condition.T0_K:g} K, gamma={condition.gamma:.5g}\nGaps retain censored, nonoscillatory or unresolved results; other parameters fixed')
        fig.tight_layout(rect=(0,0,1,.91))
        name=re.sub('[^a-zA-Z0-9_-]','_',str(scenario))
        fig.savefig(bundle.add_artifact(f'figures/local_trends_{name}.png',label=f'Controlled trends at {scenario}'),dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,7),squeeze=False)
    metrics=['frequency_scaling_error','amplitude_scaling_error','voltage_path_normalized_rmse','temperature_path_rmse_K']
    for j,((anchor,protocol),part) in enumerate(scaling.groupby(['anchor','protocol'],sort=False)):
        for ax,metric in zip(axs.flat,metrics):
            numeric=part.assign(value=pd.to_numeric(part[metric],errors='coerce'))
            data=numeric.groupby('nominal_dt_ns').value.agg(lambda x:x.abs().max()).sort_index()
            multiplier=1 if metric=='temperature_path_rmse_K' else 100
            ax.plot(data.index,data*multiplier,'o',linestyle='--' if protocol=='fixed_dt' else '-',color=colors[j//2%len(colors)],label=f'{anchor}, {protocol}')
    titles=['Maximum |f(s) s / f(1) - 1| (%)','Maximum |Vpp(s) / Vpp(1) - 1| (%)',
            'Unaligned voltage path RMSE (% of reference span)','Unaligned temperature path RMSE (K)']
    for ax,title in zip(axs.flat,titles):
        ax.set_title(title,fontsize=10);ax.set_xscale('log');ax.set_xlabel('Nominal dt (ns)');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7)
    fig.suptitle('Common-capacitance scaling: new integrations\nMaximum across prescribed scale factors and common observation horizons')
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(bundle.add_artifact('figures/scaling_validation.png',label='Independent scaling and timestep checks'),dpi=170);plt.close(fig)
    gamma=summary[summary.family=='gamma']
    fig,axs=plt.subplots(1,2,figsize=(10,3.5))
    for j,(anchor,part) in enumerate(gamma.groupby('anchor',sort=False)):
        part=part.sort_values('gamma');valid=(part.classification=='oscillatory')&part.timestep_verified&part.horizon_verified
        for ax,metric in zip(axs,['frequency_MHz','extrema_vpp_mV']):
            ax.plot(part.gamma,np.where(valid,part[metric],np.nan),'o-',color=colors[j%len(colors)],label=anchor)
    for ax,label in zip(axs,['Frequency (MHz)','Vpp (mV)']):
        ax.set_xlabel('Gamma');ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Gamma varies; C, Cth, Se, current and ambient stay fixed per anchor')
    fig.tight_layout();fig.savefig(bundle.add_artifact('figures/gamma_validation.png',label='Controlled coupled gamma tests'),dpi=170);plt.close(fig)
