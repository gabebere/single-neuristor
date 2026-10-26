"""Plots of saved parameter perturbations with raw scientific eligibility visible."""
import matplotlib.pyplot as plt

def plot_profiles(samples,path):
    """Show finite optimizer scores; explicitly avoid labeling them accepted losses."""
    names=list(samples.center_name.unique())
    coordinates=[n for n in samples.coordinate.unique() if n!='center']
    fig,axes=plt.subplots(len(names),4,figsize=(14,6),squeeze=False,constrained_layout=True)
    for i,name in enumerate(names):
        block=samples[samples.center_name==name]
        for j,coordinate in enumerate(coordinates):
            ax=axes[i,j]
            for dt,group in block.groupby('dt_ns'):
                center=group[group.coordinate=='center']
                p=group[group.coordinate==coordinate].copy()
                p['offset']=p.step*p.direction
                # Score relative to its own center at that timestep exposes
                # local steps without implying timestep-independent losses.
                baseline=center.score.iloc[0]
                ax.plot([*p.sort_values('offset').offset[:len(p)//2],0.,*p.sort_values('offset').offset[len(p)//2:]],
                    [*p.sort_values('offset').score[:len(p)//2]-baseline,0.,*p.sort_values('offset').score[len(p)//2:]-baseline],
                    '.-',label=f'{dt:g} ns')
            ax.set(title=f'{name.replace("_"," ")}\n{coordinate}',xlabel='Normalized coordinate increment',
                   ylabel='Change in finite search score',yscale='symlog')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(path,dpi=160);plt.close(fig)
