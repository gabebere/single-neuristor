"""Direct f_i(theta) profiles from saved simulations, using measured current names."""
import matplotlib.pyplot as plt


def plot_frequency_profiles(points,path):
    """Show log-capacitance perturbations for the current-ranked middle input."""
    currents=sorted(points.measured_current_A.unique()); current=currents[len(currents)//2]
    modes=list(points.center_name.unique())
    fig,axes=plt.subplots(1,len(modes),figsize=(11,3.8),squeeze=False,constrained_layout=True)
    for ax,mode in zip(axes[0],modes):
        part=points[(points.center_name==mode)&(points.measured_current_A==current)]
        for dt,frame in part.groupby('dt_ns'):
            shown=frame[frame.coordinate.isin(['center','C_pF'])].sort_values('offset')
            ax.plot(shown.offset,shown.predicted_frequency_MHz,'o-',label=f'{dt:g} ns',markersize=4)
        ax.axhline(part.measured_frequency_MHz.iloc[0],color='black',linestyle='--',label='Measured')
        ax.set(title=f'{mode.replace("_"," ")} · {current*1e6:.1f} µA',
               xlabel='Normalized log-capacitance increment',ylabel='Extracted frequency (MHz)')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.savefig(path,dpi=170);plt.close(fig)
