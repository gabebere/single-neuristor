"""Read-only candidate search; peak amplitudes are not averaged across neighbors."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import find_peaks,savgol_filter
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/double_oscillation_20261025'

def cycles(t,y,window):
 t,y=np.asarray(t,float),np.asarray(y,float)
 mask=(t>=window[0])&(t<=window[1])&np.isfinite(t)&np.isfinite(y);t,y=t[mask],y[mask]
 if len(t)<15 or np.any(np.diff(t)<=0): return pd.DataFrame()
 dt=float(np.median(np.diff(t)));width=max(5,int(round(.5/dt))|1);width=min(width,len(t)//2*2-1)
 z=savgol_filter(y,width,2);peaks,_=find_peaks(z,prominence=max(.5,.03*np.ptp(z)),distance=max(1,round(2/dt)))
 rows=[]
 for i in range(1,len(peaks)):
  left,right=peaks[i-1:i+1];bottom=left+np.argmin(z[left:right+1])
  amp=y[right]-y[bottom]
  if amp<=0:continue
  rows.append(dict(time_ns=t[right],peak_mV=y[right],trough_mV=y[bottom],vpp_mV=amp,period_ns=t[right]-t[left]))
 return pd.DataFrame(rows)

def describe(c):
 a=c.vpp_mV.to_numpy();m=np.median(a)
 lag1=np.corrcoef(a[:-1],a[1:])[0,1] if np.std(a)>1e-6 else 0
 lag2=np.corrcoef(a[:-2],a[2:])[0,1] if np.std(a)>1e-6 else 0
 odd,even=a[::2],a[1::2]
 separation=abs(np.median(odd)-np.median(even))/max(m,.01)
 spread=(np.percentile(a,90)-np.percentile(a,10))/max(m,.01)
 parity_mad=(np.median(np.abs(odd-np.median(odd)))+np.median(np.abs(even-np.median(even))))/max(m,.01)
 return dict(cycles=len(c),median_vpp_mV=m,spread_fraction=spread,parity_separation=separation,parity_mad=parity_mad,lag1=lag1,lag2=lag2,alternation_score=separation*max(-lag1,0)*max(lag2,0)/(1+10*parity_mad),period_cv=float(c.period_ns.std()/c.period_ns.mean()))

rows=[]
source=ROOT/'public_jobs/20261021_054434_loss-function-comparison-all-44-stacked-gifs_15921f'
for p in source.glob('*/*.npz'):
 d=np.load(p)
 for kind,tk,yk in [('measured','acquisition_time_ns','measured_voltage_mV'),('simulated','time_ns','simulated_voltage_mV')]:
  t,y=d[tk],d[yk]
  for lo,hi in [(100,300),(150,250),(200,400),(300,550),(150,550)]:
   c=cycles(t,y,(lo,hi))
   if len(c)>=8:rows.append(dict(source=p.relative_to(ROOT).as_posix(),kind=kind,window_start_ns=lo,window_stop_ns=hi,**describe(c)))
frame=pd.DataFrame(rows).sort_values('alternation_score',ascending=False)
frame.to_csv(OUT/'saved_trace_candidates.csv',index=False)
print(frame.head(22).to_string(index=False))
print('MEASURED:');print(frame[frame.kind=='measured'].head(15).to_string(index=False))
