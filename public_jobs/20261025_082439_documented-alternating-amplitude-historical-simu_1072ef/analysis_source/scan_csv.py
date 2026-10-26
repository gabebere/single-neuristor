from pathlib import Path
import numpy as np,pandas as pd
from scan_saved import cycles,describe,ROOT,OUT
rows=[]
for p in (ROOT/'public_jobs').glob('*/*traces*.csv'):
 if p.stat().st_size>80_000_000:continue
 header=pd.read_csv(p,nrows=0).columns
 time='time_ns' if 'time_ns' in header else 'time_us' if 'time_us' in header else None
 if time is None:continue
 signals=[c for c in ['predicted_mV','predicted_voltage_mV','V_vo2_mV','voltage_V','measured_mV','measured_voltage_mV','output_voltage_mV'] if c in header]
 if not signals:continue
 groups=[c for c in ['candidate','dt_ns','source_mV','source_label_mV','fit_mode','split','source_file','nominal_drive_mV','I_target_uA','current_uA'] if c in header]
 d=pd.read_csv(p,usecols=[time,*groups,*signals])
 for keys,g in d.groupby(groups,dropna=False) if groups else [((),d)]:
  g=g.sort_values(time);t=g[time].to_numpy()*(1000 if time=='time_us' else 1)
  if np.any(np.diff(t)<=0):continue
  windows=[(50,280),(100,280),(150,280)]
  if t.max()>450:windows.extend([(300,600),(.5*t.max(),t.max()),(.25*t.max(),.75*t.max())])
  for col in signals:
   y=g[col].to_numpy()*(1000 if col=='voltage_V' else 1)
   for lo,hi in windows:
    if hi>t.max()+1e-3:continue
    c=cycles(t,y,(lo,hi))
    if len(c)>=6:
     rows.append(dict(source=p.relative_to(ROOT).as_posix(),group=str(keys),signal=col,window_start_ns=lo,window_stop_ns=hi,**describe(c)))
frame=pd.DataFrame(rows).sort_values('alternation_score',ascending=False)
frame.to_csv(OUT/'csv_candidates.csv',index=False)
print('CSV CANDIDATES');print(frame.head(35).to_string(index=False))
