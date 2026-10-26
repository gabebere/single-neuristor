"""Fixed-parameter diagnostic of an archived alternating-amplitude waveform."""
from pathlib import Path
import dataclasses,json,ast,time
import numpy as np,pandas as pd
from scipy.signal import find_peaks,savgol_filter
from neuristor.config import load_toml
from neuristor.workflows import _current_params_from_config
from neuristor.fitting.parameter_inference import model_from_parameter_vector,PARAMETER_NAMES
from neuristor.current_drive_sim import simulate_current_waveforms,simulate_current_steps
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/double_oscillation_20261025'
source=ROOT/'public_jobs/20260907_125315_physics-anchored-specimen-waveform-inference_717797'
table=pd.read_csv(source/'parameter_comparison.csv').set_index('parameter')
vector=table.loc[list(PARAMETER_NAMES),'relaxed_fit'].to_numpy()
base,*_=_current_params_from_config(load_toml(ROOT/'experiments/current/specimen_physics_anchored_inference.toml'))
data=pd.read_csv(source/'fitted_traces.csv');labels=[350,400,450]
groups=[data[(data.fit_mode=='relaxed')&(data.nominal_drive_mV==k)] for k in labels]
t=groups[0].time_ns.to_numpy();currents=np.column_stack([g.measured_current_uA for g in groups])
medians=[float(g.measured_current_uA[(g.time_ns>=150)&(g.time_ns<=250)].median()) for g in groups]
functions=ast.parse((OUT/'scan_saved.py').read_text());exec(compile(ast.Module(body=[n for n in functions.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'<functions>','exec'))
rows=[]
for dt in [.025,.0125,.00625]:
 p=model_from_parameter_vector(vector,base,dt_ns=dt)
 p=dataclasses.replace(p,t_pre_s=-t[0]*1e-9,t_end_s=t[-1]*1e-9)
 for mode,outputs in [('measured_current',simulate_current_waveforms(currents,p,waveform_time_s=t*1e-9,audit_proximity=True)),('constant_current',simulate_current_steps(medians,dataclasses.replace(p,t_pre_s=0,t_end_s=2e-6,pulse_off_s=None)))]:
  for label,g,o in zip(labels,groups,outputs):
   time_ns=o['t'].astype(float)*1e9;v=o['V_vo2'].astype(float)*1000
   window=[50,280] if mode=='measured_current' else [1000,2000]
   c=cycles(time_ns,v,window)
   result=dict(label_mV=label,input_mode=mode,dt_ns=dt,Tmin_K=float(o['T'].min()),Tmax_K=float(o['T'].max()),**(describe(c) if len(c)>=6 else dict(cycles=len(c))))
   if mode=='measured_current':result['saved_max_difference_mV']=float(np.max(np.abs(np.interp(t,time_ns,v)-g.predicted_voltage_mV)))
   rows.append(result);c.to_csv(OUT/f'cycles_{mode}_{label}_{dt:g}.csv',index=False)
   np.savez_compressed(OUT/f'waveform_{mode}_{label}_{dt:g}.npz',time_ns=time_ns,voltage_mV=v,temperature_K=o['T'],resistance_ohm=o['R'],input_current_uA=o['I_in']*1e6)
   print(result,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'candidate_replay_summary.csv',index=False)
