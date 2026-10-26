"""Refine startup-temperature exclusions; no steady-state or boundary claims."""
from pathlib import Path
import dataclasses,json,itertools,time
import numpy as np
import pandas as pd
from neuristor.current_drive_sim import CurrentDriveParams,simulate_current_parameter_grid
from neuristor.model import YuanhangResistParams
root=Path(__file__).resolve().parents[2]
prior=root/'public_jobs/20261021_062155_m2-1600nm-budgeted-r-t-exploration-review_c7387b'
onset=prior/'studies/onset'; stages=json.loads((onset/'stages.json').read_text())
old=pd.read_csv(onset/stages['grid']/'grid.csv')
new=root/'runs/20261021_063603_m2-1600nm-approximate-r-t-dense-onset-20261021_57e853/screening_checkpoint.csv'
data=pd.concat([old,pd.read_csv(new)],ignore_index=True)
axes=['C_th_pJ_per_K','S_e_mW_per_K','C_pF']
chosen=data[data.sustained].sort_values('temperature_max_K').drop_duplicates(axes).head(12).copy()
physics=json.loads((onset/stages['grid']/'physics_parameters.json').read_text())
settings=dict(duration_ns=150.0,dt_ns=[.00078125,.000390625],current_uA=500.0,temperature_maximum_K=370.1795,selection='12 lowest-Tmax sustained entries in prior onset and expanded checkpoint',checkpoint_points=len(pd.read_csv(new)),points=chosen[axes+['temperature_max_K']].to_dict('records'),source_physics=physics)
(root/'runs/m2_1600nm_diagnostics/near_boundary_config.json').write_text(json.dumps(settings,indent=2))
physics['resist_params']=YuanhangResistParams(**physics['resist_params']);params=CurrentDriveParams(**physics)
rows=[]
for dt in settings['dt_ns']:
 p=dataclasses.replace(params,dt_s=dt*1e-9,t_end_s=settings['duration_ns']*1e-9)
 out=simulate_current_parameter_grid(500,p,C_F=chosen.C_pF.to_numpy()*1e-12,C_th_J_per_K=chosen.C_th_pJ_per_K.to_numpy()*1e-12,S_e_W_per_K=chosen.S_e_mW_per_K.to_numpy()*1e-3)
 for (_,original),o in zip(chosen.iterrows(),out):
  rows.append({**{a:float(original[a]) for a in axes},'dt_ns':dt,'duration_ns':150,'coarse_maximum_temperature_K':original.temperature_max_K,'maximum_temperature_K':float(np.max(o['T'])),'excluded_in_startup':bool(np.max(o['T'])>370.1795)})
 pd.DataFrame(rows).to_csv(root/'runs/m2_1600nm_diagnostics/near_boundary_checks.csv',index=False)
 print(pd.DataFrame(rows).tail(12).to_string(index=False),flush=True)
