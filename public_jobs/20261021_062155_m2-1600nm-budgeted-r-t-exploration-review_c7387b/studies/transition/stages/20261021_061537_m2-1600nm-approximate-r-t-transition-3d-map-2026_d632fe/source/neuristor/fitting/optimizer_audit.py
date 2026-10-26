"""Cached normalized-coordinate evaluations and bounded gradient score trials."""
from __future__ import annotations

from time import monotonic

import numpy as np
from scipy.optimize import minimize

from neuristor.fitting.joint_inference import NAMES, decode, encode
from neuristor.fitting.steady_inference import StageExpired, evaluate_settled, relative_candidate_score

ACTIVE = np.array([0,1,2,7])
COORDINATES = [NAMES[i] for i in ACTIVE]


class CachedEvaluator:
    """Evaluate the identical relative objective, gates and four search coordinates."""
    def __init__(self,dataset,base,settings,center_values,callback=lambda record:None):
        self.dataset,self.base,self.settings=dataset,base,settings
        self.bounds=np.asarray([settings['bounds'][n] for n in NAMES],float)
        self.full_unit=encode(center_values,self.bounds)
        self.cache={}
        self.callback=callback
        self.started=monotonic()

    def values(self,unit):
        full=self.full_unit.copy();full[ACTIVE]=np.asarray(unit,float)
        return decode(full,self.bounds)

    def evaluate(self,unit,dt_ns):
        unit=np.asarray(unit,float)
        if unit.shape!=(4,) or not np.isfinite(unit).all() or np.any(unit<0) or np.any(unit>1):
            raise ValueError('Expected four normalized parameters inside [0,1]')
        key=(float(dt_ns),*np.round(unit,12))
        if key not in self.cache:
            values=self.values(unit)
            rows,_,outside=evaluate_settled(self.dataset,values,self.base,self.dataset.train_indices,self.settings,dt_ns)
            result=relative_candidate_score(rows,outside)
            ferr=[abs(r['frequency_relative_error']) for r in rows if r['frequency_relative_error'] is not None]
            result.update(unit=unit.tolist(),values=values.tolist(),dt_ns=float(dt_ns),
                outside_domain=bool(outside),per_current=rows,
                defined_frequency_count=len(ferr),sustained_count=sum(r['sustained'] for r in rows),
                frequency_MAPE_defined=float(np.mean(ferr)) if ferr else None,
                amplitude_MAPE=float(np.mean([abs(r['robust_amplitude_relative_error']) for r in rows])),
                elapsed_s=monotonic()-self.started)
            self.cache[key]=result
            self.callback(result)
        return self.cache[key]


def lbfgsb_score_trial(evaluator,start,*,dt_ns,step,max_evaluations=60,max_iterations=6,max_line_search=4):
    """Test finite-difference L-BFGS-B on exactly the same finite search score.

    For infeasible starting points this is explicitly a gradient of a rejection
    score, not of an undefined all-target loss. Both the best sampled candidate
    and optimizer termination are returned, so a plateau is not called a fit.
    """
    counter=0
    scientific_requests=0
    initial=evaluator.evaluate(start,dt_ns)
    best=initial
    def objective(unit):
        nonlocal counter,best,scientific_requests
        if counter>=max_evaluations:
            raise StageExpired()
        counter+=1
        record=evaluator.evaluate(unit,dt_ns)
        scientific_requests+=int(record['eligible'])
        if record['score']<best['score']:
            best=record
        return record['score']
    result=None
    try:
        result=minimize(objective,np.asarray(start,float),method='L-BFGS-B',bounds=[(0.,1.)]*4,
            options=dict(eps=float(step),maxfun=int(max_evaluations),maxiter=int(max_iterations),
                         maxls=int(max_line_search),ftol=1e-10,gtol=1e-5))
    except StageExpired:
        pass
    return dict(step=float(step),dt_ns=float(dt_ns),objective_requests=counter,
        scientific_requests=scientific_requests,infeasible_requests=counter-scientific_requests,
        starting_score=initial['score'],best_score=best['score'],best_unit=best['unit'],
        values=best['values'],eligible=best['eligible'],invalid_count=best['invalid_count'],
        scientific_start_defined=initial['eligible'],
        optimizer_success=bool(result.success) if result is not None else False,
        termination=str(result.message) if result is not None else 'Hard evaluation cap reached',
        best=best)
