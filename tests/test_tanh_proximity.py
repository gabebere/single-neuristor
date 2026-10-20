from dataclasses import replace
import numpy as np
import pytest
from neuristor.model import HysteresisArray,YuanhangResistParams
from neuristor.steady_inference import settled_features,settled_loss


def test_tanh_kernel_monotone_and_major_branches_unchanged():
    for k in [.03,.5,.98]:
        p=YuanhangResistParams(proximity_function='tanh',gamma=k)
        h=HysteresisArray(p,2001);t=np.linspace(325,345,2001,dtype=np.float32)
        h.initialize(t)
        ref=HysteresisArray(replace(p,proximity_function='yuanhang'),len(t));ref.initialize(t)
        np.testing.assert_array_equal(h.g(t),ref.g(t))
        h.reversed[:]=1;h.Tr[:]=345;h.Tpr[:]=-6
        assert np.min(h.proximity_temperature_slope(t))>0
    with pytest.raises(ValueError):YuanhangResistParams(proximity_function='tanh',gamma=1.2)


def test_tanh_reversal_path_refines():
    """Nested heating/cooling paths converge under two step halvings."""
    results=[]
    for step in [.02,.01,.005]:
        h=HysteresisArray(YuanhangResistParams(proximity_function='tanh',gamma=.5),1)
        h.initialize(np.array([325.]))
        values=[]
        for start,end in [(325,345),(345,331),(331,340),(340,325)]:
            for t in np.linspace(start,end,round(abs(end-start)/step)+1)[1:]:
                r,_=h.evaluate(np.array([t]));values.append(float(r[0]))
        results.append(np.array(values))
    e1=np.max(np.abs(results[0]-results[1][1::2]))
    e2=np.max(np.abs(results[1]-results[2][1::2]))
    assert e2<e1
    assert e2/np.max(results[2])<.01


def test_sustained_gate_rejects_ringdown():
    t=np.arange(-200.,301.);wave=20*np.sin(2*np.pi*.05*t)
    m=settled_features(t,200+wave)
    settings=dict(frequency_tolerance_fraction=.025,amplitude_tolerance_fraction=.075,require_sustained=True)
    good=settled_loss(m,m,settings)
    bad=settled_loss(m,settled_features(t,200+wave*np.exp(-np.maximum(t-150,0)/40)),settings)
    assert good['sustained'] and good['near_target']
    assert not bad['sustained'] and not bad['frequency_valid'] and bad['loss']>=10000
