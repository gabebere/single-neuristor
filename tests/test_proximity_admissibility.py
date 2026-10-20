"""Check the diagnosed hook independently and ensure auditing cannot alter physics."""
from dataclasses import replace
import numpy as np
from neuristor.model import HysteresisArray, YuanhangResistParams
from neuristor.current_drive_sim import CurrentDriveParams, simulate_current_waveforms


def test_diagnosed_negative_slope_matches_independent_finite_difference():
    p=YuanhangResistParams(gamma=.10203429104987166)
    h=HysteresisArray(p,1)
    h.initialize(np.array([339.686767578125]))
    h.reversed[:]=1;h.Tr[:]=339.686767578125;h.Tpr[:]=-1.03155517578125
    t=338.0743408203125
    def effective(T):
        a=float(h.Tpr[0]);x=(T-float(h.Tr[0]))/(a+1e-6)
        return T+a*.5*(1-np.sin(p.gamma*x))*(1+np.tanh(np.pi**2-2*np.pi*x))
    numerical=(effective(t+1e-4)-effective(t-1e-4))/2e-4
    slope=h.proximity_temperature_slope(np.array([t]))[0]
    assert slope < -1.6
    np.testing.assert_allclose(slope,numerical,atol=1e-6)
    h.params=replace(p,gamma=.956269682)
    assert h.proximity_temperature_slope(np.array([t]))[0]>0


def test_audit_preserves_trajectories_at_three_timesteps():
    times=np.linspace(-10e-9,80e-9,91)
    currents=np.where(times[:,None]>=0, np.array([[100.,500.]]),0.)
    for dt in [.1e-9,.05e-9,.025e-9]:
        p=CurrentDriveParams(dt_s=dt,t_pre_s=10e-9,t_end_s=80e-9)
        plain=simulate_current_waveforms(currents,p,waveform_time_s=times)
        audited=simulate_current_waveforms(currents,p,waveform_time_s=times,audit_proximity=True)
        for a,b in zip(plain,audited):
            for key in a: np.testing.assert_array_equal(a[key],b[key])
            assert np.isfinite(b['minimum_proximity_slope'])
