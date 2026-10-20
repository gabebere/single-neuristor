"""Point identity, honest R(T) rendering, and offline archive contract."""
from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess
import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.config import load_toml
from neuristor.current_drive_sim import CurrentDriveParams
from neuristor.map_animations import resolved_rt_path, animation_controls_html, replay_animation_batch
from neuristor.physical_maps import AXES, slice_heatmap_values
from neuristor.runs import RunBundle

ROOT=Path(__file__).resolve().parents[1]


def test_rt_resolution_preserves_every_endpoint_and_breaks_instead_of_smoothing():
    t=np.arange(4.)
    temp=np.array([330.,330.01,335.,335.01])
    r=np.array([1000.,995.,20.,19.9])
    time,x,y,count=resolved_rt_path(t,temp,r)
    assert count == 1
    assert np.isnan(x[2]) and np.isnan(y[2])
    np.testing.assert_array_equal(x[np.isfinite(x)],temp)
    np.testing.assert_array_equal(y[np.isfinite(y)],r)
    np.testing.assert_array_equal(time[np.isfinite(x)],t)


def test_heatmap_point_identity_tracks_each_slice_and_excludes_blank_cells():
    import itertools
    axes=[np.array([1.,2.])]*3
    table=pd.DataFrame([dict(zip(AXES,v),point_index=i,included=i!=7,
        frequency_MHz=50.,extrema_vpp_mV=40.)
        for i,v in enumerate(itertools.product(*axes))])
    a=slice_heatmap_values(table,axes,AXES[0],1.)
    b=slice_heatmap_values(table,axes,AXES[0],2.)
    assert set(a['point_ids'].ravel()) == {0,1,2,3}
    assert set(b['point_ids'].ravel()) == {4,5,6,-1}


def test_offline_controls_use_relative_gifs_no_fetch_and_clear_stale_slice():
    html=animation_controls_html({'1':dict(gif='GIFs/point_00001.gif',caption='</script>',status='pass',detail='checked')})
    assert 'fetch(' not in html and 'http://' not in html and 'https://' not in html
    assert "plotly_sliderchange" in html and 'hovered=null;selected=null' in html
    assert 'contextmenu' in html and 'plotly_click' in html
    assert '\\u003c/script>' in html and 'CATALOG' not in html


def test_point_controls_runtime_click_right_click_slider_and_close():
    node=shutil.which('node')
    if node is None:
        bundled=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
        node=str(bundled) if bundled.exists() else None
    if node is None: pytest.skip('Node unavailable for DOM event logic check')
    html=animation_controls_html({'1':dict(gif='GIFs/point_00001.gif',caption='point one',status='pass',detail='checked')})
    script=re.search(r'<script>(.*?)</script>',html,re.S).group(1)
    harness='''
const vm=require('vm'),assert=require('assert');
const elements={};
function element(id){return elements[id]??={id,style:{},handlers:{},listeners:{},disabled:false,open:false,
 on(n,f){this.handlers[n]=f;},addEventListener(n,f){this.listeners[n]=f;},contains(e){return e===this;},
 showModal(){this.open=true;},close(){this.open=false;}};}
const document={getElementById:element,addEventListener(){}};
vm.runInNewContext(SCRIPT,{document,window:{innerWidth:1000,innerHeight:800}});
const plot=element('map-3d');
plot.handlers.plotly_click({points:[{customdata:[0,0,0,0,0,0,1]}]});
assert.equal(element('point-open').disabled,false);
element('point-open').listeners.click();
assert.equal(element('gif-image').src,'GIFs/point_00001.gif');
assert.equal(element('gif-dialog').open,true);
element('gif-close').listeners.click();assert.equal(element('gif-dialog').open,false);
plot.handlers.plotly_hover({points:[{customdata:[0,0,0,0,0,0,1]}]});
let prevented=false;plot.listeners.contextmenu({clientX:400,clientY:200,preventDefault(){prevented=true;}});
assert(prevented);assert.equal(element('point-menu').style.display,'block');
plot.handlers.plotly_sliderchange();assert.equal(element('point-open').disabled,true);
plot.handlers.plotly_click({points:[{customdata:new Float64Array([0,0,0,0,0,0,1])}]});
assert.equal(element('point-open').disabled,false);
plot.handlers.plotly_click({points:[{pointNumber:0,data:{customdata:[[0,0,0,0,0,0,1]]}}]});
assert.equal(element('point-open').disabled,false);
const heat=element('map-slice-0');heat.handlers.plotly_click({points:[{customdata:1}]});
assert.equal(element('point-open').disabled,false);
heat.handlers.plotly_click({points:[{pointNumber:[0,0],data:{customdata:[[1,-1]]}}]});
assert.equal(element('point-open').disabled,false);
heat.handlers.plotly_sliderchange();assert.equal(element('point-open').disabled,true);
heat.handlers.plotly_click({points:[{customdata:-1}]});assert.equal(element('point-open').disabled,true);
'''.replace('SCRIPT',json.dumps(script))
    result=subprocess.run([node,'-e',harness],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_refinement_halves_dt_when_visual_jumps_remain(monkeypatch):
    import neuristor.map_animations as module
    steps=[]
    def simulate(current,params,**columns):
        steps.append(params.dt_s*1e9)
        d=.5 if len(steps)==1 else .1
        return [dict(t=np.arange(12)*1e-9,T=330+np.arange(12)*d,R=np.full(12,100.))]
    monkeypatch.setattr(module,'simulate_current_parameter_grid',simulate)
    monkeypatch.setattr(module,'screen_trajectory',lambda *args:dict(frequency_MHz=50.,extrema_vpp_mV=40.,eligible=True))
    points=pd.DataFrame([dict(point_index=9,C_th_pJ_per_K=.05,S_e_mW_per_K=.004,C_pF=5.,frequency_MHz=50.,extrema_vpp_mV=40.)])
    settings=dict(dt_ns=.00625,maximum_refinements=2)
    out=replay_animation_batch(points,CurrentDriveParams(),500,
        dict(window_ns=[0,11],convergence_fraction=.05),settings)
    np.testing.assert_allclose(steps,[.00625,.003125],rtol=1e-14,atol=0)
    assert out[0][2]['visually_resolved'] and out[0][2]['converged']


def test_map_animation_cli_portable_zip_and_original_map_preserved(tmp_path):
    original=dict(schema_version=1,name='tiny map',model='current',kind='simulation',
        input=dict(amplitude_uA=50.),maps=dict(window_ns=[10.,20.],maximum_frequency_MHz=110.,
            verification_dt_ns=[.002,.001],convergence_fraction=.05,
            fixed_values=dict(zip(AXES,[1.,.004,5.])),
            axes={name:dict(values=[value]) for name,value in zip(AXES,[1.,.004,5.])}))
    source=RunBundle.create(name='tiny map',model='current',kind='analysis',config=original,output_root=tmp_path/'source',command='test map')
    params=CurrentDriveParams(t_end_s=20e-9,T0_K=325,T_init_K=325)
    source.write_json('physics_parameters.json',asdict(params),label='test physics')
    grid=pd.DataFrame([dict(point_index=0,C_th_pJ_per_K=1.,S_e_mW_per_K=.004,C_pF=5.,
        included=True,frequency_MHz=50.,extrema_vpp_mV=40.,temperature_max_K=335.)])
    grid.to_csv(source.add_artifact('grid.csv',label='test grid'),index=False)
    grid.to_csv(source.add_artifact('verification.csv',label='test verification'),index=False)
    source.write_json('axis_values.json',{n:[v] for n,v in zip(AXES,[1.,.004,5.])},label='test axes')
    source.complete(summary={})
    before=(source.root/'grid.csv').read_bytes()
    result=CliRunner().invoke(app,['analyze','map-animations','--config',str(ROOT/'experiments/current/physical_parameter_map_gifs.toml'),
        '--set',f'animations.source={source.root}','--set','animations.dt_ns=0.0005',
        '--set','animations.maximum_refinements=0','--set','animations.frames=2',
        '--set','animations.workers=1','--limit','1','--output-root',str(tmp_path/'result')])
    assert result.exit_code==0, result.output+repr(result.exception)
    root=next((tmp_path/'result').glob('*/run.json')).parent
    assert (root/'grid.csv').read_bytes()==before==(source.root/'grid.csv').read_bytes()
    assert json.loads((root/'metrics.json').read_text())['gifs']==1
    html=(root/'parameter_maps.html').read_text()
    assert 'contextmenu' in html and 'map-slice-2' in html
    decoder=json.JSONDecoder()
    for match in re.finditer(r'Plotly.newPlot\(\s*"(map-[^"]+)"\s*,\s*',html):
        traces,_=decoder.raw_decode(html[match.end():])
        assert isinstance(traces[0]['customdata'],list), 'Point IDs must not use binary typed-array encoding'
    with zipfile.ZipFile(root/'VO2_parameter_map_with_GIFs.zip') as archive:
        assert 'parameter_maps.html' in archive.namelist()
        assert 'GIFs/point_00000.gif' in archive.namelist()
        assert b'GIF8' in archive.read('GIFs/point_00000.gif')[:6]
    reused=CliRunner().invoke(app,['analyze','map-animations','--config',str(ROOT/'experiments/current/physical_parameter_map_gifs.toml'),
        '--set',f'animations.source={source.root}','--set','animations.dt_ns=0.0005',
        '--set','animations.maximum_refinements=0','--set','animations.frames=2',
        '--set','animations.workers=1','--limit','1','--reuse-media',str(root),
        '--output-root',str(tmp_path/'reused')])
    assert reused.exit_code==0,reused.output+repr(reused.exception)
    rebuilt=next((tmp_path/'reused').glob('*/run.json')).parent
    assert not json.loads((rebuilt/'metrics.json').read_text())['resimulated']
    assert (rebuilt/'GIFs/point_00000.gif').read_bytes()==(root/'GIFs/point_00000.gif').read_bytes()
    assert (rebuilt/'original_simulation_source/neuristor/map_animations.py').exists()
