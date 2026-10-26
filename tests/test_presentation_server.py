"""Exercise real HTTP links against synthetic bundles, including GIF dependencies."""
import json
from pathlib import Path
import subprocess
import sys
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from typer.testing import CliRunner

from neuristor.cli import app
from neuristor.interfaces.presentation_server import BUNDLES, SERVICE, VIEWERS, create_presentation_server


def test_viewer_routes_media_and_read_only_boundary(tmp_path):
    for bundle in BUNDLES:
        root = tmp_path/'runs/published'/bundle
        root.mkdir(parents=True)
        (root/'START_HERE.html').write_text('<h1>Test loss viewer</h1>')
        (root/'parameter_maps.html').write_text('<h1>Test heatmap viewer</h1>')
        (root/'GIFs').mkdir()
        (root/'GIFs'/'point.gif').write_bytes(b'GIF89a')
        (root/'figures').mkdir()
        (root/'figures'/'point.png').write_bytes(b'PNG test')
    private = tmp_path/'private'
    private.mkdir()
    (private/'paper.pdf').write_bytes(b'not for serving')
    (tmp_path/'runs/published'/BUNDLES[0]/'leak.pdf').symlink_to(private/'paper.pdf')
    with create_presentation_server(tmp_path, port=0) as server:
        thread=Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            assert json.load(urlopen(base+'/health')) == {'service':SERVICE, 'project':str(tmp_path)}
            for route, target in VIEWERS.items():
                with urlopen(base+route) as response:
                    assert response.url==base+target
                    assert response.headers['Content-Type'].startswith('text/html')
                    assert b'Test' in response.read()
            path=f'/public_jobs/{BUNDLES[2]}/GIFs/point.gif'
            with urlopen(base+path) as response:
                assert response.read()==b'GIF89a'
            with urlopen(base+f'/public_jobs/{BUNDLES[2]}/Figures/point.png') as response:
                assert response.read()==b'PNG test'
            for route in ['/private/paper.pdf','/public_jobs/'+BUNDLES[0]+'/leak.pdf',
                          '/public_jobs/'+BUNDLES[0]+'/../../private/paper.pdf']:
                with pytest.raises(HTTPError) as exc:
                    urlopen(base+route)
                assert exc.value.code==403
            with pytest.raises(HTTPError) as exc:
                urlopen(Request(base+'/health', headers={'Host':'other.example'}))
            assert exc.value.code==403
            with pytest.raises(HTTPError) as exc:
                urlopen(Request(base+path, data=b'overwrite', method='POST'))
            assert exc.value.code==501
        finally:
            server.shutdown()
            thread.join()


def test_presentation_server_cli_background(tmp_path, monkeypatch):
    from neuristor.interfaces import presentation_server
    seen=[]
    monkeypatch.setattr(presentation_server,'ensure_running',lambda:seen.append(True))
    result=CliRunner().invoke(app,['analyze','serve-presentation','--background'])
    assert result.exit_code==0,result.output
    assert seen and 'http://127.0.0.1:8767' in result.output


def test_mica_alias_advances_only_to_completed_reviews(tmp_path):
    def review(name, status):
        root=tmp_path/'runs/published'/name
        root.mkdir(parents=True)
        (root/'run.json').write_text(json.dumps({'status':status}))
        (root/'parameter_maps.html').write_text(name)
        return root
    first=review('20261021_062155_m2-1600nm-budgeted-review_first','completed')
    newer=review('20261021_080000_m2-1600nm-expanded-review_newer','running')
    with create_presentation_server(tmp_path,port=0) as server:
        thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            address=f'http://127.0.0.1:{server.server_port}/mica-map'
            with urlopen(address) as response:
                assert response.read().decode()==first.name
            (newer/'run.json').write_text(json.dumps({'status':'completed'}))
            with urlopen(address) as response:
                assert response.read().decode()==newer.name
        finally:
            server.shutdown();thread.join()


def test_portable_script_runs_without_installation(tmp_path):
    """A plain Python interpreter can start the service from a copied source tree."""
    from neuristor.interfaces import presentation_server
    script=Path(presentation_server.__file__).resolve()
    process=subprocess.Popen([sys.executable,'-I',str(script),'--project',str(tmp_path),'--port','0'],
        cwd=tmp_path,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        address=process.stdout.readline().strip().removeprefix('Presentation viewers: ')
        assert address.startswith('http://127.0.0.1:'),address
        with urlopen(address+'/health',timeout=3) as response:
            assert json.load(response)['project']==str(tmp_path)
    finally:
        process.terminate();process.communicate(timeout=5)


def test_alternating_gallery_requires_completion_and_serves_its_gifs(tmp_path):
    root=tmp_path/'runs/published'/'20261025_fixed-alternating-amplitude-classic-gifs_test'
    root.mkdir(parents=True)
    (root/'run.json').write_text(json.dumps({'status':'running'}))
    (root/'START_HERE.html').write_text('alternating gallery')
    (root/'current.gif').write_bytes(b'GIF89a')
    with create_presentation_server(tmp_path,port=0) as server:
        thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with pytest.raises(HTTPError):
                urlopen(base+'/alternating-gifs')
            (root/'run.json').write_text(json.dumps({'status':'completed'}))
            with urlopen(base+'/alternating-gifs') as response:
                assert response.read()==b'alternating gallery'
            with urlopen(base+f'/public_jobs/{root.name}/current.gif') as response:
                assert response.read()==b'GIF89a'
            private=tmp_path/'private';private.mkdir()
            (private/'secret.txt').write_text('private')
            (root/'leak.txt').symlink_to(private/'secret.txt')
            with pytest.raises(HTTPError) as exc:
                urlopen(base+f'/public_jobs/{root.name}/leak.txt')
            assert exc.value.code==403
        finally:
            server.shutdown();thread.join()


def test_present_checks_assets_starts_viewers_before_opening_pdf(tmp_path, monkeypatch):
    from neuristor.interfaces import presentation_server as ps
    events=[]
    monkeypatch.setattr(ps,'ensure_running',lambda *a,**k:events.append('service'))
    monkeypatch.setattr(ps,'open_presentation_pdf',lambda path:events.append(path))
    with pytest.raises(FileNotFoundError,match='PDF alone is insufficient'):
        ps.present(tmp_path)
    assert not events
    pdf=tmp_path/'docs/presentation'/'main.pdf';pdf.parent.mkdir(parents=True);pdf.write_bytes(b'%PDF')
    for path in VIEWERS.values():
        target=tmp_path/path.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_text('viewer')
    (tmp_path/'runs/published'/ps.MEDIA_RUN/'GIFs').mkdir(parents=True)
    mica=tmp_path/'runs/published'/'20261025_090000_m2-1600nm-presentation-review_test'
    mica.mkdir();(mica/'run.json').write_text(json.dumps({'status':'completed'}))
    (mica/'parameter_maps.html').write_text('mica')
    alternating=tmp_path/'runs/published'/'20261025_fixed-alternating-amplitude-classic-gifs_test'
    alternating.mkdir();(alternating/'run.json').write_text(json.dumps({'status':'completed'}))
    (alternating/'START_HERE.html').write_text('alternating')
    assert ps.present(tmp_path)==pdf
    assert events==['service',pdf]
    with pytest.raises(ValueError,match='requires that port'):
        ps.present(tmp_path,port=9999)


def test_present_cli(monkeypatch):
    from neuristor.interfaces import presentation_server
    seen=[]
    def open_slides():
        seen.append(True)
        return Path('/test/slides.pdf')
    monkeypatch.setattr(presentation_server,'present',open_slides)
    result=CliRunner().invoke(app,['present'])
    assert result.exit_code==0,result.output
    assert seen and 'all slide viewers are running' in result.output
