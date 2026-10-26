"""Read-only loopback delivery for the presentation's two local HTML viewers.

PDF applications may refuse file URLs and Launch actions. Ordinary HTTP links
avoid those action types. Only the three reviewed viewer/media bundles are
served; the repository, private papers and arbitrary filesystem paths are not.
"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

from neuristor.runs import find_project_root

PORT = 8767
SERVICE = 'neuristor-presentation-viewers-v1'
LOSS_RUN = '20261021_054434_loss-function-comparison-all-44-stacked-gifs_15921f'
MAP_RUN = '20261007_123036_fixed-scale-viewer-portable-screened-parameter-m_4325d1'
MEDIA_RUN = '20261007_074702_portable-screened-parameter-map-with-point-gifs_eaf6ab'
BUNDLES = (LOSS_RUN, MAP_RUN, MEDIA_RUN)
VIEWERS = {'/loss-gifs': f'/public_jobs/{LOSS_RUN}/START_HERE.html',
           '/parameter-map': f'/public_jobs/{MAP_RUN}/parameter_maps.html'}


def create_presentation_server(project_root=None, *, port=PORT):
    """Create an IPv4-loopback server restricted to the reviewed display bundles."""
    project = Path(project_root or find_project_root()).resolve()
    roots = [(project/'public_jobs'/name).resolve() for name in BUNDLES]

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def list_directory(self, path):
            self.send_error(403, 'Directory listing disabled')
            return None

        def send_head(self):
            host = self.headers.get('Host', '').split(':', 1)[0]
            if host not in ('127.0.0.1', 'localhost'):
                self.send_error(403, 'Local host required')
                return None
            route = unquote(urlsplit(self.path).path)
            if route == '/health':
                from io import BytesIO
                body = json.dumps(dict(service=SERVICE, project=str(project))).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                return BytesIO(body)
            if route in VIEWERS:
                self.send_response(302)
                self.send_header('Location', VIEWERS[route])
                self.send_header('Content-Length', '0')
                self.end_headers()
                return None
            target = (project/route.lstrip('/')).resolve()
            if not any(target.is_relative_to(root) for root in roots):
                self.send_error(403, 'Only presentation viewer bundles are served')
                return None
            return super().send_head()

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            super().end_headers()

    return ThreadingHTTPServer(('127.0.0.1', port), partial(Handler, directory=str(project)))


def ensure_running(project_root=None, *, port=PORT):
    """Reuse this project's viewer service or start a detached local process.

    No startup agent or system settings are installed. The process lasts until
    stopped or the computer restarts; the presentation launcher restarts it.
    An occupied port belonging to a different service is never taken over.
    """
    project = Path(project_root or find_project_root()).resolve()
    expected = dict(service=SERVICE, project=str(project))
    health = f'http://127.0.0.1:{port}/health'
    def check():
        try:
            with urlopen(health, timeout=1) as response:
                value = json.load(response)
        except URLError:
            return False
        except (ValueError, OSError) as exc:
            raise RuntimeError(f'Port {port} is occupied by another service') from exc
        if value != expected:
            raise RuntimeError(f'Port {port} belongs to a different service or checkout')
        return True
    if not check():
        log = project/'outputs'/'presentation_viewers.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('ab') as stream:
            process = subprocess.Popen([sys.executable, '-m', 'neuristor.interfaces.presentation_server', '--project', str(project), '--port', str(port)],
                cwd=project, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True)
        for _ in range(50):
            if check():
                return
            if process.poll() is not None:
                raise RuntimeError(f'Viewer service stopped; see {log}')
            time.sleep(.1)
        raise RuntimeError(f'Viewer service did not start; see {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    if args.background:
        ensure_running(args.project, port=args.port)
        print(f'Presentation viewers ready at http://127.0.0.1:{args.port}')
        return
    with create_presentation_server(args.project, port=args.port) as server:
        print(f'Presentation viewers: http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
