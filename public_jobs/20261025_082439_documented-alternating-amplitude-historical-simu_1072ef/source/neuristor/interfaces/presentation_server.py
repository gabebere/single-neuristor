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
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

PORT = 8767
SERVICE = 'neuristor-presentation-viewers-v2'
LOSS_RUN = '20261021_054434_loss-function-comparison-all-44-stacked-gifs_15921f'
MAP_RUN = '20261007_123036_fixed-scale-viewer-portable-screened-parameter-m_4325d1'
MEDIA_RUN = '20261007_074702_portable-screened-parameter-map-with-point-gifs_eaf6ab'
BUNDLES = (LOSS_RUN, MAP_RUN, MEDIA_RUN)
VIEWERS = {'/loss-gifs': f'/public_jobs/{LOSS_RUN}/START_HERE.html',
           '/parameter-map': f'/public_jobs/{MAP_RUN}/parameter_maps.html'}


def presentation_project_root(project_root=None):
    """Resolve relative to this file, so a copied project needs no installation."""
    return Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[3]


def completed_mica_reviews(project):
    """Find locally published M2 reviews; never link an unfinished expansion."""
    result = []
    for root in (project/'public_jobs').glob('*m2-1600nm*review*'):
        try:
            manifest = json.loads((root/'run.json').read_text())
        except (OSError, ValueError):
            continue
        if manifest.get('status') == 'completed' and (root/'parameter_maps.html').is_file():
            result.append(root.resolve())
    return sorted(result, key=lambda root: root.name)


def create_presentation_server(project_root=None, *, port=PORT):
    """Create an IPv4-loopback server restricted to the reviewed display bundles."""
    project = presentation_project_root(project_root)
    roots = [(project/'public_jobs'/name).resolve() for name in BUNDLES]

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def list_directory(self, path):
            self.send_error(403, 'Directory listing disabled')
            return None

        def translate_path(self, path):
            # Some immutable historical bundles spelled Figures/figures
            # differently in HTML and on disk. Resolve only unambiguous matches
            # so a copy also works on case-sensitive filesystems.
            candidate = Path(super().translate_path(path))
            if candidate.exists():
                return str(candidate)
            relative = candidate.relative_to(project)
            resolved = project
            for part in relative.parts:
                exact = resolved/part
                if exact.exists():
                    resolved = exact
                elif resolved.is_dir():
                    matches = [p for p in resolved.iterdir() if p.name.casefold() == part.casefold()]
                    if len(matches) != 1:
                        return str(candidate)
                    resolved = matches[0]
                else:
                    return str(candidate)
            return str(resolved)

        def send_head(self):
            host = self.headers.get('Host', '').split(':', 1)[0]
            if host not in ('127.0.0.1', 'localhost'):
                self.send_error(403, 'Local host required')
                return None
            route = unquote(urlsplit(self.path).path)
            mica = completed_mica_reviews(project)
            if route == '/health':
                from io import BytesIO
                body = json.dumps(dict(service=SERVICE, project=str(project))).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                return BytesIO(body)
            viewers = dict(VIEWERS)
            if mica:
                viewers['/mica-map'] = f'/public_jobs/{mica[-1].name}/parameter_maps.html'
            if route in viewers:
                self.send_response(302)
                self.send_header('Location', viewers[route])
                self.send_header('Content-Length', '0')
                self.end_headers()
                return None
            target = Path(self.translate_path(self.path)).resolve()
            if not any(target.is_relative_to(root) for root in [*roots, *mica]):
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
    project = presentation_project_root(project_root)
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
            options = dict(start_new_session=True) if os.name != 'nt' else dict(
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--project', str(project), '--port', str(port)],
                cwd=project, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, **options)
        for _ in range(50):
            if check():
                return
            if process.poll() is not None:
                raise RuntimeError(f'Viewer service stopped; see {log}')
            time.sleep(.1)
        raise RuntimeError(f'Viewer service did not start; see {log}')


def check_presentation_files(project_root=None):
    """Fail clearly when only the PDF, rather than its linked viewers, was copied."""
    project = presentation_project_root(project_root)
    required = [project/'Simulations_on_VO2'/'main.pdf']
    required.extend(project/path.lstrip('/') for path in VIEWERS.values())
    required.append(project/'public_jobs'/MEDIA_RUN/'GIFs')
    missing = [str(path.relative_to(project)) for path in required if not path.exists()]
    if not completed_mica_reviews(project):
        missing.append('public_jobs/<completed M2 1600 nm review>/parameter_maps.html')
    if missing:
        raise FileNotFoundError('Missing presentation files:\n  ' + '\n  '.join(missing)
            + '\nCopy the presentation and its public_jobs viewer/media bundles; the PDF alone is insufficient.')
    return required[0]


def open_presentation_pdf(path):
    """Ask the desktop's default PDF application to open the existing slide deck."""
    path = Path(path).resolve()
    if sys.platform == 'darwin':
        subprocess.run(['open', str(path)], check=True)
    elif os.name == 'nt':
        os.startfile(str(path))
    else:
        opener = shutil.which('xdg-open')
        if not opener:
            raise RuntimeError(f'Viewers are ready; open this PDF manually: {path}')
        subprocess.run([opener, str(path)], check=True)


def present(project_root=None, *, port=PORT):
    """Start/reuse all linked viewers, then open the deck, without scientific dependencies."""
    if port != PORT:
        raise ValueError(f'The PDF buttons use port {PORT}; presentation mode requires that port')
    project = presentation_project_root(project_root)
    pdf = check_presentation_files(project)
    ensure_running(project, port=port)
    open_presentation_pdf(pdf)
    return pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--present', action='store_true', help='Start all viewers in the background and open the PDF')
    args = parser.parse_args()
    if args.present:
        try:
            pdf = present(args.project, port=args.port)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.exit(1, f'{exc}\n')
        print(f'Opened {pdf}\nAll slide viewers are ready at http://127.0.0.1:{args.port}')
        return
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
