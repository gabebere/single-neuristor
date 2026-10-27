"""Read-only loopback delivery for the presentation's local HTML viewers.

PDF applications may refuse file URLs and Launch actions. Ordinary HTTP links
avoid those action types. Only the reviewed viewer/media bundles are served;
the repository, private papers and arbitrary filesystem paths are not.
"""
from __future__ import annotations

import argparse
import hashlib
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
SERVICE = 'neuristor-presentation-viewers-v4'
LOSS_RUN = '20261021_054434_loss-function-comparison-all-44-stacked-gifs_15921f'
MAP_RUN = '20261007_123036_fixed-scale-viewer-portable-screened-parameter-m_4325d1'
MEDIA_RUN = '20261007_074702_portable-screened-parameter-map-with-point-gifs_eaf6ab'
BUNDLES = (LOSS_RUN, MAP_RUN, MEDIA_RUN)
VIEWERS = {'/loss-gifs': f'/runs/published/{LOSS_RUN}/START_HERE.html',
           '/parameter-map': f'/runs/published/{MAP_RUN}/parameter_maps.html'}


def presentation_project_root(project_root=None):
    """Resolve relative to this file, so a copied project needs no installation."""
    return Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[3]


def rebuilt_presentation(project):
    """Read only a completed full reproduction's contained, checked link manifest.

    The optional pointer is administrative metadata, not another result copy.
    Smoke runs, unfinished runs, path escapes and private symlinks cannot become
    presentation dependencies. The standalone launcher keeps standard-library
    compatibility and needs no simulation packages to follow these links.
    """
    pointer = project/'runs/local/presentation.json'
    if not pointer.exists():
        return None
    value = json.loads(pointer.read_text())
    if not isinstance(value, dict) or value.get('schema_version') != 1 or not isinstance(value.get('bundle'), str):
        raise ValueError('Invalid presentation reproduction pointer')
    relative = Path(value['bundle'])
    if relative.is_absolute() or relative.parts[:2] != ('runs', 'local') or len(relative.parts) != 3:
        raise ValueError('Invalid presentation reproduction location')
    location = project/relative
    if location.is_symlink():
        raise ValueError('Presentation reproduction must not be a symlink')
    location = location.resolve()
    if not location.is_relative_to(project.resolve()):
        raise ValueError('Presentation reproduction is outside this checkout')
    run = json.loads((location/'run.json').read_text())
    raw = (location/'presentation_manifest.json').read_bytes()
    manifest = json.loads(raw)
    if not isinstance(run, dict) or not isinstance(manifest, dict):
        raise ValueError('Invalid presentation reproduction manifest')
    if (run.get('kind') != 'presentation-reproduction' or run.get('status') != 'completed'
            or manifest.get('schema_version') != 1 or manifest.get('profile') != 'full'
            or hashlib.sha256(raw).hexdigest() != value.get('manifest_sha256')):
        raise ValueError('Presentation reproduction is incomplete or its manifest changed')
    expected = {'/loss-gifs', '/parameter-map', '/mica-map', '/alternating-gifs'}
    if not isinstance(manifest.get('routes'), dict) or set(manifest['routes']) != expected:
        raise ValueError('Presentation reproduction has incomplete viewer links')
    for path in [manifest.get('entry'), *manifest['routes'].values()]:
        if not isinstance(path, str):
            raise ValueError('Invalid presentation viewer path')
        target = (location/path).resolve()
        if Path(path).is_absolute() or not target.is_relative_to(location) or not target.is_file():
            raise ValueError('Presentation viewer path is missing or escapes its bundle')
    return location, manifest


def completed_mica_reviews(project):
    """Find locally published M2 reviews; never link an unfinished expansion."""
    result = []
    for root in (project/'runs/published').glob('*m2-1600nm*review*'):
        try:
            manifest = json.loads((root/'run.json').read_text())
        except (OSError, ValueError):
            continue
        if manifest.get('status') == 'completed' and (root/'parameter_maps.html').is_file():
            result.append(root.resolve())
    return sorted(result, key=lambda root: root.name)


def completed_alternating_gifs(project):
    """Locate completed frozen-vector GIF studies for the final slide only."""
    result = []
    for root in (project/'runs/published').glob('*fixed-alternating-amplitude-classic-gifs*'):
        if root.is_symlink():
            continue
        try:
            manifest = json.loads((root/'run.json').read_text())
        except (OSError, ValueError):
            continue
        if manifest.get('status') == 'completed' and (root/'START_HERE.html').is_file():
            result.append(root.resolve())
    return sorted(result, key=lambda root: root.name)


def create_presentation_server(project_root=None, *, port=PORT):
    """Create an IPv4-loopback server restricted to the reviewed display bundles."""
    project = presentation_project_root(project_root)
    roots = [(project/'runs/published'/name).resolve() for name in BUNDLES]

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
            # Frozen viewers still request the former /public_jobs prefix.
            # Translate only that known lane; the allowlist below still checks
            # resolved targets, including symlinks and traversal attempts.
            route = unquote(urlsplit(path).path)
            if route.startswith('/public_jobs/'):
                path = '/runs/published/' + route[len('/public_jobs/'):]
            elif route.startswith('/runs/public_jobs/'):
                path = '/runs/published/' + route[len('/runs/public_jobs/'):]
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
            alternating = completed_alternating_gifs(project)
            if route == '/health':
                from io import BytesIO
                body = json.dumps(dict(service=SERVICE, project=str(project))).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                return BytesIO(body)
            try:
                recreated = rebuilt_presentation(project)
            except (OSError, ValueError, KeyError, TypeError):
                self.send_error(503, 'Invalid or incomplete presentation reproduction manifest')
                return None
            viewers = dict(VIEWERS)
            if mica:
                viewers['/mica-map'] = f'/runs/published/{mica[-1].name}/parameter_maps.html'
            if alternating:
                viewers['/alternating-gifs'] = f'/runs/published/{alternating[-1].name}/START_HERE.html'
            recreated_roots = []
            if recreated:
                location, manifest = recreated
                prefix = '/' + location.relative_to(project).as_posix() + '/'
                viewers.update({key: prefix + value for key, value in manifest['routes'].items()})
                viewers['/reproduction'] = prefix + manifest['entry']
                recreated_roots.append(location)
            if route in viewers:
                self.send_response(302)
                self.send_header('Location', viewers[route])
                self.send_header('Content-Length', '0')
                self.end_headers()
                return None
            target = Path(self.translate_path(self.path)).resolve()
            if not any(target.is_relative_to(root) for root in [*roots, *mica, *alternating, *recreated_roots]):
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
        log = project/'runs/scratch'/'presentation_viewers.log'
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
    required = [project/'docs/presentation'/'main.pdf']
    recreated = rebuilt_presentation(project)
    if recreated:
        location, manifest = recreated
        if not required[0].is_file():
            raise FileNotFoundError('Missing presentation PDF: ' + str(required[0]))
        return required[0]
    required.extend(project/path.lstrip('/') for path in VIEWERS.values())
    required.append(project/'runs/published'/MEDIA_RUN/'GIFs')
    missing = [str(path.relative_to(project)) for path in required if not path.exists()]
    if not completed_mica_reviews(project):
        missing.append('runs/published/<completed M2 1600 nm review>/parameter_maps.html')
    if not completed_alternating_gifs(project):
        missing.append('runs/published/<completed alternating amplitude GIF sweep>/START_HERE.html')
    if missing:
        raise FileNotFoundError('Missing presentation files:\n  ' + '\n  '.join(missing)
            + '\nCopy the presentation and its runs/published viewer/media bundles; the PDF alone is insufficient.')
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
