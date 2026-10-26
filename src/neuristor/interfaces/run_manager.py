"""Loopback-only artifact library with explicit, recoverable run deletion.

A file:// HTML catalog has no filesystem deletion authority. This small local
manager adds it only after a same-origin, session-token-protected confirmation.
All lifecycle rules live in RunRegistry, not in browser code. No scientific
equations, public-bundle mutations, network exposure, or permanent erase endpoint.
"""

from __future__ import annotations

import json
import secrets
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from urllib.parse import unquote, urlsplit

from neuristor.orchestration.artifact_catalog import build_artifact_catalog
from neuristor.runs import RunRegistry, find_project_root


def create_manager(project_root=None, *, port: int = 0) -> ThreadingHTTPServer:
    """Create, but do not start, a loopback manager (port 0 chooses a free port).

    The random token is injected only into the served page, never saved in the
    exported HTML. Other web origins cannot read it or issue authorized writes.
    The returned server must be closed by its caller.
    """
    root = find_project_root(project_root)
    registry = RunRegistry(root)
    build_artifact_catalog(root)
    token = secrets.token_urlsafe(32)
    lock = RLock()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, *args):
            pass  # No tokens or user-provided paths in terminal output.

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            super().end_headers()

        def trusted_host(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def reply(self, status, payload, *, content_type="application/json"):
            body = json.dumps(payload).encode() if content_type == "application/json" else payload.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.trusted_host():
                self.reply(403, {"error": "Untrusted Host header"})
                return
            path = unquote(urlsplit(self.path).path)
            if path == "/":
                self.send_response(302)
                self.send_header("Location", "/runs/catalog/index.html")
                self.end_headers()
                return
            if path == "/runs/catalog/index.html":
                with lock:
                    page = (root / "runs/catalog/index.html").read_text()
                    bootstrap = json.dumps(dict(token=token)).replace("<", "\\u003c")
                    page = page.replace("</html>", f"<script>enableRunManager({bootstrap});</script></html>")
                self.reply(200, page, content_type="text/html; charset=utf-8")
                return
            if path == "/api/trash":
                with lock:
                    self.reply(200, registry.trashed())
                return
            target = (root / path.lstrip("/")).resolve()
            if not target.is_relative_to(root) or any(p.startswith(".") for p in Path(path).parts if p not in {"/"}):
                self.reply(403, {"error": "Path is outside the visible project"})
                return
            super().do_GET()

        def do_HEAD(self):
            # Do not bypass containment/Host checks through the inherited path.
            self.send_error(405, "Use GET")

        def do_POST(self):
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if (not self.trusted_host() or self.headers.get("Origin") != origin
                    or not secrets.compare_digest(self.headers.get("X-Neuristor-Token", "").encode(), token.encode())):
                self.reply(403, {"error": "Use the local manager page to confirm this operation"})
                return
            if self.headers.get("Content-Type") != "application/json":
                self.reply(415, {"error": "JSON required"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict) or payload.get("confirm") is not True:
                    raise ValueError("Explicit confirmation required")
                key = "run_id" if self.path == "/api/trash" else "trash_id"
                if not isinstance(payload.get(key), str):
                    raise ValueError("A string run/trash ID is required")
                with lock:
                    if self.path == "/api/trash":
                        result = registry.trash(payload["run_id"])
                    elif self.path == "/api/restore":
                        result = registry.restore(payload["trash_id"])
                    else:
                        self.reply(404, {"error": "Unknown action"})
                        return
                    # Operation already succeeded. Report index-refresh errors
                    # separately rather than claiming the bundle was not moved.
                    try:
                        build_artifact_catalog(root)
                    except (OSError, ValueError) as exc:
                        result["index_warning"] = str(exc)
                self.reply(200, result)
            except (KeyError, ValueError, OSError) as exc:
                self.reply(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
