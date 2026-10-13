#!/usr/bin/env python3
"""
Shared HTTP server for the live knowledge-graph viewer (viewer/kg_viewer.html).

Used by both guida_museale_core.py (graph evolution during the conversation)
and museum_explorer.py (graph construction during the head-scan), so whichever
process is running serves the same page on the same port.
"""

import functools
import http.server
import socketserver
import threading
import console
from pathlib import Path


class _QuietViewerHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the viewer directory without logging a line per request — the
    page polls kg_snapshot.json once a second, and the default access log
    would otherwise interleave with (and garble) the interactive CLI prompt."""

    def log_message(self, format, *args):
        pass


class _ViewerServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_viewer_server(viewer_dir: Path, port: int, quiet_on_conflict: bool = False) -> bool:
    """
    Serve viewer_dir (kg_viewer.html + the live kg_snapshot.json) on a
    background thread.

    Returns True if this process bound the port and is therefore the "primary"
    instance — the only one allowed to write kg_snapshot.json. Losing the bind
    means another instance is already the live one, so the caller must NOT
    write the snapshot, or the two processes would silently overwrite each
    other's graph state.

    quiet_on_conflict: suppress the conflict message (used by periodic
    re-acquisition retries, which would otherwise spam the terminal).
    """
    handler = functools.partial(_QuietViewerHandler, directory=str(viewer_dir))
    try:
        httpd = _ViewerServer(("localhost", port), handler)
    except OSError as e:
        if not quiet_on_conflict:
            console.detail(f"[KG Viewer] Server non avviato sulla porta {port} ({e}) — un'altra istanza è già attiva: "
                           f"questo processo non scriverà lo snapshot per non sovrascrivere quello in corso. "
                           f"Ritenterò in automatico quando la porta si libera.")
        return False
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    console.detail(f"🌐 KG Viewer live: http://localhost:{port}/kg_viewer.html")
    return True
