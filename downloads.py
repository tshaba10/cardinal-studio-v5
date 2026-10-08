"""Loopback-only, token-authorized file streaming; large ZIPs stay off Streamlit RAM."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import shutil
import threading
import json
from quality_report import download_allowed, sha256_file


class DownloadServer:
    def __init__(self):
        self.files = {}
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                expected = f"127.0.0.1:{self.server.server_port}"
                if self.headers.get("Host") != expected:
                    self.send_error(403)
                    return
                with owner.lock:
                    path = owner.files.get(self.path)
                if path is None or not path.is_file():
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(path.stat().st_size))
                self.send_header(
                    "Content-Disposition", 'attachment; filename="cardinal-dataset.zip"'
                )
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                try:
                    with path.open("rb") as file:
                        shutil.copyfileobj(file, self.wfile, length=1024 * 1024)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass  # Never log capability tokens.

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def grant(self, path):
        path = Path(path).resolve()
        root = Path(__file__).resolve().parent / ".runs"
        if path.parent.parent != root or path.name != "dataset.zip" or not path.is_file():
            raise ValueError("Only a generated dataset ZIP can be downloaded.")
        result_path = path.parent / "result.json"
        if not result_path.is_file():
            raise ValueError("The generation result is missing. Regenerate before downloading.")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not download_allowed(result):
            raise ValueError(
                "The Python quality gate has not passed. AI advice cannot unlock this download."
            )
        if sha256_file(path) != result.get("dataset_sha256"):
            raise ValueError(
                "The dataset ZIP changed after verification. Regenerate before downloading."
            )
        token = "/" + secrets.token_urlsafe(32)
        with self.lock:
            self.files[token] = path
        return f"http://127.0.0.1:{self.server.server_port}{token}"

    def revoke(self, url):
        from urllib.parse import urlparse

        with self.lock:
            self.files.pop(urlparse(url).path, None)
