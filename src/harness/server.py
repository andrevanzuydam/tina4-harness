"""A single shared model server.

Every Jupyter notebook kernel is its own OS process, so a model loaded in one kernel cannot be
reached from another — open two notebooks and you pay for the weights twice. This server loads the
model ONCE in a dedicated process and serves every kernel over HTTP, so the 2B lives in ~1.6 GB of
memory total no matter how many notebooks you open.

    python -m harness.server            # start it (reads HARNESS_MODEL_PORT, HARNESS_THREADS)

Each kernel's ``load_model()`` returns a thin :class:`RemoteModel` proxy (see
``backends/remote_backend.py``) whenever ``HARNESS_MODEL_SERVER`` points here — same interface,
forwards generate()/describe()/memory() to this process. Generation is serialised on one lock
(a single MLX model can't run two at once); the thread pool just lets requests queue.

Endpoints:
    GET  /health                       -> "ok"
    GET  /info                         -> {repo_id, backend, device}
    GET  /memory                       -> model.memory()
    POST /generate {messages, max_tokens?, temperature?, stream?}
                                       -> {text}, or an ndjson stream of base64 deltas + "__END__"
    POST /describe {image, prompt?, max_tokens?}  -> {text}   (vision; image is a local path)
"""
from __future__ import annotations

import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .model import Settings, format_memory, load_model

_model = None
_model_lock = threading.Lock()      # one generation at a time on the single model
_slots: threading.BoundedSemaphore  # caps concurrent in-flight requests (the "threads")


def _get():
    """The one shared model, loaded locally (never via the remote proxy — that would recurse)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = load_model(allow_remote=False, settings=Settings())
    return _model


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):  # keep the console quiet
        pass

    # --- helpers ---------------------------------------------------------------------
    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _send_json(self, obj, status: int = 200) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str, status: int = 200) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # --- routes ----------------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send_text("ok")
        elif self.path == "/info":
            m = _get()
            self._send_json({"repo_id": getattr(m, "repo_id", ""), "backend": getattr(m, "backend", ""),
                             "device": getattr(m, "device", "")})
        elif self.path == "/memory":
            self._send_json(_get().memory())
        else:
            self._send_text("not found", 404)

    def do_POST(self):  # noqa: N802
        with _slots:  # bound concurrency to HARNESS_THREADS
            try:
                data = self._body()
                if self.path == "/generate":
                    self._generate(data)
                elif self.path == "/describe":
                    self._describe(data)
                else:
                    self._send_text("not found", 404)
            except Exception as exc:  # report the error back to the proxy
                try:
                    self._send_json({"error": str(exc)}, 500)
                except Exception:
                    pass

    def _generate(self, d: dict) -> None:
        messages = d["messages"]
        mt = d.get("max_tokens")
        temp = d.get("temperature")
        think = d.get("enable_thinking", True)
        model = _get()
        if d.get("stream"):
            # Stream base64-encoded deltas, one per line, then an "__END__" line (chunked).
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def on_text(chunk: str) -> None:
                self._write_chunk(base64.b64encode(chunk.encode("utf-8")) + b"\n")

            with _model_lock:
                model.generate(messages, on_text=on_text, max_tokens=mt, temperature=temp,
                               enable_thinking=think)
            self._write_chunk(b"__END__\n")
            self._write_chunk(b"")  # terminating 0-length chunk
        else:
            with _model_lock:
                text = model.generate(messages, max_tokens=mt, temperature=temp, enable_thinking=think)
            self._send_json({"text": text})

    def _write_chunk(self, payload: bytes) -> None:
        self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()

    def _describe(self, d: dict) -> None:
        model = _get()
        with _model_lock:
            text = model.describe(d["image"], d.get("prompt", "Describe this image."),
                                  max_tokens=int(d.get("max_tokens", 512)))
        self._send_json({"text": text})


def serve(host: str = "127.0.0.1", port: int | None = None, threads: int | None = None) -> None:
    global _slots
    port = int(port or os.environ.get("HARNESS_MODEL_PORT", "7777"))
    threads = int(threads or os.environ.get("HARNESS_THREADS", "8"))
    _slots = threading.BoundedSemaphore(threads)
    model = _get()  # load up front so the first notebook request is instant
    print(f"Harness model server · http://{host}:{port} · {threads} threads · "
          f"{format_memory(model.memory())}", flush=True)
    ThreadingHTTPServer((host, port), _Handler).serve_forever()


if __name__ == "__main__":
    serve()
