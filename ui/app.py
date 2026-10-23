import json
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset_checks.dataset_loader import build_test_case
from orchestrator.evaluation_orchestrator import run_test_case

HOST = "127.0.0.1"
PORT = int(os.environ.get("EVAL_UI_PORT", "8000"))

_PAGE_PATH = os.path.join(os.path.dirname(__file__), "index.html")

_jobs = {}
_jobs_lock = threading.Lock()

def _set_job(job_id, **fields):
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)

def _get_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        snapshot = dict(job)

    if snapshot.get("state") == "running" and snapshot.get("started_at"):
        snapshot["elapsed"] = round(time.time() - snapshot["started_at"], 1)

    snapshot.pop("started_at", None)
    return snapshot

def _run_job(job_id, form):
    """Runs one test case in a background thread."""

    started = time.time()

    try:
        test_case = build_test_case(
            row_id=form.get("id") or "ui",
            category=form.get("category") or "ui",
            category_position=0,
            input_text=(form.get("input") or "").strip(),
            context_text=(form.get("context") or "").strip(),
            ground_truth_text=(form.get("ground_truth") or "").strip(),
            pattern_text=(form.get("pattern") or "").strip(),
            source="ui",
        )

        result = run_test_case(test_case)

        _set_job(
            job_id,
            state="done",
            elapsed=round(time.time() - started, 1),
            test_case=test_case,
            result=result,
        )

    except Exception:
        _set_job(
            job_id,
            state="error",
            elapsed=round(time.time() - started, 1),
            error=traceback.format_exc(),
        )

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def _send(self, status, body, content_type):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status, data):
        self._send(status, json.dumps(data, default=str), "application/json; charset=utf-8")

    def do_GET(self):
        route = urlparse(self.path)

        if route.path in ("/", "/index.html"):
            with open(_PAGE_PATH, "r", encoding="utf-8") as page:
                self._send(200, page.read(), "text/html; charset=utf-8")
            return

        if route.path == "/api/status":
            job_id = (parse_qs(route.query).get("id") or [""])[0]
            job = _get_job(job_id)
            if job is None:
                self._send_json(404, {"state": "unknown"})
                return
            self._send_json(200, job)
            return

        if route.path == "/api/config":
            from config import settings
            from models.model_client import uses_different_models
            self._send_json(200, {
                "generator": f"{settings.GENERATOR_PROVIDER}/{settings.GENERATOR_MODEL}",
                "judge": f"{settings.JUDGE_PROVIDER}/{settings.JUDGE_MODEL}",
                "different_models": uses_different_models(),
            })
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        route = urlparse(self.path)

        if route.path != "/api/run":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            form = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        if not (form.get("input") or "").strip():
            self._send_json(400, {"error": "A question is required."})
            return

        job_id = str(uuid.uuid4())
        _set_job(job_id, state="running", elapsed=0.0, started_at=time.time())

        thread = threading.Thread(target=_run_job, args=(job_id, form), daemon=True)
        thread.start()

        self._send_json(202, {"id": job_id})

def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\n  Eval console running at  http://{HOST}:{PORT}")
    print("  Excel stays the source of truth -- this is for trying one prompt.")
    print("  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.\n")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
