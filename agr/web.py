"""
Local demo server: the demo page (docs/index.html) with live questions answered by the real system.

    python -m agr.web          (or ./demo.sh)  ->  open http://localhost:8000

The page is the same file that is published read-only (GitHub Pages from docs/). Served from here it also has a live
box: a question (typed, or picked from the public / held-out lists) is answered by the chosen pipeline (agentic,
GraphRAG or RAG) against TigerGraph, and the page shows the answer line, citations with snippets, the full trace and,
for a known question, the published answer and our score. Keys and secrets stay in .env on this machine; the browser
only talks to this local server.
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("TG_TRANSPORT", "mcp")

from agr import llm  # noqa: E402
from agr.config import ROOT, SETTINGS  # noqa: E402
from agr.pipelines import PIPELINES  # noqa: E402
from agr.run import answer_line  # noqa: E402
from agr.tools import Toolbox  # noqa: E402

PORT = int(os.getenv("PORT", "8000"))
DOCS = ROOT / "docs"
_lock = threading.Lock()          # one question at a time
_tb: Toolbox | None = None


def _questions() -> dict:
    out = {}
    for name in ("questions_public.jsonl", "questions_heldout.jsonl"):
        p = ROOT / "data" / "questions" / name
        if p.exists():
            for l in open(p, encoding="utf-8"):
                if l.strip():
                    q = json.loads(l)
                    out[q["qid"]] = q
    return out


QUESTIONS = _questions()


def toolbox() -> Toolbox:
    global _tb
    if _tb is None:
        _tb = Toolbox()
        _tb.tg.wait_until_awake()     # wake an auto-suspended workspace before the first question
        _tb.catalog()
    return _tb


def ask(question: str, pipeline: str = "agentic", qid: str = "") -> dict:
    if pipeline not in PIPELINES:
        return {"error": f"unknown pipeline {pipeline}"}
    with _lock:
        tb = toolbox()
        llm.wait_for_budget()
        t0 = time.time()
        r = PIPELINES[pipeline](tb, qid or "demo", question)
        line = answer_line(r)
        out = {**line, "pipeline": pipeline, "wall_s": round(time.time() - t0, 1),
               "citation_snippets": [{"doc_id": c["doc_id"], "title": c["title"], "text": c["text"][:600]}
                                     for c in r.citations]}
    q = QUESTIONS.get(qid)
    if q and "answer" in q:           # a published question: show the gold answer and our score
        import importlib.util
        spec = importlib.util.spec_from_file_location("score_public", ROOT / "scripts" / "score_public.py")
        sp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sp)
        out["gold"] = {"status": q.get("status"), "answer": q.get("answer"), "docs": q.get("gold_doc_ids", [])}
        out["score"] = sp.score_one(q, line)["score"]
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")   # local server; it holds no secrets in responses
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/status":
            body = {"live": True, "graph": SETTINGS.tg_graph, "model": SETTINGS.model,
                    "pipelines": list(PIPELINES), "keys": len(llm._KEYS)}
            return self._send(200, json.dumps(body).encode("utf-8"), "application/json")
        files = {"/": "index.html", "/index.html": "index.html", "/data.json": "data.json"}
        if path in files and (DOCS / files[path]).exists():
            ctype = "text/html; charset=utf-8" if path != "/data.json" else "application/json"
            return self._send(200, (DOCS / files[path]).read_bytes(), ctype)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/ask":
            return self._send(404, b"{}", "application/json")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            q = str(req.get("question", "")).strip()
            out = ask(q, str(req.get("pipeline", "agentic")), str(req.get("qid", ""))) if q \
                else {"error": "empty question"}
        except Exception as e:  # show the problem on the page instead of failing silently
            out = {"error": f"{type(e).__name__}: {e}"}
        self._send(200, json.dumps(out, ensure_ascii=False).encode("utf-8"), "application/json")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"[demo] waking TigerGraph and loading the catalogue ...", flush=True)
    toolbox()
    print(f"[demo] ready: open http://localhost:{PORT}  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
