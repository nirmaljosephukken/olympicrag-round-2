"""Minimal TigerGraph 4.x / Savanna REST client (requests only)."""
from __future__ import annotations

import datetime
import json
import os
import re
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

import requests

from agr.config import SETTINGS, ROOT


class TigerGraphError(RuntimeError):
    pass


_local = threading.local()


QUERY_LOG = Path(os.getenv("TG_QUERY_LOG", str(ROOT / "logs" / "tigergraph_queries.jsonl")))
_log_lock = threading.Lock()
CONTEXT = {"qid": ""}   # set by the runner, written to every log line


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _loggable(params: dict) -> dict:
    """Parameters as logged and traced: a query vector is summarised, everything else is kept."""
    out = {}
    for k, v in (params or {}).items():
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, float) for x in v[:4]):
            out[k] = f"<{len(v)}-d vector>"
        else:
            out[k] = list(v) if isinstance(v, (set, tuple)) else v
    return out


def _rows(out) -> int:
    """Rows / vertices returned: the length of every list in the result objects (scalars count once)."""
    if not isinstance(out, list):
        return 0
    n = 0
    for r in out:
        if isinstance(r, dict):
            for v in r.values():
                n += len(v) if isinstance(v, (list, dict)) else 1
    return n


def log_call(kind: str, name: str, params: dict, rows: int, ms: float, via: str, ts: str, error: str = "") -> dict:
    """Append one TigerGraph call to logs/tigergraph_queries.jsonl and to this thread's trace buffer."""
    rec = {"ts": ts, "qid": CONTEXT.get("qid", ""), "graph": SETTINGS.tg_graph, "kind": kind, "query": name,
           "params": _loggable(params), "rows": rows, "ms": round(ms, 1), "via": via}
    if error:
        rec["error"] = error
    with _log_lock:
        QUERY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(QUERY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if kind == "query":
        _local.calls = getattr(_local, "calls", []) + [rec]
    return rec


def take_calls() -> list[dict]:
    """The installed-query calls made on this thread since the last call (one trace step each)."""
    v = getattr(_local, "calls", [])
    _local.calls = []
    return v


def take_transports() -> list[str]:
    """Which transports served the queries made on this thread since the last call (for the trace)."""
    v = getattr(_local, "vias", [])
    _local.vias = []
    return v


MCP_CALLS: list | None = None   # when a list, every MCP call is also recorded here (used by the web page)


def _mcp_log(mcp, name: str, params: dict, out, secs: float, error: Exception | None = None) -> None:
    """TG_MCP_LOG=1 prints every call to the TigerGraph MCP server as it happens (for demos and debugging):
    the MCP tool, the installed GSQL query, its arguments, the time and how much came back."""
    if MCP_CALLS is not None:
        rows = None if error is not None else sum(len(v) for r in out if isinstance(r, dict)
                                                   for v in r.values() if isinstance(v, list))
        MCP_CALLS.append({"tool": next((n for n in getattr(mcp, "tools", {}) if n.endswith("run_installed_query")),
                                       "run_installed_query"),
                          "query": name, "params": {k: v for k, v in params.items() if not isinstance(v, (list, tuple))},
                          "ms": round(secs * 1000), "rows": rows, "error": type(error).__name__ if error else ""})
    if os.getenv("TG_MCP_LOG") != "1":
        return
    tool = next((n for n in getattr(mcp, "tools", {}) if n.endswith("run_installed_query")), "run_installed_query")
    args = ", ".join(f"{k}={(str(v)[:28] + '...') if len(str(v)) > 30 else v}" for k, v in params.items()
                     if not isinstance(v, (list, tuple)) or len(v) < 4)
    if error is not None:
        res = f"FAILED ({type(error).__name__}), falling back to RESTPP"
    else:
        rows = sum(len(v) for r in out if isinstance(r, dict) for v in r.values() if isinstance(v, list))
        res = f"{rows} row(s)"
    print(f"      [TigerGraph MCP] {tool} -> {name}({args})  {secs * 1000:.0f} ms, {res}", flush=True)


class TG:
    def __init__(self, s=SETTINGS):
        if not s.tg_host:
            raise TigerGraphError("TG_HOST is not set (copy .env.example to .env)")
        self.s, self.base, self.graph = s, s.tg_host, s.tg_graph
        self.http = requests.Session()
        self._token = s.tg_token or self._cached_token()
        self.calls = 0

    # -- auth -------------------------------------------------------------
    def _cached_token(self) -> str | None:
        p = ROOT / ".tg_token"
        if p.exists() and time.time() - p.stat().st_mtime < 20 * 86400:
            return p.read_text().strip() or None
        return None

    def wait_until_awake(self, max_wait: float | None = None) -> float:
        """Wake a Savanna workspace that auto-suspended (any request starts it) and wait until it answers, with
        back-off (5 s doubling to 60 s), for up to TG_WAKE_MAX_S seconds (default 900). Returns the seconds waited."""
        max_wait = float(os.getenv("TG_WAKE_MAX_S", "900")) if max_wait is None else max_wait
        t0, delay = time.time(), 5.0
        while True:
            try:
                r = self.http.get(f"{self.base}/api/ping", timeout=30)
                ok = r.status_code == 200 and "pong" in r.text
            except requests.RequestException:
                ok = False
            if ok:
                return time.time() - t0
            if time.time() - t0 > max_wait:
                raise TigerGraphError(f"TigerGraph workspace did not answer within {max_wait:.0f} s "
                                      f"(still starting?); check it in the Savanna console")
            print(f"[tigergraph] workspace starting (auto-resume), retrying in {delay:.0f} s ...", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)

    def token(self) -> str:
        if self._token:
            return self._token
        self.wait_until_awake()
        t0 = time.time()
        while True:
            r = self.http.post(f"{self.base}/gsql/v1/tokens",
                               json={"secret": self.s.tg_secret, "lifetime": "2592000"}, timeout=60)
            if r.ok and "token" in r.text:
                break
            if time.time() - t0 > 300:
                raise TigerGraphError(f"token request failed: {r.status_code} {r.text[:200]}")
            time.sleep(10)
        self._token = r.json()["token"]
        (ROOT / ".tg_token").write_text(self._token)
        return self._token

    def _h(self, extra: dict | None = None) -> dict:
        h = {"Authorization": f"Bearer {self.token()}", "GSQL-TIMEOUT": "600000"}
        h.update(extra or {})
        return h

    def _req(self, method: str, url: str, **kw) -> requests.Response:
        for attempt in range(6):
            try:
                r = self.http.request(method, url, **kw)
            except (requests.ConnectionError, requests.Timeout):
                self.wait_until_awake()   # connection dropped while the workspace suspends / resumes
                continue
            if r.status_code in (502, 503, 504) or re.search(r"starting workspace|is starting", r.text[:300], re.I):
                self.wait_until_awake()
                continue
            if r.status_code == 401 and attempt == 0:
                self._token = None
                kw["headers"]["Authorization"] = f"Bearer {self.token()}"
                continue
            return r
        return r

    # -- api --------------------------------------------------------------
    def gsql(self, statements: str, timeout: int = 1800) -> str:
        ts, t0 = _now_iso(), time.time()
        first = next((l.strip() for l in statements.splitlines() if l.strip() and not l.strip().startswith("//")), "")
        r = self._req("POST", f"{self.base}/gsql/v1/statements", data=statements.encode("utf-8"),
                      headers=self._h({"Content-Type": "text/plain"}), timeout=timeout)
        if r.status_code >= 400:
            log_call("gsql", "POST /gsql/v1/statements", {"first_line": first[:120]}, 0,
                     (time.time() - t0) * 1000, "gsql", ts, f"HTTP {r.status_code}")
            raise TigerGraphError(f"GSQL HTTP {r.status_code}: {r.text[:1000]}")
        log_call("gsql", "POST /gsql/v1/statements", {"first_line": first[:120]}, 0, (time.time() - t0) * 1000,
                 "gsql", ts)
        return r.text

    def query(self, name: str, params: dict[str, Any] | None = None, timeout: int = 120) -> list[dict]:
        ts, t0 = _now_iso(), time.time()
        try:
            out = self._query(name, params, timeout)
        except Exception as e:
            log_call("query", name, params or {}, 0, (time.time() - t0) * 1000, self._last_via, ts, str(e)[:300])
            raise
        log_call("query", name, params or {}, _rows(out), (time.time() - t0) * 1000, self._last_via, ts)
        return out

    _last_via = ""

    def _query(self, name: str, params: dict[str, Any] | None = None, timeout: int = 120) -> list[dict]:
        """Run an installed GSQL query. TG_TRANSPORT=mcp routes it through the TigerGraph MCP server
        (agr.mcp_bridge); on any MCP failure, or with the default TG_TRANSPORT=rest, RESTPP is used."""
        params = params or {}
        if os.getenv("TG_TRANSPORT", "rest").lower() == "mcp" and self._mcp_ready():
            t0 = time.time()
            try:
                with self._mcp_lock:
                    out = self._mcp.run_installed_query(name, params)
                self.calls += 1
                _local.vias = getattr(_local, "vias", []) + ["mcp"]
                self._last_via = "mcp"
                _mcp_log(self._mcp, name, params, out, time.time() - t0)
                return out
            except Exception as e:  # fall back to RESTPP, and say so in the trace
                _local.vias = getattr(_local, "vias", []) + [f"rest(fallback: {type(e).__name__})"]
                self._last_via = f"rest(fallback: {type(e).__name__})"
                _mcp_log(self._mcp, name, params, None, time.time() - t0, error=e)
        else:
            _local.vias = getattr(_local, "vias", []) + ["rest"]
            self._last_via = "rest"
        return self._rest_query(name, params, timeout)

    _mcp = None
    _mcp_failed = False
    _mcp_lock = threading.Lock()
    _mcp_init_lock = threading.Lock()

    def _mcp_ready(self) -> bool:
        if self._mcp is None and not self._mcp_failed:
            with TG._mcp_init_lock:  # one MCP server per process, even with several worker threads
                if self._mcp is None and not self._mcp_failed:
                    self._start_mcp()
        return self._mcp is not None

    def _start_mcp(self):
        if True:
            try:
                from agr.mcp_bridge import TigerGraphMCP
                TG._mcp = TigerGraphMCP(token=self.token())
            except Exception as e:
                TG._mcp_failed = True
                print(f"[tigergraph] MCP unavailable, using RESTPP: {e}", flush=True)

    def _rest_query(self, name: str, params: dict[str, Any], timeout: int = 120) -> list[dict]:
        url = f"{self.base}/restpp/query/{self.graph}/{name}"
        self.calls += 1
        if any(isinstance(v, (list, tuple)) for v in params.values()):
            r = self._req("POST", url, data=json.dumps(params),
                          headers=self._h({"Content-Type": "application/json"}), timeout=timeout)
        else:
            q = {k: (str(v).lower() if isinstance(v, bool) else v) for k, v in params.items()}
            qs = urllib.parse.urlencode(q, quote_via=urllib.parse.quote)
            r = self._req("GET", f"{url}?{qs}" if qs else url, headers=self._h(), timeout=timeout)
        if r.status_code >= 400:
            raise TigerGraphError(f"query {name} HTTP {r.status_code}: {r.text[:500]}")
        body = r.json()
        if body.get("error"):
            raise TigerGraphError(f"query {name}: {body.get('message')}")
        return body.get("results", [])

    def upsert(self, vertices: dict | None = None, edges: dict | None = None) -> dict:
        payload = {k: v for k, v in (("vertices", vertices), ("edges", edges)) if v}
        ts, t0 = _now_iso(), time.time()
        r = self._req("POST", f"{self.base}/restpp/graph/{self.graph}", data=json.dumps(payload),
                      headers=self._h({"Content-Type": "application/json"}), timeout=600)
        if r.status_code >= 400:
            raise TigerGraphError(f"upsert HTTP {r.status_code}: {r.text[:500]}")
        body = r.json()
        if body.get("error"):
            raise TigerGraphError(f"upsert: {body.get('message')}")
        res = body.get("results", [{}])[0]
        log_call("upsert", f"POST /restpp/graph/{self.graph}",
                 {"vertex_types": sorted((vertices or {}).keys()), "edge_source_types": sorted((edges or {}).keys())},
                 int(res.get("accepted_vertices", 0)) + int(res.get("accepted_edges", 0)), (time.time() - t0) * 1000,
                 "rest", ts)
        return res

    def count(self, vtype: str) -> int:
        r = self._req("GET", f"{self.base}/restpp/graph/{self.graph}/vertices/{vtype}?count_only=true",
                      headers=self._h(), timeout=60)
        return int(r.json()["results"][0]["count"])
