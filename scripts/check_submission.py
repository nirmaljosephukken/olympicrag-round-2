"""
Check answer files against the submission rules of the Round 2 brief, before they are sent.

    python scripts/check_submission.py answers_public.jsonl [answers_hidden.jsonl ...]
                                       [--log logs/tigergraph_queries.jsonl] [--metrics metrics.json]

For every answer file:
  * every field is present and filled in (qid, status, answer, explanation, citations, tokens, latency_ms, trace);
    `answer` may be null only for not_in_corpus
  * status is answered / partial / not_in_corpus; no qid appears twice
  * every trace step has step (1, 2, 3, ...), tool, tigergraph and ms; TigerGraph steps also have query, params, rows
  * the LLM steps' tokens add up exactly to the answer's input_tokens / output_tokens
  * the step times roughly add up to latency_ms
  * every TigerGraph step is in the TigerGraph query log (same qid and query, timestamp within 5 s)
  * the numbers in metrics.json match the values recomputed from the answer file (scripts/metrics.py)
Prints one line per problem and a summary; exit code 1 if anything fails.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ("qid", "status", "answer", "explanation", "citations", "input_tokens", "output_tokens", "latency_ms", "trace")
STATUSES = {"answered", "partial", "not_in_corpus"}


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_log(p: Path) -> dict:
    idx = defaultdict(list)                       # (qid, query) -> [timestamps]
    if p.exists():
        for l in open(p, encoding="utf-8"):
            if l.strip():
                e = json.loads(l)
                if e.get("ts"):
                    idx[(e.get("qid", ""), e.get("query", ""))].append(ts(e["ts"]))
    return idx


def check_file(path: Path, log: dict) -> list[str]:
    bad, seen = [], set()
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for a in rows:
        q = a.get("qid", "?")
        for f in FIELDS:
            if f not in a:
                bad.append(f"{q}: missing field {f}")
        if q in seen:
            bad.append(f"{q}: appears twice")
        seen.add(q)
        if a.get("status") not in STATUSES:
            bad.append(f"{q}: status {a.get('status')!r}")
        if a.get("answer") in (None, "") and a.get("status") != "not_in_corpus":
            bad.append(f"{q}: empty answer with status {a.get('status')}")
        if not a.get("explanation"):
            bad.append(f"{q}: empty explanation")
        if not isinstance(a.get("citations"), list) or (a.get("status") != "not_in_corpus" and not a.get("citations")):
            bad.append(f"{q}: no citations")
        tr = a.get("trace") or []
        if not tr:
            bad.append(f"{q}: empty trace")
            continue
        tin = tout = 0
        for i, s in enumerate(tr, 1):
            for f in ("step", "tool", "tigergraph", "ms"):
                if f not in s:
                    bad.append(f"{q} step {i}: missing {f}")
            if s.get("step") != i:
                bad.append(f"{q}: step numbers not 1, 2, 3, ... at position {i}")
            tin += s.get("input_tokens", 0) or 0
            tout += s.get("output_tokens", 0) or 0
            if s.get("tigergraph"):
                for f in ("query", "params", "rows"):
                    if f not in s:
                        bad.append(f"{q} step {i}: TigerGraph step without {f}")
                if s.get("ts"):
                    near = [t for t in log.get((q, s.get("query", "")), []) if abs(t - ts(s["ts"])) <= 5]
                    if not near:
                        bad.append(f"{q} step {i}: {s.get('query')} at {s['ts']} not in the query log")
                else:
                    bad.append(f"{q} step {i}: TigerGraph step without a timestamp")
        if (tin, tout) != (a.get("input_tokens"), a.get("output_tokens")):
            bad.append(f"{q}: step tokens {tin}/{tout} != answer tokens {a.get('input_tokens')}/{a.get('output_tokens')}")
        ms, lat = sum(s.get("ms", 0) or 0 for s in tr), a.get("latency_ms") or 0
        if lat and not (0.7 * lat <= ms <= 1.05 * lat + 50):
            bad.append(f"{q}: step times add up to {ms:.0f} ms, latency_ms is {lat}")
        if not any(s.get("tigergraph") for s in tr):
            bad.append(f"{q}: (warning) no TigerGraph step")
    print(f"{path}: {len(rows)} answers, {len(bad)} problem(s)")
    return bad


def check_metrics(metrics: Path, files: dict[str, Path]) -> list[str]:
    if not metrics.exists():
        return [f"{metrics}: missing"]
    spec = importlib.util.spec_from_file_location("metrics", ROOT / "scripts" / "metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m, bad = json.loads(metrics.read_text(encoding="utf-8")), []
    for split, path in files.items():
        if split not in m:
            bad.append(f"metrics.json: no '{split}' section")
            continue
        fresh = mod.split_metrics(path, None, None)
        for k, v in fresh.items():
            if k in ("accuracy_pct", "accuracy_method") or v is None:
                continue
            if k in m[split] and m[split][k] != v:
                bad.append(f"metrics.json {split}.{k} = {m[split][k]}, recomputed {v}")
    if "hidden" in m and m["hidden"].get("accuracy_pct") is not None:
        bad.append("metrics.json hidden.accuracy_pct must be null")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", nargs="+")
    ap.add_argument("--log", default=str(ROOT / "logs" / "tigergraph_queries.jsonl"))
    ap.add_argument("--metrics", default="")
    a = ap.parse_args()
    log, bad = load_log(Path(a.log)), []
    for f in a.answers:
        bad += check_file(Path(f), log)
    if a.metrics:
        split = lambda f: "hidden" if "hidden" in Path(f).name else "public"
        bad += check_metrics(Path(a.metrics), {split(f): Path(f) for f in a.answers})
    for b in bad:
        print("  " + b)
    fails = [b for b in bad if "(warning)" not in b]
    print("OK" if not fails else f"FAIL: {len(fails)} problem(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
