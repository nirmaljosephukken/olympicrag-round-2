"""
Extract the TigerGraph query log of one run: the lines of logs/tigergraph_queries.jsonl that belong to the questions
of an answers file and fall inside that run's time window (first to last traced TigerGraph call), and check that every
traced TigerGraph step appears in the log.

    python scripts/extract_query_log.py answers_public.jsonl logs/tigergraph_queries_public.jsonl

The full log keeps every call ever made (builds, development runs); this gives the log for one submitted run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    answers, out = Path(sys.argv[1]), Path(sys.argv[2])
    log = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "logs" / "tigergraph_queries.jsonl"
    rows = [json.loads(l) for l in open(answers, encoding="utf-8") if l.strip()]
    traced = {(r["qid"], s["ts"], s["query"]) for r in rows for s in r["trace"] if s.get("tigergraph") and s.get("ts")}
    if not traced:
        print("no traced TigerGraph steps with timestamps in", answers)
        return 1
    qids = {q for q, _, _ in traced}
    first, last = min(t for _, t, _ in traced), max(t for _, t, _ in traced)
    keep, found = [], set()
    for l in open(log, encoding="utf-8"):
        if not l.strip():
            continue
        rec = json.loads(l)
        if rec.get("qid") in qids and first <= rec.get("ts", "") <= last:
            keep.append(l if l.endswith("\n") else l + "\n")
            found.add((rec["qid"], rec["ts"], rec["query"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(keep), encoding="utf-8")
    missing = traced - found
    print(f"{len(keep)} log lines for {len(qids)} questions ({first} .. {last}) -> {out}")
    print(f"traced TigerGraph steps: {len(traced)}; found in the log: {len(traced) - len(missing)}; missing: {len(missing)}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
