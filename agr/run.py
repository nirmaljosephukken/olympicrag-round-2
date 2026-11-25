"""
Run the agentic pipeline on every question of a questions file and write Round 2 answer lines.

    python -m agr.run data/questions/questions_public.jsonl answers_public.jsonl   (what ./run.sh calls)

Each output line: qid, status (answered | partial | not_in_corpus), answer, explanation, citations (doc_ids),
input_tokens, output_tokens, latency_ms and trace (one entry per TigerGraph call, LLM call and code step, written as the
question runs). Every TigerGraph call is also appended to logs/tigergraph_queries.jsonl with an ISO timestamp.

Rate limiting happens BEFORE a question's clock starts (agr.llm.wait_for_budget), so free-tier waits are not part of
latency_ms. The run is resumable: questions already in the answers file are skipped (--fresh starts over).
A full per-question record (steps, notes, grounding) goes to <answers>.full.jsonl next to the answers file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from agr import llm, tg
from agr.pipelines import PIPELINES, run_agentic
from agr.tools import Toolbox

NOT_FOUND = {"", "unknown", "none", "n/a", "not in corpus", "not_in_corpus"}


def status_of(answer: str) -> str:
    return "not_in_corpus" if answer.strip().lower().strip(".") in NOT_FOUND else "answered"


def citations_of(r) -> list[str]:
    out = []
    for c in r.citations:
        for d in str(c.get("doc_id", "")).split(","):
            d = d.strip()
            if d and d not in out:
                out.append(d)
    return out


def explanation_of(r) -> str:
    parts = ([r.explanation] if r.explanation else []) + \
        ([f"Not in the corpus: {'; '.join(r.missing)}"] if r.missing else []) + list(r.notes) + list(r.reasoning_chain)
    text = " | ".join(parts) if parts else r.stop_reason
    if r.stop_reason and not r.stop_reason.startswith("evidence_sufficient"):
        text += f" | stop: {r.stop_reason}"
    return text[:1500]


def answer_line(r) -> dict:
    status = r.status or status_of(r.answer)
    line = {"qid": r.qid, "status": status, "answer": None if status == "not_in_corpus" else r.answer,
            "explanation": explanation_of(r), "citations": citations_of(r),
            "input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "latency_ms": int(round(r.latency_ms)),
            "trace": r.trace}
    tin = sum(s.get("input_tokens", 0) for s in r.trace)
    tout = sum(s.get("output_tokens", 0) for s in r.trace)
    assert (tin, tout) == (r.input_tokens, r.output_tokens), f"{r.qid}: trace tokens {tin}/{tout} != totals"
    return line


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("answers")
    ap.add_argument("--fresh", action="store_true", help="overwrite the answers file instead of resuming")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pipeline", default="agentic", choices=["agentic", "rag", "graphrag"],
                    help="agentic (the submitted system) or a one-call baseline for comparison")
    a = ap.parse_args()

    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8") if l.strip()]
    out, full = Path(a.answers), Path(a.answers).with_suffix(".full.jsonl")
    if a.fresh:
        for p in (out, full):
            p.unlink(missing_ok=True)
    done = {json.loads(l)["qid"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [q for q in qs if q["qid"] not in done][: a.limit or None]
    print(f"[run] {len(qs)} questions, {len(done)} already answered, {len(todo)} to run", flush=True)

    tb = Toolbox()
    # warm-up before any clock starts: wake the workspace if it auto-suspended, then load the catalogue
    waited = tb.tg.wait_until_awake(max_wait=float(__import__("os").getenv("TG_WAKE_MAX_S", "900")))
    print(f"[run] TigerGraph workspace answering (waited {waited:.0f} s)", flush=True)
    tb.catalog()
    failed = 0
    t_run = time.time()
    for i, q in enumerate(todo, 1):
        waited = llm.wait_for_budget()
        tg.CONTEXT["qid"] = q["qid"]
        llm.WAITS["in_question_s"] = 0.0
        try:
            r = PIPELINES[a.pipeline](tb, q["qid"], q["question"])
        except RuntimeError as e:
            if "daily quota" in str(e):
                print(f"[run] stopping: {e}. Re-run the same command later to resume.", flush=True)
                return 2
            if isinstance(e, tg.TigerGraphError) or "workspace" in str(e).lower():
                try:   # the workspace went to sleep mid-run: wake it and retry this question once
                    tb.tg.wait_until_awake()
                    llm.wait_for_budget()
                    r = run_agentic(tb, q["qid"], q["question"]) if a.pipeline == "agentic" else \
                        PIPELINES[a.pipeline](tb, q["qid"], q["question"])
                except RuntimeError as e2:
                    e = e2
                else:
                    e = None
            if e is not None:
                print(f"[run] {q['qid']} FAILED: {e}", flush=True)
                failed += 1
                continue
        finally:
            tg.CONTEXT["qid"] = ""
        line = answer_line(r)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        with open(full, "a", encoding="utf-8") as f:
            f.write(json.dumps({**r.to_dict(), "rate_wait_before_s": round(waited, 2),
                                "rate_wait_inside_s": round(llm.WAITS["in_question_s"], 2)}, ensure_ascii=False) + "\n")
        tgs = sum(1 for s in r.trace if s["tigergraph"])
        print(f"[run] {i}/{len(todo)} {q['qid']} {line['status']:<13} {line['latency_ms']:>6} ms "
              f"{r.input_tokens + r.output_tokens:>6} tok {tgs} TG  {str(line['answer'])[:70]!r}", flush=True)
    print(f"[run] finished in {time.time() - t_run:.0f} s; failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
