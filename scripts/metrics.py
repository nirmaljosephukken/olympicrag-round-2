"""
metrics.json from the answer files, with the fields of the Round 2 brief.

    python scripts/metrics.py [--public answers_public.jsonl] [--hidden answers_hidden.jsonl]
                              [--score results/score_public.json] [--out metrics.json]

Every number except public accuracy is recomputed from the per-question values in the answer files:
  * tokens: sums of input_tokens / output_tokens; tokens_per_question_avg = (input + output) / questions
  * latency: mean, p50 and p95 of latency_ms (nearest-rank percentiles)
  * llm_calls: trace steps that carry input_tokens; tigergraph_queries: trace steps with tigergraph == true
  * answers_with_tigergraph_step: answers whose trace has at least one TigerGraph step
  * total_run_time_s: sum of latency_ms / 1000 (rate-limit waits happen between questions and are not included)
  * cost_usd_total: tokens x LLM_PRICE_IN_PER_M / LLM_PRICE_OUT_PER_M (USD per million tokens, from .env; 0 on a
    free tier)
Public accuracy comes from scripts/score_public.py (its method is copied into accuracy_method); hidden accuracy is null.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agr.config import SETTINGS  # noqa: E402  (loads .env)


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0
    s = sorted(xs)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def split_metrics(path: Path, accuracy: float | None, method: str | None) -> dict:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = len(rows)
    tin = sum(r["input_tokens"] for r in rows)
    tout = sum(r["output_tokens"] for r in rows)
    lat = [r["latency_ms"] for r in rows]
    llm = sum(1 for r in rows for s in r["trace"] if "input_tokens" in s)
    tgq = sum(1 for r in rows for s in r["trace"] if s.get("tigergraph"))
    steps = sum(len(r["trace"]) for r in rows)
    pin, pout = float(os.getenv("LLM_PRICE_IN_PER_M", "0")), float(os.getenv("LLM_PRICE_OUT_PER_M", "0"))
    out = {"questions": n, "accuracy_pct": accuracy}
    if method is not None:
        out["accuracy_method"] = method
    out.update({
        "not_in_corpus_answers": sum(1 for r in rows if r["status"] == "not_in_corpus"),
        "input_tokens_total": tin, "output_tokens_total": tout,
        "tokens_per_question_avg": round((tin + tout) / n) if n else 0,
        "latency_ms_avg": round(sum(lat) / n) if n else 0,
        "latency_ms_p50": pct(lat, 50), "latency_ms_p95": pct(lat, 95),
        "llm_calls_per_question_avg": round(llm / n, 2) if n else 0,
        "tigergraph_queries_per_question_avg": round(tgq / n, 2) if n else 0,
        "trace_steps_per_question_avg": round(steps / n, 2) if n else 0,
        "answers_with_tigergraph_step": sum(1 for r in rows if any(s.get("tigergraph") for s in r["trace"])),
        "total_run_time_s": round(sum(lat) / 1000, 1),
        "cost_usd_total": round(tin / 1e6 * pin + tout / 1e6 * pout, 4),
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default=str(ROOT / "answers_public.jsonl"))
    ap.add_argument("--hidden", default=str(ROOT / "answers_hidden.jsonl"))
    ap.add_argument("--score", default=str(ROOT / "results/score_public.json"))
    ap.add_argument("--out", default=str(ROOT / "metrics.json"))
    a = ap.parse_args()
    acc, method, checks = None, "not scored yet: run scripts/score_public.py", None
    if Path(a.score).exists():
        s = json.loads(Path(a.score).read_text(encoding="utf-8"))["summary"]
        acc, method = s["accuracy_pct"], s["method"]
        checks = {k: s.get(k) for k in ("citation_precision_avg", "citation_recall_avg", "citation_method",
                                        "grounding_report", "tokens_per_correct_answer")}
    out = {}
    if Path(a.public).exists():
        out["public"] = split_metrics(Path(a.public), acc, method)
        if checks:
            out["public_checks"] = checks   # from scripts/score_public.py (needs the published answers)
    if Path(a.hidden).exists():
        out["hidden"] = split_metrics(Path(a.hidden), None, None)
    out["settings"] = {"llm_provider": SETTINGS.llm_provider, "llm_model": SETTINGS.model, "tg_graph": SETTINGS.tg_graph,
                       "max_agent_steps": SETTINGS.max_agent_steps,
                       "price_usd_per_m_tokens": {"input": float(os.getenv("LLM_PRICE_IN_PER_M", "0")),
                                                  "output": float(os.getenv("LLM_PRICE_OUT_PER_M", "0"))}}
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
