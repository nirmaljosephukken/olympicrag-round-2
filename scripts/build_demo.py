"""
Build the data file of the demo page (docs/data.json) from the result files in the repository.

    python scripts/build_demo.py            -> docs/data.json   (docs/index.html is the page; it reads this file)

The page has three parts: what the system is and its measured results, an explorer of every recorded answer (public and
held-out questions, with the published answer, our score, citations and the full trace), and a live box that sends a new
question to a locally running server (python -m agr.web), which runs the same agent against TigerGraph.
No secrets go into the page or the data file: only answers, traces and scores that are already in the repository.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def answers_with_gold(answers: Path, questions: Path, score: Path, label: str) -> list[dict]:
    qs = {q["qid"]: q for q in load_jsonl(questions)}
    sc = {r["qid"]: r for r in (load_json(score) or {}).get("questions", [])}
    full = {r["qid"]: r for r in load_jsonl(answers.with_suffix(".full.jsonl"))}
    out = []
    for a in load_jsonl(answers):
        q = qs.get(a["qid"], {})
        s = sc.get(a["qid"], {})
        out.append({"set": label, "qid": a["qid"], "question": q.get("question", ""), "gold_status": q.get("status"),
                    "gold_answer": q.get("answer"), "gold_docs": q.get("gold_doc_ids", []), "status": a["status"],
                    "answer": a["answer"], "explanation": a["explanation"], "citations": a["citations"],
                    "score": s.get("score"), "style": s.get("style"), "input_tokens": a["input_tokens"],
                    "output_tokens": a["output_tokens"], "latency_ms": a["latency_ms"], "trace": a["trace"],
                    "missing": full.get(a["qid"], {}).get("missing", []),
                    "stop_reason": full.get(a["qid"], {}).get("stop_reason", ""),
                    "grounding": full.get(a["qid"], {}).get("grounding", "")})
    return out


def summary(score: Path, metrics: Path | None = None, answers: Path | None = None) -> dict:
    s = (load_json(score) or {}).get("summary", {})
    m = ((load_json(metrics) or {}).get("public", {}) if metrics else {})
    if not m and answers and answers.exists():   # efficiency figures computed from the answer file itself
        import importlib.util
        spec = importlib.util.spec_from_file_location("metrics", ROOT / "scripts" / "metrics.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        m = mod.split_metrics(answers, None, None)
    keep = ("accuracy_pct", "full_credit", "status_matches", "citation_precision_avg", "citation_recall_avg",
            "tokens_per_correct_answer", "grounding_report", "accuracy_by_style")
    out = {k: s.get(k) for k in keep}
    for k in ("tokens_per_question_avg", "llm_calls_per_question_avg", "tigergraph_queries_per_question_avg",
              "latency_ms_p50", "latency_ms_p95", "latency_ms_avg", "answers_with_tigergraph_step"):
        if k in m:
            out[k] = m[k]
    return out


def main() -> int:
    r = ROOT / "results"
    data = {
        "graph_counts": load_json(ROOT / "graph_counts.json"),
        "runs": [
            {"name": "Public, final run (submitted)", "code": "f7100e4",
             **summary(r / "score_public.json", ROOT / "metrics.json")},
            {"name": "Public, previous final run (45 s LLM timeout)", "code": "50825c5",
             **summary(r / "score_public_a5ebfdc.json", r / "metrics_a5ebfdc.json")},
            {"name": "Public, fast-period run", "code": "06930bd",
             **summary(r / "score_public_06930bd.json", r / "metrics_06930bd.json")},
            {"name": "Public, repeatability run A", "code": "c0c43ec", **summary(r / "score_run_A.json", answers=r / "run_A_public.jsonl")},
            {"name": "Public, repeatability run B", "code": "c0c43ec", **summary(r / "score_run_B.json", answers=r / "run_B_public.jsonl")},
            {"name": "Held-out, frozen code (39 questions, answers computed by code)", "code": "e890a5f",
             **summary(r / "score_heldout_e890a5f.json", answers=r / "heldout_e890a5f.jsonl")},
            {"name": "Held-out, earlier code (39 questions, answers computed by code)", "code": "c0c43ec",
             **summary(r / "score_heldout.json", answers=r / "answers_heldout.jsonl")},
            {"name": "Baseline: RAG (1 LLM call)", "code": "fca70f6",
             **summary(r / "score_baseline_rag.json", r / "metrics_baseline_rag.json")},
            {"name": "Baseline: GraphRAG (1 LLM call)", "code": "fca70f6",
             **summary(r / "score_baseline_graphrag.json", r / "metrics_baseline_graphrag.json")},
        ],
        "heldout_note": "The held-out set is no longer fully untouched: some fixes made after the c0c43ec run (for example citing supporting evidence the answer left uncited) were prompted by held-out misses. The hidden set remains the real test.",
        "repeatability": (load_json(r / "compare_A_B.json") or {}).get("summary"),
        "repeatability_rows": [{k: q.get(k) for k in ("qid", "score_a", "score_b", "status_a", "status_b")}
                               for q in (load_json(r / "compare_A_B.json") or {}).get("questions", [])],
        "answers": answers_with_gold(ROOT / "answers_public.jsonl", ROOT / "data/questions/questions_public.jsonl",
                                     r / "score_public.json", "public")
        + answers_with_gold(r / "heldout_e890a5f.jsonl", ROOT / "data/questions/questions_heldout.jsonl",
                            r / "score_heldout_e890a5f.json", "held-out")
        + answers_with_gold(ROOT / "answers_hidden.jsonl", ROOT / "data/questions/questions_hidden.jsonl",
                            r / "no_score_for_hidden.json", "hidden"),   # no answer key: unscored
    }
    out = ROOT / "docs" / "data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{out}: {len(data['answers'])} answers, {out.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
