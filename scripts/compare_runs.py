"""
Repeatability: compare two independent runs of the same questions, question by question.

    python scripts/compare_runs.py RUN_A.jsonl RUN_B.jsonl [--questions data/questions/questions_public.jsonl]
                                   [--out results/compare_runs.json]

For each question: same status? same answer (normalised text)? same citations? and, when the questions file has
published answers, the score of each run (scripts/score_public.py). Prints a table and a summary; writes JSON.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agr.parse import norm  # noqa: E402


def load(path: str) -> dict:
    return {json.loads(l)["qid"]: json.loads(l) for l in open(path, encoding="utf-8") if l.strip()}


def scorer():
    spec = importlib.util.spec_from_file_location("score_public", ROOT / "scripts" / "score_public.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def compare(a: dict, b: dict, qs: dict | None = None) -> dict:
    s = scorer() if qs else None
    rows = []
    for qid in sorted(set(a) | set(b)):
        x, y = a.get(qid), b.get(qid)
        row = {"qid": qid, "in_both": bool(x and y)}
        if x and y:
            row.update({"same_status": x["status"] == y["status"],
                        "same_answer": norm(str(x.get("answer") or "")) == norm(str(y.get("answer") or "")),
                        "same_citations": set(x.get("citations") or []) == set(y.get("citations") or []),
                        "status_a": x["status"], "status_b": y["status"]})
            if s and qid in qs and "answer" in qs[qid]:
                row["score_a"] = s.score_one(qs[qid], x)["score"]
                row["score_b"] = s.score_one(qs[qid], y)["score"]
        rows.append(row)
    both = [r for r in rows if r["in_both"]]
    summary = {"questions": len(rows), "in_both": len(both),
               "same_status": sum(r["same_status"] for r in both),
               "same_answer_text": sum(r["same_answer"] for r in both),
               "same_citations": sum(r["same_citations"] for r in both)}
    scored = [r for r in both if "score_a" in r]
    if scored:
        summary.update({"accuracy_a_pct": round(100 * sum(r["score_a"] for r in scored) / len(scored), 1),
                        "accuracy_b_pct": round(100 * sum(r["score_b"] for r in scored) / len(scored), 1),
                        "same_score": sum(abs(r["score_a"] - r["score_b"]) < 0.01 for r in scored),
                        "changed_score": [r["qid"] for r in scored if abs(r["score_a"] - r["score_b"]) >= 0.01]})
    return {"summary": summary, "questions": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--questions", default=str(ROOT / "data/questions/questions_public.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results/compare_runs.json"))
    a = ap.parse_args()
    qs = load(a.questions) if a.questions and Path(a.questions).exists() else None
    res = compare(load(a.run_a), load(a.run_b), qs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in res["questions"]:
        if r["in_both"] and not (r["same_status"] and r["same_answer"]):
            sc = f"  score {r['score_a']:.2f} -> {r['score_b']:.2f}" if "score_a" in r else ""
            print(f"{r['qid']:<11} status {r['status_a']} / {r['status_b']}; answer "
                  f"{'same' if r['same_answer'] else 'differs'}{sc}")
    print(json.dumps(res["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
