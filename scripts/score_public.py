"""
Score answers_public.jsonl against the published answers, fact by fact with partial credit.

    python scripts/score_public.py [answers_public.jsonl] [--questions data/questions/questions_public.jsonl]
                                   [--judge] [--out results/score_public.json]

Method (deterministic, no LLM, the default):
  * gold `not_in_corpus`: 1 if our status is not_in_corpus, else 0 (an invented answer scores 0).
  * otherwise, if we answered not_in_corpus: 0.
  * otherwise the gold answer is split into facts and each fact is looked for in our `answer` text (normalised:
    lower case, accents and punctuation removed; numbers must match as whole numbers):
      - a row (object inside a list, e.g. {athlete, event, years}) is one fact; its credit is the share of its values
        found close together (within 250 characters of the row's most specific value);
      - a scalar or list element (name, NOC, number) is one fact, found or not;
      - a long free-text gold answer ("why" questions) is scored by content-word recall (credit = min(1, recall/0.7)).
    For gold `partial` answers the status itself is one more fact (our status must be `partial`).
  * question score = mean credit of its facts; accuracy_pct = mean question score x 100.
  Number words in the answer count as digits ('eight' = 8).
  Extra wrong list items are not penalised by this matcher (the official scorer does penalise them).
Also reported: citation precision / recall of the cited doc_ids against gold_doc_ids, the grounding verdict of each
answer (from its trace), accuracy per answer style (from the shape of the published answer, not the question
wording) and tokens per correct answer.
With --judge, an LLM judge (same provider as the system) re-scores questions whose deterministic score is below 1,
and the higher of the two scores is kept.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agr.parse import norm  # noqa: E402

STOP = set("a an the of in at on to for and or but by with from as is was were be been it its this that these those "
           "he she they his her their who whom which what why how not no did does do had has have than then so "
           "because after before during only also both all any one two".split())


def _txt(v) -> str:
    return norm(str(v))


UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "         "seventeen eighteen nineteen".split()
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def digits(text: str) -> str:
    """Number words to digits in a normalised answer ('eight' -> '8', 'twenty two' -> '22'), so a count written
    in words is matched like one written in digits."""
    w, out, i = text.split(), [], 0
    while i < len(w):
        if w[i] in TENS:
            n = TENS[w[i]]
            if i + 1 < len(w) and w[i + 1] in UNITS[1:10]:
                n += UNITS.index(w[i + 1]); i += 1
            out.append(str(n))
        elif w[i] in UNITS:
            out.append(str(UNITS.index(w[i])))
        else:
            out.append(w[i])
        i += 1
    return " ".join(out)


def _found(value, text_n: str) -> list[int]:
    """Positions where a gold value occurs in the normalised answer text."""
    v = _txt(value)
    if not v:
        return []
    return [m.start() for m in re.finditer(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", text_n)]


def _leaves(x) -> list:
    if isinstance(x, dict):
        return [l for v in x.values() for l in _leaves(v)]
    if isinstance(x, (list, tuple)):
        return [l for v in x for l in _leaves(v)]
    return [] if x is None else [x]


def row_credit(row: dict, text_n: str, window: int = 250) -> float:
    vals = [v for v in _leaves(row) if _txt(v)]
    if not vals:
        return 1.0
    anchor = max(vals, key=lambda v: (not isinstance(v, (int, float)), len(_txt(v))))
    best = 0
    for p in _found(anchor, text_n):
        near = text_n[max(0, p - window): p + window]
        best = max(best, 1 + sum(1 for v in vals if v is not anchor and _found(v, near)))
    return best / len(vals)


def text_credit(gold: str, text_n: str) -> float:
    words = [w for w in _txt(gold).split() if w not in STOP and len(w) > 1]
    if not words:
        return 0.0
    have = set(text_n.split())
    recall = sum(1 for w in words if w in have) / len(words)
    return min(1.0, recall / 0.7)


def facts(gold, text_n: str) -> list[tuple[str, float]]:
    """(fact label, credit) pairs for one gold answer."""
    out = []
    if isinstance(gold, list):
        for x in gold:
            if isinstance(x, dict):
                out.append((json.dumps(x, ensure_ascii=False), row_credit(x, text_n)))
            else:
                out += facts(x, text_n)
    elif isinstance(gold, dict):
        for k, v in gold.items():
            if k == "not_in_corpus":   # partial answers: the missing part must be named as missing
                for m in (v if isinstance(v, list) else [v]):
                    out.append((f"missing:{m}", 1.0 if _found(m, text_n) else 0.0))
            elif v is None:
                continue
            elif isinstance(v, (list, dict)):
                out += facts(v, text_n)
            else:
                out.append((f"{k}={v}", 1.0 if _found(v, text_n) else 0.0))
    elif isinstance(gold, str) and len(gold.split()) > 4:
        out.append(("text", text_credit(gold, text_n)))
    elif gold is not None:
        out.append((str(gold), 1.0 if _found(gold, text_n) else 0.0))
    return out


def score_one(q: dict, a: dict | None) -> dict:
    if a is None:
        return {"score": 0.0, "facts": [], "why": "no answer"}
    gs, s = q.get("status"), a.get("status")
    if gs == "not_in_corpus":
        return {"score": 1.0 if s == "not_in_corpus" else 0.0, "facts": [], "why": f"gold not_in_corpus, ours {s}"}
    if s == "not_in_corpus":
        return {"score": 0.0, "facts": [], "why": "answered not_in_corpus"}
    ans = a.get("answer")
    text_n = digits(_txt(ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False)))
    fs = facts(q.get("answer"), text_n)
    if gs == "partial":
        fs.append(("status=partial", 1.0 if s == "partial" else 0.0))
    score = sum(c for _, c in fs) / len(fs) if fs else 0.0
    return {"score": round(score, 3), "facts": [[f, round(c, 2)] for f, c in fs], "why": ""}


def judge_one(q: dict, a: dict) -> float | None:
    from agr.config import SETTINGS
    from agr.llm import generate
    prompt = (f"Grade a system answer against the gold answer, fact by fact with partial credit. Lists must be "
              f"complete; wrong extra items lower the score. Statuses: gold {q.get('status')}, system "
              f"{a.get('status')}.\nQuestion: {q['question']}\nGold answer: {json.dumps(q.get('answer'), ensure_ascii=False)}"
              f"\nGold note: {q.get('note', '')}\nSystem answer: {a.get('answer')}\n\n"
              f"Return JSON: {{\"score\": number between 0 and 1, \"reason\": str}}")
    r = generate(prompt, model=SETTINGS.judge_model, json_mode=True, max_output_tokens=300)
    try:
        return max(0.0, min(1.0, float(r.json().get("score"))))
    except (TypeError, ValueError):
        return None


def answer_style(q: dict) -> str:
    """A question's style, read from the SHAPE of its published answer (never from its wording): not_in_corpus,
    partial, trend (rows per edition), list (several rows), count / comparison (a dict of values), explanation (a
    sentence), short factual (a name or number)."""
    a, st = q.get("answer"), q.get("status")
    if st == "not_in_corpus":
        return "not_in_corpus"
    if st == "partial":
        return "partial"
    if isinstance(a, dict) and isinstance(a.get("rows"), list):
        return "trend across editions"
    if isinstance(a, list) and len(a) > 1:
        return "list"
    if isinstance(a, dict):
        return "count / comparison"
    if isinstance(a, str) and len(a.split()) > 6:
        return "explanation"
    return "short factual"


def citation_scores(q: dict, a: dict | None) -> tuple[float | None, float | None]:
    """Precision and recall of the cited doc_ids against the published gold_doc_ids (None when the question has no
    gold documents, e.g. not_in_corpus)."""
    gold = set(q.get("gold_doc_ids") or [])
    if not gold or a is None:
        return None, None
    cited = set(a.get("citations") or [])
    hit = len(cited & gold)
    return (round(hit / len(cited), 3) if cited else 0.0), round(hit / len(gold), 3)


def grounding_verdict(a: dict | None) -> str:
    """From the trace: was the released answer checked as fully supported by cited evidence?"""
    if a is None:
        return "no answer"
    if a.get("status") == "not_in_corpus":
        return "not_in_corpus"
    checks = [s for s in a.get("trace", []) if s.get("tool") == "evidence_evaluator"]
    if checks and checks[-1].get("ok"):
        return "supported by cited evidence"
    return "not checked" if not checks else "failed check"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", nargs="?", default=str(root / "answers_public.jsonl"))
    ap.add_argument("--questions", default=str(root / "data/questions/questions_public.jsonl"))
    ap.add_argument("--out", default=str(root / "results/score_public.json"))
    ap.add_argument("--judge", action="store_true")
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8") if l.strip()]
    ans = {}
    for l in open(a.answers, encoding="utf-8"):
        if l.strip():
            x = json.loads(l)
            ans[x["qid"]] = x
    rows = []
    for q in qs:
        r = score_one(q, ans.get(q["qid"]))
        r["deterministic"] = r["score"]
        if a.judge and q["qid"] in ans and r["score"] < 1 and q.get("status") != "not_in_corpus" \
                and ans[q["qid"]].get("status") != "not_in_corpus":
            j = judge_one(q, ans[q["qid"]])
            r["judge"] = j
            if j is not None:
                r["score"] = max(r["score"], round(j, 3))
        a_ = ans.get(q["qid"])
        prec, rec = citation_scores(q, a_)
        rows.append({"qid": q["qid"], "style": answer_style(q), "gold_status": q.get("status"),
                     "status": (a_ or {}).get("status"), **r, "citation_precision": prec, "citation_recall": rec,
                     "grounding": grounding_verdict(a_),
                     "tokens": ((a_ or {}).get("input_tokens", 0) + (a_ or {}).get("output_tokens", 0))})
    acc = 100 * sum(r["score"] for r in rows) / len(rows)
    full = sum(1 for r in rows if r["score"] >= 0.999)
    status_ok = sum(1 for r in rows if r["status"] == r["gold_status"])
    method = ("deterministic fact-by-fact matching with partial credit (scripts/score_public.py)" +
              ("; LLM judge re-scores answers below full credit, higher score kept" if a.judge else ""))
    styles = {}
    for r in rows:
        styles.setdefault(r["style"], []).append(r)
    by_style = {s: {"questions": len(rs), "accuracy_pct": round(100 * sum(r["score"] for r in rs) / len(rs), 1),
                    "failing": [r["qid"] for r in rs if r["score"] < 0.999]} for s, rs in sorted(styles.items())}
    grounding = {}
    for r in rows:
        grounding[r["grounding"]] = grounding.get(r["grounding"], 0) + 1
    total_tokens = sum(r["tokens"] for r in rows)
    correct = sum(r["score"] for r in rows)
    summary = {"questions": len(rows), "answered_files": len(ans), "accuracy_pct": round(acc, 1),
               "full_credit": full, "status_matches": status_ok, "method": method,
               "citation_precision_avg": mean(r["citation_precision"] for r in rows),
               "citation_recall_avg": mean(r["citation_recall"] for r in rows),
               "citation_method": "cited doc_ids vs gold_doc_ids, averaged over questions that have gold documents",
               "grounding_report": grounding,
               "tokens_per_correct_answer": round(total_tokens / correct) if correct else None,
               "accuracy_by_style": by_style}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "questions": rows}, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(f"{'qid':<11} {'style':<22} {'gold':<13} {'ours':<13} score  cite P/R")
    for r in rows:
        pr = "" if r["citation_precision"] is None else f"{r['citation_precision']:.2f}/{r['citation_recall']:.2f}"
        print(f"{r['qid']:<11} {r['style']:<22} {r['gold_status'] or '':<13} {r['status'] or '-':<13} "
              f"{r['score']:.2f}  {pr}" + (f"  ({r['why']})" if r["why"] else ""))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
