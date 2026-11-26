"""
Re-run submitted questions live and compare with the submitted answers (for the judging call).

    python scripts/reproduce.py answers_hidden.jsonl QID [QID ...] [--questions data/questions/questions_hidden.jsonl]

Each question is answered again by the same system that ./run.sh runs (agr.run), into results/reproduce/, and the new
answer is printed next to the submitted one: status, answer, citations, and the sequence of tools in the trace, with
the TigerGraph queries and their parameters. The LLM runs at temperature 0 with a fixed seed, but a free-tier model is
not perfectly deterministic, so small wording differences are expected; status, cited documents and the queries sent
to TigerGraph are the things to compare.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(p: Path) -> dict:
    return {r["qid"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}


def tg_steps(a: dict) -> list[str]:
    return [f"{s.get('query')}({json.dumps(s.get('params', {}), ensure_ascii=False)}) -> {s.get('rows')} rows"
            for s in a.get("trace", []) if s.get("tigergraph")]


def show(label: str, a: dict) -> None:
    print(f"  {label:9} status {a['status']}")
    print(f"  {'':9} answer {str(a.get('answer'))[:300]}")
    print(f"  {'':9} citations {', '.join(a.get('citations', []))}")
    print(f"  {'':9} tools {' > '.join(s['tool'] for s in a.get('trace', []))}")
    for t in tg_steps(a):
        print(f"  {'':9} TG {t}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", help="the submitted answers file")
    ap.add_argument("qids", nargs="+")
    ap.add_argument("--questions", default="")
    a = ap.parse_args()
    sub = load(Path(a.answers))
    qfile = Path(a.questions) if a.questions else ROOT / "data" / "questions" / (
        "questions_hidden.jsonl" if "hidden" in Path(a.answers).name else "questions_public.jsonl")
    qs = load(qfile)
    missing = [q for q in a.qids if q not in qs or q not in sub]
    if missing:
        print(f"not in {qfile.name} or {Path(a.answers).name}: {', '.join(missing)}")
        return 1
    out_dir = ROOT / "results" / "reproduce"
    out_dir.mkdir(parents=True, exist_ok=True)
    qtmp, atmp = out_dir / "questions.jsonl", out_dir / "answers.jsonl"
    qtmp.write_text("".join(json.dumps(qs[q], ensure_ascii=False) + "\n" for q in a.qids), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "agr.run", str(qtmp), str(atmp), "--fresh"], cwd=ROOT, check=True)
    new = load(atmp)
    same = 0
    for q in a.qids:
        print(f"\n{q}: {qs[q]['question']}")
        show("submitted", sub[q])
        show("now", new[q])
        ok = new[q]["status"] == sub[q]["status"] and set(new[q]["citations"]) == set(sub[q]["citations"])
        same += ok
        print(f"  -> {'same status and citations' if ok else 'DIFFERENT status or citations'}")
    print(f"\n{same} of {len(a.qids)} reproduced with the same status and citations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
