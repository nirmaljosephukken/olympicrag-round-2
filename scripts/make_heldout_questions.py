"""
Held-out question set with gold answers computed by code from the parsed corpus (build/*.json), to check that fixes
generalise beyond the 50 public questions. No LLM writes a question or an answer.

    python scripts/make_heldout_questions.py [--n-per-kind 6] [--seed 11]   -> data/questions/questions_heldout.jsonl

Kinds (templates are our own; items are sampled with a fixed seed and avoid every event, athlete and sport-year
the public questions are about):
  winners_years   gold medallist of one event at several Games                         (answered, list)
  winners_gap     same, including a Games whose article for that event is not in the corpus  (partial)
  missing_2028    winner of an event that has a 2024 article but no 2028 article         (not_in_corpus)
  repeat_winners  athletes who won the same event of a sport at 2+ Games                 (answered, list)
  trend           nations per edition of one event, with the most / fewest               (answered, rows)
  medal_leader    nation with the most golds in a sport at one Games (2024, 2026, 2028) (answered, count)
  fewest_nations  event of a sport at one Games with the fewest nations                   (answered, count)
  top_source      for an LA 2028 event with source records, the most authoritative record (answered, short)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agr import crossgames  # noqa: E402
from agr.parse import norm, person_names, CREW_EVENT  # noqa: E402


def load(name):
    return json.loads((ROOT / "build" / name).read_text(encoding="utf-8"))


def public_exclusions() -> tuple[set, set]:
    """Doc ids and names the public questions are about: held-out items never reuse them."""
    docs, text = set(), ""
    for l in open(ROOT / "data/questions/questions_public.jsonl", encoding="utf-8"):
        q = json.loads(l)
        docs |= set(q.get("gold_doc_ids") or [])
        text += " " + norm(q["question"]) + " " + norm(json.dumps(q.get("answer"), ensure_ascii=False))
    return docs, set(text.split()) and {text}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-kind", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(ROOT / "data/questions/questions_heldout.jsonl"))
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    events = load("events.json")
    records = load("records.json")
    pub_docs, (pub_text,) = public_exclusions()
    used = lambda e: e["id"] in pub_docs or norm(e["event"]) in pub_text and norm(e["sport"]) in pub_text
    by_key = defaultdict(dict)          # (sport, event, season) -> year -> event
    for e in events:
        by_key[(e["sport"], e["event"], e["season"])][e["year"]] = e
    games = defaultdict(set)
    for e in events:
        games[e["season"]].add(e["year"])
    out = []

    def add(kind, question, status, answer, docs):
        out.append({"qid": f"ho-{len(out) + 1:03d}", "kind": kind, "question": question, "status": status,
                    "answer": answer, "gold_doc_ids": sorted(set(docs)), "note": "gold computed by code"})

    def label(sport, ev, season):
        return f"the {ev} event in {sport.lower()} at the {season} Olympics"

    # winners_years / winners_gap / trend: events with several editions and a single named winner each year
    solo = lambda e: len(person_names(e["gold"], e["event"])) == 1 and not CREW_EVENT.search(e["event"])
    multi = [(k, ys) for k, ys in by_key.items() if len(ys) >= 4 and not any(used(e) for e in ys.values())]
    rnd.shuffle(multi)
    n = 0
    for (sport, ev, season), ys in multi:
        years = sorted(ys)
        if not all(solo(ys[y]) for y in years):   # individual events with one named winner per Games
            continue
        pick = sorted(rnd.sample(years, 3))
        add("winners_years", f"Name the gold medallist of {label(sport, ev, season)} in {', '.join(map(str, pick))}.",
            "answered", [{"year": y, "gold": ys[y]["gold"]} for y in pick], [ys[y]["id"] for y in pick])
        n += 1
        if n >= a.n_per_kind:
            break
    n = 0
    for (sport, ev, season), ys in multi:
        years = sorted(ys)
        gaps = [y for y in sorted(games[season]) if years[0] < y < years[-1] and y not in ys]
        if not gaps or not all(solo(ys[y]) for y in years):
            continue
        have = sorted(rnd.sample(years, 2))
        ask = sorted(have + gaps[:1])
        add("winners_gap", f"Who took gold in {label(sport, ev, season)} at the Games of {', '.join(map(str, ask))}?",
            "partial", {"found": [{"year": y, "gold": ys[y]["gold"]} for y in have], "not_in_corpus": gaps[:1]},
            [ys[y]["id"] for y in have])
        n += 1
        if n >= a.n_per_kind:
            break
    # missing_2028: 2024 event without a 2028 article
    cand = [ys[2024] for k, ys in by_key.items() if k[2] == "Summer" and 2024 in ys and 2028 not in ys
            and not used(ys[2024])]
    for e in rnd.sample(cand, min(a.n_per_kind // 2, len(cand))):
        add("missing_2028", f"According to the corpus, who won {label(e['sport'], e['event'], 'Summer')} in 2028?",
            "not_in_corpus", None, [])
    # trend: nations per edition
    n = 0
    for (sport, ev, season), ys in multi:
        evs = [{**e, "id": e["id"]} for e in ys.values()]
        p = crossgames.participation(evs, "nations")
        if p["missing_years"] or not p["max"] or len(p["max"]["years"]) > 1 or len(p["min"]["years"]) > 1:
            continue
        add("trend", f"For {label(sport, ev, season)}, give the number of participating nations at every edition in "
                     f"the corpus and say which edition had the most and which the fewest.", "answered",
            {"rows": [{"year": r["year"], "nations": r["value"]} for r in p["rows"]],
             "max_year": p["max"]["years"][0], "min_year": p["min"]["years"][0]}, [r["event_id"] for r in p["rows"]])
        n += 1
        if n >= a.n_per_kind:
            break
    # repeat winners by sport (individual events), sports not asked about publicly
    rows_by_athlete = defaultdict(lambda: {"name": "", "rows": []})
    for e in events:
        if not solo(e):
            continue
        for n_ in person_names(e["gold"], e["event"]):
            r = rows_by_athlete[norm(n_)]
            r["name"] = n_
            r["rows"].append(f"{e['id']}|{e['year']}|{e['season']}|{e['sport']}|{e['event']}|gold")
    reps = [g for g in crossgames.repeat_winners(list(rows_by_athlete.values())) if not CREW_EVENT.search(g["event"])]
    by_sport = defaultdict(list)
    for g in reps:
        by_sport[(g["sport"], g["season"])].append(g)
    sports = [k for k, gs in by_sport.items() if 2 <= len(gs) <= 6 and norm(k[0]) not in pub_text]
    for sport, season in rnd.sample(sports, min(a.n_per_kind, len(sports))):
        gs = by_sport[(sport, season)]
        add("repeat_winners", f"In {sport.lower()} at the {season} Olympics, which athletes won the same individual event "
                              f"at more than one Games according to the corpus? Give athlete, event and years.",
            "answered", [{"athlete": g["athlete"], "event": g["event"], "years": g["years"]} for g in gs],
            [i for g in gs for i in g["event_ids"]])
    # medal_leader and fewest_nations at the new Games
    for kind in ("medal_leader", "fewest_nations"):
        n = 0
        pool = sorted({(e["sport"], e["year"], e["season"]) for e in events if e["year"] in (2024, 2026, 2028)})
        rnd.shuffle(pool)
        for sport, year, season in pool:
            evs = [e for e in events if (e["sport"], e["year"], e["season"]) == (sport, year, season)]
            if len(evs) < 4 or any(e["id"] in pub_docs for e in evs):
                continue
            if kind == "medal_leader":
                count = defaultdict(list)
                for e in evs:
                    if e["gold_noc"]:
                        count[e["gold_noc"]].append(e["id"])
                ranked = sorted(count.items(), key=lambda kv: -len(kv[1]))
                if len(ranked) < 2 or len(ranked[0][1]) == len(ranked[1][1]):
                    continue   # ties are skipped: one clear answer
                add(kind, f"At the {year} {season} Olympics, which nation won the most gold medals in "
                          f"{sport.lower()}, and how many?", "answered", {"noc": ranked[0][0], "golds": len(ranked[0][1])},
                    ranked[0][1])
            else:
                rk = crossgames.rank_events(evs, "nations")
                if not rk["min"] or len(rk["min"]["events"]) != 1 or rk["missing"]:
                    continue
                ev = next(e for e in evs if e["event"] == rk["min"]["events"][0])
                add(kind, f"Of the {sport.lower()} events at the {year} {season} Olympics, which drew the fewest nations, "
                          f"and how many?", "answered", {"event": ev["event"], "nations": rk["min"]["value"]}, [ev["id"]])
            n += 1
            if n >= a.n_per_kind:
                break
    # top_source: the most authoritative source record about an LA 2028 event (one clear tier-1 record)
    by_event = defaultdict(list)
    for r in records:
        if r.get("event_id"):
            by_event[r["event_id"]].append(r)
    titles = {e["id"]: e for e in events}
    for eid, rs in sorted(by_event.items()):
        best = min(r["tier"] for r in rs)
        top = [r for r in rs if r["tier"] == best]
        if len(rs) < 2 or len(top) != 1 or eid in pub_docs or any(r["id"] in pub_docs for r in rs):
            continue
        e = titles[eid]
        add("top_source", f"Several source records concern the LA 2028 {e['event'].lower()} in {e['sport'].lower()}. "
                          f"Which of them comes from the most authoritative source, and what is that source?",
            "answered", {"record": top[0]["title"], "source": top[0]["source"]}, [top[0]["id"]])
    Path(a.out).write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in out), encoding="utf-8")
    kinds = defaultdict(int)
    for q in out:
        kinds[q["kind"]] += 1
    print(f"{len(out)} held-out questions -> {a.out}: {dict(kinds)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
