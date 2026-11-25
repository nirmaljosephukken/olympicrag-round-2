"""
An in-memory stand-in for the installed Round 2 GSQL queries, computed from the parsed build/ JSON with the same
semantics as graph/queries.gsql, so the agent tools can be tested offline (no TigerGraph).
"""
from __future__ import annotations

import json
from pathlib import Path

from agr.parse import norm, person_names

BUILD = Path(__file__).resolve().parent.parent / "build"


def _v(i, **attrs):
    return {"v_id": i, "attributes": {"id": i, **attrs}}


class FakeTG:
    def __init__(self, build: Path = BUILD):
        self.events = json.loads((build / "events.json").read_text(encoding="utf-8"))
        self.films = json.loads((build / "films.json").read_text(encoding="utf-8"))
        self.people = {p["id"]: p for p in json.loads((build / "people.json").read_text(encoding="utf-8"))}
        self.terms = json.loads((build / "terms.json").read_text(encoding="utf-8"))
        self.records = json.loads((build / "records.json").read_text(encoding="utf-8"))
        self.calls = []

    def query(self, name, params=None):
        """Same contract as agr.tg.TG.query, including the query log / trace entry (transport 'fake')."""
        from agr import tg
        self.calls.append((name, params))
        out = getattr(self, "q_" + name)(**(params or {}))
        tg.log_call("query", name, params or {}, tg._rows(out), 1.0, "fake", tg._now_iso())
        return out

    # --- cross-Games
    def q_athlete_medal_rows(self, sport_norm, season, medal):
        rows = {}
        for e in self.events:
            if (sport_norm and norm(e["sport"]) != sport_norm) or (season and e["season"] != season):
                continue
            for m in ("gold", "silver", "bronze"):
                if medal and m != medal:
                    continue
                names = person_names(e[m], e["event"]) + \
                    ([n for x in e["extra_bronze"] for n in person_names(x, e["event"])] if m == "bronze" else [])
                for n in names:
                    a = rows.setdefault(norm(n), {"name": n, "rows": []})
                    a["rows"].append(f"{e['id']}|{e['year']}|{e['season']}|{e['sport']}|{e['event']}|{m}")
        return [{"athletes": [_v(k, **a) for k, a in rows.items()]}]

    def q_event_editions(self, sport_norm, event_norm, season):
        evs = [e for e in self.events if norm(e["event"]) == event_norm and (not sport_norm or norm(e["sport"]) ==
                                                                            sport_norm) and (not season or e["season"] == season)]
        return [{"events": [_v(e["id"], title=e["title"], sport=e["sport"], year=e["year"], season=e["season"],
                               nations=e["nations"], competitors=e["competitors"], gold=e["gold"]) for e in evs]}]

    def q_medal_table(self, sport_norm, year, season, medal):
        count, events, n = {}, {}, 0
        for e in self.events:
            if e["year"] != year or (season and e["season"] != season) or (sport_norm and norm(e["sport"]) != sport_norm):
                continue
            n += 1
            for m in ("gold", "silver", "bronze"):
                noc = e[f"{m}_noc"]
                if noc and (not medal or m == medal):
                    count[noc] = count.get(noc, 0) + 1
                    events.setdefault(noc, []).append(f"{e['id']}|{e['event']}|{m}|{e[m]}")
        return [{"events_found": n, "medals_by_nation": count, "events_by_nation": events}]

    def q_find_events(self, sport_norm, year, season, venue_norm, date_norm, event_norm, lim):
        evs = [e for e in self.events if (year <= 0 or e["year"] == year) and (not season or e["season"] == season)
               and (not sport_norm or norm(e["sport"]) == sport_norm)
               and (not venue_norm or venue_norm in norm(e["venue"])) and (not event_norm or event_norm in norm(e["event"]))]
        return [{"events": [_v(e["id"], title=e["title"], sport=e["sport"], year=e["year"], season=e["season"],
                               venue=e["venue"], date=e["date"], competitors=e["competitors"], nations=e["nations"],
                               gold=e["gold"]) for e in evs[:lim]]}]

    def q_games_events(self, sport_norm, year, season):
        evs = [e for e in self.events if e["year"] == year and (not season or e["season"] == season)
               and (not sport_norm or norm(e["sport"]) == sport_norm)]
        return [{"events": [_v(e["id"], title=e["title"], sport=e["sport"], event=e["event"], year=e["year"],
                               season=e["season"], nations=e["nations"], competitors=e["competitors"], gold=e["gold"],
                               gold_noc=e["gold_noc"]) for e in evs]}]

    def q_chunk_search(self, qv, k):
        """No vectors offline: an empty result with the real query's shape."""
        return [{"hits": []}, {"distances": {}}]

    def q_text_search(self, needle, lim):
        out = []
        for c in self.chunks():
            if needle.lower() in c["text"].lower():
                out.append(_v(c["id"], doc_id=c["doc_id"], idx=c["idx"], text=c["text"]))
                if len(out) >= lim:
                    break
        return [{"hits": out}]

    def q_doc_chunks(self, doc_id, max_idx):
        cs = sorted((c for c in self.chunks() if c["doc_id"] == doc_id and c["idx"] <= max_idx), key=lambda c: c["idx"])
        return [{"doc": [_v(doc_id)]}, {"chunks": [_v(c["id"], idx=c["idx"], text=c["text"]) for c in cs]}]

    def chunks(self):
        if not hasattr(self, "_chunks"):
            self._chunks = [json.loads(l) for l in open(BUILD / "chunks.jsonl", encoding="utf-8")]
        return self._chunks

    # --- films and offices
    def q_film_facts(self, title_norm, person_norm, role, year_from, year_to):
        out = []
        for f in self.films:
            credits = [(r, n) for r, ns in f["credits"].items() for n in ns]
            if person_norm and not any(norm(n) == person_norm and (not role or role in r) for r, n in credits):
                continue
            if title_norm and title_norm not in norm(f["title"]):
                continue
            if (year_from > 0 and f["year"] < year_from) or (year_to > 0 and f["year"] > year_to):
                continue
            out.append(_v(f["id"], **{k: v for k, v in f.items() if k not in ("id", "credits")},
                          credits=[f"{r}: {n}" for r, n in credits]))
        return [{"films": out}]

    def q_office_succession(self, person_norm, office_like):
        holder = {p["doc_id"]: p["name"] for p in self.people.values()}
        nexts = {t["id"]: t["next_id"] for t in self.terms if t.get("next_id")}
        prevs = {v: k for k, v in nexts.items()}
        out = []
        for t in self.terms:
            if person_norm and norm(holder.get(t["doc_id"], "")) != person_norm:
                continue
            if office_like and office_like not in t["office_key"]:
                continue
            out.append(_v(t["id"], holder=[holder.get(t["doc_id"], "")],
                          **{k: t[k] for k in ("office", "term_start", "term_end", "start_year", "end_year",
                                               "predecessor", "successor", "doc_id")},
                          prev_term=[prevs[t["id"]]] if t["id"] in prevs else [],
                          next_term=[nexts[t["id"]]] if t["id"] in nexts else []))
        return [{"terms": out}]

    # --- source records
    def _rec(self, r):
        return _v(r["id"], title=r["title"], source=r["source"], published=r["published"],
                  published_utc=r["published_utc"].replace("T", " ").rstrip("Z"), tier=r["tier"],
                  tier_label=r["tier_label"], body=r["body"])

    def q_event_sources(self, eid):
        return [{"event": [_v(eid)]}, {"records": [self._rec(r) for r in self.records if r["event_id"] == eid]}]

    def q_list_sources(self, published_before):
        cut = published_before.replace("T", " ").rstrip("Z")
        return [{"records": [self._rec(r) for r in self.records
                             if not cut or r["published_utc"].replace("T", " ").rstrip("Z") <= cut]}]
