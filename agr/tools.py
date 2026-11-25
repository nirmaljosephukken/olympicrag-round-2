"""
Specialised retrieval / reasoning agents. Each one is a deterministic, inspectable tool over TigerGraph that
returns an Observation: a compact summary for the orchestrator plus Evidence items (with citations).

  entity_linker      question text -> years / season / sport / venue / date / exact event titles
  graph_search       filtered traversal Games -> Event (sport, venue, date, event-name filters)
  venue_events       Venue -> Event multi-hop (venue + date -> event -> medallists)
  event_profile      1-hop neighbourhood of an event (medallists, nations, previous/next edition)
  edition_hop        temporal traversal along NEXT_EDITION / PREV_EDITION edges
  aggregate          graph-native count / argmax over events (GSQL accumulators)
  similarity_search  vector search over all chunks (TigerGraph HNSW index)
  scoped_similarity  vector search restricted to chunks of graph-selected documents
  read_document      document retrieval (ordered chunks of one article)
  athlete_medals     Athlete -> Event traversal
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from agr.config import ROOT
from agr.embed import embed_query
from agr import crossgames
from agr.parse import norm, TITLE_RE, split_names, source_tier, TIER_LABELS, PROVISIONAL_RE, CREW_EVENT
from agr.tg import TG

MONTHS = "january february march april may june july august september october november december".split()


@dataclass
class Evidence:
    id: str
    kind: str          # graph | chunk | aggregate
    doc_id: str
    title: str
    text: str
    url: str = ""


@dataclass
class Observation:
    tool: str
    summary: str                       # what the orchestrator sees
    evidence: list[Evidence] = field(default_factory=list)
    found: bool = True
    data: Any = None
    ms: float = 0.0
    context_tokens: int = 0


def _attrs(v: dict) -> dict:
    return {"id": v.get("v_id"), **v.get("attributes", {})}


def _date_tokens(s: str) -> set[str]:
    s = norm(s)
    toks = set(re.findall(r"\b\d{1,2}\b", s))
    toks |= {m for m in MONTHS if m in s}
    return toks


def date_score(query: str, cand: str) -> float:
    """2.0 = the date text is identical; 1.0+ = one contains the other; else day/month token overlap (0-1)."""
    qn, cn = norm(query).replace(" to ", " "), norm(cand).replace(" to ", " ")
    if not qn or not cn:
        return 0.0
    if qn == cn:
        return 2.0
    q, c = _date_tokens(query), _date_tokens(cand)
    overlap = len(q & c) / len(q | c) if q and c else 0.0
    if qn in cn or cn in qn:
        return 1.0 + overlap / 2
    return overlap


def event_phrase_match(phrase: str, event_norm: str) -> bool:
    p = norm(phrase)
    return bool(p) and re.search(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", event_norm) is not None


class Toolbox:
    def __init__(self, tg: TG | None = None):
        self.tg = tg or TG()
        self.counter = 0
        self.docs = {d["id"]: d for d in json.loads((ROOT / "build" / "documents.json").read_text(encoding="utf-8"))}

    # ------------------------------------------------------------ helpers
    def _ev(self, kind: str, doc_id: str, text: str, title: str | None = None) -> Evidence:
        self.counter += 1
        d = self.docs.get(doc_id, {})
        return Evidence(id=f"E{self.counter}", kind=kind, doc_id=doc_id, title=title or d.get("title", doc_id),
                        text=text, url=d.get("url", ""))

    def _event_fact(self, e: dict) -> Evidence:
        parts = [f"{e.get('title')}"]
        for k in ("venue", "date", "date_text", "competitors", "nations", "gold", "silver", "bronze"):
            if e.get(k) not in (None, "", -1):
                parts.append(f"{k.replace('_text', '')}: {e[k]}")
        return self._ev("graph", e["id"], "; ".join(parts), e.get("title"))

    # ------------------------------------------------------------ entity linking
    @lru_cache(maxsize=1)
    def catalog(self) -> dict:
        p = ROOT / "build" / "catalog.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        r = self.tg.query("catalog")
        ev = self.tg.query("find_events", {"sport_norm": "", "year": -1, "season": "", "venue_norm": "",
                                           "date_norm": "", "event_norm": "", "lim": 100000})
        cat = {"sports": sorted({s["attributes"]["name"] for s in r[0]["sports"]}),
               "venues": sorted({v["attributes"]["name"] for v in r[0]["venues"]}),
               "events": {norm(e["attributes"]["title"]): e["v_id"] for e in ev[0]["events"]}}
        p.write_text(json.dumps(cat, ensure_ascii=False))
        return cat

    def entity_linker(self, question: str) -> Observation:
        t0 = time.time()
        q = norm(question)
        cat = self.catalog()
        years = [int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", question)]
        season = "Summer" if "summer" in q else ("Winter" if "winter" in q else "")
        sports = sorted([s for s in cat["sports"] if re.search(r"\b" + re.escape(norm(s)) + r"\b", q)],
                        key=len, reverse=True)
        sports = [s for s in sports if not any(norm(s) != norm(o) and norm(s) in norm(o) for o in sports)]
        venues = sorted([v for v in cat["venues"] if len(norm(v)) > 3 and norm(v) in q], key=len, reverse=True)
        titles = [(t, i) for t, i in cat["events"].items() if t in q]
        date = ""
        m = re.search(r"\bon (.+?)(?: at the \d{4}| at the|:|\?|$)", question)
        if m:
            date = m.group(1).strip()
        res = {"years": years, "season": season, "sports": sports[:3], "venues": venues[:3], "date_phrase": date,
               "exact_event_titles": [{"event_id": i, "title_norm": t} for t, i in titles[:3]]}
        named = self.names_index()
        hits = [(n, info) for n, info in named.items() if re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", q)]
        hits = [(n, i) for n, i in hits if not any(n != o and n in o for o, _ in hits)]   # longest names only
        res["people"] = [i for _, i in hits if i["kind"] in ("person", "athlete")][:4]
        res["films"] = [i for _, i in hits if i["kind"] == "film"][:3]
        return Observation("entity_linker", json.dumps(res, ensure_ascii=False), [], bool(any(res.values())),
                           res, (time.time() - t0) * 1000)

    @lru_cache(maxsize=1)
    def _names_index(self) -> dict:
        """Names a question can mention, from the parsed build data: people with an article, every medallist and
        every film title (normalised name -> what it is and where to read about it). Two-word names at least, or a
        film title of 5+ letters, so common words never link."""
        out = {}
        b = ROOT / "build"
        load = lambda f: json.loads((b / f).read_text(encoding="utf-8")) if (b / f).exists() else []
        for e in load("events.json"):
            for m in ("gold", "silver", "bronze"):
                for n in split_names(e.get(m, "")):
                    if len(n.split()) >= 2 and len(n) <= 40:
                        out.setdefault(norm(n), {"kind": "athlete", "name": n})
        for p in load("people.json"):
            if len(p["name"].split()) >= 2:
                out[norm(p["name"])] = {"kind": "person", "name": p["name"], "doc_id": p["doc_id"]}
        for f in load("films.json"):
            name = f.get("name") or f["title"]
            if len(norm(name)) >= 5:
                out[norm(name)] = {"kind": "film", "name": name, "doc_id": f["id"], "year": f.get("year")}
        return out

    def names_index(self) -> dict:
        return self._names_index()

    # ------------------------------------------------------------ graph agents
    def _split_sport(self, sport: str, event: str = "") -> tuple[str, str]:
        """(sport, event): a value passed as `sport` that is not a sport in the catalogue (e.g. "10 metre air rifle")
        is used as the event name instead, so the call still finds the events."""
        try:
            sports = {norm(s) for s in self.catalog()["sports"]}
        except Exception:
            return sport, event
        if sport and norm(sport) not in sports:
            return "", event or sport
        return sport, event

    def graph_search(self, sport: str = "", year: int = -1, season: str = "", venue: str = "", date: str = "",
                     event: str = "", limit: int = 40) -> Observation:
        t0 = time.time()
        sport, event = self._split_sport(sport, event)
        m = TITLE_RE.match(event or "")
        if m:  # a full article title was passed as the event name: use its parts
            sport, year, season, event = sport or m.group("sport"), int(year or -1) if year not in (None, "", -1) \
                else int(m.group("year")), season or m.group("season"), m.group("event")
        r = self.tg.query("find_events", {"sport_norm": norm(sport), "year": int(year or -1), "season": season or "",
                                          "venue_norm": norm(venue), "date_norm": "", "event_norm": norm(event),
                                          "lim": 2000})
        evs = [_attrs(v) for v in r[0]["events"]]
        if event:
            ev_part = lambda e: norm(e["title"].split(" – ")[-1])  # the event name after the dash
            exact = [e for e in evs if ev_part(e) == norm(event)]
            evs = exact or [e for e in evs if event_phrase_match(event, norm(e["title"]))]
        if date:
            evs = sorted(evs, key=lambda e: -date_score(date, e.get("date", "")))
        evs = evs[:limit]
        ev = [self._event_fact(e) for e in evs]
        summ = f"{len(evs)} events" + "".join(f"\n[{x.id}] {e['id']} | {x.text}" for x, e in zip(ev, evs))
        return Observation("graph_search", summ, ev, bool(evs), evs, (time.time() - t0) * 1000)

    def resolve_venues(self, venue: str) -> list[str]:
        """Venue entity resolution: spacing/punctuation-insensitive containment, then token overlap."""
        squash = lambda x: norm(x).replace(" ", "")
        q, qt = squash(venue), set(norm(venue).split())
        names = self.catalog()["venues"]
        hits = [v for v in names if q and (q in squash(v) or (squash(v) in q and len(squash(v)) >= 0.6 * len(q)))]
        if not hits:
            scored = sorted(((len(qt & set(norm(v).split())) / max(1, len(qt | set(norm(v).split()))), v)
                             for v in names), reverse=True)
            hits = [v for sc, v in scored[:3] if sc >= 0.5]
        return hits[:6]

    def venue_events(self, venue: str, date: str = "", year: int = -1) -> Observation:
        t0 = time.time()
        if year in (-1, None, 0):
            ys = re.findall(r"\b(19\d\d|20\d\d)\b", date or "")
            year = int(ys[0]) if ys else -1
        evs, venues, seen = [], [], set()
        for vn in self.resolve_venues(venue) or [venue]:
            r = self.tg.query("events_at_venue", {"venue_norm": norm(vn), "date_norm": "", "year": int(year)})
            venues += [v["attributes"]["name"] for v in r[0]["venues"]]
            for v in r[1]["events"]:
                if v["v_id"] not in seen:
                    seen.add(v["v_id"]); evs.append(_attrs(v))
        for e in evs:
            e["date_match"] = round(date_score(date, e.get("date", "")), 2) if date else None
        evs.sort(key=lambda e: -(e["date_match"] or 0))
        top = [e for e in evs if not date or e["date_match"] == evs[0]["date_match"]][:8] if evs else []
        ev = [self._event_fact(e) for e in top]
        exact = [e for e in top if (e["date_match"] or 0) >= 2.0]
        note = (f"; EXACT date-text match: {exact[0]['title']} (this is the event asked about)" if len(exact) == 1
                else f"; {len(top)} events share the best date match: compare their date text with the question"
                if len(top) > 1 else "")
        summ = (f"venues matched: {sorted(set(venues))}; {len(evs)} events there{note}; best date matches:" +
                "".join(f"\n[{x.id}] {e['id']} | date_match={e['date_match']} | {x.text}" for x, e in zip(ev, top)))
        return Observation("venue_events", summ, ev, bool(top), top, (time.time() - t0) * 1000)

    def event_profile(self, event_id: str) -> Observation:
        t0 = time.time()
        event_id = self._resolve_event(event_id)  # a title instead of an id: resolved exactly, never fuzzily
        r = self.tg.query("event_profile", {"eid": event_id})
        if not r[0]["event"]:
            return Observation("event_profile", f"no event {event_id}", [], False)
        e = _attrs(r[0]["event"][0])
        meta = r[1]
        x = self._event_fact(e)
        summ = (f"[{x.id}] {e['id']} | {x.text}\nmedallists: {meta['medallists']}\n"
                f"previous_edition: {meta['previous_edition']}\nnext_edition: {meta['next_edition']}")
        x.text += f"; previous edition: {meta['previous_edition']}; next edition: {meta['next_edition']}"
        n_rec = self.event_index().get(event_id, {}).get("records", 0)
        if n_rec:
            summ += (f"\nSOURCE RECORDS: {n_rec} record(s) about this event (some may be provisional or superseded); "
                     f"call source_records('{event_id}') before answering about results, times, venues or entries")
        claims, _ = self._claims(event_id)
        n_conf = sum(c["kind"] == "count_statement" and c["status"].startswith("conflicts") for c in claims)
        n_chg = sum(c["kind"] in ("original_result", "disqualification", "medal_reallocated", "doping_retest")
                    for c in claims)
        if n_conf or n_chg:
            summ += (f"\nVERSIONS: {n_conf} conflicting statement(s), {n_chg} result-change statement(s) in the article; "
                     f"call fact_history('{event_id}') to see them")
        conf = [{"official": f"{e.get('competitors')}/{e.get('nations')}", "text": c["sentence"],
                 "text_value": c["claim_value"]} for c in claims
                if c["kind"] == "count_statement" and c["status"].startswith("conflicts")]
        chg = [{"text": c["sentence"]} for c in claims
               if c["kind"] in ("original_result", "disqualification", "medal_reallocated", "doping_retest")]
        return Observation("event_profile", summ, [x], True, {**e, **meta, "n_conflicts": n_conf, "n_changes": n_chg,
                                                              "conflict_claims": conf, "change_claims": chg},
                           (time.time() - t0) * 1000)

    def edition_hop(self, event_id: str, direction: str = "prev", hops: int = 1) -> Observation:
        t0 = time.time()
        r = self.tg.query("edition_hop", {"eid": event_id, "direction": direction, "hops": int(hops)})
        evs = [_attrs(v) for v in r[0]["events"]]
        ev = [self._event_fact(e) for e in evs]
        summ = (f"{direction} x{hops} from {event_id}: " +
                ("".join(f"\n[{x.id}] {e['id']} | {x.text}" for x, e in zip(ev, evs)) if evs else "no linked edition"))
        return Observation("edition_hop", summ, ev, bool(evs), evs, (time.time() - t0) * 1000)

    def multi_hop(self, path: list | str = "", event_id: str = "", venue: str = "", date: str = "", year: int = -1,
                  sport: str = "", season: str = "", event: str = "") -> Observation:
        """Multi-hop reasoning agent: resolve ONE start event, then walk a planned chain of hops through the graph
        (each hop 'prev' or 'next' along the edition chain), recording the event reached at every hop as evidence.

        The start is an event id/title, or venue + date (+ year), or sport + year (+ season) + event name. If the start
        is ambiguous (several events match equally) it does NOT guess: it returns the candidates and stops, so the
        orchestrator can choose one by id. If a hop has no linked edition, the chain stops there and says so."""
        t0 = time.time()
        hops = [h.strip().lower() for h in (path.replace(">", ",").split(",") if isinstance(path, str) else path or [])
                if str(h).strip()]
        bad = [h for h in hops if h not in ("prev", "next")]
        if bad:
            return Observation("multi_hop", f"unknown hop(s) {bad}: each hop must be 'prev' or 'next'", [], False,
                               {"chain": [], "ties": []}, (time.time() - t0) * 1000)
        # 1. resolve the start event (exact, never fuzzy)
        how, cands = "", []
        if event_id:
            eid = self._resolve_event(event_id)
            prof = self.tg.query("event_profile", {"eid": eid})
            cands = [_attrs(prof[0]["event"][0])] if prof and prof[0]["event"] else []
            how = f"event {eid}"
        elif venue:
            vo = self.venue_events(venue, date, year)
            top = vo.data or []
            exact = [e for e in top if (e.get("date_match") or 0) >= 2.0]
            cands = exact if len(exact) == 1 else top
            how = f"venue '{venue}' + date '{date}'"
        elif sport or event:
            go = self.graph_search(sport=sport, year=year, season=season, event=event, date=date)
            cands = go.data or []
            how = f"graph search {sport} {year} {season} '{event}'".strip()
        if not cands:
            return Observation("multi_hop", f"start not found ({how or 'no start given'})", [], False,
                               {"chain": [], "ties": []}, (time.time() - t0) * 1000)
        if len(cands) > 1:
            ev = [self._event_fact(e) for e in cands[:8]]
            summ = (f"START IS AMBIGUOUS: {len(cands)} events match {how} equally. No hop was taken. Call multi_hop "
                    f"again with the event_id of the one the question means:" +
                    "".join(f"\n[{x.id}] {e['id']} | {x.text}" for x, e in zip(ev, cands)))
            ties = [{"id": e["id"], "title": e["title"], "date": e.get("date", "")} for e in cands[:8]]
            return Observation("multi_hop", summ, ev, False, {"chain": [], "ties": ties}, (time.time() - t0) * 1000)
        # 2. walk the chain, one hop at a time
        cur = cands[0]
        chain, ev = [cur], [self._event_fact(cur)]
        lines = [f"start ({how}): [{ev[0].id}] {cur['id']} | {ev[0].text}"]
        stopped = ""
        for i, h in enumerate(hops, 1):
            r = self.tg.query("edition_hop", {"eid": cur["id"], "direction": h, "hops": 1})
            nxt = [_attrs(v) for v in r[0]["events"]]
            if not nxt:
                stopped = f"hop {i} ({h}) from {cur['title']}: no linked edition, chain stopped"
                lines.append(stopped)
                break
            cur = nxt[0]
            x = self._event_fact(cur)
            chain.append(cur); ev.append(x)
            lines.append(f"hop {i} ({h}): [{x.id}] {cur['id']} | {x.text}")
        done = len(chain) - 1 == len(hops)
        if done:
            lines.append(f"chain complete: answer from the LAST event [{ev[-1].id}]")
        return Observation("multi_hop", "\n".join(lines), ev, done,
                           {"chain": [{"id": e["id"], "title": e["title"]} for e in chain], "ties": [], "how": how,
                            "path": hops, "stopped": stopped}, (time.time() - t0) * 1000)

    def aggregate(self, sport: str, year: int, season: str, threshold: int = -1) -> Observation:
        t0 = time.time()
        r = self.tg.query("aggregate_events", {"sport_norm": norm(sport), "year": int(year), "season": season,
                                               "threshold": int(threshold)})
        s, rows = r[0], [_attrs(v) for v in r[1]["events"]]
        rows.sort(key=lambda e: -e["competitors"])
        self.counter += 1
        table = "; ".join(f"{e['title'].split(' – ')[-1]}={e['competitors']}" for e in rows)
        text = (f"{sport} at the {year} {season} Olympics: {s['events_found']} events in corpus; "
                f"events with competitors > {threshold}: {s['over_threshold']}; max competitors {s['max_competitors']} "
                f"in {s['max_events']}; events without a competitor count: {s['events_without_count']}. Table: {table}")
        ev = Evidence(id=f"E{self.counter}", kind="aggregate", doc_id=",".join(e["id"] for e in rows),
                      title=f"aggregate({sport}, {year} {season})", text=text)
        return Observation("aggregate", f"[{ev.id}] {text}", [ev], s["events_found"] > 0, {**s, "rows": rows},
                           (time.time() - t0) * 1000)

    def athlete_medals(self, name: str) -> Observation:
        """Athlete -> Event traversal. Medal counts are computed in code from the medal fields (exact name match),
        so a count answer is grounded in a stated number rather than the LLM's own tally."""
        t0 = time.time()
        r = self.tg.query("athlete_medals", {"name_like": norm(name)})
        evs = [_attrs(v) for v in r[1]["events"]]
        athletes = [a["attributes"]["name"] for a in r[0]["athletes"]]
        exact = [a for a in athletes if norm(a) == norm(name)] or athletes[:1]
        ev, lines = [], []
        for a in exact:
            by = {"gold": [], "silver": [], "bronze": []}
            for e in evs:
                for medal in by:
                    if any(norm(x) == norm(a) for x in split_names(e.get(medal, ""))):
                        by[medal].append(e)
            for e in by["gold"] + by["silver"] + by["bronze"]:
                ev.append(self._event_fact(e))
            self.counter += 1
            text = (f"{a}: gold in {len(by['gold'])} events; silver in {len(by['silver'])} events; "
                    f"bronze in {len(by['bronze'])} events. Gold: " + "; ".join(e["title"] for e in by["gold"]))
            ev.append(Evidence(id=f"E{self.counter}", kind="aggregate",
                               doc_id=",".join(e["id"] for e in by["gold"]), title=f"medal count({a})", text=text))
            lines.append(f"[E{self.counter}] {text}")
        summ = f"athletes matched: {athletes[:10]}\n" + "\n".join(lines) + \
               "".join(f"\n[{x.id}] {x.doc_id} | {x.text}" for x in ev if x.kind == "graph")
        return Observation("athlete_medals", summ, ev, bool(ev), evs, (time.time() - t0) * 1000)

    # ------------------------------------------------------------ reasoning over time
    def _resolve_event(self, event_id: str) -> str:
        if str(event_id) in self.docs:
            return event_id
        return self.catalog()["events"].get(norm(str(event_id)), event_id)

    def _resolve_doc(self, doc_id: str) -> str:
        """A title passed instead of a document id is resolved by exact (normalised) title match, never fuzzily."""
        if doc_id in self.docs:
            return doc_id
        if not hasattr(self, "_doc_by_title"):
            self._doc_by_title = {norm(d.get("title", "")): i for i, d in self.docs.items()}
        return self._doc_by_title.get(norm(str(doc_id)), doc_id)

    def _claims(self, eid: str) -> tuple[list[dict], list[str]]:
        r = self.tg.query("event_claims", {"eid": eid})
        return [_attrs(c) for c in r[1]["claims"]], r[2]["supersedes"]

    def fact_history(self, event_id: str) -> Observation:
        """All versions of an event's facts (official record, article statements, result changes), which supersedes
        which, and conflicts. The resolution is computed in code: dated changes supersede the original result, the
        infobox is the current official record, and disagreeing statements are reported, never dropped."""
        t0 = time.time()
        eid = self._resolve_event(event_id)
        claims, sup = self._claims(eid)
        title = self.docs.get(eid, {}).get("title", eid)
        if not claims:
            return Observation("fact_history", f"{title}: no conflicting or changed facts recorded; the official "
                                               f"record is the only version", [], True,
                               {"event": eid, "conflicts": [], "changes": []}, (time.time() - t0) * 1000)
        ev, lines, conflicts, changes = [], [], [], []
        record = next((c for c in claims if c["kind"] == "official_record"), None)
        for c in sorted(claims, key=lambda c: (c["kind"] != "official_record", c["id"])):
            kind = "graph" if c["kind"] == "official_record" else "claim"
            x = self._ev(kind, eid, f"[{c['source']}; {c['status']}] {c['sentence']}" +
                         (f" (dates: {c['dates']})" if c["dates"] else ""), title)
            ev.append(x)
            lines.append(f"[{x.id}] {c['kind']}: {x.text}")
            if c["kind"] == "count_statement" and c["status"].startswith("conflicts"):
                conflicts.append({"text": c["sentence"], "official": record["sentence"] if record else ""})
            if c["kind"] in ("original_result", "disqualification", "medal_reallocated", "doping_retest"):
                changes.append({"text": c["sentence"], "dates": c["dates"]})
        res = []
        if conflicts:
            res.append(f"CONFLICT: {len(conflicts)} article statement(s) disagree with the official record; policy: the "
                       f"official record (infobox) is used and the disagreement is reported")
        if changes:
            res.append(f"RESULT CHANGED AFTER THE EVENT: {len(changes)} statement(s); the current result is the "
                       f"official record, the original result is superseded" +
                       (f" ({len(sup)} supersedes edge(s))" if sup else ""))
        summ = f"{title}: " + ("; ".join(res) or "all versions agree") + "".join("\n" + l for l in lines)
        return Observation("fact_history", summ, ev, True, {"event": eid, "title": title, "conflicts": conflicts,
                                                            "changes": changes}, (time.time() - t0) * 1000)

    # ------------------------------------------------------------ source records (Round 2)
    def source_records(self, event_id: str = "", published_before: str = "") -> Observation:
        """Source records (scoreboards, jury decisions, bulletins, blogs, emails ...) in publication-time order, each
        with its source, publication time (local and UTC), reliability tier (1 official record, 2 operational,
        3 news/media, 4 informal; agr.parse.source_tier) and any words marking its content as provisional: those
        linked to one event, or all of them (optionally published up to a UTC time)."""
        t0 = time.time()
        article = []
        if event_id:
            eid = self._resolve_event(event_id)
            r = self.tg.query("event_sources", {"eid": eid})
            recs = [_attrs(v) for v in r[1]["records"]]
            for x in recs:
                x["relation"] = "linked to the event"
            # records not linked to any event that name this event's name or venue (generic text match, reported
            # as such so the agent can judge them)
            ctx = self.event_index().get(eid)
            if ctx:
                allr = [_attrs(v) for v in self.tg.query("list_sources", {"published_before": ""})[0]["records"]]
                have = {x["id"] for x in recs}
                for x in allr:
                    if x["id"] not in have and crossgames.mentions_event(x.get("title", "") + " " + x.get("body", ""),
                                                                         ctx):
                        x["relation"] = "not linked; mentions this event's name or venue"
                        recs.append(x)
                # the event's own article (its opening passages) comes first: the record of what happened
                d = self.tg.query("doc_chunks", {"doc_id": eid, "max_idx": 1})
                for c in d[1]["chunks"][:2]:
                    article.append(self._ev("chunk", eid, c["attributes"]["text"][:1200]))
        else:
            r = self.tg.query("list_sources", {"published_before": str(published_before or "")})
            recs = [_attrs(v) for v in r[0]["records"]]
        recs.sort(key=lambda x: (str(x.get("published_utc") or ""), x.get("id", "")))
        ev, lines = [], []
        for x in recs:
            tier = int(x.get("tier") or source_tier(x.get("source", "")))
            prov = sorted({m.lower() for m in PROVISIONAL_RE.findall(x.get("title", "") + " " + x.get("body", ""))})
            text = ((f"[{x['relation']}] " if x.get("relation") else "") +
                    f"source: {x['source']} (tier {tier}: {TIER_LABELS.get(tier, '')}); published: {x['published']}"
                    f" [UTC {x.get('published_utc', '')}]" + (f"; provisional wording: {', '.join(prov)}" if prov else "")
                    + f"; {x['body']}")
            e = self._ev("source", x["id"], text, x.get("title"))
            ev.append(e)
            lines.append(f"[{e.id}] {x['id']} | {x['title']} | {text[:900]}")
        summ = ("".join(f"[{a.id}] EVENT ARTICLE {a.doc_id} | {a.text}\n" for a in article) +
                f"{len(recs)} source record(s), in publication-time order (tier 1 = most authoritative):\n" +
                "\n".join(lines))
        ev = article + ev
        return Observation("source_records", summ, ev, bool(ev), recs, (time.time() - t0) * 1000,
                           sum(len(e.text) for e in ev) // 4)

    @lru_cache(maxsize=1)
    def _event_index(self) -> dict:
        p = ROOT / "build" / "events.json"
        if not p.exists():
            return {}
        recs = ROOT / "build" / "records.json"
        linked = {}
        for r in (json.loads(recs.read_text(encoding="utf-8")) if recs.exists() else []):
            if r.get("event_id"):
                linked[r["event_id"]] = linked.get(r["event_id"], 0) + 1
        return {e["id"]: {"event": e["event"], "sport": e["sport"], "venue": e["venue"], "year": e["year"],
                          "records": linked.get(e["id"], 0)}
                for e in json.loads(p.read_text(encoding="utf-8"))}

    def event_index(self) -> dict:
        return self._event_index()

    # ------------------------------------------------------------ cross-Games (Round 2)
    def repeat_winners(self, sport: str = "", season: str = "", medal: str = "gold", min_games: int = 2,
                       individual_only: bool = False, event: str = "") -> Observation:
        """Athletes who won the same event (same sport + event name + season) at min_games or more Games;
        individual_only drops crew / team events (doubles, pairs, relays, team, boats ...: agr.parse.CREW_EVENT).
        Rows come from the athlete_medal_rows query; grouping is done in code (agr.crossgames)."""
        t0 = time.time()
        sport, event = self._split_sport(sport, event)
        r = self.tg.query("athlete_medal_rows", {"sport_norm": norm(sport), "season": season or "",
                                                 "medal": (medal or "").lower()})
        rows = [{"name": a["attributes"]["name"], "rows": a["attributes"]["rows"]} for a in r[0]["athletes"]]
        reps = crossgames.repeat_winners(rows, int(min_games or 2))
        if str(individual_only).lower() in ("true", "1", "yes"):
            reps = [g for g in reps if not CREW_EVENT.search(g["event"])]
        if event:   # only one event (by name)
            reps = [g for g in reps if event_phrase_match(event, norm(g["event"]))]
        self.counter += 1
        text = (f"athletes with {medal or 'a medal'} in the same event at {min_games}+ Games"
                f"{' in ' + sport if sport else ''}{' (' + season + ')' if season else ''}: {len(reps)}. " +
                "; ".join(f"{g['athlete']} - {g['sport']} {g['event']} ({g['season']}): "
                          f"{', '.join(map(str, g['years']))}" for g in reps))
        ev = Evidence(id=f"E{self.counter}", kind="aggregate",
                      doc_id=",".join(i for g in reps for i in g["event_ids"]), title=f"repeat_winners({sport})",
                      text=text)
        return Observation("repeat_winners", f"[{ev.id}] {text}", [ev], bool(rows), reps, (time.time() - t0) * 1000)

    def participation_by_edition(self, event: str, sport: str = "", season: str = "",
                                 field: str = "nations") -> Observation:
        """Nations (or competitors) at every edition of one event in the corpus, with the max and min editions.
        `event` is the event name, e.g. "Women's javelin throw" (a full article title also works)."""
        t0 = time.time()
        m = TITLE_RE.match(event or "")
        if m:
            sport, season, event = sport or m.group("sport"), season or m.group("season"), m.group("event")
        sport, _ = self._split_sport(sport)
        field = field if field in ("nations", "competitors") else "nations"
        r = self.tg.query("event_editions", {"sport_norm": norm(sport), "event_norm": norm(event),
                                             "season": season or ""})
        evs = [_attrs(v) for v in r[0]["events"]]
        sports = sorted({e["sport"] for e in evs})
        if len(sports) > 1 and not sport:
            names = ", ".join(sports)
            return Observation("participation_by_edition", f"'{event}' exists in several sports ({names}): call again "
                               f"with sport", [], False, {"sports": sports}, (time.time() - t0) * 1000)
        p = crossgames.participation(evs, field)
        self.counter += 1
        table = "; ".join(f"{x['year']}: {x['value'] if x['value'] >= 0 else 'not stated'}"
                          + (f" (gold: {x['gold']})" if x.get("gold") else "") for x in p["rows"])
        text = (f"{field} per edition of {sport + ' ' if sport else ''}{event}{' (' + season + ')' if season else ''}: "
                f"{len(p['rows'])} editions in the corpus. {table}." +
                (f" Most: {p['max']['value']} in {', '.join(map(str, p['max']['years']))}; fewest: {p['min']['value']} "
                 f"in {', '.join(map(str, p['min']['years']))}." if p["max"] else "") +
                (f" No count stated for: {', '.join(map(str, p['missing_years']))}." if p["missing_years"] else ""))
        ev = Evidence(id=f"E{self.counter}", kind="aggregate", doc_id=",".join(x["event_id"] for x in p["rows"]),
                      title=f"participation({event})", text=text)
        return Observation("participation_by_edition", f"[{ev.id}] {text}", [ev], bool(evs), p,
                           (time.time() - t0) * 1000)

    def medal_table(self, sport: str, year: int, season: str = "", medal: str = "gold") -> Observation:
        """Medals of one colour (or all, medal='') per nation in one sport at one Games, counted in GSQL, ranked."""
        t0 = time.time()
        r = self.tg.query("medal_table", {"sport_norm": norm(sport), "year": int(year), "season": season or "",
                                          "medal": (medal or "").lower()})
        res = r[0] if r else {}
        rows = crossgames.medal_counts(res.get("medals_by_nation", {}) or {}, res.get("events_by_nation", {}) or {})
        self.counter += 1
        def item(e: str) -> str:
            parts = (e.split("|") + ["", "", "", ""])[:4]
            return f"{parts[1]}" + (f" [{parts[2]}]" if not medal else "") + (f" - {', '.join(split_names(parts[3])) or parts[3]}" if parts[3] else "")
        text = (f"{medal or 'all'} medals by nation, {sport} at the {year} {season} Olympics "
                f"({res.get('events_found', 0)} events in the corpus; {len(rows)} nations with a medal): " +
                "; ".join(f"{x['rank']}. {x['noc']} {x['medals']} ({', '.join(item(e) for e in x['events'])})"
                          for x in rows))
        ev = Evidence(id=f"E{self.counter}", kind="aggregate",
                      doc_id=",".join(sorted({e.split('|')[0] for x in rows for e in x["events"]})),
                      title=f"medal_table({sport}, {year} {season})", text=text)
        return Observation("medal_table", f"[{ev.id}] {text}", [ev], bool(rows), rows, (time.time() - t0) * 1000)

    def rank_events(self, sport: str, year: int, season: str = "", field: str = "nations") -> Observation:
        """Every event of one sport (sport='' for all) at one Games ranked by nations or competitors, with the
        event(s) holding the most and the fewest (ties kept) and the events without a count; ranked in code."""
        t0 = time.time()
        field = field if field in ("nations", "competitors") else "nations"
        r = self.tg.query("games_events", {"sport_norm": norm(sport), "year": int(year), "season": season or ""})
        evs = [_attrs(v) for v in r[0]["events"]]
        rk = crossgames.rank_events(evs, field)
        self.counter += 1
        table = "; ".join(f"{x['event']}={x['value']}" for x in rk["rows"][:60])
        text = (f"{sport or 'all sports'} at the {year} {season} Olympics: {len(evs)} events in the corpus, ranked by "
                f"{field}. " + (f"MOST: {', '.join(rk['max']['events'])} ({rk['max']['value']}). "
                                f"FEWEST: {', '.join(rk['min']['events'])} ({rk['min']['value']}). " if rk["max"] else "")
                + (f"No count stated: {', '.join(rk['missing'])}. " if rk["missing"] else "") + f"All: {table}")
        ends = {(rk["max"] or {}).get("value"), (rk["min"] or {}).get("value")} - {None}
        ev = Evidence(id=f"E{self.counter}", kind="aggregate",
                      doc_id=",".join(sorted({x["event_id"] for x in rk["rows"] if x["value"] in ends})),
                      title=f"rank_events({sport}, {year} {season}, {field})", text=text)
        return Observation("rank_events", f"[{ev.id}] {text}", [ev], bool(evs), rk, (time.time() - t0) * 1000)

    def text_search(self, phrase: str, k: int = 8, max_chars: int = 700) -> Observation:
        """Exact phrase search (case-insensitive) over the text of every document, e.g. a film title or a person's
        name mentioned inside another article. Returns the matching passages, centred on the phrase."""
        t0 = time.time()
        r = self.tg.query("text_search", {"needle": str(phrase or "").strip(), "lim": int(k)})
        hits = [_attrs(h) for h in r[0]["hits"]] if r else []
        ev = []
        for h in hits:
            txt = h["text"]
            i = txt.lower().find(str(phrase).lower())
            start = max(0, i - max_chars // 2) if i >= 0 else 0
            x = self._ev("chunk", h["doc_id"], ("..." if start else "") + txt[start:start + max_chars])
            ev.append(x)
        summ = f"{len(ev)} passage(s) containing '{phrase}':" + "".join(f"\n[{x.id}] ({x.title}) {x.text}" for x in ev)
        return Observation("text_search", summ, ev, bool(ev), hits, (time.time() - t0) * 1000,
                           sum(len(x.text) for x in ev) // 4)

    # ------------------------------------------------------------ films and officeholders (Round 2)
    def film_facts(self, title: str = "", person: str = "", role: str = "", year_from: int = -1,
                   year_to: int = -1, ratio_below: float = 0, ratio_above: float = 0) -> Observation:
        """Films from their infoboxes: by title, by a credited person (role director / starring / music /
        cinematography / producer / writer / editing, or any) and/or release-year range. Budget and gross are given
        as written; gross / budget is computed in code when both are in the same currency, and ratio_below /
        ratio_above keep only films whose ratio is below / above a threshold (computed in code, never by the LLM).
        When no film has its own article, the title or person is searched in the text of every article."""
        t0 = time.time()
        r = self.tg.query("film_facts", {"title_norm": norm(title), "person_norm": norm(person),
                                         "role": (role or "").lower(), "year_from": int(year_from or -1),
                                         "year_to": int(year_to or -1)})
        films = sorted((_attrs(v) for v in r[0]["films"]), key=lambda f: (f.get("year", 0), f.get("title", "")))
        for f in films:
            f["ratio"] = crossgames.film_ratio(f)
        head = f"{len(films)} film(s)"
        try:
            below, above = float(ratio_below or 0), float(ratio_above or 0)
        except (TypeError, ValueError):
            below = above = 0.0
        if below or above:
            keep = [f for f in films if f["ratio"] is not None and (not below or f["ratio"][1] < below)
                    and (not above or f["ratio"][0] > above)]
            unknown = [f["title"] for f in films if f["ratio"] is None]
            head = (f"{len(films)} film(s); {len(keep)} with gross/budget" + (f" below {below:g}" if below else "") +
                    (f" above {above:g}" if above else "") + f" (computed in code): " +
                    (", ".join(f"{f['title']} ({f['ratio'][0]:.2f})" for f in keep) or "none") +
                    (f". Ratio unknown (budget or gross missing or in different currencies): {', '.join(unknown)}"
                     if unknown else ""))
            films = keep + [f for f in films if f not in keep]
        ev, lines = [], []
        for f in films[:25]:
            ratio = f"; gross/budget = {f['ratio'][0]:.2f}" + (f"-{f['ratio'][1]:.2f}" if f["ratio"] and
                                                                abs(f["ratio"][1] - f["ratio"][0]) > 0.005 else "") \
                if f["ratio"] else ""
            credits = [c for c in sorted(f.get("credits", [])) if not role or role in c.split(":")[0]][:6]
            text = (f"{f['title']} ({f.get('year')}): released {f.get('released')}; director {f.get('director')}; "
                    f"starring {f.get('starring')}; music {f.get('music')}; budget {f.get('budget') or 'not stated'}; "
                    f"gross {f.get('gross') or 'not stated'}{ratio}" + (f"; {'; '.join(credits)}" if credits else ""))
            x = self._ev("graph", f["id"], text[:900], f["title"])
            ev.append(x)
            lines.append(f"[{x.id}] {f['id']} | {text[:450]}")
        summ = head + ("" if len(films) <= 25 else " (first 25 shown)") + "".join("\n" + l for l in lines)
        if not films and (title or person):  # no film article: the film may be mentioned inside another article
            ts = self.text_search(title or person, k=5)
            summ += f"\nNo film article matches. Text search for '{title or person}' in all articles:\n" + ts.summary
            ev += ts.evidence
        return Observation("film_facts", summ, ev, bool(ev), films, (time.time() - t0) * 1000,
                           sum(len(x.text) for x in ev) // 4)

    def office_succession(self, name: str = "", office: str = "") -> Observation:
        """Offices held by a person (from the officeholder infobox, one entry per office group) with term dates,
        predecessor and successor; or every holder of an office (office='governor of wyoming'). prev_term / next_term
        link to the term of the same office held by the predecessor / successor when that person has an article."""
        t0 = time.time()
        r = self.tg.query("office_succession", {"person_norm": norm(name),
                                                "office_like": norm(re.sub(r"^\s*\d+(st|nd|rd|th)\s+", "", office or ""))})
        terms = sorted((_attrs(v) for v in r[0]["terms"]), key=lambda x: (x.get("start_year", 0), x.get("id", "")))
        ev, lines = [], []
        for x in terms[:40]:
            text = (f"{', '.join(x.get('holder', [])) or '?'}: {x.get('office') or 'office not named'}, "
                    f"{x.get('term_start') or '?'} to {x.get('term_end') or 'present/not stated'}; "
                    f"predecessor: {x.get('predecessor') or 'not stated'}; successor: {x.get('successor') or 'not stated'}"
                    + (f"; previous term: {', '.join(x['prev_term'])}" if x.get("prev_term") else "")
                    + (f"; next term: {', '.join(x['next_term'])}" if x.get("next_term") else ""))
            e = self._ev("graph", x.get("doc_id", ""), text)
            ev.append(e)
            lines.append(f"[{e.id}] {x['id']} | {text}")
        summ = f"{len(terms)} office term(s)" + "".join("\n" + l for l in lines)
        return Observation("office_succession", summ, ev, bool(terms), terms, (time.time() - t0) * 1000)

    # ------------------------------------------------------------ text agents
    def _hits(self, r, tool: str, t0: float, max_chars: int) -> Observation:
        dist = r[1]["distances"] if len(r) > 1 else {}
        hits = sorted((_attrs(h) for h in r[0]["hits"]), key=lambda h: dist.get(h["id"], 1))
        ev = [self._ev("chunk", h["doc_id"], h["text"][:max_chars]) for h in hits]
        for x, h in zip(ev, hits):
            x.kind = f"chunk:{h['id']}"
        summ = "".join(f"\n[{x.id}] ({x.title}) {x.text}" for x in ev)
        return Observation(tool, summ.strip(), ev, bool(ev), hits, (time.time() - t0) * 1000,
                           sum(len(x.text) for x in ev) // 4)

    def similarity_search(self, query: str, k: int = 5, max_chars: int = 900) -> Observation:
        t0 = time.time()
        return self._hits(self.tg.query("chunk_search", {"qv": embed_query(query), "k": int(k)}),
                          "similarity_search", t0, max_chars)

    def scoped_similarity(self, query: str, doc_ids: list[str], k: int = 4, max_chars: int = 1400) -> Observation:
        t0 = time.time()
        return self._hits(self.tg.query("chunk_search_in_docs", {"qv": embed_query(query), "doc_ids": list(doc_ids),
                                                                 "k": int(k)}), "scoped_similarity", t0, max_chars)

    def find_documents(self, query: str, k: int = 5) -> Observation:
        """Document linking: rank article titles by word overlap with the query (code, no LLM, no embeddings).
        Year numbers must match exactly, so '1992' never links to a 2000 article."""
        t0 = time.time()
        qn = set(norm(query).split())
        q_years = {w for w in qn if re.fullmatch(r"(19|20)\d\d", w)}
        scored = []
        for d in self.docs.values():
            tn = set(norm(d["title"]).split())
            t_years = {w for w in tn if re.fullmatch(r"(19|20)\d\d", w)}
            if q_years and t_years and not (q_years & t_years):
                continue
            overlap = len(qn & tn) / max(1, len(tn))
            if overlap > 0:
                scored.append((overlap, d["id"], d["title"]))
        top = sorted(scored, reverse=True)[:k]
        summ = "documents (title match):" + "".join(f"\n  {i} | {t} | match={sc:.2f}" for sc, i, t in top)
        return Observation("find_documents", summ, [], bool(top), [{"id": i, "title": t} for _, i, t in top],
                           (time.time() - t0) * 1000)

    def read_document(self, doc_id: str, max_chunks: int = 2, contains: str = "") -> Observation:
        """Document retrieval: the opening chunks of one article, or (with `contains`) only the chunks of that
        article that contain the keyword, like searching inside the page."""
        t0 = time.time()
        doc_id = self._resolve_doc(doc_id)
        r = self.tg.query("doc_chunks", {"doc_id": doc_id, "max_idx": 500 if contains else int(max_chunks) - 1})
        chunks = [c["attributes"]["text"] for c in r[1]["chunks"]]
        if contains:
            chunks = [c for c in chunks if norm(contains) in norm(c)][:max(2, int(max_chunks))]
        ev = [self._ev("chunk", doc_id, c) for c in chunks]
        summ = "".join(f"\n[{x.id}] {x.text}" for x in ev)
        return Observation("read_document", summ.strip() or f"no matching text in {doc_id}", ev, bool(ev), None,
                           (time.time() - t0) * 1000, sum(len(x.text) for x in ev) // 4)


TOOL_SPECS = {   # what the orchestrator sees: signature -> what comes back (kept short: it is sent on every call)
    "graph_search": "(sport, year, season, venue, date, event, limit) -> events matching the filters (omit unknown ones)",
    "venue_events": "(venue, date, year) -> events at a venue ranked by date match (date text as in the question)",
    "event_profile": "(event_id) -> one event's facts, medallists, prev/next edition, versions, source-record count",
    "edition_hop": "(event_id, direction='prev'|'next', hops=1) -> the same event at an adjacent Games",
    "multi_hop": "(path=['prev'|'next',...], event_id | venue+date+year | sport+year+season+event) -> resolves one "
                 "start event, walks the hops, returns the event at every hop",
    "aggregate": "(sport, year, season, threshold) -> count of a Games' events with competitors > threshold",
    "rank_events": "(sport, year, season='', field='nations'|'competitors') -> a Games' events ranked, most/fewest",
    "medal_table": "(sport, year, season='', medal='gold'|'silver'|'bronze'|'') -> medals per nation, with events "
                   "and medallists",
    "repeat_winners": "(sport='', season='', medal='gold', min_games=2, individual_only=false) -> athletes who won "
                      "the same event at several Games, with years",
    "participation_by_edition": "(event, sport='', season='', field='nations'|'competitors') -> every edition "
                                "of one event: count and gold medallist per year, max/min",
    "athlete_medals": "(name) -> an athlete's medal events and counts",
    "fact_history": "(event_id) -> versions of an event's facts: official record, result changes, conflicts",
    "source_records": "(event_id='', published_before='') -> event article + its source records (time order, "
                      "tier 1 official > 2 operational > 3 news > 4 informal); without event_id: all records",
    "film_facts": "(title='', person='', role='', year_from=-1, year_to=-1, ratio_below=0, ratio_above=0) -> films: "
                  "release, director, cast, music, budget, gross, gross/budget (filters computed in code)",
    "office_succession": "(name='', office='') -> offices held, terms, predecessor, successor",
    "text_search": "(phrase, k=8) -> passages of any article containing the exact phrase",
    "find_documents": "(query) -> article titles matching the query",
    "read_document": "(doc_id, max_chunks=2, contains='') -> an article's opening passages, or those containing a word",
    "similarity_search": "(query, k=5) -> most similar passages in the whole corpus",
    "scoped_similarity": "(query, doc_ids, k=4) -> similar passages inside the given documents",
}
