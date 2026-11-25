"""
Cross-Games reasoning done in code over rows returned by installed GSQL queries (deterministic, testable offline):

  repeat_winners      same athlete, same event (sport + event name + season), medal of one colour at 2+ Games
  participation       nations / competitors per edition of one event, with the editions holding the max and min
  medal_counts        medals per nation (from the medal_table query), ranked, ties kept
"""
from __future__ import annotations

from agr.parse import norm


def parse_medal_row(row: str) -> dict:
    """'event_id|year|season|sport|event|medal' as built by the athlete_medal_rows query."""
    eid, year, season, sport, event, medal = (row.split("|") + [""] * 6)[:6]
    return {"event_id": eid, "year": int(year) if year.lstrip("-").isdigit() else -1, "season": season,
            "sport": sport, "event": event, "medal": medal}


def repeat_winners(athletes: list[dict], min_games: int = 2) -> list[dict]:
    """athletes: [{'name', 'rows': [medal row strings]}] -> one entry per (athlete, event) won at >= min_games Games,
    sorted by athlete name. Years are distinct Games (two medals in one year count once)."""
    out = []
    for a in athletes:
        by: dict[tuple, dict] = {}
        for r in map(parse_medal_row, a.get("rows", [])):
            key = (norm(r["sport"]), norm(r["event"]), r["season"], r["medal"])
            g = by.setdefault(key, {"athlete": a["name"], "sport": r["sport"], "event": r["event"],
                                    "season": r["season"], "medal": r["medal"], "years": set(), "event_ids": []})
            g["years"].add(r["year"])
            g["event_ids"].append(r["event_id"])
        for g in by.values():
            if len(g["years"]) >= min_games:
                out.append({**g, "years": sorted(g["years"])})
    return sorted(out, key=lambda g: (norm(g["athlete"]), g["sport"], g["event"]))


def participation(events: list[dict], field: str = "nations") -> dict:
    """events: editions of one event with year / nations / competitors -> rows by year, max and min editions
    (ties kept), and the editions without a count (reported, never guessed)."""
    rows = sorted(({"year": e["year"], "season": e.get("season", ""), "value": e.get(field, -1),
                    "event_id": e.get("id", ""), "title": e.get("title", "")} for e in events),
                  key=lambda r: r["year"])
    known = [r for r in rows if isinstance(r["value"], int) and r["value"] >= 0]
    hi = max((r["value"] for r in known), default=None)
    lo = min((r["value"] for r in known), default=None)
    return {"field": field, "rows": rows,
            "max": {"value": hi, "years": [r["year"] for r in known if r["value"] == hi]} if known else None,
            "min": {"value": lo, "years": [r["year"] for r in known if r["value"] == lo]} if known else None,
            "missing_years": [r["year"] for r in rows if r not in known]}


def medal_counts(medals_by_nation: dict, events_by_nation: dict) -> list[dict]:
    """medal_table output -> [{'noc', 'medals', 'events'}] sorted by count (desc), then NOC; rank shared on ties."""
    rows = sorted(({"noc": k, "medals": int(v), "events": sorted(events_by_nation.get(k, []))}
                   for k, v in medals_by_nation.items()), key=lambda r: (-r["medals"], r["noc"]))
    rank, prev = 0, None
    for i, r in enumerate(rows, 1):
        if r["medals"] != prev:
            rank, prev = i, r["medals"]
        r["rank"] = rank
    return rows


def rank_events(events: list[dict], field: str = "nations") -> dict:
    """Events of one Games ranked by a count (desc); the events with the most and fewest (ties kept); events whose
    count is not stated are listed apart, never ranked."""
    rows = [{"event": e.get("event") or e.get("title", ""), "event_id": e.get("id", ""), "value": e.get(field, -1)}
            for e in events]
    known = sorted((r for r in rows if isinstance(r["value"], int) and r["value"] >= 0),
                   key=lambda r: (-r["value"], r["event"]))
    if not known:
        return {"field": field, "rows": [], "max": None, "min": None, "missing": [r["event"] for r in rows]}
    hi, lo = known[0]["value"], known[-1]["value"]
    return {"field": field, "rows": known,
            "max": {"value": hi, "events": [r["event"] for r in known if r["value"] == hi]},
            "min": {"value": lo, "events": [r["event"] for r in known if r["value"] == lo]},
            "missing": sorted(r["event"] for r in rows if r not in known)}


def film_ratio(f: dict) -> tuple[float, float] | None:
    """gross / budget as (low, high) when both are known in the same currency; ranges give a range."""
    if not f.get("budget_currency") or f.get("budget_currency") != f.get("gross_currency"):
        return None
    if (f.get("budget_low") or 0) <= 0 or (f.get("gross_low") is None) or f["gross_low"] < 0:
        return None
    return f["gross_low"] / f["budget_high"], f["gross_high"] / f["budget_low"]


GENERIC_VENUE_WORDS = {"stadium", "arena", "park", "centre", "center", "hall", "field", "court", "course", "pool",
                       "olympic", "national", "international", "aquatics", "aquatic", "sports", "beach", "street",
                       "range", "venue", "club", "complex", "memorial", "dome", "pavilion", "velodrome"}


def mentions_event(text: str, ev: dict) -> bool:
    """Does a record's text name this event (its event name as a phrase) or its venue (the full venue name, or a
    distinctive venue word of 6+ letters such as 'coliseum')?"""
    import re
    t = re.sub(r"\b(\d+) m\b", r"\1 metre", norm(text))   # '10 m air pistol' names the '10 metre' event
    has = lambda p: bool(p) and re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", t) is not None
    if has(norm(ev.get("event", ""))) and has(norm(ev.get("sport", "")).split(" ")[0]):
        return True
    venue = norm(ev.get("venue", ""))
    if has(venue):
        return True
    words = [w for w in venue.split() if len(w) >= 6 and w not in GENERIC_VENUE_WORDS]
    return any(has(w) for w in words)
