"""
Films and officeholders -> graph data, parsed deterministically from their infoboxes (no LLM).

  [Infobox film]          -> Film(title, year, released, director, budget/gross as text and as numbers + currency, ...)
                             Film -FILM_CREDIT(role)-> Person for director, producer, writer, starring, music, ...
  [Infobox officeholder]  -> Person -HELD-> OfficeTerm(office, term_start, term_end, years)
                             OfficeTerm -PRECEDED_BY-> Person, OfficeTerm -SUCCEEDED_BY-> Person (names as written)
                             OfficeTerm -NEXT_TERM-> OfficeTerm when the successor has an article with a term of the
                             same office (normalised: ordinal removed), which makes succession chains traversable
  [Infobox person]        -> Person(occupation, birth date)

An officeholder infobox holds several offices as numbered groups: `office2`, `term_start2`, `predecessor2`, ...
(the un-numbered group is one of them). Each group becomes one OfficeTerm. A group's office name is its `office`,
`order` or `title` field, or, for legislators, `jr/sr` + `state` (e.g. "United States Senator from Maine").
"""
from __future__ import annotations

import re

from agr.parse import norm, split_names

CREDIT_ROLES = ("director", "producer", "writer", "screenplay", "story", "starring", "music", "cinematography",
                "editing", "narrator")
SYMBOLS = [("US$", "USD"), ("HK$", "HKD"), ("A$", "AUD"), ("$", "USD"), ("₹", "INR"), ("Rs", "INR"), ("£", "GBP"),
           ("€", "EUR"), ("¥", "JPY")]
SCALE = {"thousand": 1e3, "million": 1e6, "milion": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "crore": 1e7,
         "crores": 1e7, "c": 1e7, "lakh": 1e5, "lakhs": 1e5}
MONEY_RE = re.compile(r"(?P<sym>US\$|HK\$|A\$|\$|₹|Rs\.?|£|€|¥)?\s*(?P<lo>\d[\d,]*(?:\.\d+)?)"
                      r"(?:\s*[–\-‒—]\s*(?:\$|₹|£|€)?\s*(?P<hi>\d[\d,]*(?:\.\d+)?))?\s*"
                      r"(?P<scale>thousand|million|milion|billion|bn|crores?|lakhs?|m\b|c\b)?", re.I)


def split_list(v: str) -> list[str]:
    """'Tom Hanks, Edward Burns and Matt Damon' -> names: commas, semicolons, ' and ', and names stored run together
    ('Lowell GanzBabaloo Mandel'), with the same rules as medal fields (agr.parse.split_names)."""
    return split_names(v or "")


def parse_money(text: str) -> dict:
    """First amount in a budget / box-office field: {'currency', 'low', 'high', 'note'}; low = high = -1 if none.
    '$65–70million' -> USD 65e6..70e6; '15 crore' -> INR 1.5e8; the note keeps a scope such as '(domestic only)'."""
    t = (text or "").replace(" ", " ")
    m = MONEY_RE.search(t)
    if not m:
        return {"currency": "", "low": -1.0, "high": -1.0, "note": ""}
    scale_word = (m.group("scale") or "").lower()
    scale = SCALE.get(scale_word, 1.0)
    sym = (m.group("sym") or "").rstrip(".")
    cur = dict(SYMBOLS).get(sym, "")
    if not cur and scale_word.startswith(("crore", "lakh", "c")):
        cur = "INR"
    num = lambda s: float(s.replace(",", "")) * scale
    lo = num(m.group("lo"))
    hi = num(m.group("hi")) if m.group("hi") else lo
    note = re.search(r"\(([^)]*)\)", t[m.end(): m.end() + 60])
    return {"currency": cur, "low": lo, "high": hi, "note": note.group(1) if note else ""}


def parse_film(doc: dict, fields: dict) -> dict:
    rel = fields.get("released", "")
    y = re.search(r"\b(1[89]\d\d|20\d\d)\b", rel)
    rt = re.search(r"\d+", fields.get("runtime", ""))
    b, g = parse_money(fields.get("budget", "")), parse_money(fields.get("gross", ""))
    return {"id": doc["doc_id"], "title": doc["title"], "name": fields.get("name") or doc["title"],
            "year": int(y.group(1)) if y else -1, "released": rel, "director": fields.get("director", ""),
            "starring": fields.get("starring", ""), "music": fields.get("music", ""),
            "country": fields.get("country", ""), "language": fields.get("language", ""),
            "runtime_min": int(rt.group(0)) if rt else -1,
            "budget": fields.get("budget", ""), "gross": fields.get("gross", ""),
            "budget_currency": b["currency"], "budget_low": b["low"], "budget_high": b["high"],
            "gross_currency": g["currency"], "gross_low": g["low"], "gross_high": g["high"], "gross_note": g["note"],
            "credits": {r: split_list(fields.get(r, "")) for r in CREDIT_ROLES if fields.get(r)}}


ORDINAL = re.compile(r"^\s*\d+(?:st|nd|rd|th)\s+", re.I)


def office_key(office: str) -> str:
    """'32nd Governor of Wyoming' -> 'governor of wyoming' (the same office across holders)."""
    return norm(ORDINAL.sub("", office or ""))


def _year(s: str) -> int:
    m = re.search(r"\b(1[6-9]\d\d|20\d\d)\b", s or "")
    return int(m.group(1)) if m else -1


def parse_offices(doc: dict, fields: dict) -> dict:
    """One officeholder infobox -> {'person': ..., 'terms': [...]}, one term per numbered group."""
    suffixes = sorted({m.group(1) for k in fields for m in [re.match(r"(?:term_start|office|order|title|predecessor|"
                                                                     r"successor)(\d*)$", k)] if m},
                      key=lambda s: int(s or 0))
    terms = []
    for s in suffixes:
        office = fields.get(f"office{s}") or fields.get(f"order{s}") or fields.get(f"title{s}") or ""
        g = lambda k: fields.get(f"{k}{s}", "")
        if not office and g("jr/sr"):
            office = g("jr/sr") + (f" from {g('state')}" if g("state") else "")
        # legislature seats, as the officeholder infobox template names them
        dist = f" ({g('district')} district)" if g("district") else ""
        if not office and g("state_house"):
            office = f"Member of the {g('state_house')} House of Representatives{dist}"
        if not office and g("state_senate"):
            office = f"Member of the {g('state_senate')} Senate{dist}"
        if not office and g("state_assembly"):
            office = f"Member of the {g('state_assembly')} State Assembly{dist}"
        if not office and g("state_delegate"):
            office = f"Member of the {g('state_delegate')} House of Delegates{dist}"
        if not office and g("state") and g("district"):
            office = f"Member of the U.S. House of Representatives from {g('state')}{dist}"
        if not office and g("constituency_MP"):
            office = f"Member of {g('parliament') or 'Parliament'} for {g('constituency_MP')}"
        if not office and g("constituency"):
            office = f"Member for {g('constituency')}"
        start, end = fields.get(f"term_start{s}", ""), fields.get(f"term_end{s}", "")
        pred, succ = fields.get(f"predecessor{s}", ""), fields.get(f"successor{s}", "")
        if not (office or start or pred or succ):
            continue
        terms.append({"id": f"{doc['doc_id']}#term{s or 0}", "doc_id": doc["doc_id"], "group": int(s or 0),
                      "office": office, "office_key": office_key(office), "term_start": start, "term_end": end,
                      "start_year": _year(start), "end_year": _year(end), "predecessor": pred, "successor": succ})
    name = fields.get("name") or doc["title"]
    return {"person": {"id": norm(name), "name": name, "doc_id": doc["doc_id"], "kind": "officeholder",
                       "birth_date": fields.get("birth_date", ""), "party": fields.get("party", ""),
                       "occupation": ""}, "terms": terms}


def parse_person(doc: dict, fields: dict) -> dict:
    name = fields.get("name") or doc["title"]
    return {"id": norm(name), "name": name, "doc_id": doc["doc_id"], "kind": "person",
            "birth_date": fields.get("birth_date", ""), "party": "", "occupation": fields.get("occupation", "")}


def link_terms(terms: list[dict], people: dict) -> None:
    """term -> next_id: the successor's term of the same office, when the successor has an article in the corpus.
    Exact (normalised) name and office key only; if the successor holds that office in several groups, the group
    whose start year is closest after this term's start is taken, never a different office."""
    by_doc: dict[str, list[dict]] = {}
    for t in terms:
        by_doc.setdefault(t["doc_id"], []).append(t)
    for t in terms:
        t["next_id"] = ""
        succ = people.get(norm(t["successor"]))
        if not succ or not t["office_key"]:
            continue
        same = [u for u in by_doc.get(succ["doc_id"], []) if u["office_key"] == t["office_key"] and u is not t]
        if t["start_year"] > 0:
            later = [u for u in same if u["start_year"] >= t["start_year"]]
            same = sorted(later, key=lambda u: u["start_year"]) or same
        if same:
            t["next_id"] = same[0]["id"]


def parse_entities(docs_with_fields: list[tuple[dict, str | None, dict]]) -> dict:
    """[(doc, infobox kind, fields)] -> {'films', 'people', 'terms'} (people keyed by normalised name)."""
    films, people, terms = [], {}, []
    for doc, kind, fields in docs_with_fields:
        if kind == "film":
            films.append(parse_film(doc, fields))
        elif kind == "officeholder":
            o = parse_offices(doc, fields)
            people[o["person"]["id"]] = o["person"]
            terms += o["terms"]
        elif kind == "person":
            p = parse_person(doc, fields)
            people.setdefault(p["id"], p)
    link_terms(terms, people)
    return {"films": films, "people": people, "terms": terms}
