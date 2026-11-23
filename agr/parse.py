"""
Corpus -> graph entities + text chunks.

The corpus is Wikipedia plain text where most articles start with an infobox block:

    [Infobox Olympic event]
      event: Men's 400 metre freestyle
      games: 2016 Summer
      ...

Olympic infoboxes are parsed deterministically (no LLM extraction cost, no hallucinated edges) into:
  Event, Games, Sport, Venue, Athlete (medal winners), NOC, plus PREV/NEXT edition links.
Every document (Olympic or not: films, people, companies ...) becomes a Document vertex whose
text is split into Chunk vertices for vector search, so nothing in the corpus is unreachable.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

TITLE_RE = re.compile(r"^(?P<sport>.+?) at the (?P<year>\d{4}) (?P<season>Summer|Winter) Olympics – (?P<event>.+)$")
INFOBOX_RE = re.compile(r"^\[Infobox (?P<kind>[^\]]+)\]\n(?P<body>(?:  .*\n?)+)")
DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")


def norm(s: str) -> str:
    """Lower-case, accent-free, dash-normalised, single-spaced key for matching names."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = s.translate(DASHES).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def to_int(v: str | None) -> int:
    if not v:
        return -1
    m = re.search(r"\d[\d,]*", v)
    return int(m.group(0).replace(",", "")) if m else -1


@dataclass
class Event:
    id: str                 # = doc_id (Wikidata QID)
    title: str
    sport: str
    event: str
    year: int
    season: str
    games: str              # "2016 Summer"
    venue: str = ""
    date: str = ""
    competitors: int = -1
    nations: int = -1
    gold: str = ""
    silver: str = ""
    bronze: str = ""
    gold_noc: str = ""
    silver_noc: str = ""
    bronze_noc: str = ""
    win_value: str = ""
    prev_year: int = -1
    next_year: int = -1
    extra_bronze: list = field(default_factory=list)


def parse_infobox(text: str) -> tuple[str | None, dict, str]:
    m = INFOBOX_RE.match(text)
    if not m:
        return None, {}, text
    fields = {}
    for line in m.group("body").splitlines():
        if ":" in line:
            k, v = line.strip().split(":", 1)
            fields[k.strip()] = v.strip()
    return m.group("kind"), fields, text[m.end():]


OLYMPIC_BLOCK_RE = re.compile(r"\[Infobox Olympic event\]\n((?:  .*\n?)+)")
TOURNAMENT_KINDS = {"international football competition", "international handball competition",
                    "international ice hockey competition", "field hockey", "olympic water polo tournament",
                    "rugby tournament"}


def olympic_infobox(text: str) -> dict | None:
    """The [Infobox Olympic event] block wherever it appears (tennis articles carry it as a second infobox)."""
    m = OLYMPIC_BLOCK_RE.search(text[:6000])
    if not m:
        return None
    fields = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.strip().split(":", 1)
            fields[k.strip()] = v.strip()
    return fields


def tournament_fields(fields: dict) -> dict:
    """Map a team-tournament infobox onto Olympic-event field names. Only fields that exist are copied;
    the champion is a team (usually an NOC code such as 'USA'), never an invented person."""
    pick = lambda *ks: next((fields[k] for k in ks if fields.get(k)), "")
    venue = pick("venue", "venues")
    venue = "" if re.fullmatch(r"\d+", venue) else venue  # football infoboxes put the venue COUNT in 'venues'
    return {"venue": venue, "dates": pick("dates", "date"),
            "competitors": pick("competitors"), "nations": pick("num_teams", "teams", "nations"),
            "gold": pick("champion", "champions", "champion_other", "winners"),
            "silver": pick("second", "second_other", "runnerup"), "bronze": pick("third", "third_other")}


def parse_event(doc: dict, fields: dict) -> Event | None:
    m = TITLE_RE.match(doc["title"])
    games = fields.get("games", "")
    if m:
        sport, year, season, ev = m.group("sport"), int(m.group("year")), m.group("season"), m.group("event")
    else:
        gm = re.match(r"(\d{4}) (Summer|Winter)", games)
        if not gm:
            return None
        year, season = int(gm.group(1)), gm.group(2)
        sport, ev = doc["title"].split(" at the ")[0], fields.get("event", doc["title"])
    return Event(
        id=doc["doc_id"], title=doc["title"], sport=sport, event=ev, year=year, season=season,
        games=f"{year} {season}", venue=fields.get("venue", ""),
        date=fields.get("date") or fields.get("dates", ""),
        competitors=to_int(fields.get("competitors")), nations=to_int(fields.get("nations")),
        gold=fields.get("gold", ""), silver=fields.get("silver", ""), bronze=fields.get("bronze", ""),
        gold_noc=fields.get("goldNOC", ""), silver_noc=fields.get("silverNOC", ""),
        bronze_noc=fields.get("bronzeNOC", ""), win_value=fields.get("win_value", ""),
        prev_year=to_int(fields.get("prev")), next_year=to_int(fields.get("next")),
        extra_bronze=[fields[k] for k in ("bronze2",) if fields.get(k)],
    )


def chunk_text(title: str, text: str, size: int = 1400, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunks; every chunk is prefixed with the article title so it stands alone."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size:  # very long paragraph/table
            if cur:
                chunks.append(cur); cur = ""
            chunks.append(p[:size]); p = p[size - overlap:]
        if len(cur) + len(p) + 2 > size and cur:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n" if overlap else ""
        cur += ("\n\n" if cur else "") + p
    if cur.strip():
        chunks.append(cur)
    return [f"[{title}]\n{c}" for c in chunks]


SOURCE_RE = re.compile(r"^Source:[ \t]*(?P<v>.+)$", re.M)
PUBLISHED_RE = re.compile(r"^Published:[ \t]*(?P<v>.+)$", re.M)
STAMP_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})(?:[ T](?P<time>\d{1,2}:\d{2}))?(?:\s*(?P<zone>[A-Z]{2,5}))?"
                      r"(?:\s*\(UTC(?P<off>[+\-−–]\d{1,2}(?::?\d{2})?)\))?")


def parse_published(raw: str) -> dict:
    """'2028-07-18 09:00 PDT (UTC−07:00)' -> local ISO time with offset, and the same instant in UTC."""
    import datetime as dt
    m = STAMP_RE.search(raw or "")
    if not m:
        return {"published_local": "", "published_utc": ""}
    clock = m.group("time") or "00:00"
    off = (m.group("off") or "+00:00").translate(DASHES).replace("−", "-")
    sign = -1 if off[0] == "-" else 1
    hh, _, mm = off[1:].partition(":")
    mm = mm or (hh[2:] if len(hh) > 2 else "0")
    tz = dt.timezone(sign * dt.timedelta(hours=int(hh[:2]), minutes=int(mm or 0)))
    local = dt.datetime.fromisoformat(f"{m.group('date')}T{int(clock.split(':')[0]):02d}:{clock.split(':')[1]}")
    local = local.replace(tzinfo=tz)
    return {"published_local": local.isoformat(), "published_utc":
            local.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def source_header(text: str) -> dict | None:
    """Source records state who published them and when, as 'Source: ...' and 'Published: ...' lines near the top.
    Generic: any document with both lines in its first 1,000 characters is a source record."""
    head = text[:1000]
    s, p = SOURCE_RE.search(head), PUBLISHED_RE.search(head)
    if not (s and p):
        return None
    body = text[p.end():].strip()
    return {"source": s.group("v").strip(), "published": p.group("v").strip(), **parse_published(p.group("v")),
            "body": body}


def link_records(records: list[dict], events: list["Event"]) -> None:
    """Link a source record to the ONE event its title names: an event name of that Games' events must appear as a
    phrase in the title (year from the title, else the publication year). When several events share that name
    (e.g. 'men's singles' in three sports), the sport or venue named in the title or body must single one out;
    otherwise the record stays unlinked. Never guesses."""
    by_year: dict[int, list[Event]] = {}
    for e in events:
        by_year.setdefault(e.year, []).append(e)
    phrase = lambda p, t: bool(p) and re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", t) is not None
    for r in records:
        tn, bn = norm(r["title"]), norm(r.get("body", ""))
        ctx = tn + " " + bn
        ys = [int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", r["title"])]
        if not ys and r.get("published_utc"):
            ys = [int(r["published_utc"][:4])]
        pool = by_year.get(ys[0], []) if ys else []
        # sports named anywhere in the record (the sport name or a form of it, e.g. 'archery', 'cycling')
        mentions = lambda e: phrase(norm(e.sport), ctx) or (bool(e.venue) and phrase(norm(e.venue), ctx))
        cands, how = [], ""
        for where, text, need_sport in (("title", tn, False), ("text", bn, True)):
            cands = [e for e in pool if phrase(norm(e.event), text)]
            # a name that is part of a longer matched name ("men's 10" in "men's 10 metre air rifle") is dropped;
            # two different names left (e.g. men's and women's individual) means the record is about both: no link
            names = {norm(e.event) for e in cands}
            names = {n for n in names if not any(o != n and phrase(n, o) for o in names)}
            cands = [e for e in cands if norm(e.event) in names] if len(names) == 1 else []
            if cands and (need_sport or len(cands) > 1):  # text matches, and shared names, need the sport or venue
                cands = [e for e in cands if mentions(e)]
            if len(cands) == 1:
                how = where
                break
        r["event_id"] = cands[0].id if len(cands) == 1 else ""
        r["event_candidates"] = len(cands)
        r["link_how"] = how if r["event_id"] else ""


# Reliability tiers of a source record, from its Source line (general rules, applied in this order):
#   4 informal: blogs, fan or spectator posts, volunteer notes, tourism or third-party sites, memos, automated emails
#   1 official record: signed / certified documents, juries, certificates, or an official body (jury, federation,
#     inspector, director, organiser) issuing a register, registration record, notice, bulletin, decision or timing data
#   3 news and media: news, bulletins, newsletters, headlines, roundups, aggregators, delegation statements
#   2 operational: feeds, schedules, scoreboards, logs, desks, operations, planning and allocation documents
# A tier says how authoritative the source is, not whether this record is current: content words such as 'draft',
# 'provisional' or 'capture' found in the record are reported separately (provisional_terms).
TIER_LABELS = {1: "official record", 2: "operational", 3: "news/media", 4: "informal"}
_T4 = r"blog|fan|spectator|volunteer|tourism|third[- ]party|memo|analytics|automated|e-?mail|ticket|social media|voice note"
_T1_ALONE = r"signed|certified|certificate|jury"
_T1_BODY = r"jury|federation|inspector|director|chief|referee|organi[sz]ing|committee|LA\d\d|IOC|official"
_T1_FORM = r"register|registration|notice|bulletin|decision|ruling|official|binding|audit|timing data|verification"
_T3 = r"news|bulletin|newsletter|headline|roundup|aggregator|delegation|press|magazine|media desk"
_T2 = r"feed|schedule|scoreboard|log\b|desk|operations|planning|allocation|timing|export|inspector|system|office"
PROVISIONAL_RE = re.compile(r"\b(draft|provisional|preliminary|unofficial|pending|unconfirmed|capture|approx(?:imately)?"
                            r"|estimate[sd]?|unverified)\b", re.I)


def source_tier(source: str) -> int:
    s = source or ""
    has = lambda pat: re.search(pat, s, re.I) is not None
    if has(_T4):
        return 4
    if has(_T1_ALONE) or (has(_T1_BODY) and has(_T1_FORM)):
        return 1
    if has(_T3):
        return 3
    if has(_T2):
        return 2
    return 3


def tag_records(records: list[dict]) -> None:
    for r in records:
        r["tier"] = source_tier(r["source"])
        r["tier_label"] = TIER_LABELS[r["tier"]]
        # words that mark content as provisional or unverified, as found in the record (reported, not judged)
        r["provisional_terms"] = sorted({m.lower() for m in PROVISIONAL_RE.findall(r["title"] + " " + r.get("body", ""))})


def parse_corpus(path: str | Path):
    docs, events, chunks, records, typed = [], [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            kind, fields, _ = parse_infobox(d["text"])
            hdr = None if kind else source_header(d["text"])
            docs.append({"id": d["doc_id"], "title": d["title"], "url": d.get("url") or "",
                         "kind": kind or ("source record" if hdr else "article"),
                         "tokens": d.get("approx_tokens", 0), "synthetic": bool(d.get("synthetic"))})
            if hdr:
                records.append({"id": d["doc_id"], "title": d["title"], **hdr})
            if kind in ("film", "officeholder", "person"):
                typed.append((d, kind, fields))
            ev = None
            oly = fields if kind == "Olympic event" else olympic_infobox(d["text"])
            if oly:
                ev = parse_event(d, oly)
            elif kind and kind.lower() in TOURNAMENT_KINDS and TITLE_RE.match(d["title"]):
                ev = parse_event(d, tournament_fields(fields))
            if ev:
                events.append(ev)
            for i, c in enumerate(chunk_text(d["title"], d["text"])):
                chunks.append({"id": f"{d['doc_id']}#{i}", "doc_id": d["doc_id"], "idx": i, "text": c})
    link_editions(events)
    link_records(records, events)
    tag_records(records)
    from agr.entities import parse_entities
    return docs, events, chunks, records, parse_entities(typed)


def link_editions(events: list[Event]) -> None:
    """Resolve prev/next infobox years into concrete Event ids (same sport + event name + season)."""
    by_key = {}
    for e in events:
        by_key[(norm(e.sport), norm(e.event), e.season, e.year)] = e
    for e in events:
        e.prev_id = getattr(by_key.get((norm(e.sport), norm(e.event), e.season, e.prev_year)), "id", "")
        e.next_id = getattr(by_key.get((norm(e.sport), norm(e.event), e.season, e.next_year)), "id", "")


def _split_joined(seg: str) -> list[str]:
    """Split names stored run together ('Dani KingLaura Trott', 'Herta AnitașMarioara Trașcă'): a lowercase letter
    directly followed by an uppercase letter that starts a lowercase word. 'McKeon' / 'MacDonald' are kept."""
    parts, start = [], 0
    for i in range(1, len(seg) - 1):
        if seg[i - 1].islower() and seg[i].isupper() and seg[i + 1].islower():
            word_start = seg.rfind(" ", 0, i) + 1
            if seg[word_start:i] in ("Mc", "Mac", "De", "Di", "La", "Le"):
                continue
            parts.append(seg[start:i]); start = i
    parts.append(seg[start:])
    return parts


def split_names(s: str) -> list[str]:
    """Split a medal field into names using only clear separators: commas / semicolons / 'and' / 'cox:', and names
    stored run together. Removes '*' marks, '(h)'-style notes, footnote text and glued NOC codes ('USAIan Crocker')."""
    if not s:
        return []
    s = re.sub(r"\([^)]*\)", " ", s).replace("*", " ")
    s = re.sub(r"Indicates .*$", " ", s)
    s = re.sub(r"(?<=[^\W\d_])and (?=[A-ZÀ-Þ])", ", ", s)      # 'Virtueand Scott' -> 'Virtue, Scott'
    s = re.sub(r"\s*\bcox:\s*", ", ", s)
    out = []
    for seg in re.split(r",|;|\n|\s+and\s+", s):
        for p in _split_joined(seg.strip()):
            p = re.sub(r"^(?:[A-Z]{3}\s?)(?=[A-ZÀ-Þ][^\W\d_]*[a-zà-ÿ])", "", p.strip())  # glued / leading NOC code
            p = " ".join(p.split())
            if p:
                out.append(p)
    return out


CREW_EVENT = re.compile(r"doubles|pairs?\b|team|relay|two|four|eight|quadruple|double sculls|coxless|coxed|tandem|"
                        r"synchron|beach|madison|dance|duet|\b470\b|49er|tornado|\bstar\b|soling|yngling|"
                        r"flying dutchman|tempest|nacra|k-2|k-4|c-2|c-4|2-man|4-man|two-man|four-man|keelboat|"
                        r"sprint\b.*team|mixed", re.I)


def person_names(s: str, event: str = "") -> list[str]:
    """Names that are safe to store as Athlete vertices, or [] when the field cannot be split reliably
    (e.g. 'Viktor Ahn Semion Elistratov Vladimir Grigorev' has no separators). Never guesses a split:
    in crew events (doubles, pairs, relays, boats ...) a part longer than two words is treated as unsplit."""
    parts = split_names(s)
    max_words = 2 if CREW_EVENT.search(event or "") else 4
    max_words = 3 if max_words == 2 and len(parts) > 1 else max_words  # separated crew lists may hold 3-word names
    ok = lambda p: (1 <= len(p.split()) <= max_words and not re.search(r"\d|:", p) and len(p) <= 40
                    and not p.isupper() and p.lower() != "not awarded")
    return parts if parts and all(ok(p) for p in parts) else []



if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "data/corpus/corpus.jsonl"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "build")
    out.mkdir(parents=True, exist_ok=True)
    docs, events, chunks, records, ent = parse_corpus(src)
    (out / "films.json").write_text(json.dumps(ent["films"], ensure_ascii=False), encoding="utf-8")
    (out / "people.json").write_text(json.dumps(list(ent["people"].values()), ensure_ascii=False), encoding="utf-8")
    (out / "terms.json").write_text(json.dumps(ent["terms"], ensure_ascii=False), encoding="utf-8")
    (out / "records.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (out / "documents.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    (out / "events.json").write_text(json.dumps([{**asdict(e), "prev_id": e.prev_id, "next_id": e.next_id}
                                                for e in events], ensure_ascii=False))
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    linked = sum(1 for e in events if e.prev_id or e.next_id)
    by_games = {}
    for e in events:
        by_games[e.games] = by_games.get(e.games, 0) + 1
    print(f"documents={len(docs)} events={len(events)} chunks={len(chunks)} events_with_edition_links={linked} "
          f"source_records={len(records)} records_linked_to_event={sum(1 for r in records if r['event_id'])}")
    f, tm = ent["films"], ent["terms"]
    print(f"films={len(f)} (year {sum(x['year'] > 0 for x in f)}, budget {sum(x['budget_low'] >= 0 for x in f)}, "
          f"gross {sum(x['gross_low'] >= 0 for x in f)}) people={len(ent['people'])} "
          f"officeholders={sum(1 for x in ent['people'].values() if x['kind'] == 'officeholder')} office_terms={len(tm)} "
          f"(with predecessor {sum(1 for x in tm if x['predecessor'])}, successor {sum(1 for x in tm if x['successor'])}, "
          f"linked to the successor's term {sum(1 for x in tm if x['next_id'])})")
    print("events per Games (2024+):", {k: v for k, v in sorted(by_games.items()) if int(k[:4]) >= 2024})
