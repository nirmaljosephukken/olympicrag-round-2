"""Cross-Games, film and office tools: logic on synthetic rows, and the tools end to end on the parsed build/ JSON
through an in-memory stand-in for the installed queries (tests/fake_tg.py). No TigerGraph, no LLM."""
import pytest

from agr import crossgames
from agr.entities import link_terms, office_key, parse_money, parse_offices

from tests.fake_tg import BUILD, FakeTG


def test_repeat_winners_groups_by_event_and_distinct_years():
    rows = [{"name": "A B", "rows": ["e1|2012|Summer|Judo|Men's 60 kg|gold", "e2|2016|Summer|Judo|Men's 60 kg|gold",
                                     "e3|2016|Summer|Judo|Men's 66 kg|gold"]},
            {"name": "C D", "rows": ["e4|2012|Summer|Judo|Men's 73 kg|gold", "e4|2012|Summer|Judo|Men's 73 kg|gold"]}]
    out = crossgames.repeat_winners(rows)
    assert [(g["athlete"], g["event"], g["years"]) for g in out] == [("A B", "Men's 60 kg", [2012, 2016])]
    assert crossgames.repeat_winners(rows, 3) == []


def test_participation_max_min_ties_and_missing():
    evs = [{"year": 2000, "nations": 14, "id": "a"}, {"year": 2004, "nations": 16, "id": "b"},
           {"year": 2008, "nations": 16, "id": "c"}, {"year": 2012, "nations": -1, "id": "d"}]
    p = crossgames.participation(evs)
    assert p["max"] == {"value": 16, "years": [2004, 2008]} and p["min"] == {"value": 14, "years": [2000]}
    assert p["missing_years"] == [2012]


def test_medal_counts_rank_ties():
    rows = crossgames.medal_counts({"JPN": 3, "USA": 3, "FRA": 1}, {})
    assert [(r["noc"], r["rank"]) for r in rows] == [("JPN", 1), ("USA", 1), ("FRA", 3)]


def test_money_and_offices():
    assert parse_money("$65–70million")["low"] == 65e6 and parse_money("$65–70million")["high"] == 70e6
    assert parse_money("15 crore") == {"currency": "INR", "low": 1.5e8, "high": 1.5e8, "note": ""}
    assert parse_money("$12.5 million (domestic only)")["note"] == "domestic only"
    assert parse_money("")["low"] == -1
    assert office_key("72nd Governor of Maine") == "governor of maine"
    a = parse_offices({"doc_id": "A", "title": "Ann"}, {"name": "Ann", "order": "5th Governor of X",
                                                         "term_start": "1990", "term_end": "1994", "successor": "Bo",
                                                         "office2": "Mayor of Y", "term_start2": "1980"})
    b = parse_offices({"doc_id": "B", "title": "Bo"}, {"name": "Bo", "order1": "6th Governor of X",
                                                       "term_start1": "1994", "predecessor1": "Ann"})
    terms = a["terms"] + b["terms"]
    link_terms(terms, {a["person"]["id"]: a["person"], b["person"]["id"]: b["person"]})
    assert [(t["office"], t["next_id"]) for t in terms] == [("5th Governor of X", "B#term1"), ("Mayor of Y", ""),
                                                            ("6th Governor of X", "")]


needs_build = pytest.mark.skipif(not (BUILD / "films.json").exists(), reason="run: python -m agr.parse ... build")


@pytest.fixture(scope="module")
def tb():
    from agr.tools import Toolbox
    t = Toolbox.__new__(Toolbox)
    import json
    t.tg, t.counter = FakeTG(), 0
    t.docs = {d["id"]: d for d in json.loads((BUILD / "documents.json").read_text(encoding="utf-8"))}
    return t


@needs_build
def test_repeat_winners_tool_on_corpus(tb):
    o = tb.repeat_winners(sport="Judo")
    assert o.found and o.data and all(len(set(g["years"])) >= 2 for g in o.data)
    assert all(g["sport"] == "Judo" and g["medal"] == "gold" for g in o.data)
    assert o.evidence[0].kind == "aggregate" and o.evidence[0].doc_id


@needs_build
def test_participation_tool_on_corpus(tb):
    o = tb.participation_by_edition("Men's hammer throw", sport="Athletics")
    years = [r["year"] for r in o.data["rows"]]
    assert years == sorted(years) and len(years) >= 5 and o.data["max"]["value"] >= o.data["min"]["value"]
    assert tb.tg.calls[-1] == ("event_editions", {"sport_norm": "athletics", "event_norm": "men s hammer throw",
                                                  "season": ""})


@needs_build
def test_medal_table_tool_on_corpus(tb):
    o = tb.medal_table("Wrestling", 2024, "Summer")
    events = [e for e in tb.tg.events if e["year"] == 2024 and e["sport"] == "Wrestling" and e["gold_noc"]]
    assert sum(r["medals"] for r in o.data) == len(events) and o.data[0]["rank"] == 1


@needs_build
def test_film_and_office_tools_on_corpus(tb):
    o = tb.film_facts(person="Steven Spielberg", role="director")
    assert o.found and all("director: Steven Spielberg" in f["credits"] for f in o.data)
    assert any("gross/budget" in e.text for e in o.evidence)
    t = tb.office_succession(office="governor of missouri")
    assert t.found and all("Governor of Missouri" in x["office"] for x in t.data)


@needs_build
def test_source_records_in_time_order_with_tiers(tb):
    o = tb.source_records()
    utc = [r["published_utc"] for r in o.data]
    assert utc == sorted(utc) and len(o.data) == len(tb.tg.records)
    assert all("(tier " in e.text for e in o.evidence)
