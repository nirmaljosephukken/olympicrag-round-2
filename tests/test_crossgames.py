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
def tb(tmp_path_factory):
    from agr import tg
    from agr.tools import Toolbox
    tg.QUERY_LOG = tmp_path_factory.mktemp("log") / "q.jsonl"
    t = Toolbox.__new__(Toolbox)
    import json
    t.tg, t.counter = FakeTG(), 0
    t.docs = {d["id"]: d for d in json.loads((BUILD / "documents.json").read_text(encoding="utf-8"))}
    return t


@needs_build
def test_repeat_winners_tool_on_corpus(tb):
    o = tb.repeat_winners(sport="Sailing")
    assert o.found and o.data and all(len(set(g["years"])) >= 2 for g in o.data)
    assert all(g["sport"] == "Sailing" and g["medal"] == "gold" for g in o.data)
    assert o.evidence[0].kind == "aggregate" and o.evidence[0].doc_id


@needs_build
def test_participation_tool_on_corpus(tb):
    o = tb.participation_by_edition("Men's javelin throw", sport="Athletics")
    years = [r["year"] for r in o.data["rows"]]
    assert years == sorted(years) and len(years) >= 5 and o.data["max"]["value"] >= o.data["min"]["value"]
    assert tb.tg.calls[-1] == ("event_editions", {"sport_norm": "athletics", "event_norm": "men s javelin throw",
                                                  "season": ""})


@needs_build
def test_medal_table_tool_on_corpus(tb):
    o = tb.medal_table("Sailing", 2008, "Summer")
    events = [e for e in tb.tg.events if e["year"] == 2008 and e["sport"] == "Sailing" and e["gold_noc"]]
    assert sum(r["medals"] for r in o.data) == len(events) and o.data[0]["rank"] == 1


@needs_build
def test_film_and_office_tools_on_corpus(tb):
    o = tb.film_facts(person="Tony Scott", role="director")
    assert o.found and all("director: Tony Scott" in f["credits"] for f in o.data)
    assert any("gross/budget" in e.text for e in o.evidence)
    t = tb.office_succession(office="governor of wyoming")
    assert t.found and all("Governor of Wyoming" in x["office"] for x in t.data)


@needs_build
def test_source_records_in_time_order_with_tiers(tb):
    o = tb.source_records()
    utc = [r["published_utc"] for r in o.data]
    assert utc == sorted(utc) and len(o.data) == len(tb.tg.records)
    assert all("(tier " in e.text for e in o.evidence)


def test_rank_events_ties_and_missing():
    evs = [{"event": "A", "id": "1", "nations": 13}, {"event": "B", "id": "2", "nations": 14},
           {"event": "C", "id": "3", "nations": 14}, {"event": "D", "id": "4", "nations": -1}]
    r = crossgames.rank_events(evs)
    assert r["max"] == {"value": 14, "events": ["B", "C"]} and r["min"] == {"value": 13, "events": ["A"]}
    assert r["missing"] == ["D"]


def test_film_ratio_needs_same_currency():
    f = {"budget_currency": "USD", "gross_currency": "USD", "budget_low": 10e6, "budget_high": 20e6,
         "gross_low": 30e6, "gross_high": 30e6}
    assert crossgames.film_ratio(f) == (1.5, 3.0)
    assert crossgames.film_ratio({**f, "gross_currency": "INR"}) is None


def test_mentions_event_by_name_with_sport_or_by_distinctive_venue_word():
    ev = {"event": "Women's sprint", "sport": "Cycling", "venue": "Dominguez Velodrome"}
    assert crossgames.mentions_event("Cycling bulletin: women's sprint heats", ev)
    assert crossgames.mentions_event("Doors open early at Dominguez today", ev)
    assert not crossgames.mentions_event("Women's sprint (canoe) results", ev)
    assert not crossgames.mentions_event("New seats at the velodrome", ev)
    assert crossgames.mentions_event("Shooting: 30 lined up in the 50 m rifle", {"event": "Men's 50 metre rifle"
                                     if False else "50 metre rifle", "sport": "Shooting", "venue": ""})


@needs_build
def test_rank_events_tool_on_corpus(tb):
    o = tb.rank_events("Sailing", 2008, "Summer")
    vals = [r["value"] for r in o.data["rows"]]
    assert vals == sorted(vals, reverse=True) and "MOST:" in o.summary and "FEWEST:" in o.summary


@needs_build
def test_medal_table_lists_winners(tb):
    o = tb.medal_table("Sailing", 2008, "Summer")
    assert " - " in o.evidence[0].text   # event - medallist


@needs_build
def test_film_ratio_filter_in_code_and_text_fallback(tb):
    o = tb.film_facts(person="Tony Scott", role="director", ratio_below=2)
    assert "(computed in code)" in o.summary
    keep = [f for f in o.data if f["ratio"] and f["ratio"][1] < 2]
    assert all(f["title"] in o.summary for f in keep)
    miss = tb.film_facts(title="Top Gun")   # not an article of its own here: falls back to a text search
    assert miss.found and "Text search" in miss.summary


@needs_build
def test_one_credit_edge_keeps_every_role_of_a_person():
    # a director who also produced the film must stay findable as director (one edge per film-person pair)
    from graph.setup_graph import entity_payload
    from agr.parse import norm
    V, E = entity_payload()
    film = next(i for i, v in V["Film"].items() if v["title"]["value"] == "The Terminal")
    role = E["Film"][film]["FILM_CREDIT"]["Person"][norm("Steven Spielberg")]["role"]["value"]
    assert set(role.split(",")) == {"director", "producer"}


@needs_build
def test_entity_linker_finds_people_athletes_and_films(tb):
    tb.catalog = lambda: {"sports": [], "venues": [], "events": {}}
    o = tb.entity_linker("What did Tony Scott direct, and did Ben Ainslie win in 2008?")
    assert any(p["name"] == "Ben Ainslie" for p in o.data["people"])
    o = tb.entity_linker("Who scored the music for True Romance?")
    assert any(f["name"] == "True Romance" for f in o.data["films"])


@needs_build
def test_event_name_passed_as_sport_still_finds_events(tb):
    import json
    from tests.fake_tg import BUILD
    events = json.loads((BUILD / "events.json").read_text(encoding="utf-8"))
    tb.catalog = lambda: {"sports": sorted({e["sport"] for e in events}), "venues": [], "events": {}}
    o = tb.graph_search(sport="Men's javelin throw", year=2008)
    assert o.found and all("javelin" in e["title"].lower() for e in o.data)
    p = tb.participation_by_edition("Men's javelin throw", sport="javelin throw")
    assert p.found and "(gold:" in p.summary


@needs_build
def test_text_search_is_deterministic_and_one_passage_per_document(tb):
    a, b = tb.text_search("Ben Ainslie"), tb.text_search("Ben Ainslie")
    assert [e.doc_id for e in a.evidence] == [e.doc_id for e in b.evidence]
    assert len({e.doc_id for e in a.evidence}) == len(a.evidence)


@needs_build
def test_linker_names_events_with_years(tb):
    tb.catalog = lambda: {"sports": ["Athletics", "Sailing"], "venues": [], "events": {}}
    o = tb.entity_linker("Who won the men's javelin throw at the 2008 Summer Olympics?")
    assert o.data["events"] and all("2008" in e["title"] and "javelin" in e["title"].lower() for e in o.data["events"])
    assert tb.entity_linker("Who won the women's race?").data["events"] == []
