"""Round 2 additions: source-record header parsing, record -> event linking, scoring and metrics helpers."""
import importlib.util
from pathlib import Path

from agr.parse import Event, link_records, parse_published, source_header

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_published_time_to_utc():
    p = parse_published("2028-07-18 09:00 PDT (UTC−07:00)")
    assert p["published_utc"] == "2028-07-18T16:00:00Z"
    assert p["published_local"] == "2028-07-18T09:00:00-07:00"
    assert parse_published("2029-02-10 14:00 CET (UTC+01:00)")["published_utc"] == "2029-02-10T13:00:00Z"
    assert parse_published("no date")["published_utc"] == ""


def test_source_header():
    text = "[Synthetic document: notice.]\n\nSource: Venue volunteer notebook\nPublished: 2028-07-26 21:55 PDT " \
           "(UTC−07:00)\n\nWomen's individual: equipment pause, approx. 14 min."
    h = source_header(text)
    assert h["source"] == "Venue volunteer notebook"
    assert h["published_utc"] == "2028-07-27T04:55:00Z"
    assert h["body"].startswith("Women's individual")
    assert source_header("An ordinary article without a header.") is None


def _ev(i, sport, event, year=2028, venue=""):
    return Event(id=i, title=f"{sport} at the {year} Summer Olympics – {event}", sport=sport, event=event, year=year,
                 season="Summer", games=f"{year} Summer", venue=venue)


def test_link_records_unique_or_nothing():
    events = [_ev("A", "Athletics", "Men's 100 metres"), _ev("B", "Athletics", "Men's 10,000 metres"),
              _ev("T", "Tennis", "Men's singles"), _ev("TT", "Table tennis", "Men's singles", venue="Westside Pavilion")]
    recs = [{"title": "LA 2028 men's 100 metres – scoreboard capture", "body": "", "published_utc": "2028-07-24T04:09:00Z"},
            {"title": "Schedule – men's singles", "body": "", "published_utc": "2028-07-15T23:00:00Z"},
            {"title": "Schedule – men's singles", "body": "Westside Pavilion, table 1", "published_utc": "2028-07-15T23:00:00Z"},
            {"title": "Morning headlines", "body": "", "published_utc": "2028-07-17T19:00:00Z"}]
    link_records(recs, events)
    assert [r["event_id"] for r in recs] == ["A", "", "TT", ""]


def test_score_facts_and_status():
    s = _load("score_public")
    q = {"status": "answered", "answer": [{"year": 2012, "gold": "Yi Siling"}, {"year": 2016, "gold": "Virginia Thrasher"}]}
    full = s.score_one(q, {"status": "answered", "answer": "2012: Yi Siling; 2016: Virginia Thrasher"})
    half = s.score_one(q, {"status": "answered", "answer": "2012: Yi Siling"})
    assert full["score"] == 1.0 and half["score"] == 0.5
    nic = {"status": "not_in_corpus", "answer": None}
    assert s.score_one(nic, {"status": "not_in_corpus", "answer": None})["score"] == 1.0
    assert s.score_one(nic, {"status": "answered", "answer": "Someone"})["score"] == 0.0
    assert s.score_one({"status": "answered", "answer": {"noc": "JPN", "golds": 8}},
                       {"status": "answered", "answer": "JPN with 8 golds"})["score"] == 1.0


def test_metrics_percentiles():
    m = _load("metrics")
    assert m.pct([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert m.pct([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10


def test_trace_has_one_step_per_tigergraph_call(tmp_path, monkeypatch):
    from agr import tg
    from agr.llm import LLMResult
    from agr.pipelines import Harness
    from agr.tools import Observation
    monkeypatch.setattr(tg, "QUERY_LOG", tmp_path / "q.jsonl")
    h = Harness("q1", "agentic", "question?")
    tg.log_call("query", "event_profile", {"eid": "X"}, 3, 12.0, "rest", "2028-01-01T00:00:00.000Z")
    tg.log_call("query", "event_claims", {"eid": "X", "qv": [0.1] * 512}, 0, 8.0, "rest", "2028-01-01T00:00:00.010Z")
    h.observe("event_profile", "event_profile", {"event_id": "X"}, Observation("event_profile", "s", [], True, ms=25.0))
    h.log_llm_step("orchestrator", "decide:final_answer", LLMResult("{}", 100, 20, 300.0, "m"))
    t = h.res.trace
    assert [s["query"] for s in t if s["tigergraph"]] == ["event_profile", "event_claims"]
    assert t[1]["params"]["qv"] == "<512-d vector>"
    assert t[-1]["input_tokens"] == 100 and [s["step"] for s in t] == [1, 2, 3]
    assert len((tmp_path / "q.jsonl").read_text().splitlines()) == 2


def test_source_tiers_from_source_line():
    from agr.parse import source_tier
    assert source_tier("LA28 shooting jury (signed certificate)") == 1
    assert source_tier("World archery federation (registration database)") == 1
    assert source_tier("Venue scoreboard operator") == 2
    assert source_tier("LA28 competition operations (published schedule)") == 2
    assert source_tier("Local sports bulletin") == 3
    assert source_tier("Trackside Notes (spectator blog)") == 4
    assert source_tier("LA28 ticketing (automated email)") == 4
    assert source_tier("Third-party medal data feed (legacy format)") == 4


def test_text_link_needs_sport_and_one_event_name():
    events = [_ev("AW", "Archery", "Women's individual"), _ev("AM", "Archery", "Men's individual"),
              _ev("EW", "Equestrian", "Women's individual")]
    recs = [{"title": "Volunteer notebook, archery field", "body": "Women's individual: pause, approx. 14 min.",
             "published_utc": "2028-07-27T04:55:00Z"},
            {"title": "Registration extract", "body": "Archery. Men's individual: X. Women's individual: Y.",
             "published_utc": "2028-07-27T04:55:00Z"},
            {"title": "Notebook", "body": "Women's individual: pause.", "published_utc": "2028-07-27T04:55:00Z"}]
    link_records(recs, events)
    assert [(r["event_id"], r["link_how"]) for r in recs] == [("AW", "text"), ("", ""), ("", "")]
