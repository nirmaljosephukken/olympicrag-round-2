"""The agent loop end to end without network: scripted LLM replies, in-memory graph queries (tests/fake_tg.py).
Checks statuses decided by the agent and checked in code, multi-evidence grounding, and the Round 2 answer line."""
import json

import pytest

from agr import pipelines, run
from agr.llm import LLMResult
from agr.parse import norm
from tests.fake_tg import BUILD, FakeTG

pytestmark = pytest.mark.skipif(not (BUILD / "films.json").exists(), reason="needs the parsed build/ JSON")


@pytest.fixture()
def tb(tmp_path, monkeypatch):
    from agr import tg
    from agr.tools import Toolbox
    monkeypatch.setattr(tg, "QUERY_LOG", tmp_path / "q.jsonl")
    t = Toolbox.__new__(Toolbox)
    t.tg, t.counter = FakeTG(), 0
    t.docs = {d["id"]: d for d in json.loads((BUILD / "documents.json").read_text(encoding="utf-8"))}
    events = json.loads((BUILD / "events.json").read_text(encoding="utf-8"))
    cat = {"sports": sorted({e["sport"] for e in events}), "venues": sorted({e["venue"] for e in events if e["venue"]}),
           "events": {norm(e["title"]): e["id"] for e in events}}
    t.catalog = lambda: cat
    return t


def script(monkeypatch, replies):
    it = iter(replies)
    monkeypatch.setattr(pipelines, "generate",
                        lambda *a, **k: LLMResult(json.dumps(next(it)), 1000, 50, 5.0, "fake"))


def test_list_answer_from_one_tool_is_answered(tb, monkeypatch):
    script(monkeypatch, [
        {"reasoning": "count by nation", "action": "medal_table",
         "args": {"sport": "Sailing", "year": 2008, "season": "Summer", "medal": "gold"}},
        {"action": "final_answer", "answer": "GBR, with 4 golds", "status": "answered", "citations": ["E1"],
         "explanation": "Counted sailing golds by nation."}])
    r = pipelines.run_agentic(tb, "t1", "Sailing golds by nation at the 2008 Summer Olympics?")
    line = run.answer_line(r)
    assert line["status"] == "answered" and line["answer"] == "GBR, with 4 golds"
    assert line["citations"] and line["input_tokens"] == 2000
    assert [s["query"] for s in line["trace"] if s["tigergraph"]] == ["medal_table"]
    assert sum(s.get("input_tokens", 0) for s in line["trace"]) == line["input_tokens"]


def test_invented_name_is_rejected_then_withheld(tb, monkeypatch):
    bad = {"action": "final_answer", "answer": "GBR and Nobody Atall", "status": "answered", "citations": ["E1"]}
    script(monkeypatch, [{"action": "medal_table", "args": {"sport": "Sailing", "year": 2008, "season": "Summer"}},
                         bad, bad, bad])
    r = pipelines.run_agentic(tb, "t2", "Sailing golds by nation at the 2008 Summer Olympics?")
    assert r.status == "not_in_corpus" and r.stop_reason.startswith("withheld_after_review")


def test_not_in_corpus_needs_a_search_first_and_partial_needs_missing(tb, monkeypatch):
    nic = {"action": "final_answer", "answer": "", "status": "not_in_corpus", "explanation": "no such event"}
    script(monkeypatch, [nic, {"action": "graph_search", "args": {"sport": "Athletics", "year": 2028}}, nic,
                         {"action": "similarity_search", "args": {"query": "made-up event 2028"}},
                         {"action": "text_search", "args": {"phrase": "made-up event"}}, nic])
    r = pipelines.run_agentic(tb, "t3", "Who won the made-up event at the 2028 Summer Olympics?")
    assert r.status == "not_in_corpus" and r.tool_calls == 4 and run.answer_line(r)["answer"] is None
    assert pipelines.answer_fields({"answer": "A", "status": "partial", "missing": []})[2] == "answered"
    assert pipelines.answer_fields({"answer": "A", "status": "answered", "missing": ["2008"]})[2] == "partial"
    assert pipelines.answer_fields({"answer": "unknown", "status": "answered"})[2] == "not_in_corpus"
    assert pipelines.answer_fields({"answer": "No duration is recorded", "status": "answered",
                                    "fact_stated": "no"})[2] == "not_in_corpus"
    assert pipelines.answer_fields({"answer": "A, B", "status": "answered", "fact_stated": "partly"})[2] == "partial"


def test_answer_with_inline_evidence_ids_is_accepted_with_them_as_citations(tb, monkeypatch):
    script(monkeypatch, [{"action": "medal_table", "args": {"sport": "Sailing", "year": 2008, "season": "Summer"}},
                         {"action": "final_answer", "answer": "GBR won 4 golds [E1].", "status": "answered",
                          "citations": []}])
    r = pipelines.run_agentic(tb, "t4", "Sailing golds by nation at the 2008 Summer Olympics?")
    assert r.status == "answered" and r.answer == "GBR won 4 golds." and r.citations[0]["evidence"] == "E1"


def test_history_keeps_recent_observations_in_full():
    h = [("step 1 link", False)] + [(f"step {i} obs " + "x" * 2000, True) for i in range(2, 6)]
    out = pipelines.render_history(h).split("\n")
    assert [len(l) > 1500 for l in out] == [False, False, False, True, True]


def test_status_from_asked_parts():
    f = pipelines.answer_fields
    parts = lambda *xs: [{"part": p, "stated": v} for p, v in xs]
    assert f({"answer": "No duration was ever confirmed.", "asked_parts": parts(("official duration", False))})[2] \
        == "not_in_corpus"
    a = f({"answer": "Two informal claims (11 and 14 min); nothing confirmed.",
           "asked_parts": parts(("what the accounts say", True), ("what is confirmed", False))})
    assert a[2] == "partial" and a[3] == ["what is confirmed"]
    assert f({"answer": "No, absence of a log entry proves nothing.",
              "asked_parts": parts(("does the absence prove it", "true"))})[2] == "answered"
    assert f({"answer": "X", "asked_parts": parts(("a", True)), "missing": ["stale"]})[2] == "answered"


def test_evidence_ids_in_the_answer_text_move_to_citations():
    text, ids = pipelines.strip_evidence_ids("Kim won gold [E15, E16]; Lee took silver (E3). See E7 and E8.")
    assert text == "Kim won gold; Lee took silver. See." and ids == ["E15", "E16", "E3", "E7", "E8"]
    ans, cites, _, _ = pipelines.answer_fields({"answer": "GBR won 4 [E1]", "citations": ["E2"]})
    assert ans == "GBR won 4" and cites == ["E2", "E1"]
    assert pipelines.strip_evidence_ids("Event 2028") == ("Event 2028", [])
