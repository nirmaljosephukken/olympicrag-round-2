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


def test_rejected_answer_with_evidence_ids_gets_an_explicit_hint(tb, monkeypatch):
    seen = []
    replies = iter([{"action": "medal_table", "args": {"sport": "Sailing", "year": 2008, "season": "Summer"}},
                    {"action": "final_answer", "answer": "E1 and E2", "status": "answered", "citations": ["E1"]},
                    {"action": "final_answer", "answer": "GBR", "status": "answered", "citations": ["E1"]}])

    def gen(prompt, **k):
        seen.append(prompt)
        return LLMResult(json.dumps(next(replies)), 100, 5, 1.0, "fake")
    monkeypatch.setattr(pipelines, "generate", gen)
    r = pipelines.run_agentic(tb, "t4", "Sailing golds by nation at the 2008 Summer Olympics?")
    assert r.status == "answered" and "Ids belong only" in seen[-1]


def test_history_keeps_recent_observations_in_full():
    h = [("step 1 link", False)] + [(f"step {i} obs " + "x" * 2000, True) for i in range(2, 6)]
    out = pipelines.render_history(h).split("\n")
    assert [len(l) > 1500 for l in out] == [False, False, False, True, True]
