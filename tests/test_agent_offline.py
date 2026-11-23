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
         "args": {"sport": "Rowing", "year": 2024, "season": "Summer", "medal": "gold"}},
        {"action": "final_answer", "answer": "NED, with 4 golds", "status": "answered", "citations": ["E1"],
         "explanation": "Counted rowing golds by nation."}])
    r = pipelines.run_agentic(tb, "t1", "Which nation won the most rowing golds at the 2024 Summer Olympics?")
    line = run.answer_line(r)
    assert line["status"] == "answered" and line["answer"] == "NED, with 4 golds"
    assert line["citations"] and line["input_tokens"] == 2000
    assert [s["query"] for s in line["trace"] if s["tigergraph"]] == ["medal_table"]
    assert sum(s.get("input_tokens", 0) for s in line["trace"]) == line["input_tokens"]


def test_invented_name_is_rejected_then_withheld(tb, monkeypatch):
    bad = {"action": "final_answer", "answer": "NED and Nobody Atall", "status": "answered", "citations": ["E1"]}
    script(monkeypatch, [{"action": "medal_table", "args": {"sport": "Rowing", "year": 2024, "season": "Summer"}},
                         bad, bad, bad])
    r = pipelines.run_agentic(tb, "t2", "Which nation won the most rowing golds at the 2024 Summer Olympics?")
    assert r.status == "not_in_corpus" and r.stop_reason.startswith("withheld_after_review")


def test_not_in_corpus_needs_a_search_first_and_partial_needs_missing(tb, monkeypatch):
    nic = {"action": "final_answer", "answer": "", "status": "not_in_corpus", "explanation": "no such event"}
    script(monkeypatch, [nic, {"action": "graph_search", "args": {"sport": "Athletics", "year": 2028}}, nic])
    r = pipelines.run_agentic(tb, "t3", "Who won the made-up event at the 2028 Summer Olympics?")
    assert r.status == "not_in_corpus" and r.tool_calls == 2 and run.answer_line(r)["answer"] is None
    assert pipelines.answer_fields({"answer": "A", "status": "partial", "missing": []})[2] == "answered"
    assert pipelines.answer_fields({"answer": "A", "status": "answered", "missing": ["2008"]})[2] == "partial"
    assert pipelines.answer_fields({"answer": "unknown", "status": "answered"})[2] == "not_in_corpus"
