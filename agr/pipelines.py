"""
Three pipelines over the same TigerGraph instance and the same LLM:

  rag       vector search over all chunks -> answer                               (1 LLM call)
  graphrag  entity linking -> fixed graph retrieval + 1-hop expansion
            + graph-scoped vector search -> answer                                (1 LLM call)
  agentic   orchestrator LLM picks the next specialised agent from the evidence so far, loops,
            an evidence evaluator checks grounding before it is allowed to stop   (2-8 LLM calls)

Every run returns a Result with answer, citations, token accounting and a step-by-step trace.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from agr.config import SETTINGS
from agr.llm import generate, approx_tokens
from agr.parse import norm
from agr import grounding
from agr.tg import take_transports, take_calls
from agr.tools import Toolbox, Observation, Evidence, TOOL_SPECS


# ----------------------------------------------------------------------------- harness
@dataclass
class Step:
    n: int
    agent: str
    action: str
    args: dict
    observation: str
    found: bool
    ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    context_tokens: int = 0
    reasoning: str = ""
    via: str = ""               # transport that served the graph queries of this step (rest / mcp)


@dataclass
class Result:
    qid: str
    pipeline: str
    question: str
    answer: str
    citations: list[dict]
    confidence: str = ""
    steps: list[Step] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    context_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0
    chunks_retrieved: int = 0
    strategy_changes: int = 0
    stop_reason: str = ""
    grounded: bool = False      # every name / number of the answer stated in cited evidence (agr/grounding.py)
    grounding: str = ""         # reason / problem from the grounding check
    notes: list[str] = field(default_factory=list)   # code-written notes (e.g. ambiguity), never LLM text
    reasoning_chain: list[str] = field(default_factory=list)  # multi-hop path built by code from tool results
    trace: list[dict] = field(default_factory=list)
    status: str = ""            # answered | partial | not_in_corpus (decided by the agent, checked in code)
    missing: list[str] = field(default_factory=list)   # requested parts the corpus does not contain (partial)
    explanation: str = ""       # the agent's short account of how it reached the answer   # Round 2 trace: one entry per TigerGraph call / LLM call / step

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


def chain_entry(obs: Observation, args: dict) -> str:
    """One link of the multi-hop reasoning chain, written by code from what the tool actually returned."""
    d, t = obs.data, obs.tool
    titles = lambda xs: "; ".join(x.get("title", "")[:70] for x in xs[:3])
    if not obs.found:
        return f"{t}({json.dumps(args, ensure_ascii=False)[:80]}) -> nothing found"
    if t == "entity_linker" and isinstance(d, dict):
        parts = [f"{k}={v}" for k, v in d.items() if v and k != "exact_event_titles"]
        return "linked: " + ", ".join(parts)[:160]
    if t in ("graph_search", "venue_events", "edition_hop", "athlete_medals") and isinstance(d, list):
        head = {"venue_events": f"venue '{args.get('venue', '')}' + date '{args.get('date', '')}'",
                "edition_hop": f"{args.get('direction', 'prev')} edition of {args.get('event_id', '')}",
                "graph_search": "graph search", "athlete_medals": f"athlete '{args.get('name', '')}'"}[t]
        return f"{head} -> {len(d)} event(s): {titles(d)}"
    if t == "event_profile" and isinstance(d, dict):
        return f"event profile -> {d.get('title', '')[:80]}"
    if t == "aggregate" and isinstance(d, dict):
        return (f"aggregate {args.get('sport', '')} {args.get('year', '')} {args.get('season', '')}: "
                f"{d.get('over_threshold')} of {d.get('events_found')} events over {args.get('threshold')}")
    if t == "multi_hop" and isinstance(d, dict):
        if d.get("ties"):
            return f"multi-hop start ambiguous: {len(d['ties'])} candidate events, no hop taken"
        steps = " -> ".join(c["title"][:70] for c in d.get("chain", []))
        return f"multi-hop ({d.get('how', '')}; path {'>'.join(d.get('path', [])) or 'none'}): {steps}" + \
            (f" [{d['stopped']}]" if d.get("stopped") else "")
    if t == "fact_history" and isinstance(d, dict):
        return (f"versions of {d.get('title', '')[:70]}: {len(d.get('conflicts', []))} conflict(s), "
                f"{len(d.get('changes', []))} result change(s)")
    if t == "find_documents" and isinstance(d, list):
        return f"documents -> {titles(d)}"
    if t in ("read_document", "similarity_search", "scoped_similarity"):
        return f"{t} -> {len(obs.evidence)} passage(s) from {', '.join(sorted({e.title[:60] for e in obs.evidence})[:2])}"
    return ""


class Harness:
    """State for one investigation: evidence ledger, token/time accounting, trace."""

    def __init__(self, qid: str, pipeline: str, question: str):
        self.res = Result(qid, pipeline, question, "", [])
        self.ledger: dict[str, Evidence] = {}
        self.ties: list[list[dict]] = []   # groups of events that match the question equally well
        self.versions: dict[str, dict] = {}  # event id -> fact_history data (conflicts, changes)
        self.t0 = time.time()
        take_calls()  # drop calls made before this question started (e.g. catalogue warm-up)

    def _trace(self, **kw) -> None:
        self.res.trace.append({"step": len(self.res.trace) + 1, **kw})

    def observe(self, agent: str, action: str, args: dict, obs: Observation, reasoning: str = "", llm=None):
        for e in obs.evidence:
            self.ledger[e.id] = e
        if obs.tool == "fact_history" and isinstance(obs.data, dict):
            self.versions[obs.data["event"]] = obs.data
        if obs.tool == "event_profile" and isinstance(obs.data, dict) and \
                (obs.data.get("n_conflicts") or obs.data.get("n_changes")):
            self.versions.setdefault(obs.data.get("id", ""), {"flag": obs.data, "title": obs.data.get("title", "")})
        link = chain_entry(obs, args)
        if link:
            self.res.reasoning_chain.append(link)
        if obs.tool == "venue_events" and isinstance(obs.data, list) and len(obs.data) > 1:
            self.ties.append([{"id": e["id"], "title": e["title"], "date": e.get("date", "")} for e in obs.data])
        if obs.tool == "multi_hop" and isinstance(obs.data, dict) and len(obs.data.get("ties", [])) > 1:
            self.ties.append(obs.data["ties"])
        if obs.tool in ("similarity_search", "scoped_similarity", "read_document"):
            self.res.chunks_retrieved += len(obs.evidence)
        self.res.tool_calls += 1
        ctx = obs.context_tokens or approx_tokens(obs.summary)
        self.res.context_tokens += ctx
        vias = take_transports()
        calls = take_calls()
        for c in calls:   # one trace step per TigerGraph call, as logged in logs/tigergraph_queries.jsonl
            self._trace(tool=action, tigergraph=True, query=c["query"], params=c["params"], rows=c["rows"],
                        ms=c["ms"], ts=c["ts"], via=c["via"], **({"error": c["error"]} if c.get("error") else {}))
        local_ms = round(obs.ms - sum(c["ms"] for c in calls), 1)
        if not calls or local_ms > 50:   # work done in code (entity linking, title matching, embedding ...)
            self._trace(tool=action if not calls else f"{action}:local", tigergraph=False, ms=max(0.0, local_ms))
        self.res.steps.append(Step(len(self.res.steps) + 1, agent, action, args, obs.summary[:1500], obs.found,
                                   round(obs.ms, 1), getattr(llm, "input_tokens", 0),
                                   getattr(llm, "output_tokens", 0), ctx, reasoning, ",".join(sorted(set(vias)))))

    def llm(self, agent: str, prompt: str, system: str | None = None, json_mode: bool = False, max_out: int = 1024):
        r = generate(prompt, system=system, json_mode=json_mode, max_output_tokens=max_out)
        self.res.input_tokens += r.input_tokens
        self.res.output_tokens += r.output_tokens
        self.res.llm_calls += 1
        return r

    def log_llm_step(self, agent: str, action: str, r, reasoning: str = "", args: dict | None = None):
        self._trace(tool="llm_" + agent, tigergraph=False, decision=action, input_tokens=r.input_tokens,
                    output_tokens=r.output_tokens, ms=round(getattr(r, "total_ms", 0) or r.ms, 1),
                    llm_ms=round(r.ms, 1), llm_key=getattr(r, "key", 0), attempts=getattr(r, "attempts", 1))
        self.res.steps.append(Step(len(self.res.steps) + 1, agent, action, args or {}, r.text[:600], True,
                                   round(r.ms, 1), r.input_tokens, r.output_tokens, 0, reasoning))

    def cite(self, ids: list[str]) -> list[dict]:
        out = []
        for i in ids:
            e = self.ledger.get(str(i).strip("[]"))
            if e:
                out.append({"evidence": e.id, "doc_id": e.doc_id, "title": e.title, "url": e.url, "kind": e.kind,
                            "text": e.text[:1500]})
        return out

    def ground(self, answer: str, ids: list) -> "grounding.Grounding":
        """Run the code grounding check on the cited evidence; add a code-written ambiguity note if the answer's
        event was one of several that matched the question equally (so the ambiguity is reported, not merged)."""
        g = grounding.check(answer, [self.ledger[str(i).strip("[]")] for i in ids if str(i).strip("[]") in self.ledger],
                            self.res.question)
        self.res.grounded, self.res.grounding = g.ok, (g.problem + ": " if g.problem else "") + g.reason
        if g.ok:
            used = next((self.ledger[str(i).strip("[]")] for i in ids if str(i).strip("[]") in self.ledger), None)
            if used:
                self.res.reasoning_chain.append(f"answer '{answer[:80]}' <- {used.id} ({used.title})")
            for eid in g.events:
                v = self.versions.get(eid)
                if not v:
                    continue
                if v.get("conflicts"):
                    self.res.notes.append(f"Conflicting versions in the corpus for {v['title']}: the official record "
                                          f"({v['conflicts'][0]['official']}) and the article text "
                                          f"('{v['conflicts'][0]['text']}') disagree. The answer uses the official record.")
                elif v.get("flag", {}).get("conflict_claims"):
                    c = v["flag"]["conflict_claims"][0]
                    self.res.notes.append(f"Conflicting versions in the corpus for {v['title']}: the official record "
                                          f"(competitors/nations {c['official']}) and the article text "
                                          f"('{c['text']}', i.e. {c['text_value']}) disagree. "
                                          f"The answer uses the official record.")
                if v.get("changes") or v.get("flag", {}).get("change_claims"):
                    v = {**v, "changes": v.get("changes") or v["flag"]["change_claims"]}
                    self.res.notes.append(f"The result of {v['title']} changed after the event "
                                          f"({len(v['changes'])} statement(s), e.g. '{v['changes'][0]['text'][:160]}'). "
                                          f"The official record reflects the current result.")
            for group in self.ties:
                if any(e["id"] in g.events for e in group):
                    chosen = next(e["title"] for e in group if e["id"] in g.events)
                    note = (f"Ambiguous in the corpus: {len(group)} events match this venue and date equally "
                            f"({'; '.join(e['title'] for e in group)}). This answer is for: {chosen}.")
                    if note not in self.res.notes:
                        self.res.notes.append(note)
        return g

    def finish(self) -> Result:
        self.res.latency_ms = round((time.time() - self.t0) * 1000, 1)
        return self.res


ANSWER_RULES = """Answer rules:
- Use ONLY the evidence. Every name and number in the answer must be stated in an evidence item you cite (several
  items may be combined). Evidence ids (E3) go in "citations", never in the answer text. Write names as the
  evidence writes them (a nation given as a code such as JPN stays JPN, not a country name you supply).
- Answer what is asked, completely: a list holds every item the evidence supports and nothing else, each labelled
  (year, event, medal, winner); a count gives the number and what was counted; write numbers as digits (8, not eight);
  a "why" question gives the reason the evidence states.
- asked_parts: split the question into the facts it asks for and mark each one: "stated": true only if the
  evidence gives that fact itself (a name, number, time, place, reason, list). A fact the evidence only calls unknown,
  anonymous, unverified, unconfirmed, provisional or not recorded is NOT stated (false). When the question asks
  whether something is established, proven or superseded, that judgement is the fact: true when you can give it.
- status: "answered" when the evidence answers every part; "partial" when some parts are present and others are not in
  the corpus (answer the present parts, list the absent ones in "missing"); "not_in_corpus" when the specific fact
  asked for is absent. If the evidence only shows that the asked fact is unknown, anonymous, unverified or not
  recorded, that fact is not in the corpus: not_in_corpus, or partial when other asked parts are present. A question
  ASKING whether something is established or proven (e.g. "does X prove Y?") is answered directly: say what the
  evidence does and does not establish.
Evidence policy (sources can disagree or change over time):
- The most authoritative evidence for the time asked wins: tier 1 (signed or official decisions, juries,
  certificates, registers) > tier 2 (scoreboards, schedules, feeds, logs) > tier 3 (news, newsletters) > tier 4
  (blogs, fan posts, volunteer or spectator notes, tourism sites, automated emails).
- A later publication does not override a signed decision; a record that repeats older data does not supersede it.
  For "as of" questions use what was effective at that time; state the effective time when the evidence gives one.
- Registrations or entries are not starters or competitors.
- Missing evidence is not a negative fact: no record means not established, not that it did not happen.
- State what was provisional, superseded, disputed or never confirmed, and by which source."""


def _answer_json(h: Harness, agent: str, question: str, context: str) -> dict:
    prompt = (f"{ANSWER_RULES}\n\nEvidence (each item has an id in brackets):\n{context}\n\n"
              f"Question: {question}\n\nReturn JSON: {{\"answer\": str, "
              f"\"asked_parts\": [{{\"part\": str, \"stated\": bool}}], "
              f"\"status\": \"answered|partial|not_in_corpus\", "
              f"\"missing\": [str], \"explanation\": str, \"citations\": [evidence ids], "
              f"\"confidence\": \"high|medium|low\"}}")
    r = h.llm(agent, prompt, json_mode=True, max_out=900)
    j = r.json()
    h.log_llm_step(agent, "generate_answer", r)
    return j


# ----------------------------------------------------------------------------- 1. RAG
def run_rag(tb: Toolbox, qid: str, question: str, k: int | None = None) -> Result:
    h = Harness(qid, "rag", question)
    obs = tb.similarity_search(question, k=k or SETTINGS.rag_k)
    h.observe("similarity_search", "chunk_search", {"k": k or SETTINGS.rag_k}, obs)
    j = _answer_json(h, "answer_generator", question, obs.summary)
    h.res.answer, h.res.confidence = str(j.get("answer", "")), j.get("confidence", "")
    h.res.citations = h.cite(j.get("citations", []))
    h.ground(h.res.answer, j.get("citations", []))
    h.res.stop_reason = "single_pass"
    return h.finish()


# ----------------------------------------------------------------------------- 2. GraphRAG
def run_graphrag(tb: Toolbox, qid: str, question: str) -> Result:
    """Fixed retrieval sequence: link -> graph retrieval (+1-hop expansion) -> scoped chunks -> answer."""
    h = Harness(qid, "graphrag", question)
    link = tb.entity_linker(question)
    h.observe("entity_linker", "link", {}, link)
    L = link.data
    year = L["years"][0] if L["years"] else -1
    sport = L["sports"][0] if L["sports"] else ""
    obs_list = []
    if L["exact_event_titles"]:
        for t in L["exact_event_titles"][:2]:
            obs_list.append(("event_profile", {"event_id": t["event_id"]}, tb.event_profile(t["event_id"])))
    elif L["venues"]:
        obs_list.append(("venue_events", {"venue": L["venues"][0], "date": L["date_phrase"], "year": year},
                         tb.venue_events(L["venues"][0], L["date_phrase"], year)))
    elif sport or year > 0:
        o = tb.graph_search(sport=sport, year=year, season=L["season"], limit=40)
        obs_list.append(("graph_search", {"sport": sport, "year": year, "season": L["season"]}, o))
    for name, args, o in obs_list:
        h.observe("graph_retriever", name, args, o)
    # 1-hop expansion: neighbourhood of the (few) retrieved events, as a standard GraphRAG context
    events = [e for _, _, o in obs_list for e in (o.data if isinstance(o.data, list) else [o.data] if o.data else [])]
    if 0 < len(events) <= 3:
        for e in events:
            if "previous_edition" not in e:
                p = tb.event_profile(e["id"])
                h.observe("graph_retriever", "event_profile(1-hop)", {"event_id": e["id"]}, p)
    doc_ids = list({e["id"] for e in events})[:30]
    if doc_ids:
        sc = tb.scoped_similarity(question, doc_ids, k=3)
        h.observe("scoped_similarity", "chunk_search_in_docs", {"docs": len(doc_ids)}, sc)
    else:
        sc = tb.similarity_search(question, k=6)
        h.observe("similarity_search", "chunk_search(fallback)", {"k": 6}, sc)
    context = "\n".join(f"[{e.id}] ({e.title}) {e.text}" for e in h.ledger.values())
    j = _answer_json(h, "answer_generator", question, context)
    h.res.answer, h.res.confidence = str(j.get("answer", "")), j.get("confidence", "")
    h.res.citations = h.cite(j.get("citations", []))
    h.ground(h.res.answer, j.get("citations", []))
    h.res.stop_reason = "single_pass"
    return h.finish()


# ----------------------------------------------------------------------------- 3. Agentic GraphRAG
ORCH_SYSTEM = """You orchestrate tools over a TigerGraph graph built from a corpus: Wikipedia articles (Olympic events
1988-2026, films, people), fictional LA 2028 event articles, and LA 2028 source records that state their source and
publication time (some are provisional, out of date or wrong).
Graph: Event -IN_GAMES-> Games; -OF_SPORT-> Sport; -AT_VENUE-> Venue; -WON_MEDAL-> Athlete; -NEXT/PREV_EDITION-> Event;
Doc -HAS_CHUNK-> Chunk(text, vector); SourceRecord -ABOUT_EVENT-> Event; Film -FILM_CREDIT-> Person -HELD-> OfficeTerm.

Tools (one per turn):
{tools}

Choose the most useful next action. Use the tool that computes what is asked; never count, rank or do arithmetic
yourself: repeat_winners (same event won at several Games), participation_by_edition (one event across Games),
rank_events (events of one Games by nations/competitors), medal_table (medals per nation, with winners),
athlete_medals (one athlete's medals), film_facts (films; ratio filters), office_succession (politicians).
Chains ("held immediately before/after", "two Games earlier"): multi_hop. Changed results: fact_history.
LA 2028 results, incidents, times, venues, entries: source_records(event_id) gives the event article plus every
source record about it, in time order with tiers; use it before answering. Reasons ("why"): read_document /
text_search / similarity_search. A name not found by a graph tool may be mentioned inside another article: text_search.
If a call returns nothing, change tool or filter. Before not_in_corpus, try similarity_search and a document search
(find_documents or text_search).
Stop when the evidence answers the question: action "final_answer".

{rules}

Reply with JSON only:
{{"reasoning": "<short>", "action": "<tool or final_answer>", "args": {{...}},
  "answer": "<final_answer only>", "asked_parts": [{{"part": "<fact asked>", "stated": true}}],
  "status": "answered|partial|not_in_corpus",
  "missing": ["<part>"],
  "explanation": "<one sentence>", "citations": ["E3"], "confidence": "high|medium|low"}}"""

TOOL_FAMILY = {"entity_linker": "link", "graph_search": "graph", "venue_events": "graph", "event_profile": "graph",
               "edition_hop": "graph", "aggregate": "graph", "athlete_medals": "graph",
               "similarity_search": "vector", "scoped_similarity": "vector", "read_document": "document",
               "find_documents": "document", "fact_history": "graph", "multi_hop": "graph", "source_records": "source",
               "repeat_winners": "graph", "participation_by_edition": "graph", "medal_table": "graph",
               "film_facts": "graph", "office_succession": "graph", "rank_events": "graph", "text_search": "document"}


FEEDBACK = {
    "unsupported": "Part of the answer is not in the cited evidence. Cite every item that states it, write names and "
                   "numbers exactly as the evidence does (e.g. a nation code such as JPN, not a country name), "
                   "or gather it first. If no evidence states it, remove it from the answer and list that part "
                   "as not stated in asked_parts (the status becomes partial); never keep an unstated name.",
    "uncited": "Cite the evidence ids that state the answer.",
    "unknown": "Keep investigating with a different tool or filter before giving up.",
    "ids": "You wrote evidence ids (like E7) inside the answer text. Ids belong only in \"citations\"; write the names, "
           "values and years themselves in the answer.",
}
TEXT_TOOLS = {"find_documents", "text_search", "read_document"}
KEEP_FULL = 2          # the last N tool observations are shown in full; older ones are shortened
OLD_OBS_CHARS = 700
OBS_CHARS = 3500


def render_history(history: list[tuple[str, bool]]) -> str:
    """Investigation log for the orchestrator: recent observations in full, older ones shortened (token budget)."""
    obs_idx = [i for i, (_, is_obs) in enumerate(history) if is_obs]
    recent = set(obs_idx[-KEEP_FULL:])
    out = []
    for i, (text, is_obs) in enumerate(history):
        if is_obs and i not in recent and len(text) > OLD_OBS_CHARS:
            text = text[:OLD_OBS_CHARS] + " ...[shortened]"
        out.append(text)
    return "\n".join(out)


def ground_args(action: str, args: dict, linked: dict) -> dict:
    """Argument grounding: keep tool arguments faithful to the question text.
    If the orchestrator shortened the question's date phrase, restore the verbatim phrase the linker found
    (exact date-text matching in venue_events depends on it)."""
    if not isinstance(args, dict):
        return {}
    phrase = (linked or {}).get("date_phrase", "")
    if action in ("venue_events", "graph_search", "multi_hop") and phrase and args.get("date"):
        if norm(args["date"]) and norm(args["date"]) in norm(phrase) and norm(args["date"]) != norm(phrase):
            args = {**args, "date": phrase}
    return args


def run_agentic(tb: Toolbox, qid: str, question: str, max_steps: int | None = None) -> Result:
    max_steps = max_steps or SETTINGS.max_agent_steps
    h = Harness(qid, "agentic", question)
    system = ORCH_SYSTEM.format(tools="\n".join(f"- {k}{v}" for k, v in TOOL_SPECS.items()), rules=ANSWER_RULES)
    # the entity linker always runs first: it is cheap (no LLM) and grounds every later decision
    link = tb.entity_linker(question)
    h.observe("entity_linker", "link", {"question": question}, link)
    history: list[tuple[str, bool]] = [(f"step 1 entity_linker -> {link.summary}", False)]
    used: set[str] = set()
    last_family, last_found, rejections, nic_pushbacks = "link", True, 0, 0
    max_rejections = 2
    for step in range(max_steps):
        remaining = max_steps - step
        prompt = (f"Question: {question}\n\nInvestigation so far:\n" + render_history(history) +
                  f"\n\nSteps remaining: {remaining}. "
                  + ("You MUST give final_answer now." if remaining == 1 else "Choose the next action."))
        r = h.llm("orchestrator", prompt, system=system, json_mode=True, max_out=500)
        d = r.json()
        action, args = d.get("action", ""), d.get("args") or {}
        if action == "final_answer" and isinstance(args, dict):
            d = {**{k: args[k] for k in ANSWER_KEYS if k in args}, **{k: v for k, v in d.items() if v}}
        args = ground_args(action, args, link.data)
        h.log_llm_step("orchestrator", f"decide:{action}", r, d.get("reasoning", ""), args)
        if action == "final_answer" or not action:
            ans, cites, status, missing = answer_fields(d)
            if status == "not_in_corpus":
                todo = ([] if "similarity_search" in used else ["similarity_search"]) + \
                       ([] if used & TEXT_TOOLS else ["find_documents or text_search"])
                if todo and remaining > 2 and nic_pushbacks < 2:
                    nic_pushbacks += 1   # never conclude not_in_corpus after a single empty search
                    history.append((f"step {len(history) + 1} evidence_evaluator: before not_in_corpus, also try "
                                    f"{' and '.join(todo)} (missing evidence is not a negative fact).", False))
                    continue
                h._trace(tool="evidence_evaluator", tigergraph=False, ok=True, ms=0.0)
                return not_in_corpus(h, d, cites, "the agent found no evidence that answers the question")
            g = h.ground(ans, cites)
            h.res.steps.append(Step(len(h.res.steps) + 1, "evidence_evaluator", "check_grounding",
                                    {"answer": ans, "citations": cites}, h.res.grounding, g.ok, 0.0))
            h._trace(tool="evidence_evaluator", tigergraph=False, ok=g.ok, ms=0.0)
            if g.ok:
                return accept(h, d, ans, cites, missing, "evidence_sufficient: " + g.reason)
            if rejections >= max_rejections or remaining <= 1:
                return withhold(h, g)
            rejections += 1
            hint = FEEDBACK["ids"] if re.search(r"\bE\d+\b", g.reason) else FEEDBACK.get(g.problem, "")
            history.append((f"step {len(history) + 1} evidence_evaluator REJECTED answer '{ans[:300]}': {g.reason}. "
                            f"{hint}", False))
            continue
        fn = getattr(tb, action, None)
        if action not in TOOL_SPECS or fn is None:
            history.append((f"step {len(history) + 1} invalid action '{action}'", False))
            continue
        try:
            obs = fn(**args)
        except Exception as ex:  # bad args from the LLM -> tell it, keep going
            obs = Observation(action, f"error: {type(ex).__name__}: {str(ex)[:200]}", [], False)
        fam = TOOL_FAMILY.get(action, action)
        if fam != last_family and not last_found:
            h.res.strategy_changes += 1
        last_family, last_found = fam, obs.found
        h.observe(action, action, args, obs, d.get("reasoning", ""))
        used.add(action)
        summary = obs.summary if len(obs.summary) < OBS_CHARS else obs.summary[:OBS_CHARS] + " ...[truncated]"
        history.append((f"step {len(history) + 1} {action}({json.dumps(args, ensure_ascii=False)}) -> {summary}", True))
    # budget exhausted: answer from what we have
    context = "\n".join(f"[{e.id}] ({e.title}) {e.text}" for e in h.ledger.values())
    j = _answer_json(h, "answer_generator", question, context[:24000])
    ans, cites, status, missing = answer_fields(j)
    if status == "not_in_corpus":
        return not_in_corpus(h, j, cites, "step budget exhausted without evidence that answers the question")
    g = h.ground(ans, cites)
    return accept(h, j, ans, cites, missing, "step_budget_exhausted: " + g.reason) if g.ok else withhold(h, g)


ANSWER_KEYS = ("answer", "asked_parts", "fact_stated", "status", "missing", "explanation", "citations",
               "confidence")
NOT_FOUND = {"", "unknown", "none", "n a", "not in corpus", "not in the corpus", "not_in_corpus", "null"}


def answer_fields(d: dict) -> tuple[str, list[str], str, list[str]]:
    """(answer text, citation ids, status, missing parts) from the agent's JSON, with the status decided in code:
    an empty / 'unknown' answer is not_in_corpus; otherwise the agent's asked_parts decide it: no part stated ->
    not_in_corpus, some parts not stated -> partial (those parts are the missing ones), all stated -> answered.
    Without asked_parts, the missing list (or the older fact_stated field) decides."""
    ans = d.get("answer")
    ans = json.dumps(ans, ensure_ascii=False) if isinstance(ans, (list, dict)) else str(ans or "")
    cites = [str(c) for c in (d.get("citations") or [])] if isinstance(d.get("citations"), list) else []
    missing = [str(m) for m in (d.get("missing") or []) if str(m).strip()] if isinstance(d.get("missing"), list) else []
    status = str(d.get("status") or "").strip().lower()
    stated = str(d.get("fact_stated") or "").strip().lower()
    parts = [x for x in (d.get("asked_parts") or []) if isinstance(x, dict) and str(x.get("part", "")).strip()] \
        if isinstance(d.get("asked_parts"), list) else []
    if parts:
        is_true = lambda v: v is True or str(v).strip().lower() in ("true", "yes", "1")
        absent = [str(x["part"]) for x in parts if not is_true(x.get("stated"))]
        stated = "no" if len(absent) == len(parts) else ("partly" if absent else "yes")
        missing = missing or absent
    if status == "not_in_corpus" or norm(ans) in NOT_FOUND or stated == "no":
        return ans, cites, "not_in_corpus", missing
    if stated == "partly" and not missing:
        missing = ["part of what was asked is not stated in the corpus"]
    if stated == "yes":
        return ans, cites, "answered", []
    return ans, cites, "partial" if missing else "answered", missing


def accept(h: Harness, d: dict, ans: str, cites: list[str], missing: list[str], reason: str) -> Result:
    h.res.answer, h.res.confidence = ans, d.get("confidence", "")
    h.res.citations = h.cite(cites)
    h.res.status, h.res.missing = ("partial" if missing else "answered"), missing
    h.res.explanation = str(d.get("explanation") or "")
    h.res.stop_reason = reason
    return h.finish()


def not_in_corpus(h: Harness, d: dict, cites: list[str], reason: str) -> Result:
    """The agent concluded the corpus does not answer the question. Cited items (e.g. a record that lists anonymous
    entries) are kept as the evidence for that conclusion."""
    h.res.answer, h.res.status, h.res.confidence = "", "not_in_corpus", d.get("confidence", "")
    said = d.get("answer")
    if said and norm(str(said)) not in NOT_FOUND:   # what the agent did find goes into the explanation
        h.res.notes.append(f"What the corpus does say: {str(said)[:600]}")
    h.res.citations = h.cite(cites)
    h.res.missing = [str(m) for m in (d.get("missing") or [])] if isinstance(d.get("missing"), list) else []
    h.res.explanation = str(d.get("explanation") or "") or reason
    h.res.grounded, h.res.grounding = True, "not_in_corpus: " + reason
    h.res.stop_reason = "not_in_corpus: " + reason
    return h.finish()


def withhold(h: Harness, g) -> Result:
    """Never release a rejected answer: report not_in_corpus and say why, in code."""
    h.res.answer, h.res.status, h.res.confidence, h.res.citations = "", "not_in_corpus", "low", []
    h.res.grounded, h.res.grounding = False, "withheld after review"
    h.res.notes.append(f"Answer withheld: the proposed answer failed the grounding check ({g.reason}).")
    for group in h.ties:
        h.res.notes.append("Candidates that match the question equally: " + "; ".join(e["title"] for e in group) + ".")
    h.res.explanation = "No answer could be grounded in the evidence found."
    h.res.stop_reason = f"withheld_after_review: {g.reason}"
    return h.finish()


PIPELINES = {"rag": run_rag, "graphrag": run_graphrag, "agentic": run_agentic}
