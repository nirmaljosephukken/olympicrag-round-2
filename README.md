# OlympicRAG, Round 2: agentic GraphRAG on TigerGraph

Round 2 system for the TigerGraph **Agentic GraphRAG Hackathon**. One TigerGraph graph holds the whole Round 2 corpus
(3,530 documents: the 2,951 Round 1 Wikipedia articles, 299 Paris 2024 and 114 Milano Cortina 2026 event articles,
127 fictional LA 2028 event articles and 39 fictional LA 2028 source records). An orchestrator LLM answers each question
by calling specialised tools over that graph, one at a time, and code checks every answer before it is released.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.lock   # Python 3.12.10; Linux/macOS: .venv/bin/pip
cp .env.example .env                                                     # TigerGraph host + secret, Gemini key(s)
./build.sh data/corpus/corpus.jsonl                                      # wake workspace, parse, embed, schema, load,
                                                                         # install queries, graph_counts.json
./run.sh data/questions/questions_public.jsonl answers_public.jsonl      # resumable; --fresh starts over
./run.sh data/questions/questions_hidden.jsonl answers_hidden.jsonl
python scripts/score_public.py answers_public.jsonl                     # fact-by-fact score against the public key
python scripts/metrics.py                                                # metrics.json from the answer files
python scripts/extract_query_log.py answers_public.jsonl logs/tigergraph_queries_public.jsonl
python scripts/check_submission.py answers_public.jsonl answers_hidden.jsonl --metrics metrics.json   # brief's rules
python -m pytest -q tests                                                # offline tests (no TigerGraph, no LLM)
```

On Windows run the `.sh` scripts from Git Bash. `requirements.lock` is the exact `pip freeze` of the environment the results
were produced with; `requirements.txt` lists the direct dependencies, pinned. `.env.example` lists every setting.
Other entry points: `./run.sh QUESTIONS ANSWERS --pipeline rag|graphrag` (one-call baselines), `python -m agr.ask "..."`
(one question with its full trace), `python scripts/compare_runs.py RUN_A RUN_B` (repeatability),
`python scripts/make_heldout_questions.py` (held-out questions with code-computed answers), `python scripts/reproduce.py answers_hidden.jsonl QID ...` (re-run submitted questions and compare).
The judging-call runbook is [docs/judging.md](docs/judging.md).

## Architecture

### 1. Documents become graph data (no LLM)

`agr/parse.py`, `agr/entities.py` and `graph/setup_graph.py` turn the corpus into graph data deterministically:

| From | Vertices | Edges |
|---|---|---|
| every document | `Doc`, `Chunk` (text + 512-d vector, HNSW index) | `HAS_CHUNK` |
| `[Infobox Olympic event]` (2,743 events: 2,203 Round 1 + 540 new) | `Event`, `Games`, `Sport`, `Venue`, `Athlete`, `Nation` | `DESCRIBES`, `IN_GAMES`, `OF_SPORT`, `AT_VENUE`, `WON_MEDAL(medal)`, `MEDAL_FOR(medal)`, `NEXT_EDITION` / `PREV_EDITION`, `NEXT_GAMES` |
| result changes and count statements in event text | `Claim` | `HAS_CLAIM`, `SUPERSEDES` |
| `Source:` / `Published:` header (39 LA 2028 records) | `SourceRecord` (source, published, UTC time, reliability tier) | `HAS_RECORD`, `ABOUT_EVENT` (23 linked) |
| `[Infobox film]` (546) and `[Infobox officeholder]` (73) | `Film`, `Person`, `OfficeTerm` (240) | `FILM_CREDIT(role)`, `HELD`, `PRECEDED_BY`, `SUCCEEDED_BY`, `NEXT_TERM` |

Counts after loading `corpus.jsonl` are in `graph_counts.json` (47,074 vertices, 65,948 edges). The graph name comes
from `TG_GRAPH` (default `OlympicRAG2`); the GSQL files use an `@GRAPH@` placeholder. Embeddings are local and
deterministic (`minishlab/potion-retrieval-32M`, no API). A source record is linked to an event only when its title, or
its text together with the sport or venue, names exactly one event; otherwise it stays unlinked. Each record gets a
reliability tier from its Source line: 1 official record (signed decisions, juries, certificates, registers, official
timing), 2 operational (feeds, schedules, scoreboards, logs), 3 news / media, 4 informal (blogs, fan or spectator
posts, volunteer notes, tourism sites, automated emails).

### 2. Tools (installed GSQL queries + code)

All tools query TigerGraph (`graph/queries.gsql`, 20 installed queries) through the official TigerGraph MCP server
(`TG_TRANSPORT=mcp`, RESTPP fallback). Counting, ranking and max / min are done in code (`agr/crossgames.py`), never by
the LLM.

| Tool | What it returns |
|---|---|
| `entity_linker` (always first, no LLM) | years, season, sports, venues, event titles, named events with their source-record count, people, medallists and films named in the question |
| `graph_search`, `venue_events`, `event_profile`, `edition_hop`, `multi_hop` | events by filters / venue + date; one event's facts, medallists, editions, versions; chains such as "the Games immediately before" |
| `rank_events`, `aggregate` | a Games' events of a sport ranked by nations or competitors; threshold counts |
| `medal_table`, `repeat_winners`, `participation_by_edition`, `athlete_medals` | medals per nation with medallists; athletes who won the same event at several Games; count and winner per edition of one event, max / min; one athlete's medals |
| `fact_history` | the official record, result changes and conflicting statements of an event |
| `source_records` | an event's article plus every source record about it (linked, or naming the event or its venue), in publication-time order with tiers and provisional wording |
| `film_facts`, `office_succession` | films (credits, budget, gross, gross / budget filters in code); offices, terms, predecessors, successors |
| `text_search`, `find_documents`, `read_document`, `similarity_search`, `scoped_similarity` | exact phrase search over all text (deterministic, one passage per document); title match; an article's passages; vector search (global or within documents) |

### 3. The agent loop and how it decides (`agr/pipelines.py: run_agentic`)

1. The entity linker runs in code; its result is the first line of the investigation log.
2. Up to 8 steps: the orchestrator (Gemini) sees the question, the tool list, the evidence policy and the investigation
   so far, and returns JSON: either the next tool with arguments, or `final_answer` with the answer, the facts the
   question asks for (`asked_parts`, each marked stated or not), citations and an explanation. Each tool result is an
   observation with evidence items `[E1]`, `[E2]`, ... kept in a ledger. The most recent observations are shown in full,
   older ones shortened.
3. Checks done in code before an answer is released:
   * **Grounding** (`agr/grounding.py`): every name and number in the answer must be stated in a cited evidence item
     (several items may be combined); a bare count must come from a count field; a name paired with a medal must hold
     that medal in the cited facts. Evidence ids written in the answer text become citations; an answer with no
     citations, or one missing an item it read, is cited from the gathered evidence that states it; citations are then
     limited to the items that state part of the answer. A rejected answer goes back with the reason (twice at most),
     then it is withheld as `not_in_corpus`.
   * **Completeness**: an answer citing a full table (every edition, every repeat winner) must include every row.
   * **Status**: no fact stated -> `not_in_corpus`; some -> `partial` (the others listed as missing); all -> `answered`.
     A fact the evidence only calls unknown, anonymous, unverified or provisional is not stated.
   * **Minimum search before `not_in_corpus`**: code searches every name in the question (and its surname or short
     form) over all text, runs a similarity search and reads the source records, then the agent decides again.
   * **Evidence policy** (in the answer rules): the most authoritative source for the time asked wins (tier 1 > 2 > 3
     > 4); a later publication does not override a signed decision; registrations are not starters; missing evidence is
     not a negative fact; say what was provisional, superseded or never confirmed.
4. The trace (one step per TigerGraph call, LLM call and code step, with tokens and times) and the answer line are
   written as the question runs (`agr/run.py`).

### 4. Running reliably

* **Gemini key pool** (`agr/llm.py`): any number of keys (`GEMINI_API_KEYS`, `GEMINI_API_KEY`, `GEMINI_API_KEY2`, ...);
  each call goes to the fastest key with a free per-minute slot, a rate-limited key rests 60 s while the call moves to
  another key, a key with its daily quota spent is dropped. Rate-limit waits happen before a question starts, so they
  are not part of `latency_ms`. Temperature 0 and a fixed seed (`LLM_SEED`).
* **Sleeping TigerGraph workspace**: Savanna suspends an idle workspace. `run.sh` and `build.sh` wake it first and wait
  with back-off (up to `TG_WAKE_MAX_S`, 900 s); any call that gets "starting", 502-504 or a dropped connection waits and
  retries; MCP calls time out after 90 s and fall back to RESTPP; a question that fails because the workspace slept is
  retried once.
* **Query log**: every TigerGraph call (queries, GSQL, upserts) is appended to `logs/tigergraph_queries.jsonl` with an
  ISO-8601 UTC timestamp, question id, query, parameters, rows, time and transport. `scripts/extract_query_log.py` cuts
  the log of one run and checks that every traced TigerGraph step is in it.
* LLM provider: `LLM_PROVIDER=gemini` (used for all results here; `gemini-3.1-flash-lite`); `anthropic` and `openai`
  are implemented but untested.

## Demo page

`docs/index.html` is one page in two modes:

* **Read-only (public)**: served as static files (e.g. GitHub Pages from the `docs/` folder). It shows how the system
  works, the graph counts, every result table (public runs, held-out, baselines, by style, repeatability) and an
  explorer of every recorded answer (public and held-out): our answer, status, score against the published answer,
  citations, explanation and the full trace with each TigerGraph query (parameters, rows, ms) and each LLM call
  (tokens). No keys, no live calls: it reads `docs/data.json`, built from the committed result files by
  `python scripts/build_demo.py`.
* **Live (local)**: `./demo.sh` (or `python -m agr.web`) wakes the TigerGraph workspace, rebuilds `docs/data.json` and
  serves the same page at http://localhost:8000 with an extra box: type a question, or pick a public / held-out one,
  choose the pipeline (Agentic GraphRAG, or the GraphRAG / RAG baselines) and press Ask. The page shows the answer line,
  citations with snippets, the trace and, for a published question, the published answer and our score. The browser
  only talks to the local server; keys and the TigerGraph secret stay in `.env` on that machine.

## Results

All scores are our own deterministic fact-by-fact scoring against the published public answers
(`scripts/score_public.py`: partial credit per fact, number words read as digits; extra wrong list items are not
penalised, unlike the official scorer). Citation precision / recall compare cited doc_ids with `gold_doc_ids`.

### Public questions (50), full runs

| Run | Code | Accuracy | Full credit | Status right | Citation P / R | Tokens / q | LLM calls / q | TG queries / q | Latency p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|
| Baseline port (25 Nov) | `6a1dc16` | 69.2% | 27 | 41 | n/a | 11,871 | 4.04 | 2.32 | 6.9 / 40.2 s |
| After accuracy pass | `4c31905` | 85.7% | 35 | 46 | 0.77 / 0.83 | 9,237 | 3.44 | 2.36 | 22.3 / 167.5 s * |
| Repeatability run A | `c0c43ec` | 88.6% | 38 | 47 | 0.81 / 0.85 | 8,846 | 3.30 | 2.56 | 28.5 / 288.0 s * |
| Repeatability run B | `c0c43ec` | 86.6% | 38 | 45 | 0.74 / 0.79 | 9,046 | 3.42 | 2.54 | 10.3 / 91.0 s * |
| Fast-period run | `06930bd` | 82.5% | 35 | 45 | 0.73 / 0.81 | 8,821 | 3.26 | 2.64 | **7.0 / 18.7 s** |
| Previous final run (45 s LLM timeout) | `50825c5` | 89.9% | 40 | 48 | 0.76 / 0.87 | 7,639 | 2.72 | 2.42 | 6.1 / 72.5 s * |
| **Final run, submitted** (20 s LLM timeout) | `f7100e4` | **90.1%** | 40 | 48 | 0.76 / 0.87 | 7,646 | 2.72 | 2.42 | **6.3 / 25.5 s** |

\* Gemini was 5-10x slower than usual during these runs (and the workspace auto-suspended during run A), so their
latency is not representative; the fast-period run is. In the previous final run a few Gemini calls hung until the
45 s request timeout (25 of 307 calls; the retry then took about 2 s), which set its p95.

**Which run is submitted.** The agent code is the same in both final runs (`e890a5f`); the only change in `f7100e4` is
that a hung Gemini request is abandoned after 20 s instead of 45 s and retried at once on another key (`agr/llm.py`).
Both final runs answered the public and hidden sets on one commit each. We submit the `f7100e4` run, chosen because it
scored higher on the public set (90.1% against 89.9%; the difference is within run-to-run variation) and was faster
(public p95 25.5 s against 72.5 s, hidden p95 33.1 s against 63.1 s). The previous run's public answers are kept in
`results/answers_public_a5ebfdc.jsonl`. Every answer in every run has at least one TigerGraph step, and
no answer was released without being supported by cited evidence (grounding report in `results/score_*.json`).

**Variance.** Run-to-run variation is real: across the six runs since the accuracy pass the score ranges from 82.5% to
90.1% (mean about 87%). Runs A and B on identical code agree on the status of 46 of 50 questions and on the score of 40
(`results/compare_A_B.json`). The movement is almost entirely in explanation and source-reliability questions; counts,
lists, trends and `not_in_corpus` are stable. Two later changes target it (deterministic text search, evidence ids
resolved to documents, event-first linking): on the unstable questions, two repeated runs now agree on 12 of 14 scores
(4 of 14 before). The submitted figures come from one final run of the public and hidden sets on one frozen code version.

**By answer style** (fast-period run; style read from the shape of the published answer): counts / comparisons 96.7%,
lists 94.9%, trends across editions 100%, `not_in_corpus` 100%, explanations 63.9%, short factual 60%, partial 30%.

### Held-out questions (39, answers computed by code)

`scripts/make_heldout_questions.py` builds questions from the parsed corpus with our own templates, avoiding every
document the public questions are about; no LLM writes a question or an answer.

| Code | Accuracy | Full credit | Statuses right | Citation recall |
|---|---|---|---|---|
| `e890a5f` (frozen, used for the final runs) | **94.9%** | 36 of 39 | 38 of 39 | 0.97 |
| `c0c43ec` (earlier) | 85.9% | 33 of 39 | 35 of 39 | 0.91 |

On the earlier code, by kind: trends, repeat winners, gold leader per sport and 2028 events with no article 100%;
winners with a missing edition (partial) 83%; fewest-nations event 70%; winners at several Games 67%. On the frozen
code, two count / comparison questions get half credit and one partial question scores 0: the model gave an answer
without citing it, so the grounding check withheld it.

**Caveat:** the held-out set is no longer fully untouched. Some fixes made after the `c0c43ec` run (for example citing
supporting evidence that an answer left uncited) were prompted by held-out misses, so 94.9% overstates performance on
truly unseen questions. The hidden set remains the real test.

### Baselines: when does a question need an agent?

Same graph, same LLM, same 50 public questions (code `fca70f6`):

| Pipeline | What it does | Accuracy | Full credit | Citation P / R | Tokens / q | Tokens / correct answer |
|---|---|---|---|---|---|---|
| RAG | vector search, 1 LLM call | 35.4% | 10 | 0.53 / 0.33 | 3,278 | 9,256 |
| GraphRAG | entity linking + fixed graph retrieval + scoped vector search, 1 LLM call | 44.4% | 15 | 0.48 / 0.42 | 4,029 | 9,078 |
| Agentic GraphRAG | this system | 86.6-88.6% | 38 | 0.74-0.81 / 0.79-0.85 | about 8,900 | about 10,000 |

The baselines lose most on counts, rankings, complete lists and trends (17-38% against 87-100%), which the agent answers
with graph tools that compute the result. Per correct answer the agent costs about the same as a one-call pipeline.

## No question-specific rules

The system has no rule written for a particular question: no question ids, no answers in code, and no pattern that
matches the wording of a question in order to pick an answer or a tool. What the code reads from a question:

* **Entity linking** (`agr/tools.py: entity_linker`), the same rules for every question: four-digit years 1900-2099; the
  word "summer" or "winter"; sports, venues, event titles, event names, people, medallists and film titles from the
  graph's own data that occur in the question; a date phrase after "on". The result is a hint to the orchestrator.
* **Argument grounding** (`agr/pipelines.py: ground_args`): a shortened date phrase is restored to the question's
  wording; an evidence id passed where a document id is expected is replaced by that evidence's document.
* **Names in the question** are accepted in an answer (restating the question is not a new claim) and are searched in
  the minimum search before `not_in_corpus`.

Removed for Round 2: the Round 1 medal-role rule that looked for "gold medal" and "originally" in the question; the
medal-role check now compares the answer with the cited evidence only. Answer styles in the scorer are read from the
shape of the published answers, never from question wording. A `git grep` for every 4-word phrase of both question files
finds only generic wording in comments and tool descriptions, and test fixtures; tests use corpus facts that no public or hidden question asks about.

## Limitations

* Accuracy varies by a few points between runs on the same code, mostly on explanation and source-reliability
  questions (see Results); the LLM is a small free-tier model.
* Explanation answers are scored by our own word-level matcher; it under-credits correct answers worded differently.
* Some source records stay unlinked by design when their text does not name exactly one event; the agent reaches them
  through the source_records search.
* Latency depends on Gemini's free tier, which varied 5-10x during development.

## Round 1

The Round 1 system, its results and its design notes are in [docs/round1.md](docs/round1.md) (Round 1 repository: https://github.com/nirmaljosephukken/olympicrag-agentic-graphrag).
