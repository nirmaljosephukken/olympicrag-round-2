# Judging-call runbook

What to run before and during the live judging call, and what to do if something is slow.

## The day before

* Check the Savanna workspace has credits left for the call and is set to auto-resume.
* Leave the Gemini keys alone on the day of the call. The free-tier daily quota resets at midnight Pacific time
  (08:00 UTC, 13:30 IST); a full public + hidden run uses most of a day's quota.
* `git checkout <submission tag>` so the code on screen is the code that produced the submitted answers.
* `python -m pytest -q tests` (offline, about 10 s).

## 30 minutes before

```bash
./demo.sh                     # wakes the workspace (up to ~1 min after auto-suspend), loads the catalogue,
                              # serves the demo page with live questions on http://localhost:8000
```

Then ask one warm-up question on the page (the first question after a restart is slower: the MCP server starts and
Gemini connections open). The status line at the top of the page shows `live · graph OlympicRAG2 · model · keys`.

Keep a second terminal open in the repository for the commands below.

## During the call

1. **Architecture** (Explore the page, no live calls): *How it works* shows the seven layers, the graph schema with
   counts and the agent loop. *Results* has the public, held-out and baseline numbers.
2. **New questions from the judges**: type them into *Ask a question* (agentic pipeline). The page shows the answer,
   citations with snippets and every step: LLM decisions with tokens, TigerGraph queries with parameters and rows,
   checks that passed or rejected. Optionally run the same question through the GraphRAG and RAG baselines to show
   the difference. Same thing in a terminal: `python -m agr.ask "question"` (`--all` for all three pipelines).
3. **Reproducing submitted hidden answers**: for the question ids the judges pick,
   ```bash
   python scripts/reproduce.py answers_hidden.jsonl r2-hid-007 r2-hid-031
   ```
   It answers them again with the same system as `./run.sh` and prints the submitted and new status, answer, citations
   and TigerGraph queries side by side. The LLM runs at temperature 0 with a fixed seed; on the held-out set two
   questions reproduced exactly (same answer, citations and queries), but a free-tier model is not perfectly
   deterministic, so wording can differ.
4. **Traces and logs**: `python scripts/check_submission.py answers_public.jsonl answers_hidden.jsonl --metrics metrics.json`
   checks every answer's fields, that step tokens add up to the totals, that step times add up to the latency, that
   every TigerGraph step is in `logs/tigergraph_queries.jsonl`, and that metrics.json matches the answer files.

## If something is slow or fails

| Problem | What to do |
|---|---|
| First call hangs for a minute | The workspace was suspended; `./demo.sh` and `./run.sh` wait for it (up to `TG_WAKE_MAX_S`). |
| Gemini slow (a step over 20 s) | Free-tier latency varies 5-10x by time of day. Keep talking through the recorded traces in *Explore answers* while it runs; a hung request is abandoned after `LLM_TIMEOUT_S` (20 s) and retried at once on another key. |
| "daily quota exhausted on every key" | Add another key to `GEMINI_API_KEYS` in `.env` and restart `./demo.sh`. |
| TigerGraph MCP call times out | It falls back to RESTPP automatically after `TG_MCP_TIMEOUT_S`. |
| Nothing live works | The page still shows every recorded answer and trace; the published copy needs no server. |

`LLM_PROVIDER=anthropic` and `openai` are implemented but were not used for any result; switching provider changes the
model, so answers will not match the submitted ones.

## Questions to have ready

None of these is in the public, held-out or hidden sets. Each was run on the frozen code on 26 Nov (Gemini was slow
that afternoon: 22-82 s per question).

| Question | Shows | Answer then |
|---|---|---|
| Who won the gold medal in the men's pole vault at the 2012 Summer Olympics? | event lookup, 1 citation | Renaud Lavillenie |
| Which nation won the most gold medals in table tennis at the 2024 Summer Olympics, and how many? | medal table computed in the graph | CHN, 5 |
| How did the number of nations in the men's pole vault change across the Summer Olympics in the corpus? Give the highest and lowest. | trend across editions, 9 citations | highest 25 (2004, 2008), lowest 13 (1988) |
| Was the LA 2028 men's triathlon final moved, and when did it finally take place according to the most authoritative source? | source records: bulletin vs race director's log | moved from 29 to 30 July 2028 |
| Who won the 2030 Winter Olympics curling gold? | minimum search, then `not_in_corpus` | not_in_corpus |
