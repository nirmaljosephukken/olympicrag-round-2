#!/usr/bin/env bash
# One-command build: ./build.sh corpus.jsonl
# parse (no LLM) -> local embeddings -> schema -> load -> install queries -> graph_counts.json
set -euo pipefail
CORPUS_IN=$(realpath "${1:?usage: ./build.sh corpus.jsonl}")
source "$(dirname "$0")/scripts/env.sh"
export CORPUS="$CORPUS_IN"
mkdir -p build logs
rm -f build/catalog.json                       # entity-linking catalogue is rebuilt from the new graph
$PY -m agr.parse "$CORPUS" build
$PY -m agr.embed build
$PY -c "from agr.tg import TG; print('TigerGraph workspace answering after', round(TG().wait_until_awake()), 's')"   # wake it if auto-suspended
$PY -m graph.setup_graph                       # schema (if missing), load, install queries, write graph_counts.json
$PY -c "from agr.tools import Toolbox; c = Toolbox().catalog(); print('catalog:', len(c['events']), 'events')"
echo "build done: graph_counts.json"
