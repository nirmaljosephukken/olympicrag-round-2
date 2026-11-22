#!/usr/bin/env bash
# One-command run: ./run.sh questions.jsonl answers.jsonl
# Runs the agentic pipeline on every question; resumable (answered qids are skipped). Add --fresh to start over.
set -euo pipefail
Q=$(realpath "${1:?usage: ./run.sh questions.jsonl answers.jsonl}")
A=$(realpath -m "${2:?usage: ./run.sh questions.jsonl answers.jsonl}")
shift 2
source "$(dirname "$0")/scripts/env.sh"
$PY -m agr.run "$Q" "$A" "$@"
