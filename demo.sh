#!/usr/bin/env bash
# Local demo: the demo page with live questions answered by the system (same agent, graph and .env as run.sh).
#   ./demo.sh   ->  open http://localhost:8000
set -euo pipefail
source "$(dirname "$0")/scripts/env.sh"
$PY scripts/build_demo.py
exec $PY -m agr.web
