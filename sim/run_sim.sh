#!/usr/bin/env bash
# Opens the interactive viewer. Extra args go to house_env.py, e.g.
#   ./run_sim.sh --layout door_shut --target study_desk --seed 3
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { echo "Run ./setup.sh first"; exit 1; }
if [ "$(uname)" = "Darwin" ]; then
  exec .venv/bin/mjpython house_env.py --view "$@"   # macOS viewer must run under mjpython
else
  exec .venv/bin/python house_env.py --view "$@"
fi
