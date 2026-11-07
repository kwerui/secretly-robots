#!/usr/bin/env bash
# One-off setup: makes sim/.venv, installs deps, checks every layout and renders snapshots.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(not ((3,10) <= sys.version_info[:2] <= (3,13)))'; then
    PY="$cand"; break
  fi
done
[ -n "$PY" ] || { echo "Need Python 3.10-3.13 (e.g. brew install python@3.12)"; exit 1; }
echo "Using $($PY --version)"

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

.venv/bin/python house_env.py --check
.venv/bin/python house_env.py --snapshot
echo
echo "Done. Open the viewer with: ./run_sim.sh   (snapshots are in sim/snapshots/)"
