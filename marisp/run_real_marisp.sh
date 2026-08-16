#!/usr/bin/env bash
# run_real_marisp.sh — run MARISP over the REAL MARIS modules.
# Must be run from inside ~/MARIS/maris so `import llm_modules` resolves,
# with ANTHROPIC_API_KEY set (same as MARIS).
set -euo pipefail
if [[ ! -f llm_modules.py ]]; then
  echo "error: run this from ~/MARIS/maris (where llm_modules.py lives)." >&2
  echo "  cp ../marisp/*.py .   # bring MARISP files in, then run again" >&2
  exit 1
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "error: set ANTHROPIC_API_KEY first (same key MARIS uses)." >&2
  exit 1
fi
python3 marisp_live_demo.py
