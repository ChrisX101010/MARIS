#!/usr/bin/env bash
# run_marisp_demo.sh — verify the MARISP prototype on your machine.
set -euo pipefail
echo "==> Python version:"; python3 --version
echo "==> running MARISP capability tests (mock modules)"
python3 test_marisp.py
echo
echo "  MARISP prototype verified. Next step: wire to real MARIS modules."
