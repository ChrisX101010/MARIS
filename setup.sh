#!/bin/bash
echo "=== MARIS v6 Setup ==="
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install anthropic --quiet
rm -f strategy_memory.json meta_strategies.json progression_metrics.json insights.json
echo "Ready. Run: source venv/bin/activate && rlwrap python main.py"
