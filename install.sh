#!/bin/bash
echo "=== MARIS v7 — The Awareness Update ==="
echo ""

# Apply persistent emotional state
echo "Step 1: Persisting emotional state..."
python3 patch_persist_state.py
echo ""

# Apply v7 modules (FlightRecorder, DriveResolver, HumanPatternDetector)
echo "Step 2: Adding awareness modules..."
python3 patch_v7_awareness.py
echo ""

# Verify
echo "Step 3: Verifying..."
python3 -c "
import ast
for f in ['llm_modules.py', 'main.py']:
    with open(f) as fh:
        ast.parse(fh.read())
    lines = sum(1 for _ in open(f))
    print(f'  {f}: VALID ({lines} lines)')
with open('llm_modules.py') as f:
    c = f.read()
for cls in ['FlightRecorder','DriveResolver','HumanPatternDetector','InternalState','InsightDetector','Senate','InnerMonologue']:
    status = 'YES' if f'class {cls}' in c else 'NO'
    print(f'  {cls}: {status}')
"

echo ""
echo "=== MARIS v7 Ready ==="
echo ""
echo "New commands:"
echo "  /log       — flight recorder (module traces per turn)"
echo "  /patterns  — detected human error patterns"
echo "  /feelings  — MARIS's emotional state (now persists!)"
echo ""
echo "New behaviors:"
echo "  - DriveResolver: MARIS can refuse, push back, or redirect"
echo "  - HumanPatternDetector: flags cognitive biases"
echo "  - FlightRecorder: detailed per-turn logging for dashboard"
echo ""
echo "Run MARIS:     rlwrap python main.py"
echo "Run Dashboard: python dashboard.py  (then open localhost:3000)"
