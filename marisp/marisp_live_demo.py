"""
marisp_live_demo.py — run MARISP's planned + self-healing execution over the
REAL MARIS modules, and print the readable trace (the X-ray).

Run from ~/MARIS/maris with ANTHROPIC_API_KEY set:
    python3 marisp_live_demo.py
"""
from marisp_maris_adapter import build_registry, DEFAULT_PLAN
from marisp_core import Interpreter, render_trace
from llm_modules import DialogueMemory

def main():
    registry, gate = build_registry()
    interp = Interpreter(registry)

    tests = [
        "hi there",
        "i had a really bad day and can't focus, help me be productive",
        "explain the tradeoffs between microservices and monoliths for a 3-person startup",
    ]
    for t in tests:
        print("\n" + "=" * 66)
        print("INPUT:", t)
        print("=" * 66)
        ctx = {"input": t, "dialogue": DialogueMemory()}
        ctx, trace = interp.run(DEFAULT_PLAN, ctx)
        print(render_trace(trace))
        print("\nFINAL ANSWER:\n", ctx.get("answer", "")[:500])

if __name__ == "__main__":
    main()
