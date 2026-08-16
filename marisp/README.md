# MARISP — planning + self-healing layer for MARIS (prototype)

**Status: standalone prototype.** This runs today against *mock* modules and
proves the planning logic. It is **not yet wired to your real MARIS modules** —
that's the next step, and it needs `main.py` + `llm_modules.py` to be connected.

## What this is

MARISP adds a layer on top of MARIS that turns the fixed pipeline into a
**composed plan per input**, with **explicit fallbacks** (self-healing) and a
**readable execution trace** (the X-ray).

- `marisp_core.py` — the plan format (`Step`, `Plan`), a `ModuleRegistry`
  adapter, and the `Interpreter` that runs a plan and records a trace.
- `marisp_planner.py` — the `Planner` that composes a plan by consulting learned
  `Strategy` objects from `StrategyMemory`, with a safe default plan and the
  standard fallback chain.
- `test_marisp.py` — six capability tests that all pass, driving mock modules.

## Run it (proves the logic)

```bash
python3 test_marisp.py
```

You'll see: simple input → minimal plan; complex input → full deliberation;
a learned strategy overriding the default; reinforcement raising a strategy's
weight; a Senate rejection triggering an explicit improve→re-judge fallback;
and a readable trace.

## How it will connect to real MARIS (next step)

The mock registry gets replaced with adapters onto your real modules:

```python
from llm_modules import EmotionModule, ReasoningModule, Senate  # etc.
registry.register("reason", lambda ctx: reasoning.run(ctx["input"], ctx))
registry.register("senate", lambda ctx: senate.evaluate(...))
# ...one line per real module
```

Once wired, the planner reads your real `strategy_memory.json` to compose plans,
and the fallbacks call your real modules. That's the version worth benchmarking
against plain MARIS.

## The measurable claim (the paper)

Does planned + self-healing execution reduce failed/hallucinated outputs versus
the fixed linear pipeline, and at what token cost? Decide the metric and inputs
*before* running, then A/B MARISP vs. MARIS. That's a real result, grounded in
2026 self-healing-orchestration research (arXiv 2606.01416, 2605.06737,
2602.19843).

## License

MIT, matching MARIS.
