"""
MARISP capability tests — proves the planning layer works, using mock modules
that stand in for MARIS's real ones. Every claim is checked, not asserted in prose.

Run:  python3 test_marisp.py
"""

from marisp_core import ModuleRegistry, Interpreter, render_trace
from marisp_planner import Planner, StrategyMemory, Strategy


# --------------------------------------------------------------------------
# Mock MARIS modules. Each takes the shared context and returns a result.
# These stand in for EmotionModule, ReasoningModule, Senate, etc.
# --------------------------------------------------------------------------
def mk_registry(senate_rejects_first=False):
    reg = ModuleRegistry()
    state = {"senate_calls": 0}

    reg.register("detect_emotion", lambda ctx: {"ok": True, "mood": "neutral"})
    reg.register("inner_monologue", lambda ctx: {"ok": True, "thought": "deliberating"})
    reg.register("reason", lambda ctx: {"ok": True, "answer": "42"})
    reg.register("clarify", lambda ctx: {"ok": True, "question": "could you clarify?"})
    reg.register("improve", lambda ctx: {"ok": True, "answer": "42 (improved)"})
    reg.register("memory_fallback", lambda ctx: {"ok": True, "answer": "known-good from memory"})

    def senate(ctx):
        state["senate_calls"] += 1
        # optionally reject the first time to exercise the fallback
        if senate_rejects_first and state["senate_calls"] == 1:
            return {"ok": False, "verdict": "reject"}
        return {"ok": True, "verdict": "accept"}
    reg.register("senate", senate)

    reg.register("hallucination_probe", lambda ctx: {"ok": True, "grounded": True})
    return reg


def test_simple_input_gets_minimal_plan():
    mem = StrategyMemory()
    planner = Planner(mem)
    plan, why = planner.compose(["simple_factual"])
    modules = [s.module for s in plan.steps]
    assert modules == ["reason"], modules
    assert "minimal" in why
    print("PASS  simple input -> minimal plan (reason only)")


def test_complex_input_gets_full_deliberation():
    mem = StrategyMemory()
    planner = Planner(mem)
    plan, why = planner.compose(["ambiguous", "code"])
    modules = [s.module for s in plan.steps]
    assert modules == ["detect_emotion", "inner_monologue", "reason", "senate", "hallucination_probe"]
    assert "full deliberation" in why
    print("PASS  complex input -> full deliberation plan")


def test_learned_strategy_overrides_default():
    mem = StrategyMemory()
    # MARIS "learned" that for a sad user asking about code, this sequence worked.
    mem.learn(Strategy(
        trigger="sad_user",
        plan_modules=["detect_emotion", "reason", "senate"],
        weight=5,
        note="validate feelings before answering code questions",
    ))
    planner = Planner(mem)
    plan, why = planner.compose(["sad_user"])
    modules = [s.module for s in plan.steps]
    assert modules == ["detect_emotion", "reason", "senate"], modules
    assert "learned strategy" in why
    print(f"PASS  learned strategy shaped the plan  ({why})")


def test_strategy_reinforcement_increases_trust():
    mem = StrategyMemory()
    s = Strategy(trigger="greeting", plan_modules=["reason"], weight=1)
    mem.learn(s)
    mem.learn(Strategy(trigger="greeting", plan_modules=["reason"], weight=1))  # reinforce
    retrieved = mem.retrieve(["greeting"])
    assert retrieved.weight == 2, retrieved.weight
    print("PASS  repeated experience reinforces a strategy (weight 1 -> 2)")


def test_senate_rejection_triggers_explicit_fallback():
    reg = mk_registry(senate_rejects_first=True)
    interp = Interpreter(reg)
    mem = StrategyMemory()
    planner = Planner(mem)
    plan, _ = planner.compose(["ambiguous"])  # full plan includes senate w/ fallback

    ctx, trace = interp.run(plan, {"input": "explain recursion"})

    # the senate step must show a fallback firing, then a re-judge
    modules_run = [t.module for t in trace]
    assert "improve" in modules_run, modules_run          # fallback ran
    assert modules_run.count("senate") == 2, modules_run  # re-judged after improve
    # and a fallback was recorded
    assert any(t.took_fallback for t in trace)
    print("PASS  Senate rejection -> improve -> re-judge (explicit fallback fired)")
    return trace


def test_full_run_produces_readable_trace():
    reg = mk_registry(senate_rejects_first=False)
    interp = Interpreter(reg)
    mem = StrategyMemory()
    planner = Planner(mem)
    plan, why = planner.compose(["ambiguous"])
    ctx, trace = interp.run(plan, {"input": "what is a monad?"})
    out = render_trace(trace)
    assert "MARISP execution trace" in out
    assert "reason" in out
    print("PASS  full run produces a readable trace")
    return plan, why, trace


if __name__ == "__main__":
    print("=" * 60)
    print("MARISP capability tests")
    print("=" * 60)
    test_simple_input_gets_minimal_plan()
    test_complex_input_gets_full_deliberation()
    test_learned_strategy_overrides_default()
    test_strategy_reinforcement_increases_trust()
    test_senate_rejection_triggers_explicit_fallback()
    plan, why, trace = test_full_run_produces_readable_trace()

    print("\n" + "=" * 60)
    print("DEMO: a complex input, planned and executed")
    print("=" * 60)
    print("\nplan rationale:", why)
    print("\nthe plan MARISP composed:")
    print(plan.describe())
    print()

    # show the fallback demo trace too
    print("=" * 60)
    print("DEMO: Senate rejects, explicit fallback recovers")
    print("=" * 60)
    fb_trace = test_senate_rejection_triggers_explicit_fallback.__wrapped__ if hasattr(
        test_senate_rejection_triggers_explicit_fallback, "__wrapped__") else None
    # re-run to display
    reg = mk_registry(senate_rejects_first=True)
    interp = Interpreter(reg)
    mem = StrategyMemory()
    planner = Planner(mem)
    p, _ = planner.compose(["ambiguous"])
    _, tr = interp.run(p, {"input": "explain recursion"})
    print("\n" + render_trace(tr))

    print("\n" + "=" * 60)
    print("ALL MARISP TESTS PASSED")
    print("=" * 60)
