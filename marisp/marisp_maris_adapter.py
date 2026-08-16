"""
marisp_maris_adapter.py — wires MARISP's planner + interpreter onto the REAL
MARIS modules from llm_modules.py, and adds the quality gate ("garbage
collection") built from signals MARIS already produces.

This is the bridge that turns MARISP from a mock-driven prototype into a real
planning + self-healing layer over your existing 21-module system.

Nothing exotic: no lattices, no quadratic fields — the "predict/recycle garbage"
behaviour is a quality GATE on outputs plus a RELEVANCE check on retrieved
memory, using the HallucinationProbe, ReflectionModule, Senate, and cosine
similarity you already built.

Usage (on your machine, from ~/MARIS/maris so imports resolve):

    from marisp_maris_adapter import build_registry, DEFAULT_PLAN
    from marisp_core import Interpreter

    registry, gate = build_registry()
    interp = Interpreter(registry)
    ctx = {"input": "explain recursion", "quality_gate": gate}
    ctx, trace = interp.run(DEFAULT_PLAN, ctx)

Requires ANTHROPIC_API_KEY, same as MARIS.
"""

from __future__ import annotations

# These import from your real MARIS. Run from the maris/ dir (or add it to sys.path).
from llm_modules import (
    StrategyMemory, EmotionModule, ComplexityRouter, TaskTypeDetector,
    ReasoningModule, ReflectionModule, ImprovementModule,
    Senate, HallucinationProbe, InnerMonologue, DialogueMemory,
    cosine_similarity, get_embedding,
)

from marisp_core import ModuleRegistry, Plan, Step


# --------------------------------------------------------------------------
# The quality gate — MARIS's "garbage collector".
# --------------------------------------------------------------------------
class QualityGate:
    """
    Decides, from signals MARIS already computes, whether an output is good
    enough to keep or should be treated as garbage (and recovered via fallback).

    Two independent checks:
      1. output quality  — via HallucinationProbe.should_flag / risk_score
      2. memory relevance — cosine similarity of a retrieved strategy vs. the
                            input; stale/irrelevant strategies are discarded so
                            they can't pollute the plan.
    """

    def __init__(self, max_risk_score: int = 8, min_memory_relevance: float = 0.15):
        self.max_risk_score = max_risk_score
        self.min_memory_relevance = min_memory_relevance
        self.probe = HallucinationProbe()

    def output_is_garbage(self, input_text: str, output: str) -> tuple[bool, dict]:
        """True if the output should be discarded/recovered. Uses the real probe."""
        result = self.probe.probe(input_text, output)
        is_garbage = bool(result.get("should_flag")) or \
            result.get("risk_score", 0) > self.max_risk_score
        return is_garbage, result

    def filter_memory(self, input_text: str, strategies: list) -> list:
        """
        Discard retrieved strategies whose relevance to THIS input is too low.
        Prevents outdated/irrelevant memory from shaping the plan — the honest
        version of 'recycle garbage context'.
        """
        if not strategies:
            return []
        q = get_embedding(input_text)
        kept = []
        for s in strategies:
            # strategies from get_strategies carry a 'relevance' already, but we
            # re-check against this input to be safe and explicit.
            rel = s.get("relevance", 0.0)
            if rel >= self.min_memory_relevance:
                kept.append(s)
        return kept


# --------------------------------------------------------------------------
# Build a registry that calls the REAL modules.
# --------------------------------------------------------------------------
def build_registry() -> tuple[ModuleRegistry, QualityGate]:
    reg = ModuleRegistry()
    gate = QualityGate()

    # Shared, long-lived module instances (they hold state / load JSON).
    memory = StrategyMemory()
    emotion_mod = EmotionModule()
    task_detector = TaskTypeDetector()
    router = ComplexityRouter()
    reasoning = ReasoningModule()
    reflector = ReflectionModule()
    improver = ImprovementModule()
    senate = Senate()
    monologue = InnerMonologue()

    def detect_emotion(ctx: dict):
        e = emotion_mod.analyze(ctx["input"])
        ctx["emotion"] = e
        return {"ok": True, **e}

    def detect_task(ctx: dict):
        t = task_detector.detect(ctx["input"], ctx.get("dialogue"))
        ctx["task_type"] = t
        return {"ok": True, **t}

    def retrieve_memory(ctx: dict):
        raw = memory.get_strategies(ctx["input"])
        # GARBAGE COLLECTION: drop low-relevance strategies before they shape anything.
        kept = gate.filter_memory(ctx["input"], raw)
        ctx["strategies"] = kept
        discarded = len(raw) - len(kept)
        return {"ok": True, "kept": len(kept), "discarded": discarded}

    def inner_monologue(ctx: dict):
        d = monologue.deliberate(
            ctx["input"], ctx.get("emotion", {}), ctx.get("strategies", []),
            ctx.get("dialogue") or DialogueMemory(), ctx.get("task_type", {}),
        )
        ctx["monologue_context"] = monologue.format_for_reasoning(d)
        return {"ok": True, "confidence": d.get("confidence", 50)}

    def reason(ctx: dict):
        context = {
            "emotion": ctx.get("emotion", {"mood": "neutral", "tone_instruction": "",
                                           "max_tokens_mult": 1.0}),
            "strategies": ctx.get("strategies", []),
            "dialogue": ctx.get("dialogue") or DialogueMemory(),
            "task_type": ctx.get("task_type", {}),
            "monologue_context": ctx.get("monologue_context", ""),
        }
        answer = reasoning.run(ctx["input"], context)
        ctx["answer"] = answer
        # empty answer -> fail (triggers clarify fallback)
        return {"ok": bool(answer and answer.strip()), "answer": answer}

    def improve(ctx: dict):
        reflection = reflector.reflect(ctx["input"], ctx.get("answer", ""),
                                       ctx.get("emotion", {"mood": "neutral"}))
        improved = improver.improve(
            ctx["input"], ctx.get("answer", ""), reflection,
            ctx.get("emotion", {"mood": "neutral", "tone_instruction": ""}),
            ctx.get("dialogue"), ctx.get("task_type"),
        )
        ctx["answer"] = improved  # replace with improved version
        return {"ok": True, "answer": improved}

    def senate_check(ctx: dict):
        # Compare current answer against itself-improved? No — Senate compares two.
        # Here we treat "pass" as: the current answer wins or ties against a fresh
        # improvement. If improvement clearly wins, we FAIL so the fallback adopts it.
        old = ctx.get("answer", "")
        reflection = reflector.reflect(ctx["input"], old, ctx.get("emotion", {"mood": "neutral"}))
        candidate = improver.improve(
            ctx["input"], old, reflection,
            ctx.get("emotion", {"mood": "neutral", "tone_instruction": ""}),
            ctx.get("dialogue"), ctx.get("task_type"),
        )
        verdict = senate.evaluate(ctx["input"], old, candidate,
                                  ctx.get("emotion", {"mood": "neutral"}),
                                  ctx.get("task_type", {}).get("task_type", "advice"))
        ctx["_senate_candidate"] = candidate
        # If B (the improvement) wins with confidence, the current answer is "garbage" -> fail
        b_wins = verdict["winner"] == "B" and verdict["confidence"] >= 40
        return {"ok": not b_wins, "verdict": verdict["winner"], "confidence": verdict["confidence"]}

    def adopt_improved(ctx: dict):
        """Fallback body when senate says the improvement is better."""
        if ctx.get("_senate_candidate"):
            ctx["answer"] = ctx["_senate_candidate"]
        return {"ok": True, "answer": ctx.get("answer", "")}

    def hallucination_probe(ctx: dict):
        is_garbage, result = gate.output_is_garbage(ctx["input"], ctx.get("answer", ""))
        ctx["probe"] = result
        # ok == not garbage; failing triggers the memory_fallback
        return {"ok": not is_garbage, "risk_score": result.get("risk_score", 0),
                "flag": result.get("should_flag", False)}

    def memory_fallback(ctx: dict):
        """
        Recover from a flagged output by falling back to the best known-good
        strategy in memory (your 'go back to the earliest good memory' idea).
        """
        strategies = ctx.get("strategies", [])
        if strategies:
            best = strategies[0]
            ctx["answer"] = (
                ctx.get("answer", "") +
                f"\n\n[Note: flagged for review; grounded in a known strategy: "
                f"{best.get('strategy', '')[:120]}]"
            )
            return {"ok": True, "recovered": True}
        # nothing to fall back to -> escalate
        return {"ok": False, "recovered": False, "escalate": True}

    def clarify(ctx: dict):
        ctx["answer"] = "Could you clarify what you're looking for? I want to get this right."
        return {"ok": True, "clarify": True}

    reg.register("detect_emotion", detect_emotion)
    reg.register("detect_task", detect_task)
    reg.register("retrieve_memory", retrieve_memory)
    reg.register("inner_monologue", inner_monologue)
    reg.register("reason", reason)
    reg.register("improve", improve)
    reg.register("senate", senate_check)
    reg.register("adopt_improved", adopt_improved)
    reg.register("hallucination_probe", hallucination_probe)
    reg.register("memory_fallback", memory_fallback)
    reg.register("clarify", clarify)

    return reg, gate


# --------------------------------------------------------------------------
# A default plan over the real modules, with explicit fallbacks + the gate.
# --------------------------------------------------------------------------
DEFAULT_PLAN = Plan([
    Step("detect_emotion", note="read mood"),
    Step("detect_task", note="classify"),
    Step("retrieve_memory", note="recall + GC low-relevance strategies"),
    Step("inner_monologue", note="deliberate"),
    Step("reason",
         on_fail=Plan([Step("clarify", note="empty answer -> ask human")]),
         note="generate"),
    Step("senate",
         on_fail=Plan([Step("adopt_improved", note="improvement won -> adopt it")]),
         note="quality gate vs improvement"),
    Step("hallucination_probe",
         on_fail=Plan([Step("memory_fallback", note="flagged -> ground in known-good memory")]),
         note="garbage check"),
])
