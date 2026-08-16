"""
MARISP planner — composes a plan for each input by consulting learned strategies.

This is the piece that closes the loop you described:
    MARIS learns strategies  ->  the planner retrieves them  ->  they shape the
    plan  ->  executing the plan produces new experience  ->  consolidation.

The planner is deliberately simple and rule-based here (no LLM call), so its
behavior is fully inspectable and testable. In real MARISP you can optionally
replace `compose` with an LLM-emitted plan — but the retrieval-driven default
below already demonstrates "plans informed by what it learned," which is the
novel claim, without any opaque step.
"""

from __future__ import annotations
from dataclasses import dataclass
from marisp_core import Plan, Step


# --------------------------------------------------------------------------
# A learned strategy, mirroring MARIS's Tier-1/Tier-2 memory shape.
# --------------------------------------------------------------------------
@dataclass
class Strategy:
    # A cue the planner matches against the input's detected features.
    trigger: str                 # e.g. "simple_factual", "sad_user+code", "ambiguous"
    # The recommended module sequence (module names) this strategy encodes.
    plan_modules: list[str]
    # How many times this strategy was reinforced — higher = more trusted.
    weight: int = 1
    note: str = ""


class StrategyMemory:
    """Stand-in for MARIS's consolidated memory; real MARISP reads strategy_memory.json."""
    def __init__(self) -> None:
        self._by_trigger: dict[str, Strategy] = {}

    def learn(self, strat: Strategy) -> None:
        existing = self._by_trigger.get(strat.trigger)
        if existing:
            existing.weight += strat.weight  # reinforce
        else:
            self._by_trigger[strat.trigger] = strat

    def retrieve(self, triggers: list[str]) -> Strategy | None:
        """Return the highest-weighted strategy whose trigger matches a feature."""
        candidates = [self._by_trigger[t] for t in triggers if t in self._by_trigger]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.weight)

    def all(self) -> list[Strategy]:
        return list(self._by_trigger.values())


# --------------------------------------------------------------------------
# The planner.
# --------------------------------------------------------------------------
class Planner:
    """
    Turns detected input-features into an explicit Plan, preferring a learned
    strategy when one matches, and otherwise falling back to a safe default
    pipeline. Fallbacks are attached explicitly so cascade failures are handled.
    """

    def __init__(self, memory: StrategyMemory) -> None:
        self.memory = memory

    def compose(self, features: list[str]) -> tuple[Plan, str]:
        """
        Returns (plan, rationale). `features` are cheap detectors MARIS already
        has: e.g. ["simple_factual"], ["sad_user", "code"], ["ambiguous"].
        """
        strat = self.memory.retrieve(features)

        if strat is not None:
            plan = self._plan_from_modules(strat.plan_modules)
            rationale = (
                f"used learned strategy '{strat.trigger}' "
                f"(weight {strat.weight}): {strat.note or 'no note'}"
            )
            return plan, rationale

        # No learned strategy -> conservative default, scaled by complexity.
        if "simple_factual" in features:
            plan = self._plan_from_modules(["reason"])
            return plan, "no strategy; simple input -> minimal plan (reason only)"

        plan = self._default_deliberate_plan()
        return plan, "no strategy; complex input -> full deliberation plan"

    # ---- plan builders ----------------------------------------------------

    def _plan_from_modules(self, modules: list[str]) -> Plan:
        """Build a plan from a module-name list, wiring standard fallbacks."""
        steps = []
        for m in modules:
            steps.append(self._step_with_standard_fallback(m))
        return Plan(steps)

    def _step_with_standard_fallback(self, module: str) -> Step:
        """
        Attach the fallback chain you specified, per module type:
          senate reject      -> improve, then re-judge
          hallucination fail -> fall back to a retrieved memory
          reasoning empty    -> escalate to human clarification
        """
        if module == "senate":
            return Step(
                "senate",
                on_fail=Plan([Step("improve", note="Senate rejected -> rewrite"),
                              Step("senate", note="re-judge improved answer")]),
                note="quality gate",
            )
        if module == "hallucination_probe":
            return Step(
                "hallucination_probe",
                on_fail=Plan([Step("memory_fallback", note="probe failed -> use known-good memory")]),
                note="self-check",
            )
        if module == "reason":
            return Step(
                "reason",
                on_fail=Plan([Step("clarify", note="empty answer -> ask the human")]),
                note="generate",
            )
        return Step(module)

    def _default_deliberate_plan(self) -> Plan:
        """The safe, full path when nothing is learned yet (INFANT-stage behavior)."""
        return Plan([
            self._step_with_standard_fallback("detect_emotion"),
            self._step_with_standard_fallback("inner_monologue"),
            self._step_with_standard_fallback("reason"),
            self._step_with_standard_fallback("senate"),
            self._step_with_standard_fallback("hallucination_probe"),
        ])
