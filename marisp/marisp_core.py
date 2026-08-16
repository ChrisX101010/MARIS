"""
MARISP core — the planning layer that turns MARIS from a fixed pipeline into a
system that *composes a plan* for each input and executes it with explicit
fallbacks.

Design goals (all grounded, none magical):
  1. A plan is an explicit, readable instruction list (the "LISP" intuition).
  2. Plans are composed by consulting learned strategies (learning feeds planning).
  3. Fallbacks are first-class, not emergent — attacking cascade failure directly.
  4. Every run produces a trace you can read (the "X-ray").

This module is dependency-free and wraps MARIS's modules through a thin adapter,
so it can be tested standalone here and dropped onto the real modules later.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any


# --------------------------------------------------------------------------
# 1. The plan format — an explicit instruction list (the symbolic layer)
# --------------------------------------------------------------------------
#
# A Step names a module to invoke plus optional fallback handling. A Plan is an
# ordered list of Steps. This is deliberately small: readable by a human,
# emittable by a planner, executable by the interpreter.

@dataclass
class Step:
    """One instruction: run `module`, optionally react to a weak/failed result."""
    module: str                          # which MARIS module to invoke
    # A predicate on the module's result deciding if it "passed". Defaults to
    # "any non-None, non-empty result passes."
    passed: Callable[[Any], bool] | None = None
    # If the step fails, run this sub-plan instead of aborting (the fallback).
    on_fail: "Plan | None" = None
    # Optional human label for the trace.
    note: str = ""


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)

    def describe(self, indent: int = 0) -> str:
        """Human-readable rendering of the plan — the readability guarantee."""
        pad = "  " * indent
        lines = []
        for s in self.steps:
            tag = f"  # {s.note}" if s.note else ""
            lines.append(f"{pad}- {s.module}{tag}")
            if s.on_fail:
                lines.append(f"{pad}  on_fail:")
                lines.append(s.on_fail.describe(indent + 2))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 2. The module registry — thin adapter over MARIS's real modules
# --------------------------------------------------------------------------
#
# MARISP does not care how a module is implemented. It only needs a callable
# that takes the shared context dict and returns a result. To wire in real
# MARIS, register each module: registry.register("reason", maris.reasoning.run)

class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: dict[str, Callable[[dict], Any]] = {}

    def register(self, name: str, fn: Callable[[dict], Any]) -> None:
        self._modules[name] = fn

    def has(self, name: str) -> bool:
        return name in self._modules

    def call(self, name: str, ctx: dict) -> Any:
        if name not in self._modules:
            raise KeyError(f"module '{name}' is not registered")
        return self._modules[name](ctx)


# --------------------------------------------------------------------------
# 3. The interpreter — executes a plan, applies fallbacks, records the trace
# --------------------------------------------------------------------------

@dataclass
class TraceEntry:
    module: str
    passed: bool
    took_fallback: bool
    note: str
    result_summary: str


class Interpreter:
    def __init__(self, registry: ModuleRegistry, max_depth: int = 8) -> None:
        self.registry = registry
        self.max_depth = max_depth

    @staticmethod
    def _default_passed(result: Any) -> bool:
        if result is None:
            return False
        if isinstance(result, str):
            return result.strip() != ""
        if isinstance(result, dict):
            # convention: a module may return {"ok": bool, ...}
            return bool(result.get("ok", True))
        return True

    def run(self, plan: Plan, ctx: dict) -> tuple[dict, list[TraceEntry]]:
        trace: list[TraceEntry] = []
        self._run_plan(plan, ctx, trace, depth=0)
        return ctx, trace

    def _run_plan(self, plan: Plan, ctx: dict, trace: list[TraceEntry], depth: int) -> None:
        if depth > self.max_depth:
            trace.append(TraceEntry("<max-depth>", False, False, "aborted: fallback too deep", ""))
            return

        for step in plan.steps:
            result = self.registry.call(step.module, ctx)
            ctx[step.module] = result  # results accumulate in shared context

            check = step.passed or self._default_passed
            ok = bool(check(result))

            took_fallback = False
            if not ok and step.on_fail is not None:
                took_fallback = True
                trace.append(TraceEntry(
                    step.module, ok, took_fallback, step.note,
                    self._summarize(result),
                ))
                # execute the fallback sub-plan, then continue
                self._run_plan(step.on_fail, ctx, trace, depth + 1)
                continue

            trace.append(TraceEntry(
                step.module, ok, took_fallback, step.note,
                self._summarize(result),
            ))

    @staticmethod
    def _summarize(result: Any) -> str:
        s = str(result)
        return s if len(s) <= 60 else s[:57] + "..."


def render_trace(trace: list[TraceEntry]) -> str:
    """The X-ray: exactly what ran, whether it passed, and where fallbacks fired."""
    lines = ["MARISP execution trace", "=" * 40]
    for i, e in enumerate(trace, 1):
        status = "PASS" if e.passed else "FAIL"
        fb = "  ->FALLBACK" if e.took_fallback else ""
        note = f"  ({e.note})" if e.note else ""
        lines.append(f"{i:2d}. [{status}]{fb} {e.module}{note}")
        if e.result_summary:
            lines.append(f"      = {e.result_summary}")
    return "\n".join(lines)
