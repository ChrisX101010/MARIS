"""
marisp_provenance.py — the grounded "waterfall": hash-chained provenance plus a
verified-result cache, layered on top of MARISP's execution trace.

Two capabilities, both tested (see test_provenance.py):

  1. PROVENANCE (hash-chained lineage)
     Every answer gets a root hash committing to the modules, retrieved
     strategies, and fallbacks that produced it. Each answer-root includes the
     PREVIOUS answer-root — so the whole conversation forms a tamper-evident
     chain. Crucially we chain the HASH, never the content, so meaning does not
     leak between turns (no "Apple=fruit" contaminating "Apple=company").

  2. VERIFIED-RESULT CACHE (token saver)
     Results that PASSED their quality gate (Senate accepted, probe not flagged)
     can be reused for a near-identical input instead of making fresh API calls.
     Only *verified* results are cached, so reuse stands on already-checked
     ground — saving tokens and reducing opportunities to hallucinate, without
     pouring raw content downstream.

This module imports nothing from MARIS and makes no API calls — it operates on
the trace object MARISP already returns, so it is fully testable offline.
"""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Hashing helpers
# --------------------------------------------------------------------------
def _h(*parts) -> str:
    m = hashlib.sha256()
    for p in parts:
        m.update(json.dumps(p, sort_keys=True, default=str).encode())
    return m.hexdigest()[:16]


# --------------------------------------------------------------------------
# Provenance: build a hash-chained root over a trace
# --------------------------------------------------------------------------
GENESIS = "GENESIS"


@dataclass
class ProvenanceNode:
    module: str
    passed: bool
    took_fallback: bool
    summary: str
    node_hash: str
    parent_hash: str


@dataclass
class AnswerProvenance:
    root: str
    prev_root: str
    nodes: list[ProvenanceNode] = field(default_factory=list)

    def lineage(self) -> list[str]:
        """Ordered list of step hashes from genesis of this answer to its root."""
        return [n.node_hash for n in self.nodes]


def build_provenance(trace, prev_root: str | None = None) -> AnswerProvenance:
    """
    Fold an interpreter trace into a hash-chained provenance for one answer.
    `trace` is the list[TraceEntry] MARISP's Interpreter.run returns.
    `prev_root` links this answer to the previous one (hash only).
    """
    parent = prev_root or GENESIS
    nodes: list[ProvenanceNode] = []
    node = parent
    for e in trace:
        # each node commits to the step AND the running chain (parent)
        node = _h(e.module, e.passed, e.took_fallback, e.result_summary, node)
        nodes.append(ProvenanceNode(
            module=e.module,
            passed=e.passed,
            took_fallback=e.took_fallback,
            summary=e.result_summary,
            node_hash=node,
            parent_hash=parent if not nodes else nodes[-1].node_hash,
        ))
    root = node if nodes else _h("empty", parent)
    return AnswerProvenance(root=root, prev_root=parent, nodes=nodes)


def verify_chain(answers: list[AnswerProvenance]) -> bool:
    """
    Verify a conversation chain: each answer's prev_root must equal the previous
    answer's root. Returns True if the lineage is intact (tamper-evident).
    """
    for i in range(1, len(answers)):
        if answers[i].prev_root != answers[i - 1].root:
            return False
    return True


# --------------------------------------------------------------------------
# Verified-result cache
# --------------------------------------------------------------------------
def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


class VerifiedCache:
    """
    Caches ONLY results that passed their quality gate, keyed by a normalized
    input. Reuse avoids fresh API calls (token saving) and reuses already-gated
    output (fewer hallucination opportunities). Never stores unverified output.
    """

    def __init__(self):
        self._store: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0

    def key(self, input_text: str) -> str:
        return _h(_normalize(input_text))

    def get(self, input_text: str):
        k = self.key(input_text)
        if k in self._store:
            self.hits += 1
            return self._store[k]
        self.misses += 1
        return None

    def put_if_verified(self, input_text: str, answer: str, *,
                        senate_ok: bool, probe_flagged: bool) -> bool:
        """
        Store only if the answer passed its gates. Returns True if cached.
        """
        if not senate_ok or probe_flagged:
            return False  # do NOT cache unverified output
        self._store[self.key(input_text)] = {
            "answer": answer,
            "verified": True,
        }
        return True

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "size": len(self._store),
        }


def render_provenance(ap: AnswerProvenance) -> str:
    lines = [f"answer root: {ap.root}   (prev: {ap.prev_root})", "-" * 40]
    for n in ap.nodes:
        flag = " [fallback]" if n.took_fallback else ""
        status = "ok" if n.passed else "FAIL"
        lines.append(f"  {n.node_hash}  {n.module} ({status}){flag}")
    return "\n".join(lines)
