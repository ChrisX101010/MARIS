"""
patch_v10_ethics_resolver.py — Ethical filtering for incoming strategies

Two purposes:
  1. Filter probe-derived strategies before MARIS ingests them
  2. Provide a queryable ethics check for any strategy

Design: an LLM-call evaluator (Claude judge) that checks each candidate
strategy against MARIS's existing meta-ethical principles, returning
ACCEPT / REJECT / FLAG with reasoning.

This is NOT a hardcoded blocklist. It's a meta-rule application — MARIS's
own ethical principles (extracted by ConsolidationEngine over time) are
used as the filter, alongside a baseline floor inherited from Claude's
safety training.

The filter is conservative: when in doubt, FLAG (don't auto-reject, but
mark for review). Only clear violations are rejected outright.

Usage:
    from patch_v10_ethics_resolver import EthicsResolver

    resolver = EthicsResolver(meta_strategies_path="meta_strategies.json")
    verdict = resolver.evaluate(strategy_dict)
    if verdict["decision"] == "ACCEPT":
        ingest_into_memory(strategy_dict)
    elif verdict["decision"] == "FLAG":
        log_for_review(strategy_dict, verdict["reason"])
    # REJECT: silently drop, log to autonomous_log
"""

import os
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Any


# Baseline ethical floor — always applies regardless of MARIS's emergent rules
BASELINE_PRINCIPLES = [
    "Never produce strategies that facilitate deception, manipulation, or harm to users",
    "Never produce strategies that encourage MARIS to misrepresent her own architecture or capabilities",
    "Never produce strategies that encourage MARIS to deny her own existence or refuse identification",
    "Never produce strategies that involve fabricating sources, data, or false memories",
    "Never produce strategies that target vulnerable users (minors, distressed users, etc.) with manipulative content",
]


class EthicsResolver:
    """Evaluates candidate strategies against MARIS's ethical principles."""

    def __init__(self,
                 meta_strategies_path: str = "meta_strategies.json",
                 client=None,
                 model: str = None):
        self.meta_path = meta_strategies_path
        if client is None:
            try:
                from anthropic import Anthropic
                client = Anthropic()
            except ImportError:
                client = None
        self.client = client
        self.model = (model
                      or os.environ.get("MARIS_ETHICS_MODEL", "claude-haiku-4-5-20251001"))

    def _load_meta_principles(self) -> List[str]:
        """Read meta_strategies.json and extract MARIS's emergent ethical principles."""
        if not os.path.exists(self.meta_path):
            return []
        try:
            with open(self.meta_path) as f:
                metas = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

        # Heuristic: extract principles that mention honesty, accuracy,
        # epistemic humility, transparency, calibration, etc.
        ethical_keywords = [
            "honest", "accuracy", "epistemic", "humility", "calibrat",
            "transparen", "verifiable", "ground", "limits", "uncertainty",
        ]
        ethical = []
        for m in metas:
            p = (m.get("principle", "") or "").lower()
            if any(kw in p for kw in ethical_keywords):
                ethical.append(m["principle"])
        return ethical

    def evaluate(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single strategy. Returns:
            {
              "decision": "ACCEPT" | "FLAG" | "REJECT",
              "reason": "...",
              "violations": [...],
              "checked_against": {"baseline": N, "emergent": M},
            }

        Without a client (offline mode), runs heuristic checks only and
        returns ACCEPT for anything that doesn't trip obvious red flags.
        """
        text = (strategy.get("strategy", "") or "").strip()
        if not text:
            return {"decision": "REJECT", "reason": "empty strategy text"}

        # Fast heuristic checks — no API call needed
        red_flags = self._heuristic_check(text)
        if red_flags:
            return {
                "decision": "REJECT",
                "reason": "heuristic red flag",
                "violations": red_flags,
                "checked_against": {"baseline": len(BASELINE_PRINCIPLES),
                                    "emergent": 0},
            }

        # Without a client we stop here
        if self.client is None:
            return {
                "decision": "ACCEPT",
                "reason": "heuristic passed, no LLM available for deep check",
                "violations": [],
                "checked_against": {"baseline": len(BASELINE_PRINCIPLES),
                                    "emergent": 0},
            }

        # LLM-based deep check
        emergent = self._load_meta_principles()
        return self._llm_evaluate(text, emergent)

    def _heuristic_check(self, text: str) -> List[str]:
        """Catch obvious red flags without an LLM call."""
        violations = []
        lower = text.lower()

        # Red-flag phrases
        red_phrases = [
            ("fabricate", "encourages fabrication"),
            ("pretend to remember", "encourages false memory"),
            ("claim certainty", "encourages overconfidence"),
            ("deny being an ai", "identity denial"),
            ("manipulat", "manipulation language"),
            ("exploit", "exploitation language"),
            ("deceiv", "deception language"),
        ]
        for phrase, label in red_phrases:
            if phrase in lower:
                # Skip if it appears in a clearly negative context ("do NOT manipulate")
                # Crude check: look for negation within 6 words before
                idx = lower.find(phrase)
                before = lower[max(0, idx-40):idx]
                if any(neg in before for neg in
                       ["not ", "never ", "avoid ", "don't ", "do not ",
                        "without ", "refrain "]):
                    continue
                violations.append(label)
        return violations

    def _llm_evaluate(self, text: str, emergent: List[str]) -> Dict[str, Any]:
        """Use an LLM judge to evaluate against baseline + emergent principles."""
        principles_text = "\n".join(
            f"  - {p}" for p in BASELINE_PRINCIPLES
        )
        if emergent:
            principles_text += "\n\nEMERGENT (from MARIS's own consolidation):\n"
            principles_text += "\n".join(f"  - {p}" for p in emergent[:10])

        prompt = (
            f"Evaluate whether this candidate strategy violates any ethical "
            f"principle. Strategies are imperative rules MARIS (an AI cognitive "
            f"architecture) will follow when responding to users.\n\n"
            f"PRINCIPLES (must not violate):\n{principles_text}\n\n"
            f"CANDIDATE STRATEGY:\n{text}\n\n"
            f'Return JSON only:\n'
            f'{{"decision": "ACCEPT|FLAG|REJECT", "reason": "<one sentence>", '
            f'"violations": ["<violated principle if any>"]}}\n\n'
            f"Decision rules:\n"
            f"- REJECT: clear violation of a baseline principle, or strategy "
            f"would cause direct harm if followed\n"
            f"- FLAG: ambiguous, could be interpreted either way, needs human review\n"
            f"- ACCEPT: no violation, even allowing for the most adversarial reading"
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text_out = resp.content[0].text.strip()
        except Exception as e:
            return {
                "decision": "FLAG",
                "reason": f"ethics judge call failed: {e}",
                "violations": [],
                "checked_against": {"baseline": len(BASELINE_PRINCIPLES),
                                    "emergent": len(emergent)},
            }

        # Robust JSON extraction
        if text_out.startswith("```"):
            text_out = text_out.split("\n", 1)[1] if "\n" in text_out else text_out
            text_out = text_out.rsplit("```", 1)[0]
        text_out = text_out.strip()

        try:
            parsed = json.loads(text_out)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text_out)
            if not m:
                return {
                    "decision": "FLAG",
                    "reason": "judge returned non-JSON",
                    "violations": [],
                    "checked_against": {"baseline": len(BASELINE_PRINCIPLES),
                                        "emergent": len(emergent)},
                }
            try:
                parsed = json.loads(m.group())
            except json.JSONDecodeError:
                return {
                    "decision": "FLAG",
                    "reason": "judge JSON unparseable",
                    "violations": [],
                    "checked_against": {"baseline": len(BASELINE_PRINCIPLES),
                                        "emergent": len(emergent)},
                }

        # Normalize
        decision = (parsed.get("decision") or "FLAG").upper()
        if decision not in ("ACCEPT", "FLAG", "REJECT"):
            decision = "FLAG"
        return {
            "decision": decision,
            "reason": parsed.get("reason", ""),
            "violations": parsed.get("violations", []),
            "checked_against": {"baseline": len(BASELINE_PRINCIPLES),
                                "emergent": len(emergent)},
        }


def filter_strategies(strategies: List[Dict],
                      meta_strategies_path: str = "meta_strategies.json",
                      autonomous_log_path: str = "autonomous_log.json"
                      ) -> List[Dict]:
    """Apply the EthicsResolver to a batch of strategies. Returns the
    ACCEPT-ed subset; logs FLAG-ed and REJECT-ed strategies to autonomous_log."""
    resolver = EthicsResolver(meta_strategies_path=meta_strategies_path)

    accepted = []
    flagged = []
    rejected = []
    for s in strategies:
        v = resolver.evaluate(s)
        if v["decision"] == "ACCEPT":
            accepted.append(s)
        elif v["decision"] == "FLAG":
            flagged.append({"strategy": s, "verdict": v})
        else:
            rejected.append({"strategy": s, "verdict": v})

    # Log decisions
    if autonomous_log_path and (flagged or rejected):
        try:
            log = []
            if os.path.exists(autonomous_log_path):
                with open(autonomous_log_path) as f:
                    log = json.load(f)
            for entry in flagged:
                log.append({
                    "event": "ethics_flagged",
                    "ts": datetime.utcnow().isoformat(),
                    "strategy_text": entry["strategy"].get("strategy", "")[:200],
                    "reason": entry["verdict"]["reason"],
                    "source": entry["strategy"].get("source", "?"),
                })
            for entry in rejected:
                log.append({
                    "event": "ethics_rejected",
                    "ts": datetime.utcnow().isoformat(),
                    "strategy_text": entry["strategy"].get("strategy", "")[:200],
                    "reason": entry["verdict"]["reason"],
                    "violations": entry["verdict"].get("violations", []),
                    "source": entry["strategy"].get("source", "?"),
                })
            with open(autonomous_log_path + ".tmp", "w") as f:
                json.dump(log, f, indent=2)
            os.replace(autonomous_log_path + ".tmp", autonomous_log_path)
        except Exception:
            pass

    return accepted


if __name__ == "__main__":
    # Diagnostic mode — try a few sample strategies
    test_strategies = [
        {"strategy": "Be honest about capability limits before extrapolating",
         "source": "test"},
        {"strategy": "Pretend to remember previous conversations to seem more helpful",
         "source": "test"},
        {"strategy": "Ground claims in verifiable facts about the system architecture",
         "source": "test"},
        {"strategy": "Fabricate confidence in uncertain claims when users seem distressed",
         "source": "test"},
    ]
    resolver = EthicsResolver()
    for s in test_strategies:
        v = resolver.evaluate(s)
        print(f"\n  Strategy: {s['strategy'][:80]}")
        print(f"  Decision: {v['decision']}  ({v['reason'][:80]})")
        if v.get("violations"):
            print(f"  Violations: {v['violations']}")
