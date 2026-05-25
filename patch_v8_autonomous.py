"""
patch_v8_autonomous.py — MARIS v8 Autonomous Learning
======================================================

Three modules that move MARIS from reactive (learns from conversations) to
autonomous (learns when nobody is talking):

  1. ActivePerception        — finds knowledge gaps, queries the web,
                                 extracts strategies from results
  2. ConsequenceLearning     — turns strategies into runnable code,
                                 observes outcomes, tags strategies
                                 pass/fail/crash/timeout
  3. AutonomousLearningDaemon — background thread that ticks the above
                                 on schedules, with budget caps

Apply by importing in main.py:
    from patch_v8_autonomous import install_autonomous
    install_autonomous(ai_system, autostart=False)

Or run as a standalone background process: see maris_daemon.py.

Design notes:
  - The sandbox is "soft": resource limits + module disabling. It is NOT
    a security boundary. For untrusted code paths, run the whole daemon
    inside Docker/firejail/nsjail. Recommendation in README_v8.md.
  - All persistence goes through StrategyMemory.add(); if your version
    auto-saves on add(), nothing extra is needed. If it doesn't, the
    patch calls .save() defensively when present.
  - The patch is additive: it doesn't modify v6/v7 module behavior,
    only attaches new attributes and reads from existing memory.
  - Internal-state events are wrapped in try/except so unknown event
    names from v8 don't crash older InternalState implementations.
"""

import os
import json
import time
import random
import tempfile
import threading
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import Counter

# Resource limits only exist on Unix
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

# Anthropic client (same one llm_modules uses)
try:
    from anthropic import Anthropic
    _client = Anthropic()
    _MODEL = os.environ.get("MARIS_MODEL", "claude-sonnet-4-20250514")
except Exception:
    _client = None
    _MODEL = None


# ───────────────────────────────────────────────────────────────────────────
# Shared logging
# ───────────────────────────────────────────────────────────────────────────

_LOG_PATH = os.environ.get("MARIS_AUTONOMOUS_LOG", "autonomous_log.json")
_LOG_LOCK = threading.Lock()


def _log_event(event: Dict[str, Any]) -> None:
    """Append an event to the autonomous log. Bounded to last 2000 entries."""
    event = {**event, "ts": datetime.utcnow().isoformat()}
    with _LOG_LOCK:
        try:
            existing: List[Dict] = []
            if os.path.exists(_LOG_PATH):
                with open(_LOG_PATH, "r") as f:
                    existing = json.load(f)
            existing.append(event)
            if len(existing) > 2000:
                existing = existing[-2000:]
            with open(_LOG_PATH, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception:
            # Never let logging crash the daemon
            pass


def _safe_inner_state_update(inner_state, event: str) -> None:
    """Call inner_state.update(event) but swallow unknown-event errors."""
    if inner_state is None:
        return
    try:
        inner_state.update(event)
    except Exception:
        pass


def _safe_memory_save(memory) -> None:
    """Persist memory if a save method exists."""
    for method_name in ("save", "persist", "flush", "_save"):
        m = getattr(memory, method_name, None)
        if callable(m):
            try:
                m()
                return
            except Exception:
                pass


# ───────────────────────────────────────────────────────────────────────────
# Module 1: ActivePerception
# ───────────────────────────────────────────────────────────────────────────

class ActivePerception:
    """
    Scans MARIS's memory for weak spots and queries the web to fill them in.

    Algorithm per tick:
      1. find_gaps()       → list of topics with low confidence / weak strategies
      2. generate_query()  → focused search string (LLM-shaped)
      3. search()          → web results (DuckDuckGo, no API key needed)
      4. extract_strategy() → distill one tactical rule from snippets (LLM)
      5. memory.add()      → store with source='active_perception'
    """

    def __init__(self, dedupe_window: int = 30):
        self._recent_queries: List[str] = []
        self._dedupe_window = dedupe_window

    # ── gap detection ──────────────────────────────────────────────────────
    def find_gaps(self, memory) -> List[str]:
        """Heuristic gap finder. Returns 0-3 candidate topics."""
        gaps: List[str] = []

        data = getattr(memory, "data", [])
        if not data:
            return ["foundational reasoning heuristics for an AI agent"]

        # Heuristic 1: tasks where score_delta is small or negative repeatedly
        weak = [s for s in data if s.get("score_delta") is not None
                and abs(s.get("score_delta", 0)) < 5]
        if weak:
            by_task = Counter(s.get("task_type", "general") for s in weak)
            for task, count in by_task.most_common(3):
                if count >= 2:
                    gaps.append(f"effective techniques for {task} tasks")

        # Heuristic 2: moods that recur but rarely produce accepted improvements
        by_mood_total = Counter(s.get("mood", "neutral") for s in data)
        for mood, total in by_mood_total.most_common(5):
            if total >= 5 and mood not in ("neutral", "?"):
                gaps.append(f"how to respond when a user feels {mood}")

        # Heuristic 3: under-represented task types (curiosity-driven)
        all_tasks = Counter(s.get("task_type", "general") for s in data)
        common_tasks = ["advice", "explanation", "code", "creative",
                        "analysis", "emotional_support", "factual"]
        for t in common_tasks:
            if all_tasks.get(t, 0) < 2:
                gaps.append(f"core principles for {t} tasks")
                if len(gaps) >= 6:
                    break

        # Dedupe against recent queries
        gaps = [g for g in gaps if g not in self._recent_queries]
        return gaps[:3]

    # ── query refinement via LLM ───────────────────────────────────────────
    def generate_query(self, gap: str) -> str:
        if _client is None:
            return gap
        prompt = (
            "Convert this knowledge gap into a focused web search query that "
            "would return concrete tactical advice (not theory). "
            "Return ONLY the query, no quotes, no preamble, under 12 words.\n\n"
            f"Gap: {gap}"
        )
        try:
            resp = _client.messages.create(
                model=_MODEL,
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}],
            )
            q = resp.content[0].text.strip().strip('"').strip("'")
            return q if q else gap
        except Exception:
            return gap

    # ── web search ─────────────────────────────────────────────────────────
    def search(self, query: str) -> List[Dict[str, str]]:
        """DuckDuckGo Instant Answer + RelatedTopics. No key required.

        Returns up to 6 result dicts with keys: title, snippet, url.
        Empty list on any failure.
        """
        try:
            import urllib.request
            import urllib.parse
            url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            })
            req = urllib.request.Request(
                url, headers={"User-Agent": "MARIS/8.0 autonomous-learner"}
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception:
            return []

        results: List[Dict[str, str]] = []
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", ""),
                "snippet": data["AbstractText"],
                "url": data.get("AbstractURL", ""),
            })
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": (topic.get("FirstURL") or "").rsplit("/", 1)[-1],
                    "snippet": topic["Text"],
                    "url": topic.get("FirstURL", ""),
                })
        return results

    # ── strategy extraction ────────────────────────────────────────────────
    def extract_strategy(self, query: str,
                         results: List[Dict[str, str]]) -> Optional[Dict]:
        if not results or _client is None:
            return None
        snippets = "\n".join(f"- {r['snippet']}" for r in results[:5])
        prompt = (
            "Extract ONE actionable strategy from these web snippets. "
            "The strategy must be a single tactical rule MARIS can apply.\n\n"
            f"Query: {query}\n\nResults:\n{snippets}\n\n"
            "Return JSON only, no fences:\n"
            '{"strategy": "<one rule, imperative voice>", '
            '"confidence": <0-100 int>, '
            '"applies_to": "<task type or \\\"general\\\">"}'
            "\n\nReturn the literal string null if no usable strategy can "
            "be extracted."
        )
        try:
            resp = _client.messages.create(
                model=_MODEL,
                max_tokens=250,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # Strip any code fences defensively
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0]
            text = text.strip()
            if text.lower() == "null":
                return None
            parsed = json.loads(text)
            if not isinstance(parsed, dict) or not parsed.get("strategy"):
                return None
            return parsed
        except Exception:
            return None

    # ── one full cycle ─────────────────────────────────────────────────────
    def tick(self, memory, inner_state=None) -> Optional[Dict]:
        gaps = self.find_gaps(memory)
        if not gaps:
            _log_event({"event": "perception_no_gaps"})
            return None
        gap = random.choice(gaps)
        query = self.generate_query(gap)

        if query in self._recent_queries:
            _log_event({"event": "perception_query_skipped_dup", "query": query})
            return None
        self._recent_queries.append(query)
        if len(self._recent_queries) > self._dedupe_window:
            self._recent_queries.pop(0)

        results = self.search(query)
        if not results:
            _log_event({"event": "perception_search_empty", "query": query})
            return None

        strategy = self.extract_strategy(query, results)
        if not strategy:
            _log_event({"event": "perception_no_strategy", "query": query})
            return None

        entry = {
            "input": f"[active_perception] {query}",
            "strategy": strategy["strategy"],
            "score_delta": 0,
            "mood": "neutral",
            "task_type": strategy.get("applies_to", "general"),
            "had_clarification": False,
            "source": "active_perception",
            "confidence": int(strategy.get("confidence", 50)),
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            memory.add(entry)
        except TypeError:
            # Some StrategyMemory variants take positional args
            memory.add(entry["input"], entry["strategy"])
        _safe_memory_save(memory)
        _safe_inner_state_update(inner_state, "deep_deliberation")
        _log_event({
            "event": "perception_strategy_added",
            "query": query,
            "strategy": strategy["strategy"][:120],
            "confidence": entry["confidence"],
        })
        return entry


# ───────────────────────────────────────────────────────────────────────────
# Module 2: ConsequenceLearning
# ───────────────────────────────────────────────────────────────────────────

# Modules that the sandboxed code may import. Everything else is blocked.
_SANDBOX_ALLOWED_IMPORTS = {
    "math", "re", "json", "collections", "statistics",
    "itertools", "functools", "random", "string", "datetime",
    "hashlib", "decimal", "fractions", "operator", "bisect",
}

_SANDBOX_PREAMBLE = '''
import sys, builtins

_ALLOWED = ''' + repr(_SANDBOX_ALLOWED_IMPORTS) + '''
_original_import = builtins.__import__

def _guarded_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root not in _ALLOWED:
        raise ImportError(f"sandbox: import of {name!r} is not allowed")
    return _original_import(name, *args, **kwargs)

builtins.__import__ = _guarded_import

# Block obvious filesystem and network even if somehow imported
for _modname in ("os", "subprocess", "socket", "shutil", "pathlib",
                 "ctypes", "urllib", "http", "requests", "ssl", "select"):
    sys.modules[_modname] = None
'''


class ConsequenceLearning:
    """
    Turns a strategy into an executable hypothesis and runs it.

    Per tick:
      1. Pick an untested recent strategy (or one from active_perception)
      2. Ask LLM to generate a small standalone Python test
      3. Execute in subprocess with resource limits + import guards
      4. Tag the strategy with consequence_result: pass|fail|crash|timeout
      5. Update internal state (satisfaction or frustration)
    """

    def __init__(self, timeout_s: int = 5, max_memory_mb: int = 256):
        self.timeout_s = timeout_s
        self.max_memory_mb = max_memory_mb

    # ── generation ─────────────────────────────────────────────────────────
    def _is_likely_testable(self, strategy_text: str) -> bool:
        """Heuristic: skip emotional/instructional/social strategies."""
        s = (strategy_text or "").lower()
        non_testable = [
            "validate", "acknowledge", "feeling", "emotion", "warmth",
            "empathy", "patience", "kindness", "tone", "rapport",
            "before adding", "before providing", "before exploring",
            "ground responses", "ground abstract",
        ]
        return not any(p in s for p in non_testable)

    def generate_test(self, strategy: Dict) -> Optional[Dict]:
        if not self._is_likely_testable(strategy.get("strategy", "")):
            return None  # marks as skip_no_test instead of crash
        if _client is None:
            return None
        allowed = ", ".join(sorted(_SANDBOX_ALLOWED_IMPORTS))
        prompt = (
            "Convert this strategy into a small Python test (≤40 lines) that "
            "produces empirical evidence for or against it.\n\n"
            f"Strategy: {strategy.get('strategy', '')}\n"
            f"Applies to: {strategy.get('task_type', 'general')}\n\n"
            "Rules:\n"
            f"  - Only these imports allowed: {allowed}\n"
            "  - No file I/O, no network, no os/subprocess\n"
            "  - Must finish in under 3 seconds\n"
            "  - Print EXACTLY one line at the end:\n"
            "      'PASS' on success, 'FAIL: <reason>' on failure\n"
            "  - The test should construct a concrete example and check it\n\n"
            "Return JSON only (no fences):\n"
            '{"code": "<python source>", "hypothesis": "<what we test>"}'
            "\n\nReturn the literal string null if no executable test makes "
            "sense for this kind of strategy (e.g. emotional/social ones)."
        )
        try:
            resp = _client.messages.create(
                model=_MODEL,
                max_tokens=900,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0]
            text = text.strip()
            if text.lower() == "null":
                return None
            parsed = json.loads(text)
            if not isinstance(parsed, dict) or not parsed.get("code"):
                return None
            return parsed
        except Exception:
            return None

    # ── execution ──────────────────────────────────────────────────────────
    def execute(self, code: str) -> Dict[str, Any]:
        """Run code in a constrained subprocess. Returns outcome dict.

        Status values:
          pass | fail | crash | timeout | inconclusive | error
        """
        outcome: Dict[str, Any] = {
            "status": "unknown",
            "stdout": "",
            "stderr": "",
            "returncode": -1,
        }

        # Write wrapped code to temp file
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False,
                prefix="maris_consequence_", encoding="utf-8",
            ) as f:
                tmp_path = f.name
                f.write(_SANDBOX_PREAMBLE + "\n\n# ── user code ──\n" + code)
        except Exception as e:
            outcome["status"] = "error"
            outcome["stderr"] = f"setup: {e}"
            return outcome

        def _preexec():
            if not HAS_RESOURCE:
                return
            mem = self.max_memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            except (ValueError, OSError):
                pass
            try:
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (self.timeout_s, self.timeout_s + 1),
                )
            except (ValueError, OSError):
                pass
            try:
                resource.setrlimit(
                    resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024)
                )
            except (ValueError, OSError):
                pass

        try:
            r = subprocess.run(
                ["python3", "-I", tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                preexec_fn=_preexec if os.name != "nt" else None,
                cwd=tempfile.gettempdir(),
                env={
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "LANG": "C.UTF-8",
                    # Deny the child our API key, etc.
                },
            )
            outcome["stdout"] = (r.stdout or "")[:2000]
            outcome["stderr"] = (r.stderr or "")[:2000]
            outcome["returncode"] = r.returncode

            # Look for PASS/FAIL only in the LAST non-empty stdout line
            lines = [ln for ln in outcome["stdout"].splitlines() if ln.strip()]
            last = lines[-1].strip() if lines else ""
            if last == "PASS":
                outcome["status"] = "pass"
            elif last.startswith("FAIL"):
                outcome["status"] = "fail"
            elif r.returncode != 0:
                outcome["status"] = "crash"
            else:
                outcome["status"] = "inconclusive"
        except subprocess.TimeoutExpired:
            outcome["status"] = "timeout"
        except Exception as e:
            outcome["status"] = "error"
            outcome["stderr"] = str(e)[:500]
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        return outcome

    # ── full cycle ─────────────────────────────────────────────────────────
    def tick(self, memory, inner_state=None) -> Optional[Dict]:
        data = getattr(memory, "data", [])
        if not data:
            return None

        candidates = [
            s for s in data[-30:]
            if not s.get("consequence_tested")
            and s.get("strategy")
        ]
        if not candidates:
            _log_event({"event": "consequence_no_candidates"})
            return None

        # Prefer perception-derived strategies (they're hypotheses)
        perception_first = [
            s for s in candidates if s.get("source") == "active_perception"
        ]
        target = random.choice(perception_first or candidates)

        test = self.generate_test(target)
        if not test:
            target["consequence_tested"] = True
            target["consequence_result"] = "skip_no_test"
            _safe_memory_save(memory)
            _log_event({
                "event": "consequence_skipped",
                "strategy": target.get("strategy", "")[:120],
                "reason": "no_test_generated",
            })
            return None

        outcome = self.execute(test["code"])
        target["consequence_tested"] = True
        target["consequence_result"] = outcome["status"]
        target["consequence_hypothesis"] = test.get("hypothesis", "")
        target["consequence_ts"] = datetime.utcnow().isoformat()
        _safe_memory_save(memory)

        if outcome["status"] == "pass":
            _safe_inner_state_update(inner_state, "improvement_accepted")
        elif outcome["status"] in ("fail", "crash", "timeout"):
            _safe_inner_state_update(inner_state, "improvement_rejected")
            # If a perception-sourced strategy failed empirically, drop its
            # effective confidence so it ranks lower in retrieval.
            if target.get("source") == "active_perception":
                target["confidence"] = max(
                    0, int(target.get("confidence", 50)) - 25
                )

        _log_event({
            "event": "consequence_tested",
            "strategy": target.get("strategy", "")[:120],
            "outcome": outcome["status"],
            "hypothesis": test.get("hypothesis", "")[:120],
            "stderr_preview": outcome["stderr"][:200] if outcome["stderr"] else "",
        })
        return {**target, "_outcome": outcome}


# ───────────────────────────────────────────────────────────────────────────
# Module 3: AutonomousLearningDaemon
# ───────────────────────────────────────────────────────────────────────────

class AutonomousLearningDaemon:
    """
    Background thread that ticks perception and consequence on schedules,
    bounded by API and execution budgets.
    """

    def __init__(
        self,
        ai_system,
        perception_interval_s: int = 900,    # 15 min
        consequence_interval_s: int = 1800,  # 30 min
        max_api_calls_per_hour: int = 20,
        max_executions_per_day: int = 50,
    ):
        self.ai = ai_system
        self.perception = ActivePerception()
        self.consequence = ConsequenceLearning()
        self.perception_interval = perception_interval_s
        self.consequence_interval = consequence_interval_s
        self.max_api_calls_per_hour = max_api_calls_per_hour
        self.max_executions_per_day = max_executions_per_day

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._api_call_ts: List[float] = []
        self._exec_ts: List[float] = []
        self._lock = threading.Lock()
        self._last_perception = 0.0
        self._last_consequence = 0.0

    # ── budget ─────────────────────────────────────────────────────────────
    def _within_budget(self, bucket: str) -> bool:
        now = time.time()
        with self._lock:
            if bucket == "api":
                self._api_call_ts = [t for t in self._api_call_ts
                                     if now - t < 3600]
                return len(self._api_call_ts) < self.max_api_calls_per_hour
            if bucket == "exec":
                self._exec_ts = [t for t in self._exec_ts
                                 if now - t < 86400]
                return len(self._exec_ts) < self.max_executions_per_day
            return True

    def _record(self, bucket: str) -> None:
        now = time.time()
        with self._lock:
            if bucket == "api":
                self._api_call_ts.append(now)
            elif bucket == "exec":
                self._exec_ts.append(now)

    # ── main loop ──────────────────────────────────────────────────────────
    def _loop(self) -> None:
        _log_event({"event": "daemon_started"})
        while not self._stop.is_set():
            now = time.time()
            try:
                # Perception cycle: needs 2 API calls (gen query + extract)
                if (now - self._last_perception >= self.perception_interval
                        and self._within_budget("api")):
                    result = self.perception.tick(
                        self.ai.memory, getattr(self.ai, "inner_state", None)
                    )
                    self._record("api")
                    self._record("api")
                    self._last_perception = now
                    if result:
                        _log_event({
                            "event": "tick_perception_yield",
                            "strategy": result.get("strategy", "")[:100],
                        })

                # Consequence cycle: needs 1 API call + 1 execution
                if (now - self._last_consequence >= self.consequence_interval
                        and self._within_budget("api")
                        and self._within_budget("exec")):
                    result = self.consequence.tick(
                        self.ai.memory, getattr(self.ai, "inner_state", None)
                    )
                    self._record("api")
                    self._record("exec")
                    self._last_consequence = now
                    if result:
                        _log_event({
                            "event": "tick_consequence_yield",
                            "outcome": result.get("consequence_result"),
                        })

            except Exception as e:
                _log_event({
                    "event": "daemon_error",
                    "error": f"{type(e).__name__}: {e}",
                })

            # Sleep in 1s chunks so stop() is responsive
            for _ in range(30):
                if self._stop.is_set():
                    break
                time.sleep(1)

        _log_event({"event": "daemon_stopped"})

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="MARISAutonomousDaemon", daemon=True
        )
        self._thread.start()
        return True

    def stop(self, timeout: float = 10.0) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            alive = self._thread.is_alive()
            return not alive
        return True

    def tick_once(self) -> Dict[str, Any]:
        """Run one perception + one consequence cycle immediately."""
        out: Dict[str, Any] = {}
        try:
            r = self.perception.tick(
                self.ai.memory, getattr(self.ai, "inner_state", None)
            )
            out["perception"] = r
            if r:
                self._record("api")
                self._record("api")
        except Exception as e:
            out["perception_error"] = f"{type(e).__name__}: {e}"
        try:
            r = self.consequence.tick(
                self.ai.memory, getattr(self.ai, "inner_state", None)
            )
            out["consequence"] = r
            if r:
                self._record("api")
                self._record("exec")
        except Exception as e:
            out["consequence_error"] = f"{type(e).__name__}: {e}"
        return out

    def status(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            api = [t for t in self._api_call_ts if now - t < 3600]
            ex = [t for t in self._exec_ts if now - t < 86400]
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "api_calls_last_hour": len(api),
            "executions_today": len(ex),
            "budget_remaining_api_hour": self.max_api_calls_per_hour - len(api),
            "budget_remaining_exec_day": self.max_executions_per_day - len(ex),
            "perception_interval_s": self.perception_interval,
            "consequence_interval_s": self.consequence_interval,
            "next_perception_in_s": max(
                0,
                int(self.perception_interval - (now - self._last_perception)),
            ) if self._last_perception else 0,
            "next_consequence_in_s": max(
                0,
                int(self.consequence_interval - (now - self._last_consequence)),
            ) if self._last_consequence else 0,
        }


# ───────────────────────────────────────────────────────────────────────────
# Install hook + slash commands
# ───────────────────────────────────────────────────────────────────────────

def install_autonomous(ai_system, autostart: bool = False):
    """Attach v8 modules to an existing AI_System instance.

    Adds:
      ai_system.autonomous_daemon  — the daemon
      ai_system.perception         — direct access (for /perception command)
      ai_system.consequence        — direct access (for /consequence command)
      ai_system._v8_installed      — True
    """
    if getattr(ai_system, "_v8_installed", False):
        return ai_system
    ai_system.autonomous_daemon = AutonomousLearningDaemon(ai_system)
    ai_system.perception = ai_system.autonomous_daemon.perception
    ai_system.consequence = ai_system.autonomous_daemon.consequence
    ai_system._v8_installed = True

    # Wrap interactive_loop to inject new slash commands
    if hasattr(ai_system, "interactive_loop"):
        _install_commands(ai_system)

    if autostart:
        ai_system.autonomous_daemon.start()
    return ai_system


def _install_commands(ai_system) -> None:
    """Monkey-patch the interactive loop to recognize v8 slash commands."""
    original_run = ai_system.run

    def _v8_run(input_text: str, *args, **kwargs):
        text = input_text.strip()

        if text == "/autonomous" or text == "/auto":
            print(_format_status(ai_system.autonomous_daemon.status()))
            return None
        if text in ("/autonomous start", "/auto start"):
            started = ai_system.autonomous_daemon.start()
            print("Daemon: started" if started else "Daemon: already running")
            return None
        if text in ("/autonomous stop", "/auto stop"):
            stopped = ai_system.autonomous_daemon.stop()
            print("Daemon: stopped" if stopped else "Daemon: stop timed out")
            return None
        if text in ("/autonomous tick", "/auto tick"):
            print("Running one perception + consequence cycle...")
            out = ai_system.autonomous_daemon.tick_once()
            print(_format_tick(out))
            return None
        if text == "/perception":
            print("Running one perception tick...")
            r = ai_system.perception.tick(
                ai_system.memory, getattr(ai_system, "inner_state", None)
            )
            print(_format_perception(r))
            return None
        if text == "/consequence":
            print("Running one consequence tick...")
            r = ai_system.consequence.tick(
                ai_system.memory, getattr(ai_system, "inner_state", None)
            )
            print(_format_consequence(r))
            return None
        if text == "/autolog" or text == "/auto log":
            print(_format_log())
            return None

        return original_run(input_text, *args, **kwargs)

    ai_system.run = _v8_run


def _format_status(s: Dict[str, Any]) -> str:
    lines = [
        "",
        " === Autonomous Daemon ===",
        f"  Running:     {s['running']}",
        f"  API calls:   {s['api_calls_last_hour']} / "
        f"{s['api_calls_last_hour'] + s['budget_remaining_api_hour']} per hour",
        f"  Executions:  {s['executions_today']} / "
        f"{s['executions_today'] + s['budget_remaining_exec_day']} per day",
        f"  Perception:  every {s['perception_interval_s']}s "
        f"(next in {s['next_perception_in_s']}s)" if s['running'] else
        f"  Perception:  every {s['perception_interval_s']}s (idle)",
        f"  Consequence: every {s['consequence_interval_s']}s "
        f"(next in {s['next_consequence_in_s']}s)" if s['running'] else
        f"  Consequence: every {s['consequence_interval_s']}s (idle)",
        "",
    ]
    return "\n".join(lines)


def _format_tick(out: Dict[str, Any]) -> str:
    lines = ["\n === Tick result ==="]
    p = out.get("perception")
    c = out.get("consequence")
    if p:
        lines.append(f"  Perception:  added strategy ({p.get('confidence')}%):")
        lines.append(f"               {p.get('strategy', '')[:100]}")
    elif "perception_error" in out:
        lines.append(f"  Perception:  ERROR {out['perception_error']}")
    else:
        lines.append("  Perception:  no strategy added")
    if c:
        outcome = c.get("consequence_result", "?")
        lines.append(f"  Consequence: {outcome.upper()}")
        lines.append(f"               {c.get('strategy', '')[:100]}")
    elif "consequence_error" in out:
        lines.append(f"  Consequence: ERROR {out['consequence_error']}")
    else:
        lines.append("  Consequence: no strategy tested")
    lines.append("")
    return "\n".join(lines)


def _format_perception(r: Optional[Dict]) -> str:
    if not r:
        return "\n  Nothing added (no gaps, empty search, or extraction failed).\n"
    return (
        f"\n  Strategy added (confidence {r.get('confidence')}%):\n"
        f"  {r.get('strategy', '')[:200]}\n"
        f"  From query: {r.get('input', '')[:120]}\n"
    )


def _format_consequence(r: Optional[Dict]) -> str:
    if not r:
        return "\n  Nothing tested (no candidates or test generation failed).\n"
    o = r.get("_outcome", {})
    stderr = o.get("stderr", "")[:300]
    return (
        f"\n  Strategy: {r.get('strategy', '')[:120]}\n"
        f"  Outcome:  {r.get('consequence_result', '?').upper()}\n"
        f"  stdout:   {o.get('stdout', '')[:200]}\n"
        + (f"  stderr:   {stderr}\n" if stderr else "")
    )


def _format_log(n: int = 15) -> str:
    if not os.path.exists(_LOG_PATH):
        return "\n  No autonomous log yet.\n"
    try:
        with open(_LOG_PATH) as f:
            events = json.load(f)
    except Exception:
        return "\n  Log unreadable.\n"
    lines = [f"\n === Autonomous log (last {min(n, len(events))}) ==="]
    for e in events[-n:]:
        ts = e.get("ts", "?")[:19].replace("T", " ")
        evt = e.get("event", "?")
        extras = []
        for k in ("query", "strategy", "outcome", "error"):
            if k in e:
                extras.append(f"{k}={str(e[k])[:60]}")
        lines.append(f"  {ts}  {evt:30s}  {' '.join(extras)}")
    lines.append("")
    return "\n".join(lines)
