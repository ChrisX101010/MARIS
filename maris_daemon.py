"""
maris_daemon.py — MARIS v8 standalone autonomous learner

Runs ActivePerception + ConsequenceLearning continuously without an
interactive session. Shares JSON state files (strategy_memory.json,
meta_strategies.json, insights.json) with the interactive main.py via
file locking — both can run safely at the same time.

Use cases:
  - Always-on learning between human conversations
  - Run on a small VPS / Raspberry Pi
  - Spawned by systemd (see maris.service)

Usage:
  python maris_daemon.py                # run forever, default intervals
  python maris_daemon.py --once         # one tick then exit
  python maris_daemon.py --status       # print status from log + exit
  python maris_daemon.py --perception-interval 600 --consequence-interval 1200

Environment:
  ANTHROPIC_API_KEY        — required
  MARIS_MODEL              — defaults to claude-sonnet-4-20250514
  MARIS_DATA_DIR           — where JSON state lives (default: cwd)
  MARIS_AUTONOMOUS_LOG     — path to log file (default: autonomous_log.json)

The daemon is intentionally minimal: it imports only the pieces of MARIS
it needs (StrategyMemory, InternalState) and the v8 patch. It does NOT
load EmotionModule, ReasoningModule, etc. — those exist for handling
human input, not background learning.
"""

import os
import sys
import time
import json
import signal
import argparse
import fcntl
from pathlib import Path
from datetime import datetime


# ───────────────────────────────────────────────────────────────────────────
# File-locking wrapper around StrategyMemory so the daemon and interactive
# session can both touch strategy_memory.json without corrupting it.
# ───────────────────────────────────────────────────────────────────────────

class LockedStrategyMemory:
    """Wraps the real StrategyMemory and serializes reads/writes via flock.

    Drop-in compatible with the parts ActivePerception and ConsequenceLearning
    use: .data (list), .add(entry), .save() if present, plus an optional
    .reload() that re-reads from disk so we see new entries written by
    other processes.
    """

    def __init__(self, real_memory, lock_path=".maris_memory.lock"):
        self._real = real_memory
        self._lock_path = lock_path

    def _acquire(self):
        # Open a sidecar lock file (don't lock the data file itself —
        # the data file gets rewritten on save, breaking the lock).
        f = open(self._lock_path, "a+")
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        return f

    @property
    def data(self):
        return self._real.data

    @property
    def meta_strategies(self):
        return getattr(self._real, "meta_strategies", [])

    def add(self, *args, **kwargs):
        lock = self._acquire()
        try:
            # Reload before mutating so we don't blow away another
            # process's recent writes.
            self.reload()
            result = self._real.add(*args, **kwargs)
            self.save()
            return result
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def save(self):
        for name in ("save", "persist", "flush", "_save"):
            m = getattr(self._real, name, None)
            if callable(m):
                try:
                    m()
                    return
                except Exception:
                    continue

    def reload(self):
        """Re-read state from disk if the underlying class supports it."""
        for name in ("reload", "load", "_load"):
            m = getattr(self._real, name, None)
            if callable(m):
                try:
                    m()
                    return
                except Exception:
                    pass

    def strategy_count(self):
        if hasattr(self._real, "strategy_count"):
            return self._real.strategy_count()
        return len(self.data)

    def meta_count(self):
        if hasattr(self._real, "meta_count"):
            return self._real.meta_count()
        return len(self.meta_strategies)

    def get_strategies(self, *args, **kwargs):
        return self._real.get_strategies(*args, **kwargs)


# ───────────────────────────────────────────────────────────────────────────
# Minimal AI_System stand-in for the daemon
# ───────────────────────────────────────────────────────────────────────────

class _DaemonAISystem:
    """Just enough surface area for the autonomous daemon to operate on."""
    def __init__(self, data_dir: str):
        os.chdir(data_dir)
        from llm_modules import StrategyMemory, InternalState
        self.memory = LockedStrategyMemory(StrategyMemory())
        self.inner_state = InternalState()


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MARIS v8 autonomous learning daemon"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run one tick (perception + consequence) and exit",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print daemon status from log and exit",
    )
    parser.add_argument(
        "--perception-interval", type=int, default=900,
        help="Seconds between perception cycles (default: 900 = 15 min)",
    )
    parser.add_argument(
        "--consequence-interval", type=int, default=1800,
        help="Seconds between consequence cycles (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--max-api-per-hour", type=int, default=20,
        help="Cap on Anthropic API calls per hour (default: 20)",
    )
    parser.add_argument(
        "--max-exec-per-day", type=int, default=50,
        help="Cap on sandbox executions per day (default: 50)",
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=os.environ.get("MARIS_DATA_DIR", os.getcwd()),
        help="Directory containing strategy_memory.json (default: cwd)",
    )
    args = parser.parse_args()

    # Status path: no need to import LLM modules
    if args.status:
        _print_status_from_log()
        return 0

    # Sanity check
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 1

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: data dir {args.data_dir!r} not found.", file=sys.stderr)
        return 1

    # Bring up minimal AI system + daemon
    ai = _DaemonAISystem(args.data_dir)
    from patch_v8_autonomous import AutonomousLearningDaemon

    daemon = AutonomousLearningDaemon(
        ai,
        perception_interval_s=args.perception_interval,
        consequence_interval_s=args.consequence_interval,
        max_api_calls_per_hour=args.max_api_per_hour,
        max_executions_per_day=args.max_exec_per_day,
    )

    print(f"[{_ts()}] MARIS v8 daemon starting in {args.data_dir}")
    print(f"[{_ts()}] strategies={ai.memory.strategy_count()} "
          f"meta={ai.memory.meta_count()}")
    print(f"[{_ts()}] perception every {args.perception_interval}s, "
          f"consequence every {args.consequence_interval}s")
    print(f"[{_ts()}] budget: {args.max_api_per_hour} API/hr, "
          f"{args.max_exec_per_day} exec/day")

    if args.once:
        print(f"[{_ts()}] running one tick...")
        result = daemon.tick_once()
        print(f"[{_ts()}] result: {json.dumps(_summarize(result), indent=2)}")
        return 0

    # Graceful shutdown on SIGTERM/SIGINT
    def _shutdown(signum, frame):
        print(f"\n[{_ts()}] signal {signum}, stopping daemon...")
        daemon.stop(timeout=15)
        print(f"[{_ts()}] stopped.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    daemon.start()
    print(f"[{_ts()}] running. Ctrl-C or SIGTERM to stop.")

    # Periodic heartbeat
    try:
        while True:
            time.sleep(300)  # 5 min heartbeat
            s = daemon.status()
            print(f"[{_ts()}] heartbeat: api={s['api_calls_last_hour']} "
                  f"exec={s['executions_today']} "
                  f"strategies={ai.memory.strategy_count()}")
    except KeyboardInterrupt:
        _shutdown(signal.SIGINT, None)
    return 0


def _summarize(result):
    """Compact summary of tick_once result for logging."""
    out = {}
    p = result.get("perception")
    c = result.get("consequence")
    if p:
        out["perception"] = {
            "strategy": p.get("strategy", "")[:120],
            "confidence": p.get("confidence"),
        }
    elif "perception_error" in result:
        out["perception_error"] = result["perception_error"]
    if c:
        out["consequence"] = {
            "strategy": c.get("strategy", "")[:120],
            "outcome": c.get("consequence_result"),
        }
    elif "consequence_error" in result:
        out["consequence_error"] = result["consequence_error"]
    return out


def _print_status_from_log():
    log_path = os.environ.get("MARIS_AUTONOMOUS_LOG", "autonomous_log.json")
    if not os.path.exists(log_path):
        print("No log found at", log_path)
        return
    try:
        with open(log_path) as f:
            events = json.load(f)
    except Exception as e:
        print("Could not read log:", e)
        return

    print(f"\n=== MARIS Autonomous Daemon Status ===")
    print(f"Log:    {log_path}")
    print(f"Events: {len(events)}")
    if not events:
        return

    last_event = events[-1]
    print(f"Last:   {last_event.get('ts', '?')} {last_event.get('event', '?')}")

    # Count events by type in last 24h
    from collections import Counter
    cutoff = datetime.utcnow().timestamp() - 86400
    recent = []
    for e in events:
        try:
            t = datetime.fromisoformat(e.get("ts", "")).timestamp()
            if t >= cutoff:
                recent.append(e)
        except Exception:
            pass

    counts = Counter(e.get("event", "?") for e in recent)
    print(f"\nLast 24h: {len(recent)} events")
    for evt, n in counts.most_common():
        print(f"  {evt:35s} {n}")
    print()


def _ts():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")


if __name__ == "__main__":
    sys.exit(main())
