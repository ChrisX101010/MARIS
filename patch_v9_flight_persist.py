"""
patch_v9_flight_persist.py — v2
================================

Make FlightRecorder persist to flight_log.json.

v1 hooked into end_turn() and tried to discover the turn structure
heuristically. That didn't work reliably across FlightRecorder versions,
so v2 takes the simpler, more robust approach:

  Hook into record() directly. Every event the pipeline records gets
  appended to flight_log.json as a flat entry: {kind, payload, ts}.

This matches the shape the dashboard expects without needing to introspect
turn objects. It also means events appear in the log as they happen —
even if MARIS crashes mid-turn, partial data is preserved.

On first event you'll see a one-time confirmation in your terminal, then
it goes quiet.
"""

import os
import json
import threading
from datetime import datetime
from typing import Any, Dict


_LOCK = threading.Lock()
_FLIGHT_PATH = os.environ.get("MARIS_FLIGHT_LOG", "flight_log.json")
_FIRST_CALL_FLAG = {"announced": False}


def _atomic_append(path: str, entry: Dict[str, Any],
                   max_entries: int = 10000) -> bool:
    """Append entry to JSON list at path, atomically."""
    with _LOCK:
        try:
            existing = []
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
                except json.JSONDecodeError:
                    existing = []
            existing.append(entry)
            if len(existing) > max_entries:
                existing = existing[-max_entries:]
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(existing, f, indent=2, default=str)
            os.replace(tmp, path)
            return True
        except Exception as e:
            if not _FIRST_CALL_FLAG.get("error_printed"):
                print(f"  [flight_persist] write error: {e}")
                _FIRST_CALL_FLAG["error_printed"] = True
            return False


def _safe_payload(payload: Any) -> Any:
    """Make payload JSON-serializable."""
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload
    if isinstance(payload, dict):
        return {k: _safe_payload(v) for k, v in payload.items()
                if not k.startswith("_")}
    if isinstance(payload, (list, tuple)):
        return [_safe_payload(x) for x in payload]
    if hasattr(payload, "__dict__"):
        try:
            return {k: _safe_payload(v) for k, v in payload.__dict__.items()
                    if not k.startswith("_") and not callable(v)}
        except Exception:
            pass
    return str(payload)[:500]


def install_flight_persistence(ai_system, log_path: str = None) -> bool:
    """Hook FlightRecorder.record() so every event writes to flight_log.json."""
    fr = getattr(ai_system, "flight_recorder", None)
    if fr is None:
        print("  [flight_persist] no flight_recorder attribute -- skipping")
        return False
    if getattr(fr, "_v9_persistence_v2_installed", False):
        return False

    path = log_path or _FLIGHT_PATH
    fr._v9_flight_path = path

    _orig_record = getattr(fr, "record", None)
    if _orig_record is None or not callable(_orig_record):
        print("  [flight_persist] flight_recorder has no record() -- skipping")
        return False

    def _persistent_record(*args, **kwargs):
        kind = None
        payload = None
        if args:
            kind = args[0]
            if len(args) > 1:
                payload = args[1]
            elif "payload" in kwargs:
                payload = kwargs["payload"]
        elif "kind" in kwargs:
            kind = kwargs["kind"]
            payload = kwargs.get("payload")

        result = _orig_record(*args, **kwargs)

        if kind is not None:
            entry = {
                "kind": str(kind),
                "payload": _safe_payload(payload) if payload else {},
                "ts": datetime.utcnow().isoformat(),
            }
            ok = _atomic_append(path, entry)
            if ok and not _FIRST_CALL_FLAG["announced"]:
                _FIRST_CALL_FLAG["announced"] = True
                print(f"  [flight_persist] writing events to {path}")
        return result

    fr.record = _persistent_record

    _orig_start = getattr(fr, "start_turn", None)
    if _orig_start and callable(_orig_start):
        def _persistent_start(*args, **kwargs):
            result = _orig_start(*args, **kwargs)
            input_text = (args[0] if args
                          else kwargs.get("input_text", kwargs.get("text", "")))
            _atomic_append(path, {
                "kind": "turn_start",
                "payload": {"input": str(input_text)[:1000]},
                "ts": datetime.utcnow().isoformat(),
            })
            return result
        fr.start_turn = _persistent_start

    _orig_end = getattr(fr, "end_turn", None)
    if _orig_end and callable(_orig_end):
        def _persistent_end(*args, **kwargs):
            result = _orig_end(*args, **kwargs)
            _atomic_append(path, {
                "kind": "turn_end",
                "payload": {},
                "ts": datetime.utcnow().isoformat(),
            })
            return result
        fr.end_turn = _persistent_end

    fr._v9_persistence_v2_installed = True
    return True


install = install_flight_persistence


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else _FLIGHT_PATH
    print(f"\n  Inspecting: {path}")
    if not os.path.exists(path):
        print(f"  X File does not exist yet")
        print(f"  -> Have a conversation through main_v9.py, then re-run this")
        sys.exit(0)
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  X Invalid JSON: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print(f"  X Root is not a list (got {type(data).__name__})")
        sys.exit(1)

    print(f"  OK {len(data)} entries\n")

    from collections import Counter
    kinds = Counter(e.get("kind", "?") for e in data)
    print(f"  Event counts:")
    for k, n in kinds.most_common():
        print(f"    {k:25s} {n}")

    senate = [e for e in data if e.get("kind") == "senate"]
    if senate:
        print(f"\n  Senate verdict samples:")
        for e in senate[:5]:
            payload = e.get("payload", {})
            acc = payload.get("accepted") if isinstance(payload, dict) else "?"
            print(f"    {e.get('ts', '?')[:19]}  accepted={acc}")

    print(f"\n  Last 3 events:")
    for e in data[-3:]:
        ts = e.get("ts", "?")[:19]
        kind = e.get("kind", "?")
        payload = e.get("payload", {})
        if isinstance(payload, dict):
            keys = list(payload.keys())[:4]
            preview = f"{{{', '.join(keys)}}}" if keys else "{}"
        else:
            preview = str(payload)[:60]
        print(f"    {ts}  {kind:20s}  {preview}")
    print()
