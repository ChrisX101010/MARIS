"""
main_v9.py — MARIS v9 entry point
===================================

Installs v8 autonomous + v9 fixes, then runs the interactive REPL.

  python main_v9.py              # interactive, daemon idle
  python main_v9.py --auto       # interactive, daemon started
  python main_v9.py --demo       # automated demo
  python main_v9.py --once       # boot, verify, exit (smoke test)

All v6/v7/v8/v9 commands work:
  /memory /history /consolidate /insights /stage /progress
  /feelings /log /patterns /stats /clear           (v6/v7)
  /autonomous /perception /consequence /autolog    (v8)
  /search-test /state /drives /embed-info          (v9)
"""

import sys
import os
try:
    import readline  # arrow-key history
except ImportError:
    pass


def smoke_test(ai):
    """Verify each module loaded and is reachable. Used by --once."""
    print("\n  === v9 Smoke Test ===")
    checks = [
        ("memory",            hasattr(ai, "memory")),
        ("inner_state",       hasattr(ai, "inner_state")),
        ("flight_recorder",   hasattr(ai, "flight_recorder")),
        ("drive_resolver",    hasattr(ai, "drive_resolver")),
        ("autonomous_daemon", hasattr(ai, "autonomous_daemon")),
        ("perception",        hasattr(ai, "perception")),
        ("consequence",       hasattr(ai, "consequence")),
        ("v8 installed",      getattr(ai, "_v8_installed", False)),
        ("v9 installed",      getattr(ai, "_v9_installed", False)),
        ("state persistence",
            getattr(ai.inner_state, "_v9_persistence_installed", False)
            if hasattr(ai, "inner_state") else False),
        ("drive tuning",
            getattr(ai.drive_resolver, "_v9_tuned", False)
            if hasattr(ai, "drive_resolver") else False),
        ("v9 search backend",
            hasattr(getattr(ai, "perception", None), "_search_backend")),
        ("embedding upgrade",
            getattr(ai.memory, "_v9_embed_patched", False)
            if hasattr(ai, "memory") else False),
    ]
    all_ok = True
    for name, ok in checks:
        mark = "✓" if ok else "✗"
        print(f"    {mark} {name}")
        if not ok:
            all_ok = False

    # Bonus: actually test the search backend if available
    if hasattr(ai, "perception") and hasattr(ai.perception, "_search_backend"):
        backend = ai.perception._search_backend
        print(f"\n    Search backend selection: {backend.backend}")
        print(f"    Available backends: {backend._available}")

    print(f"\n  Result: {'ALL OK' if all_ok else 'SOME ISSUES'}\n")
    return all_ok


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MARIS v9")
    parser.add_argument("--auto", action="store_true",
                        help="Start autonomous daemon immediately")
    parser.add_argument("--demo", action="store_true",
                        help="Run automated demo")
    parser.add_argument("--once", action="store_true",
                        help="Boot, smoke-test, exit (no REPL)")
    parser.add_argument("--embed", choices=["voyage", "tfidf", "bow"],
                        help="Embedding provider (default: tfidf)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        return 1

    print("\n  Booting MARIS v9...")
    from main import AI_System
    ai = AI_System()

    # v8: autonomous learning
    try:
        from patch_v8_autonomous import install_autonomous
        install_autonomous(ai, autostart=False)
        print("    v8 (autonomous) installed")
    except ImportError:
        print("    v8 (autonomous) NOT installed — patch_v8_autonomous.py missing")

    # v9: fixes + upgrades
    try:
        from patch_v9 import install_v9
        install_v9(ai, embedding_provider=args.embed)
        try:
            from patch_v10_mood_colors import install_mood_colors
            install_mood_colors(ai)
        except ImportError:
            pass
        try:
            from patch_v9_flight_persist import install_flight_persistence
            install_flight_persistence(ai)
            print("    v9 flight persistence installed")
        except ImportError:
            pass
        print("    v9 (fixes + embed) installed")
    except ImportError as e:
        print(f"    v9 NOT installed: {e}")

    print(f"\n    Strategies: {ai.memory.strategy_count()}")
    print(f"    Meta-rules: {ai.memory.meta_count()}")
    if hasattr(ai, "insight_detector"):
        try:
            print(f"    Insights:   {ai.insight_detector.insight_count()}")
        except Exception:
            pass

    if args.once:
        return 0 if smoke_test(ai) else 1

    if args.demo:
        from main import demo
        demo()
        return 0

    if args.auto and hasattr(ai, "autonomous_daemon"):
        ai.autonomous_daemon.start()
        print("\n    Autonomous daemon: STARTED\n")

    print()
    print("    v9 commands: /search-test /state /drives /embed-info")
    print("    v8 commands: /autonomous /perception /consequence /autolog")
    print("    v6 commands: /memory /insights /feelings /stage /log\n")

    try:
        ai.interactive_loop()
    finally:
        if hasattr(ai, "autonomous_daemon"):
            ai.autonomous_daemon.stop(timeout=5)
        # Persist internal state on exit
        if hasattr(ai, "inner_state") and hasattr(ai.inner_state, "persist"):
            ai.inner_state.persist()
            print("    Internal state saved.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
