"""
main_v8.py — MARIS v8 interactive entry point

Same as main.py but with the autonomous learning patch installed.
The daemon does NOT autostart — you control it with slash commands:

    /autonomous           show daemon status
    /autonomous start     start background learning
    /autonomous stop      stop background learning
    /autonomous tick      run one cycle manually (perception + consequence)
    /perception           run a single perception tick
    /consequence          run a single consequence tick
    /autolog              show last 15 entries from autonomous log

If you'd rather use the daemon separately (e.g. via systemd) and keep
the interactive session purely reactive, just run main.py and skip
this file. The daemon and interactive session can run simultaneously —
strategy_memory.json is locked on writes.
"""

import sys
from main import AI_System
from patch_v8_autonomous import install_autonomous


def main():
    ai = AI_System()
    install_autonomous(ai, autostart=False)

    print()
    print(" v8 autonomous learning installed.")
    print(" Commands: /autonomous /perception /consequence /autolog")
    print(" Type /autonomous start to begin background learning.")
    print()

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        from main import demo
        demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "--auto":
        # Convenience: run with autonomous daemon already started
        ai.autonomous_daemon.start()
        print(" Daemon: started (will tick every 15/30 min)\n")
        ai.interactive_loop()
        ai.autonomous_daemon.stop()
    else:
        ai.interactive_loop()


if __name__ == "__main__":
    main()
