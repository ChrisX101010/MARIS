"""
patch_v10_mood_colors.py — terminal text colors driven by MARIS's mood

Maps her dominant internal emotion to an ANSI color, applied to the FINAL
OUTPUT only (not the pipeline trace, so the diagnostics stay readable).

Color scheme matches the dashboard:
  curiosity     → bright blue
  satisfaction  → bright green
  excitement    → magenta
  warmth        → yellow
  anxiety       → dim yellow
  frustration   → red
  neutral       → default (no color)

Install:
  In main_v9.py, after the existing install_v9() call, add:
      from patch_v10_mood_colors import install_mood_colors
      install_mood_colors(ai)
"""

import os
import json
from typing import Optional


# ANSI color codes
RESET = "\033[0m"
COLORS = {
    "curiosity":    "\033[94m",   # bright blue
    "satisfaction": "\033[92m",   # bright green
    "excitement":   "\033[95m",   # magenta
    "warmth":       "\033[93m",   # yellow
    "anxiety":      "\033[33m",   # dim yellow
    "frustration":  "\033[91m",   # red
}


def _detect_dominant_emotion(state_path: str = "internal_state.json"
                             ) -> Optional[str]:
    """Return the strongest non-trivial emotion, or None if neutral."""
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Try the saved 'dominant' field first
    dom = data.get("dominant", {})
    if isinstance(dom, dict) and dom.get("emotion"):
        intensity = dom.get("intensity", 0)
        if abs(intensity) > 0.1:  # threshold — below this is "neutral"
            return dom["emotion"]

    # Fall back to scanning state dict for the strongest value
    state = data.get("state", {})
    if not state:
        return None
    strongest = max(state.items(), key=lambda kv: abs(kv[1]), default=(None, 0))
    name, val = strongest
    if val is None or abs(val) < 0.1:
        return None
    return name


def color_for_emotion(emotion: Optional[str]) -> str:
    """Return the ANSI prefix for an emotion, or empty string for neutral."""
    if not emotion:
        return ""
    return COLORS.get(emotion, "")


def install_mood_colors(ai_system, state_path: str = "internal_state.json"
                        ) -> bool:
    """Wrap ai_system.run() so the FINAL OUTPUT print uses mood colors.

    The wrapping is light-touch: we patch the print of the final response
    by monkey-patching builtins.print only during the colored region. To
    avoid breaking the pipeline trace (which has its own structure), we
    only color lines that look like response output.

    Returns True if installed.
    """
    if getattr(ai_system, "_v10_mood_colors_installed", False):
        return False

    # Strategy: wrap the run() method. After the response is generated,
    # read internal_state.json, get dominant emotion, set ANSI prefix on
    # the response string.
    import builtins
    _real_print = builtins.print

    # We track whether we're "inside" a FINAL OUTPUT block via state
    state = {"in_final": False, "color": ""}

    def _colored_print(*args, **kwargs):
        # Detect FINAL OUTPUT boundary
        if args and isinstance(args[0], str):
            line = args[0]
            if "FINAL OUTPUT:" in line:
                # Refresh emotion at the boundary
                emo = _detect_dominant_emotion(state_path)
                state["color"] = color_for_emotion(emo)
                state["in_final"] = True
                if state["color"]:
                    _real_print(state["color"], end="")
                _real_print(*args, **kwargs)
                return
            if line.startswith("=" * 10) and state["in_final"]:
                # Closing boundary of the FINAL OUTPUT block
                if state["color"]:
                    _real_print(RESET, end="")
                state["in_final"] = False
                state["color"] = ""
        _real_print(*args, **kwargs)

    builtins.print = _colored_print
    ai_system._v10_mood_colors_installed = True
    ai_system._v10_real_print = _real_print
    return True


def uninstall_mood_colors(ai_system) -> None:
    """Restore the original print (mostly for tests)."""
    import builtins
    if getattr(ai_system, "_v10_real_print", None):
        builtins.print = ai_system._v10_real_print
        ai_system._v10_mood_colors_installed = False


if __name__ == "__main__":
    # Diagnostic mode: print sample lines in each color
    print("MARIS mood color palette:")
    for emo, code in COLORS.items():
        print(f"  {code}{emo:14s} — sample response text in this color{RESET}")
    print("\nCurrent state:")
    emo = _detect_dominant_emotion()
    if emo:
        print(f"  Dominant emotion: {emo}")
        print(f"  {COLORS.get(emo, '')}Response would appear in this color{RESET}")
    else:
        print(f"  Neutral / no internal_state.json found")
