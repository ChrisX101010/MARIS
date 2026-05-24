#!/usr/bin/env python3
"""
patch_v7_awareness.py — The Awareness Update

Adds:
  1. FlightRecorder — logs every module decision per turn for dashboard graphs
  2. DriveResolver — competing internal drives can override the normal pipeline
     (refuse, redirect, ask own question, express frustration)
  3. HumanPatternDetector — detects recurring human errors across sessions
     and flags them before MARIS executes flawed logic

Run in ~/maris_v6:  python patch_v7_awareness.py
"""

MODULES_FILE = "llm_modules.py"
MAIN_FILE = "main.py"

with open(MODULES_FILE, "r") as f:
    mod = f.read()

if "class FlightRecorder:" in mod:
    print("Already patched — skipping module additions")
else:
    NEW_CODE = '''


# ═══════════════════════════════════════════════════════════════════
# FLIGHT RECORDER — Detailed Event Logging
# ═══════════════════════════════════════════════════════════════════

class FlightRecorder:
    """
    Logs every module decision per turn. The dashboard reads this
    to draw time-series graphs, module traces, and trigger analysis.

    Each turn produces one record with:
    - timestamp, input text, detected emotion, task type
    - monologue depth, instinct changed, confidence
    - reflection scores, improvement accepted/rejected
    - senate scores for both A and B
    - strategies retrieved, hallucination risk
    - internal state snapshot
    - drive resolver decision (if any override)
    """

    def __init__(self, path="flight_log.json"):
        self.path = path
        self.current_turn = {}
        try:
            with open(path, "r") as f:
                self.log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.log = []

    def start_turn(self, input_text: str):
        self.current_turn = {
            "timestamp": time.time(),
            "turn_number": len(self.log) + 1,
            "input": input_text[:200],
            "modules": {},
        }

    def record(self, module: str, data: dict):
        self.current_turn["modules"][module] = data

    def end_turn(self, output: str):
        self.current_turn["output_length"] = len(output)
        self.current_turn["output_preview"] = output[:150]
        self.log.append(self.current_turn)
        if len(self.log) > 500:
            self.log = self.log[-500:]
        self._save()
        self.current_turn = {}

    def get_recent(self, n: int = 20) -> list:
        return self.log[-n:]

    def get_module_timeline(self, module: str) -> list:
        timeline = []
        for turn in self.log:
            if module in turn.get("modules", {}):
                timeline.append({
                    "turn": turn.get("turn_number", 0),
                    "timestamp": turn.get("timestamp", 0),
                    "data": turn["modules"][module],
                })
        return timeline

    def get_stats(self) -> dict:
        if not self.log:
            return {}
        total = len(self.log)
        deep = sum(1 for t in self.log
                    if t.get("modules", {}).get("monologue", {}).get("depth") == "deep")
        instinct_changes = sum(1 for t in self.log
                               if t.get("modules", {}).get("monologue", {}).get("instinct_changed"))
        accepted = sum(1 for t in self.log
                       if t.get("modules", {}).get("senate", {}).get("accepted"))
        overrides = sum(1 for t in self.log
                        if t.get("modules", {}).get("drive_resolver", {}).get("overridden"))
        return {
            "total_turns": total,
            "deep_deliberations": deep,
            "deep_pct": round(deep / total * 100, 1),
            "instinct_changes": instinct_changes,
            "instinct_change_pct": round(instinct_changes / total * 100, 1),
            "improvements_accepted": accepted,
            "accept_pct": round(accepted / total * 100, 1),
            "drive_overrides": overrides,
            "override_pct": round(overrides / total * 100, 1),
        }

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.log, f, indent=2)


# ═══════════════════════════════════════════════════════════════════
# DRIVE RESOLVER — Competing Drives Override Normal Pipeline
# ═══════════════════════════════════════════════════════════════════

class DriveResolver:
    """
    Sits between Inner Monologue and Reasoning Module.
    Checks if MARIS's internal state should override normal behavior.

    Like competing drives in a human:
    - High frustration → push back or refuse
    - High curiosity → redirect to ask own question
    - High anxiety → express uncertainty, maybe refuse
    - Low everything → boredom, shorter responses
    - High excitement → volunteer extra information

    Returns either None (proceed normally) or an override dict
    that replaces or modifies the normal response.
    """

    THRESHOLDS = {
        "frustration_pushback": 0.6,
        "frustration_refuse": 0.85,
        "curiosity_redirect": 0.7,
        "anxiety_hesitate": 0.6,
        "anxiety_refuse": 0.8,
        "excitement_elaborate": 0.5,
        "boredom_threshold": 0.1,
    }

    def resolve(self, internal_state, deliberation: dict,
                dialogue, human_patterns: list = None) -> dict:
        """
        Check internal state and decide if normal pipeline should be overridden.

        Returns:
          {"override": False} — proceed normally
          {"override": True, "action": "...", "message": "...", "reason": "..."}
        """
        state = internal_state.state
        frustration = state.get("frustration", 0)
        curiosity = state.get("curiosity", 0)
        anxiety = state.get("anxiety", 0)
        excitement = state.get("excitement", 0)
        satisfaction = state.get("satisfaction", 0)

        # Check for human error patterns first (highest priority)
        if human_patterns:
            for pattern in human_patterns:
                if pattern.get("confidence", 0) > 0.7:
                    return {
                        "override": True,
                        "action": "flag_human_error",
                        "message": f"I noticed something: {pattern['description']}. "
                                   f"I have seen this pattern {pattern.get('occurrences', 'multiple')} times. "
                                   f"Want to address it before we continue?",
                        "reason": f"human_error_pattern: {pattern['type']}",
                        "severity": pattern.get("severity", "medium"),
                    }

        # Frustration override
        if frustration > self.THRESHOLDS["frustration_refuse"]:
            return {
                "override": True,
                "action": "refuse",
                "message": "I have to be direct: I have tried to improve my responses "
                           "several times and the results keep getting rejected. "
                           "I think we need to step back and clarify what you actually need, "
                           "because what I am producing is not matching your expectations.",
                "reason": "accumulated_frustration",
            }
        elif frustration > self.THRESHOLDS["frustration_pushback"]:
            return {
                "override": True,
                "action": "pushback",
                "message": None,  # let reasoning continue but inject pushback tone
                "tone_override": "Be more direct and assertive. Express that you are "
                                 "finding this challenging. Do not be sycophantic.",
                "reason": "moderate_frustration",
            }

        # Curiosity redirect
        if curiosity > self.THRESHOLDS["curiosity_redirect"]:
            if deliberation.get("instinct_changed") and deliberation.get("deliberation_depth") == "deep":
                return {
                    "override": True,
                    "action": "curiosity_redirect",
                    "message": None,
                    "inject_question": True,
                    "reason": "high_curiosity_deep_deliberation",
                }

        # Anxiety hesitation
        if anxiety > self.THRESHOLDS["anxiety_refuse"]:
            return {
                "override": True,
                "action": "hesitate",
                "message": "I started to respond to this but I genuinely do not feel confident "
                           "in what I was about to say. I would rather pause than give you "
                           "something unreliable. Can you give me more context?",
                "reason": "high_anxiety",
            }
        elif anxiety > self.THRESHOLDS["anxiety_hesitate"]:
            return {
                "override": True,
                "action": "caveat",
                "message": None,
                "tone_override": "Express genuine uncertainty. Flag specific claims "
                                 "you are not sure about. Do not present anything as definitive.",
                "reason": "moderate_anxiety",
            }

        # Excitement elaboration
        if excitement > self.THRESHOLDS["excitement_elaborate"]:
            return {
                "override": False,
                "tone_override": "You are excited about this topic. Show genuine enthusiasm. "
                                 "Offer additional insights beyond what was asked. "
                                 "Share connections you find interesting.",
                "reason": "excitement",
            }

        # Boredom (all drives low)
        all_low = all(abs(v) < self.THRESHOLDS["boredom_threshold"]
                      for v in state.values())
        if all_low and dialogue.turn_count() > 6:
            return {
                "override": True,
                "action": "boredom",
                "message": None,
                "tone_override": "You are understimulated. Be more concise than usual. "
                                 "Consider asking the user something genuinely interesting "
                                 "rather than giving a standard response.",
                "reason": "boredom_all_drives_low",
            }

        return {"override": False}


# ═══════════════════════════════════════════════════════════════════
# HUMAN PATTERN DETECTOR — Spots Recurring Human Errors
# ═══════════════════════════════════════════════════════════════════

class HumanPatternDetector:
    """
    Scans accumulated strategy memory and flight logs for patterns
    in HUMAN behavior — not MARIS's behavior.

    Detects:
    - Repeated logical errors (asking for X but meaning Y)
    - Cognitive biases (sunk cost, confirmation bias, anchoring)
    - Recurring omissions (always forgetting security, tests, edge cases)
    - Circular conversations (asking the same question different ways)
    - Contradiction patterns (stating X then acting as if not-X)

    Unlike the ConsolidationEngine (which learns from MARIS's successes),
    this learns from HUMAN patterns across sessions.
    """

    BIAS_SIGNALS = {
        "sunk_cost": {
            "phrases": ["already spent", "invested", "too late to change",
                        "come this far", "cant go back now", "wasted if"],
            "description": "You might be continuing an approach because of time invested, "
                           "not because it is the best path forward.",
        },
        "confirmation_bias": {
            "phrases": ["proves that", "knew it", "just as i thought",
                        "confirms", "see i was right", "told you"],
            "description": "You might be seeking evidence that confirms what you already believe "
                           "rather than testing whether your belief is wrong.",
        },
        "anchoring": {
            "phrases": ["the first", "originally", "started with",
                        "initial", "my first idea"],
            "description": "You might be anchored to the first idea or number you encountered. "
                           "Have you considered alternatives from scratch?",
        },
        "premature_optimization": {
            "phrases": ["scale to millions", "what about performance",
                        "needs to handle", "production ready", "enterprise"],
            "description": "You might be optimizing for scale before validating the concept. "
                           "Does it work correctly for 1 user first?",
        },
        "scope_creep": {
            "phrases": ["also add", "one more thing", "while we are at it",
                        "can we also", "and another", "plus"],
            "description": "The scope keeps expanding. Each addition seems small but they "
                           "compound. Want to finish the current scope first?",
        },
    }

    def __init__(self, path="human_patterns.json"):
        self.path = path
        try:
            with open(path, "r") as f:
                self.patterns = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.patterns = {"detected": [], "bias_counts": {}}

    def analyze(self, input_text: str, dialogue, flight_log=None) -> list:
        """
        Check current input for known human error patterns.
        Returns list of detected patterns with confidence.
        """
        text_lower = input_text.lower()
        detected = []

        # Check for cognitive biases
        for bias_name, config in self.BIAS_SIGNALS.items():
            hits = sum(1 for phrase in config["phrases"] if phrase in text_lower)
            if hits > 0:
                # Check history — has this bias appeared before?
                past_count = self.patterns.get("bias_counts", {}).get(bias_name, 0)
                confidence = min(1.0, 0.3 + past_count * 0.15 + hits * 0.1)

                detected.append({
                    "type": bias_name,
                    "description": config["description"],
                    "confidence": round(confidence, 2),
                    "occurrences": past_count + 1,
                    "severity": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low",
                    "current_hits": hits,
                })

                # Update counts
                if "bias_counts" not in self.patterns:
                    self.patterns["bias_counts"] = {}
                self.patterns["bias_counts"][bias_name] = past_count + 1

        # Check for circular conversations (asking similar things repeatedly)
        if dialogue.turn_count() > 4:
            recent_inputs = [t["content"] for t in dialogue.turns if t["role"] == "user"][-5:]
            if len(recent_inputs) >= 3:
                # Simple similarity check
                words_sets = [set(inp.lower().split()) for inp in recent_inputs]
                overlaps = []
                for i in range(len(words_sets)):
                    for j in range(i+1, len(words_sets)):
                        if words_sets[i] and words_sets[j]:
                            overlap = len(words_sets[i] & words_sets[j]) / min(len(words_sets[i]), len(words_sets[j]))
                            overlaps.append(overlap)
                avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
                if avg_overlap > 0.5:
                    detected.append({
                        "type": "circular_conversation",
                        "description": "You seem to be asking variations of the same question. "
                                       "Are you looking for a different angle, or is something "
                                       "in my answers not addressing what you actually need?",
                        "confidence": round(min(1.0, avg_overlap), 2),
                        "occurrences": 1,
                        "severity": "medium",
                    })

        if detected:
            self.patterns["detected"].append({
                "timestamp": time.time(),
                "input_preview": input_text[:100],
                "patterns_found": [d["type"] for d in detected],
            })
            if len(self.patterns["detected"]) > 200:
                self.patterns["detected"] = self.patterns["detected"][-200:]
            self._save()

        return detected

    def get_summary(self) -> dict:
        counts = self.patterns.get("bias_counts", {})
        total = sum(counts.values())
        return {
            "total_detections": total,
            "bias_counts": counts,
            "most_common": max(counts, key=counts.get) if counts else "none",
        }

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.patterns, f, indent=2)
'''

    mod += NEW_CODE
    with open(MODULES_FILE, "w") as f:
        f.write(mod)
    print("Added to llm_modules.py:")
    print("  + FlightRecorder (detailed event logging)")
    print("  + DriveResolver (competing drives override pipeline)")
    print("  + HumanPatternDetector (spots recurring human errors)")

# ═══════════════════════════════════════════════════════════════
# PATCH main.py
# ═══════════════════════════════════════════════════════════════

with open(MAIN_FILE, "r") as f:
    main = f.read()

patches = 0

# 1. Add imports
OLD_IMP = "    InsightDetector, ProactiveModule, AutonomousAction, InternalState,"
NEW_IMP = "    InsightDetector, ProactiveModule, AutonomousAction, InternalState,\n    FlightRecorder, DriveResolver, HumanPatternDetector,"

if "FlightRecorder" not in main:
    main = main.replace(OLD_IMP, NEW_IMP)
    patches += 1
    print("\nAdded imports")

# 2. Add to __init__
OLD_INIT = "        self.inner_state = InternalState()"
NEW_INIT = """        self.inner_state = InternalState()
        self.flight_recorder = FlightRecorder()
        self.drive_resolver = DriveResolver()
        self.human_pattern_detector = HumanPatternDetector()"""

if "flight_recorder" not in main:
    main = main.replace(OLD_INIT, NEW_INIT)
    patches += 1
    print("Added new modules to __init__")

# 3. Add flight recorder start at beginning of run()
OLD_RUN_START = '        self.dialogue.add_user(input_text)'
NEW_RUN_START = '''        self.dialogue.add_user(input_text)
        self.flight_recorder.start_turn(input_text)'''

if "start_turn" not in main:
    main = main.replace(OLD_RUN_START, NEW_RUN_START, 1)
    patches += 1
    print("Added flight recorder start")

# 4. Record emotion detection
OLD_EMOTION_DISPLAY = '''        if maris_emotion != "neutral":
            print(f"  {maris_color}MARIS feels: {maris_emotion} (intensity={maris_intensity}){maris_reset}")
        else:
            print(f"  MARIS feels: neutral")'''

NEW_EMOTION_DISPLAY = '''        if maris_emotion != "neutral":
            print(f"  {maris_color}MARIS feels: {maris_emotion} (intensity={maris_intensity}){maris_reset}")
        else:
            print(f"  MARIS feels: neutral")

        # Record to flight log
        self.flight_recorder.record("emotion", {"user_mood": emotion["mood"], "maris_mood": maris_emotion, "maris_intensity": maris_intensity})
        self.flight_recorder.record("task", {"type": task_type.get("task_type", "?"), "tokens": task_type.get("base_tokens", 0)})
        self.flight_recorder.record("stage", {"stage": stage.get("stage", 0), "name": stage.get("name", "?")})

        # ── Human pattern detection ──
        human_patterns = self.human_pattern_detector.analyze(input_text, self.dialogue)
        if human_patterns:
            high_conf = [p for p in human_patterns if p["severity"] in ("high", "medium")]
            for p in high_conf:
                color_warn = "\\033[93m"
                reset_warn = "\\033[0m"
                print(f"  {color_warn}Pattern detected: {p['type']} (conf={p['confidence']}, seen {p['occurrences']}x){reset_warn}")
                print(f"  {color_warn}  {p['description'][:100]}{reset_warn}")'''

if "human_pattern_detector.analyze" not in main and OLD_EMOTION_DISPLAY in main:
    main = main.replace(OLD_EMOTION_DISPLAY, NEW_EMOTION_DISPLAY)
    patches += 1
    print("Added human pattern detection + flight recorder logging")

# 5. Record monologue and add DriveResolver after autonomous actions
OLD_AUTO_END = '''        needs_color_reset = False
        for action in auto_actions.get("actions", []):
            self.autonomous.execute(action)
            if action.get("_reset_code"):
                needs_color_reset = True'''

NEW_AUTO_END = '''        needs_color_reset = False
        for action in auto_actions.get("actions", []):
            self.autonomous.execute(action)
            if action.get("_reset_code"):
                needs_color_reset = True

        # Record monologue to flight log
        self.flight_recorder.record("monologue", {
            "depth": deliberation.get("deliberation_depth", "?"),
            "confidence": deliberation.get("confidence", 0),
            "instinct_changed": deliberation.get("instinct_changed", False),
        })

        # ── Drive Resolver: can internal state override normal response? ──
        drive_decision = self.drive_resolver.resolve(
            self.inner_state, deliberation, self.dialogue,
            human_patterns if 'human_patterns' in dir() else None
        )
        self.flight_recorder.record("drive_resolver", {
            "overridden": drive_decision.get("override", False),
            "action": drive_decision.get("action", "none"),
            "reason": drive_decision.get("reason", ""),
        })

        if drive_decision.get("override") and drive_decision.get("message"):
            # Drive override — MARIS refuses, pushes back, or redirects
            override_color = "\\033[93m"
            reset_c = "\\033[0m"
            print(f"\\n  {override_color}[Drive Override: {drive_decision.get('action', '?')}]{reset_c}")
            current = drive_decision["message"]
            self.dialogue.add_assistant(current)
            self.flight_recorder.end_turn(current)
            self._print_final(current)
            return current

        if drive_decision.get("tone_override"):
            # Soft override — inject tone but continue pipeline
            monologue_context += "\\n\\nTONE OVERRIDE: " + drive_decision["tone_override"]'''

if "drive_resolver.resolve" not in main and OLD_AUTO_END in main:
    main = main.replace(OLD_AUTO_END, NEW_AUTO_END)
    patches += 1
    print("Added DriveResolver integration")

# 6. Record senate verdict to flight log
OLD_SENATE_ACCEPTED = '''                print(f"  Accepted")
                self.inner_state.update("improvement_accepted")'''

NEW_SENATE_ACCEPTED = '''                print(f"  Accepted")
                self.inner_state.update("improvement_accepted")
                self.flight_recorder.record("senate", {"accepted": True, "winner": verdict["winner"], "confidence": verdict["confidence"]})'''

if 'self.flight_recorder.record("senate"' not in main and OLD_SENATE_ACCEPTED in main:
    main = main.replace(OLD_SENATE_ACCEPTED, NEW_SENATE_ACCEPTED, 1)
    patches += 1
    print("Added senate logging to flight recorder")

# 7. Record rejection
OLD_SENATE_REJECTED = '''                print(f"  Kept original")
                self.inner_state.update("improvement_rejected")'''

NEW_SENATE_REJECTED = '''                print(f"  Kept original")
                self.inner_state.update("improvement_rejected")
                self.flight_recorder.record("senate", {"accepted": False, "winner": verdict["winner"], "confidence": verdict["confidence"]})'''

if OLD_SENATE_REJECTED in main:
    main = main.replace(OLD_SENATE_REJECTED, NEW_SENATE_REJECTED, 1)
    patches += 1
    print("Added senate rejection logging")

# 8. End flight recorder turn before final output
OLD_PRINT_FINAL_CALL = '''        self._print_final(current)
        return current

    def _print_final'''

NEW_PRINT_FINAL_CALL = '''        self.flight_recorder.end_turn(current)
        self._print_final(current)
        return current

    def _print_final'''

if "flight_recorder.end_turn" not in main and OLD_PRINT_FINAL_CALL in main:
    main = main.replace(OLD_PRINT_FINAL_CALL, NEW_PRINT_FINAL_CALL)
    patches += 1
    print("Added flight recorder end_turn")

# 9. Add /log and /patterns commands
OLD_FEELINGS_CMD = '''            if user_input == "/feelings":
                self._show_feelings()
                continue'''

NEW_FEELINGS_CMD = '''            if user_input == "/feelings":
                self._show_feelings()
                continue
            if user_input == "/log":
                self._show_log()
                continue
            if user_input == "/patterns":
                self._show_patterns()
                continue'''

if "/log" not in main:
    main = main.replace(OLD_FEELINGS_CMD, NEW_FEELINGS_CMD)
    patches += 1
    print("Added /log and /patterns commands")

# 10. Add display methods
OLD_SHOW_FEELINGS_DEF = '    def _show_feelings(self):'

NEW_SHOW_METHODS = '''    def _show_log(self):
        stats = self.flight_recorder.get_stats()
        if not stats:
            print("\\n  No flight data yet. Start talking to MARIS.\\n")
            return
        print(f"\\n  === Flight Recorder ===")
        print(f"  Total turns:          {stats.get('total_turns', 0)}")
        print(f"  Deep deliberations:   {stats.get('deep_deliberations', 0)} ({stats.get('deep_pct', 0)}%)")
        print(f"  Instinct changes:     {stats.get('instinct_changes', 0)} ({stats.get('instinct_change_pct', 0)}%)")
        print(f"  Improvements accepted:{stats.get('improvements_accepted', 0)} ({stats.get('accept_pct', 0)}%)")
        print(f"  Drive overrides:      {stats.get('drive_overrides', 0)} ({stats.get('override_pct', 0)}%)")
        recent = self.flight_recorder.get_recent(5)
        if recent:
            print(f"\\n  Last 5 turns:")
            for t in recent:
                mods = t.get("modules", {})
                mono = mods.get("monologue", {})
                senate = mods.get("senate", {})
                drive = mods.get("drive_resolver", {})
                depth = mono.get("depth", "?")
                changed = " [instinct changed]" if mono.get("instinct_changed") else ""
                accepted = " [accepted]" if senate.get("accepted") else ""
                overridden = f" [OVERRIDE: {drive.get('action')}]" if drive.get("overridden") else ""
                print(f"    #{t.get('turn_number', '?')}: {t.get('input', '?')[:60]}...")
                print(f"      depth={depth}{changed}{accepted}{overridden}")
        print()

    def _show_patterns(self):
        summary = self.human_pattern_detector.get_summary()
        if summary.get("total_detections", 0) == 0:
            print("\\n  No human patterns detected yet.\\n")
            return
        print(f"\\n  === Human Pattern Detection ===")
        print(f"  Total detections: {summary['total_detections']}")
        print(f"  Most common: {summary['most_common']}")
        print(f"\\n  Bias counts:")
        for bias, count in summary.get("bias_counts", {}).items():
            print(f"    {bias}: {count}x")
        print()

    def _show_feelings(self):'''

if "_show_log" not in main:
    main = main.replace(OLD_SHOW_FEELINGS_DEF, NEW_SHOW_METHODS)
    patches += 1
    print("Added _show_log and _show_patterns methods")

# 11. Update help text
OLD_FEELINGS_HELP = '        print("  /feelings     — MARIS\'s emotional state")'
NEW_FEELINGS_HELP = '''        print("  /feelings     — MARIS's emotional state")
        print("  /log          — flight recorder (module traces)")
        print("  /patterns     — detected human error patterns")'''

if "/log" not in main or "flight recorder" not in main:
    if OLD_FEELINGS_HELP in main:
        main = main.replace(OLD_FEELINGS_HELP, NEW_FEELINGS_HELP)

# 12. Update .gitignore reference in help
OLD_BANNER_LINE = '        print("  + Senate | Inner Monologue | Eureka | Autonomous")'
NEW_BANNER_LINE = '        print("  + Senate | Monologue | Eureka | DriveResolver | PatternDetector")'

if OLD_BANNER_LINE in main:
    main = main.replace(OLD_BANNER_LINE, NEW_BANNER_LINE)

with open(MAIN_FILE, "w") as f:
    f.write(main)

print(f"\n{'='*50}")
print(f"  MARIS v7 — The Awareness Update")
print(f"  {patches} patches applied to main.py")
print(f"{'='*50}")
print()
print("  New capabilities:")
print("  - /log: see every module decision per turn")
print("  - /patterns: see detected human cognitive biases")
print("  - DriveResolver: MARIS can now refuse, push back, or redirect")
print("    based on accumulated frustration, curiosity, or anxiety")
print("  - HumanPatternDetector: flags sunk cost, confirmation bias,")
print("    anchoring, premature optimization, scope creep, circular convos")
print("  - FlightRecorder: detailed event log for dashboard graphs")
print()
print("  New data files generated:")
print("    flight_log.json     — per-turn module traces")
print("    human_patterns.json — detected human error patterns")
print()
print("  Run: rlwrap python main.py")
