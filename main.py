"""
main.py — MARIS v6: Modular Adaptive Reasoning with Interactive Self-improvement
=================================================================================
  python main.py              # interactive mode
  python main.py --demo       # automated demo
"""

from llm_modules import (
    StrategyMemory, EmotionModule, ComplexityRouter, TaskTypeDetector,
    ReasoningModule, ReflectionModule, ImprovementModule, JudgeModule,
    UncertaintyDetector, ClarificationModule, MidLoopClarifier,
    DialogueMemory, ConsolidationEngine,
    Senate, DevelopmentTracker, HallucinationProbe, InnerMonologue,
    InsightDetector, ProactiveModule, AutonomousAction, InternalState,
)


class AI_System:
    def __init__(self):
        self.memory = StrategyMemory()
        self.emotion_module = EmotionModule()
        self.router = ComplexityRouter()
        self.task_detector = TaskTypeDetector()

        self.reasoning = ReasoningModule()
        self.reflector = ReflectionModule()
        self.improver = ImprovementModule()
        self.judge = JudgeModule()

        self.uncertainty = UncertaintyDetector()
        self.clarifier = ClarificationModule()
        self.mid_clarifier = MidLoopClarifier()
        self.dialogue = DialogueMemory()

        # v4 modules
        self.consolidation = ConsolidationEngine(min_strategies=3)

        # v5 modules — Senate, Development, Hallucination, InnerMonologue
        self.senate = Senate()
        self.dev_tracker = DevelopmentTracker()
        self.hallucination_probe = HallucinationProbe()
        self.monologue = InnerMonologue()
        self.insight_detector = InsightDetector()
        self.proactive = ProactiveModule()
        self.autonomous = AutonomousAction()
        self.inner_state = InternalState()

    def run(self, input_text: str, max_steps: int = 3, interactive: bool = True):
        print(f"\n{'='*60}")
        print(f"INPUT: {input_text}")
        print(f"{'='*60}")

        self.dialogue.add_user(input_text)

        # ── Step 1: Emotion ──
        emotion = self.emotion_module.analyze(input_text)
        print(f"\n  Emotion: {emotion['mood']} "
              f"(confidence={emotion['confidence']}, valence={emotion['valence']})")
        print(f"  Tone: {emotion['tone_instruction']}")

        # ── Step 1b: Development stage ──
        stage = self.dev_tracker.compute_stage(self.memory)
        print(f"  Stage: {stage['name']} ({stage['description'][:50]}...)")

        # ── MARIS's own emotional state (not the user's) ──
        self.inner_state.decay()  # emotions fade between turns
        maris_emotion, maris_intensity = self.inner_state.get_dominant_emotion()
        maris_color = self.inner_state.get_color()
        maris_reset = "\033[0m"
        if maris_emotion != "neutral":
            print(f"  {maris_color}MARIS feels: {maris_emotion} (intensity={maris_intensity}){maris_reset}")
        else:
            print(f"  MARIS feels: neutral")
        print(f"  Progress to next: {stage['progress_to_next']}")

        # ── Step 2: Task type detection (NEW) ──
        task_type = self.task_detector.detect(input_text, self.dialogue)
        print(f"  Task: {task_type['task_type']} "
              f"(tokens={task_type['base_tokens']}, scores={task_type['all_scores']})")

        # ── Step 3: Complexity routing (FIXED) ──
        complexity = self.router.classify(input_text, task_type, self.dialogue)
        steps = min(complexity["recommended_steps"], max_steps)
        print(f"  Complexity: {complexity['complexity']} "
              f"-> {steps} step(s) ({complexity['reason']})")

        # ── Step 4: Pre-answer clarification ──
        if interactive and steps > 1:
            unc = self.uncertainty.should_clarify(input_text, emotion, self.dialogue)
            print(f"\n  Uncertainty: {unc['reason']}")

            if unc["should_ask"]:
                clarif = self.clarifier.generate_questions(
                    input_text, emotion, self.dialogue, max_questions=unc["max_questions"],
                )
                questions = clarif.get("questions", [])
                conf = clarif.get("confidence_without_answers", 70)

                if questions and conf < 75:
                    print(f"  Confidence without clarification: {conf}%")
                    print(f"  Missing: {clarif.get('missing_dimensions', [])}")
                    print(f"\n  --- AI wants to ask you something first ---")

                    for i, q in enumerate(questions):
                        print(f"\n  Q{i+1}: {q}")
                        try:
                            answer = input("  Your answer (or 'skip'): ").strip()
                        except (EOFError, KeyboardInterrupt):
                            answer = "skip"
                        if answer.lower() != "skip":
                            self.dialogue.add_clarification(q, answer)
                            self.dialogue.add_user(answer)
                            print(f"  Noted.")
                        else:
                            print(f"  Skipped.")

                    print(f"\n  --- Proceeding with context ---\n")

        # ── Step 5: Strategy retrieval (now includes meta-strategies) ──
        strategies = self.memory.get_strategies(input_text)
        if strategies:
            print(f"\n  Retrieved {len(strategies)} strategies:")
            for s in strategies:
                meta_tag = " [META]" if s.get("is_meta") else ""
                print(f"    [{s['relevance']}]{meta_tag} {s['strategy'][:80]}...")
        else:
            print(f"\n  No relevant strategies in memory yet")

        # ── Step 6: INNER MONOLOGUE — the daemon deliberates ──
        print(f"\n  [Inner Monologue — deliberating...]")
        deliberation = self.monologue.deliberate(
            input_text, emotion, strategies, self.dialogue, task_type
        )
        monologue_context = self.monologue.format_for_reasoning(deliberation)

        instinct_flag = " (instinct CHANGED)" if deliberation.get("instinct_changed") else ""
        print(f"  Depth: {deliberation.get('deliberation_depth', '?')}, "
              f"Confidence: {deliberation.get('confidence', '?')}%{instinct_flag}")
        if deliberation.get("first_instinct"):
            print(f"  First instinct: {deliberation['first_instinct'][:100]}...")
        if deliberation.get("instinct_changed") and deliberation.get("challenge"):
            print(f"  Changed because: {deliberation['challenge'][:100]}...")
        if deliberation.get("final_position"):
            print(f"  Final position: {deliberation['final_position'][:120]}...")
        if deliberation.get("blind_spots"):
            print(f"  Blind spots: {deliberation['blind_spots'][:3]}")

        # ── Update MARIS's internal state from deliberation ──
        if deliberation.get("deliberation_depth") == "deep":
            self.inner_state.update("deep_deliberation")
        if deliberation.get("instinct_changed"):
            self.inner_state.update("instinct_changed")
        if deliberation.get("confidence", 50) < 40:
            self.inner_state.update("low_confidence")
        elif deliberation.get("confidence", 50) > 75:
            self.inner_state.update("high_confidence")

        # ── Autonomous actions based on MARIS's internal state ──
        # Use MARIS's OWN emotional state for autonomous actions, not the user's
        maris_state_for_action = {
            "mood": self.inner_state.get_dominant_emotion()[0],
            "valence": self.inner_state.get_dominant_emotion()[1],
        }
        auto_actions = self.autonomous.decide_action(deliberation, maris_state_for_action, stage)
        needs_color_reset = False
        for action in auto_actions.get("actions", []):
            self.autonomous.execute(action)
            if action.get("_reset_code"):
                needs_color_reset = True

        # ── Step 7: Reasoning (guided by inner monologue) ──
        context = {
            "emotion": emotion,
            "strategies": strategies,
            "dialogue": self.dialogue,
            "task_type": task_type,
            "monologue_context": monologue_context,
        }
        current = self.reasoning.run(input_text, context)
        print(f"\n-- Initial Response --\n{current}")
        self.dialogue.add_assistant(current)

        if steps <= 1:
            self._print_final(current)
            return current

        # ── Step 7: Improvement loop ──
        prev_score = 0
        stagnation_count = 0
        asked_midloop = False

        for i in range(steps - 1):
            print(f"\n-- Iteration {i+1} --")

            reflection = self.reflector.reflect(input_text, current, emotion)
            score = reflection.get("overall_score", 50)
            print(f"  Scores: acc={reflection.get('accuracy')}/10 "
                  f"rel={reflection.get('relevance')}/10 "
                  f"tone={reflection.get('tone_match')}/10 "
                  f"comp={reflection.get('completeness')}/10")
            print(f"  Overall: {score}/100")
            print(f"  Weaknesses: {reflection.get('weaknesses', [])}")

            # Mid-loop clarification
            if interactive and not asked_midloop:
                mid = self.mid_clarifier.needs_human_input(reflection)
                if mid["needs_human_input"] and mid["suggested_question"]:
                    print(f"\n  --- AI interrupt ---")
                    print(f"  {mid['suggested_question']}")
                    try:
                        answer = input("  Your answer (or 'skip'): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        answer = "skip"
                    if answer.lower() != "skip":
                        self.dialogue.add_clarification(mid['suggested_question'], answer)
                        self.dialogue.add_user(answer)
                        print(f"  Incorporating.")
                    asked_midloop = True

            # Convergence
            score_delta = score - prev_score
            if i > 0 and abs(score_delta) < 5:
                stagnation_count += 1
                if stagnation_count >= 1:
                    print(f"  Converged (delta={score_delta}) — stopping")
                    break
            else:
                stagnation_count = 0
            prev_score = score

            # Improve
            improved = self.improver.improve(
                input_text, current, reflection, emotion, self.dialogue, task_type,
            )
            print(f"\n  Improved ({len(improved)} chars):\n  {improved[:300]}{'...' if len(improved) > 300 else ''}")

            # Judge
            verdict = self.senate.evaluate(
                input_text, current, improved, emotion, task_type.get("task_type", "advice")
            )
            print(f"\n  Senate: {verdict['winner']} ({verdict['confidence']}%)")
            print(f"    A={verdict.get('scores_A', {})}")
            print(f"    B={verdict.get('scores_B', {})}")
            print(f"    Reason: {verdict.get('reason', '')}")

            if verdict["winner"] == "B" and verdict["confidence"] >= 40:
                current = improved
                self.dialogue.add_assistant(current)
                print(f"  Accepted")
                self.inner_state.update("improvement_accepted")

                self.memory.add({
                    "input": input_text,
                    "strategy": reflection.get("strategy", ""),
                    "score_delta": score_delta,
                    "mood": emotion["mood"],
                    "task_type": task_type["task_type"],
                    "had_clarification": len(self.dialogue.clarifications) > 0,
                })
            else:
                print(f"  Kept original")
                self.inner_state.update("improvement_rejected")

        # ── Auto-consolidation check ──
        if self.consolidation.should_consolidate(self.memory):
            print(f"\n  Consolidation triggered ({self.memory.strategy_count()} strategies)...")
            new_metas = self.consolidation.consolidate(self.memory)
            if new_metas:
                print(f"  Extracted {len(new_metas)} new meta-principles:")
                for m in new_metas:
                    print(f"    -> {m['principle']}")

                # ── Insight Detection (Eureka moments) ──
                insights = self.insight_detector.detect(self.memory)
                if insights:
                    cyan = "\033[96m"
                    bold = "\033[1m"
                    reset = "\033[0m"
                    print(f"\n  {cyan}{bold}*** EUREKA! ***{reset}")
                    print(f"  {cyan}Discovered {len(insights)} deeper principle(s):{reset}")
                    for ins in insights:
                        print(f"  {cyan}  Insight: {ins['principle']}{reset}")
                        print(f"  {cyan}  From: {ins['source_rules'][0][:60]}...{reset}")
                        print(f"  {cyan}  And:  {ins['source_rules'][1][:60]}...{reset}")
                        print(f"  {cyan}  Depth: {ins.get('depth', '?')}{reset}")
                    self.inner_state.update("eureka_moment")
            else:
                print(f"  No new meta-principles extracted this round")

        # ── Hallucination probe (before showing final output) ──
        probe_result = self.hallucination_probe.probe(input_text, current)
        if probe_result.get("should_flag"):
            print(f"\n  HALLUCINATION WARNING (risk={probe_result.get('hallucination_risk')})")
            if probe_result.get("uncertain_claims"):
                print(f"    Uncertain: {probe_result['uncertain_claims'][:3]}")
            if probe_result.get("assumptions_made"):
                print(f"    Assumptions: {probe_result['assumptions_made'][:3]}")
            self.inner_state.update("hallucination_found")
            self.dev_tracker.record_task({"hallucination_detected": True,
                                          "output_chars": len(current)})
        else:
            print(f"\n  Confidence: {probe_result.get('overall_confidence', '?')}% "
                  f"(risk={probe_result.get('hallucination_risk', '?')})")

        # Record task metrics
        any_accepted = any(
            "Accepted" in str(s) for s in self.memory.data[-3:]
        ) if self.memory.data else False
        has_meta = any(s.get("is_meta") for s in strategies) if strategies else False
        self.dev_tracker.record_task({
            "improvement_accepted": any_accepted,
            "clarification_asked": len(self.dialogue.clarifications) > 0,
            "meta_rule_applied": has_meta,
            "output_chars": len(current),
            "stage": stage.get("stage", 0) if 'stage' in dir() else 0,
        })

        self._print_final(current)
        return current

    def _print_final(self, output: str):
        maris_color = self.inner_state.get_color()
        reset = "\033[0m"
        print(f"\n{'='*60}")
        print(f"{maris_color}FINAL OUTPUT:\n{output}{reset}")
        print(f"{'='*60}")
        self.inner_state.snapshot()

    def interactive_loop(self):
        print("\n" + "="*60)
        print("  MARIS v6 — Modular Adaptive Reasoning")
        print("  with Interactive Self-improvement")
        print("  + Senate | Inner Monologue | Eureka | Autonomous")
        print("="*60)
        print(f"\n  Memory: {self.memory.strategy_count()} strategies, "
              f"{self.memory.meta_count()} meta-rules, "
              f"{self.insight_detector.insight_count()} insights")
        print(f"  Stage: {self.dev_tracker.compute_stage(self.memory)['name']}")
        emo, intensity = self.inner_state.get_dominant_emotion()
        if emo != "neutral":
            c = self.inner_state.get_color()
            print(f"  {c}Feeling: {emo} ({intensity})\033[0m")
        print("\nCommands:")
        print("  /memory       — strategies + meta-rules")
        print("  /history      — conversation so far")
        print("  /consolidate  — force knowledge extraction")
        print("  /insights     — Eureka moments (Tier 3)")
        print("  /stage        — development level")
        print("  /progress     — learning metrics")
        print("  /feelings     — MARIS's emotional state")
        print("  /stats        — system statistics")
        print("  /clear        — reset conversation")
        print("  quit          — exit")
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            if user_input == "/memory":
                self._show_memory()
                continue
            if user_input == "/history":
                self._show_history()
                continue
            if user_input == "/consolidate":
                self._do_consolidate()
                continue
            if user_input == "/clear":
                self.dialogue = DialogueMemory()
                print("Conversation cleared. Memory retained.\n")
                continue
            if user_input == "/stats":
                self._show_stats()
                continue
            if user_input == "/stage":
                self._show_stage()
                continue
            if user_input == "/insights":
                self._show_insights()
                continue
            if user_input == "/feelings":
                self._show_feelings()
                continue
            if user_input == "/progress":
                self._show_progress()
                continue

            self.run(user_input, interactive=True)
            print()

            # ── Proactive questioning: MARIS asks YOU ──
            stage_info = self.dev_tracker.compute_stage(self.memory)
            proactive = self.proactive.should_initiate(
                self.dialogue, stage_info, self.memory
            )
            if proactive.get("should_ask") and proactive.get("question"):
                cyan = "\033[96m"
                dim = "\033[2m"
                reset = "\033[0m"
                print(f"{cyan}  MARIS has a question for you:{reset}")
                print(f"  {proactive['question']}")
                print(f"{dim}  (Reason: {proactive.get('reason', '?')}){reset}")
                print(f"  {dim}(Answer naturally, or just continue with your own question){reset}")
                print()

    def _show_memory(self):
        print(f"\n  === Strategy Memory ===")
        if self.memory.meta_strategies:
            print(f"\n  Meta-rules ({len(self.memory.meta_strategies)}):")
            for i, m in enumerate(self.memory.meta_strategies):
                mood = m.get("mood_pattern", "all")
                sources = m.get("source_count", "?")
                conf = m.get("confidence", "?")
                print(f"  {i+1}. [{mood}, conf={conf}%, from {sources} exp] {m['principle']}")

        if self.memory.data:
            print(f"\n  Recent strategies (last 10 of {len(self.memory.data)}):")
            for i, entry in enumerate(self.memory.data[-10:]):
                mood = entry.get("mood", "?")
                task = entry.get("task_type", "?")
                strategy = entry.get("strategy", "?")[:60]
                cl = " [+clarif]" if entry.get("had_clarification") else ""
                print(f"  {i+1}. [{mood}/{task}]{cl} {strategy}...")
        else:
            print("\n  No strategies stored yet.")
        print()

    def _show_history(self):
        if not self.dialogue.turns:
            print("\n  No conversation history.\n")
            return
        print(f"\n  Conversation ({self.dialogue.turn_count()} turns):")
        for t in self.dialogue.turns[-12:]:
            role = "You" if t["role"] == "user" else "AI "
            print(f"  {role}: {t['content'][:120]}...")
        if self.dialogue.clarifications:
            print(f"\n  Clarifications: {len(self.dialogue.clarifications)}")
            for c in self.dialogue.clarifications:
                print(f"    Q: {c['question'][:60]}...")
                print(f"    A: {c['answer'][:60]}...")
        print()

    def _do_consolidate(self):
        count = self.memory.strategy_count()
        if count < 3:
            print(f"\n  Not enough strategies yet ({count}). Need at least 5.\n")
            return
        print(f"\n  Running consolidation on {count} strategies...")
        new_metas = self.consolidation.consolidate(self.memory)
        if new_metas:
            print(f"  Extracted {len(new_metas)} new meta-principles:")
            for m in new_metas:
                print(f"    -> {m['principle']}")
        else:
            print(f"  No new patterns found (may need more diverse experiences)")
        print()

    def _show_stats(self):
        print(f"\n  === MARIS Statistics ===")
        print(f"  Strategies stored: {self.memory.strategy_count()}")
        print(f"  Meta-rules learned: {self.memory.meta_count()}")
        print(f"  Conversation turns: {self.dialogue.turn_count()}")
        print(f"  Clarifications gathered: {len(self.dialogue.clarifications)}")
        print()


    def _show_feelings(self):
        color = self.inner_state.get_color() or ""
        reset = "\033[0m"
        print(f"\n  {color}=== MARIS Internal State ==={reset}")
        print(f"  {self.inner_state.get_state_summary()}")
        print(f"\n  All dimensions:")
        for dim, val in self.inner_state.state.items():
            bar_len = int(abs(val) * 20)
            direction = "+" if val >= 0 else "-"
            bar = direction * bar_len if bar_len > 0 else "."
            print(f"    {dim:14s} [{bar:>20s}] {val:+.2f}")
        print()

    def _show_insights(self):
        insights = self.insight_detector.get_insights()
        if not insights:
            print("\n  No Eureka moments yet. Keep learning!\n")
            return
        cyan = "\033[96m"
        reset = "\033[0m"
        print(f"\n  {cyan}=== Eureka Moments (Tier 3 Insights) ==={reset}")
        for i, ins in enumerate(insights):
            print(f"  {cyan}{i+1}. {ins['principle']}{reset}")
            print(f"     From: {ins['source_rules'][0][:70]}...")
            print(f"     And:  {ins['source_rules'][1][:70]}...")
            print(f"     Depth: {ins.get('depth', '?')}")
        print()

    def _show_stage(self):
        stage = self.dev_tracker.compute_stage(self.memory)
        print(f"\n  === Development Stage ===")
        print(f"  Current: Stage {stage['stage']} — {stage['name']}")
        print(f"  {stage['description']}")
        print(f"  Strategies: {stage['strategies']}")
        print(f"  Meta-rules: {stage['meta_rules']}")
        print(f"  Progress to next stage: {stage['progress_to_next']}")
        print()

    def _show_progress(self):
        summary = self.dev_tracker.summary()
        print(f"\n  === Progression Metrics ===")
        print(f"  Total tasks:        {summary['total_tasks']}")
        print(f"  Acceptance rate:    {summary['acceptance_rate']}%")
        print(f"  Hallucination rate: {summary['hallucination_rate']}%")
        print(f"  Clarification rate: {summary['clarification_rate']}%")
        print(f"  Meta-rule usage:    {summary['meta_rule_usage']}%")
        print(f"  Est. tokens used:   {summary['estimated_tokens']}")
        print(f"  Trend:              {summary['trend']}")
        print()


def demo():
    ai = AI_System()
    examples = [
        "hi there",
        "i had a really bad day and can't focus, help me be productive",
        "explain tradeoffs between microservices and monoliths for a startup with 3 engineers",
        "UGH this code keeps breaking!!! How do I fix a segfault in C?",
        "feeling down again and need to concentrate on a deadline",
    ]
    for ex in examples:
        ai.run(ex, interactive=False)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        ai = AI_System()
        ai.interactive_loop()
