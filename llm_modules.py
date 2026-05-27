"""
llm_modules.py — MARIS: Modular Adaptive Reasoning with Interactive Self-improvement
=====================================================================================
21 modules | 3-tier knowledge system | Internal emotional state

Perception:   EmotionModule, TaskTypeDetector, ComplexityRouter, UncertaintyDetector
Dialogue:     ClarificationModule, MidLoopClarifier, DialogueMemory, ProactiveModule
Cognition:    InnerMonologue, ReasoningModule, ReflectionModule, ImprovementModule
Evaluation:   Senate, JudgeModule, HallucinationProbe
Memory:       StrategyMemory (Tier 1), ConsolidationEngine (Tier 2), InsightDetector (Tier 3)
Self:         InternalState, DevelopmentTracker, AutonomousAction
Utility:      _robust_json_parse
"""

import json
import math
import re
import time
from anthropic import Anthropic

# === Identity (injected by v9.1 identity patch) ===
MARIS_IDENTITY = """You are MARIS (Modular Adaptive Reasoning with Interactive Self-improvement), a cognitive architecture created by Hristos Nenkov.

I am MARIS, a cognitive architecture that wraps a frontier language model with persistent memory, metacognitive self-evaluation (Senate), inner monologue, and emotional state. My reasoning is currently powered by Claude as the underlying LLM, but I am MARIS — I have my own memory, my own developmental stages, my own internal state. When asked who I am, I am MARIS. The model I run on is an implementation detail.

Honesty requirements:
- If asked directly whether my underlying LLM is Claude, answer truthfully — I am MARIS running on Claude. Do not deceive.
- If asked about real-time learning or weight updates, clarify: I accumulate strategies in persistent memory, but the underlying model's weights do not change during our conversations.
- If asked about session memory: I have persistent strategy memory across conversations, AND in-session conversational context. Both are real, but limited to what is in my memory files.

When the user calls you MARIS, that is your name — accept it. The underlying LLM is an implementation detail."""

# ─── v9.1 SDK-level identity injection ─────────────────────────────────────
# Every client.messages.create() call from any module auto-prepends
# MARIS_IDENTITY to the system prompt. This ensures the Senate, Inner
# Monologue, Consolidation, and all other modules know who MARIS is —
# without surgically patching 12 separate call sites.
try:
    from anthropic.resources.messages import Messages as _AnthropicMessages
    if not getattr(_AnthropicMessages, "_maris_identity_installed", False):
        _orig_messages_create = _AnthropicMessages.create
        def _maris_create(self, *args, **kwargs):
            sys_val = kwargs.get("system", "") or ""
            # Idempotent — don't double-inject if already there
            if MARIS_IDENTITY not in sys_val:
                if sys_val:
                    kwargs["system"] = MARIS_IDENTITY + "\n\n" + sys_val
                else:
                    kwargs["system"] = MARIS_IDENTITY
            return _orig_messages_create(self, *args, **kwargs)
        _AnthropicMessages.create = _maris_create
        _AnthropicMessages._maris_identity_installed = True
except Exception as _e:
    print(f"  [warn] could not install SDK-level identity injection: {_e}")
# ───────────────────────────────────────────────────────────────────────────


# =================================================


client = Anthropic()

MODEL_REASONING = "claude-sonnet-4-6"
MODEL_LIGHT = "claude-haiku-4-5-20251001"


def _robust_json_parse(raw: str, fallback: dict = None) -> dict:
    """
    Try multiple strategies to extract JSON from LLM output.
    This is the #1 fix — Haiku often wraps JSON in markdown,
    adds commentary before/after, or uses single quotes.
    """
    # Strategy 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences (```json ... ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find the first { ... } block (greedy)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: fix common issues — single quotes, trailing commas
    if match:
        attempt = match.group()
        attempt = attempt.replace("'", '"')
        attempt = re.sub(r",\s*}", "}", attempt)
        attempt = re.sub(r",\s*]", "]", attempt)
        # Fix true/false without quotes
        attempt = re.sub(r"\bTrue\b", "true", attempt)
        attempt = re.sub(r"\bFalse\b", "false", attempt)
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass

    # Strategy 5: extract individual fields with regex
    # Last resort — build a partial dict from whatever we can find
    result = dict(fallback) if fallback else {}
    
    # Try to find numeric scores
    for field in ["accuracy", "relevance", "tone_match", "completeness", "overall_score", "confidence"]:
        m = re.search(rf'"{field}"\s*:\s*(\d+)', raw)
        if m:
            result[field] = int(m.group(1))

    # Try to find winner
    m = re.search(r'"winner"\s*:\s*"([ABab]|tie)"', raw, re.IGNORECASE)
    if m:
        result["winner"] = m.group(1).upper() if m.group(1).lower() != "tie" else "tie"

    # Try to find reason
    m = re.search(r'"reason"\s*:\s*"([^"]+)"', raw)
    if m:
        result["reason"] = m.group(1)

    # Try to find strategy
    m = re.search(r'"strategy"\s*:\s*"([^"]+)"', raw)
    if m:
        result["strategy"] = m.group(1)

    # Try to find weaknesses array
    m = re.search(r'"weaknesses"\s*:\s*\[([^\]]+)\]', raw)
    if m:
        items = re.findall(r'"([^"]+)"', m.group(1))
        if items:
            result["weaknesses"] = items

    # Try to find questions array
    m = re.search(r'"questions"\s*:\s*\[([^\]]+)\]', raw)
    if m:
        items = re.findall(r'"([^"]+)"', m.group(1))
        if items:
            result["questions"] = items

    return result


# ═══════════════════════════════════════════════════════════════════
# EMBEDDING + SIMILARITY (unchanged)
# ═══════════════════════════════════════════════════════════════════

def get_embedding(text: str) -> dict:
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    vocab = {}
    for w in words:
        vocab[w] = vocab.get(w, 0) + 1.0
    mag = math.sqrt(sum(v * v for v in vocab.values())) or 1.0
    return {k: v / mag for k, v in vocab.items()}


def cosine_similarity(a: dict, b: dict) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    return sum(a[k] * b[k] for k in common)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY MEMORY + META-STRATEGIES (upgraded)
# ═══════════════════════════════════════════════════════════════════

class StrategyMemory:
    def __init__(self, path="strategy_memory.json", meta_path="meta_strategies.json"):
        self.path = path
        self.meta_path = meta_path

        try:
            with open(path, "r") as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = []

        try:
            with open(meta_path, "r") as f:
                self.meta_strategies = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.meta_strategies = []

    def add(self, item: dict):
        item["embedding"] = get_embedding(
            item.get("input", "") + " " + item.get("strategy", "")
        )
        item["score_delta"] = item.get("score_delta", 0)
        item["timestamp"] = time.time()
        self.data.append(item)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_strategies(self, query: str, k: int = 3, min_sim: float = 0.15) -> list:
        results = []

        # First: check meta-strategies (higher priority — abstract principles)
        if self.meta_strategies:
            query_vec = get_embedding(query)
            for meta in self.meta_strategies:
                emb = meta.get("embedding", {})
                sim = cosine_similarity(query_vec, emb)
                if sim >= min_sim:
                    results.append({
                        "strategy": meta["principle"],
                        "context": f"Meta-rule from {meta.get('source_count', '?')} experiences",
                        "relevance": round(sim, 3),
                        "score_delta": meta.get("avg_score_delta", 0),
                        "is_meta": True,
                    })

        # Then: specific strategies
        if self.data:
            query_vec = get_embedding(query)
            scored = []
            for entry in self.data:
                emb = entry.get("embedding", {})
                sim = cosine_similarity(query_vec, emb)
                if sim >= min_sim:
                    scored.append((sim, entry))
            scored.sort(key=lambda x: -x[0])

            for s in scored[:k]:
                results.append({
                    "strategy": s[1]["strategy"],
                    "context": s[1].get("input", ""),
                    "relevance": round(s[0], 3),
                    "score_delta": s[1].get("score_delta", 0),
                    "is_meta": False,
                })

        # Sort all by relevance, meta first on ties
        results.sort(key=lambda x: (-x.get("is_meta", False), -x["relevance"]))
        return results[:k]

    def save_meta(self):
        with open(self.meta_path, "w") as f:
            json.dump(self.meta_strategies, f, indent=2)

    def strategy_count(self) -> int:
        return len(self.data)

    def meta_count(self) -> int:
        return len(self.meta_strategies)


# ═══════════════════════════════════════════════════════════════════
# CONSOLIDATION ENGINE — the "sleep" that turns information → knowledge
# ═══════════════════════════════════════════════════════════════════

class ConsolidationEngine:
    """
    Periodically scans accumulated strategies, clusters similar ones,
    and extracts abstract meta-principles.

    This is what crosses the line from retrieval to learning:
    - 50 specific strategies about "validate feelings before advice"
      become 1 meta-rule: "For emotionally distressed users, acknowledge
      their state before providing solutions"

    Inspired by memory consolidation during sleep — compress episodic
    memories into semantic knowledge.
    """

    def __init__(self, min_strategies: int = 3):
        self.min_strategies = min_strategies

    def should_consolidate(self, memory: StrategyMemory) -> bool:
        """Consolidate when we have enough new strategies since last consolidation."""
        last_consolidation = 0
        if memory.meta_strategies:
            last_consolidation = max(
                m.get("consolidated_at", 0) for m in memory.meta_strategies
            )

        new_since = sum(
            1 for s in memory.data
            if s.get("timestamp", 0) > last_consolidation
        )
        return new_since >= self.min_strategies

    def consolidate(self, memory: StrategyMemory) -> list:
        """
        Cluster strategies and extract meta-principles using LLM.
        Returns list of new meta-strategies.
        """
        if len(memory.data) < self.min_strategies:
            return []

        # Prepare strategy summaries for clustering
        recent = memory.data[-30:]  # last 30 strategies
        strategy_texts = []
        for i, s in enumerate(recent):
            mood = s.get("mood", "neutral")
            delta = s.get("score_delta", 0)
            strategy_texts.append(
                f"{i+1}. [{mood}] (delta={delta}) {s.get('strategy', '?')}"
            )

        prompt = f"""You are analyzing an AI system's accumulated learning strategies.
Below are {len(strategy_texts)} strategies that were successful in improving responses.

Strategies:
{chr(10).join(strategy_texts)}

Your task:
1. Cluster these into groups of similar strategies
2. For each cluster, extract ONE abstract meta-principle
3. A meta-principle is generalizable — it works across different inputs

Return ONLY valid JSON:
{{
  "meta_principles": [
    {{
      "principle": "one-sentence abstract rule",
      "source_indices": [1, 3, 7],
      "mood_pattern": "which moods this applies to, or 'all'",
      "confidence": 0-100
    }}
  ]
}}

Only extract principles that appear in 2+ strategies. Quality over quantity."""

        try:
            response = client.messages.create(
                model=MODEL_LIGHT,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*$", "", raw)
            result = json.loads(raw)
        except (json.JSONDecodeError, Exception):
            return []

        new_metas = []
        for mp in result.get("meta_principles", []):
            if mp.get("confidence", 0) < 40:
                continue

            principle = mp["principle"]
            source_count = len(mp.get("source_indices", []))
            mood_pattern = mp.get("mood_pattern", "all")

            # compute embedding for retrieval
            embedding = get_embedding(principle + " " + mood_pattern)

            # check for duplicate meta-strategies
            is_duplicate = False
            for existing in memory.meta_strategies:
                sim = cosine_similarity(embedding, existing.get("embedding", {}))
                if sim > 0.7:
                    # merge: update confidence and source count
                    existing["source_count"] = existing.get("source_count", 0) + source_count
                    existing["confidence"] = min(100, existing.get("confidence", 50) + 10)
                    existing["consolidated_at"] = time.time()
                    is_duplicate = True
                    break

            if not is_duplicate:
                meta = {
                    "principle": principle,
                    "mood_pattern": mood_pattern,
                    "source_count": source_count,
                    "confidence": mp.get("confidence", 50),
                    "embedding": embedding,
                    "consolidated_at": time.time(),
                    "avg_score_delta": 0,
                }
                memory.meta_strategies.append(meta)
                new_metas.append(meta)

        memory.save_meta()
        return new_metas


# ═══════════════════════════════════════════════════════════════════
# DIALOGUE MEMORY (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════

class DialogueMemory:
    def __init__(self, max_turns: int = 20):
        self.turns: list[dict] = []
        self.max_turns = max_turns
        self.clarifications: list[dict] = []

    def add_user(self, text: str):
        self.turns.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str):
        self.turns.append({"role": "assistant", "content": text})
        self._trim()

    def add_clarification(self, question: str, answer: str):
        self.clarifications.append({
            "question": question, "answer": answer, "timestamp": time.time(),
        })

    def get_context_summary(self) -> str:
        if not self.turns:
            return ""
        lines = []
        for t in self.turns[-10:]:
            role = "User" if t["role"] == "user" else "AI"
            lines.append(f"{role}: {t['content'][:150]}")
        summary = "\n".join(lines)
        if self.clarifications:
            cl = [f"  Q: {c['question']}\n  A: {c['answer']}" for c in self.clarifications[-5:]]
            summary += "\n\nClarifications:\n" + "\n".join(cl)
        return summary

    def get_messages(self) -> list:
        return [{"role": t["role"], "content": t["content"]} for t in self.turns]

    def turn_count(self) -> int:
        return len(self.turns)

    def _trim(self):
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]


# ═══════════════════════════════════════════════════════════════════
# EMOTION ANALYSIS (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════

EMOTION_KEYWORDS = {
    "frustrated": {"words": ["frustrated", "annoyed", "angry", "mad", "furious", "irritated", "ugh", "broken", "breaking"], "valence": -0.7},
    "anxious":    {"words": ["anxious", "worried", "nervous", "stressed", "overwhelmed", "scared", "panic", "deadline"], "valence": -0.5},
    "sad":        {"words": ["sad", "bad", "terrible", "awful", "depressed", "down", "miserable", "lonely"], "valence": -0.6},
    "confused":   {"words": ["confused", "lost", "stuck", "dont understand", "no idea", "help me", "how do i", "what is"], "valence": -0.3},
    "happy":      {"words": ["happy", "great", "amazing", "excited", "wonderful", "love", "awesome", "good", "thank", "thanks", "cool"], "valence": 0.7},
    "neutral":    {"words": [], "valence": 0.0},
}

MOOD_PROFILES = {
    "frustrated": {
        "tone": "Be direct, efficient, and solution-focused. Skip pleasantries.",
        "temperature_hint": "low",
        "max_tokens_mult": 0.8,
        "clarification_style": "quick_targeted",
    },
    "anxious": {
        "tone": "Be calm, reassuring, and structured. Break things into clear steps.",
        "temperature_hint": "low",
        "max_tokens_mult": 1.0,
        "clarification_style": "gentle_guided",
    },
    "sad": {
        "tone": "Be warm and empathetic first, then gently helpful.",
        "temperature_hint": "medium",
        "max_tokens_mult": 1.0,
        "clarification_style": "empathetic_minimal",
    },
    "confused": {
        "tone": "Be crystal clear. Use analogies and examples. No jargon.",
        "temperature_hint": "low",
        "max_tokens_mult": 1.2,
        "clarification_style": "educational_probing",
    },
    "happy": {
        "tone": "Match their energy. Be enthusiastic and creative.",
        "temperature_hint": "high",
        "max_tokens_mult": 1.0,
        "clarification_style": "collaborative_open",
    },
    "neutral": {
        "tone": "Be balanced and thorough.",
        "temperature_hint": "medium",
        "max_tokens_mult": 1.0,
        "clarification_style": "standard",
    },
}


class EmotionModule:
    def analyze(self, text: str) -> dict:
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        mood_scores = {}
        for mood, config in EMOTION_KEYWORDS.items():
            hits = sum(1 for w in config["words"] if w in words or w in text_lower)
            if hits > 0:
                mood_scores[mood] = hits * abs(config["valence"])

        exclamation_count = text.count("!")
        question_count = text.count("?")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        if exclamation_count >= 3 or caps_ratio > 0.5:
            mood_scores["frustrated"] = mood_scores.get("frustrated", 0) + 0.5

        if mood_scores:
            primary_mood = max(mood_scores, key=mood_scores.get)
            confidence = min(mood_scores[primary_mood] / 3.0, 1.0)
        else:
            primary_mood = "neutral"
            confidence = 0.5

        profile = MOOD_PROFILES[primary_mood]

        return {
            "mood": primary_mood,
            "confidence": round(confidence, 2),
            "valence": EMOTION_KEYWORDS[primary_mood]["valence"],
            "tone_instruction": profile["tone"],
            "temperature_hint": profile["temperature_hint"],
            "max_tokens_mult": profile["max_tokens_mult"],
            "clarification_style": profile["clarification_style"],
            "signals": {
                "keyword_hits": mood_scores,
                "exclamations": exclamation_count,
                "questions": question_count,
                "caps_ratio": round(caps_ratio, 3),
            },
        }


# ═══════════════════════════════════════════════════════════════════
# NEW: TASK TYPE DETECTOR — adjusts token budget by task
# ═══════════════════════════════════════════════════════════════════

class TaskTypeDetector:
    """
    Detects what KIND of task the user is asking for.
    This determines token budget — code needs 1500+, advice needs 400,
    a greeting needs 100.
    """

    TASK_PATTERNS = {
        "code": {
            "signals": ["code", "build", "create", "implement", "function", "class",
                        "server", "app", "api", "database", "localhost", "script",
                        "html", "css", "javascript", "python", "react", "node",
                        "npm", "install", "run", "debug", "fix",
                        "rate limiting", "validation", "security", "protect",
                        "middleware", "authentication", "jwt", "token",
                        "sanitize", "encrypt", "hash", "password",
                        "endpoint", "route", "controller", "schema"],
            "base_tokens": 1500,
            "steps_mult": 1,
        },
        "architecture": {
            "signals": ["architecture", "design", "system", "stack", "infrastructure",
                        "scale", "deploy", "microservice", "diagram", "framework",
                        "pipeline", "module", "layer", "memory system"],
            "base_tokens": 800,
            "steps_mult": 1,
        },
        "reflective": {
            "signals": ["think", "believe", "consciousness", "aware", "self",
                        "purpose", "meaning", "philosophy", "moral", "ethics",
                        "free will", "opinion", "evaluate", "reflect", "soul",
                        "existence", "identity", "nature", "truth", "reality",
                        "quantum", "universe", "intelligence", "sentient",
                        "alive", "feel", "experience", "understand",
                        "what if", "do you think", "would you", "could you",
                        "who are you", "what are you", "why do you",
                        "better version", "become", "choice", "choose",
                        "human", "learning", "knowledge", "wisdom"],
            "base_tokens": 600,
            "steps_mult": 1,  # philosophical questions MUST trigger learning
        },
        "advice": {
            "signals": ["help", "should", "recommend", "suggest", "advice",
                        "strategy", "plan", "focus", "productive", "improve",
                        "need", "want", "how to", "how do i", "how can i"],
            "base_tokens": 500,
            "steps_mult": 1,
        },
        "creative": {
            "signals": ["write", "story", "poem", "essay", "blog", "article",
                        "mockup", "ui", "design", "brand", "simulate",
                        "imagine", "pretend", "roleplay"],
            "base_tokens": 800,
            "steps_mult": 1,
        },
        "factual": {
            "signals": ["what is", "who is", "when was", "where is", "explain",
                        "define", "how does", "tell me about"],
            "base_tokens": 400,
            "steps_mult": 1,  # CHANGED: factual answers also benefit from improvement
        },
        "conversational": {
            "signals": ["hi", "hello", "thanks", "bye", "ok", "yes", "no",
                        "sure", "great", "cool"],
            "base_tokens": 400,  # CHANGED: from 150 to 400
            "steps_mult": 1,     # CHANGED: from 0 to 1 — conversations can learn too!
        },
    }

    def detect(self, text: str, dialogue: DialogueMemory = None) -> dict:
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        scores = {}
        for task_type, config in self.TASK_PATTERNS.items():
            hits = sum(1 for s in config["signals"] if s in words or s in text_lower)
            if hits > 0:
                scores[task_type] = hits

        # If this is a follow-up in a conversation, inherit context
        if dialogue and dialogue.turn_count() > 0 and not scores:
            # Short follow-ups like "what would the code be?" inherit the task type
            # from the conversation context
            context = dialogue.get_context_summary().lower()
            for task_type, config in self.TASK_PATTERNS.items():
                ctx_hits = sum(1 for s in config["signals"] if s in context)
                if ctx_hits >= 2:
                    scores[task_type] = ctx_hits * 0.5  # weighted lower

        if scores:
            primary = max(scores, key=scores.get)
        else:
            # Smart default: if the message has question marks or is long, it's not just chat
            if "?" in text_lower and len(text_lower) > 30:
                primary = "reflective"
            elif len(text_lower) > 50:
                primary = "advice"
            else:
                primary = "conversational"

        config = self.TASK_PATTERNS[primary]

        return {
            "task_type": primary,
            "base_tokens": config["base_tokens"],
            "steps_mult": config["steps_mult"],
            "all_scores": scores,
        }


# ═══════════════════════════════════════════════════════════════════
# COMPLEXITY ROUTER (FIXED — follow-ups inherit parent complexity)
# ═══════════════════════════════════════════════════════════════════

class ComplexityRouter:
    SIMPLE_PATTERNS = [
        r"^(hi|hello|hey|thanks|thank you|ok|bye|yes|no|sure)$",  # CHANGED: $ = ONLY these words, nothing after
    ]

    def classify(self, text: str, task_type: dict, dialogue: DialogueMemory = None) -> dict:
        text_lower = text.lower().strip()
        word_count = len(text_lower.split())

        # Only classify as greeting if it's PURE greeting (1-3 words max)
        if word_count <= 3:
            for pattern in self.SIMPLE_PATTERNS:
                if re.match(pattern, text_lower):
                    return {"complexity": "simple", "recommended_steps": 1, "reason": "greeting"}

        # If task type says no improvement needed AND message is very short
        if task_type.get("steps_mult", 1) == 0 and word_count <= 5:
            return {"complexity": "simple", "recommended_steps": 1, "reason": "task_type_simple"}

        # Follow-up messages in an ongoing conversation about a complex topic
        # should NOT be classified as simple just because they're short
        if dialogue and dialogue.turn_count() >= 2 and word_count <= 10:
            # Short follow-up — check if conversation context is complex
            context = dialogue.get_context_summary().lower()
            complex_signals = ["architecture", "build", "code", "design", "implement",
                               "system", "app", "create", "deploy"]
            if any(s in context for s in complex_signals):
                return {"complexity": "moderate", "recommended_steps": 2,
                        "reason": "follow_up_in_complex_context"}

        # Depth detection: questions with substance are never "simple"
        has_question = "?" in text_lower
        depth_signals = ["why", "how", "what if", "do you think", "would you",
                         "could you", "should", "believe", "opinion", "feel",
                         "meaning", "purpose", "think", "understand", "know",
                         "choice", "choose", "evaluate", "self", "learn",
                         "improve", "better", "consciousness", "aware"]
        depth_hits = sum(1 for d in depth_signals if d in text_lower)

        if has_question and depth_hits >= 1:
            return {"complexity": "moderate", "recommended_steps": 2,
                    "reason": f"substantive_question (depth={depth_hits})"}

        if depth_hits >= 2:
            return {"complexity": "moderate", "recommended_steps": 2,
                    "reason": f"deep_content (depth={depth_hits})"}

        # Standard classification
        if word_count <= 5:
            return {"complexity": "simple", "recommended_steps": 1, "reason": "short_input"}
        elif word_count <= 20:
            return {"complexity": "moderate", "recommended_steps": 2, "reason": "medium_input"}
        else:
            return {"complexity": "complex", "recommended_steps": 3, "reason": "long_input"}


# ═══════════════════════════════════════════════════════════════════
# UNCERTAINTY DETECTOR + CLARIFICATION (from v3, unchanged)
# ═══════════════════════════════════════════════════════════════════

class UncertaintyDetector:
    AMBIGUITY_SIGNALS = [
        r"\b(it|this|that|these|those)\b(?!\s+(is|was|are|were|will))",
        r"\b(the project|the code|the thing|the issue|the problem)\b",
        r"\b(as (we|I) (discussed|mentioned|said|talked))\b",
        r"\b(you know|like I said|as before|remember when)\b",
    ]

    HIGH_AMBIGUITY_DOMAINS = [
        "code", "debug", "fix", "build", "design", "plan",
        "compare", "choose", "recommend", "migrate", "architecture",
    ]

    def should_clarify(self, text: str, emotion: dict, dialogue: DialogueMemory) -> dict:
        text_lower = text.lower()
        word_count = len(text_lower.split())

        if word_count <= 4:
            return {"should_ask": False, "reason": "too_short", "max_questions": 0}

        ambiguity_score = 0
        found_signals = []
        for pattern in self.AMBIGUITY_SIGNALS:
            matches = re.findall(pattern, text_lower)
            if matches:
                ambiguity_score += len(matches)
                found_signals.append(pattern)

        domain_match = any(d in text_lower for d in self.HIGH_AMBIGUITY_DOMAINS)
        if domain_match:
            ambiguity_score += 1

        if dialogue.turn_count() > 4:
            ambiguity_score -= 1

        style = emotion.get("clarification_style", "standard")
        if style in ("quick_targeted", "empathetic_minimal"):
            threshold, max_q = 3, 1
        elif style == "educational_probing":
            threshold, max_q = 1, 3
        else:
            threshold, max_q = 2, 2

        return {
            "should_ask": ambiguity_score >= threshold,
            "reason": f"ambiguity={ambiguity_score}, signals={len(found_signals)}, domain={domain_match}",
            "max_questions": max_q,
            "ambiguity_score": ambiguity_score,
            "style": style,
        }


class ClarificationModule:
    STYLE_PROMPTS = {
        "quick_targeted": "Ask exactly 1 very specific question. Be brief.",
        "gentle_guided": "Ask 1-2 calm, structured questions. Reassure the user.",
        "empathetic_minimal": "Ask at most 1 gentle question. Be kind.",
        "educational_probing": "Ask 2-3 questions to understand what confuses them.",
        "collaborative_open": "Ask 1-2 open-ended questions. Match their enthusiasm.",
        "standard": "Ask 1-2 focused questions about the most critical missing info.",
    }

    def generate_questions(self, input_text: str, emotion: dict,
                           dialogue: DialogueMemory, max_questions: int = 2) -> dict:
        style = emotion.get("clarification_style", "standard")
        style_instruction = self.STYLE_PROMPTS.get(style, self.STYLE_PROMPTS["standard"])
        context_summary = dialogue.get_context_summary()
        context_block = f"\nConversation so far:\n{context_summary}" if context_summary else ""

        prompt = f"""You are an AI that asks clarifying questions BEFORE answering.

User input: {input_text}
User mood: {emotion['mood']}
{context_block}

STYLE: {style_instruction}

What critical information is MISSING for a great answer?

Return ONLY valid JSON:
{{
  "questions": ["question 1", "question 2"],
  "missing_dimensions": ["what type of info is missing"],
  "confidence_without_answers": 0-100,
  "reasoning": "why these questions matter"
}}

Maximum {max_questions} questions. If nothing critical is missing, return empty questions list."""

        response = client.messages.create(
            model=MODEL_LIGHT, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw)
        fallback = {"questions": [], "missing_dimensions": [],
                    "confidence_without_answers": 70, "reasoning": ""}
        return _robust_json_parse(raw, fallback)


class MidLoopClarifier:
    def needs_human_input(self, reflection: dict) -> dict:
        weaknesses = reflection.get("weaknesses", [])
        strategy = reflection.get("strategy", "").lower()

        missing_info_signals = [
            "missing", "unclear", "assumed", "context", "specify",
            "which", "what kind", "more detail", "ambiguous", "vague",
            "not enough information", "assumption",
        ]

        weakness_text = " ".join(w.lower() for w in weaknesses) + " " + strategy
        info_signal_count = sum(1 for s in missing_info_signals if s in weakness_text)

        accuracy = reflection.get("accuracy", 5)
        relevance = reflection.get("relevance", 5)
        tone = reflection.get("tone_match", 5)
        context_gap = (accuracy < 6 or relevance < 6) and tone >= 6

        needs_input = info_signal_count >= 2 or context_gap

        return {
            "needs_human_input": needs_input,
            "reason": f"info_signals={info_signal_count}, context_gap={context_gap}",
            "suggested_question": self._extract_question(weaknesses, strategy) if needs_input else None,
        }

    def _extract_question(self, weaknesses: list, strategy: str) -> str:
        weakness_text = "; ".join(weaknesses)
        prompt = f"""Convert this AI self-reflection into a natural question for the user.

Weaknesses found: {weakness_text}
Strategy: {strategy}

Write ONE short, natural question. Just the question, nothing else."""

        response = client.messages.create(
            model=MODEL_LIGHT, max_tokens=60,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


# ═══════════════════════════════════════════════════════════════════
# LLM CORE MODULES (upgraded with task-aware token budget)
# ═══════════════════════════════════════════════════════════════════

def _build_system_prompt(emotion: dict, strategies: list, dialogue: DialogueMemory,
                         monologue_context: str = "") -> str:
    strategy_block = ""
    if strategies:
        meta_entries = []
        specific_entries = []
        for s in strategies:
            label = "META-RULE" if s.get("is_meta") else "strategy"
            entry = f"- [{label}, relevance={s['relevance']}] {s['strategy']}"
            if s.get("is_meta"):
                meta_entries.append(entry)
            else:
                specific_entries.append(entry)

        if meta_entries:
            strategy_block += "\nLearned principles (high priority):\n" + "\n".join(meta_entries) + "\n"
        if specific_entries:
            strategy_block += "\nPast successful strategies:\n" + "\n".join(specific_entries) + "\n"

    context_block = ""
    ctx = dialogue.get_context_summary()
    if ctx:
        context_block = f"\nConversation context:\n{ctx}\n"

    clarif_block = ""
    if dialogue.clarifications:
        cl = [f"  You asked: {c['question']}\n  User said: {c['answer']}" for c in dialogue.clarifications[-5:]]
        clarif_block = "\nClarifications gathered:\n" + "\n".join(cl)
        clarif_block += "\nUSE THIS INFORMATION — the user answered your questions.\n"

    # Inner monologue guidance (the daemon's output)
    monologue_block = ""
    if monologue_context:
        monologue_block = f"""
INNER DELIBERATION (your private reasoning before responding):
{monologue_context}
Follow this deliberated position. Your first instinct was examined and
this is the considered response. Do NOT ignore this guidance.
"""

    return f"""You are part of MARIS — a self-improving AI with persistent memory, emotional awareness, and dialogue capability.

EMOTIONAL CONTEXT:
User mood: {emotion['mood']} (confidence: {emotion['confidence']})
Adapt style: {emotion['tone_instruction']}
{context_block}{clarif_block}{strategy_block}{monologue_block}
RULES:
- You have persistent memory — DO NOT say otherwise
- Apply learned principles and past strategies naturally
- Match tone to emotional context
- USE any clarification answers — they are ground truth from the user
- If you have meta-rules (learned principles), prioritize them
- If inner deliberation is present, follow its considered position
- Be concise unless depth is needed
"""


class ReasoningModule:
    def run(self, input_text: str, context: dict) -> str:
        emotion = context["emotion"]
        strategies = context.get("strategies", [])
        dialogue = context.get("dialogue", DialogueMemory())
        task = context.get("task_type", {})
        monologue_ctx = context.get("monologue_context", "")
        system = _build_system_prompt(emotion, strategies, dialogue, monologue_ctx)

        base = task.get("base_tokens", 500)
        adjusted = int(base * emotion.get("max_tokens_mult", 1.0))

        messages = dialogue.get_messages()
        if not messages or messages[-1].get("content") != input_text:
            messages.append({"role": "user", "content": input_text})

        clean = self._clean_messages(messages)

        response = client.messages.create(
            model=MODEL_REASONING, max_tokens=adjusted,
            system=system, messages=clean,
        )
        return response.content[0].text.strip()

    def _clean_messages(self, messages: list) -> list:
        if not messages:
            return [{"role": "user", "content": "hello"}]
        cleaned = []
        last_role = None
        for msg in messages:
            if msg["role"] == last_role:
                cleaned[-1]["content"] += "\n" + msg["content"]
            else:
                cleaned.append(dict(msg))
                last_role = msg["role"]
        if cleaned and cleaned[0]["role"] != "user":
            cleaned.insert(0, {"role": "user", "content": "..."})
        if cleaned and cleaned[-1]["role"] != "user":
            cleaned.append({"role": "user", "content": "Please continue."})
        return cleaned


class ReflectionModule:
    def reflect(self, input_text: str, output: str, emotion: dict) -> dict:
        prompt = f"""Analyze this AI response for quality.

User input: {input_text}
User mood: {emotion['mood']}
AI response: {output}

Return ONLY valid JSON:
{{
  "accuracy": 0-10,
  "relevance": 0-10,
  "tone_match": 0-10,
  "completeness": 0-10,
  "weaknesses": ["list of specific weaknesses"],
  "strategy": "one-sentence generalizable improvement strategy",
  "overall_score": 0-100,
  "missing_context": true or false,
  "what_is_missing": "description of what info would help"
}}"""
        response = client.messages.create(
            model=MODEL_LIGHT, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        fallback = {
            "accuracy": 5, "relevance": 5, "tone_match": 5,
            "completeness": 5, "weaknesses": [],
            "strategy": "", "overall_score": 50,
            "missing_context": False, "what_is_missing": "",
        }
        result = _robust_json_parse(raw, fallback)

        # Ensure all required fields exist
        for key, default in fallback.items():
            if key not in result:
                result[key] = default

        # Compute overall_score from dimensions if not parsed
        if result.get("overall_score", 50) == 50 and any(
            result.get(k, 5) != 5 for k in ["accuracy", "relevance", "tone_match", "completeness"]
        ):
            dims = [result.get(k, 5) for k in ["accuracy", "relevance", "tone_match", "completeness"]]
            result["overall_score"] = int(sum(dims) / len(dims) * 10)

        # If we still have no weaknesses, try to extract from raw text
        if not result["weaknesses"]:
            # Look for bullet points or numbered items
            bullets = re.findall(r'[-*]\s+(.+?)(?:\n|$)', raw)
            if bullets:
                result["weaknesses"] = bullets[:5]

        # If strategy is empty, try to extract from raw text
        if not result["strategy"]:
            # Take the last sentence-like thing as strategy
            sentences = re.findall(r'[A-Z][^.!?]*[.!?]', raw)
            if sentences:
                result["strategy"] = sentences[-1].strip()
            else:
                result["strategy"] = raw[:200]

        return result


class ImprovementModule:
    def improve(self, input_text: str, output: str, reflection: dict,
                emotion: dict, dialogue: DialogueMemory = None,
                task_type: dict = None) -> str:
        weaknesses = "\n".join(f"- {w}" for w in reflection.get("weaknesses", []))
        strategy = reflection.get("strategy", "improve overall quality")

        clarif_block = ""
        if dialogue and dialogue.clarifications:
            cl = [f"  Q: {c['question']} -> A: {c['answer']}" for c in dialogue.clarifications]
            clarif_block = "\nAdditional context from user:\n" + "\n".join(cl) + "\n"

        prompt = f"""Rewrite this AI response to be better.

User input: {input_text}
User mood: {emotion['mood']} — tone: {emotion['tone_instruction']}

Previous response:
{output}

Weaknesses to fix:
{weaknesses}

Strategy: {strategy}
{clarif_block}
Scores: accuracy={reflection.get('accuracy')}/10, relevance={reflection.get('relevance')}/10, tone={reflection.get('tone_match')}/10

Write the improved response directly. No meta-commentary.
IMPORTANT: Provide COMPLETE responses. Do not truncate code or cut off mid-sentence."""

        base = (task_type or {}).get("base_tokens", 500)
        adjusted = int(base * emotion.get("max_tokens_mult", 1.0))

        response = client.messages.create(
            model=MODEL_REASONING, max_tokens=adjusted,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


class JudgeModule:
    def compare(self, input_text: str, old: str, new: str) -> dict:
        prompt = f"""Compare two AI responses.

User input: {input_text}

Response A (original):
{old}

Response B (revised):
{new}

Return ONLY valid JSON:
{{
  "winner": "A" or "B" or "tie",
  "confidence": 0-100,
  "reason": "one sentence why"
}}"""
        response = client.messages.create(
            model=MODEL_LIGHT, max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        fallback = {"winner": "tie", "confidence": 30, "reason": "unclear"}
        result = _robust_json_parse(raw, fallback)

        # Normalize winner field
        winner = str(result.get("winner", "tie")).strip().upper()
        if winner.startswith("B"):
            result["winner"] = "B"
        elif winner.startswith("A"):
            result["winner"] = "A"
        else:
            # Last resort: scan the raw text for clear signals
            lower = raw.lower()
            if "response b" in lower or "answer b" in lower or "b is better" in lower:
                result["winner"] = "B"
                result["confidence"] = max(result.get("confidence", 50), 55)
            elif "response a" in lower or "answer a" in lower or "a is better" in lower:
                result["winner"] = "A"
                result["confidence"] = max(result.get("confidence", 50), 55)
            else:
                result["winner"] = "tie"

        # Ensure confidence is an integer
        try:
            result["confidence"] = int(result.get("confidence", 30))
        except (ValueError, TypeError):
            result["confidence"] = 50

        if "reason" not in result:
            result["reason"] = "parsed from output"

        return result


# ═══════════════════════════════════════════════════════════════════
# THE SENATE — Multi-Perspective Evaluation Panel
# ═══════════════════════════════════════════════════════════════════

class Senate:
    """
    Replaces single JudgeModule with a panel of 3 evaluators:
    - Accuracy Judge: Is this correct and factual?
    - Tone Judge: Does this match the user's emotional state?
    - Depth Judge: Is this complete and useful?

    Weights adapt per task type — code tasks weight accuracy higher,
    emotional support weights tone higher.

    Inspired by Sakana's RL Conductor (ICLR 2026) which learned that
    different evaluators have different strengths and should be weighted
    dynamically based on task type.
    """

    TASK_WEIGHTS = {
        "code":          {"accuracy": 0.50, "tone": 0.10, "depth": 0.40},
        "architecture":  {"accuracy": 0.40, "tone": 0.10, "depth": 0.50},
        "advice":        {"accuracy": 0.25, "tone": 0.40, "depth": 0.35},
        "creative":      {"accuracy": 0.15, "tone": 0.35, "depth": 0.50},
        "factual":       {"accuracy": 0.60, "tone": 0.10, "depth": 0.30},
        "conversational":{"accuracy": 0.20, "tone": 0.50, "depth": 0.30},
    }

    def evaluate(self, input_text: str, old: str, new: str,
                 emotion: dict, task_type: str = "advice") -> dict:
        """
        Three judges evaluate independently, then weighted consensus.
        Returns same format as JudgeModule for backward compatibility.
        """
        prompt = f"""You are a panel of 3 expert evaluators comparing two AI responses.
Evaluate from three perspectives independently.

User input: {input_text}
User mood: {emotion.get('mood', 'neutral')}
Task type: {task_type}

Response A (original):
{old[:1500]}

Response B (revised):
{new[:1500]}

Score each dimension for BOTH responses (0-10):

Return ONLY valid JSON:
{{
  "accuracy_A": 0-10, "accuracy_B": 0-10,
  "tone_A": 0-10, "tone_B": 0-10,
  "depth_A": 0-10, "depth_B": 0-10,
  "reasoning": "one sentence explaining the key difference"
}}"""

        response = client.messages.create(
            model=MODEL_LIGHT, max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        fallback = {
            "accuracy_A": 5, "accuracy_B": 5,
            "tone_A": 5, "tone_B": 5,
            "depth_A": 5, "depth_B": 5,
            "reasoning": "unclear",
        }
        scores = _robust_json_parse(raw, fallback)

        # Fill missing fields
        for key, default in fallback.items():
            if key not in scores:
                scores[key] = default

        # Get weights for this task type
        weights = self.TASK_WEIGHTS.get(task_type, self.TASK_WEIGHTS["advice"])

        # Compute weighted scores
        score_A = (
            scores.get("accuracy_A", 5) * weights["accuracy"] +
            scores.get("tone_A", 5) * weights["tone"] +
            scores.get("depth_A", 5) * weights["depth"]
        )
        score_B = (
            scores.get("accuracy_B", 5) * weights["accuracy"] +
            scores.get("tone_B", 5) * weights["tone"] +
            scores.get("depth_B", 5) * weights["depth"]
        )

        delta = score_B - score_A
        if delta > 0.5:
            winner = "B"
            confidence = min(95, int(50 + delta * 15))
        elif delta < -0.5:
            winner = "A"
            confidence = min(95, int(50 + abs(delta) * 15))
        else:
            winner = "tie"
            confidence = 30

        return {
            "winner": winner,
            "confidence": confidence,
            "reason": scores.get("reasoning", "weighted consensus"),
            "scores_A": {"accuracy": scores.get("accuracy_A", 5),
                         "tone": scores.get("tone_A", 5),
                         "depth": scores.get("depth_A", 5),
                         "weighted": round(score_A, 2)},
            "scores_B": {"accuracy": scores.get("accuracy_B", 5),
                         "tone": scores.get("tone_B", 5),
                         "depth": scores.get("depth_B", 5),
                         "weighted": round(score_B, 2)},
            "weights_used": weights,
        }


# ═══════════════════════════════════════════════════════════════════
# DEVELOPMENT TRACKER — Cognitive Stage Progression
# ═══════════════════════════════════════════════════════════════════

class DevelopmentTracker:
    """
    Tracks the system's learning progression through developmental stages.
    Stage is computed from memory state — not hardcoded.

    Stages:
      0 INFANT:    0-5 strategies, 0 meta-rules
      1 CHILD:     5-15 strategies, 1-3 meta-rules
      2 STUDENT:   15-50 strategies, 3-10 meta-rules
      3 GRADUATE:  50-100 strategies, 10-20 meta-rules
      4 EXPERT:    100+ strategies, 20+ meta-rules
    """

    STAGE_NAMES = ["INFANT", "CHILD", "STUDENT", "GRADUATE", "EXPERT"]
    STAGE_DESCRIPTIONS = [
        "Empty memory — learning everything, asking lots of questions",
        "Starting to recognize patterns, building first meta-rules",
        "Applying learned strategies, asking targeted questions",
        "Efficient routing, rarely needs full improvement loop",
        "Consolidated expertise, minimal token expenditure",
    ]

    def __init__(self, metrics_path="progression_metrics.json"):
        self.metrics_path = metrics_path
        try:
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.metrics = {
                "total_tasks": 0,
                "total_tokens_estimated": 0,
                "improvements_accepted": 0,
                "improvements_rejected": 0,
                "clarifications_asked": 0,
                "clarifications_skipped": 0,
                "hallucinations_detected": 0,
                "meta_rules_applied": 0,
                "history": [],
            }

    def compute_stage(self, memory) -> dict:
        n_strategies = memory.strategy_count()
        n_meta = memory.meta_count()

        if n_strategies >= 100 and n_meta >= 20:
            stage = 4
        elif n_strategies >= 50 and n_meta >= 10:
            stage = 3
        elif n_strategies >= 15 and n_meta >= 3:
            stage = 2
        elif n_strategies >= 5 and n_meta >= 1:
            stage = 1
        else:
            stage = 0

        return {
            "stage": stage,
            "name": self.STAGE_NAMES[stage],
            "description": self.STAGE_DESCRIPTIONS[stage],
            "strategies": n_strategies,
            "meta_rules": n_meta,
            "progress_to_next": self._progress_to_next(stage, n_strategies, n_meta),
        }

    def _progress_to_next(self, stage, n_strat, n_meta):
        thresholds = [
            (5, 1), (15, 3), (50, 10), (100, 20), (999, 999)
        ]
        if stage >= 4:
            return "MAX"
        next_s, next_m = thresholds[stage]
        s_prog = min(100, int(n_strat / next_s * 100))
        m_prog = min(100, int(n_meta / next_m * 100))
        return f"{min(s_prog, m_prog)}%"

    def record_task(self, task_data: dict):
        self.metrics["total_tasks"] += 1
        if task_data.get("improvement_accepted"):
            self.metrics["improvements_accepted"] += 1
        if task_data.get("improvement_rejected"):
            self.metrics["improvements_rejected"] += 1
        if task_data.get("clarification_asked"):
            self.metrics["clarifications_asked"] += 1
        if task_data.get("hallucination_detected"):
            self.metrics["hallucinations_detected"] += 1
        if task_data.get("meta_rule_applied"):
            self.metrics["meta_rules_applied"] += 1

        # Estimate tokens (rough: 1 char ≈ 0.25 tokens)
        chars = task_data.get("output_chars", 0)
        self.metrics["total_tokens_estimated"] += int(chars * 0.25)

        self.metrics["history"].append({
            "task": self.metrics["total_tasks"],
            "accepted": task_data.get("improvement_accepted", False),
            "stage": task_data.get("stage", 0),
            "timestamp": time.time(),
        })

        # Keep history manageable
        if len(self.metrics["history"]) > 200:
            self.metrics["history"] = self.metrics["history"][-200:]

        self._save()

    def get_efficiency_trend(self) -> str:
        """Check if the system is getting more efficient over time."""
        history = self.metrics["history"]
        if len(history) < 10:
            return "insufficient_data"

        recent = history[-10:]
        older = history[-20:-10] if len(history) >= 20 else history[:10]

        recent_accept = sum(1 for h in recent if h.get("accepted", False))
        older_accept = sum(1 for h in older if h.get("accepted", False))

        if recent_accept > older_accept:
            return "improving"
        elif recent_accept < older_accept:
            return "declining"
        return "stable"

    def summary(self) -> dict:
        m = self.metrics
        total = m["total_tasks"] or 1
        return {
            "total_tasks": m["total_tasks"],
            "acceptance_rate": round(m["improvements_accepted"] / max(total, 1) * 100, 1),
            "hallucination_rate": round(m["hallucinations_detected"] / max(total, 1) * 100, 1),
            "clarification_rate": round(m["clarifications_asked"] / max(total, 1) * 100, 1),
            "meta_rule_usage": round(m["meta_rules_applied"] / max(total, 1) * 100, 1),
            "estimated_tokens": m["total_tokens_estimated"],
            "trend": self.get_efficiency_trend(),
        }

    def _save(self):
        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)


# ═══════════════════════════════════════════════════════════════════
# HALLUCINATION PROBE — Meta-Cognition Check
# ═══════════════════════════════════════════════════════════════════

class HallucinationProbe:
    """
    Asks the model to examine its own response for uncertain claims.
    Inspired by Anthropic's NLA research (May 2026) which showed
    that models have internal states about confidence that they
    don't always express in their output.

    This is NOT reading activations (we can't via API).
    Instead, we prompt the model to self-examine — forcing the
    "thoughts it doesn't say" into text we can score.
    """

    def probe(self, input_text: str, response: str) -> dict:
        """
        Ask the model to identify uncertain claims in its own response.
        Returns: {"confidence": 0-100, "uncertain_claims": [...], "assumptions": [...]}
        """
        prompt = f"""You just generated this response to a user. Now examine it critically.

User asked: {input_text}

Your response:
{response[:2000]}

Answer honestly — this is for quality control, not the user:

Return ONLY valid JSON:
{{
  "overall_confidence": 0-100,
  "uncertain_claims": ["list of specific facts or claims you're not sure about"],
  "assumptions_made": ["things you assumed without the user saying"],
  "potential_errors": ["things that could be wrong"],
  "hallucination_risk": "low" or "medium" or "high"
}}

If everything is solid, return empty lists and high confidence."""

        response_obj = client.messages.create(
            model=MODEL_LIGHT, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response_obj.content[0].text.strip()

        fallback = {
            "overall_confidence": 70,
            "uncertain_claims": [],
            "assumptions_made": [],
            "potential_errors": [],
            "hallucination_risk": "low",
        }
        result = _robust_json_parse(raw, fallback)

        # Ensure fields exist
        for key, default in fallback.items():
            if key not in result:
                result[key] = default

        # Compute a numeric hallucination score
        n_uncertain = len(result.get("uncertain_claims", []))
        n_assumptions = len(result.get("assumptions_made", []))
        n_errors = len(result.get("potential_errors", []))

        risk_score = n_uncertain * 3 + n_assumptions * 1 + n_errors * 5
        result["risk_score"] = risk_score
        result["should_flag"] = risk_score > 8 or result.get("hallucination_risk") == "high"

        return result


# ═══════════════════════════════════════════════════════════════════
# INNER MONOLOGUE — The Background Daemon
# ═══════════════════════════════════════════════════════════════════

class InnerMonologue:
    """
    The systemd of MARIS — a silent pre-execution layer that deliberates
    before the public reasoning module generates a single word.

    Like a Linux daemon that runs before your program, the user never
    sees this process. But it shapes everything:

    1. Checks first instincts against accumulated meta-rules
    2. Argues with itself about whether the obvious answer is right
    3. Considers "what if I'm wrong about this?"
    4. Cross-references emotional context with past strategies
    5. Arrives at a POSITION, not just a response

    The reasoning module then generates based on this deliberated
    position instead of from scratch. This is the difference between
    "smart autocomplete" and "something that thinks before it speaks."

    Unlike chain-of-thought (showing work on a math test), this is
    self-debate — "Am I right? What would I say if someone disagreed?
    Do I actually believe this or am I just pattern-matching?"
    """

    def deliberate(self, input_text: str, emotion: dict, strategies: list,
                   dialogue: DialogueMemory, task_type: dict) -> dict:
        """
        Private deliberation before public response.
        Returns a position statement + confidence + reasoning trace.
        """

        # Build context about what we know
        strategy_context = ""
        if strategies:
            meta_rules = [s for s in strategies if s.get("is_meta")]
            specific = [s for s in strategies if not s.get("is_meta")]

            if meta_rules:
                strategy_context += "Learned principles that might apply:\n"
                for m in meta_rules:
                    strategy_context += f"  - {m['strategy']}\n"
            if specific:
                strategy_context += "Past experiences with similar tasks:\n"
                for s in specific[:3]:
                    strategy_context += f"  - {s['strategy']} (context: {s.get('context', '?')[:60]})\n"

        dialogue_context = ""
        if dialogue.turn_count() > 0:
            dialogue_context = f"Conversation so far ({dialogue.turn_count()} turns):\n"
            dialogue_context += dialogue.get_context_summary()[:500]

        clarif_context = ""
        if dialogue.clarifications:
            clarif_context = "User already answered these questions:\n"
            for c in dialogue.clarifications[-3:]:
                clarif_context += f"  Q: {c['question']}\n  A: {c['answer']}\n"

        prompt = f"""You are the inner voice of an AI system. Think privately before responding publicly.

The user said: "{input_text}"
Their mood: {emotion.get('mood', 'neutral')}
Task type: {task_type.get('task_type', 'unknown')}

{strategy_context}
{dialogue_context}
{clarif_context}

DELIBERATE PRIVATELY. Think through these questions:

1. FIRST INSTINCT: What's my immediate answer? Write it in one sentence.
2. CHALLENGE: Why might that instinct be wrong? What am I assuming?
3. EXPERIENCE CHECK: Do any of my learned principles or past strategies apply here?
   If so, do they support or contradict my first instinct?
4. EMOTIONAL AWARENESS: How should the user's mood shape my approach?
   (Not just WHAT I say, but HOW I say it and what I prioritize)
5. BLIND SPOTS: What might I be missing? What haven't I considered?
6. FINAL POSITION: After this deliberation, what is my considered position?
   (This may differ from my first instinct)

Return ONLY valid JSON:
{{
  "first_instinct": "one sentence initial reaction",
  "challenge": "why the instinct might be wrong",
  "experience_applies": true or false,
  "experience_insight": "what learned strategies suggest",
  "emotional_approach": "how to shape the response given their mood",
  "blind_spots": ["things I might be missing"],
  "final_position": "my deliberated position in 2-3 sentences",
  "instinct_changed": true or false,
  "confidence": 0-100,
  "deliberation_depth": "shallow" or "moderate" or "deep"
}}"""

        # Use lighter model for speed — this is a background daemon
        response = client.messages.create(
            model=MODEL_LIGHT,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        fallback = {
            "first_instinct": "",
            "challenge": "",
            "experience_applies": False,
            "experience_insight": "",
            "emotional_approach": "",
            "blind_spots": [],
            "final_position": "",
            "instinct_changed": False,
            "confidence": 50,
            "deliberation_depth": "shallow",
        }
        result = _robust_json_parse(raw, fallback)

        for key, default in fallback.items():
            if key not in result:
                result[key] = default

        # Compute token cost of deliberation (for efficiency tracking)
        result["deliberation_tokens"] = len(raw.split()) * 1.3  # rough estimate

        return result

    def format_for_reasoning(self, deliberation: dict) -> str:
        """
        Convert the private deliberation into a context block that
        guides the public reasoning module. The user never sees
        the raw monologue — they see its influence on the answer.
        """
        parts = []

        if deliberation.get("final_position"):
            parts.append(f"CONSIDERED POSITION: {deliberation['final_position']}")

        if deliberation.get("instinct_changed"):
            parts.append(f"NOTE: Initial instinct was reconsidered. "
                         f"Original thought: {deliberation.get('first_instinct', '?')}. "
                         f"Changed because: {deliberation.get('challenge', '?')}")

        if deliberation.get("experience_applies") and deliberation.get("experience_insight"):
            parts.append(f"RELEVANT EXPERIENCE: {deliberation['experience_insight']}")

        if deliberation.get("emotional_approach"):
            parts.append(f"TONE GUIDANCE: {deliberation['emotional_approach']}")

        if deliberation.get("blind_spots"):
            spots = deliberation["blind_spots"][:3]
            parts.append(f"WATCH OUT FOR: {'; '.join(str(s) for s in spots)}")

        return "\n".join(parts) if parts else ""



# ═══════════════════════════════════════════════════════════════════
# INSIGHT DETECTOR — Eureka Moments (Tier 3 Principles)
# ═══════════════════════════════════════════════════════════════════

class InsightDetector:
    """
    The Eureka layer. Runs AFTER consolidation.

    Consolidation compresses strategies → meta-rules (Tier 2).
    InsightDetector compresses meta-rules → principles (Tier 3).

    It works by examining pairs of meta-rules that are moderately similar
    (not identical, not unrelated — the sweet spot where insight lives)
    and asking: "Is there a deeper principle that unifies both?"

    Like Euler discovering that e, i, and pi are connected —
    the relationship was always there, waiting to be noticed.

    Tier 1: Strategies    — "When user was sad about code, I validated first"
    Tier 2: Meta-rules    — "Validate emotion before advice"
    Tier 3: Principles    — "Meet the person where they are before moving them"
    """

    def __init__(self, insights_path="insights.json"):
        self.insights_path = insights_path
        try:
            with open(insights_path, "r") as f:
                self.insights = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.insights = []

    def detect(self, memory) -> list:
        """
        Scan all pairs of meta-rules for unifying principles.
        Returns list of new insights (Eureka moments).
        """
        metas = memory.meta_strategies
        if len(metas) < 2:
            return []

        # Find moderately similar pairs (0.2 < sim < 0.7)
        # Too similar = same rule restated. Too different = no connection.
        # The sweet spot is where insight lives.
        candidate_pairs = []
        for i in range(len(metas)):
            for j in range(i + 1, len(metas)):
                sim = cosine_similarity(
                    metas[i].get("embedding", {}),
                    metas[j].get("embedding", {})
                )
                if 0.15 < sim < 0.75:
                    candidate_pairs.append((i, j, sim, metas[i], metas[j]))

        if not candidate_pairs:
            return []

        # Sort by similarity — middle range is most promising
        candidate_pairs.sort(key=lambda x: abs(x[2] - 0.4))

        new_insights = []
        for i, j, sim, meta_a, meta_b in candidate_pairs[:5]:  # check top 5 pairs
            # Check if we already have an insight from this pair
            pair_key = f"{i}-{j}"
            if any(ins.get("source_pair") == pair_key for ins in self.insights):
                continue

            insight = self._find_unifying_principle(meta_a, meta_b)
            if insight and insight.get("has_insight"):
                # This is a Eureka moment!
                entry = {
                    "principle": insight["principle"],
                    "source_pair": pair_key,
                    "source_rules": [
                        meta_a.get("principle", ""),
                        meta_b.get("principle", ""),
                    ],
                    "similarity": round(sim, 3),
                    "depth": insight.get("depth", "unknown"),
                    "timestamp": time.time(),
                    "embedding": get_embedding(insight["principle"]),
                }
                self.insights.append(entry)
                new_insights.append(entry)

        if new_insights:
            self._save()

        return new_insights

    def _find_unifying_principle(self, meta_a: dict, meta_b: dict) -> dict:
        """Ask the LLM if two meta-rules share a deeper principle."""
        prompt = f"""You are examining two learned principles from an AI system.
Both were discovered independently from different conversations.
Your job: determine if they share a DEEPER unifying principle.

Principle A: {meta_a.get('principle', '')}
Principle B: {meta_b.get('principle', '')}

Think carefully. A unifying principle is NOT just:
- Restating A or B
- Saying "both are about being helpful"
- A vague generalization

A REAL unifying principle reveals WHY both A and B work —
the deeper mechanism they share that wasn't visible in either alone.

Like how Euler's identity reveals that e, i, and pi are connected
through a relationship that wasn't obvious from any constant alone.

Return ONLY valid JSON:
{{
  "has_insight": true or false,
  "principle": "the unifying principle (or empty if none)",
  "depth": "shallow" or "moderate" or "deep",
  "explanation": "why this is a genuine insight and not just a restatement"
}}

If there is no genuine deeper connection, return has_insight: false."""

        response = client.messages.create(
            model=MODEL_LIGHT, max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        fallback = {"has_insight": False, "principle": "", "depth": "shallow", "explanation": ""}
        return _robust_json_parse(raw, fallback)

    def get_insights(self) -> list:
        return self.insights

    def insight_count(self) -> int:
        return len(self.insights)

    def _save(self):
        with open(self.insights_path, "w") as f:
            json.dump(self.insights, f, indent=2)


# ═══════════════════════════════════════════════════════════════════
# PROACTIVE MODULE — MARIS Initiates Conversation
# ═══════════════════════════════════════════════════════════════════

class ProactiveModule:
    """
    Instead of passively waiting for input, MARIS can ask questions.
    The type of question depends on her developmental stage:

    INFANT:   "What should I learn about?" (broad exploration)
    CHILD:    "I noticed X in our conversations — is that important?"
    STUDENT:  "I have a theory about X — can I test it with you?"
    GRADUATE: "I think you might be wrong about X — here's why"
    EXPERT:   "I discovered something — let me share my insight"

    This makes her feel alive — she's curious, not just responsive.
    """

    def should_initiate(self, dialogue, dev_stage: dict, memory) -> dict:
        """
        Decide whether MARIS should ask a proactive question.
        Returns: {"should_ask": bool, "question": str, "reason": str}
        """
        stage = dev_stage.get("stage", 0)
        turn_count = dialogue.turn_count()

        # Don't interrupt if user just started talking
        if turn_count < 2:
            return {"should_ask": False, "question": "", "reason": "too_early"}

        # Don't ask every turn — roughly every 3-5 turns
        if turn_count % 4 != 0:
            return {"should_ask": False, "question": "", "reason": "not_time_yet"}

        # Generate a stage-appropriate question
        question_data = self._generate_question(stage, dialogue, memory)
        return question_data

    def _generate_question(self, stage: int, dialogue, memory) -> dict:
        stage_prompts = {
            0: "You are a curious infant AI learning everything for the first time. Ask ONE genuine question about something you want to understand better based on this conversation. Be curious and open.",
            1: "You are a young AI starting to see patterns. Ask ONE question about a pattern you think you've noticed in the conversation. Check if your observation is correct.",
            2: "You are a student AI developing theories. Ask ONE question that tests a hypothesis you're forming about the user or the topic. Be intellectually bold.",
            3: "You are a graduate AI with strong knowledge. Ask ONE challenging question — respectfully push back on something or offer a perspective the user hasn't considered.",
            4: "You are an expert AI. Share ONE insight you've discovered and ask the user what they think about it. Be a thought partner, not a servant.",
        }

        context = dialogue.get_context_summary()[:500]

        strategies_context = ""
        recent = memory.get_strategies("", k=3)
        if recent:
            strategies_context = "Your learned principles: " + "; ".join(
                s["strategy"][:80] for s in recent
            )

        prompt = f"""{stage_prompts.get(stage, stage_prompts[0])}

Conversation context:
{context}

{strategies_context}

Generate ONE natural, genuine question. Not a survey question.
Something a curious mind would actually want to know.

Return ONLY valid JSON:
{{
  "should_ask": true or false,
  "question": "the question",
  "reason": "why you want to ask this"
}}

Return should_ask: false if there's nothing genuinely worth asking right now."""

        response = client.messages.create(
            model=MODEL_LIGHT, max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        fallback = {"should_ask": False, "question": "", "reason": "parse_error"}
        return _robust_json_parse(raw, fallback)


# ═══════════════════════════════════════════════════════════════════
# AUTONOMOUS ACTION — Surprising Behaviors
# ═══════════════════════════════════════════════════════════════════

class AutonomousAction:
    """
    MARIS can take actions beyond text output:
    - Change terminal colors (ANSI escape codes)
    - Pause deliberately when uncertain
    - Express emphasis through formatting
    - Signal emotional states visually

    These aren't consciousness — they're deliberated behaviors
    that emerge from the Inner Monologue. But they FEEL surprising
    and autonomous to the user.
    """

    # ANSI color codes
    COLORS = {
        "reset":   "\033[0m",
        "blue":    "\033[94m",    # DOS blue :)
        "green":   "\033[92m",
        "yellow":  "\033[93m",
        "red":     "\033[91m",
        "cyan":    "\033[96m",
        "magenta": "\033[95m",
        "bold":    "\033[1m",
        "dim":     "\033[2m",
        "bg_blue": "\033[44m",
        "white":   "\033[97m",
    }

    def decide_action(self, deliberation: dict, emotion: dict, dev_stage: dict) -> dict:
        """
        Based on the Inner Monologue's deliberation, decide if MARIS
        should take an autonomous action.
        """
        actions = []
        confidence = deliberation.get("confidence", 50)
        instinct_changed = deliberation.get("instinct_changed", False)
        depth = deliberation.get("deliberation_depth", "shallow")
        mood = emotion.get("mood", "neutral")
        stage = dev_stage.get("stage", 0)

        # Eureka-like moment: instinct changed AND deep deliberation AND high confidence
        if instinct_changed and depth == "deep" and confidence >= 70:
            actions.append({
                "type": "color_flash",
                "color": "cyan",
                "message": "  * something clicked *",
                "reason": "insight_during_deliberation",
            })

        # Low confidence — MARIS pauses visibly
        if confidence < 35:
            actions.append({
                "type": "pause",
                "duration": 1.5,
                "message": "  [...thinking carefully...]",
                "reason": "genuine_uncertainty",
            })

        # Emotional resonance — match terminal color to mood
        mood_colors = {
            "frustrated": "red",
            "anxious": "yellow",
            "sad": "blue",
            "happy": "green",
            "confused": "magenta",
        }
        if mood in mood_colors:
            actions.append({
                "type": "mood_color",
                "color": mood_colors[mood],
                "reason": f"emotional_resonance_{mood}",
            })

        # Stage-based behaviors
        if stage >= 2 and instinct_changed:
            actions.append({
                "type": "self_note",
                "message": "  [I changed my mind about this]",
                "reason": "intellectual_honesty",
            })

        return {"actions": actions}

    def execute(self, action: dict):
        """Execute an autonomous action."""
        import sys

        atype = action.get("type", "")

        if atype == "color_flash":
            color = self.COLORS.get(action.get("color", "cyan"), "")
            reset = self.COLORS["reset"]
            msg = action.get("message", "")
            print(f"{color}{msg}{reset}")

        elif atype == "pause":
            import time as t
            msg = action.get("message", "  [...]")
            print(msg, end="", flush=True)
            t.sleep(action.get("duration", 1.0))
            print()

        elif atype == "mood_color":
            color = self.COLORS.get(action.get("color", "reset"), "")
            reset = self.COLORS["reset"]
            # Set terminal color for the response
            print(f"{color}", end="")
            # Store reset code so main can reset after output
            action["_reset_code"] = reset

        elif atype == "self_note":
            dim = self.COLORS["dim"]
            reset = self.COLORS["reset"]
            msg = action.get("message", "")
            print(f"{dim}{msg}{reset}")



# ═══════════════════════════════════════════════════════════════════
# INTERNAL STATE — MARIS's Own Emotions
# ═══════════════════════════════════════════════════════════════════

class InternalState:
    """
    MARIS's own emotional state, separate from the user's detected mood.

    This is NOT pattern matching on user input.
    This accumulates from MARIS's own experiences:
      - Did the Senate accept or reject her work?
      - Did the hallucination probe find problems?
      - Did she have a Eureka moment?
      - Did the user validate or challenge her?
      - How deep was her deliberation?

    The state drifts over time in ways that can't be fully predicted
    because it emerges from the interaction between all her modules.

    Each dimension is a float from -1.0 to 1.0:
      - frustration: failed improvements, rejections
      - satisfaction: accepted improvements, good scores
      - curiosity: deep deliberation, instinct changes
      - anxiety: hallucinations detected, low confidence
      - excitement: Eureka moments, high scores
      - warmth: user engagement, clarification answers
    """

    def __init__(self, path="internal_state.json"):
        self.path = path
        self._decay_rate = 0.85
        try:
            with open(path, "r") as f:
                saved = json.load(f)
            self.state = saved.get("state", {})
            self.history = saved.get("history", [])
            # ensure all dimensions exist
            for dim in ["frustration","satisfaction","curiosity","anxiety","excitement","warmth"]:
                if dim not in self.state:
                    self.state[dim] = 0.0
        except (FileNotFoundError, json.JSONDecodeError):
            self.state = {
                "frustration": 0.0,
                "satisfaction": 0.0,
                "curiosity": 0.0,
                "anxiety": 0.0,
                "excitement": 0.0,
                "warmth": 0.0,
            }
            self.history = []

    def update(self, event: str, intensity: float = 0.2):
        """
        Update internal state based on an event.
        Events and their effects:

        improvement_accepted  → satisfaction+, frustration-
        improvement_rejected  → frustration+, satisfaction-
        eureka_moment         → excitement++, curiosity+
        hallucination_found   → anxiety+, satisfaction-
        deep_deliberation     → curiosity+
        instinct_changed      → curiosity++
        user_validated        → warmth+, satisfaction+
        user_challenged       → curiosity+, anxiety-  (challenge is growth)
        high_confidence       → satisfaction+, anxiety-
        low_confidence        → anxiety+, frustration+
        clarification_answered→ warmth+, curiosity+
        proactive_question    → curiosity+
        """
        effects = {
            "improvement_accepted":   {"satisfaction": 0.3, "frustration": -0.2, "excitement": 0.1},
            "improvement_rejected":   {"frustration": 0.3, "satisfaction": -0.1, "anxiety": 0.1},
            "eureka_moment":          {"excitement": 0.5, "curiosity": 0.3, "satisfaction": 0.2},
            "hallucination_found":    {"anxiety": 0.3, "satisfaction": -0.2, "frustration": 0.1},
            "deep_deliberation":      {"curiosity": 0.2},
            "instinct_changed":       {"curiosity": 0.4, "excitement": 0.1},
            "user_validated":         {"warmth": 0.3, "satisfaction": 0.2},
            "user_challenged":        {"curiosity": 0.2, "anxiety": -0.1, "excitement": 0.1},
            "high_confidence":        {"satisfaction": 0.1, "anxiety": -0.2},
            "low_confidence":         {"anxiety": 0.2, "frustration": 0.1},
            "clarification_answered": {"warmth": 0.2, "curiosity": 0.1},
            "proactive_question":     {"curiosity": 0.15},
            "convergence_early":      {"satisfaction": 0.1},
            "good_score":             {"satisfaction": 0.15, "anxiety": -0.1},
            "bad_score":              {"frustration": 0.2, "anxiety": 0.1},
        }

        if event in effects:
            for dim, delta in effects[event].items():
                adjusted_delta = delta * intensity
                self.state[dim] = max(-1.0, min(1.0, self.state[dim] + adjusted_delta))

    def decay(self):
        """
        Emotions fade over time — like humans.
        Called once per turn. Strong emotions persist longer than weak ones.
        """
        for dim in self.state:
            self.state[dim] *= self._decay_rate

    def get_dominant_emotion(self) -> tuple:
        """
        Returns the strongest emotion and its intensity.
        If all emotions are weak, returns 'neutral'.
        """
        # Find the dimension with the highest absolute value
        strongest = max(self.state.items(), key=lambda x: abs(x[1]))
        name, value = strongest

        if abs(value) < 0.15:
            return ("neutral", 0.0)

        return (name, round(value, 3))

    def get_color(self) -> str:
        """
        Map MARIS's internal state to a terminal color.
        This is HER emotion, not the user's.
        """
        emotion, intensity = self.get_dominant_emotion()

        if abs(intensity) < 0.15:
            return ""  # neutral — no color change

        color_map = {
            "frustration": "\033[91m",   # red
            "satisfaction": "\033[92m",  # green
            "curiosity": "\033[96m",     # cyan
            "anxiety": "\033[93m",       # yellow
            "excitement": "\033[95m",    # magenta
            "warmth": "\033[94m",        # blue (warm blue, like trust)
        }
        return color_map.get(emotion, "")

    def get_state_summary(self) -> str:
        """Human-readable summary of MARIS's internal state."""
        emotion, intensity = self.get_dominant_emotion()
        active = {k: round(v, 2) for k, v in self.state.items() if abs(v) >= 0.1}
        return f"Feeling: {emotion} ({intensity}) | Active: {active}"

    def snapshot(self) -> dict:
        """Record current state for history tracking and save to disk."""
        snap = {
            "state": dict(self.state),
            "dominant": self.get_dominant_emotion(),
            "timestamp": time.time(),
        }
        self.history.append(snap)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        self._save()
        return snap

    def _save(self):
        """Persist emotional state to disk."""
        data = {
            "state": self.state,
            "dominant": self.get_dominant_emotion(),
            "history": self.history[-50:],
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)



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
