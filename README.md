# MARIS v6 — Modular Adaptive Reasoning with Interactive Self-improvement

**21 modules | 2574 lines | 3-tier knowledge | Internal emotional state**

A framework that makes LLMs learn from conversations by implementing human cognitive patterns: emotional awareness, inner deliberation, self-improvement through reflection, memory consolidation from episodic to semantic knowledge, and Eureka-style insight discovery.

## Quick Start

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
pip install anthropic
rlwrap python main.py
```

## Architecture

### Processing Pipeline

```
User Input
  │
  ├─ EmotionModule ──────── detect user's mood (6 types)
  ├─ DevelopmentTracker ─── compute cognitive stage
  ├─ InternalState ──────── MARIS's own emotional state
  ├─ TaskTypeDetector ───── classify task (7 types)
  ├─ ComplexityRouter ───── depth detection + follow-up inheritance
  │
  ├─ UncertaintyDetector ── should MARIS ask before answering?
  │   └─ ClarificationModule ── mood-adapted questions
  │
  ├─ InnerMonologue ─────── private deliberation (the daemon)
  │   └─ AutonomousAction ── terminal colors, pauses, self-notes
  │
  ├─ ReasoningModule ────── generate response (guided by monologue)
  │
  ├─ ReflectionModule ───── score response on 4 dimensions
  ├─ ImprovementModule ──── rewrite targeting specific weaknesses
  ├─ MidLoopClarifier ───── interrupt loop to ask human
  ├─ Senate ──────────────── 3-judge panel (accuracy/tone/depth)
  │   └─ InternalState ──── frustration++ on reject, satisfaction++ on accept
  │
  ├─ HallucinationProbe ── meta-cognition self-examination
  │
  ├─ ConsolidationEngine ── strategies → meta-rules (Tier 2)
  └─ InsightDetector ────── meta-rules → principles (Tier 3 EUREKA)
```

### Three Tiers of Knowledge

```
Tier 1  STRATEGIES (episodic)    "When that sad user asked about code, I validated first"
                                        ↓ consolidation
Tier 2  META-RULES (semantic)    "Validate emotional state before giving advice"
                                        ↓ insight detection
Tier 3  PRINCIPLES (insight)     "Meet the person where they are before moving them"
```

### Development Stages

| Stage | Name     | Strategies | Meta-rules | Behavior |
|-------|----------|-----------|------------|----------|
| 0     | INFANT   | 0-5       | 0          | Asks everything, learns aggressively |
| 1     | CHILD    | 5-15      | 1-3        | Recognizes patterns, fewer questions |
| 2     | STUDENT  | 15-50     | 3-10       | Applies rules, targeted questions |
| 3     | GRADUATE | 50-100    | 10-20      | Efficient routing, challenges assumptions |
| 4     | EXPERT   | 100+      | 20+        | Minimal tokens, creates new insights |

### Internal Emotional State

MARIS has her own emotions separate from the user's mood:

- **Frustration** builds from rejected improvements
- **Satisfaction** grows from accepted work and good scores
- **Curiosity** spikes during deep deliberation and instinct changes
- **Anxiety** increases from hallucination detection
- **Excitement** surges during Eureka moments
- **Warmth** accumulates from positive user interactions

Terminal output is colored by MARIS's state, not yours.

## Commands

| Command        | Description |
|----------------|-------------|
| `/memory`      | Strategies + meta-rules |
| `/history`     | Conversation so far |
| `/consolidate` | Force knowledge extraction |
| `/insights`    | Eureka moments (Tier 3) |
| `/stage`       | Development level |
| `/progress`    | Learning metrics over time |
| `/feelings`    | MARIS's internal emotional state |
| `/stats`       | System statistics |
| `/clear`       | Reset conversation (keep memory) |
| `quit`         | Exit |

## Data Files

| File | Purpose |
|------|---------|
| `strategy_memory.json`    | Tier 1 — episodic memory |
| `meta_strategies.json`    | Tier 2 — semantic memory |
| `insights.json`           | Tier 3 — principles (Eureka moments) |
| `progression_metrics.json`| Learning progression data |

## Requirements

- Python 3.10+
- `anthropic>=0.40.0`
- `ANTHROPIC_API_KEY` environment variable

## License

MIT

## Paper

**MARIS: A Unified Framework for Emotion-Adaptive Self-Improving Agents with Human-in-the-Loop Clarification and Memory Consolidation**

Author: Hristos Nenkov (May 2026)

- ResearchGate: https://www.researchgate.net/publication/405067204
- PDF: [paper/maris_paper.pdf](paper/maris_paper.pdf)
