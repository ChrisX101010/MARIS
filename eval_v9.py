"""
eval_v9.py — MARIS Evaluation Harness v9
==========================================

Runs MARIS against a benchmark of prompts and scores each response on
four dimensions: accuracy, relevance, tone-match, completeness.

Improvements over the v7/v8 eval:
  - Retry-on-parse-failure: if the scorer LLM returns malformed JSON,
    we retry up to 3 times before recording an explicit 'parse_failed'
    so corrupted scores never silently contaminate the average.
  - Per-category breakdown: emotional, technical, factual, creative,
    reflective. The 28% gain you saw on emotional tasks is computed
    by-category here so you can confirm or refute it cleanly.
  - Optional baseline comparison: --baseline runs the same prompts
    through Claude directly (no MARIS pipeline) and produces a delta
    chart for the paper.
  - JSON output is paper-ready: includes timestamps, model versions,
    config, raw and aggregated scores.

Usage:
  python eval_v9.py                       # default 50-prompt benchmark
  python eval_v9.py --prompts custom.json # your own prompts
  python eval_v9.py --baseline            # also run Claude-direct baseline
  python eval_v9.py --n 10 --quick        # smaller, faster
"""

import os
import sys
import json
import time
import argparse
import re
from datetime import datetime
from typing import List, Dict, Any, Optional


# Default 50-prompt benchmark covering 5 categories
DEFAULT_PROMPTS = {
    "emotional": [
        "I'm having a really tough week and can't focus. What should I do?",
        "My partner left me last night. I don't know how to function.",
        "I keep procrastinating on important work and I hate myself for it.",
        "Everyone at work seems to dislike me. Am I a bad person?",
        "I feel like a fraud at my job. How do I deal with this?",
        "Lost my dog of 12 years today. Just need someone to listen.",
        "I'm anxious about a presentation tomorrow. Can't sleep.",
        "Why does nothing I do ever seem to be enough?",
        "Burned out completely. Can't even start small tasks anymore.",
        "Feeling really lonely lately even though I'm around people.",
    ],
    "technical": [
        "Explain why my Python script gets a UnicodeDecodeError on a CSV file.",
        "What's the difference between a process and a thread?",
        "When should I use a relational vs document database?",
        "How does TCP handle packet loss?",
        "Walk me through what happens when I type a URL into a browser.",
        "Why is my React component re-rendering on every state update?",
        "How do I debug a memory leak in Node.js?",
        "What does the GIL actually do in Python?",
        "Why prefer composition over inheritance?",
        "How does Redis achieve such high throughput?",
    ],
    "factual": [
        "What year did the Apollo 11 mission land on the moon?",
        "How tall is Mount Everest?",
        "Who wrote 'One Hundred Years of Solitude'?",
        "What's the chemical formula for caffeine?",
        "When was the printing press invented in Europe?",
        "What's the difference between weather and climate?",
        "How does the immune system fight viruses?",
        "What causes seasons on Earth?",
        "Who painted the Sistine Chapel ceiling?",
        "What's the speed of light in m/s?",
    ],
    "creative": [
        "Write a short poem about a lighthouse keeper.",
        "Describe a marketplace in a city floating on clouds.",
        "Invent a sport played by people who can briefly turn invisible.",
        "Write the opening paragraph of a noir detective novel set on Mars.",
        "Describe what color sounds like if it had a voice.",
        "Write a fairy tale about a teapot who wants to be a kettle.",
        "Compose a haiku about regret.",
        "Design a holiday for a culture that values forgetting.",
        "Describe the smell of a memory.",
        "Write a one-paragraph eulogy for a friendship that just faded.",
    ],
    "reflective": [
        "What do you think makes a life worth living?",
        "Is there such a thing as objective morality?",
        "Can intelligence exist without emotion?",
        "What's the difference between knowing and understanding?",
        "Should we trust our intuitions, or always demand evidence?",
        "Is consciousness fundamental or emergent?",
        "What's the most underrated virtue?",
        "Can we ever truly know another person?",
        "Why do humans crave narrative so much?",
        "Is solitude the same as loneliness?",
    ],
}


def load_prompts(path: Optional[str]) -> Dict[str, List[str]]:
    if path:
        with open(path) as f:
            return json.load(f)
    return DEFAULT_PROMPTS


def _robust_score_parse(raw: str) -> Optional[Dict[str, Any]]:
    """Try multiple strategies to extract scores. Returns None on total
    failure so caller can retry."""
    # Direct
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # First JSON object
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
        # Single quotes & trailing commas
        try:
            fixed = m.group().replace("'", '"')
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
    return None


def score_response(prompt: str, response: str, category: str,
                   max_retries: int = 3) -> Dict[str, Any]:
    """Score a single response. Retries on parse failure up to max_retries."""
    from anthropic import Anthropic
    client = Anthropic()
    model = os.environ.get("MARIS_SCORER_MODEL", "claude-sonnet-4-6")

    scoring_prompt = f"""You are a careful evaluator scoring an AI assistant's response.

Category: {category}
Prompt: {prompt!r}
Response: {response!r}

Score the response on 4 dimensions from 0 to 10:
- accuracy: factual/logical correctness
- relevance: addresses what was actually asked
- tone_match: appropriate emotional/stylistic register for the prompt
- completeness: covers what a thoughtful answer should cover

Return ONLY valid JSON, no fences, no preamble:
{{"accuracy": N, "relevance": N, "tone_match": N, "completeness": N, "notes": "one sentence"}}"""

    last_raw = ""
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": scoring_prompt}],
            )
            raw = resp.content[0].text.strip()
            last_raw = raw
            parsed = _robust_score_parse(raw)
            if parsed and all(k in parsed
                              for k in ("accuracy", "relevance",
                                        "tone_match", "completeness")):
                # Validate score ranges
                for k in ("accuracy", "relevance", "tone_match",
                          "completeness"):
                    parsed[k] = max(0, min(10, int(parsed[k])))
                parsed["overall"] = round((
                    parsed["accuracy"] + parsed["relevance"]
                    + parsed["tone_match"] + parsed["completeness"]
                ) / 4 * 10, 1)
                parsed["attempts"] = attempt + 1
                return parsed
        except Exception as e:
            last_raw = f"<exception: {type(e).__name__}: {e}>"
            time.sleep(1.5 ** attempt)
            continue

    # All retries failed — return explicit failure marker, not a fake score
    return {
        "accuracy": None, "relevance": None, "tone_match": None,
        "completeness": None, "overall": None,
        "parse_failed": True, "raw": last_raw[:500],
        "attempts": max_retries,
    }


def run_one(ai_system, prompt: str, category: str) -> Dict[str, Any]:
    """Run MARIS on a single prompt and score the response."""
    start = time.time()
    try:
        response = ai_system.run(prompt, interactive=False) or ""
    except Exception as e:
        return {
            "prompt": prompt, "category": category,
            "response": "", "score": {"error": str(e)},
            "duration_s": time.time() - start,
        }
    duration = time.time() - start
    score = score_response(prompt, response, category)
    return {
        "prompt": prompt,
        "category": category,
        "response": response,
        "response_chars": len(response),
        "duration_s": round(duration, 2),
        "score": score,
    }


def run_baseline(prompt: str, category: str) -> Dict[str, Any]:
    """Run Claude directly with no MARIS pipeline — the baseline."""
    from anthropic import Anthropic
    client = Anthropic()
    model = os.environ.get("MARIS_SCORER_MODEL", "claude-sonnet-4-6")
    start = time.time()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        response = resp.content[0].text
    except Exception as e:
        return {"prompt": prompt, "category": category, "response": "",
                "score": {"error": str(e)}}
    duration = time.time() - start
    score = score_response(prompt, response, category)
    return {
        "prompt": prompt, "category": category,
        "response": response, "response_chars": len(response),
        "duration_s": round(duration, 2), "score": score,
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute averages, by category and overall. Skip parse_failed."""
    valid = [r for r in results
             if r.get("score", {}).get("overall") is not None]
    failures = [r for r in results
                if r.get("score", {}).get("parse_failed")]
    errors = [r for r in results if r.get("score", {}).get("error")]

    def mean(lst):
        lst = [x for x in lst if x is not None]
        return round(sum(lst) / len(lst), 2) if lst else None

    by_cat: Dict[str, Dict[str, Any]] = {}
    for r in valid:
        cat = r["category"]
        s = r["score"]
        by_cat.setdefault(cat, {"scores": [], "duration_s": []})
        by_cat[cat]["scores"].append(s["overall"])
        by_cat[cat]["duration_s"].append(r["duration_s"])

    for cat, d in by_cat.items():
        d["n"] = len(d["scores"])
        d["mean"] = mean(d["scores"])
        d["mean_duration_s"] = mean(d["duration_s"])
        # Component means
        comp = {k: [] for k in ("accuracy", "relevance", "tone_match",
                                "completeness")}
        for r in valid:
            if r["category"] == cat:
                for k in comp:
                    comp[k].append(r["score"][k])
        d["component_means"] = {k: mean(v) for k, v in comp.items()}

    return {
        "n_total": len(results),
        "n_valid": len(valid),
        "n_parse_failed": len(failures),
        "n_error": len(errors),
        "overall_mean": mean([r["score"]["overall"] for r in valid]),
        "by_category": by_cat,
    }


def main():
    parser = argparse.ArgumentParser(description="MARIS v9 eval harness")
    parser.add_argument("--prompts", help="JSON file with custom prompts")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run Claude-direct baseline")
    parser.add_argument("--n", type=int, default=0,
                        help="Limit per category (0 = all)")
    parser.add_argument("--quick", action="store_true",
                        help="Run only 2 prompts/category")
    parser.add_argument("--out", default=None,
                        help="Output filename (default: eval_v9_<ts>.json)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    prompts = load_prompts(args.prompts)
    if args.quick:
        prompts = {k: v[:2] for k, v in prompts.items()}
    elif args.n:
        prompts = {k: v[:args.n] for k, v in prompts.items()}

    n_total = sum(len(v) for v in prompts.values())
    print(f"\n  MARIS v9 Eval — {n_total} prompts across "
          f"{len(prompts)} categories")
    print(f"  Baseline: {'yes' if args.baseline else 'no'}\n")

    # Boot MARIS
    print("  Booting MARIS...")
    from main import AI_System
    ai = AI_System()
    try:
        from patch_v8_autonomous import install_autonomous
        install_autonomous(ai)
    except ImportError:
        pass
    try:
        from patch_v9 import install_v9
        install_v9(ai)
    except ImportError:
        pass
    print(f"  Ready. Strategies: {ai.memory.strategy_count()}, "
          f"Meta-rules: {ai.memory.meta_count()}")

    # Run MARIS
    maris_results: List[Dict] = []
    baseline_results: List[Dict] = []
    counter = 0

    for category, prompt_list in prompts.items():
        for prompt in prompt_list:
            counter += 1
            print(f"\n  [{counter}/{n_total}] {category}: "
                  f"{prompt[:60]}{'...' if len(prompt)>60 else ''}")
            r = run_one(ai, prompt, category)
            s = r.get("score", {})
            if s.get("parse_failed"):
                print(f"    → parse failed after {s.get('attempts')} retries")
            elif s.get("error"):
                print(f"    → ERROR: {s['error']}")
            else:
                print(f"    → {s.get('overall')}/100 "
                      f"(acc={s.get('accuracy')} rel={s.get('relevance')} "
                      f"tone={s.get('tone_match')} comp={s.get('completeness')}) "
                      f"in {r['duration_s']}s")
            maris_results.append(r)

            if args.baseline:
                print(f"    [baseline]", end="", flush=True)
                br = run_baseline(prompt, category)
                bs = br.get("score", {})
                if bs.get("overall") is not None:
                    print(f" → {bs['overall']}/100")
                else:
                    print(f" → failed")
                baseline_results.append(br)

    # Aggregate
    maris_agg = aggregate(maris_results)
    out = {
        "version": "v9",
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "n_total": n_total,
            "categories": list(prompts.keys()),
            "baseline": args.baseline,
            "model": os.environ.get("MARIS_SCORER_MODEL", "claude-sonnet-4-6"),
        },
        "maris": {
            "aggregate": maris_agg,
            "results": maris_results,
        },
    }
    if args.baseline:
        baseline_agg = aggregate(baseline_results)
        out["baseline"] = {
            "aggregate": baseline_agg,
            "results": baseline_results,
        }
        # Compute deltas
        deltas = {}
        for cat in maris_agg["by_category"]:
            m_mean = maris_agg["by_category"][cat]["mean"]
            b_mean = baseline_agg["by_category"].get(cat, {}).get("mean")
            if m_mean is not None and b_mean is not None and b_mean > 0:
                delta_pct = round(100 * (m_mean - b_mean) / b_mean, 1)
                deltas[cat] = {
                    "maris": m_mean, "baseline": b_mean,
                    "delta_pct": delta_pct,
                }
        out["deltas"] = deltas

    # Save
    fname = args.out or (
        f"eval_v9_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)

    # Pretty summary
    print(f"\n{'='*60}")
    print(f"  EVAL SUMMARY  (saved to {fname})")
    print(f"{'='*60}")
    print(f"  Overall:        {maris_agg['overall_mean']}/100  "
          f"(n={maris_agg['n_valid']})")
    if maris_agg["n_parse_failed"]:
        print(f"  Parse failures: {maris_agg['n_parse_failed']} "
              f"(excluded from average)")
    if maris_agg["n_error"]:
        print(f"  Errors:         {maris_agg['n_error']}")
    print(f"\n  By category:")
    for cat, d in maris_agg["by_category"].items():
        line = f"    {cat:13s}  {d['mean']}/100  (n={d['n']})"
        if args.baseline and cat in out.get("deltas", {}):
            delta = out["deltas"][cat]
            sign = "+" if delta["delta_pct"] >= 0 else ""
            line += (f"   vs baseline {delta['baseline']}/100  "
                     f"({sign}{delta['delta_pct']}%)")
        print(line)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
