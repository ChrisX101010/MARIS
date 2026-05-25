"""
dashboard_v9.py — MARIS Brain Dashboard v9 (with Anatomy view)
=================================================================

Runs on http://localhost:3000

Two tabs:
  - Cognitive Map: existing v9 view — stats, knowledge graph, autonomous log
  - Anatomy:       new brain SVG view — modules mapped to functional regions

Reads from the actual data files:
  strategy_memory.json, meta_strategies.json, insights.json,
  internal_state.json, flight_log.json, autonomous_log.json,
  progression_metrics.json, eval_*.json

Usage:
    pip install flask
    python dashboard_v9.py
    # → http://localhost:3000

On the Anatomy view: hovering over any brain region shows the MARIS module(s)
mapped to it, its functional role, and current activity. Region color
saturation reflects activity intensity from internal_state.json + flight_log.
"""

import os
import re
import json
import glob
from datetime import datetime
from collections import Counter, defaultdict

try:
    from flask import Flask, jsonify, render_template_string
except ImportError:
    print("ERROR: dashboard requires flask. Install with: pip install flask")
    raise SystemExit(1)


DATA_DIR = os.environ.get("MARIS_DATA_DIR", os.getcwd())
PORT = int(os.environ.get("MARIS_DASHBOARD_PORT", "3000"))


# ───────────────────────────────────────────────────────────────────────────
# Data loaders
# ───────────────────────────────────────────────────────────────────────────
def _load(name, default):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def load_strategies():     return _load("strategy_memory.json", [])
def load_metas():          return _load("meta_strategies.json", [])
def load_insights():       return _load("insights.json", [])
def load_internal_state(): return _load("internal_state.json", {})
def load_autonomous_log(): return _load("autonomous_log.json", [])


def load_flight_log():
    log = _load("flight_log.json", [])
    if isinstance(log, dict):
        return log.get("turns", log.get("events", []))
    return log


# ───────────────────────────────────────────────────────────────────────────
# Metric aggregation (unchanged from v9.0)
# ───────────────────────────────────────────────────────────────────────────
def compute_metrics():
    log = load_flight_log()
    accepted = rejected = halluc = overrides = eureka = consolid = 0
    for turn in log:
        if not isinstance(turn, dict):
            continue
        kinds = []
        if "kind" in turn:
            kinds.append((turn["kind"], turn.get("payload", turn)))
        if "events" in turn and isinstance(turn["events"], list):
            for e in turn["events"]:
                if isinstance(e, dict) and "kind" in e:
                    kinds.append((e["kind"], e.get("payload", e)))
        for kind, payload in kinds:
            if kind == "senate":
                if isinstance(payload, dict):
                    if payload.get("accepted"):
                        accepted += 1
                    else:
                        rejected += 1
            elif kind in ("hallucination", "halluc"):
                halluc += 1
            elif kind == "drive_resolver" and isinstance(payload, dict):
                if payload.get("overridden"):
                    overrides += 1
            elif kind == "eureka":
                eureka += 1
            elif kind == "consolidation":
                consolid += 1

    strategies = load_strategies()
    metas = load_metas()
    insights = load_insights()
    auto = [e for e in load_autonomous_log()
            if e.get("event", "").startswith("perception_strategy_added")]
    return {
        "turns": len(log),
        "strategies": len(strategies),
        "meta_rules": len(metas),
        "insights": len(insights),
        "senate_accepted": accepted,
        "senate_rejected": rejected,
        "hallucinations_caught": halluc,
        "drive_overrides": overrides,
        "eurekas": eureka,
        "consolidations": consolid,
        "autonomous_strategies": len(auto),
        "acceptance_rate": (
            round(100 * accepted / max(accepted + rejected, 1), 1)
        ),
    }


def compute_graph():
    strategies = load_strategies()
    metas = load_metas()
    insights = load_insights()
    nodes, edges = [], []
    for i, s in enumerate(strategies[-50:]):
        nid = f"s{i}"
        nodes.append({
            "id": nid, "type": "strategy",
            "label": s.get("strategy", "")[:80],
            "task": s.get("task_type", "?"),
            "mood": s.get("mood", "?"),
            "source": s.get("source", "interactive"),
            "confidence": s.get("confidence", 50),
            "consequence_result": s.get("consequence_result"),
        })
    for i, m in enumerate(metas):
        mid = f"m{i}"
        nodes.append({
            "id": mid, "type": "meta",
            "label": m.get("principle", "")[:100],
            "confidence": m.get("confidence", 50),
            "source_count": m.get("source_count", 0),
            "mood_pattern": m.get("mood_pattern", "all"),
        })
        for j, s in enumerate(strategies[-30:]):
            if (m.get("mood_pattern") in ("all", s.get("mood"))
                    and _text_overlap(m.get("principle", ""),
                                      s.get("strategy", "")) > 0.15):
                edges.append({
                    "source": f"s{j + max(0, len(strategies)-30)}",
                    "target": mid, "kind": "consolidates"
                })
    for i, ins in enumerate(insights):
        iid = f"i{i}"
        nodes.append({
            "id": iid, "type": "insight",
            "label": ins.get("principle", "")[:120],
            "depth": ins.get("depth", "?"),
        })
        for src in ins.get("source_rules", []):
            for j, m in enumerate(metas):
                if _text_overlap(src, m.get("principle", "")) > 0.3:
                    edges.append({"source": f"m{j}", "target": iid,
                                  "kind": "abstracts"})
    return {"nodes": nodes, "edges": edges}


def _text_overlap(a, b):
    if not a or not b:
        return 0.0
    aw = set(re.findall(r"\w+", a.lower()))
    bw = set(re.findall(r"\w+", b.lower()))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / min(len(aw), len(bw))


# ───────────────────────────────────────────────────────────────────────────
# NEW: Anatomy data — modules mapped to brain regions
# ───────────────────────────────────────────────────────────────────────────
def compute_anatomy():
    """Map MARIS modules to functional brain regions with activity levels.

    Activity sources:
      - emotion-related regions: from internal_state.json values
      - cognitive regions: from autonomous log event frequencies + metrics
      - memory regions: from strategy/meta counts (growth = activity)
    """
    state = load_internal_state().get("state", {})
    auto = load_autonomous_log()
    metrics = compute_metrics()
    strategies = load_strategies()

    # Count recent autonomous events by type (last 100)
    event_counts = Counter(e.get("event", "") for e in auto[-100:])
    perception_count = (event_counts.get("perception_strategy_added", 0)
                        + event_counts.get("search_success", 0))
    consequence_count = event_counts.get("consequence_tested", 0)
    drive_count = event_counts.get("drive_override", 0)

    # Normalize activity to 0-1 range
    def norm(value, ceiling):
        if value is None:
            return 0.0
        return min(1.0, abs(float(value)) / ceiling)

    # Compose total state magnitude for "energy" baseline
    state_energy = sum(abs(v) for v in state.values()) if state else 0

    regions = {
        # ── Cerebrum: cortical regions ─────────────────────────────────────
        "prefrontal_cortex": {
            "label": "Prefrontal Cortex",
            "function": "Executive function, planning, decision evaluation",
            "modules": ["Reasoning", "Senate", "Inner Monologue"],
            "activity": min(1.0,
                (metrics["senate_accepted"] + metrics["senate_rejected"]) / 30
            ),
            "color": "#a78bfa",  # purple — executive
            "detail": (
                f"{metrics['senate_accepted']} accepted, "
                f"{metrics['senate_rejected']} rejected verdicts"
            ),
        },
        "anterior_cingulate": {
            "label": "Anterior Cingulate Cortex (ACC)",
            "function": "Error/conflict monitoring, self-correction",
            "modules": ["Hallucination Probe", "UncertaintyDetector"],
            "activity": min(1.0, metrics["hallucinations_caught"] / 8),
            "color": "#fbbf24",  # amber — vigilance
            "detail": f"{metrics['hallucinations_caught']} hallucinations caught",
        },
        "default_mode_network": {
            "label": "Default Mode Network",
            "function": "Self-narrative, internal reflection, deliberation",
            "modules": ["Inner Monologue", "Reflection"],
            "activity": min(1.0, len(strategies) / 50),  # grows with thinking
            "color": "#8b5cf6",  # violet
            "detail": (
                f"{len(strategies)} strategies accumulated through reflection"
            ),
        },
        "parietal_cortex": {
            "label": "Parietal Cortex",
            "function": "Integration, sensory synthesis",
            "modules": ["TaskTypeDetector", "ComplexityRouter"],
            "activity": min(1.0, metrics["turns"] / 50),
            "color": "#5eb3ff",  # blue — integrative
            "detail": f"{metrics['turns']} turns integrated",
        },
        "temporal_cortex": {
            "label": "Temporal Cortex",
            "function": "Pattern recognition, language",
            "modules": ["HumanPatternDetector", "EmotionModule"],
            "activity": min(1.0,
                (metrics["turns"] + metrics["consolidations"]) / 30
            ),
            "color": "#22d3ee",  # cyan
            "detail": f"{metrics['consolidations']} consolidation events",
        },
        "occipital_cortex": {
            "label": "Occipital Cortex (Sensory)",
            "function": "External perception",
            "modules": ["Active Perception (v8)", "Web Search"],
            "activity": min(1.0, perception_count / 10),
            "color": "#4ade80",  # green — sensory
            "detail": f"{perception_count} perception events from autonomous learning",
        },

        # ── Deep brain structures ──────────────────────────────────────────
        "hippocampus": {
            "label": "Hippocampus",
            "function": "Long-term memory formation, consolidation",
            "modules": ["StrategyMemory", "ConsolidationEngine"],
            "activity": min(1.0,
                (metrics["strategies"] + 3 * metrics["meta_rules"]
                 + 5 * metrics["insights"]) / 80
            ),
            "color": "#fb923c",  # orange — memory
            "detail": (
                f"{metrics['strategies']} episodic + {metrics['meta_rules']} "
                f"semantic + {metrics['insights']} insight"
            ),
        },
        "amygdala": {
            "label": "Amygdala",
            "function": "Emotion processing, threat detection",
            "modules": ["EmotionModule", "InternalState (frustration/anxiety)"],
            "activity": norm(
                state.get("frustration", 0) + state.get("anxiety", 0), 2.0
            ),
            "color": "#f87171",  # red — emotion
            "detail": (
                f"frustration={state.get('frustration', 0):.2f} "
                f"anxiety={state.get('anxiety', 0):.2f}"
            ),
        },
        "insula": {
            "label": "Insula",
            "function": "Interoception, internal-state awareness",
            "modules": ["InternalState (full)", "DriveResolver"],
            "activity": min(1.0, state_energy / 3),
            "color": "#ec4899",  # pink — bodily awareness
            "detail": (
                f"total emotional energy = {state_energy:.2f} "
                f"({drive_count} drive overrides)"
            ),
        },
        "thalamus": {
            "label": "Thalamus",
            "function": "Information relay, attention gating",
            "modules": ["MidLoopClarifier", "ClarificationModule"],
            "activity": min(1.0, metrics["turns"] / 40),
            "color": "#e879f9",  # magenta
            "detail": "Routes between modules during each turn",
        },

        # ── Subcortical / brainstem ────────────────────────────────────────
        "cerebellum": {
            "label": "Cerebellum",
            "function": "Fine motor control, behavioral coordination",
            "modules": ["DriveResolver", "ProactiveModule"],
            "activity": min(1.0, (drive_count + 1) / 10),
            "color": "#10b981",  # teal — coordination
            "detail": f"{drive_count} drive-based response adjustments",
        },
        "brainstem": {
            "label": "Brainstem",
            "function": "Background regulation, autonomic processes",
            "modules": ["Autonomous Daemon (v8)", "FlightRecorder"],
            "activity": min(1.0, (perception_count + consequence_count) / 20),
            "color": "#f59e0b",  # amber-orange — autonomic
            "detail": (
                f"{perception_count + consequence_count} autonomous "
                f"background events"
            ),
        },
    }

    return {
        "regions": regions,
        "dominant_emotion": load_internal_state().get("dominant", {}),
        "saved_at": load_internal_state().get("saved_at"),
    }


# ───────────────────────────────────────────────────────────────────────────
# Flask app
# ───────────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/state")
def api_state():
    return jsonify(load_internal_state())


@app.route("/api/memory")
def api_memory():
    return jsonify({
        "strategies": load_strategies()[-30:],
        "metas": load_metas(),
        "insights": load_insights(),
    })


@app.route("/api/metrics")
def api_metrics():
    return jsonify(compute_metrics())


@app.route("/api/graph")
def api_graph():
    return jsonify(compute_graph())


@app.route("/api/autonomous")
def api_autonomous():
    log = load_autonomous_log()
    return jsonify({"events": log[-50:], "total": len(log)})


@app.route("/api/anatomy")
def api_anatomy():
    return jsonify(compute_anatomy())


# ───────────────────────────────────────────────────────────────────────────
# HTML — tabbed dashboard with Cognitive Map + Anatomy
# ───────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>MARIS v9 — Brain Dashboard</title>
<style>
  :root {
    --bg: #0a0e1a; --panel: #131826; --border: #1f2940;
    --text: #d8def0; --muted: #6b7596; --accent: #5eb3ff;
    --green: #4ade80; --red: #f87171; --yellow: #fbbf24;
    --purple: #a78bfa; --cyan: #22d3ee;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace;
    font-size: 13px; }
  header { padding: 14px 24px; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center; }
  h1 { margin: 0; font-size: 18px; font-weight: 600; }
  h2 { margin: 0 0 12px 0; font-size: 13px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.5px; }
  .tabs { display: flex; gap: 4px; padding: 0 24px; background: var(--bg);
    border-bottom: 1px solid var(--border); }
  .tab { padding: 12px 20px; cursor: pointer; color: var(--muted);
    font-size: 13px; border-bottom: 2px solid transparent;
    transition: all 0.15s; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .grid { display: grid;
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
    gap: 16px; padding: 16px 24px; }
  .panel { background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; min-height: 200px; }
  .panel.wide { grid-column: 1 / -1; }
  .metric { display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04); }
  .metric:last-child { border: none; }
  .metric .v { font-weight: 600; color: var(--accent); }
  .metric .v.green { color: var(--green); }
  .metric .v.red { color: var(--red); }
  .metric .v.yellow { color: var(--yellow); }
  .bar-container { display: flex; align-items: center; gap: 8px;
    padding: 4px 0; font-size: 12px; }
  .bar-container .label { width: 110px; color: var(--muted); }
  .bar-container .bar-bg { flex: 1; height: 6px;
    background: rgba(255,255,255,0.06); border-radius: 3px;
    overflow: hidden; position: relative; }
  .bar-container .bar-fill { position: absolute; left: 50%; height: 100%;
    transition: all 0.3s; }
  .bar-container .bar-fill.pos { background: var(--green); }
  .bar-container .bar-fill.neg { background: var(--red); }
  .bar-container .v { width: 60px; text-align: right; font-size: 11px;
    color: var(--muted); font-variant-numeric: tabular-nums; }
  .strategy-list { max-height: 280px; overflow-y: auto; }
  .strategy-item { padding: 8px; margin-bottom: 6px;
    background: rgba(255,255,255,0.02);
    border-left: 2px solid var(--border); border-radius: 4px;
    font-size: 12px; line-height: 1.4; }
  .strategy-item.meta { border-left-color: var(--purple); }
  .strategy-item.insight { border-left-color: var(--yellow); }
  .strategy-item.autonomous { border-left-color: var(--cyan); }
  .strategy-item .tag { display: inline-block; padding: 1px 6px;
    background: rgba(94,179,255,0.15); border-radius: 3px;
    font-size: 10px; color: var(--accent); margin-right: 4px; }
  .strategy-item .tag.meta { background: rgba(167,139,250,0.15); color: var(--purple); }
  .strategy-item .tag.insight { background: rgba(251,191,36,0.15); color: var(--yellow); }
  .strategy-item .tag.autonomous { background: rgba(34,211,238,0.15); color: var(--cyan); }
  .strategy-item .tag.pass { background: rgba(74,222,128,0.15); color: var(--green); }
  .strategy-item .tag.fail { background: rgba(248,113,113,0.15); color: var(--red); }
  .stat-grid { display: grid; grid-template-columns: repeat(2, 1fr);
    gap: 8px 16px; }
  .log-line { font-size: 11px; padding: 3px 0; color: var(--muted);
    font-family: 'SF Mono', Menlo, monospace; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; }
  .log-line .ts { color: var(--muted); margin-right: 8px; }
  .log-line .evt { color: var(--accent); margin-right: 8px;
    display: inline-block; min-width: 200px; }
  .log-line.error .evt { color: var(--red); }
  .log-line.success .evt { color: var(--green); }
  #graph-svg { width: 100%; height: 380px; }
  .node-strategy circle { fill: rgba(94,179,255,0.2); stroke: var(--accent); }
  .node-meta circle { fill: rgba(167,139,250,0.2); stroke: var(--purple); }
  .node-insight circle { fill: rgba(251,191,36,0.2); stroke: var(--yellow); }
  .node text { font-size: 9px; fill: var(--text); pointer-events: none; }
  .link { stroke: var(--border); stroke-opacity: 0.5; fill: none; }
  .ok { color: var(--green); }
  .warn { color: var(--yellow); }
  .bad { color: var(--red); }

  /* Anatomy tab styles */
  #anatomy-layout { display: grid; grid-template-columns: 2fr 1fr;
    gap: 16px; padding: 16px 24px; }
  #brain-svg-container { background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px; }
  #brain-svg { width: 100%; height: 580px; }
  .region { cursor: pointer; transition: filter 0.2s, opacity 0.2s;
    stroke-width: 2; }
  .region:hover { filter: brightness(1.4); }
  .region.dim { opacity: 0.35; }
  .region-label { fill: var(--text); font-size: 11px; font-weight: 500;
    pointer-events: none; text-anchor: middle; }
  .region-internal-label { fill: var(--text); font-size: 9px;
    pointer-events: none; text-anchor: middle; }
  .anatomy-side { display: flex; flex-direction: column; gap: 12px; }
  .anatomy-card { background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px; }
  .anatomy-card h3 { margin: 0 0 8px 0; font-size: 11px;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  #region-detail .empty { color: var(--muted); font-style: italic; }
  #region-detail .name { font-size: 15px; font-weight: 600;
    color: var(--accent); margin-bottom: 4px; }
  #region-detail .function { font-size: 12px; color: var(--text);
    margin-bottom: 10px; }
  #region-detail .modules-label { font-size: 10px; color: var(--muted);
    text-transform: uppercase; margin-bottom: 4px; }
  #region-detail .module { display: inline-block; padding: 2px 8px;
    background: rgba(94,179,255,0.1); border: 1px solid var(--border);
    border-radius: 12px; font-size: 11px; margin: 2px 4px 2px 0; }
  #region-detail .activity-bar { margin-top: 10px; }
  #region-detail .activity-bar .lbl { display: flex;
    justify-content: space-between; font-size: 11px; color: var(--muted);
    margin-bottom: 4px; }
  #region-detail .activity-bar .bg { height: 6px;
    background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; }
  #region-detail .activity-bar .fill { height: 100%;
    background: var(--accent); transition: width 0.3s; }
  #region-detail .detail-text { font-size: 11px; color: var(--muted);
    margin-top: 8px; font-style: italic; }
  .legend-item { display: flex; align-items: center; gap: 8px;
    padding: 4px 0; font-size: 11px; }
  .legend-swatch { width: 12px; height: 12px; border-radius: 50%;
    flex-shrink: 0; }
  .anatomy-disclaimer { font-size: 10px; color: var(--muted);
    padding: 12px; text-align: center; font-style: italic; }
</style>
</head>
<body>
<header>
  <h1>🧠 MARIS v9 — Brain Dashboard</h1>
  <span style="color: var(--muted); font-size: 11px;">
    refreshes every 5s · <span id="last-update">—</span>
  </span>
</header>

<div class="tabs">
  <div class="tab active" data-tab="cognitive">Cognitive Map</div>
  <div class="tab" data-tab="anatomy">Anatomy</div>
</div>

<!-- ── Cognitive Map tab ─────────────────────────────────────────────── -->
<div id="tab-cognitive" class="tab-content">
  <div class="grid">
    <div class="panel">
      <h2>System Stats</h2>
      <div id="metrics-stats" class="stat-grid"></div>
    </div>
    <div class="panel">
      <h2>Internal State</h2>
      <div id="state-bars"></div>
      <div id="state-meta" style="margin-top: 12px; font-size: 11px;
        color: var(--muted);"></div>
    </div>
    <div class="panel">
      <h2>Senate Verdicts</h2>
      <div id="senate-stats"></div>
    </div>
    <div class="panel wide">
      <h2>Knowledge Graph — strategies → meta-rules → insights</h2>
      <svg id="graph-svg"></svg>
      <div style="font-size: 11px; color: var(--muted); margin-top: 8px;">
        <span style="color: var(--accent);">●</span> strategies &nbsp;
        <span style="color: var(--purple);">●</span> meta-rules &nbsp;
        <span style="color: var(--yellow);">●</span> insights (eureka)
      </div>
    </div>
    <div class="panel">
      <h2>Recent Strategies</h2>
      <div id="strategies-list" class="strategy-list"></div>
    </div>
    <div class="panel">
      <h2>Meta-Rules + Insights</h2>
      <div id="metas-list" class="strategy-list"></div>
    </div>
    <div class="panel wide">
      <h2>Autonomous Activity Log</h2>
      <div id="auto-log" style="max-height: 260px; overflow-y: auto;"></div>
    </div>
  </div>
</div>

<!-- ── Anatomy tab ────────────────────────────────────────────────────── -->
<div id="tab-anatomy" class="tab-content" style="display: none;">
  <div id="anatomy-layout">
    <div id="brain-svg-container">
      <h2>Functional Brain Map</h2>
      <p style="font-size: 11px; color: var(--muted); margin: 0 0 8px 0;">
        Hover or tap a region to see its MARIS module(s).
        Saturation reflects current activity.
      </p>
      <svg id="brain-svg" viewBox="0 0 700 580"></svg>
      <div class="anatomy-disclaimer">
        Visualization metaphor only — MARIS modules don't literally map to
        neural anatomy. The mapping highlights functional correspondences.
      </div>
    </div>
    <div class="anatomy-side">
      <div class="anatomy-card" id="region-detail">
        <h3>Selected Region</h3>
        <div class="empty">Hover over a region to see details</div>
      </div>
      <div class="anatomy-card">
        <h3>Internal Emotional State</h3>
        <div id="anatomy-state-bars"></div>
      </div>
      <div class="anatomy-card">
        <h3>Legend</h3>
        <div id="anatomy-legend"></div>
      </div>
    </div>
  </div>
</div>

<script>
// ── Tab switching ───────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(t => {
  t.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.querySelectorAll('.tab-content').forEach(x => x.style.display = 'none');
    document.getElementById('tab-' + t.dataset.tab).style.display = '';
    if (t.dataset.tab === 'anatomy') refreshAnatomy();
  };
});

// ── Utility ─────────────────────────────────────────────────────────────
async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) return null;
  return await r.json();
}
function el(tag, props = {}, ...children) {
  const e = document.createElement(tag);
  Object.assign(e, props);
  for (const c of children) {
    if (c == null) continue;
    e.append(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return e;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[c]));
}

// ── Cognitive tab renderers (same as before) ───────────────────────────
function renderMetrics(m) {
  const c = document.getElementById('metrics-stats');
  c.innerHTML = '';
  const pairs = [
    ['Turns', m.turns], ['Strategies', m.strategies],
    ['Meta-rules', m.meta_rules], ['Insights', m.insights],
    ['Eurekas', m.eurekas], ['Consolidations', m.consolidations],
    ['Hallucinations', m.hallucinations_caught],
    ['Drive overrides', m.drive_overrides],
    ['From autonomous', m.autonomous_strategies],
    ['Accept rate', m.acceptance_rate + '%'],
  ];
  for (const [k, v] of pairs) {
    c.append(el('div', {className: 'metric'},
      el('span', {textContent: k, style: 'color: var(--muted);'}),
      el('span', {className: 'v', textContent: v})));
  }
}

function renderSenate(m) {
  const c = document.getElementById('senate-stats');
  c.innerHTML = '';
  c.append(el('div', {className: 'metric'},
    el('span', {style: 'color: var(--muted);'}, 'Accepted'),
    el('span', {className: 'v green', textContent: m.senate_accepted})));
  c.append(el('div', {className: 'metric'},
    el('span', {style: 'color: var(--muted);'}, 'Rejected'),
    el('span', {className: 'v red', textContent: m.senate_rejected})));
  const total = m.senate_accepted + m.senate_rejected;
  c.append(el('div', {className: 'metric'},
    el('span', {style: 'color: var(--muted);'}, 'Total verdicts'),
    el('span', {className: 'v', textContent: total})));
  c.append(el('div', {className: 'metric'},
    el('span', {style: 'color: var(--muted);'}, 'Accept rate'),
    el('span', {className: 'v yellow', textContent: m.acceptance_rate + '%'})));
  if (total === 0) {
    c.append(el('div', {style: 'margin-top: 12px; padding: 8px; ' +
      'background: rgba(251,191,36,0.08); border-radius: 4px; ' +
      'font-size: 11px; color: var(--yellow);',
      textContent: 'No verdicts in flight log yet. Have a conversation with MARIS to see real counts.'}));
  }
}

function renderState(s, containerId) {
  const bars = document.getElementById(containerId || 'state-bars');
  bars.innerHTML = '';
  const state = s.state || {};
  for (const [dim, val] of Object.entries(state)) {
    const v = parseFloat(val) || 0;
    const pct = Math.abs(v) * 50;
    bars.append(el('div', {className: 'bar-container'},
      el('span', {className: 'label', textContent: dim}),
      el('div', {className: 'bar-bg'},
        el('div', {
          className: 'bar-fill ' + (v >= 0 ? 'pos' : 'neg'),
          style: v >= 0 ? `left: 50%; width: ${pct}%;`
                       : `left: ${50 - pct}%; width: ${pct}%;`,
        })),
      el('span', {className: 'v', textContent: v.toFixed(2)})));
  }
  if (containerId !== 'anatomy-state-bars') {
    const meta = document.getElementById('state-meta');
    meta.innerHTML = '';
    if (s.dominant && s.dominant.emotion) {
      meta.append(el('div', {textContent:
        `Dominant: ${s.dominant.emotion} (intensity ${s.dominant.intensity})`}));
    }
    if (s.saved_at) {
      meta.append(el('div', {textContent: `Saved: ${s.saved_at}`}));
    }
  }
  if (!Object.keys(state).length && containerId !== 'anatomy-state-bars') {
    bars.append(el('div', {style: 'color: var(--yellow); padding: 8px;',
      textContent: 'No internal_state.json yet. Make sure v9 patch is applied.'}));
  }
}

function renderMemory(d) {
  const sl = document.getElementById('strategies-list');
  sl.innerHTML = '';
  const strategies = (d.strategies || []).slice(-20).reverse();
  if (!strategies.length) {
    sl.append(el('div', {style: 'color: var(--muted);',
      textContent: 'No strategies yet.'}));
  }
  for (const s of strategies) {
    const isAuto = (s.source === 'active_perception');
    const item = el('div', {className: 'strategy-item' + (isAuto ? ' autonomous' : '')});
    item.append(el('span', {
      className: 'tag' + (isAuto ? ' autonomous' : ''),
      textContent: isAuto ? 'AUTO' : (s.task_type || 'task'),
    }));
    if (s.consequence_result) {
      item.append(el('span', {
        className: 'tag ' + (s.consequence_result === 'pass' ? 'pass' : 'fail'),
        textContent: s.consequence_result.toUpperCase(),
      }));
    }
    if (s.mood && s.mood !== 'neutral') {
      item.append(el('span', {className: 'tag', textContent: s.mood}));
    }
    item.append(document.createTextNode((s.strategy || '').slice(0, 200)));
    sl.append(item);
  }
  const ml = document.getElementById('metas-list');
  ml.innerHTML = '';
  for (const m of d.metas || []) {
    const item = el('div', {className: 'strategy-item meta'});
    item.append(el('span', {className: 'tag meta',
      textContent: `META · ${m.confidence || '?'}%`}));
    item.append(el('span', {className: 'tag',
      textContent: `from ${m.source_count || '?'} exp`}));
    item.append(document.createTextNode(m.principle || ''));
    ml.append(item);
  }
  for (const ins of d.insights || []) {
    const item = el('div', {className: 'strategy-item insight'});
    item.append(el('span', {className: 'tag insight', textContent: 'EUREKA'}));
    if (ins.depth) {
      item.append(el('span', {className: 'tag', textContent: ins.depth}));
    }
    item.append(document.createTextNode(ins.principle || ''));
    ml.append(item);
  }
  if (!d.metas?.length && !d.insights?.length) {
    ml.append(el('div', {style: 'color: var(--muted);',
      textContent: 'No meta-rules or insights yet. Need more strategies first.'}));
  }
}

function renderAutoLog(events) {
  const c = document.getElementById('auto-log');
  c.innerHTML = '';
  const recent = (events || []).slice(-50).reverse();
  if (!recent.length) {
    c.append(el('div', {style: 'color: var(--muted);',
      textContent: 'No autonomous events yet. Start the daemon with /autonomous start.'}));
    return;
  }
  for (const e of recent) {
    const isError = (e.event || '').includes('failed') || (e.event || '').includes('error');
    const isSuccess = (e.event || '').includes('added') || (e.event || '').includes('pass') || (e.event || '').includes('success');
    const line = el('div', {className: 'log-line' + (isError ? ' error' : isSuccess ? ' success' : '')});
    line.append(el('span', {className: 'ts', textContent: (e.ts || '').slice(0, 19).replace('T', ' ')}));
    line.append(el('span', {className: 'evt', textContent: e.event || '?'}));
    const extras = [];
    for (const k of ['query', 'strategy', 'outcome', 'backend', 'count']) {
      if (e[k] != null) extras.push(`${k}=${String(e[k]).slice(0, 60)}`);
    }
    line.append(document.createTextNode(extras.join(' · ')));
    c.append(line);
  }
}

function renderGraph(g) {
  const svg = document.getElementById('graph-svg');
  svg.innerHTML = '';
  const W = svg.clientWidth, H = 380;
  if (!g.nodes?.length) {
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('x', W/2); t.setAttribute('y', H/2);
    t.setAttribute('text-anchor', 'middle'); t.setAttribute('fill', '#6b7596');
    t.textContent = 'Empty — strategies will appear here once MARIS has memory.';
    svg.append(t); return;
  }
  const byType = {strategy: [], meta: [], insight: []};
  for (const n of g.nodes) if (n.type in byType) byType[n.type].push(n);
  const cols = {strategy: W*0.18, meta: W*0.5, insight: W*0.82};
  const positions = {};
  for (const [type, list] of Object.entries(byType)) {
    list.forEach((n, i) => {
      positions[n.id] = {x: cols[type], y: 30 + (H - 60) * (i + 1) / (list.length + 1)};
    });
  }
  for (const e of g.edges || []) {
    const s = positions[e.source], t = positions[e.target];
    if (!s || !t) continue;
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', s.x); line.setAttribute('y1', s.y);
    line.setAttribute('x2', t.x); line.setAttribute('y2', t.y);
    line.setAttribute('class', 'link');
    svg.append(line);
  }
  for (const n of g.nodes) {
    const p = positions[n.id]; if (!p) continue;
    const grp = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    grp.setAttribute('class', `node node-${n.type}`);
    grp.setAttribute('transform', `translate(${p.x}, ${p.y})`);
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    const r = n.type === 'insight' ? 8 : n.type === 'meta' ? 7 : 5;
    circle.setAttribute('r', r); circle.setAttribute('stroke-width', 1.5);
    grp.append(circle);
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', n.type === 'insight' ? -12 : 12);
    text.setAttribute('y', 3);
    text.setAttribute('text-anchor', n.type === 'insight' ? 'end' : 'start');
    text.textContent = (n.label || '').slice(0, 40);
    grp.append(text);
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = n.label || n.id;
    grp.append(title);
    svg.append(grp);
  }
}

// ── ANATOMY: render the brain SVG ───────────────────────────────────────
// Region definitions: SVG paths for each brain region (side view, facing left)
const REGION_GEOMETRY = {
  // ─ Outer cortex ─
  prefrontal_cortex: {
    type: 'path',
    d: 'M 110,210 Q 60,160 90,110 Q 140,60 230,80 Q 280,110 270,180 Q 250,230 200,240 Q 150,245 110,210 Z',
    labelX: 175, labelY: 160,
  },
  parietal_cortex: {
    type: 'path',
    d: 'M 270,180 Q 280,110 360,90 Q 440,80 470,140 Q 480,200 430,230 Q 350,250 270,180 Z',
    labelX: 370, labelY: 165,
  },
  occipital_cortex: {
    type: 'path',
    d: 'M 470,140 Q 540,150 560,210 Q 555,275 500,290 Q 450,290 430,230 Q 425,180 470,140 Z',
    labelX: 500, labelY: 220,
  },
  temporal_cortex: {
    type: 'path',
    d: 'M 130,240 Q 180,270 240,275 Q 300,270 350,285 Q 380,310 360,340 Q 290,365 200,355 Q 130,335 115,290 Q 110,260 130,240 Z',
    labelX: 240, labelY: 318,
  },
  cerebellum: {
    type: 'path',
    d: 'M 430,300 Q 460,290 490,300 Q 525,310 535,355 Q 525,400 480,410 Q 430,405 415,365 Q 410,325 430,300 Z',
    labelX: 475, labelY: 360,
  },
  brainstem: {
    type: 'path',
    d: 'M 450,400 Q 455,420 460,460 Q 465,500 478,530 Q 470,540 455,540 Q 442,510 440,470 Q 437,430 442,402 Z',
    labelX: 458, labelY: 480,
  },

  // ─ Inner structures ─
  default_mode_network: {
    type: 'ellipse',
    cx: 300, cy: 175, rx: 60, ry: 18,
    labelX: 300, labelY: 178, internal: true,
  },
  anterior_cingulate: {
    type: 'path',
    d: 'M 165,170 Q 220,140 280,150 Q 285,165 275,175 Q 215,185 165,170 Z',
    labelX: 220, labelY: 165, internal: true,
  },
  thalamus: {
    type: 'ellipse',
    cx: 330, cy: 210, rx: 24, ry: 16,
    labelX: 330, labelY: 213, internal: true,
  },
  hippocampus: {
    type: 'path',
    d: 'M 220,305 Q 260,295 305,305 Q 295,325 250,330 Q 215,322 220,305 Z',
    labelX: 260, labelY: 315, internal: true,
  },
  amygdala: {
    type: 'ellipse',
    cx: 200, cy: 308, rx: 14, ry: 10,
    labelX: 200, labelY: 311, internal: true,
  },
  insula: {
    type: 'ellipse',
    cx: 270, cy: 260, rx: 26, ry: 12,
    labelX: 270, labelY: 263, internal: true,
  },
};

function svgEl(name, attrs = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function renderAnatomy(data) {
  const svg = document.getElementById('brain-svg');
  svg.innerHTML = '';

  // Background subtle outline of full brain
  const bgOutline = svgEl('path', {
    d: 'M 100,200 Q 55,140 110,80 Q 200,40 320,60 Q 450,70 520,140 Q 555,210 525,290 Q 510,360 470,400 Q 450,440 460,500 Q 460,530 445,540 L 90,540 Q 60,440 80,360 Q 55,300 75,250 Q 80,220 100,200',
    fill: 'rgba(255,255,255,0.02)',
    stroke: 'rgba(255,255,255,0.08)',
    'stroke-width': 1.5,
    'stroke-dasharray': '3,4',
  });
  svg.append(bgOutline);

  // Render each region
  const regions = data.regions || {};
  // Render external regions first, internal on top
  const ordered = Object.entries(regions).sort((a, b) => {
    const aInt = REGION_GEOMETRY[a[0]]?.internal ? 1 : 0;
    const bInt = REGION_GEOMETRY[b[0]]?.internal ? 1 : 0;
    return aInt - bInt;
  });

  for (const [key, region] of ordered) {
    const geom = REGION_GEOMETRY[key];
    if (!geom) continue;
    const activity = region.activity || 0;
    // Saturation/opacity based on activity; minimum visible
    const fillOpacity = 0.15 + 0.55 * activity;
    const strokeOpacity = 0.4 + 0.6 * activity;
    
    let shape;
    if (geom.type === 'ellipse') {
      shape = svgEl('ellipse', {
        cx: geom.cx, cy: geom.cy, rx: geom.rx, ry: geom.ry,
      });
    } else {
      shape = svgEl('path', { d: geom.d });
    }
    shape.setAttribute('class', 'region' + (geom.internal ? ' internal' : ''));
    shape.setAttribute('fill', region.color);
    shape.setAttribute('fill-opacity', fillOpacity);
    shape.setAttribute('stroke', region.color);
    shape.setAttribute('stroke-opacity', strokeOpacity);
    shape.setAttribute('data-key', key);
    shape.addEventListener('mouseenter', () => showRegionDetail(key, region));
    shape.addEventListener('click', () => showRegionDetail(key, region));
    svg.append(shape);

    // Label
    const label = svgEl('text', {
      x: geom.labelX, y: geom.labelY,
      class: geom.internal ? 'region-internal-label' : 'region-label',
    });
    label.textContent = region.label.split('(')[0].trim()
      .replace(' Cortex', '').replace(' Network', '');
    svg.append(label);
  }
}

function showRegionDetail(key, region) {
  const c = document.getElementById('region-detail');
  c.innerHTML = '<h3>Selected Region</h3>';
  const name = el('div', {className: 'name', textContent: region.label});
  name.style.color = region.color;
  c.append(name);
  c.append(el('div', {className: 'function', textContent: region.function}));
  c.append(el('div', {className: 'modules-label', textContent: 'MARIS modules'}));
  const modBox = el('div');
  for (const mod of region.modules) {
    modBox.append(el('span', {className: 'module', textContent: mod}));
  }
  c.append(modBox);

  const bar = el('div', {className: 'activity-bar'});
  bar.append(el('div', {className: 'lbl'},
    el('span', {textContent: 'Activity'}),
    el('span', {textContent: (region.activity * 100).toFixed(0) + '%'})));
  const bg = el('div', {className: 'bg'});
  const fill = el('div', {className: 'fill',
    style: `width: ${region.activity * 100}%; background: ${region.color};`});
  bg.append(fill); bar.append(bg);
  c.append(bar);

  if (region.detail) {
    c.append(el('div', {className: 'detail-text', textContent: region.detail}));
  }
}

function renderAnatomyLegend(data) {
  const c = document.getElementById('anatomy-legend');
  c.innerHTML = '';
  const groups = {
    'Cortical': ['prefrontal_cortex', 'parietal_cortex', 'temporal_cortex',
                  'occipital_cortex', 'anterior_cingulate', 'default_mode_network'],
    'Deep brain': ['hippocampus', 'amygdala', 'insula', 'thalamus'],
    'Autonomic': ['cerebellum', 'brainstem'],
  };
  for (const [group, keys] of Object.entries(groups)) {
    c.append(el('div', {style: 'font-size: 10px; color: var(--muted); ' +
      'text-transform: uppercase; margin-top: 8px; margin-bottom: 4px;',
      textContent: group}));
    for (const k of keys) {
      const region = data.regions[k];
      if (!region) continue;
      const item = el('div', {className: 'legend-item'});
      item.append(el('div', {className: 'legend-swatch',
        style: `background: ${region.color};`}));
      item.append(el('span', {textContent: region.label.split('(')[0].trim()}));
      item.style.cursor = 'pointer';
      item.onmouseenter = () => showRegionDetail(k, region);
      c.append(item);
    }
  }
}

async function refreshAnatomy() {
  const data = await fetchJSON('/api/anatomy');
  if (!data) return;
  renderAnatomy(data);
  renderAnatomyLegend(data);
  const state = await fetchJSON('/api/state');
  if (state) renderState(state, 'anatomy-state-bars');
}

// ── Master refresh ──────────────────────────────────────────────────────
async function refresh() {
  const activeTab = document.querySelector('.tab.active').dataset.tab;
  if (activeTab === 'cognitive') {
    const [m, s, mem, auto, g] = await Promise.all([
      fetchJSON('/api/metrics'), fetchJSON('/api/state'),
      fetchJSON('/api/memory'), fetchJSON('/api/autonomous'),
      fetchJSON('/api/graph'),
    ]);
    if (m) { renderMetrics(m); renderSenate(m); }
    if (s) renderState(s);
    if (mem) renderMemory(mem);
    if (auto) renderAutoLog(auto.events);
    if (g) renderGraph(g);
  } else {
    await refreshAnatomy();
  }
  document.getElementById('last-update').textContent =
    new Date().toLocaleTimeString();
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  MARIS v9 Brain Dashboard (tabbed: Cognitive + Anatomy)")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  http://localhost:{PORT}\n")
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)
