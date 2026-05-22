"""
dashboard.py — MARIS Brain Dashboard
======================================
Run: python dashboard.py
Open: http://localhost:3000

Reads MARIS's JSON files in real-time and visualizes:
- Brain cortex map with development levels
- 3-tier knowledge hierarchy
- Emotional state trajectory
- Development stage progression
- Learning metrics
- Search across all knowledge
"""

import http.server
import json
import os
import socketserver

PORT = 3000
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def read_json(filename):
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def get_maris_data():
    strategies = read_json("strategy_memory.json")
    meta_rules = read_json("meta_strategies.json")
    insights = read_json("insights.json")
    metrics = read_json("progression_metrics.json")
    internal = read_json("internal_state.json")

    n_strat = len(strategies) if isinstance(strategies, list) else 0
    n_meta = len(meta_rules) if isinstance(meta_rules, list) else 0
    n_insights = len(insights) if isinstance(insights, list) else 0

    if n_strat >= 100 and n_meta >= 20: stage = 4
    elif n_strat >= 50 and n_meta >= 10: stage = 3
    elif n_strat >= 15 and n_meta >= 3: stage = 2
    elif n_strat >= 5 and n_meta >= 1: stage = 1
    else: stage = 0

    stage_names = ["INFANT", "CHILD", "STUDENT", "GRADUATE", "EXPERT"]

    return json.dumps({
        "strategies": strategies if isinstance(strategies, list) else [],
        "meta_rules": meta_rules if isinstance(meta_rules, list) else [],
        "insights": insights if isinstance(insights, list) else [],
        "metrics": metrics if isinstance(metrics, dict) else {},
        "internal_state": internal if isinstance(internal, dict) else {},
        "stage": stage,
        "stage_name": stage_names[stage],
        "counts": {"strategies": n_strat, "meta_rules": n_meta, "insights": n_insights},
    })

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MARIS — Brain Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0B0F19; color: #E5E7EB; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; overflow-x: hidden; }
a { color: #60A5FA; text-decoration: none; }

.header { padding: 20px 24px; border-bottom: 1px solid #1F2937; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 20px; font-weight: 600; letter-spacing: -0.02em; }
.header h1 span { color: #6B7280; font-weight: 400; font-size: 13px; margin-left: 8px; }
.stage-badge { padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 500; background: #1E293B; border: 1px solid #334155; }

.tabs { display: flex; gap: 2px; padding: 12px 24px; border-bottom: 1px solid #1F2937; }
.tab { padding: 8px 16px; border-radius: 6px; font-size: 13px; cursor: pointer; border: none; background: transparent; color: #6B7280; transition: all 0.2s; }
.tab.active { background: #1E293B; color: #E5E7EB; border: 1px solid #334155; }
.tab:hover { color: #E5E7EB; }

.content { padding: 24px; max-width: 1200px; margin: 0 auto; }

/* Stats bar */
.stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 24px; }
.stat { background: #111827; border: 1px solid #1F2937; border-radius: 10px; padding: 16px; }
.stat-label { font-size: 11px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-value { font-size: 28px; font-weight: 600; margin-top: 4px; }
.stat-sub { font-size: 11px; color: #6B7280; margin-top: 2px; }

/* Brain SVG */
.brain-container { position: relative; }
.brain-container svg { width: 100%; max-height: 460px; }
.cortex-circle { cursor: pointer; transition: all 0.3s; }
.cortex-circle:hover { filter: brightness(1.3); }
@keyframes pulse { 0%,100%{opacity:0.6} 50%{opacity:1} }
.pulse { animation: pulse 2.5s ease-in-out infinite; }
@keyframes glow { 0%,100%{filter:drop-shadow(0 0 4px #C084FC)} 50%{filter:drop-shadow(0 0 18px #C084FC)} }
.eureka-glow { animation: glow 2s ease-in-out infinite; }

/* Detail panel */
.detail { background: #111827; border: 1px solid #1F2937; border-radius: 10px; padding: 16px; margin-top: 16px; display: none; }
.detail.active { display: block; }
.detail h3 { font-size: 15px; margin-bottom: 8px; }
.detail p { font-size: 13px; color: #9CA3AF; line-height: 1.5; }
.module-tag { display: inline-block; padding: 2px 8px; margin: 2px; border-radius: 4px; font-size: 11px; font-family: monospace; }

/* Knowledge list */
.knowledge-item { padding: 10px 14px; margin: 6px 0; border-radius: 6px; background: #111827; border-left: 3px solid; font-size: 13px; line-height: 1.5; }
.tier-header { font-size: 14px; font-weight: 500; margin: 20px 0 8px; display: flex; align-items: center; gap: 8px; }

/* Dev bars */
.dev-bar { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
.dev-label { width: 110px; font-size: 12px; color: #9CA3AF; font-family: monospace; }
.dev-track { flex: 1; height: 8px; background: #1F2937; border-radius: 4px; overflow: hidden; }
.dev-fill { height: 100%; border-radius: 4px; transition: width 0.8s ease; }
.dev-val { width: 36px; text-align: right; font-size: 11px; color: #6B7280; font-family: monospace; }

/* Feelings */
.feeling-bar { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.feeling-label { width: 100px; font-size: 13px; color: #9CA3AF; }
.feeling-track { flex: 1; height: 12px; background: #1F2937; border-radius: 6px; overflow: hidden; position: relative; }
.feeling-fill { height: 100%; border-radius: 6px; transition: width 0.5s; }
.feeling-center { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #374151; }

/* Search */
.search-input { width: 100%; padding: 10px 14px; background: #111827; border: 1px solid #1F2937; border-radius: 8px; color: #E5E7EB; font-size: 14px; outline: none; margin-bottom: 16px; }
.search-input:focus { border-color: #334155; }
.search-empty { text-align: center; padding: 40px; color: #4B5563; font-size: 13px; }

/* Stage progress */
.stage-track { display: flex; align-items: center; gap: 0; margin: 16px 0; }
.stage-dot { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 13px; border: 2px solid #1F2937; }
.stage-dot.active { border-color: #60A5FA; }
.stage-dot.done { border-color: #2DD4A8; }
.stage-line { flex: 1; height: 2px; background: #1F2937; }
.stage-line.done { background: #2DD4A8; }

/* Responsive */
@media (max-width: 768px) { .stats { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>

<div class="header">
  <h1>MARIS <span>cognitive dashboard</span></h1>
  <div>
    <span class="stage-badge" id="stage-badge">Loading...</span>
    <button onclick="loadData()" style="margin-left: 8px; padding: 4px 12px; background: #1E293B; border: 1px solid #334155; border-radius: 6px; color: #9CA3AF; font-size: 12px; cursor: pointer;">↻ Refresh</button>
  </div>
</div>

<div class="tabs">
  <button class="tab active" onclick="showTab('brain')">🧠 Brain</button>
  <button class="tab" onclick="showTab('knowledge')">📊 Knowledge</button>
  <button class="tab" onclick="showTab('feelings')">💜 Feelings</button>
  <button class="tab" onclick="showTab('metrics')">📈 Metrics</button>
  <button class="tab" onclick="showTab('search')">🔍 Search</button>
</div>

<div class="content">
  <div class="stats">
    <div class="stat"><div class="stat-label">Strategies</div><div class="stat-value" id="n-strat" style="color:#FBBF24">0</div><div class="stat-sub">Tier 1 — Episodic</div></div>
    <div class="stat"><div class="stat-label">Meta-rules</div><div class="stat-value" id="n-meta" style="color:#2DD4A8">0</div><div class="stat-sub">Tier 2 — Semantic</div></div>
    <div class="stat"><div class="stat-label">Insights</div><div class="stat-value" id="n-insights" style="color:#C084FC">0</div><div class="stat-sub">Tier 3 — Eureka</div></div>
    <div class="stat"><div class="stat-label">Stage</div><div class="stat-value" id="stage-num" style="color:#60A5FA">0</div><div class="stat-sub" id="stage-name">INFANT</div></div>
    <div class="stat"><div class="stat-label">Tasks</div><div class="stat-value" id="n-tasks" style="color:#E5E7EB">0</div><div class="stat-sub">Total processed</div></div>
  </div>

  <!-- BRAIN TAB -->
  <div id="tab-brain">
    <div class="stage-track" id="stage-track"></div>
    <div class="brain-container">
      <svg viewBox="0 0 720 460">
        <ellipse cx="360" cy="230" rx="320" ry="210" fill="none" stroke="#1F2937" stroke-width="1" stroke-dasharray="3 5"/>
        <!-- Connections -->
        <line x1="190" y1="130" x2="280" y2="110" stroke="#2DD4A8" stroke-width="1" opacity="0.25"/>
        <line x1="530" y1="130" x2="440" y2="110" stroke="#60A5FA" stroke-width="1" opacity="0.25"/>
        <line x1="360" y1="170" x2="360" y2="220" stroke="#A78BFA" stroke-width="1.5" opacity="0.3"/>
        <line x1="280" y1="270" x2="190" y2="310" stroke="#F87171" stroke-width="1" opacity="0.25"/>
        <line x1="440" y1="270" x2="530" y2="310" stroke="#F87171" stroke-width="1" opacity="0.25"/>
        <line x1="190" y1="340" x2="280" y2="270" stroke="#FBBF24" stroke-width="1" opacity="0.2" stroke-dasharray="3 3"/>
        <line x1="530" y1="340" x2="440" y2="270" stroke="#F472B6" stroke-width="1" opacity="0.2" stroke-dasharray="3 3"/>

        <!-- Perception -->
        <g class="cortex-circle" onclick="selectCortex('perception')">
          <circle cx="190" cy="130" r="60" fill="#2DD4A8" fill-opacity="0.08" stroke="#2DD4A8" stroke-width="1.5" class="pulse"/>
          <circle id="ring-perception" cx="190" cy="130" r="68" fill="none" stroke="#1F2937" stroke-width="3"/>
          <circle id="dev-perception" cx="190" cy="130" r="68" fill="none" stroke="#2DD4A8" stroke-width="3" stroke-dasharray="0 999" stroke-dashoffset="107" stroke-linecap="round"/>
          <text x="190" y="126" text-anchor="middle" fill="#E5E7EB" font-size="13" font-weight="600">Perception</text>
          <text x="190" y="144" text-anchor="middle" fill="#9CA3AF" font-size="10" id="pct-perception">0%</text>
        </g>

        <!-- Cognition (center, largest) -->
        <g class="cortex-circle" onclick="selectCortex('cognition')">
          <circle cx="360" cy="105" r="75" fill="#A78BFA" fill-opacity="0.1" stroke="#A78BFA" stroke-width="1.5" class="pulse"/>
          <circle id="ring-cognition" cx="360" cy="105" r="83" fill="none" stroke="#1F2937" stroke-width="3"/>
          <circle id="dev-cognition" cx="360" cy="105" r="83" fill="none" stroke="#A78BFA" stroke-width="3" stroke-dasharray="0 999" stroke-dashoffset="130" stroke-linecap="round"/>
          <text x="360" y="101" text-anchor="middle" fill="#E5E7EB" font-size="14" font-weight="600">Cognition</text>
          <text x="360" y="119" text-anchor="middle" fill="#9CA3AF" font-size="10" id="pct-cognition">0%</text>
        </g>

        <!-- Dialogue -->
        <g class="cortex-circle" onclick="selectCortex('dialogue')">
          <circle cx="530" cy="130" r="55" fill="#60A5FA" fill-opacity="0.08" stroke="#60A5FA" stroke-width="1.5"/>
          <circle id="ring-dialogue" cx="530" cy="130" r="63" fill="none" stroke="#1F2937" stroke-width="3"/>
          <circle id="dev-dialogue" cx="530" cy="130" r="63" fill="none" stroke="#60A5FA" stroke-width="3" stroke-dasharray="0 999" stroke-dashoffset="99" stroke-linecap="round"/>
          <text x="530" y="126" text-anchor="middle" fill="#E5E7EB" font-size="13" font-weight="600">Dialogue</text>
          <text x="530" y="144" text-anchor="middle" fill="#9CA3AF" font-size="10" id="pct-dialogue">0%</text>
        </g>

        <!-- Evaluation -->
        <g class="cortex-circle" onclick="selectCortex('evaluation')">
          <circle cx="360" cy="260" r="65" fill="#F87171" fill-opacity="0.08" stroke="#F87171" stroke-width="1.5" class="pulse"/>
          <circle id="ring-evaluation" cx="360" cy="260" r="73" fill="none" stroke="#1F2937" stroke-width="3"/>
          <circle id="dev-evaluation" cx="360" cy="260" r="73" fill="none" stroke="#F87171" stroke-width="3" stroke-dasharray="0 999" stroke-dashoffset="115" stroke-linecap="round"/>
          <text x="360" y="256" text-anchor="middle" fill="#E5E7EB" font-size="13" font-weight="600">Evaluation</text>
          <text x="360" y="274" text-anchor="middle" fill="#9CA3AF" font-size="10" id="pct-evaluation">0%</text>
        </g>

        <!-- Memory -->
        <g class="cortex-circle" onclick="selectCortex('memory')">
          <circle cx="170" cy="340" r="62" fill="#FBBF24" fill-opacity="0.08" stroke="#FBBF24" stroke-width="1.5"/>
          <circle id="ring-memory" cx="170" cy="340" r="70" fill="none" stroke="#1F2937" stroke-width="3"/>
          <circle id="dev-memory" cx="170" cy="340" r="70" fill="none" stroke="#FBBF24" stroke-width="3" stroke-dasharray="0 999" stroke-dashoffset="110" stroke-linecap="round"/>
          <text x="170" y="336" text-anchor="middle" fill="#E5E7EB" font-size="13" font-weight="600">Memory</text>
          <text x="170" y="354" text-anchor="middle" fill="#9CA3AF" font-size="10" id="pct-memory">0%</text>
        </g>

        <!-- Self -->
        <g class="cortex-circle" onclick="selectCortex('self')">
          <circle cx="550" cy="340" r="58" fill="#F472B6" fill-opacity="0.08" stroke="#F472B6" stroke-width="1.5"/>
          <circle id="ring-self" cx="550" cy="340" r="66" fill="none" stroke="#1F2937" stroke-width="3"/>
          <circle id="dev-self" cx="550" cy="340" r="66" fill="none" stroke="#F472B6" stroke-width="3" stroke-dasharray="0 999" stroke-dashoffset="104" stroke-linecap="round"/>
          <text x="550" y="336" text-anchor="middle" fill="#E5E7EB" font-size="13" font-weight="600">Self</text>
          <text x="550" y="354" text-anchor="middle" fill="#9CA3AF" font-size="10" id="pct-self">0%</text>
        </g>

        <!-- Eureka star -->
        <g class="eureka-glow" id="eureka-star" style="display:none">
          <circle cx="360" cy="30" r="14" fill="#1a0a2e" stroke="#C084FC" stroke-width="1.5"/>
          <text x="360" y="35" text-anchor="middle" font-size="14">✦</text>
        </g>
      </svg>
    </div>

    <div class="detail" id="detail-perception"><h3 style="color:#2DD4A8">Perception Cortex</h3><p>Detects user emotion (6 types: frustrated, anxious, sad, confused, happy, neutral), classifies task type (7 categories), analyzes complexity with depth detection, and identifies uncertainty/ambiguity.</p><div style="margin-top:8px" id="modules-perception"></div></div>
    <div class="detail" id="detail-cognition"><h3 style="color:#A78BFA">Cognition Cortex</h3><p>The thinking core. Inner Monologue deliberates privately before each response — challenging first instincts, checking blind spots, arriving at considered positions. Reasoning generates guided by deliberation. Reflection scores on 4 dimensions. Improvement rewrites targeting specific weaknesses.</p><div style="margin-top:8px" id="modules-cognition"></div></div>
    <div class="detail" id="detail-dialogue"><h3 style="color:#60A5FA">Dialogue Cortex</h3><p>Mood-adapted clarification before answering (fewer questions for frustrated users, more for confused). Mid-loop interrupts where MARIS pauses self-improvement to ask the human. Proactive questioning based on developmental stage.</p><div style="margin-top:8px" id="modules-dialogue"></div></div>
    <div class="detail" id="detail-evaluation"><h3 style="color:#F87171">Evaluation Cortex</h3><p>Senate: 3-judge panel (accuracy, tone, depth) with task-adaptive weights. Code tasks weight accuracy 50%; emotional tasks weight tone 50%. Hallucination Probe forces meta-cognition self-examination on every response.</p><div style="margin-top:8px" id="modules-evaluation"></div></div>
    <div class="detail" id="detail-memory"><h3 style="color:#FBBF24">Memory Cortex</h3><p>Three-tier knowledge: Tier 1 strategies (episodic, specific experiences), Tier 2 meta-rules (semantic, consolidated principles), Tier 3 insights (Eureka moments from connecting meta-rule pairs). Mirrors human episodic → semantic memory consolidation.</p><div style="margin-top:8px" id="modules-memory"></div></div>
    <div class="detail" id="detail-self"><h3 style="color:#F472B6">Self Cortex</h3><p>Internal emotional state driven by system events (not user mood). Six dimensions: frustration, satisfaction, curiosity, anxiety, excitement, warmth. Development tracking through 5 stages. Autonomous actions (terminal colors, pauses, self-notes).</p><div style="margin-top:8px" id="modules-self"></div></div>

    <!-- Module activity -->
    <div style="margin-top:20px; background:#111827; border:1px solid #1F2937; border-radius:10px; padding:16px;">
      <h3 style="font-size:14px; color:#9CA3AF; margin-bottom:10px; font-family:monospace;">Module activity</h3>
      <div id="module-bars"></div>
    </div>
  </div>

  <!-- KNOWLEDGE TAB -->
  <div id="tab-knowledge" style="display:none">
    <div id="knowledge-content"></div>
  </div>

  <!-- FEELINGS TAB -->
  <div id="tab-feelings" style="display:none">
    <h3 style="margin-bottom:16px;">MARIS Internal Emotional State</h3>
    <div id="feelings-content"></div>
  </div>

  <!-- METRICS TAB -->
  <div id="tab-metrics" style="display:none">
    <div id="metrics-content"></div>
  </div>

  <!-- SEARCH TAB -->
  <div id="tab-search" style="display:none">
    <input type="text" class="search-input" id="search-box" placeholder="Search strategies, meta-rules, insights..." oninput="doSearch()">
    <div id="search-results"><div class="search-empty">Type to search across all knowledge tiers</div></div>
  </div>
</div>

<script>
let DATA = null;

async function loadData() {
  try {
    const resp = await fetch('/api/data');
    DATA = await resp.json();
    render();
  } catch(e) { console.error('Failed to load data:', e); }
}

function render() {
  if (!DATA) return;
  const {strategies, meta_rules, insights, metrics, internal_state, stage, stage_name, counts} = DATA;

  // Stats
  document.getElementById('n-strat').textContent = counts.strategies;
  document.getElementById('n-meta').textContent = counts.meta_rules;
  document.getElementById('n-insights').textContent = counts.insights;
  document.getElementById('stage-num').textContent = stage;
  document.getElementById('stage-name').textContent = stage_name;
  document.getElementById('stage-badge').textContent = `Stage ${stage}: ${stage_name}`;
  document.getElementById('n-tasks').textContent = metrics.total_tasks || 0;

  // Eureka star
  document.getElementById('eureka-star').style.display = counts.insights > 0 ? 'block' : 'none';

  // Development levels per cortex (computed from data)
  const devLevels = {
    perception: Math.min(1, 0.4 + counts.strategies * 0.04),
    cognition: Math.min(1, 0.3 + counts.strategies * 0.05 + counts.meta_rules * 0.1),
    dialogue: Math.min(1, 0.2 + counts.strategies * 0.03),
    evaluation: Math.min(1, 0.3 + counts.strategies * 0.04 + counts.meta_rules * 0.08),
    memory: Math.min(1, counts.strategies * 0.04 + counts.meta_rules * 0.1 + counts.insights * 0.15),
    self: Math.min(1, 0.1 + counts.meta_rules * 0.05 + counts.insights * 0.1),
  };

  Object.entries(devLevels).forEach(([id, val]) => {
    const el = document.getElementById('dev-' + id);
    if (el) {
      const r = parseFloat(el.getAttribute('r'));
      const circ = 2 * Math.PI * r;
      el.setAttribute('stroke-dasharray', `${circ * val} ${circ * (1-val)}`);
    }
    const pct = document.getElementById('pct-' + id);
    if (pct) pct.textContent = Math.round(val * 100) + '%';
  });

  // Stage track
  const stages = ['👶 INFANT','🧒 CHILD','📚 STUDENT','🎓 GRADUATE','🧠 EXPERT'];
  document.getElementById('stage-track').innerHTML = stages.map((s,i) =>
    `<div class="stage-dot ${i===stage?'active':''} ${i<stage?'done':''}">${s.split(' ')[0]}</div>` +
    (i < stages.length-1 ? `<div class="stage-line ${i<stage?'done':''}"></div>` : '')
  ).join('');

  // Module activity bars
  const modules = [
    {label:'Monologue', v:devLevels.cognition, c:'#A78BFA'},
    {label:'Senate', v:devLevels.evaluation, c:'#F87171'},
    {label:'Emotion', v:devLevels.perception, c:'#2DD4A8'},
    {label:'Consolidate', v:devLevels.memory * 0.9, c:'#FBBF24'},
    {label:'Insight', v:counts.insights > 0 ? 0.5 : 0.05, c:'#C084FC'},
    {label:'Self-state', v:devLevels.self, c:'#F472B6'},
    {label:'Proactive', v:devLevels.dialogue * 0.5, c:'#60A5FA'},
    {label:'Autonomous', v:devLevels.self * 0.4, c:'#F472B6'},
  ];
  document.getElementById('module-bars').innerHTML = modules.map(m =>
    `<div class="dev-bar"><span class="dev-label">${m.label}</span><div class="dev-track"><div class="dev-fill" style="width:${m.v*100}%;background:${m.c}"></div></div><span class="dev-val">${Math.round(m.v*100)}%</span></div>`
  ).join('');

  // Cortex module tags
  const cortexModules = {
    perception: [{n:'EmotionModule',c:'#2DD4A8'},{n:'TaskTypeDetector',c:'#2DD4A8'},{n:'ComplexityRouter',c:'#2DD4A8'},{n:'UncertaintyDetector',c:'#2DD4A8'}],
    cognition: [{n:'InnerMonologue',c:'#A78BFA'},{n:'ReasoningModule',c:'#A78BFA'},{n:'ReflectionModule',c:'#A78BFA'},{n:'ImprovementModule',c:'#A78BFA'}],
    dialogue: [{n:'ClarificationModule',c:'#60A5FA'},{n:'MidLoopClarifier',c:'#60A5FA'},{n:'ProactiveModule',c:'#60A5FA'}],
    evaluation: [{n:'Senate',c:'#F87171'},{n:'HallucinationProbe',c:'#F87171'},{n:'JudgeModule',c:'#F87171'}],
    memory: [{n:'StrategyMemory',c:'#FBBF24'},{n:'ConsolidationEngine',c:'#FBBF24'},{n:'InsightDetector',c:'#FBBF24'}],
    self: [{n:'InternalState',c:'#F472B6'},{n:'DevelopmentTracker',c:'#F472B6'},{n:'AutonomousAction',c:'#F472B6'}],
  };
  Object.entries(cortexModules).forEach(([id, mods]) => {
    const el = document.getElementById('modules-' + id);
    if (el) el.innerHTML = mods.map(m => `<span class="module-tag" style="background:${m.c}15;border:1px solid ${m.c}30;color:${m.c}">${m.n}</span>`).join('');
  });

  // Knowledge tab
  let khtml = '';
  if (insights.length) {
    khtml += `<div class="tier-header" style="color:#C084FC">✦ Tier 3 — Insights (${insights.length})</div>`;
    insights.forEach(i => {
      khtml += `<div class="knowledge-item" style="border-color:#C084FC;background:#1a0a2e"><strong>${i.principle || i.text || 'Insight'}</strong>`;
      if (i.source_rules) khtml += `<div style="font-size:11px;color:#9CA3AF;margin-top:6px">From: ${(i.source_rules[0]||'').slice(0,70)}...<br>And: ${(i.source_rules[1]||'').slice(0,70)}...</div>`;
      khtml += `</div>`;
    });
  }
  if (meta_rules.length) {
    khtml += `<div class="tier-header" style="color:#2DD4A8">Tier 2 — Meta-rules (${meta_rules.length})</div>`;
    meta_rules.forEach(m => {
      khtml += `<div class="knowledge-item" style="border-color:#2DD4A8">${m.principle || m.text || JSON.stringify(m).slice(0,100)}<span style="color:#2DD4A8;font-size:11px;margin-left:8px">${m.confidence||'?'}%</span></div>`;
    });
  }
  if (strategies.length) {
    khtml += `<div class="tier-header" style="color:#FBBF24">Tier 1 — Strategies (${strategies.length})</div>`;
    strategies.forEach(s => {
      khtml += `<div class="knowledge-item" style="border-color:#FBBF24"><span style="color:#6B7280;font-size:11px">[${s.mood||'?'}/${s.task_type||'?'}]</span> ${(s.strategy||'').slice(0,120)}</div>`;
    });
  }
  if (!khtml) khtml = '<div class="search-empty">No knowledge yet. Talk to MARIS to start learning.</div>';
  document.getElementById('knowledge-content').innerHTML = khtml;

  // Feelings tab
  const dims = ['frustration','satisfaction','curiosity','anxiety','excitement','warmth'];
  const colors = {frustration:'#F87171',satisfaction:'#2DD4A8',curiosity:'#60A5FA',anxiety:'#FBBF24',excitement:'#C084FC',warmth:'#F472B6'};
  const state = internal_state.state || {};
  let fhtml = dims.map(d => {
    const val = state[d] || 0;
    const pct = 50 + val * 50;
    return `<div class="feeling-bar"><span class="feeling-label" style="color:${colors[d]}">${d}</span><div class="feeling-track"><div class="feeling-center"></div><div class="feeling-fill" style="width:${pct}%;background:${colors[d]};opacity:${0.3+Math.abs(val)*0.7}"></div></div><span style="width:50px;text-align:right;font-size:12px;font-family:monospace;color:${colors[d]}">${val >= 0 ? '+' : ''}${val.toFixed(2)}</span></div>`;
  }).join('');
  const dominant = internal_state.dominant || ['neutral', 0];
  fhtml += `<div style="margin-top:16px;padding:12px;background:#111827;border-radius:8px;font-size:13px">Dominant: <strong style="color:${colors[dominant[0]]||'#9CA3AF'}">${dominant[0]}</strong> (${typeof dominant[1]==='number'?dominant[1].toFixed(3):dominant[1]})</div>`;
  document.getElementById('feelings-content').innerHTML = fhtml;

  // Metrics tab
  const m = metrics || {};
  document.getElementById('metrics-content').innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
      <div class="stat"><div class="stat-label">Tasks</div><div class="stat-value">${m.total_tasks||0}</div></div>
      <div class="stat"><div class="stat-label">Accepted</div><div class="stat-value" style="color:#2DD4A8">${m.improvements_accepted||0}</div></div>
      <div class="stat"><div class="stat-label">Rejected</div><div class="stat-value" style="color:#F87171">${m.improvements_rejected||0}</div></div>
      <div class="stat"><div class="stat-label">Clarifications</div><div class="stat-value">${m.clarifications_asked||0}</div></div>
      <div class="stat"><div class="stat-label">Hallucinations</div><div class="stat-value" style="color:#FBBF24">${m.hallucinations_detected||0}</div></div>
      <div class="stat"><div class="stat-label">Est. Tokens</div><div class="stat-value">${m.total_tokens_estimated||0}</div></div>
    </div>
    <div style="margin-top:20px">
      <div class="dev-bar"><span class="dev-label">Accept rate</span><div class="dev-track"><div class="dev-fill" style="width:${m.total_tasks?(m.improvements_accepted/m.total_tasks*100):0}%;background:#2DD4A8"></div></div><span class="dev-val">${m.total_tasks?Math.round(m.improvements_accepted/m.total_tasks*100):0}%</span></div>
    </div>
  `;
}

function showTab(name) {
  ['brain','knowledge','feelings','metrics','search'].forEach(t => {
    document.getElementById('tab-'+t).style.display = t===name ? 'block' : 'none';
  });
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
}

function selectCortex(id) {
  document.querySelectorAll('.detail').forEach(d => d.classList.remove('active'));
  const el = document.getElementById('detail-'+id);
  if (el) el.classList.toggle('active', true);
}

function doSearch() {
  const q = document.getElementById('search-box').value.toLowerCase();
  if (!q || !DATA) { document.getElementById('search-results').innerHTML = '<div class="search-empty">Type to search</div>'; return; }

  let html = '';
  const match = (text) => (text||'').toLowerCase().includes(q);

  const matchedInsights = DATA.insights.filter(i => match(i.principle) || match(i.text));
  const matchedMeta = DATA.meta_rules.filter(m => match(m.principle) || match(m.text));
  const matchedStrat = DATA.strategies.filter(s => match(s.strategy) || match(s.mood) || match(s.task_type));

  if (matchedInsights.length) {
    html += `<div class="tier-header" style="color:#C084FC">Insights (${matchedInsights.length})</div>`;
    matchedInsights.forEach(i => { html += `<div class="knowledge-item" style="border-color:#C084FC">${i.principle||i.text}</div>`; });
  }
  if (matchedMeta.length) {
    html += `<div class="tier-header" style="color:#2DD4A8">Meta-rules (${matchedMeta.length})</div>`;
    matchedMeta.forEach(m => { html += `<div class="knowledge-item" style="border-color:#2DD4A8">${m.principle||m.text}</div>`; });
  }
  if (matchedStrat.length) {
    html += `<div class="tier-header" style="color:#FBBF24">Strategies (${matchedStrat.length})</div>`;
    matchedStrat.forEach(s => { html += `<div class="knowledge-item" style="border-color:#FBBF24">[${s.mood||'?'}] ${(s.strategy||'').slice(0,120)}</div>`; });
  }
  if (!html) html = `<div class="search-empty">No results for "${q}"</div>`;
  document.getElementById('search-results').innerHTML = html;
}

loadData();
setInterval(loadData, 5000);
</script>
</body>
</html>"""

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(get_maris_data().encode())
        elif self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress logs

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"\n  MARIS Brain Dashboard")
        print(f"  Open: http://localhost:{PORT}")
        print(f"  Data: {DATA_DIR}")
        print(f"  Auto-refreshes every 5 seconds")
        print(f"  Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Dashboard stopped.")
