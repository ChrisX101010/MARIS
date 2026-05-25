"""
patch_v9.py — MARIS v9 Live Patches
=====================================

Fixes applied at runtime (no file modification):
  1. ActivePerception.search()  — multi-backend search with fallbacks
                                    (Anthropic web_search → googlesearch-python
                                     → DuckDuckGo HTML scrape)
  2. InternalState persistence  — auto-save every update; load on init
  3. DriveResolver tuning       — configurable thresholds + telemetry
  4. EmbeddingMemory hook       — optional voyage-3-lite embeddings
                                    with graceful TF-IDF fallback
  5. FlightRecorder bridge      — exposes accept/reject counts so the
                                    dashboard can read real numbers

Apply by importing AFTER install_autonomous (v8):
    from patch_v8_autonomous import install_autonomous
    from patch_v9 import install_v9
    install_autonomous(ai_system)
    install_v9(ai_system)

Or one-shot via main_v9.py.

Design principles:
  - All fixes are runtime monkey-patches; reverting is just `git checkout`.
  - Persistence uses atomic writes (tmp + rename) so a crash mid-write
    never corrupts state files.
  - All new I/O is logged via patch_v8's _log_event for dashboard visibility.
  - Embedding upgrade is opt-in (set MARIS_EMBED_PROVIDER=voyage); default
    keeps existing bag-of-words so old data stays compatible.
"""

import os
import re
import json
import time
import math
import urllib.parse
import urllib.request
from datetime import datetime
from typing import List, Dict, Optional, Any


# ───────────────────────────────────────────────────────────────────────────
# Reuse log facility from v8 if present
# ───────────────────────────────────────────────────────────────────────────
try:
    from patch_v8_autonomous import _log_event
except Exception:
    _LOG_PATH = os.environ.get("MARIS_AUTONOMOUS_LOG", "autonomous_log.json")
    def _log_event(event: Dict[str, Any]) -> None:
        event = {**event, "ts": datetime.utcnow().isoformat()}
        try:
            existing = []
            if os.path.exists(_LOG_PATH):
                with open(_LOG_PATH) as f:
                    existing = json.load(f)
            existing.append(event)
            if len(existing) > 2000:
                existing = existing[-2000:]
            tmp = _LOG_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(existing, f, indent=2)
            os.replace(tmp, _LOG_PATH)
        except Exception:
            pass


# ───────────────────────────────────────────────────────────────────────────
# Atomic JSON write (used by InternalState persistence)
# ───────────────────────────────────────────────────────────────────────────
def _atomic_write_json(path: str, data: Any) -> bool:
    """Write JSON atomically: tmp file + os.replace. Returns True on success."""
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
        return True
    except Exception as e:
        _log_event({"event": "atomic_write_failed", "path": path, "err": str(e)})
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Fix 1: Multi-backend web search
# ═══════════════════════════════════════════════════════════════════════════

class SearchBackend:
    """
    Multi-backend search with graceful fallback. Tries in order:
      1. anthropic_websearch — uses Claude's web_search tool
      2. googlesearch        — googlesearch-python package
      3. duckduckgo_html     — scrape duckduckgo.com/html

    Pick backend with MARIS_SEARCH_BACKEND env var (auto = try all).
    """

    def __init__(self):
        self.backend = os.environ.get("MARIS_SEARCH_BACKEND", "auto").lower()
        self._available = self._probe_backends()

    def _probe_backends(self) -> List[str]:
        """Detect which backends are actually usable right now."""
        available = []
        # 1) Anthropic web_search (cheap to check — just verify API key)
        if os.environ.get("ANTHROPIC_API_KEY"):
            available.append("anthropic_websearch")
        # 2) googlesearch-python (import test)
        try:
            import googlesearch  # noqa
            available.append("googlesearch")
        except ImportError:
            pass
        # 3) DuckDuckGo HTML — always available as last resort
        available.append("duckduckgo_html")
        return available

    def search(self, query: str, max_results: int = 6) -> List[Dict[str, str]]:
        """Try backends in priority order. Return first non-empty result set."""
        order = (
            [self.backend] if self.backend != "auto" else self._available
        )
        for backend in order:
            try:
                results = getattr(self, f"_{backend}")(query, max_results)
                if results:
                    _log_event({
                        "event": "search_success",
                        "backend": backend,
                        "query": query[:80],
                        "count": len(results),
                    })
                    return results
            except Exception as e:
                _log_event({
                    "event": "search_backend_failed",
                    "backend": backend,
                    "query": query[:80],
                    "err": str(e)[:200],
                })
                continue
        _log_event({"event": "search_all_failed", "query": query[:80]})
        return []

    # ── Backend 1: Anthropic web_search ────────────────────────────────────
    def _anthropic_websearch(self, query: str, max_results: int
                             ) -> List[Dict[str, str]]:
        """Ask Claude to web-search and return structured results."""
        from anthropic import Anthropic
        client = Anthropic()
        # The web_search tool returns natural-language results; we ask Claude
        # to format them as JSON so we can parse cleanly.
        prompt = (
            f'Search the web for: "{query}"\n\n'
            f"Return up to {max_results} of the most useful results as a JSON "
            f"array. Each item: {{title, snippet (≤200 chars), url}}.\n"
            "Return ONLY the JSON array, no preamble, no fences."
        )
        try:
            resp = client.messages.create(
                model=os.environ.get("MARIS_MODEL", "claude-sonnet-4-6"),
                max_tokens=1200,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{"role": "user", "content": prompt}],
            )
            # Walk the content blocks for the final text
            text_blocks = [b.text for b in resp.content
                           if getattr(b, "type", "") == "text"]
            if not text_blocks:
                return []
            text = "".join(text_blocks).strip()
            # Strip fences defensively
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0].strip()
            # Find the JSON array
            m = re.search(r"\[[\s\S]*\]", text)
            if not m:
                return []
            data = json.loads(m.group())
            results = []
            for item in data[:max_results]:
                if isinstance(item, dict) and item.get("snippet"):
                    results.append({
                        "title": str(item.get("title", ""))[:200],
                        "snippet": str(item["snippet"])[:500],
                        "url": str(item.get("url", "")),
                    })
            return results
        except Exception:
            return []

    # ── Backend 2: googlesearch-python ─────────────────────────────────────
    def _googlesearch(self, query: str, max_results: int
                      ) -> List[Dict[str, str]]:
        """Use googlesearch-python. Newer versions return SearchResult objects
        with .title, .description, .url; older versions yield URL strings."""
        from googlesearch import search as gsearch
        results = []
        try:
            # Newer API: advanced=True gives title + description
            for r in gsearch(query, num_results=max_results, advanced=True):
                results.append({
                    "title": getattr(r, "title", "") or query,
                    "snippet": getattr(r, "description", "") or "",
                    "url": getattr(r, "url", str(r)),
                })
        except TypeError:
            # Older API: yields URL strings only
            for url in gsearch(query, num_results=max_results):
                results.append({
                    "title": query, "snippet": url, "url": url,
                })
        return [r for r in results if r["snippet"] or r["url"]]

    # ── Backend 3: DuckDuckGo HTML scrape ──────────────────────────────────
    def _duckduckgo_html(self, query: str, max_results: int
                         ) -> List[Dict[str, str]]:
        """Scrape duckduckgo.com/html — last-resort backend, no key needed."""
        url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({
            "q": query
        })
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 MARIS/9.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="replace")
        # Each result: <a class="result__a" href="...">title</a>
        # plus <a class="result__snippet">snippet</a>
        results = []
        pattern = re.compile(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>'
            r'(.+?)</a>'
            r'[\s\S]*?<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>'
            r'(.+?)</a>',
            re.IGNORECASE,
        )
        for m in pattern.finditer(html):
            if len(results) >= max_results:
                break
            url_raw, title, snippet = m.groups()
            title = re.sub(r"<[^>]+>", "", title).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            # DDG wraps URLs in a redirect; extract the actual target
            if url_raw.startswith("//duckduckgo.com/l/?uddg="):
                qs = urllib.parse.urlparse(url_raw).query
                params = urllib.parse.parse_qs(qs)
                url_raw = params.get("uddg", [url_raw])[0]
            results.append({
                "title": title[:200],
                "snippet": snippet[:500],
                "url": url_raw,
            })
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Fix 2: InternalState persistence
# ═══════════════════════════════════════════════════════════════════════════

def _patch_internal_state(inner_state, path: str = "internal_state.json"):
    """
    Wraps InternalState.update so every change persists. Loads any prior
    state at install time. Survives across sessions.
    """
    if getattr(inner_state, "_v9_persistence_installed", False):
        return inner_state

    inner_state._v9_state_path = path

    # ── load prior state if present ────────────────────────────────────────
    if os.path.exists(path):
        try:
            with open(path) as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                # State dimensions
                if "state" in saved and isinstance(saved["state"], dict):
                    for k, v in saved["state"].items():
                        if k in inner_state.state:
                            try:
                                inner_state.state[k] = float(v)
                            except (TypeError, ValueError):
                                pass
                # History if the class has it
                if "history" in saved and hasattr(inner_state, "history"):
                    try:
                        inner_state.history = list(saved["history"])[-100:]
                    except Exception:
                        pass
            _log_event({
                "event": "internal_state_loaded",
                "state": dict(inner_state.state),
            })
        except Exception as e:
            _log_event({"event": "internal_state_load_failed", "err": str(e)})

    # ── wrap update() to auto-save ─────────────────────────────────────────
    _orig_update = inner_state.update

    def _persistent_update(*args, **kwargs):
        result = _orig_update(*args, **kwargs)
        _save_internal_state(inner_state)
        return result

    inner_state.update = _persistent_update

    # ── wrap decay() too, if present ───────────────────────────────────────
    if hasattr(inner_state, "decay"):
        _orig_decay = inner_state.decay
        def _persistent_decay(*args, **kwargs):
            result = _orig_decay(*args, **kwargs)
            _save_internal_state(inner_state)
            return result
        inner_state.decay = _persistent_decay

    # ── snapshot for explicit save ─────────────────────────────────────────
    inner_state.persist = lambda: _save_internal_state(inner_state)

    inner_state._v9_persistence_installed = True
    return inner_state


def _save_internal_state(inner_state) -> bool:
    path = getattr(inner_state, "_v9_state_path", "internal_state.json")
    payload = {
        "state": dict(inner_state.state),
        "saved_at": datetime.utcnow().isoformat(),
    }
    # Capture dominant emotion if available
    if hasattr(inner_state, "get_dominant_emotion"):
        try:
            emo, intensity = inner_state.get_dominant_emotion()
            payload["dominant"] = {"emotion": emo, "intensity": intensity}
        except Exception:
            pass
    # Capture history if available
    if hasattr(inner_state, "history"):
        try:
            h = list(inner_state.history)[-100:]
            payload["history"] = h
        except Exception:
            pass
    return _atomic_write_json(path, payload)


# ═══════════════════════════════════════════════════════════════════════════
# Fix 3: DriveResolver tuning
# ═══════════════════════════════════════════════════════════════════════════

def _patch_drive_resolver(drive_resolver, thresholds: Optional[Dict] = None):
    """
    Attach tunable thresholds to DriveResolver and instrument override
    decisions for the dashboard.

    Default thresholds (gentle — won't fire on normal conversation):
      frustration_override:  0.75   (above this, MARIS pushes back)
      anxiety_override:      0.70   (above this, MARIS slows down)
      curiosity_override:    0.80   (above this, MARIS asks a question back)
      excitement_override:   0.85   (above this, MARIS proposes ideas)
    """
    if getattr(drive_resolver, "_v9_tuned", False):
        return drive_resolver

    defaults = {
        "frustration_override": 0.75,
        "anxiety_override":     0.70,
        "curiosity_override":   0.80,
        "excitement_override":  0.85,
        "warmth_override":      0.85,
    }
    drive_resolver.thresholds = {**defaults, **(thresholds or {})}
    drive_resolver._override_count = {"total": 0, "by_kind": {}}

    # Wrap resolve() to record override telemetry
    _orig_resolve = drive_resolver.resolve

    def _instrumented_resolve(*args, **kwargs):
        result = _orig_resolve(*args, **kwargs)
        if isinstance(result, dict) and result.get("override"):
            drive_resolver._override_count["total"] += 1
            kind = result.get("action", "unknown")
            drive_resolver._override_count["by_kind"][kind] = (
                drive_resolver._override_count["by_kind"].get(kind, 0) + 1
            )
            _log_event({
                "event": "drive_override",
                "action": kind,
                "reason": result.get("reason", "")[:200],
            })
        return result

    drive_resolver.resolve = _instrumented_resolve
    drive_resolver.get_override_stats = lambda: dict(
        drive_resolver._override_count
    )
    drive_resolver._v9_tuned = True
    return drive_resolver


# ═══════════════════════════════════════════════════════════════════════════
# Fix 4: Embedding upgrade (opt-in, TF-IDF fallback)
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingProvider:
    """
    Plug-in embedding provider with three options:
      - 'voyage'  : Voyage AI voyage-3-lite (requires VOYAGE_API_KEY, ~$0 cost
                    for embedding usage). Owned by Anthropic.
      - 'tfidf'   : Local TF-IDF (no API, much better than bag-of-words)
      - 'bow'     : Original bag-of-words (the v6/v7 behavior — kept as
                    fallback so existing memory stays compatible)

    Set MARIS_EMBED_PROVIDER env var to pick. Default: 'tfidf'.
    """

    def __init__(self, provider: Optional[str] = None,
                 corpus_path: str = "strategy_memory.json"):
        self.provider = provider or os.environ.get(
            "MARIS_EMBED_PROVIDER", "tfidf"
        ).lower()
        self.corpus_path = corpus_path
        self._idf_cache: Optional[Dict[str, float]] = None
        self._idf_cache_time: float = 0
        # Probe voyage key
        if self.provider == "voyage" and not os.environ.get("VOYAGE_API_KEY"):
            _log_event({"event": "voyage_no_key_fallback_to_tfidf"})
            self.provider = "tfidf"

    # ── public API ─────────────────────────────────────────────────────────
    def embed(self, text: str) -> Dict[str, float]:
        """
        Return an embedding compatible with the existing cosine_similarity
        function (which expects dict[str, float] with set-intersection on
        keys). For dense vectors (voyage), we still return a dict by binning
        — keeps drop-in compatibility.
        """
        if self.provider == "voyage":
            return self._embed_voyage(text)
        if self.provider == "tfidf":
            return self._embed_tfidf(text)
        return self._embed_bow(text)

    # ── implementations ────────────────────────────────────────────────────
    def _embed_bow(self, text: str) -> Dict[str, float]:
        """Original bag-of-words (kept for v6/v7 compatibility)."""
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        vocab = {}
        for w in words:
            vocab[w] = vocab.get(w, 0) + 1.0
        mag = math.sqrt(sum(v * v for v in vocab.values())) or 1.0
        return {k: v / mag for k, v in vocab.items()}

    def _embed_tfidf(self, text: str) -> Dict[str, float]:
        """TF-IDF using a corpus-derived IDF table refreshed every 60s."""
        idf = self._get_idf()
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        tf = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1.0
        # Multiply by IDF — rare words get more weight
        vec = {w: count * idf.get(w, 1.0) for w, count in tf.items()}
        # L2 normalize
        mag = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / mag for k, v in vec.items()}

    def _embed_voyage(self, text: str) -> Dict[str, float]:
        """Use Voyage AI for dense embeddings. Hashed to dict for
        compatibility with the existing sparse cosine_similarity."""
        try:
            import urllib.request
            api_key = os.environ["VOYAGE_API_KEY"]
            req = urllib.request.Request(
                "https://api.voyageai.com/v1/embeddings",
                data=json.dumps({
                    "input": [text[:2000]],
                    "model": "voyage-4-lite",
                }).encode(),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                resp = json.loads(r.read().decode())
            vec = resp["data"][0]["embedding"]  # list of floats
            # Convert to {dim_i: value} dict, compatible with cosine_similarity
            return {f"d{i}": float(v) for i, v in enumerate(vec)}
        except Exception as e:
            _log_event({"event": "voyage_fallback_to_tfidf", "err": str(e)})
            return self._embed_tfidf(text)

    # ── IDF table (cached) ─────────────────────────────────────────────────
    def _get_idf(self) -> Dict[str, float]:
        now = time.time()
        if self._idf_cache and (now - self._idf_cache_time) < 60:
            return self._idf_cache
        try:
            with open(self.corpus_path) as f:
                corpus = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._idf_cache = {}
            self._idf_cache_time = now
            return {}
        n = max(len(corpus), 1)
        doc_freq: Dict[str, int] = {}
        for item in corpus:
            text = (item.get("input", "") + " "
                    + item.get("strategy", ""))
            words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
            for w in words:
                doc_freq[w] = doc_freq.get(w, 0) + 1
        # idf = log((N + 1) / (df + 1)) + 1
        idf = {w: math.log((n + 1) / (df + 1)) + 1.0
               for w, df in doc_freq.items()}
        self._idf_cache = idf
        self._idf_cache_time = now
        return idf


def _patch_strategy_memory_embedding(memory, provider: EmbeddingProvider
                                     ) -> None:
    """
    Replace bag-of-words embedding generation in StrategyMemory.add() with
    the configured provider. Existing entries keep their old embeddings
    (forward-compatible).
    """
    if getattr(memory, "_v9_embed_patched", False):
        return
    _orig_add = memory.add

    def _add_with_new_embed(item: dict):
        # Pre-compute embedding with the new provider so the v6 add()'s
        # bag-of-words line is harmless (gets overwritten).
        text = item.get("input", "") + " " + item.get("strategy", "")
        new_embed = provider.embed(text)
        result = _orig_add(item)
        # The v6 add() overwrote item["embedding"]; restore the v9 one
        # if the item is still the same object in memory.data
        if memory.data and memory.data[-1] is item:
            item["embedding"] = new_embed
            # Persist the corrected embedding
            try:
                with open(memory.path, "w") as f:
                    json.dump(memory.data, f, indent=2)
            except Exception:
                pass
        return result

    memory.add = _add_with_new_embed
    memory._v9_embed_provider = provider
    memory._v9_embed_patched = True


# ═══════════════════════════════════════════════════════════════════════════
# Fix 5: FlightRecorder → Dashboard bridge
# ═══════════════════════════════════════════════════════════════════════════

def _patch_flight_recorder(flight_recorder) -> None:
    """
    Add a get_metrics() helper that walks flight_log.json and produces
    real accept/reject counts the dashboard can display.
    """
    if getattr(flight_recorder, "_v9_metrics_installed", False):
        return

    def get_metrics(log_path: str = "flight_log.json") -> Dict[str, Any]:
        if not os.path.exists(log_path):
            return {
                "total_turns": 0, "senate_accepted": 0, "senate_rejected": 0,
                "hallucinations_caught": 0, "drive_overrides": 0,
                "eurekas": 0, "consolidations": 0,
            }
        try:
            with open(log_path) as f:
                log = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"error": "log_unreadable"}

        # The flight log structure isn't fully known, so handle both
        # list-of-turns and list-of-events forms.
        if isinstance(log, dict):
            log = log.get("turns", log.get("events", []))

        accepted = rejected = halluc = overrides = eureka = consolid = 0
        for turn in log:
            if not isinstance(turn, dict):
                continue
            # The user said record() takes a kind + payload
            kind = turn.get("kind") or turn.get("event")
            payload = turn.get("payload", turn)

            if kind == "senate":
                if payload.get("accepted"):
                    accepted += 1
                else:
                    rejected += 1
            elif kind in ("hallucination", "halluc"):
                halluc += 1
            elif kind == "drive_resolver" and payload.get("overridden"):
                overrides += 1
            elif kind == "eureka":
                eureka += 1
            elif kind == "consolidation":
                consolid += 1

            # Also walk subevents inside a turn
            for k, v in payload.items() if isinstance(payload, dict) else []:
                if k == "senate" and isinstance(v, dict):
                    if v.get("accepted"):
                        accepted += 1
                    else:
                        rejected += 1

        return {
            "total_turns": len(log),
            "senate_accepted": accepted,
            "senate_rejected": rejected,
            "hallucinations_caught": halluc,
            "drive_overrides": overrides,
            "eurekas": eureka,
            "consolidations": consolid,
        }

    flight_recorder.get_metrics = get_metrics
    flight_recorder._v9_metrics_installed = True


# ═══════════════════════════════════════════════════════════════════════════
# Public installer
# ═══════════════════════════════════════════════════════════════════════════

def install_v9(ai_system, embedding_provider: Optional[str] = None,
               drive_thresholds: Optional[Dict] = None) -> Any:
    """
    Apply all v9 patches. Idempotent. Call AFTER install_autonomous (v8).

    Args:
      ai_system: the AI_System instance
      embedding_provider: 'voyage'|'tfidf'|'bow'|None (None = env var or tfidf)
      drive_thresholds: optional dict to override DriveResolver thresholds
    """
    if getattr(ai_system, "_v9_installed", False):
        return ai_system

    # ── Fix 1: swap search backend in ActivePerception ─────────────────────
    if hasattr(ai_system, "perception"):
        new_backend = SearchBackend()
        ai_system.perception._search_backend = new_backend
        # Monkey-patch the search method itself
        ai_system.perception.search = lambda q, _backend=new_backend: (
            _backend.search(q, max_results=6)
        )
        _log_event({
            "event": "search_backend_installed",
            "available": new_backend._available,
            "selected": new_backend.backend,
        })

    # ── Fix 2: persistent InternalState ────────────────────────────────────
    if hasattr(ai_system, "inner_state"):
        _patch_internal_state(ai_system.inner_state)

    # ── Fix 3: DriveResolver tuning ────────────────────────────────────────
    if hasattr(ai_system, "drive_resolver"):
        _patch_drive_resolver(ai_system.drive_resolver, drive_thresholds)

    # ── Fix 4: embedding upgrade ───────────────────────────────────────────
    if hasattr(ai_system, "memory"):
        provider = EmbeddingProvider(
            provider=embedding_provider,
            corpus_path=getattr(ai_system.memory, "path",
                                "strategy_memory.json"),
        )
        _patch_strategy_memory_embedding(ai_system.memory, provider)
        _log_event({
            "event": "embedding_provider_installed",
            "provider": provider.provider,
        })

    # ── Fix 5: FlightRecorder.get_metrics() ────────────────────────────────
    if hasattr(ai_system, "flight_recorder"):
        _patch_flight_recorder(ai_system.flight_recorder)

    # ── New slash commands ─────────────────────────────────────────────────
    _install_v9_commands(ai_system)

    ai_system._v9_installed = True
    _log_event({"event": "v9_installed"})
    return ai_system


def _install_v9_commands(ai_system) -> None:
    """Add /search-test, /state, /drives, /embed-info commands."""
    if not hasattr(ai_system, "run"):
        return
    _orig_run = ai_system.run

    def _v9_run(input_text: str, *args, **kwargs):
        text = input_text.strip()

        if text.startswith("/search-test"):
            q = text[len("/search-test"):].strip() or "AI cognition research"
            print(f"\n  Testing search for: {q!r}")
            backend = getattr(ai_system.perception, "_search_backend", None)
            if not backend:
                print("  No v9 search backend installed.")
                return None
            results = backend.search(q, max_results=4)
            print(f"  Got {len(results)} results from "
                  f"{backend._available[0] if backend._available else 'none'}")
            for i, r in enumerate(results, 1):
                print(f"\n  [{i}] {r.get('title', '')[:70]}")
                print(f"      {r.get('snippet', '')[:120]}")
                print(f"      {r.get('url', '')}")
            print()
            return None

        if text == "/state":
            inner = getattr(ai_system, "inner_state", None)
            if not inner:
                print("  No internal state.")
                return None
            print("\n  === Internal State (persistent) ===")
            for k, v in inner.state.items():
                bar = "#" * int(abs(v) * 20)
                sign = "+" if v >= 0 else "-"
                print(f"    {k:14s} {sign}{bar:<20s}  {v:+.3f}")
            path = getattr(inner, "_v9_state_path", "internal_state.json")
            print(f"  Saved to: {path}")
            print()
            return None

        if text == "/drives":
            dr = getattr(ai_system, "drive_resolver", None)
            if not dr or not hasattr(dr, "get_override_stats"):
                print("  DriveResolver not v9-tuned.")
                return None
            stats = dr.get_override_stats()
            print("\n  === Drive Resolver ===")
            print(f"    Total overrides: {stats['total']}")
            for k, n in stats.get("by_kind", {}).items():
                print(f"    {k:20s} {n}")
            print(f"\n    Thresholds:")
            for k, v in dr.thresholds.items():
                print(f"      {k:25s} {v}")
            print()
            return None

        if text == "/embed-info":
            mem = getattr(ai_system, "memory", None)
            prov = getattr(mem, "_v9_embed_provider", None) if mem else None
            print("\n  === Embedding Provider ===")
            if not prov:
                print("    Bag-of-words (v6 default)")
            else:
                print(f"    Provider: {prov.provider}")
                if prov.provider == "tfidf":
                    idf = prov._get_idf()
                    print(f"    IDF terms: {len(idf)}")
            print()
            return None

        return _orig_run(input_text, *args, **kwargs)

    ai_system.run = _v9_run
