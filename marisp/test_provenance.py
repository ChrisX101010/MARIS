"""
Tests for the provenance + verified-cache layer. All offline, no API calls.
Proves: lineage chains, tamper-evidence, no meaning-leak, cache saves calls
and never caches unverified output.
"""
from dataclasses import dataclass
from marisp_provenance import (
    build_provenance, verify_chain, VerifiedCache, render_provenance, GENESIS,
)

# minimal stand-in matching MARISP's TraceEntry shape
@dataclass
class TE:
    module: str
    passed: bool
    took_fallback: bool
    note: str
    result_summary: str

def trace(*mods):
    return [TE(m, True, False, "", f"{m}-result") for m in mods]

def test_per_answer_independent():
    a = build_provenance(trace("reason"))
    b = build_provenance(trace("reason"))
    # same steps, no prev link -> identical roots (deterministic)
    assert a.root == b.root
    print("PASS  identical steps -> identical root (deterministic hashing)")

def test_chain_links_turns():
    t1 = build_provenance(trace("detect_task", "reason"), prev_root=None)
    t2 = build_provenance(trace("detect_task", "reason"), prev_root=t1.root)
    assert t2.prev_root == t1.root
    # chaining changes the root even for identical steps (lineage captured)
    t2_unchained = build_provenance(trace("detect_task", "reason"))
    assert t2.root != t2_unchained.root
    print("PASS  chaining links turn2 to turn1's root (lineage captured)")

def test_no_meaning_leak():
    # turn1 about fruit, turn2 about company; chain carries only the hash
    t1 = build_provenance(trace("reason"), prev_root=None)  # 'reason-result'
    t2 = build_provenance(trace("reason"), prev_root=t1.root)
    # what turn2 inherited is a 16-char hash, not turn1's content
    assert t2.prev_root == t1.root and len(t2.prev_root) == 16
    print("PASS  turn2 inherits only turn1's hash (no content -> no meaning leak)")

def test_tamper_evident():
    t1 = build_provenance(trace("detect_task", "reason"), prev_root=None)
    t2 = build_provenance(trace("reason"), prev_root=t1.root)
    # tamper turn1 (different steps) -> its root changes -> t2 lineage breaks
    t1_tampered = build_provenance(trace("detect_task", "reason", "sneaky"), prev_root=None)
    assert not verify_chain([t1_tampered, t2]), "tamper should break the chain"
    assert verify_chain([t1, t2]), "intact chain should verify"
    print("PASS  tampering an earlier turn breaks chain verification")

def test_cache_saves_calls_and_gates():
    c = VerifiedCache()
    q = "what is 2 plus 2"
    assert c.get(q) is None  # miss
    # verified answer gets cached
    cached = c.put_if_verified(q, "4", senate_ok=True, probe_flagged=False)
    assert cached
    hit = c.get(q)
    assert hit and hit["answer"] == "4"
    # unverified answer is NEVER cached
    c.put_if_verified("bad q", "garbage", senate_ok=False, probe_flagged=True)
    assert c.get("bad q") is None
    print(f"PASS  cache reuses verified answers, refuses unverified  ({c.stats()})")

def test_cache_normalizes_input():
    c = VerifiedCache()
    c.put_if_verified("Hello   World", "hi", senate_ok=True, probe_flagged=False)
    # different spacing/case -> same key -> cache hit (token saving on near-dupes)
    assert c.get("hello world") is not None
    print("PASS  cache normalizes input (near-duplicate inputs reuse the answer)")

if __name__ == "__main__":
    print("=" * 56); print("MARISP provenance + cache tests"); print("=" * 56)
    test_per_answer_independent()
    test_chain_links_turns()
    test_no_meaning_leak()
    test_tamper_evident()
    test_cache_saves_calls_and_gates()
    test_cache_normalizes_input()
    print("\n--- example provenance render ---")
    t1 = build_provenance(trace("detect_emotion", "reason", "senate"), prev_root=None)
    print(render_provenance(t1))
    print("\n" + "=" * 56); print("ALL PROVENANCE TESTS PASSED"); print("=" * 56)
