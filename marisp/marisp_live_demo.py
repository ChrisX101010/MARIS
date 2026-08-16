"""
marisp_live_demo.py — MARISP over REAL MARIS modules, now with hash-chained
provenance and a verified-result cache (the grounded "waterfall").

Run from ~/MARIS/maris with OPENROUTER_API_KEY (via the shim) or a real key set.
"""
from marisp_maris_adapter import build_registry, DEFAULT_PLAN
from marisp_core import Interpreter, render_trace
from marisp_provenance import build_provenance, verify_chain, VerifiedCache, render_provenance
from llm_modules import DialogueMemory


def main():
    registry, gate = build_registry()
    interp = Interpreter(registry)
    cache = VerifiedCache()

    tests = [
        "hi there",
        "explain the tradeoffs between microservices and monoliths for a 3-person startup",
        "hi there",  # repeat — should hit the verified cache, saving API calls
    ]

    prev_root = None
    provenances = []

    for t in tests:
        print("\n" + "=" * 66)
        print("INPUT:", t)
        print("=" * 66)

        cached = cache.get(t)
        if cached:
            print("  [cache hit] reusing a previously VERIFIED answer — no API calls made")
            print("\nFINAL ANSWER:\n", cached["answer"][:400])
            continue

        ctx = {"input": t, "dialogue": DialogueMemory()}
        ctx, trace = interp.run(DEFAULT_PLAN, ctx)
        print(render_trace(trace))

        # provenance (chained to the previous answer via hash only)
        prov = build_provenance(trace, prev_root=prev_root)
        provenances.append(prov)
        prev_root = prov.root
        print("\n" + render_provenance(prov))

        # cache only if verified: senate passed AND probe not flagged
        senate_ok = next((tr.passed for tr in trace if tr.module == "senate"), True)
        probe_flagged = next((not tr.passed for tr in trace if tr.module == "hallucination_probe"), False)
        answer = ctx.get("answer", "")
        cached_now = cache.put_if_verified(t, answer, senate_ok=senate_ok, probe_flagged=probe_flagged)
        print(f"\n  cached (verified)?: {cached_now}")
        print("\nFINAL ANSWER:\n", answer[:400])

    print("\n" + "=" * 66)
    print("CONVERSATION PROVENANCE")
    print("=" * 66)
    print("chain intact (tamper-evident):", verify_chain(provenances))
    print("cache stats:", cache.stats())
    print("\nEach answer's root commits to its full reasoning path, and links to")
    print("the previous answer by hash only — lineage without meaning-leak.")


if __name__ == "__main__":
    main()
