"""
Retrieval eval harness.

Seeds a test database with known knowledge items, runs each VP query
against the brain server's search_knowledge(), and measures recall@k.

Usage:
    python -m tests.eval_retrieval                  # all queries
    python -m tests.eval_retrieval --id q_001       # single query
    python -m tests.eval_retrieval --k 5            # recall@5 (default: 3)
    python -m tests.eval_retrieval --verbose        # show ranked results

Scoring:
    recall@k = fraction of expected items appearing in top-k results.
    A result "matches" an expected item if:
      - domain matches, AND
      - all must_surface terms appear in the result summary (case-insensitive)

    Target: recall@3 >= 0.70 before moving to Phase 4 (brain server).

Note: This eval is fully deterministic — no LLM judge needed. Embeddings do
the work; we just check if the right content surfaces.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EVENTS_FILE = FIXTURES_DIR / "distillation_events.jsonl"
EXPECTED_FILE = FIXTURES_DIR / "distillation_expected.jsonl"
QUERIES_FILE = FIXTURES_DIR / "vp_queries.jsonl"


# ── Fixtures ───────────────────────────────────────────────────────────────────

def load_queries(query_id: str | None = None) -> list[dict]:
    queries = [json.loads(l) for l in QUERIES_FILE.read_text().splitlines() if l.strip()]
    if query_id:
        queries = [q for q in queries if q["id"] == query_id]
    return queries


def build_corpus_from_expected() -> list[dict]:
    """
    Build a minimal knowledge corpus from the expected extraction file.
    Each expected item becomes a knowledge record for seeding the test DB.
    Used when Phase 2 (distillation) is not yet complete.
    """
    corpus = []
    events = {e["id"]: e for e in [
        json.loads(l) for l in EVENTS_FILE.read_text().splitlines() if l.strip()
    ]}
    for line in EXPECTED_FILE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        event = events.get(row["event_id"], {})
        for i, item in enumerate(row["expected_items"]):
            corpus.append({
                "id": f"{row['event_id']}_item_{i}",
                "domain": item["domain"],
                "primitive": item["primitive"],
                "entity_type": item["entity_type"],
                "entity_name": item["entity_name"],
                "summary": " ".join(item["must_contain"]) + ". " + event.get("summary", ""),
                "source": event.get("source", "fixture"),
                "confidence": 0.9,
            })
    return corpus


# ── Store seeding ──────────────────────────────────────────────────────────────

async def seed_test_store(corpus: list[dict]):
    """Seed a temporary in-memory store with the corpus for retrieval testing."""
    try:
        from subrosa.store import Store
    except ImportError:
        logger.error("Cannot import subrosa.store — run from repo root")
        sys.exit(1)

    store = Store(db_path=":memory:")
    await store.initialize()

    # Seed ontology
    for domain in set(item["domain"] for item in corpus):
        await store.upsert_ontology_domain(domain, "")

    for primitive in set(item["primitive"] for item in corpus):
        await store.upsert_ontology_primitive(primitive, "", "")

    # Seed knowledge items
    inserted_ids = []
    for item in corpus:
        kid = await store.insert_knowledge(
            domain=item["domain"],
            primitive=item["primitive"],
            entity_type=item["entity_type"],
            entity_name=item["entity_name"],
            summary=item["summary"],
            source=item["source"],
            confidence=item["confidence"],
        )
        inserted_ids.append((item["id"], kid))

    return store, dict(inserted_ids)


# ── Matching ───────────────────────────────────────────────────────────────────

def result_matches_query(result: dict, query: dict) -> bool:
    """
    A result matches the query if:
    1. Its domain is in the query's expected domains, AND
    2. At least one must_surface term appears in the summary (case-insensitive)
    """
    summary_lower = result.get("summary", "").lower()
    domain_ok = result.get("domain") in query.get("domains", [])
    terms = query.get("must_surface", [])
    terms_ok = any(t.lower() in summary_lower for t in terms) if terms else True
    return domain_ok and terms_ok


def recall_at_k(results: list[dict], query: dict, k: int) -> float:
    """Fraction of must_surface terms found in top-k results."""
    top_k = results[:k]
    must_surface = query.get("must_surface", [])
    if not must_surface:
        return 1.0

    found_terms = set()
    for result in top_k:
        summary_lower = result.get("summary", "").lower()
        for term in must_surface:
            if term.lower() in summary_lower:
                found_terms.add(term.lower())

    return len(found_terms) / len(must_surface)


# ── Main eval loop ─────────────────────────────────────────────────────────────

async def run_eval(query_id: str | None = None, k: int = 3, verbose: bool = False) -> None:
    queries = load_queries(query_id)
    if not queries:
        logger.error("No queries found (id=%s)", query_id)
        sys.exit(1)

    corpus = build_corpus_from_expected()
    logger.info("Seeding test store with %d knowledge items...", len(corpus))

    try:
        store, _id_map = await seed_test_store(corpus)
    except Exception as e:
        logger.error("Could not seed store: %s — run after Phase 1 schema is built", e)
        sys.exit(1)

    query_scores: list[float] = []
    results_log = []

    for query in queries:
        try:
            results = await store.search_knowledge_semantic(
                query["query"],
                limit=k + 5,  # retrieve slightly more for ranking analysis
            )
        except AttributeError:
            logger.warning("search_knowledge_semantic not yet implemented — scores will be 0.0")
            results = []

        score = recall_at_k(results, query, k)
        query_scores.append(score)
        results_log.append({"query_id": query["id"], "score": score, "results": results[:k]})

        if verbose:
            print(f"\n{'='*60}")
            print(f"QUERY {query['id']}: {query['query']}")
            print(f"recall@{k}: {score:.2f}  |  expected domains: {query['domains']}")
            print(f"must_surface: {query['must_surface']}")
            print(f"Top-{k} results:")
            for i, r in enumerate(results[:k], 1):
                match = "✓" if result_matches_query(r, query) else " "
                print(f"  {i}. {match} [{r.get('domain','?')} / {r.get('primitive','?')}] {r.get('entity_name','?')}")
                print(f"      {r.get('summary','')[:100]}...")
            if query.get("notes"):
                print(f"Note: {query['notes']}")

    overall = sum(query_scores) / len(query_scores) if query_scores else 0.0

    print(f"\n{'='*60}")
    print(f"RETRIEVAL EVAL RESULTS  (recall@{k})")
    print(f"{'='*60}")
    print(f"Queries evaluated: {len(queries)}")
    print(f"Overall recall@{k}: {overall:.2f}  (target: >= 0.70)")
    print(f"{'='*60}")

    for r in sorted(results_log, key=lambda x: x["score"]):
        q = next(q for q in queries if q["id"] == r["query_id"])
        bar = "█" * int(r["score"] * 10) + "░" * (10 - int(r["score"] * 10))
        print(f"  {r['query_id']}  [{bar}] {r['score']:.2f}  {q['query'][:60]}")

    if overall >= 0.70:
        print(f"\n✓ PASS — retrieval quality sufficient for Phase 4 (brain server)")
    elif overall >= 0.50:
        print(f"\n~ PARTIAL — check embedding quality and threshold settings")
    else:
        print(f"\n✗ FAIL — embedding model or search implementation needs work")

    await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval retrieval quality")
    parser.add_argument("--id", help="Run a single query by ID (e.g. q_001)")
    parser.add_argument("--k", type=int, default=3, help="Recall@k (default: 3)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show ranked results")
    args = parser.parse_args()
    asyncio.run(run_eval(query_id=args.id, k=args.k, verbose=args.verbose))


if __name__ == "__main__":
    main()
