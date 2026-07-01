"""
Distillation eval harness.

Runs the Haiku distillation prompt against fixture events and scores the output
against hand-labeled expected extractions.

Usage:
    python -m tests.eval_distillation                # all events
    python -m tests.eval_distillation --id evt_001   # single event
    python -m tests.eval_distillation --verbose       # show full output

Scoring:
    Each expected item is scored 0-1 by a Claude call (LLM-as-judge):
    "Did the distillation output capture this expected knowledge item?"

    Per-event score = mean of item scores.
    Overall score   = mean of per-event scores.

    Target: >= 0.75 before moving to Phase 3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

# Allow running as: python -m tests.eval_distillation
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EVENTS_FILE = FIXTURES_DIR / "distillation_events.jsonl"
EXPECTED_FILE = FIXTURES_DIR / "distillation_expected.jsonl"


# ── Fixtures ───────────────────────────────────────────────────────────────────

def load_events(event_id: str | None = None) -> list[dict]:
    events = [json.loads(l) for l in EVENTS_FILE.read_text().splitlines() if l.strip()]
    if event_id:
        events = [e for e in events if e["id"] == event_id]
    return events


def load_expected() -> dict[str, list[dict]]:
    """Returns {event_id: [expected_item, ...]}"""
    result = {}
    for line in EXPECTED_FILE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result[row["event_id"]] = row["expected_items"]
    return result


# ── Distillation (calls the real prompt builder) ───────────────────────────────

async def run_distillation(events: list[dict]) -> dict[str, list[dict]]:
    """
    Calls the distillation pipeline on the given events.
    Returns {event_id: [extracted_item, ...]}
    """
    from subrosa.store import Store
    from subrosa.distiller import Distiller

    store = Store(db_path=":memory:")
    await store.initialize()

    distiller = Distiller(store=store)

    # Collect items per event_id by running distill() on individual events
    # so we can map extracted items back to their source event
    result: dict[str, list[dict]] = {e["id"]: [] for e in events}

    # Run distill in batches; items have event_id field we can use for grouping
    # We call distill() which writes to store, but we also capture raw items
    # by monkey-patching insert_knowledge to collect them.
    raw_items: list[dict] = []
    original_insert = store.insert_knowledge

    async def _capturing_insert(**kwargs):
        raw_items.append(kwargs)
        return await original_insert(**kwargs)

    store.insert_knowledge = _capturing_insert  # type: ignore[method-assign]

    # Run on all events; distiller will batch them
    await distiller.distill(events)

    # Close the store to release aiosqlite threads before returning
    await store.close()

    # Group captured items by source_ref (which is the event_id)
    for item in raw_items:
        eid = item.get("source_ref", "")
        if eid in result:
            result[eid].append(item)

    return result


# ── Scoring ────────────────────────────────────────────────────────────────────


def score_item_fast(expected: dict, output_items: list[dict]) -> tuple[float, str]:
    """
    Fast deterministic scoring: checks if must_contain terms appear in any output
    item's summary. Returns 1.0 if all terms found, partial credit for partial match.
    Does not call Claude.
    """
    must = [t.lower() for t in expected.get("must_contain", [])]
    if not must:
        return 1.0, "no must_contain terms"

    best_hits = 0
    for item in output_items:
        text = (item.get("summary", "") + " " + item.get("entity_name", "")).lower()
        hits = sum(1 for t in must if t in text)
        best_hits = max(best_hits, hits)

    score = best_hits / len(must)
    found = [t for t in must if any(t in (it.get("summary", "") + " " + it.get("entity_name", "")).lower() for it in output_items)]
    reason = f"{best_hits}/{len(must)} terms found: {found}"
    return score, reason


# ── LLM-as-judge (slower, requires Claude CLI) ─────────────────────────────────

JUDGE_PROMPT = """You are evaluating whether a distillation output captured an expected knowledge item.

Expected item:
  domain:      {domain}
  primitive:   {primitive}
  entity_type: {entity_type}
  entity_name: {entity_name}
  must contain: {must_contain}

Distillation output for this event:
{output}

Did the distillation output capture the essential information from the expected item?
Score 0.0-1.0 where:
  1.0 = fully captured (correct domain, primitive, entity, and all must_contain terms present)
  0.7 = mostly captured (right domain/primitive, entity close, most must_contain terms)
  0.4 = partially captured (some relevant info extracted but misclassified or incomplete)
  0.0 = not captured at all

Respond with ONLY a JSON object: {{"score": 0.0, "reason": "one sentence"}}"""


_CLAUDE_CLI = "/home/ati/.local/bin/claude"
_JUDGE_SYSTEM = "You are a precise evaluator. Respond only with JSON."


async def judge_item(
    expected: dict,
    output_items: list[dict],
) -> tuple[float, str]:
    """Ask Haiku (via Claude CLI subprocess) to score whether expected was captured in output_items."""
    output_text = json.dumps(output_items, indent=2) if output_items else "(no items extracted)"
    prompt = JUDGE_PROMPT.format(
        domain=expected["domain"],
        primitive=expected["primitive"],
        entity_type=expected["entity_type"],
        entity_name=expected["entity_name"],
        must_contain=expected["must_contain"],
        output=output_text,
    )
    cmd = [
        _CLAUDE_CLI,
        "--model", "claude-haiku-4-5-20251001",
        "--output-format", "text",
        "--no-session-persistence",
        "--system-prompt", _JUDGE_SYSTEM,
        "-p", prompt,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            logger.warning("Judge CLI exited %d: %s", proc.returncode, err[:200])
            return 0.0, f"cli error (exit {proc.returncode})"
        result_text = stdout.decode(errors="replace").strip()
        result_text = re.sub(r"^```(?:json)?\s*", "", result_text)
        result_text = re.sub(r"\s*```$", "", result_text).strip()
        result = json.loads(result_text)
        return float(result["score"]), result.get("reason", "")
    except Exception as e:
        logger.warning("Judge call failed: %s", e)
        return 0.0, f"judge error: {e}"


# ── Main eval loop ─────────────────────────────────────────────────────────────

async def run_eval(event_id: str | None = None, verbose: bool = False, fast: bool = False) -> None:
    events = load_events(event_id)
    expected_map = load_expected()

    if not events:
        logger.error("No events found (id=%s)", event_id)
        sys.exit(1)

    logger.info("Running distillation on %d event(s)...", len(events))
    extracted = await run_distillation(events)

    if fast:
        # Fast mode: deterministic must_contain scoring, no Claude calls
        logger.info("Fast mode: scoring %d item checks deterministically...",
                    sum(len(v) for v in expected_map.values()))
        scores_by_event: dict[str, list[tuple[float, str, dict]]] = {e["id"]: [] for e in events}
        for eid in scores_by_event:
            output_items = extracted.get(eid, [])
            for exp in expected_map.get(eid, []):
                score, reason = score_item_fast(exp, output_items)
                scores_by_event[eid].append((score, reason, exp))
    else:
        # LLM-as-judge: parallel Claude CLI calls
        sem = asyncio.Semaphore(4)

        async def judge_limited(exp: dict, items: list[dict]) -> tuple[float, str]:
            async with sem:
                return await judge_item(exp, items)

        judge_tasks: list[tuple[str, dict, asyncio.Task]] = []
        for event in events:
            eid = event["id"]
            output_items = extracted.get(eid, [])
            for exp in expected_map.get(eid, []):
                task = asyncio.create_task(judge_limited(exp, output_items))
                judge_tasks.append((eid, exp, task))

        logger.info("Judging %d item checks via LLM (concurrency=4)...", len(judge_tasks))
        if judge_tasks:
            await asyncio.gather(*[t for _, _, t in judge_tasks])

        scores_by_event = {e["id"]: [] for e in events}
        for eid, exp, task in judge_tasks:
            score, reason = task.result()
            scores_by_event[eid].append((score, reason, exp))

    event_scores: list[float] = []
    results = []

    for event in events:
        eid = event["id"]
        output_items = extracted.get(eid, [])
        judge_results = scores_by_event.get(eid, [])

        if verbose:
            print(f"\n{'='*60}")
            print(f"EVENT: {eid}")
            print(f"Summary: {event['summary'][:120]}...")
            print(f"Extracted {len(output_items)} items, expecting {len(judge_results)} checks")

        item_scores: list[float] = []
        for score, reason, exp in judge_results:
            item_scores.append(score)
            if verbose:
                status = "✓" if score >= 0.7 else ("~" if score >= 0.4 else "✗")
                print(f"  {status} [{score:.1f}] {exp['domain']} / {exp['primitive']} / {exp['entity_name']}")
                print(f"       {reason}")

        event_score = sum(item_scores) / len(item_scores) if item_scores else 0.0
        event_scores.append(event_score)
        results.append({
            "event_id": eid,
            "score": event_score,
            "item_scores": item_scores,
            "extracted_count": len(output_items),
            "expected_checks": len(judge_results),
        })

    overall = sum(event_scores) / len(event_scores) if event_scores else 0.0

    mode = "fast (must_contain)" if fast else "LLM judge"
    print(f"\n{'='*60}")
    print(f"DISTILLATION EVAL RESULTS  [{mode}]")
    print(f"{'='*60}")
    print(f"Events evaluated: {len(events)}")
    print(f"Overall score:    {overall:.2f}  (target: >= 0.75)")
    print(f"{'='*60}")

    # Per-event summary
    for r in sorted(results, key=lambda x: x["score"]):
        bar = "█" * int(r["score"] * 10) + "░" * (10 - int(r["score"] * 10))
        print(f"  {r['event_id']}  [{bar}] {r['score']:.2f}  ({r['extracted_count']} extracted, {r['expected_checks']} checks)")

    if overall >= 0.75:
        print(f"\n✓ PASS — ready to move to Phase 3 (retrieval)")
    elif overall >= 0.5:
        print(f"\n~ PARTIAL — tune the distillation prompt and re-run")
    else:
        print(f"\n✗ FAIL — significant prompt work needed before proceeding")

    import sys
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval distillation quality")
    parser.add_argument("--id", help="Run a single event by ID (e.g. evt_001)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-item scores")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: score by must_contain term matching, no Claude calls")
    args = parser.parse_args()
    asyncio.run(run_eval(event_id=args.id, verbose=args.verbose, fast=args.fast))


if __name__ == "__main__":
    main()
