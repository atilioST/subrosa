#!/usr/bin/env python3
"""Deliberate backfill: distill historical events that predate the id cursor.

The live distiller only processes events after _meta.last_distilled_event_id;
everything logged before Open Brain went online was skipped. This script
distills a chosen slice of that backlog without touching the cursor, so it is
safe to run while the subrosa service is up (WAL + busy_timeout handle the
concurrent writes).

Events whose id already appears in knowledge.source_ref are skipped, so
re-running after a partial failure only processes what's left.

Usage:
    # What would run, without calling Haiku
    python scripts/backfill_events.py --filter telegram:message \
        --filter scheduler:person_digest --dry-run

    # The real thing
    python scripts/backfill_events.py --filter telegram:message \
        --filter scheduler:person_digest
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subrosa.distiller import Distiller  # noqa: E402
from subrosa.store import Store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--filter", action="append", required=True, metavar="SOURCE:EVENT_TYPE",
        help="event slice to backfill, e.g. telegram:message (repeatable)",
    )
    p.add_argument(
        "--before-id", type=int, default=None,
        help="only events with id <= this (default: current distiller cursor)",
    )
    p.add_argument("--limit", type=int, default=None, help="cap total events processed")
    p.add_argument("--model", default="haiku", help="model for distillation calls")
    p.add_argument("--dry-run", action="store_true", help="report the slice, call nothing")
    return p.parse_args()


async def fetch_backlog(store: Store, pairs: list[tuple[str, str]],
                        before_id: int, limit: int | None) -> list[dict]:
    where = " OR ".join("(source = ? AND event_type = ?)" for _ in pairs)
    params: list = [v for pair in pairs for v in pair]
    query = (
        f"SELECT * FROM events WHERE ({where}) AND id <= ? "
        "AND CAST(id AS TEXT) NOT IN "
        "(SELECT source_ref FROM knowledge WHERE source_ref != '') "
        "ORDER BY id ASC"
    )
    params.append(before_id)
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    cursor = await store._db.execute(query, params)
    return [dict(r) for r in await cursor.fetchall()]


async def main() -> None:
    args = parse_args()
    pairs = []
    for f in args.filter:
        source, _, event_type = f.partition(":")
        if not source or not event_type:
            sys.exit(f"Bad --filter {f!r}: expected SOURCE:EVENT_TYPE")
        pairs.append((source, event_type))

    store = Store()
    await store.initialize()
    try:
        before_id = args.before_id
        if before_id is None:
            cursor = await store.get_meta("last_distilled_event_id")
            if cursor is None:
                sys.exit("No distiller cursor set and no --before-id given")
            before_id = int(cursor)

        events = await fetch_backlog(store, pairs, before_id, args.limit)
        if not events:
            logger.info("Nothing to backfill for %s before id %d", pairs, before_id)
            return

        by_type: dict[str, int] = {}
        for e in events:
            key = f"{e['source']}:{e['event_type']}"
            by_type[key] = by_type.get(key, 0) + 1
        logger.info(
            "Backlog: %d events (ids %d..%d) before cursor %d — %s",
            len(events), events[0]["id"], events[-1]["id"], before_id,
            ", ".join(f"{k}={n}" for k, n in sorted(by_type.items())),
        )

        if args.dry_run:
            logger.info("Dry run — ~%d Haiku batches, nothing called", -(-len(events) // 15))
            return

        before_count = await store.knowledge_count()
        items = await Distiller(store, model=args.model).distill(events)
        after_count = await store.knowledge_count()
        await store.log_diagnostic(
            "backfill",
            f"backfilled {len(events)} events → {items} knowledge items "
            f"(knowledge {before_count} → {after_count})",
        )
        logger.info(
            "Done: %d events → %d items (knowledge count %d → %d)",
            len(events), items, before_count, after_count,
        )
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
