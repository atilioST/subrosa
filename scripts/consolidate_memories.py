#!/usr/bin/env python3
"""Aggressive memory consolidation script.

Strategy:
1. Merge duplicate person topics (e.g., "Brock" + "Brock Yaeger" → "Brock")
2. Merge duplicate project/fact topics (e.g., "Task System" + "Task Management System")
3. Merge "User" facts into "Atilio" (same person)
4. Delete stale/transient facts (implementation details, debugging notes, obsolete state)
5. Deduplicate near-identical facts within merged topics
6. Delete topics that are entirely redundant with briefing.md or system prompt
7. Delete meta-memories (memories about the memory system itself)

Safety: Creates a backup before any changes.
"""

import shutil
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("~/.subrosa/subrosa.db").expanduser()
BACKUP_PATH = DB_PATH.parent / f"subrosa_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

def backup():
    shutil.copy2(DB_PATH, BACKUP_PATH)
    print(f"Backup created: {BACKUP_PATH}")

def connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def soft_delete_topic(conn, topic_id, reason=""):
    """Soft-delete a topic by marking it inactive."""
    conn.execute("UPDATE topics SET active = 0 WHERE id = ?", (topic_id,))
    if reason:
        print(f"  ✗ Deleted topic {topic_id}: {reason}")

def delete_fact(conn, fact_id, reason=""):
    """Hard-delete a specific fact."""
    conn.execute("DELETE FROM topic_facts WHERE id = ?", (fact_id,))
    if reason:
        print(f"    ✗ Deleted fact {fact_id}: {reason}")

def move_facts(conn, from_topic_id, to_topic_id):
    """Move all facts from one topic to another."""
    conn.execute(
        "UPDATE topic_facts SET topic_id = ? WHERE topic_id = ?",
        (to_topic_id, from_topic_id)
    )

def move_tags(conn, from_topic_id, to_topic_id):
    """Move tags, ignoring conflicts."""
    conn.execute(
        "INSERT OR IGNORE INTO topic_tags (topic_id, tag) "
        "SELECT ?, tag FROM topic_tags WHERE topic_id = ?",
        (to_topic_id, from_topic_id)
    )

def move_attributes(conn, from_topic_id, to_topic_id):
    """Move attributes, ignoring conflicts."""
    conn.execute(
        "INSERT OR IGNORE INTO topic_attributes (topic_id, key, value) "
        "SELECT ?, key, value FROM topic_attributes WHERE topic_id = ?",
        (to_topic_id, from_topic_id)
    )

def merge_topic_into(conn, from_id, to_id, delete_from=True):
    """Merge all data from one topic into another."""
    move_facts(conn, from_id, to_id)
    move_tags(conn, from_id, to_id)
    move_attributes(conn, from_id, to_id)
    if delete_from:
        soft_delete_topic(conn, from_id, f"merged into topic {to_id}")

def get_facts_for_topic(conn, topic_id):
    """Get all facts for a topic."""
    cur = conn.execute(
        "SELECT id, fact, confidence FROM topic_facts WHERE topic_id = ?",
        (topic_id,)
    )
    return cur.fetchall()

def count_active(conn):
    cur = conn.execute("SELECT COUNT(*) FROM topics WHERE active = 1")
    return cur.fetchone()[0]

def count_active_facts(conn):
    cur = conn.execute(
        "SELECT COUNT(*) FROM topic_facts tf "
        "JOIN topics t ON tf.topic_id = t.id WHERE t.active = 1"
    )
    return cur.fetchone()[0]


def phase1_merge_person_duplicates(conn):
    """Merge duplicate person entries (first name → full name variants)."""
    print("\n=== Phase 1: Merge duplicate person topics ===")

    # Map: (keep_id, delete_ids)
    merges = [
        # User → Atilio (same person, merge User INTO Atilio since User has more facts)
        # Actually: keep "Atilio" (id=3) as canonical, merge "User" (id=4) into it
        # But "User" is special - it's how the system refers to the active user
        # Keep both: "Atilio" for org context, "User" for preferences
        # Actually for aggressive: merge Atilio Jobson into Atilio
        (3, [216]),  # Atilio Jobson → Atilio

        # Brock: keep 114, delete
        # (no duplicate full-name topic found)

        # Bailee: keep 131 (Bailee Warsing), merge 118 (Bailee)
        (131, [118]),

        # Brittany/Brittney Cruz: keep 208, merge 222
        (208, [222]),

        # Evgeny: keep 36, no dup

        # Geoff + Geoff Manning: keep 204, merge 218
        (204, [218]),

        # Jake + Jake Cuevas: keep 127 (Jake Cuevas), merge 207
        (127, [207]),

        # Jared + Jared Onnen: keep 217 (Jared Onnen), merge 203
        (217, [203]),

        # Luke + Luke Chavez: keep 112 (Luke), merge 220
        (112, [220]),

        # Nathan + Nathan Z + Nathan Zorndorf: keep 210 (Nathan Z), merge 113, 221
        (210, [113, 221]),

        # Neeraj + Neeraj Nema: keep 130 (Neeraj Nema), merge 236
        (130, [236]),

        # Roberto Moller: keep 205, no dup (235=Santiago is different)

        # Sean Glover: keep 206, no dup

        # Walter + Walter Thorn: keep 116 (Walter), merge 129
        (116, [129]),

        # Yuval + Yuval Klein: keep 168 (Yuval), merge 219
        (168, [219]),

        # Mohamed Seliman: keep 201, no dup
        # Himanshu Patel: keep 202, no dup
    ]

    for keep_id, delete_ids in merges:
        keep_name = conn.execute("SELECT name FROM topics WHERE id = ?", (keep_id,)).fetchone()[0]
        for del_id in delete_ids:
            del_name = conn.execute("SELECT name FROM topics WHERE id = ?", (del_id,)).fetchone()[0]
            print(f"  Merging '{del_name}' (id={del_id}) → '{keep_name}' (id={keep_id})")
            merge_topic_into(conn, del_id, keep_id)


def phase2_merge_duplicate_topics(conn):
    """Merge duplicate fact/project/insight topics."""
    print("\n=== Phase 2: Merge duplicate non-person topics ===")

    merges = [
        # Scout-related consolidation: keep "Scout" (id=6)
        (6, [
            135,  # Scout Platform
            140,  # Scout Product Subsystems → already in briefing
            141,  # Scout Operational Structure → already in briefing
            139,  # Scout Slack Channels → already in briefing
            146,  # Scout System
            136,  # Scout subsystems → already in briefing
            237,  # Scout org structure
            196,  # Scout organization size
            197,  # Scout leadership
            244,  # Scout leadership structure
            245,  # Scout support functions
            278,  # Scout Implementation
            277,  # Team Practice
        ]),

        # Scout team structure insights → Scout
        (6, [132]),  # Scout team structure insight

        # Task System consolidation: keep "Task Management System" (id=22), merge "Task System" (id=33)
        (22, [33, 41, 32]),  # Task System, Task Integration Feature, task memory system

        # Subrosa consolidation: keep "Subrosa" (id=9)
        (9, [
            62,   # subrosa orchestrator
            63,   # subrosa SessionConfig
            189,  # subrosa1
            251,  # subrosa skill system
            293,  # subrosa project
            153,  # Telegram bot
        ]),

        # Briefing consolidation: keep one
        (252, [  # briefing skill (keep)
            193,  # briefing (fact)
            179,  # Briefing System
            180,  # Briefing Content Structure
            291,  # Scout briefing schedule
            295,  # briefing skill scheduling
        ]),

        # Memory System consolidation: keep 174
        (174, [
            1,    # Memory System Migration (done, stale)
            2,    # Memory Distribution (stale count)
            88,   # Memory store implementation
            89,   # Memory retrieval workflow
            99,   # MemoryStore API
            98,   # MemoryStore implementation
            309,  # Memory Database
            310,  # Memory Consolidation Opportunity
        ]),

        # Slack integration: keep one
        (306, [  # Slack MCP Server (keep)
            273,  # MCP Slack server
            254,  # Slack MCP Tool
            275,  # Slack integration
        ]),

        # CU Internship: keep 47 (project), merge facts
        (47, [
            12,   # CU Intern Hiring
            16,   # hiring interns
            66,   # CU Internship Recruitment Process
            67,   # CU Internship Job Description Requirements
            68,   # CU Internship Deadline
        ]),

        # Sitetracker internship (general company): keep 69
        (69, [70, 71, 72, 73, 74]),

        # Scout outage: keep 46
        (46, [13, 38]),

        # Subrosa system prompt: keep 157
        (157, [185, 187]),

        # Subrosa config: keep 188
        (188, [
            194,  # config folder contents
            192,  # config folder structure
            191,  # configuration management
        ]),

        # Scout channels: keep 177
        (177, [178]),  # Scout operations channels → Scout channels

        # Compass project: keep 120
        (120, [
            256,  # Compass NL-to-SOQL orchestrator
            257,  # Compass query tool registration
            166,  # Compass architecture
        ]),

        # Evals project: keep 126
        (126, [
            104,  # Scout Evals
            108,  # Scout Org-Specific Evals
            109,  # Scout Eval UI
            259,  # Invoice evaluation taxonomy
        ]),

        # OpenClaw: consolidate
        (58, [59, 61]),

        # Communication style: keep 14
        (14, [52, 54, 64, 134, 95]),

        # User workflow / work style: keep 42
        (42, [65, 40, 60]),

        # Management philosophy: keep 100
        (100, [81, 82, 101, 79, 80]),

        # Attention queue: keep 27
        (27, [28, 31, 29]),

        # VP task management: keep 18
        (18, [19, 20, 21, 24, 30, 39, 23, 26, 15]),

        # Slack briefing insights: keep 266
        (266, [267, 268, 274, 276, 308]),

        # Team workflow / Yuval tension: keep 167
        (167, [281, 282, 279, 169]),

        # Brock-Yuval insight: keep 284
        (284, []),

        # Scout risks: keep 133
        (133, [107, 110, 271, 270, 264]),

        # subrosa setup: keep 96
        (96, [94, 97]),

        # Memory quality: keep 172
        (172, [175, 173, 171, 138, 8, 17, 137, 170, 83]),

        # Scheduler: keep 147
        (147, [148, 149, 150, 151]),
    ]

    for keep_id, delete_ids in merges:
        if not delete_ids:
            continue
        keep_name = conn.execute("SELECT name FROM topics WHERE id = ?", (keep_id,)).fetchone()[0]
        for del_id in delete_ids:
            row = conn.execute("SELECT name FROM topics WHERE id = ? AND active = 1", (del_id,)).fetchone()
            if row:
                print(f"  Merging '{row[0]}' (id={del_id}) → '{keep_name}' (id={keep_id})")
                merge_topic_into(conn, del_id, keep_id)


def phase3_delete_stale_and_transient(conn):
    """Delete topics that are stale, transient, or purely implementation details."""
    print("\n=== Phase 3: Delete stale/transient/implementation-detail topics ===")

    delete_ids = [
        # Implementation debugging (no longer relevant)
        92,   # aiosqlite (install instruction)
        91,   # aiosqlite dependency
        93,   # Python virtual environment troubleshooting
        90,   # store_management_memories (working with script)
        87,   # store_management_memories.py script

        # Stale/completed state
        35,   # restart message (didn't work, 0.3 confidence)
        160,  # rosy system (APScheduler, covered by Subrosa)
        78,   # Background Jobs (covered by Subrosa)
        77,   # Orchestrator System (stable, no crashes - not useful)
        155,  # Bot status monitoring (/status refactored)
        154,  # Polling reliability improvements

        # Redundant with briefing.md
        272,  # #scout channel (in briefing)
        181,  # Scout Monitoring (in briefing)
        142,  # Scout Storage Product (in briefing)
        143,  # Scout Customer Pilots (in briefing, 4 duplicate facts)

        # One-time events that are done
        161,  # Group DM conversation (old event)
        162,  # Slack API limitation (old debugging)
        43,   # Q1 roadmap review (stale deadline)

        # Subrosa implementation details (not needed in memory)
        55,   # Subrosa Agent - Capabilities (in system prompt)
        56,   # Subrosa Agent - Deliverables (in system prompt)
        49,   # Subrosa Continuous Execution
        50,   # Subrosa Interactive Message Handling
        51,   # Subrosa Streaming Output Problem
        57,   # Streaming Output Architecture
        158,  # subrosa context assembly
        159,  # subrosa briefing document (redundant)
        156,  # Subrosa system (proactive messaging)
        152,  # Subrosa PRD (insight about PRD)
        190,  # old subrosa repository
        212,  # subrosa1 team structure (merged)

        # Scout weekly snapshots (ephemeral, not durable knowledge)
        296,  # Weekly PR throughput
        288,  # SCOUT project status
        297,  # Compass subsystem activity
        298,  # Auth and multi-tenant focus
        299,  # Eval taxonomy progress
        301,  # Geoff Manning responsibilities (weekly snapshot)
        300,  # Jared Onnen responsibilities (weekly snapshot)
        302,  # Kyle Morand responsibilities (weekly snapshot)
        303,  # Alistair Stark responsibilities (weekly snapshot)
        304,  # SharePoint connector development (weekly snapshot)
        305,  # Multi-connector routing infrastructure (weekly snapshot)
        289,  # Scott Taylor follow-up (stale action item)

        # Insights that are now stale or obvious
        85,   # execution_vs_planning (corrected behavior)
        186,  # briefing.md content issue (fixed)
        184,  # briefing.md organization (fixed)
        102,  # Decision-making approach (vague)
        292,  # Output streaming issue (known)
        247,  # subrosa1 architecture (merged/stale)
        249,  # skill auto-registration approach (implemented)
        255,  # Slack Group DMs (known limitation)
        246,  # Scout team analysis capability (done)
        215,  # subsystem ownership mapping (future plan)

        # Meta-memories about memory system (self-referential noise)
        84,   # atilio/documentation

        # Team data structure details
        199,  # Current team.json completeness
        200,  # Missing team data elements
        211,  # Team Structure (in briefing)
        213,  # team.json data fields
        224,  # Scout team members (stale snapshot)

        # Specific ticket references (ephemeral)
        115,  # SCOUT-1563 (captured in evals project)
        117,  # taxonomy bottleneck (captured in evals)
        263,  # Open technical issues

        # Redundant preferences (covered by Communication Style merge)
        53,   # User - Work Categorization
        198,  # Assistant improvement priority
        103,  # Documentation and learning (0.6 confidence)

        # Implementation details for skill system
        250,  # skill system implementation
        253,  # skill routing in Telegram
        248,  # subrosa1 project structure

        # Band team facts (in team.json, not needed in memory)
        238,  # Incubus team
        239,  # Pink Floyd team
        240,  # The Killers team
        241,  # B-52s team
        242,  # RHCP team
        243,  # Queen team

        # Individual team member entries that are just "X is on Y team" (in team.json)
        225,  # Hugo
        226,  # Behan
        229,  # Jemil
        230,  # Hager Shahin
        231,  # Michael
        232,  # Tom Sziler
        233,  # Uma
        234,  # Kyle (just "member of RHCP")
        235,  # Santiago
        227,  # Abe Dolinger
        228,  # Alistair Stark (person - just team membership)

        # Stale project references
        258,  # SharePoint browser UI (PR snapshot)
        260,  # PDF image extraction (PR snapshot)
        261,  # Legacy Salesforce columns removal (PR snapshot)
        262,  # Login UI and sign-in flows (PR snapshot)

        # Misc transient
        307,  # Integration Status (just checked, transient)
        286,  # API token costs (ephemeral data point)
        287,  # Token governance (insight from one convo)
        290,  # Scout briefing generation (preference dup)
        294,  # system prompt behavior (preference dup)
        10,   # Subrosa user (vague, old)
        7,    # context_management (old preference)

        # Already-implemented features
        34,   # Subrosa restart notification
        144,  # briefing configuration
        145,  # briefing schedule
        182,  # Briefing Location
        183,  # DMs priority (in briefing.md)
        176,  # Scout briefings preference
        269,  # Scout Briefing preference
        111,  # Scout Taxonomy Work preference

        # Stale action items
        11,   # HR Agent (close out)
        5,    # User's Team (stale count)
        44,   # PR backlog process
        37,   # Jira Cleanup Need (dup of 45)
        45,   # In Progress ticket cleanup (in risk register)
        165,  # Brock's visibility initiative (context captured in Yuval situation)
        164,  # Langfuse integration work (stale snapshot)
        163,  # Scout team culture (captured elsewhere)
        285,  # Yuval situation (captured in Yuval person + insights)

        # Redundant Subrosa preferences
        86,   # Claude Agent (plan mode pref - generic)
        283,  # Claude models (Opus preference)

        # People with minimal info now merged elsewhere
        209,  # Akanksha (1 fact, team membership)
        208,  # Brittany Cruz (merged GitHub handle elsewhere)
    ]

    for tid in delete_ids:
        row = conn.execute("SELECT name, topic_type FROM topics WHERE id = ? AND active = 1", (tid,)).fetchone()
        if row:
            soft_delete_topic(conn, tid, f"{row[1]}: {row[0]}")


def phase4_deduplicate_facts(conn):
    """Remove duplicate/near-duplicate facts within surviving topics."""
    print("\n=== Phase 4: Deduplicate facts within topics ===")

    # Get all active topics with their facts
    topics = conn.execute(
        "SELECT t.id, t.name FROM topics t WHERE t.active = 1"
    ).fetchall()

    total_removed = 0
    for topic_id, topic_name in topics:
        facts = get_facts_for_topic(conn, topic_id)
        if len(facts) <= 1:
            continue

        # Find duplicates: facts that say essentially the same thing
        seen = []
        to_delete = []

        for fact_id, fact_text, confidence in sorted(facts, key=lambda x: -x[2]):
            # Keep highest confidence version
            normalized = fact_text.lower().strip().rstrip('.')

            is_dup = False
            for seen_text in seen:
                # Check for high overlap
                seen_words = set(seen_text.split())
                fact_words = set(normalized.split())
                if len(seen_words) == 0 or len(fact_words) == 0:
                    continue
                overlap = len(seen_words & fact_words) / min(len(seen_words), len(fact_words))
                if overlap > 0.7:
                    is_dup = True
                    break

            if is_dup:
                to_delete.append((fact_id, fact_text[:80]))
            else:
                seen.append(normalized)

        for fact_id, preview in to_delete:
            delete_fact(conn, fact_id, f"[{topic_name}] dup: {preview}")
            total_removed += 1

    print(f"  Removed {total_removed} duplicate facts")


def phase5_prune_low_value_facts(conn):
    """Remove low-confidence and low-value facts."""
    print("\n=== Phase 5: Prune low-confidence facts ===")

    # Delete facts with very low confidence
    cur = conn.execute(
        "SELECT tf.id, t.name, tf.fact FROM topic_facts tf "
        "JOIN topics t ON tf.topic_id = t.id "
        "WHERE t.active = 1 AND tf.confidence <= 0.6"
    )
    low_conf = cur.fetchall()

    for fact_id, topic_name, fact_text in low_conf:
        delete_fact(conn, fact_id, f"[{topic_name}] low confidence: {fact_text[:60]}")

    print(f"  Removed {len(low_conf)} low-confidence facts")

    # Delete topics that now have zero facts
    orphans = conn.execute(
        "SELECT t.id, t.name FROM topics t "
        "WHERE t.active = 1 AND NOT EXISTS "
        "(SELECT 1 FROM topic_facts tf WHERE tf.topic_id = t.id)"
    ).fetchall()

    for tid, name in orphans:
        soft_delete_topic(conn, tid, f"orphan (no facts remain): {name}")

    print(f"  Removed {len(orphans)} orphaned topics")


def main():
    print("=== Memory Consolidation (Aggressive) ===")
    print(f"Database: {DB_PATH}")

    backup()
    conn = connect()

    before_topics = count_active(conn)
    before_facts = count_active_facts(conn)
    print(f"\nBefore: {before_topics} topics, {before_facts} facts")

    phase1_merge_person_duplicates(conn)
    phase2_merge_duplicate_topics(conn)
    phase3_delete_stale_and_transient(conn)
    phase4_deduplicate_facts(conn)
    phase5_prune_low_value_facts(conn)

    conn.commit()

    after_topics = count_active(conn)
    after_facts = count_active_facts(conn)

    print(f"\n=== Results ===")
    print(f"Before: {before_topics} topics, {before_facts} facts")
    print(f"After:  {after_topics} topics, {after_facts} facts")
    print(f"Removed: {before_topics - after_topics} topics, {before_facts - after_facts} facts")

    # Show surviving topics
    print(f"\n=== Surviving Topics ({after_topics}) ===")
    cur = conn.execute(
        "SELECT t.id, t.name, t.topic_type, COUNT(tf.id) as fact_count "
        "FROM topics t LEFT JOIN topic_facts tf ON t.id = tf.topic_id "
        "WHERE t.active = 1 GROUP BY t.id ORDER BY t.topic_type, t.name"
    )
    for tid, name, ttype, fcount in cur.fetchall():
        print(f"  [{ttype:10s}] {name} ({fcount} facts)")

    conn.close()
    print(f"\nBackup at: {BACKUP_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
