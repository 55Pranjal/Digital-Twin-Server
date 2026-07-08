"""
migrate_to_supabase.py
-----------------------
One-off migration: imports the legacy enhanced_students_dataset.csv +
student_memory.json into the Supabase tables defined in schema.sql.

Prerequisites
-------------
1. Create a Supabase project and run schema.sql in the SQL editor.
2. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env (see .env.example).
3. `enhanced_students_dataset.csv` and `student_memory.json` must still be
   present in this directory (they are safe to delete after migrating).

Run once:
    python migrate_to_supabase.py

Safe to re-run: students whose id already exists in Supabase are skipped.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

import db

CSV_FILE = "enhanced_students_dataset.csv"
MEMORY_FILE = "student_memory.json"
BATCH_SIZE = 500


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def migrate_students_and_topics() -> set[int]:
    if not Path(CSV_FILE).exists():
        print(f"  {CSV_FILE} not found — skipping student/topic migration.")
        return set()

    df = pd.read_csv(CSV_FILE)
    client = db.get_client()

    existing = {row["id"] for row in client.table("students").select("id").execute().data}
    all_ids = set(int(x) for x in df["student_id"].unique())
    to_migrate = sorted(all_ids - existing)

    if not to_migrate:
        print("  All students already present in Supabase — nothing to do.")
        return existing

    print(f"  Migrating {len(to_migrate)} students (skipping {len(existing)} already present)…")

    meta_rows = []
    for sid in to_migrate:
        first = df[df["student_id"] == sid].iloc[0]
        meta_rows.append({
            "id": sid,
            "name": str(first["name"]),
            "year": int(first["year"]),
            "branch": str(first["branch"]),
            "archetype": str(first["archetype"]),
        })
    for batch in _chunks(meta_rows, BATCH_SIZE):
        client.table("students").insert(batch).execute()

    topic_rows = []
    for _, row in df[df["student_id"].isin(to_migrate)].iterrows():
        topic_rows.append({
            "student_id": int(row["student_id"]),
            "topic":      row["topic"],
            "score":      float(row["score"]),
            "time_spent": float(row["time_spent"]),
            "attempts":   float(row["attempts"]),
            "confidence": float(row["confidence"]),
            "engagement": float(row["engagement"]),
            "fatigue":    float(row["fatigue"]),
        })
    for batch in _chunks(topic_rows, BATCH_SIZE):
        client.table("student_topics").insert(batch).execute()

    print(f"  Inserted {len(meta_rows)} students, {len(topic_rows)} topic rows.")
    return existing | set(to_migrate)


def migrate_memory(migrated_ids: set[int]) -> None:
    if not Path(MEMORY_FILE).exists():
        print(f"  {MEMORY_FILE} not found — skipping memory migration.")
        return

    with open(MEMORY_FILE, "r") as f:
        memory: dict[str, list[dict]] = json.load(f)

    client = db.get_client()
    rows = []
    for sid_str, snapshots in memory.items():
        sid = int(sid_str)
        if sid not in migrated_ids:
            continue
        for snap in snapshots:
            ts = snap.get("timestamp")
            created_at = (
                datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                if ts else None
            )
            row = {"student_id": sid, "profile": snap}
            if created_at:
                row["created_at"] = created_at
            rows.append(row)

    if not rows:
        print("  No memory snapshots to migrate.")
        return

    print(f"  Migrating {len(rows)} profile snapshots…")
    for batch in _chunks(rows, BATCH_SIZE):
        client.table("student_memory").insert(batch).execute()
    print("  Done.")


def main():
    print("EduTwin — CSV/JSON → Supabase migration\n")
    migrated_ids = migrate_students_and_topics()
    migrate_memory(migrated_ids)

    if migrated_ids:
        max_id = max(migrated_ids)
        print(
            "\nMigration complete. Run this once in the Supabase SQL editor so "
            "future sign-ups don't collide with migrated ids:\n\n"
            f"  select setval(pg_get_serial_sequence('students','id'), {max_id});\n"
        )


if __name__ == "__main__":
    main()
