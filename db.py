"""
db.py
-----
Supabase-backed data layer for EduTwin.

Replaces the old CSV (enhanced_students_dataset.csv) + JSON
(student_memory.json) persistence with Postgres tables on Supabase:

  students        — one row per student (metadata)
  student_topics  — one row per (student, topic) knowledge/behavior record
  student_memory  — append-only Live Learner Profile snapshot history
  user_profiles   — links a Supabase Auth user to a role + student_id

Uses the service_role key so it bypasses Row Level Security — this module
is only ever imported server-side, never exposed to the client.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv  # type: ignore[import-untyped]
from supabase import create_client, Client  # type: ignore[import-untyped]

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

COURSE_TOPICS = [
    "linear_algebra", "probability", "python_basics",
    "data_structures", "ai_ml", "databases", "os_concepts", "networking",
]

_TOPIC_DEFAULTS = {
    "score": 0.50, "time_spent": 60.0, "attempts": 2.0,
    "confidence": 0.50, "engagement": 0.50, "fatigue": 0.30,
}

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set "
                "(see .env.example)."
            )
        _client = create_client(url, key)
    return _client


# ═══════════════════════════════════════════════════════════
# STUDENTS
# ═══════════════════════════════════════════════════════════

def list_students() -> list[dict]:
    """Lightweight list of all students (id, name, year, branch, archetype, avatar_id)."""
    res = (
        get_client()
        .table("students")
        .select("id, name, year, branch, archetype, avatar_id")
        .order("id")
        .execute()
    )
    return [
        {
            "student_id": row["id"],
            "name": row["name"],
            "year": row["year"],
            "branch": row["branch"],
            "archetype": row["archetype"],
            "avatar_id": row.get("avatar_id", "rogue"),
        }
        for row in res.data
    ]


def get_all_student_ids() -> list[int]:
    res = get_client().table("students").select("id").order("id").execute()
    return [row["id"] for row in res.data]


def get_student_meta(student_id: int) -> dict | None:
    res = (
        get_client()
        .table("students")
        .select("id, name, year, branch, archetype, avatar_id, auth_user_id")
        .eq("id", student_id)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def get_student_topics(student_id: int) -> dict[str, dict]:
    """Returns {topic: {score, time_spent, attempts, confidence, engagement, fatigue}}."""
    res = (
        get_client()
        .table("student_topics")
        .select("*")
        .eq("student_id", student_id)
        .execute()
    )
    return {row["topic"]: row for row in res.data}


def add_student(student_data: dict) -> int:
    """
    Inserts a new student + all 8 topic rows (missing topics get neutral
    defaults). Returns the new student_id.
    """
    client = get_client()
    meta = {
        "name": student_data.get("name", "New Student"),
        "year": int(student_data.get("year", 1)),
        "branch": student_data.get("branch", "CSE"),
        "archetype": student_data.get("archetype", "unknown"),
        "avatar_id": student_data.get("avatar_id", "rogue"),
    }
    inserted = client.table("students").insert(meta).execute()
    new_id = inserted.data[0]["id"]

    topic_map = {t["topic"]: t for t in student_data.get("topics", []) if "topic" in t}
    rows = []
    for topic in COURSE_TOPICS:
        td = topic_map.get(topic, {})
        row = {"student_id": new_id, "topic": topic}
        for field, default in _TOPIC_DEFAULTS.items():
            row[field] = float(td.get(field, default))
        rows.append(row)
    client.table("student_topics").insert(rows).execute()

    return new_id


def update_student_meta(student_id: int, updates: dict) -> None:
    fields = {
        k: v for k, v in updates.items()
        if k in ("name", "year", "branch", "archetype", "avatar_id")
    }
    if fields:
        get_client().table("students").update(fields).eq("id", student_id).execute()


def update_student_topic(student_id: int, topic: str, updates: dict) -> None:
    fields = {k: v for k, v in updates.items() if k in _TOPIC_DEFAULTS}
    if fields:
        get_client().table("student_topics").update(fields).eq(
            "student_id", student_id
        ).eq("topic", topic).execute()


def delete_student(student_id: int) -> bool:
    """Deletes the student row; student_topics/student_memory cascade via FK."""
    res = get_client().table("students").delete().eq("id", student_id).execute()
    return len(res.data) > 0


def student_exists(student_id: int) -> bool:
    return get_student_meta(student_id) is not None


# ═══════════════════════════════════════════════════════════
# MEMORY (profile snapshot history)
# ═══════════════════════════════════════════════════════════

def get_latest_profile(student_id: int) -> dict | None:
    res = (
        get_client()
        .table("student_memory")
        .select("profile")
        .eq("student_id", student_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0]["profile"] if res.data else None


def append_profile(student_id: int, profile: dict) -> None:
    get_client().table("student_memory").insert(
        {"student_id": student_id, "profile": profile}
    ).execute()


def get_profile_history(student_id: int) -> list[dict]:
    res = (
        get_client()
        .table("student_memory")
        .select("profile, created_at")
        .eq("student_id", student_id)
        .order("created_at")
        .execute()
    )
    return [row["profile"] for row in res.data]


def invalidate_cache(student_id: int) -> None:
    """No-op placeholder kept for call-site compatibility — memory is append-only."""
    return None


# ═══════════════════════════════════════════════════════════
# QUIZ ATTEMPTS (the objective signal — see digital_twin.py)
# ═══════════════════════════════════════════════════════════

def record_quiz_attempt(
    student_id: int, topic: str, question: str, answer: str,
    correctness: float, feedback: str,
) -> None:
    get_client().table("quiz_attempts").insert({
        "student_id": student_id,
        "topic": topic,
        "question": question,
        "answer": answer,
        "correctness": correctness,
        "feedback": feedback,
    }).execute()


def get_quiz_stats(student_id: int) -> dict:
    """Returns {count, avg_correctness} across all of this student's graded quizzes."""
    res = (
        get_client()
        .table("quiz_attempts")
        .select("correctness")
        .eq("student_id", student_id)
        .execute()
    )
    values = [row["correctness"] for row in res.data]
    return {
        "count": len(values),
        "avg_correctness": (sum(values) / len(values)) if values else None,
    }


# ═══════════════════════════════════════════════════════════
# USER PROFILES (auth role / student linkage)
# ═══════════════════════════════════════════════════════════

def get_user_profile(user_id: str) -> dict | None:
    res = (
        get_client()
        .table("user_profiles")
        .select("id, role, student_id")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def create_user_profile(user_id: str, role: str, student_id: int | None = None) -> dict:
    row = {"id": user_id, "role": role, "student_id": student_id}
    res = get_client().table("user_profiles").upsert(row).execute()
    return res.data[0]


def link_student_to_user(student_id: int, user_id: str) -> None:
    get_client().table("students").update({"auth_user_id": user_id}).eq(
        "id", student_id
    ).execute()
