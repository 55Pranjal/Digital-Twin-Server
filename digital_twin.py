"""
digital_twin.py
---------------
Core EduTwin engine.

Capabilities:
  1.  Live Learner Profile (LLP) builder + human-readable summary
  2.  Weakness diagnosis (rule-based + LLM-reasoned)
  3.  Personalized concept explanation  (LLM, profile-injected)
  4.  Performance prediction / risk scoring
  5.  Exam answer simulation
  6.  Intervention simulation + best-strategy selection
  7.  Study plan generation
  8.  Feedback loop (real-outcome correction)
  9.  Temporal memory with JSON persistence
  10. RAG retrieval (FAISS + sentence-transformers, optional)
  11. Progress charts (line + radar)
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
from dotenv import load_dotenv  # type: ignore[import-untyped]
import google.generativeai as genai  # type: ignore[import-untyped]

env_path = Path(r"C:\College\Digital Twin(Server side)\.env")
load_dotenv(dotenv_path=env_path)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MEMORY_FILE    = "student_memory.json"
CSV_FILE       = "enhanced_students_dataset.csv"
TARGET_SCORE   = 0.70
WEAK_THRESHOLD = 0.50

COURSE_TOPICS = [
    "linear_algebra", "probability", "python_basics",
    "data_structures", "ai_ml", "databases", "os_concepts", "networking",
]

COURSE_MATERIAL = {
    "linear_algebra":  "Vectors, matrices, eigenvalues, dot products, transformations.",
    "probability":     "Bayes theorem, distributions, expectation, variance, CLT.",
    "python_basics":   "Variables, loops, functions, OOP, list comprehensions.",
    "data_structures": "Arrays, linked lists, trees, graphs, heaps, hash maps.",
    "ai_ml":           "Supervised/unsupervised learning, gradient descent, neural nets.",
    "databases":       "SQL, ER diagrams, normalisation, transactions, indexing.",
    "os_concepts":     "Processes, threads, scheduling, memory management, file systems.",
    "networking":      "OSI model, TCP/IP, DNS, HTTP, sockets, routing.",
}

INTERVENTION_EFFECTS = {
    "concept_learning":  {"score_boost": 0.08, "conf_boost": 0.05, "fatigue_delta": +0.05},
    "increase_practice": {"score_boost": 0.10, "conf_boost": 0.12, "fatigue_delta": +0.10},
    "reduce_fatigue":    {"score_boost": 0.03, "conf_boost": 0.08, "fatigue_delta": -0.20},
    "peer_study":        {"score_boost": 0.07, "conf_boost": 0.10, "fatigue_delta": -0.05},
    "tutoring_session":  {"score_boost": 0.12, "conf_boost": 0.15, "fatigue_delta": +0.02},
}

TOPIC_DIFFICULTY = {"ai_ml": 0.5, "probability": 0.7, "linear_algebra": 0.8}

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
_df: pd.DataFrame | None = None
student_memory: dict[str, list[Any]] = {}

_rag_index = None
_rag_docs: list[str] = []
_embedder  = None


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════
def load_dataset() -> pd.DataFrame:
    global _df
    if _df is None:
        if not Path(CSV_FILE).exists():
            raise FileNotFoundError(
                f"Dataset '{CSV_FILE}' not found. Run:  python generate_data.py"
            )
        _df = pd.read_csv(CSV_FILE)
    return _df


# ═══════════════════════════════════════════════════════════
# STUDENT CRUD
# ═══════════════════════════════════════════════════════════

def list_students() -> list[dict]:
    """Return a lightweight list of all students (id, name, year, branch, archetype)."""
    df = load_dataset()
    meta_cols = ["student_id", "name", "year", "branch", "archetype"]
    available = [c for c in meta_cols if c in df.columns]
    return (
        df[available]
        .drop_duplicates(subset=["student_id"])
        .sort_values("student_id")
        .to_dict(orient="records")
    )


def add_student(student_data: dict) -> dict:
    """
    Add a new student to the CSV dataset.

    Expected keys in student_data
    ──────────────────────────────
    name       str         (required)
    year       int         (default 1)
    branch     str         (default "CSE")
    archetype  str         (default "unknown")
    topics     list[dict]  each dict may contain:
                 topic, score, time_spent, attempts,
                 confidence, engagement, fatigue
                 Missing topic entries are auto-filled with neutral defaults.

    Returns the freshly built profile dict for the new student.
    """
    global _df
    df = load_dataset()

    # Auto-assign next available student_id
    new_id = int(df["student_id"].max()) + 1 if not df.empty else 1

    topic_map = {t["topic"]: t for t in student_data.get("topics", []) if "topic" in t}

    rows: list[dict] = []
    for topic in COURSE_TOPICS:
        td = topic_map.get(topic, {})
        rows.append({
            "student_id":  new_id,
            "name":        student_data.get("name", f"Student {new_id}"),
            "year":        int(student_data.get("year", 1)),
            "branch":      student_data.get("branch", "CSE"),
            "archetype":   student_data.get("archetype", "unknown"),
            "topic":       topic,
            "score":       float(td.get("score",       0.50)),
            "time_spent":  float(td.get("time_spent",  60.0)),
            "attempts":    float(td.get("attempts",    2.0)),
            "confidence":  float(td.get("confidence",  0.50)),
            "engagement":  float(td.get("engagement",  0.50)),
            "fatigue":     float(td.get("fatigue",     0.30)),
        })

    _df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    _df.to_csv(CSV_FILE, index=False)

    profile = build_profile(new_id, force_rebuild=True)
    update_memory(new_id, profile)
    save_memory()
    return profile


def delete_student(student_id: int | str) -> bool:
    """
    Remove a student from the CSV and from in-memory history.

    Returns True if the student was found and deleted, False if not found.
    """
    global _df
    df  = load_dataset()
    sid = int(student_id)

    if sid not in df["student_id"].values:
        return False

    _df = df[df["student_id"] != sid].reset_index(drop=True)
    _df.to_csv(CSV_FILE, index=False)

    str_sid = str(student_id)
    if str_sid in student_memory:
        student_memory.pop(str_sid, None)
    save_memory()
    return True


def update_student(student_id: int | str, updates: dict) -> dict:
    """
    Partially update a student's metadata and/or per-topic values.

    Accepted keys in updates
    ────────────────────────
    name, year, branch, archetype   — metadata fields
    topics  list[dict]              — each dict: {topic, score?, time_spent?,
                                      attempts?, confidence?, engagement?, fatigue?}
                                      Only supplied fields are overwritten.

    Returns the rebuilt profile dict after saving.
    Raises ValueError if the student does not exist.
    """
    global _df
    load_dataset()              # ensures _df is populated
    assert _df is not None, "Dataset failed to load"
    df: pd.DataFrame = _df          # local non-None reference for type checker
    sid = int(student_id)

    if sid not in df["student_id"].values:  # type: ignore[index]
        raise ValueError(f"Student ID {sid} not found in dataset.")

    base_mask = df["student_id"] == sid  # type: ignore[index]

    # ── metadata fields ──────────────────────────────────────
    for field in ("name", "year", "branch", "archetype"):
        if field in updates:
            df.loc[base_mask, field] = updates[field]  # type: ignore[index]

    # ── per-topic fields ─────────────────────────────────────
    numeric_fields = ("score", "time_spent", "attempts", "confidence", "engagement", "fatigue")
    for topic_upd in updates.get("topics", []):
        topic = topic_upd.get("topic")
        if not topic:
            continue
        topic_mask = base_mask & (df["topic"] == topic)  # type: ignore[index]
        if not topic_mask.any():
            warnings.warn(f"Topic '{topic}' not found for student {sid}; skipping.")
            continue
        for field in numeric_fields:
            if field in topic_upd:
                df.loc[topic_mask, field] = float(topic_upd[field])  # type: ignore[index]

    _df = df
    _df.to_csv(CSV_FILE, index=False)

    # Invalidate cached profile so next call rebuilds from fresh CSV data
    str_sid = str(student_id)
    if str_sid in student_memory:
        student_memory[str_sid] = []

    profile = build_profile(sid, force_rebuild=True)
    update_memory(sid, profile)
    save_memory()
    return profile


# ═══════════════════════════════════════════════════════════
# MEMORY
# ═══════════════════════════════════════════════════════════
def load_memory() -> None:
    global student_memory
    try:
        with open(MEMORY_FILE, "r") as f:
            student_memory = json.load(f)
    except FileNotFoundError:
        student_memory = {}
    except Exception as exc:
        warnings.warn(f"Could not load memory: {exc}")
        student_memory = {}


def save_memory() -> None:
    with open(MEMORY_FILE, "w") as f:
        json.dump(student_memory, f, indent=2)


def update_memory(student_id: int | str, profile: dict) -> None:
    sid = str(student_id)
    if sid not in student_memory:
        student_memory[sid] = []
    copy = json.loads(json.dumps(profile))
    copy["timestamp"] = time.time()
    student_memory[sid].append(copy)


# ═══════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════
def _clean(x: float) -> float:
    return float(round(x, 4))  # type: ignore[arg-type]

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _sep(char: str = "─", width: int = 60) -> str:
    return char * width


# ═══════════════════════════════════════════════════════════
# PROFILE BUILDER
# ═══════════════════════════════════════════════════════════
def build_profile(student_id: int | str, force_rebuild: bool = False) -> dict:
    """
    Build the Live Learner Profile (LLP) for a student.
    Returns latest cached profile unless force_rebuild=True.
    """
    sid = str(student_id)
    df  = load_dataset()

    if not force_rebuild and sid in student_memory and student_memory[sid]:
        return student_memory[sid][-1]

    student_df = df[df["student_id"] == int(sid)]
    if student_df.empty:
        raise ValueError(f"Student ID {sid} not found in dataset.")

    meta = student_df.iloc[0]

    profile: dict[str, Any] = {
        "student_id":    sid,
        "name":          meta.get("name",      f"Student {sid}"),
        "year":          int(meta.get("year",   1)),
        "branch":        meta.get("branch",    "CSE"),
        "archetype":     meta.get("archetype", "unknown"),
        "knowledge":     {},
        "weak_topics":   [],
        "strong_topics": [],
        "behavior":      {},
        "psychology":    {},
        "summary":       "",
    }

    for _, row in student_df.iterrows():
        topic = row["topic"]
        score = _clean(row["score"])
        profile["knowledge"][topic] = score
        if score < WEAK_THRESHOLD:
            profile["weak_topics"].append(topic)
        elif score >= TARGET_SCORE:
            profile["strong_topics"].append(topic)

    profile["behavior"] = {
        "avg_time":       _clean(student_df["time_spent"].mean()),
        "avg_attempts":   _clean(student_df["attempts"].mean()),
        "avg_confidence": _clean(student_df["confidence"].mean()),
        "engagement":     _clean(student_df["engagement"].mean()),
        "fatigue":        _clean(student_df["fatigue"].mean()),
    }

    profile = _add_psychology(profile)
    profile["summary"] = generate_llp_summary(profile)
    return profile


def _add_psychology(profile: dict) -> dict:
    conf    = profile["behavior"]["avg_confidence"]
    fatigue = profile["behavior"]["fatigue"]
    engage  = profile["behavior"]["engagement"]

    profile["psychology"] = {
        "confidence_level": "low"    if conf    < 0.40 else ("medium" if conf    < 0.65 else "high"),
        "focus":            "low"    if fatigue > 0.60 else ("medium" if fatigue > 0.40 else "good"),
        "engagement_level": "low"    if engage  < 0.40 else ("medium" if engage  < 0.65 else "high"),
        "learning_pace":    "slow"   if conf    < 0.40 else ("normal" if conf    < 0.70 else "fast"),
    }
    return profile


# ═══════════════════════════════════════════════════════════
# HUMAN-READABLE SUMMARY
# ═══════════════════════════════════════════════════════════
def generate_llp_summary(profile: dict) -> str:
    perf     = compute_performance(profile)
    strong   = profile.get("strong_topics", [])
    weak     = profile.get("weak_topics",   [])
    psych    = profile.get("psychology",    {})
    behavior = profile.get("behavior",      {})

    return (
        f"{profile['name']} is a Year-{profile['year']} {profile['branch']} student "
        f"with an overall average of {round(perf * 100, 1)}%. "  # type: ignore[arg-type]
        f"Strong in: {', '.join(strong) if strong else 'none yet'}. "
        f"Weak in: {', '.join(weak) if weak else 'none'}. "
        f"Confidence: {psych.get('confidence_level', 'unknown')}, "
        f"engagement: {psych.get('engagement_level', 'unknown')}, "
        f"focus: {psych.get('focus', 'unknown')}. "
        f"Avg time/topic: {behavior.get('avg_time', 0):.1f} min, "
        f"avg attempts: {behavior.get('avg_attempts', 0):.1f}."
    )


# ═══════════════════════════════════════════════════════════
# PERFORMANCE & GOAL
# ═══════════════════════════════════════════════════════════
def compute_performance(profile: dict) -> float:
    knowledge = profile.get("knowledge", {})
    if not knowledge:
        return 0.0
    return round(sum(knowledge.values()) / len(knowledge), 4)  # type: ignore[arg-type]


def check_goal(profile: dict) -> bool:
    return all(v >= TARGET_SCORE for v in profile["knowledge"].values())


# ═══════════════════════════════════════════════════════════
# RISK PREDICTION
# ═══════════════════════════════════════════════════════════
def predict_struggle(profile: dict) -> dict:
    weak_ratio = len(profile["weak_topics"]) / max(len(profile["knowledge"]), 1)
    low_conf   = 1 - profile["behavior"]["avg_confidence"]
    fatigue    = profile["behavior"]["fatigue"]
    score      = 0.40 * weak_ratio + 0.30 * low_conf + 0.30 * fatigue

    if score > 0.60:
        risk = "High Risk"
    elif score > 0.35:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {"risk": risk, "score": round(score, 4)}  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════
def recommend_actions(profile: dict) -> list[str]:
    recs    = []
    psych   = profile.get("psychology", {})
    behavior = profile.get("behavior",  {})
    weak    = profile.get("weak_topics", [])

    if weak:
        recs.append(f"📚 Focus on weak topics: {', '.join(weak)}")
    if psych.get("confidence_level") in ("low", "medium"):
        recs.append("🔁 Revise fundamentals and work through easier practice problems first.")
    if behavior.get("fatigue", 0) > 0.55:
        recs.append("😴 High fatigue — reduce daily study hours and add short breaks.")
    if behavior.get("engagement", 0) > 0.70:
        recs.append("🚀 High engagement — ready for advanced problems and projects.")
    if behavior.get("avg_attempts", 0) > 3:
        recs.append("🛠  Multiple attempts needed — try spaced repetition.")
    if psych.get("learning_pace") == "slow":
        recs.append("⏱  Slow pace — allocate extra time for ai_ml and probability.")
    if not recs:
        recs.append("✅ Student is on track — maintain current study patterns.")

    return recs


# ─────────────────────────────────────────────
# LLM (Gemini API)
# ─────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY") or "")

def call_llm(prompt: str, system: str = "") -> str:
    """
    Calls Google Gemini model using API key from .env
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")

        full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt

        response = model.generate_content(full_prompt)

        return response.text.strip()

    except Exception as exc:
        return f"[ERROR] Gemini API: {exc}"
    

# ═══════════════════════════════════════════════════════════
# LLM CAPABILITY 1 — Weakness Diagnosis
# ═══════════════════════════════════════════════════════════
def diagnose_weaknesses_with_llm(profile: dict) -> str:
    knowledge_str = "\n".join(
        f"  - {topic}: {round(score * 100, 1)}%"
        for topic, score in sorted(profile["knowledge"].items(), key=lambda x: x[1])
    )
    system = (
        "You are an expert academic tutor analysing a university student's performance. "
        "Be concise, specific, and constructive."
    )
    prompt = (
        f"Student: {profile['name']} | Year {profile['year']} {profile['branch']}\n"
        f"Confidence: {profile['psychology']['confidence_level']} | "
        f"Focus: {profile['psychology']['focus']}\n\n"
        f"Topic scores:\n{knowledge_str}\n\n"
        f"Identify the 2-3 most critical weak areas this student must address before "
        f"the next assessment. For each, briefly explain WHY it is weak and what "
        f"they should do specifically. Under 200 words."
    )
    return call_llm(prompt, system)


# ═══════════════════════════════════════════════════════════
# LLM CAPABILITY 2 — Personalized Explanation
# ═══════════════════════════════════════════════════════════
def generate_personalized_explanation(profile: dict, concept: str) -> str:
    context = COURSE_MATERIAL.get(concept, "")
    rag     = retrieve_relevant_material(concept, top_k=2)
    if rag:
        context += " " + rag

    system = (
        "You are a highly adaptive university tutor. You tailor every explanation "
        "precisely to the individual student's background, confidence, and weak areas."
    )
    prompt = (
        f"Student profile:\n"
        f"  Name: {profile['name']} | Year {profile['year']} {profile['branch']}\n"
        f"  Overall avg: {round(compute_performance(profile) * 100, 1)}%\n"  # type: ignore[arg-type]
        f"  Weak topics: {', '.join(profile['weak_topics']) or 'none'}\n"
        f"  Strong topics: {', '.join(profile.get('strong_topics', [])) or 'none'}\n"
        f"  Confidence: {profile['psychology']['confidence_level']}\n"
        f"  Learning pace: {profile['psychology']['learning_pace']}\n"
        f"  Focus: {profile['psychology']['focus']}\n\n"
        f"Topic context: {context}\n\n"
        f"Explain '{concept}' for this specific student.\n"
        f"- If confidence is low or pace is slow: use simple analogies, avoid jargon.\n"
        f"- If they are strong: be technical and link to advanced ideas.\n"
        f"- Reference their weak topics where relevant.\n"
        f"Under 250 words."
    )
    return call_llm(prompt, system)


# ═══════════════════════════════════════════════════════════
# LLM CAPABILITY 3 — Performance Prediction
# ═══════════════════════════════════════════════════════════
def predict_performance_llm(profile: dict, upcoming_topic: str) -> str:
    system = "You are an academic performance analyst. Be analytical and data-driven."
    prompt = (
        f"Student: {profile['name']} | {profile['branch']} Year {profile['year']}\n"
        f"Profile summary: {profile['summary']}\n\n"
        f"Upcoming assessment topic: {upcoming_topic}\n"
        f"Current score in this topic: "
        f"{round(profile['knowledge'].get(upcoming_topic, 0) * 100, 1)}%\n\n"
        "Predict whether this student will score High (>70%), Medium (50-70%), or "
        "Low (<50%) on the upcoming quiz. Give a short justification citing specific "
        "profile data. End with: 'Prediction: High / Medium / Low'."
    )
    return call_llm(prompt, system)


# ═══════════════════════════════════════════════════════════
# LLM CAPABILITY 4 — Exam Answer Simulation
# ═══════════════════════════════════════════════════════════
def simulate_exam_answer(profile: dict, question: str, topic: str = "") -> str:
    topic_score = profile["knowledge"].get(topic, compute_performance(profile))
    system = (
        "You are roleplaying as a university student taking an exam. "
        "Write the answer exactly as this student would — including their knowledge "
        "gaps and misconceptions. Do NOT give a perfect answer."
    )
    prompt = (
        f"You are playing the role of this student:\n"
        f"  Name: {profile['name']}\n"
        f"  Score in '{topic}': {round(topic_score * 100, 1)}%\n"
        f"  Weak areas: {', '.join(profile['weak_topics']) or 'none'}\n"
        f"  Confidence: {profile['psychology']['confidence_level']}\n"
        f"  Learning pace: {profile['psychology']['learning_pace']}\n\n"
        f"Exam question: {question}\n\n"
        "Write the answer this student would actually give under exam conditions. "
        "If score < 50%, include clear mistakes. If > 70%, mostly correct with minor gaps. "
        "Stay in character. Max 200 words."
    )
    return call_llm(prompt, system)


# ═══════════════════════════════════════════════════════════
# LLM CAPABILITY 5 — Study Plan
# ═══════════════════════════════════════════════════════════
def generate_study_plan(profile: dict, days: int = 7) -> str:
    system = "You are an academic coach creating personalised study plans."
    prompt = (
        f"Create a {days}-day study plan for:\n"
        f"  {profile['summary']}\n\n"
        f"Constraints:\n"
        f"  - Fatigue level: {profile['psychology']['focus']} (schedule accordingly)\n"
        f"  - Engagement: {profile['psychology']['engagement_level']}\n"
        f"  - Prioritise weak topics: {', '.join(profile['weak_topics']) or 'none'}\n\n"
        "Format: Day 1: [topic] — [task] (duration). One line per day. "
        "Keep it realistic and specific."
    )
    return call_llm(prompt, system)


# ═══════════════════════════════════════════════════════════
# INTERVENTION SIMULATION
# ═══════════════════════════════════════════════════════════
def simulate_intervention(profile: dict, action: str) -> dict:
    effects = INTERVENTION_EFFECTS.get(
        action, {"score_boost": 0.05, "conf_boost": 0.0, "fatigue_delta": 0.0}
    )
    sim = json.loads(json.dumps(profile))

    for topic in sim["weak_topics"]:
        diff_mult = TOPIC_DIFFICULTY.get(topic, 1.0)
        sim["knowledge"][topic] = _clamp(
            sim["knowledge"][topic] + effects["score_boost"] * diff_mult
        )

    sim["behavior"]["avg_confidence"] = _clamp(
        sim["behavior"]["avg_confidence"] + effects["conf_boost"]
    )
    sim["behavior"]["fatigue"] = _clamp(
        sim["behavior"]["fatigue"] + effects["fatigue_delta"]
    )
    sim["weak_topics"]   = [t for t, v in sim["knowledge"].items() if v < WEAK_THRESHOLD]
    sim["strong_topics"] = [t for t, v in sim["knowledge"].items() if v >= TARGET_SCORE]
    sim["summary"]       = generate_llp_summary(sim)
    return sim


def run_multiple_simulations(profile: dict) -> dict[str, float]:
    return {
        action: compute_performance(simulate_intervention(profile, action))
        for action in INTERVENTION_EFFECTS
    }


def best_intervention(profile: dict) -> str:
    sims = run_multiple_simulations(profile)
    return max(sims, key=lambda k: sims[k])


# ═══════════════════════════════════════════════════════════
# FEEDBACK LOOP
# ═══════════════════════════════════════════════════════════
def update_from_real_outcome(profile: dict, actual_score: float) -> dict:
    predicted = profile["behavior"]["avg_confidence"]
    error     = actual_score - predicted
    profile["behavior"]["avg_confidence"] = _clamp(predicted + 0.2 * error)
    profile["summary"] = generate_llp_summary(profile)
    return profile


# ═══════════════════════════════════════════════════════════
# RAG — RETRIEVAL ENGINE  (optional FAISS)
# ═══════════════════════════════════════════════════════════
def _init_rag() -> bool:
    global _rag_index, _rag_docs, _embedder
    if _rag_index is not None:
        return True
    try:
        import faiss  # type: ignore[import-untyped]
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        _embedder  = SentenceTransformer("all-MiniLM-L6-v2")
        _rag_docs  = list(COURSE_MATERIAL.values())
        embeddings = _embedder.encode(_rag_docs, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        _rag_index = faiss.IndexFlatIP(embeddings.shape[1])
        _rag_index.add(embeddings.astype("float32"))
        return True
    except ImportError:
        return False


def retrieve_relevant_material(query: str, top_k: int = 2) -> str:
    if not _init_rag() or _embedder is None or _rag_index is None:
        return COURSE_MATERIAL.get(query, "")
    query_vec = _embedder.encode([query], convert_to_numpy=True)  # type: ignore[union-attr]
    query_vec = query_vec / np.linalg.norm(query_vec)  # type: ignore[operator]
    _, indices = _rag_index.search(query_vec.astype("float32"), top_k)  # type: ignore[union-attr]
    return " | ".join(_rag_docs[i] for i in indices[0] if i < len(_rag_docs))


# ═══════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════
def plot_subject_progress(student_id: int | str, save_path: str | None = None) -> None:
    sid     = str(student_id)
    history = student_memory.get(sid, [])
    if not history:
        print("  No history yet.")
        return

    _, ax = plt.subplots(figsize=(10, 5))
    for topic in history[0]["knowledge"]:
        vals = [step["knowledge"].get(topic, 0) for step in history]
        ax.plot(vals, marker="o", label=topic)

    ax.axhline(TARGET_SCORE, color="gray", linestyle="--", linewidth=0.8, label="Target (0.7)")
    ax.set_xlabel("Memory Step")
    ax.set_ylabel("Score")
    ax.set_title(f"Subject Progress — Student {sid}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Chart saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_radar(profile: dict, save_path: str | None = None) -> None:
    topics  = list(profile["knowledge"].keys())
    scores  = [profile["knowledge"][t] for t in topics]
    N       = len(topics)
    angles  = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    _, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.plot(angles, scores + [scores[0]], linewidth=1.5)
    ax.fill(angles, scores + [scores[0]], alpha=0.25)
    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], topics)  # type: ignore[arg-type]
    ax.set_ylim(0, 1)
    ax.axhline(TARGET_SCORE, color="red", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title(f"{profile['name']} — Knowledge Radar", pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Radar saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════
# CLASS-WIDE TEACHER OVERVIEW
# ═══════════════════════════════════════════════════════════
def teacher_overview(num_students: int | None = None) -> None:
    df = load_dataset()
    if num_students is None:
        num_students = int(df["student_id"].nunique())

    print(f"\n{_sep('═')}")
    print("  EDUTWIN — TEACHER OVERVIEW")
    print(_sep('═'))

    profiles = []
    for sid in range(1, num_students + 1):
        try:
            profiles.append(build_profile(sid, force_rebuild=True))
        except Exception:
            pass

    perfs = [compute_performance(p) for p in profiles]
    risks = [predict_struggle(p)["risk"] for p in profiles]

    print(f"\n  Students      : {len(profiles)}")
    print(f"  Class avg     : {round(np.mean(perfs) * 100, 1)}%")
    print(f"  🔴 High Risk  : {risks.count('High Risk')}")
    print(f"  🟠 Medium Risk: {risks.count('Medium Risk')}")
    print(f"  🟢 Low Risk   : {risks.count('Low Risk')}")
    print(f"  ✅ Goal met   : {sum(1 for p in profiles if check_goal(p))}")

    print(f"\n  {'ID':<5} {'Name':<22} {'Avg%':<7} {'Risk':<14} {'Weak Topics'}")
    print(f"  {_sep('-', 75)}")
    for p in sorted(profiles, key=lambda x: predict_struggle(x)["score"], reverse=True):
        risk  = predict_struggle(p)["risk"]
        emoji = {"High Risk": "🔴", "Medium Risk": "🟠", "Low Risk": "🟢"}.get(risk, "")
        weak  = ", ".join(p["weak_topics"]) or "—"
        print(
            f"  {p['student_id']:<5} {p['name'][:20]:<22} "
            f"{round(compute_performance(p)*100,1):<7} "  # type: ignore[arg-type]
            f"{emoji} {risk:<12} {weak}"
        )

    # Topic averages
    print(f"\n  Topic averages:")
    topic_avgs = {
        t: round(np.mean([p["knowledge"].get(t, 0) for p in profiles]) * 100, 1)
        for t in COURSE_TOPICS
    }
    for topic, avg in sorted(topic_avgs.items(), key=lambda x: x[1]):
        bar   = "█" * int(avg / 5)
        flag  = "  ⚠️" if avg < WEAK_THRESHOLD * 100 else ""
        print(f"  {topic:<20} {avg:>5}%  {bar}{flag}")


# ═══════════════════════════════════════════════════════════
# SINGLE STUDENT FULL PIPELINE
# ═══════════════════════════════════════════════════════════
def run_student(
    student_id: int,
    demo_concept: str = "ai_ml",
    show_charts: bool = True,
    save_charts: bool = False,
) -> None:
    print(f"\n{_sep('═')}")
    print(f"  EDUTWIN — Student {student_id}")
    print(_sep('═'))

    load_memory()
    profile = build_profile(student_id, force_rebuild=True)
    update_memory(student_id, profile)

    # ── Profile ──────────────────────────────
    print(f"\n📋 PROFILE SUMMARY")
    print(f"  {profile['summary']}")

    print(f"\n{'Topic':<22} {'Score':>6}  {'Status'}")
    print(f"  {_sep('-', 44)}")
    for topic, score in sorted(profile["knowledge"].items(), key=lambda x: x[1]):
        pct    = round(score * 100, 1)
        status = "✅ Strong" if score >= TARGET_SCORE else ("⚠️ Weak" if score < WEAK_THRESHOLD else "🔶 Fair")
        print(f"  {topic:<22} {pct:>5}%  {status}")

    print(f"\n📊 Performance : {round(compute_performance(profile) * 100, 1)}%")  # type: ignore[arg-type]
    print(f"⚠️  Risk         : {predict_struggle(profile)['risk']}")

    print(f"\n💡 RECOMMENDATIONS")
    for r in recommend_actions(profile):
        print(f"  {r}")

    # ── LLM Capabilities ─────────────────────
    print(f"\n{_sep()}")
    print(f"🧠 LLM WEAKNESS DIAGNOSIS")
    print(_sep())
    print(diagnose_weaknesses_with_llm(profile))

    print(f"\n{_sep()}")
    print(f"📘 PERSONALIZED EXPLANATION — '{demo_concept}'")
    print(_sep())
    print(generate_personalized_explanation(profile, demo_concept))

    print(f"\n{_sep()}")
    print(f"📝 EXAM ANSWER SIMULATION")
    print(_sep())
    question = "Explain the bias-variance tradeoff in machine learning."
    print(f"  Q: {question}")
    print(f"\n  A: {simulate_exam_answer(profile, question, 'ai_ml')}")

    print(f"\n{_sep()}")
    print(f"📅 7-DAY STUDY PLAN")
    print(_sep())
    print(generate_study_plan(profile, days=7))

    # ── Simulations ───────────────────────────
    print(f"\n{_sep()}")
    print(f"🧪 INTERVENTION SIMULATIONS")
    print(_sep())
    sims = run_multiple_simulations(profile)
    for action, perf in sorted(sims.items(), key=lambda x: -x[1]):
        bar = "█" * int(perf * 40)
        print(f"  {action:<22} {round(perf*100,1):>5}%  {bar}")

    best = max(sims, key=lambda k: sims[k])
    print(f"\n🏆 Best strategy: {best}  →  {round(sims[best]*100,1)}%")  # type: ignore[arg-type]

    # Apply best + store
    profile = simulate_intervention(profile, best)
    update_memory(student_id, profile)

    profile = update_from_real_outcome(profile, actual_score=0.72)
    update_memory(student_id, profile)
    save_memory()

    print(f"\n🎯 Goal achieved: {check_goal(profile)}")

    # ── Charts ───────────────────────────────
    if show_charts or save_charts:
        radar_path    = f"radar_{student_id}.png"    if save_charts else None
        progress_path = f"progress_{student_id}.png" if save_charts else None
        plot_radar(profile, save_path=radar_path)
        plot_subject_progress(student_id, save_path=progress_path)


# ═══════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════
def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        prog="digital_twin",
        description="EduTwin — LLM-Powered Digital Twin of University Students",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # student command
    s = sub.add_parser("student", help="Run full pipeline for one student")
    s.add_argument("id",      type=int,           help="Student ID (1–60)")
    s.add_argument("--topic", default="ai_ml",    help="Concept to explain (default: ai_ml)")
    s.add_argument("--no-charts",  action="store_true", help="Skip all charts")
    s.add_argument("--save-charts",action="store_true", help="Save charts as PNG instead of showing")

    # teacher command
    t = sub.add_parser("teacher", help="Class-wide overview")
    t.add_argument("--n", type=int, default=None, help="How many students to include")

    # explain command
    e = sub.add_parser("explain", help="Get a personalized explanation for a concept")
    e.add_argument("id",      type=int)
    e.add_argument("concept", choices=COURSE_TOPICS)

    # predict command
    pr = sub.add_parser("predict", help="Predict performance on an upcoming topic")
    pr.add_argument("id",    type=int)
    pr.add_argument("topic", choices=COURSE_TOPICS)

    # simulate command
    si = sub.add_parser("simulate", help="Run intervention simulations for a student")
    si.add_argument("id", type=int)

    # plan command
    pl = sub.add_parser("plan", help="Generate a study plan")
    pl.add_argument("id",     type=int)
    pl.add_argument("--days", type=int, default=7)

    # exam command
    ex = sub.add_parser("exam", help="Simulate a student's exam answer")
    ex.add_argument("id",       type=int)
    ex.add_argument("topic",    choices=COURSE_TOPICS)
    ex.add_argument("question", type=str)

    return p.parse_args()


def main():
    args = _parse_args()
    load_memory()

    if args.command == "student":
        run_student(
            args.id,
            demo_concept=args.topic,
            show_charts=not args.no_charts,
            save_charts=args.save_charts,
        )

    elif args.command == "teacher":
        teacher_overview(num_students=args.n)

    elif args.command == "explain":
        profile = build_profile(args.id, force_rebuild=True)
        print(f"\n📘 Explanation of '{args.concept}' for {profile['name']}\n")
        print(generate_personalized_explanation(profile, args.concept))

    elif args.command == "predict":
        profile = build_profile(args.id, force_rebuild=True)
        print(f"\n📈 Performance prediction for {profile['name']} on '{args.topic}'\n")
        print(predict_performance_llm(profile, args.topic))

    elif args.command == "simulate":
        profile = build_profile(args.id, force_rebuild=True)
        sims    = run_multiple_simulations(profile)
        print(f"\n🧪 Intervention simulations for {profile['name']}\n")
        for action, perf in sorted(sims.items(), key=lambda x: -x[1]):
            bar = "█" * int(perf * 40)
            print(f"  {action:<22} {round(perf*100,1):>5}%  {bar}")  # type: ignore[arg-type]
        print(f"\n🏆 Best: {max(sims, key=lambda k: sims[k])}")

    elif args.command == "plan":
        profile = build_profile(args.id, force_rebuild=True)
        print(f"\n📅 {args.days}-day study plan for {profile['name']}\n")
        print(generate_study_plan(profile, days=args.days))

    elif args.command == "exam":
        profile = build_profile(args.id, force_rebuild=True)
        print(f"\n📝 Simulated answer by {profile['name']}\n")
        print(f"Q: {args.question}\n")
        print(f"A: {simulate_exam_answer(profile, args.question, args.topic)}")


if __name__ == "__main__":
    main()