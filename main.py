"""
main.py
-------
EduTwin FastAPI server.

Run locally:
    uvicorn main:app --reload --port 8000

Production:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

Environment variables
---------------------
ALLOWED_ORIGINS   Comma-separated list of allowed CORS origins.
                  Defaults to localhost dev ports when unset.
                  Example: https://app.edutwin.com,https://www.edutwin.com
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import auth
import db
from digital_twin import (
    COURSE_TOPICS,
    INTERVENTION_EFFECTS,
    add_student,
    build_profile,
    check_goal,
    compute_performance,
    delete_student,
    diagnose_weaknesses_with_llm,
    generate_personalized_explanation,
    generate_study_plan,
    list_students,
    load_dataset,
    load_memory,
    predict_performance_llm,
    predict_struggle,
    recommend_actions,
    run_multiple_simulations,
    save_memory,
    simulate_exam_answer,
    simulate_intervention,
    update_from_real_outcome,
    update_memory,
    update_student,
)
from evaluate import run_full_evaluation


# ═══════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("edutwin")


# ═══════════════════════════════════════════════════════════
# CORS ORIGINS  (read from environment in production)
# ═══════════════════════════════════════════════════════════

def _get_allowed_origins() -> list[str]:
    """
    In production set ALLOWED_ORIGINS to a comma-separated list:
        ALLOWED_ORIGINS=https://app.edutwin.com,https://www.edutwin.com

    Leaving the variable unset falls back to localhost dev ports only.
    Never include "*" when allow_credentials=True — browsers reject it.
    """
    raw = os.getenv("ALLOWED_ORIGINS", "")
    if raw.strip():
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        logger.info("CORS origins loaded from environment: %s", origins)
        return origins

    # Development fallback — safe defaults only
    dev_origins = [
        "http://localhost:5173",   # Vite default
        "http://localhost:3000",   # CRA / Next.js default
        "http://localhost:4173",   # Vite preview
    ]
    logger.warning(
        "ALLOWED_ORIGINS not set — using dev-only CORS origins %s. "
        "Set ALLOWED_ORIGINS in production.",
        dev_origins,
    )
    return dev_origins


# ═══════════════════════════════════════════════════════════
# LIFESPAN  (replaces deprecated @app.on_event)
# ═══════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load dataset + memory once when the server starts."""
    logger.info("EduTwin starting up — loading memory and dataset …")
    load_memory()
    load_dataset()
    logger.info("Startup complete.")
    yield
    logger.info("EduTwin shutting down.")


# ═══════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="EduTwin API",
    description="LLM-Powered Digital Twin of University Students",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════
# REQUEST BODIES
# ═══════════════════════════════════════════════════════════

class ExamBody(BaseModel):
    topic: str
    question: str

class InterventionBody(BaseModel):
    action: str

class FeedbackBody(BaseModel):
    actual_score: float

class EvaluateBody(BaseModel):
    student_ids: Optional[List[int]] = None


# ── Student CRUD models ──────────────────────────────────────

class TopicInput(BaseModel):
    topic: str
    score: float = 0.50
    time_spent: float = 60.0
    attempts: float = 2.0
    confidence: float = 0.50
    engagement: float = 0.50
    fatigue: float = 0.30

class AddStudentBody(BaseModel):
    name: str
    year: int = 1
    branch: str = "CSE"
    archetype: str = "unknown"
    topics: List[TopicInput] = []

class TopicUpdate(BaseModel):
    topic: str
    score: Optional[float] = None
    time_spent: Optional[float] = None
    attempts: Optional[float] = None
    confidence: Optional[float] = None
    engagement: Optional[float] = None
    fatigue: Optional[float] = None

class UpdateStudentBody(BaseModel):
    name: Optional[str] = None
    year: Optional[int] = None
    branch: Optional[str] = None
    archetype: Optional[str] = None
    topics: Optional[List[TopicUpdate]] = None


# ── Auth models ──────────────────────────────────────────────

class RegisterTeacherBody(BaseModel):
    invite_code: str

class ClaimStudentBody(BaseModel):
    student_id: int


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _get_profile_or_404(student_id: int, force_rebuild: bool = False) -> dict:
    """Build profile or raise 404 if student not found."""
    try:
        return build_profile(student_id, force_rebuild=force_rebuild)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error building profile for student %d", student_id)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Auth dependencies ────────────────────────────────────────
# Thin FastAPI-dependency wrappers around auth.py's role checks. Declaring
# `student_id: int` as a plain parameter lets FastAPI resolve it from the
# same path parameter as the route these are used on.

def require_teacher(
    user: auth.CurrentUser = Depends(auth.get_current_user),
) -> auth.CurrentUser:
    return auth.require_teacher(user)


def require_self_or_teacher(
    student_id: int,
    user: auth.CurrentUser = Depends(auth.get_current_user),
) -> auth.CurrentUser:
    return auth.require_self_or_teacher(user, student_id)


# ═══════════════════════════════════════════════════════════
# ENDPOINT 1 — BUILD PROFILE
# GET /api/v1/profile/{student_id}
# ═══════════════════════════════════════════════════════════

@app.get("/api/v1/profile/{student_id}")
def get_profile(
    student_id: int,
    force_rebuild: bool = Query(default=False),
    user: auth.CurrentUser = Depends(require_self_or_teacher),
):
    """
    Returns the Live Learner Profile (LLP) for a student.
    Cached unless force_rebuild=true.
    """
    profile = _get_profile_or_404(student_id, force_rebuild=force_rebuild)
    return profile


# ═══════════════════════════════════════════════════════════
# ENDPOINT 2 — DIAGNOSE WEAKNESSES
# GET /api/v1/diagnose/{student_id}
# ═══════════════════════════════════════════════════════════

@app.get("/api/v1/diagnose/{student_id}")
def diagnose(student_id: int, user: auth.CurrentUser = Depends(require_self_or_teacher)):
    """
    LLM identifies 2-3 critical weak areas with remediation advice.
    """
    profile = _get_profile_or_404(student_id)
    diagnosis = diagnose_weaknesses_with_llm(profile)
    return {
        "student_id":    str(student_id),
        "name":          profile["name"],
        "weak_topics":   profile["weak_topics"],
        "llm_diagnosis": diagnosis,
    }


# ═══════════════════════════════════════════════════════════
# ENDPOINT 3 — PERSONALISED EXPLANATION
# GET /api/v1/explain/{student_id}/{concept}
# ═══════════════════════════════════════════════════════════

VALID_TOPICS = [
    "linear_algebra", "probability", "python_basics", "data_structures",
    "ai_ml", "databases", "os_concepts", "networking",
]

@app.get("/api/v1/explain/{student_id}/{concept}")
def explain(
    student_id: int,
    concept: str,
    user: auth.CurrentUser = Depends(require_self_or_teacher),
):
    """
    Generates a personalised concept explanation tuned to the student's profile.
    """
    if concept not in VALID_TOPICS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid concept '{concept}'. Must be one of: {VALID_TOPICS}",
        )
    profile     = _get_profile_or_404(student_id)
    explanation = generate_personalized_explanation(profile, concept)
    return {
        "student_id":  str(student_id),
        "name":        profile["name"],
        "concept":     concept,
        "explanation": explanation,
        "profile_used": {
            "confidence_level": profile["psychology"]["confidence_level"],
            "learning_pace":    profile["psychology"]["learning_pace"],
            "weak_topics":      profile["weak_topics"],
            "strong_topics":    profile.get("strong_topics", []),
        },
    }


# ═══════════════════════════════════════════════════════════
# ENDPOINT 4 — PERFORMANCE PREDICTION
# GET /api/v1/predict/{student_id}/{topic}
# ═══════════════════════════════════════════════════════════

@app.get("/api/v1/predict/{student_id}/{topic}")
def predict(
    student_id: int,
    topic: str,
    user: auth.CurrentUser = Depends(require_self_or_teacher),
):
    """
    Predicts High / Medium / Low performance on an upcoming assessment.
    Returns both LLM prediction and rule-based prediction.
    """
    if topic not in VALID_TOPICS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid topic '{topic}'. Must be one of: {VALID_TOPICS}",
        )
    profile          = _get_profile_or_404(student_id)
    risk_data        = predict_struggle(profile)
    llm_prediction   = predict_performance_llm(profile, topic)
    current_score    = profile["knowledge"].get(topic, 0.0)

    risk_score = risk_data["score"]
    if risk_score > 0.60:
        rule_prediction = "Low"
    elif risk_score > 0.35:
        rule_prediction = "Medium"
    else:
        rule_prediction = "High"

    return {
        "student_id":            str(student_id),
        "name":                  profile["name"],
        "topic":                 topic,
        "current_score_pct":     round(current_score * 100, 1),
        "risk":                  risk_data["risk"],
        "risk_score":            risk_data["score"],
        "llm_prediction":        llm_prediction,
        "rule_based_prediction": rule_prediction,
    }


# ═══════════════════════════════════════════════════════════
# ENDPOINT 5 — EXAM SIMULATION
# POST /api/v1/exam/{student_id}
# ═══════════════════════════════════════════════════════════

TOPIC_KEYWORDS = {
    "ai_ml":           ["gradient", "overfitting", "neural", "training", "loss", "bias", "variance"],
    "probability":     ["bayes", "distribution", "probability", "expected", "event", "random"],
    "linear_algebra":  ["matrix", "vector", "eigenvalue", "dot", "transformation", "rank"],
    "data_structures": ["tree", "graph", "hash", "linked", "stack", "queue", "heap"],
    "databases":       ["sql", "query", "index", "join", "normalisation", "transaction"],
    "python_basics":   ["function", "loop", "class", "variable", "list", "import"],
    "os_concepts":     ["process", "thread", "scheduling", "memory", "file", "kernel"],
    "networking":      ["tcp", "ip", "http", "socket", "dns", "router", "packet"],
}

def _keyword_overlap(answer: str, topic: str) -> float:
    keywords = TOPIC_KEYWORDS.get(topic, [])
    if not keywords:
        return 0.5
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw in answer_lower)
    return round(hits / len(keywords), 4)  # type: ignore[arg-type]


@app.post("/api/v1/exam/{student_id}")
def exam_simulation(
    student_id: int,
    body: ExamBody,
    user: auth.CurrentUser = Depends(require_self_or_teacher),
):
    """
    Simulates the exam answer a student would write for a given question.
    """
    if body.topic not in VALID_TOPICS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid topic '{body.topic}'. Must be one of: {VALID_TOPICS}",
        )
    profile          = _get_profile_or_404(student_id)
    simulated_answer = simulate_exam_answer(profile, body.question, body.topic)
    topic_score      = profile["knowledge"].get(body.topic, 0.0)
    overlap          = _keyword_overlap(simulated_answer, body.topic)

    return {
        "student_id":       str(student_id),
        "name":             profile["name"],
        "topic":            body.topic,
        "topic_score_pct":  round(topic_score * 100, 1),
        "question":         body.question,
        "simulated_answer": simulated_answer,
        "keyword_overlap":  overlap,
    }


# ═══════════════════════════════════════════════════════════
# ENDPOINT 6 — STUDY PLAN
# GET /api/v1/plan/{student_id}
# ═══════════════════════════════════════════════════════════

@app.get("/api/v1/plan/{student_id}")
def study_plan(
    student_id: int,
    days: int = Query(default=7, ge=1, le=30),
    user: auth.CurrentUser = Depends(require_self_or_teacher),
):
    """
    Generates a day-by-day personalised study plan.
    """
    profile    = _get_profile_or_404(student_id)
    plan       = generate_study_plan(profile, days=days)
    fatigue    = profile["behavior"]["fatigue"]
    engagement = profile["behavior"]["engagement"]

    fatigue_label    = "low" if fatigue    < 0.40 else ("medium" if fatigue    < 0.60 else "good")
    engagement_label = "low" if engagement < 0.40 else ("medium" if engagement < 0.65 else "high")

    return {
        "student_id":       str(student_id),
        "name":             profile["name"],
        "days":             days,
        "weak_topics":      profile["weak_topics"],
        "fatigue_level":    fatigue_label,
        "engagement_level": engagement_label,
        "study_plan":       plan,
    }


# ═══════════════════════════════════════════════════════════
# ENDPOINT 7 — INTERVENTION SIMULATION
# GET /api/v1/simulate/{student_id}
# ═══════════════════════════════════════════════════════════

@app.get("/api/v1/simulate/{student_id}")
def intervention_simulation(
    student_id: int, user: auth.CurrentUser = Depends(require_self_or_teacher)
):
    """
    Runs all 5 intervention strategies and returns projected performance for each.
    """
    profile     = _get_profile_or_404(student_id)
    sims        = run_multiple_simulations(profile)
    best        = max(sims, key=sims.get)
    current_pct = round(compute_performance(profile) * 100, 2)

    return {
        "student_id":                    str(student_id),
        "name":                          profile["name"],
        "current_performance_pct":       current_pct,
        "simulations":                   {k: round(v * 100, 2) for k, v in sims.items()},
        "best_strategy":                 best,
        "best_projected_performance_pct": round(sims[best] * 100, 2),
        "intervention_effects_reference": INTERVENTION_EFFECTS,
    }


# ═══════════════════════════════════════════════════════════
# ENDPOINT 8 — APPLY INTERVENTION
# POST /api/v1/simulate/{student_id}/apply
# ═══════════════════════════════════════════════════════════

VALID_INTERVENTIONS = list(INTERVENTION_EFFECTS.keys())

@app.post("/api/v1/simulate/{student_id}/apply")
def apply_intervention(
    student_id: int,
    body: InterventionBody,
    user: auth.CurrentUser = Depends(require_self_or_teacher),
):
    """
    Applies a chosen intervention to the student's profile and persists it.
    """
    if body.action not in VALID_INTERVENTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action '{body.action}'. Must be one of: {VALID_INTERVENTIONS}",
        )
    profile      = _get_profile_or_404(student_id)
    perf_before  = round(compute_performance(profile) * 100, 2)

    updated      = simulate_intervention(profile, body.action)
    perf_after   = round(compute_performance(updated) * 100, 2)
    goal         = check_goal(updated)

    update_memory(student_id, updated)
    save_memory()

    return {
        "student_id":             str(student_id),
        "name":                   updated["name"],
        "action_applied":         body.action,
        "performance_before_pct": perf_before,
        "performance_after_pct":  perf_after,
        "goal_achieved":          goal,
        "updated_profile":        updated,
    }


# ═══════════════════════════════════════════════════════════
# ENDPOINT 9 — FEEDBACK LOOP
# POST /api/v1/feedback/{student_id}
# ═══════════════════════════════════════════════════════════

@app.post("/api/v1/feedback/{student_id}")
def feedback_loop(
    student_id: int,
    body: FeedbackBody,
    user: auth.CurrentUser = Depends(require_self_or_teacher),
):
    """
    Corrects the student's confidence estimate based on a real exam outcome.
    new_confidence = old + 0.2 * (actual - old)
    """
    if not (0.0 <= body.actual_score <= 1.0):
        raise HTTPException(
            status_code=422,
            detail="actual_score must be between 0.0 and 1.0",
        )
    profile            = _get_profile_or_404(student_id)
    confidence_before  = round(profile["behavior"]["avg_confidence"], 4)

    updated            = update_from_real_outcome(profile, body.actual_score)
    confidence_after   = round(updated["behavior"]["avg_confidence"], 4)
    goal               = check_goal(updated)

    update_memory(student_id, updated)
    save_memory()

    return {
        "student_id":        str(student_id),
        "name":              updated["name"],
        "actual_score":      body.actual_score,
        "confidence_before": confidence_before,
        "confidence_after":  confidence_after,
        "updated_summary":   updated["summary"],
        "goal_achieved":     goal,
    }


# ═══════════════════════════════════════════════════════════
# ENDPOINT 10 — RECOMMEND ACTIONS
# GET /api/v1/recommend/{student_id}
# ═══════════════════════════════════════════════════════════

@app.get("/api/v1/recommend/{student_id}")
def get_recommendations(
    student_id: int, user: auth.CurrentUser = Depends(require_self_or_teacher)
):
    """
    Returns the rule-based list of actionable recommendations.
    """
    profile = _get_profile_or_404(student_id)
    recs    = recommend_actions(profile)
    return {
        "student_id":      str(student_id),
        "name":            profile["name"],
        "recommendations": recs,
    }


# ═══════════════════════════════════════════════════════════
# ENDPOINT 11 — TEACHER OVERVIEW
# GET /api/v1/teacher
# ═══════════════════════════════════════════════════════════

@app.get("/api/v1/teacher")
def teacher_overview(
    num_students: Optional[int] = Query(default=None, ge=1, le=60),
    user: auth.CurrentUser = Depends(require_teacher),
):
    """
    Returns class-wide summary: per-student risk levels, avg scores,
    and per-topic class averages. Students sorted by risk score descending.
    """
    all_ids = sorted(load_dataset())

    if num_students is not None:
        all_ids = all_ids[:num_students]  # type: ignore[arg-type]

    profiles = []
    for sid in all_ids:
        try:
            profiles.append(build_profile(sid, force_rebuild=False))
        except Exception:
            logger.warning("Could not build profile for student %d — skipping.", sid)
            continue

    if not profiles:
        raise HTTPException(status_code=500, detail="Could not load any student profiles.")

    # Compute per-profile values once to avoid redundant calls
    perf_list      = [compute_performance(p) for p in profiles]
    risk_list      = [predict_struggle(p) for p in profiles]   # called once per profile
    risk_labels    = [r["risk"] for r in risk_list]

    students_out = sorted(
        [
            {
                "student_id":    p["student_id"],
                "name":          p["name"],
                "avg_score_pct": round(perf * 100, 1),
                "risk":          risk["risk"],
                "risk_score":    risk["score"],
                "weak_topics":   p["weak_topics"],
            }
            for p, perf, risk in zip(profiles, perf_list, risk_list)
        ],
        key=lambda x: x["risk_score"],
        reverse=True,
    )

    topic_averages = {
        t: round(
            float(sum(p["knowledge"].get(t, 0) for p in profiles) / len(profiles)) * 100, 1  # type: ignore[arg-type]
        )
        for t in COURSE_TOPICS
    }

    return {
        "class_summary": {
            "total_students":    len(profiles),
            "class_avg_pct":     round(float(sum(perf_list) / len(perf_list)) * 100, 1),  # type: ignore[arg-type]
            "high_risk_count":   risk_labels.count("High Risk"),
            "medium_risk_count": risk_labels.count("Medium Risk"),
            "low_risk_count":    risk_labels.count("Low Risk"),
            "goal_met_count":    sum(1 for p in profiles if check_goal(p)),
        },
        "students":       students_out,
        "topic_averages": topic_averages,
    }


# ═══════════════════════════════════════════════════════════
# ENDPOINT 12 — EVALUATION PIPELINE
# POST /api/v1/evaluate
# ═══════════════════════════════════════════════════════════

@app.post("/api/v1/evaluate")
def evaluation_pipeline(
    body: EvaluateBody, user: auth.CurrentUser = Depends(require_teacher)
):
    """
    Runs the full 4-part evaluation pipeline.
    Long-running — may take several minutes.
    """
    try:
        results = run_full_evaluation(student_ids=body.student_ids)
    except Exception as exc:
        logger.exception("Evaluation pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))

    # Sanitise numpy floats / NaN values for JSON serialisation
    def sanitise(obj):
        if isinstance(obj, dict):
            return {k: sanitise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitise(i) for i in obj]
        if isinstance(obj, float) and (obj != obj):   # NaN
            return None
        if hasattr(obj, "item"):                       # numpy scalar
            return obj.item()
        return obj

    return sanitise(results)


# ═══════════════════════════════════════════════════════════
# ENDPOINT 13 — LIST ALL STUDENTS
# GET /api/v1/students
# ═══════════════════════════════════════════════════════════

@app.get("/api/v1/students")
def get_all_students(user: auth.CurrentUser = Depends(auth.get_current_user)):
    """
    Returns a lightweight list of every student in the dataset
    (student_id, name, year, branch, archetype). Useful for populating
    dashboards and dropdown menus without building full profiles.

    Any authenticated user may call this — including someone who has just
    signed up and hasn't completed profile setup yet (role/student_id may
    still be None at this point), since the student sign-up flow needs the
    roster to offer the "pick your name" step.
    """
    try:
        students = list_students()
    except Exception as exc:
        logger.exception("Failed to list students")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"total": len(students), "students": students}


# ═══════════════════════════════════════════════════════════
# ENDPOINT 14 — ADD STUDENT
# POST /api/v1/students
# ═══════════════════════════════════════════════════════════

@app.post("/api/v1/students", status_code=201)
def create_student(
    body: AddStudentBody, user: auth.CurrentUser = Depends(require_teacher)
):
    """
    Adds a new student to the roster and returns their freshly built profile.
    Teacher-only — students register themselves via /api/v1/auth/register-student.

    - A unique student_id is assigned automatically.
    - You may supply per-topic values inside `topics`; any topic omitted
      from the list is created with neutral defaults (score=0.5, etc.).
    - All eight course topics are always created so the profile is complete.
    """
    student_data = body.model_dump()
    try:
        profile = add_student(student_data)
    except Exception as exc:
        logger.exception("Failed to add student")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message":    "Student created successfully.",
        "student_id": profile["student_id"],
        "profile":    profile,
    }


# ═══════════════════════════════════════════════════════════
# ENDPOINT 15 — UPDATE STUDENT
# PUT /api/v1/students/{student_id}
# ═══════════════════════════════════════════════════════════

@app.put("/api/v1/students/{student_id}")
def modify_student(
    student_id: int,
    body: UpdateStudentBody,
    user: auth.CurrentUser = Depends(require_self_or_teacher),
):
    """
    Partially updates a student's metadata and/or per-topic scores.

    Only fields present in the request body are changed; everything else
    is left untouched. Returns the rebuilt profile after saving.

    Example — update name and two topic scores:
    ```json
    {
      "name": "Alice Updated",
      "topics": [
        {"topic": "ai_ml", "score": 0.82},
        {"topic": "probability", "confidence": 0.65}
      ]
    }
    ```
    """
    updates = {k: v for k, v in body.model_dump().items() if v is not None}

    # Convert TopicUpdate objects to plain dicts, dropping None fields
    if "topics" in updates:
        updates["topics"] = [
            {k: v for k, v in t.items() if v is not None}
            for t in updates["topics"]
        ]

    try:
        profile = update_student(student_id, updates)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to update student %d", student_id)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message":    "Student updated successfully.",
        "student_id": str(student_id),
        "profile":    profile,
    }


# ═══════════════════════════════════════════════════════════
# ENDPOINT 16 — DELETE STUDENT
# DELETE /api/v1/students/{student_id}
# ═══════════════════════════════════════════════════════════

@app.delete("/api/v1/students/{student_id}", status_code=200)
def remove_student(
    student_id: int, user: auth.CurrentUser = Depends(require_teacher)
):
    """
    Permanently removes a student from the CSV dataset and clears their
    memory history. Returns 404 if the student does not exist.
    """
    try:
        found = delete_student(student_id)
    except Exception as exc:
        logger.exception("Failed to delete student %d", student_id)
        raise HTTPException(status_code=500, detail=str(exc))

    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Student ID {student_id} not found.",
        )

    return {
        "message":    f"Student {student_id} deleted successfully.",
        "student_id": str(student_id),
    }


# ═══════════════════════════════════════════════════════════
# AUTH — profile completion after Supabase Auth sign-up/sign-in
# ═══════════════════════════════════════════════════════════
# The client authenticates directly against Supabase Auth (supabase-js).
# These endpoints only handle the app-specific step of linking that auth
# user to a role ("teacher" | "student") and, for students, a student_id.

@app.get("/api/v1/auth/me")
def get_me(user: auth.CurrentUser = Depends(auth.get_current_user)):
    """
    Returns the caller's role and (for students) linked student_id.
    role is null until register-teacher / register-student / claim-student
    has been called once for this account.
    """
    return {
        "user_id":    user.user_id,
        "email":      user.email,
        "role":       user.role,
        "student_id": user.student_id,
    }


@app.post("/api/v1/auth/register-teacher")
def register_teacher(
    body: RegisterTeacherBody,
    user: auth.CurrentUser = Depends(auth.get_current_user),
):
    """
    Grants the teacher role to the calling account, gated by a shared
    invite code (set via the TEACHER_INVITE_CODE environment variable) so
    that self-serve sign-up can't silently hand out teacher access.
    """
    expected = os.getenv("TEACHER_INVITE_CODE", "")
    if not expected or body.invite_code != expected:
        raise HTTPException(status_code=403, detail="Invalid teacher invite code.")

    profile = db.create_user_profile(user.user_id, role="teacher")
    return {"message": "Teacher role granted.", "role": profile["role"]}


@app.post("/api/v1/auth/claim-student")
def claim_student(
    body: ClaimStudentBody,
    user: auth.CurrentUser = Depends(auth.get_current_user),
):
    """
    Links the calling account to an existing (unclaimed) student record —
    the "pick your name from the roster" sign-up path.
    """
    meta = db.get_student_meta(body.student_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Student not found.")
    if meta.get("auth_user_id"):
        raise HTTPException(
            status_code=409, detail="This student record is already claimed."
        )

    db.link_student_to_user(body.student_id, user.user_id)
    profile = db.create_user_profile(user.user_id, role="student", student_id=body.student_id)
    return {
        "message":    "Student account linked.",
        "role":       profile["role"],
        "student_id": profile["student_id"],
    }


@app.post("/api/v1/auth/register-student", status_code=201)
def register_student(
    body: AddStudentBody,
    user: auth.CurrentUser = Depends(auth.get_current_user),
):
    """
    Creates a brand-new student record for the calling account in one
    step — the "I'm new here" sign-up path.
    """
    student_data = body.model_dump()
    try:
        profile = add_student(student_data)
    except Exception as exc:
        logger.exception("Failed to register new student")
        raise HTTPException(status_code=500, detail=str(exc))

    new_id = int(profile["student_id"])
    db.link_student_to_user(new_id, user.user_id)
    db.create_user_profile(user.user_id, role="student", student_id=new_id)

    return {
        "message":    "Student registered successfully.",
        "student_id": new_id,
        "profile":    profile,
    }


# ═══════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════

@app.get("/health")
@app.get("/")
def health_check():
    return {"status": "ok", "service": "EduTwin API", "version": "1.0.0"}


# ═══════════════════════════════════════════════════════════
# LOCAL ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)