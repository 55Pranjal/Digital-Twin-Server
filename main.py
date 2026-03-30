"""
main.py
-------
EduTwin FastAPI server.

Run with:
    uvicorn main:app --reload --port 8000

All 12 endpoints from edutwin_api_contract.json are implemented here.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from digital_twin import (
    INTERVENTION_EFFECTS,
    build_profile,
    check_goal,
    compute_performance,
    diagnose_weaknesses_with_llm,
    generate_personalized_explanation,
    generate_study_plan,
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
)
from evaluate import run_full_evaluation


# ═══════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="EduTwin API",
    description="LLM-Powered Digital Twin of University Students",
    version="1.0.0",
)

# ── CORS ─────────────────────────────────────────────────────────────
# Allow the Vite dev server and any localhost port.
# In production, replace "*" with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite default
        "http://localhost:3000",   # CRA / Next.js default
        "http://localhost:4173",   # Vite preview
        "*",                       # remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    """Load dataset + memory once when server starts."""
    load_memory()
    load_dataset()


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


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _safe_float(val) -> float:
    """Convert numpy floats / NaN to plain Python float."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (f != f) else f   # NaN check
    except Exception:
        return None


def _get_profile_or_404(student_id: int, force_rebuild: bool = False) -> dict:
    """Build profile or raise 404 if student not found."""
    try:
        return build_profile(student_id, force_rebuild=force_rebuild)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ═══════════════════════════════════════════════════════════
# ENDPOINT 1 — BUILD PROFILE
# GET /api/v1/profile/{student_id}
# ═══════════════════════════════════════════════════════════

@app.get("/api/v1/profile/{student_id}")
def get_profile(
    student_id: int,
    force_rebuild: bool = Query(default=False),
):
    """
    Returns the Live Learner Profile (LLP) for a student.
    Cached unless force_rebuild=true.
    """
    profile = _get_profile_or_404(student_id, force_rebuild=force_rebuild)
    update_memory(student_id, profile)
    save_memory()
    return profile


# ═══════════════════════════════════════════════════════════
# ENDPOINT 2 — DIAGNOSE WEAKNESSES
# GET /api/v1/diagnose/{student_id}
# ═══════════════════════════════════════════════════════════

@app.get("/api/v1/diagnose/{student_id}")
def diagnose(student_id: int):
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
def explain(student_id: int, concept: str):
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
def predict(student_id: int, topic: str):
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
    return round(hits / len(keywords), 4)


@app.post("/api/v1/exam/{student_id}")
def exam_simulation(student_id: int, body: ExamBody):
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
def intervention_simulation(student_id: int):
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
def apply_intervention(student_id: int, body: InterventionBody):
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
def feedback_loop(student_id: int, body: FeedbackBody):
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
def get_recommendations(student_id: int):
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

from digital_twin import COURSE_TOPICS

@app.get("/api/v1/teacher")
def teacher_overview(
    num_students: Optional[int] = Query(default=None, ge=1, le=60),
):
    """
    Returns class-wide summary: per-student risk levels, avg scores,
    and per-topic class averages. Students sorted by risk score descending.
    """
    df      = load_dataset()
    all_ids = sorted(df["student_id"].unique().tolist())

    if num_students is not None:
        all_ids = all_ids[:num_students]

    profiles = []
    for sid in all_ids:
        try:
            profiles.append(build_profile(sid, force_rebuild=False))
        except Exception:
            continue

    if not profiles:
        raise HTTPException(status_code=500, detail="Could not load any student profiles.")

    perfs      = [compute_performance(p) for p in profiles]
    risk_data  = [predict_struggle(p) for p in profiles]
    risk_labels = [r["risk"] for r in risk_data]

    # Per-student rows sorted by risk score descending
    students_out = sorted(
        [
            {
                "student_id":    p["student_id"],
                "name":          p["name"],
                "avg_score_pct": round(compute_performance(p) * 100, 1),
                "risk":          predict_struggle(p)["risk"],
                "risk_score":    predict_struggle(p)["score"],
                "weak_topics":   p["weak_topics"],
            }
            for p in profiles
        ],
        key=lambda x: x["risk_score"],
        reverse=True,
    )

    # Topic averages
    topic_averages = {
        t: round(
            float(sum(p["knowledge"].get(t, 0) for p in profiles) / len(profiles)) * 100, 1
        )
        for t in COURSE_TOPICS
    }

    return {
        "class_summary": {
            "total_students":    len(profiles),
            "class_avg_pct":     round(float(sum(perfs) / len(perfs)) * 100, 1),
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
def evaluation_pipeline(body: EvaluateBody):
    """
    Runs the full 4-part evaluation pipeline.
    Long-running — may take several minutes.
    """
    try:
        results = run_full_evaluation(student_ids=body.student_ids)
    except Exception as exc:
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
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════

@app.get("/")
def health_check():
    return {"status": "ok", "service": "EduTwin API", "version": "1.0.0"}