"""
evaluate.py
-----------
Evaluation pipeline for EduTwin capabilities.

Metrics per capability:
  1. Weakness Diagnosis  → Precision / Recall / F1 vs ground-truth weak topics
  2. Explanation Quality → LLM self-rating 1–5 (proxy; swap with human ratings)
  3. Performance Prediction → Accuracy / Weighted F1 over 3-class labels
  4. Exam Simulation     → Similarity to expected answer profile (keyword overlap)
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from digital_twin import (
    COURSE_TOPICS,
    TARGET_SCORE,
    WEAK_THRESHOLD,
    build_profile,
    call_llm,
    compute_performance,
    diagnose_weaknesses_with_llm,
    generate_personalized_explanation,
    load_dataset,
    load_memory,
    predict_struggle,
    simulate_exam_answer,
)

random.seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _ground_truth_weak(profile: dict) -> set[str]:
    """True weak topics: score < WEAK_THRESHOLD in the raw dataset."""
    return {t for t, v in profile["knowledge"].items() if v < WEAK_THRESHOLD}


def _performance_label(avg_score: float) -> str:
    if avg_score >= TARGET_SCORE:
        return "High"
    elif avg_score >= 0.50:
        return "Medium"
    else:
        return "Low"


# ─────────────────────────────────────────────
# EVAL 1: WEAKNESS DIAGNOSIS
# ─────────────────────────────────────────────

def eval_weakness_rule_based(student_ids: List[int]) -> dict:
    """
    Evaluates the rule-based weak-topic detection (score < threshold)
    against ground truth from the dataset.
    Since both use the same threshold it should be near-perfect — this
    acts as a sanity-check baseline.
    """
    all_precision, all_recall, all_f1 = [], [], []

    for sid in student_ids:
        try:
            profile = build_profile(sid, force_rebuild=True)
        except Exception:
            continue

        gt_weak   = _ground_truth_weak(profile)
        pred_weak = set(profile["weak_topics"])

        tp = len(pred_weak & gt_weak)
        precision = tp / len(pred_weak) if pred_weak else 0.0
        recall    = tp / len(gt_weak)   if gt_weak   else 1.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

    return {
        "mean_precision": round(float(np.mean(all_precision)), 4),
        "mean_recall":    round(float(np.mean(all_recall)),    4),
        "mean_f1":        round(float(np.mean(all_f1)),        4),
        "n_students":     len(all_precision),
    }


# ─────────────────────────────────────────────
# EVAL 2: EXPLANATION QUALITY (LLM as judge)
# ─────────────────────────────────────────────

def eval_explanation_quality(student_ids: List[int],
                              concepts: List[str] | None = None,
                              sample_n: int = 5) -> dict:
    """
    Generates personalized explanations for a sample of students and
    uses a separate LLM call to rate relevance (1–5).
    In production, replace with human ratings collected via Google Form.
    """
    if concepts is None:
        concepts = random.sample(COURSE_TOPICS, min(3, len(COURSE_TOPICS)))

    sampled = random.sample(student_ids, min(sample_n, len(student_ids)))
    ratings = []

    for sid in sampled:
        concept = random.choice(concepts)
        try:
            profile = build_profile(sid, force_rebuild=True)
            explanation = generate_personalized_explanation(profile, concept)

            # LLM-as-judge
            judge_prompt = (
                f"Student profile summary: {profile['summary']}\n\n"
                f"Concept being explained: {concept}\n\n"
                f"Explanation given:\n{explanation}\n\n"
                "Rate how well this explanation is tailored to the student's specific "
                "profile (confidence level, weak areas, learning pace) on a scale of 1-5.\n"
                "1 = generic, ignores profile | 5 = perfectly personalised.\n"
                "Reply with ONLY a single integer between 1 and 5."
            )
            rating_str = call_llm(judge_prompt).strip()
            try:
                rating = int(rating_str[0])
                rating = max(1, min(5, rating))
            except (ValueError, IndexError):
                rating = 3

            ratings.append({
                "student_id": sid,
                "concept":    concept,
                "rating":     rating,
            })
        except Exception as exc:
            print(f"  [WARN] Student {sid} explanation eval failed: {exc}")

    avg_rating = round(float(np.mean([r["rating"] for r in ratings])), 2) if ratings else 0.0

    return {
        "avg_llm_rating":  avg_rating,
        "max_rating":      5,
        "n_evaluated":     len(ratings),
        "details":         ratings,
    }


# ─────────────────────────────────────────────
# EVAL 3: PERFORMANCE PREDICTION (3-class)
# ─────────────────────────────────────────────

def eval_performance_prediction(student_ids: List[int]) -> dict:
    """
    Rule-based baseline: predict the performance class from the risk score.
    Ground truth: actual avg score from dataset labels.
    """
    y_true, y_pred = [], []
    baseline_pred  = []

    for sid in student_ids:
        try:
            profile = build_profile(sid, force_rebuild=True)
        except Exception:
            continue

        gt_label   = _performance_label(compute_performance(profile))
        risk_score = predict_struggle(profile)["score"]

        # Simple rule: map risk score to performance label (inverse)
        if risk_score > 0.60:
            pred_label = "Low"
        elif risk_score > 0.35:
            pred_label = "Medium"
        else:
            pred_label = "High"

        y_true.append(gt_label)
        y_pred.append(pred_label)
        baseline_pred.append("Medium")   # naive baseline: always predict Medium

    if not y_true:
        return {"error": "No students evaluated"}

    labels     = ["High", "Medium", "Low"]
    model_f1   = round(f1_score(y_true, y_pred,     average="weighted",
                                labels=labels, zero_division=0), 4)
    baseline_f1 = round(f1_score(y_true, baseline_pred, average="weighted",
                                  labels=labels, zero_division=0), 4)

    report = classification_report(y_true, y_pred, labels=labels,
                                   zero_division=0, output_dict=True)

    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = round(correct / len(y_true), 4)

    return {
        "accuracy":      accuracy,
        "weighted_f1":   model_f1,
        "baseline_f1":   baseline_f1,
        "classification_report": report,
        "n_students":    len(y_true),
    }


# ─────────────────────────────────────────────
# EVAL 4: EXAM ANSWER SIMULATION
# ─────────────────────────────────────────────

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

SAMPLE_QUESTIONS = {
    "ai_ml":           "What is overfitting and how do you prevent it?",
    "probability":     "Explain Bayes' theorem with an example.",
    "linear_algebra":  "What is an eigenvalue? When is it useful?",
    "data_structures": "Compare a stack and a queue.",
    "databases":       "What is database normalisation and why does it matter?",
    "python_basics":   "Explain list comprehensions in Python with an example.",
    "os_concepts":     "What is the difference between a process and a thread?",
    "networking":      "Describe the TCP three-way handshake.",
}


def _keyword_overlap(answer: str, topic: str) -> float:
    keywords = TOPIC_KEYWORDS.get(topic, [])
    if not keywords:
        return 0.5
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw in answer_lower)
    return round(hits / len(keywords), 4)


def eval_exam_simulation(student_ids: List[int], sample_n: int = 10) -> dict:
    """
    For each sampled student, simulate an answer for one topic.
    Check whether weak students use fewer relevant keywords than strong students.
    """
    sampled = random.sample(student_ids, min(sample_n, len(student_ids)))
    results = []

    for sid in sampled:
        topic = random.choice(COURSE_TOPICS)
        question = SAMPLE_QUESTIONS.get(topic, f"Explain {topic}.")
        try:
            profile  = build_profile(sid, force_rebuild=True)
            answer   = simulate_exam_answer(profile, question, topic)
            overlap  = _keyword_overlap(answer, topic)
            topic_score = profile["knowledge"].get(topic, 0)
            is_weak  = topic_score < WEAK_THRESHOLD

            results.append({
                "student_id":  sid,
                "topic":       topic,
                "topic_score": topic_score,
                "is_weak":     is_weak,
                "kw_overlap":  overlap,
            })
        except Exception as exc:
            print(f"  [WARN] Sim eval student {sid}: {exc}")

    if not results:
        return {"error": "No results"}

    df = pd.DataFrame(results)
    weak_overlap   = df[df["is_weak"]]["kw_overlap"].mean()
    strong_overlap = df[~df["is_weak"]]["kw_overlap"].mean()

    return {
        "avg_kw_overlap":         round(df["kw_overlap"].mean(), 4),
        "weak_student_overlap":   round(weak_overlap,   4) if not np.isnan(weak_overlap)   else None,
        "strong_student_overlap": round(strong_overlap, 4) if not np.isnan(strong_overlap) else None,
        "gap_weak_vs_strong":     round(strong_overlap - weak_overlap, 4)
                                  if not (np.isnan(weak_overlap) or np.isnan(strong_overlap)) else None,
        "n_evaluated":            len(results),
        "note": "Higher gap = twin correctly differentiates weak vs strong students",
    }


# ─────────────────────────────────────────────
# FULL EVALUATION RUNNER
# ─────────────────────────────────────────────

def run_full_evaluation(student_ids: List[int] | None = None) -> dict:
    load_memory()
    all_ids = load_dataset()

    if student_ids is None:
        student_ids = all_ids

    print("\n" + "=" * 60)
    print("  EDUTWIN — EVALUATION PIPELINE")
    print("=" * 60)

    # ── Eval 1 ──────────────────────────────────
    print("\n[1/4] Weakness Diagnosis (rule-based baseline)…")
    eval1 = eval_weakness_rule_based(student_ids)
    print(f"      Precision: {eval1['mean_precision']} | "
          f"Recall: {eval1['mean_recall']} | F1: {eval1['mean_f1']}")

    # ── Eval 2 ──────────────────────────────────
    print("\n[2/4] Explanation Quality (LLM judge, n=5)…")
    eval2 = eval_explanation_quality(student_ids, sample_n=5)
    print(f"      Avg LLM rating: {eval2['avg_llm_rating']} / 5  "
          f"(n={eval2['n_evaluated']})")

    # ── Eval 3 ──────────────────────────────────
    print("\n[3/4] Performance Prediction (3-class)…")
    eval3 = eval_performance_prediction(student_ids)
    print(f"      Accuracy: {eval3.get('accuracy')} | "
          f"Weighted F1: {eval3.get('weighted_f1')} | "
          f"Baseline F1: {eval3.get('baseline_f1')}")

    # ── Eval 4 ──────────────────────────────────
    print("\n[4/4] Exam Simulation (n=10)…")
    eval4 = eval_exam_simulation(student_ids, sample_n=10)
    print(f"      Keyword overlap — weak: {eval4.get('weak_student_overlap')} | "
          f"strong: {eval4.get('strong_student_overlap')} | "
          f"gap: {eval4.get('gap_weak_vs_strong')}")

    results = {
        "weakness_diagnosis":    eval1,
        "explanation_quality":   eval2,
        "performance_prediction": eval3,
        "exam_simulation":       eval4,
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅  Results saved to evaluation_results.json")
    print("=" * 60)
    return results


if __name__ == "__main__":
    run_full_evaluation()
