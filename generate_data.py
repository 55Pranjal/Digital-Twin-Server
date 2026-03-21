"""
generate_data.py
----------------
Generates a synthetic student dataset for EduTwin.
Produces: enhanced_students_dataset.csv
"""

import pandas as pd
import numpy as np
import random
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NUM_STUDENTS   = 60
TOPICS         = ["linear_algebra", "probability", "python_basics",
                  "data_structures", "ai_ml", "databases", "os_concepts", "networking"]
ROWS_PER_TOPIC = 3          # repeated assessments per topic

STUDENT_ARCHETYPES = {
    "strong":    {"score_mu": 0.78, "score_sd": 0.10, "conf_mu": 0.75, "fatigue_mu": 0.25},
    "average":   {"score_mu": 0.60, "score_sd": 0.12, "conf_mu": 0.55, "fatigue_mu": 0.40},
    "weak":      {"score_mu": 0.42, "score_sd": 0.12, "conf_mu": 0.35, "fatigue_mu": 0.65},
    "struggling":{"score_mu": 0.30, "score_sd": 0.10, "conf_mu": 0.25, "fatigue_mu": 0.75},
}

ARCHETYPE_WEIGHTS = [0.30, 0.40, 0.20, 0.10]   # % of student population


def clamp(val, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, val)))


def generate_student(student_id):
    archetype_name = random.choices(list(STUDENT_ARCHETYPES), weights=ARCHETYPE_WEIGHTS)[0]
    arch = STUDENT_ARCHETYPES[archetype_name]

    rows = []
    for topic in TOPICS:
        # ai_ml is harder — drag scores down a bit
        difficulty_penalty = 0.08 if topic == "ai_ml" else 0.0

        for _ in range(ROWS_PER_TOPIC):
            score       = clamp(np.random.normal(arch["score_mu"] - difficulty_penalty,
                                                  arch["score_sd"]))
            confidence  = clamp(np.random.normal(arch["conf_mu"],  0.10))
            fatigue     = clamp(np.random.normal(arch["fatigue_mu"], 0.10))
            engagement  = clamp(1.0 - fatigue + np.random.normal(0, 0.05))
            time_spent  = clamp(np.random.normal(45 + (1 - score) * 20, 10), lo=5, hi=120)
            attempts    = max(1, int(np.random.normal(2 + (1 - score) * 2, 0.8)))

            rows.append({
                "student_id": student_id,
                "name":       fake.name(),
                "archetype":  archetype_name,
                "year":       random.choice([1, 2, 3, 4]),
                "branch":     random.choice(["CSE", "IT", "ECE", "EEE"]),
                "topic":      topic,
                "score":      round(score, 4),
                "confidence": round(confidence, 4),
                "fatigue":    round(fatigue, 4),
                "engagement": round(engagement, 4),
                "time_spent": round(time_spent, 2),
                "attempts":   attempts,
            })
    return rows


def main():
    all_rows = []
    for sid in range(1, NUM_STUDENTS + 1):
        all_rows.extend(generate_student(sid))

    df = pd.DataFrame(all_rows)

    # Aggregate score per student+topic (last attempt wins for profile)
    df_agg = (
        df.groupby(["student_id", "topic"])
          .agg(score=("score", "mean"),
               confidence=("confidence", "mean"),
               fatigue=("fatigue", "mean"),
               engagement=("engagement", "mean"),
               time_spent=("time_spent", "mean"),
               attempts=("attempts", "mean"),
               name=("name", "first"),
               archetype=("archetype", "first"),
               year=("year", "first"),
               branch=("branch", "first"))
          .reset_index()
    )

    df_agg.to_csv("enhanced_students_dataset.csv", index=False)
    print(f"✅  Generated {len(df_agg)} rows for {NUM_STUDENTS} students × {len(TOPICS)} topics")
    print(df_agg.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
