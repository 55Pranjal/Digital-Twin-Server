import pandas as pd
import requests
import matplotlib.pyplot as plt
import time
import json
from dotenv import load_dotenv

load_dotenv()

df = pd.read_csv("enhanced_students_dataset.csv")

MEMORY_FILE = "student_memory.json"
student_memory = {}

# -------------------------------
# MEMORY LOAD/SAVE
# -------------------------------
def load_memory():
    global student_memory
    try:
        with open(MEMORY_FILE, "r") as f:
            student_memory = json.load(f)
    except:
        student_memory = {}

def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(student_memory, f)

# -------------------------------
# UTILITY
# -------------------------------
def clean_value(x):
    return float(round(x, 2))

# -------------------------------
# BUILD PROFILE
# -------------------------------
def build_profile(student_id):
    student_id = str(student_id)

    if student_id in student_memory and len(student_memory[student_id]) > 0:
        return student_memory[student_id][-1]

    student_df = df[df["student_id"] == int(student_id)]

    profile = {
        "student_id": student_id,
        "knowledge": {},
        "weak_topics": [],
        "behavior": {}
    }

    for _, row in student_df.iterrows():
        topic = row["topic"]
        score = row["score"]

        profile["knowledge"][topic] = score
        if score < 0.5:
            profile["weak_topics"].append(topic)

    profile["behavior"] = {
        "avg_time": clean_value(student_df["time_spent"].mean()),
        "avg_attempts": clean_value(student_df["attempts"].mean()),
        "avg_confidence": clean_value(student_df["confidence"].mean()),
        "engagement": clean_value(student_df["engagement"].mean()),
        "fatigue": clean_value(student_df["fatigue"].mean())
    }

    return profile

# -------------------------------
# PSYCHOLOGY
# -------------------------------
def add_psychology(profile):
    confidence = profile["behavior"]["avg_confidence"]
    fatigue = profile["behavior"]["fatigue"]

    profile["psychology"] = {
        "confidence_level": "low" if confidence < 0.5 else "high",
        "focus": "low" if fatigue > 0.5 else "good",
        "learning_style": "slow" if confidence < 0.4 else "normal"
    }

    return profile

# -------------------------------
# MEMORY UPDATE
# -------------------------------
def update_memory(student_id, profile):
    student_id = str(student_id)

    if student_id not in student_memory:
        student_memory[student_id] = []

    profile["timestamp"] = time.time()
    student_memory[student_id].append(profile)

# -------------------------------
# PREDICTION
# -------------------------------
def predict_struggle(profile):
    score = (
        0.4 * len(profile["weak_topics"]) +
        0.3 * (1 - profile["behavior"]["avg_confidence"]) +
        0.3 * profile["behavior"]["fatigue"]
    )

    if score > 1.5:
        return {"risk": "High Risk"}
    elif score > 0.8:
        return {"risk": "Medium Risk"}
    else:
        return {"risk": "Low Risk"}

# -------------------------------
# PERFORMANCE SCORE
# -------------------------------
def compute_performance(profile):
    return sum(profile["knowledge"].values()) / len(profile["knowledge"])

# -------------------------------
# GOAL CHECK
# -------------------------------
TARGET = 0.7

def check_goal(profile):
    return all(v >= TARGET for v in profile["knowledge"].values())

# -------------------------------
# RECOMMENDATION ENGINE
# -------------------------------
def recommend_actions(profile):
    recs = []

    if profile["weak_topics"]:
        recs.append(f"Focus on: {profile['weak_topics']}")

    if profile["behavior"]["avg_confidence"] < 0.5:
        recs.append("Revise basics and practice easy questions")

    if profile["behavior"]["fatigue"] > 0.5:
        recs.append("Reduce study time / take breaks")

    if profile["behavior"]["engagement"] > 0.7:
        recs.append("Student motivated → increase difficulty")

    return recs

# -------------------------------
# SIMULATION (ADVANCED)
# -------------------------------
def simulate_intervention(profile, action):
    simulated = json.loads(json.dumps(profile))

    for topic in simulated["weak_topics"]:
        improvement = 0.1

        # subject difficulty
        if topic == "ai_ml":
            improvement = 0.05

        simulated["knowledge"][topic] += improvement
        simulated["knowledge"][topic] = min(1.0, simulated["knowledge"][topic])

    if action == "increase_practice":
        simulated["behavior"]["avg_confidence"] += 0.1

    elif action == "reduce_fatigue":
        simulated["behavior"]["fatigue"] -= 0.2

    simulated["weak_topics"] = [
        t for t, v in simulated["knowledge"].items() if v < 0.5
    ]

    return simulated

# -------------------------------
# MULTI SIMULATION
# -------------------------------
def run_multiple_simulations(profile):
    actions = ["concept_learning", "increase_practice", "reduce_fatigue"]
    results = {}

    for action in actions:
        sim = simulate_intervention(profile, action)
        results[action] = compute_performance(sim)

    return results

# -------------------------------
# FEEDBACK LOOP
# -------------------------------
def update_from_real_outcome(profile, actual_score):
    predicted = profile["behavior"]["avg_confidence"]
    error = actual_score - predicted

    profile["behavior"]["avg_confidence"] += 0.2 * error
    profile["behavior"]["avg_confidence"] = max(0, min(1, profile["behavior"]["avg_confidence"]))

    return profile

# -------------------------------
# LLM
# -------------------------------
def call_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:8b",
                "prompt": prompt,
                "stream": False
            },
            timeout=500
        )
        return response.json().get("response", "")
    except Exception as e:
        return str(e)

# -------------------------------
# GRAPH
# -------------------------------
def plot_subject_progress(student_id):
    student_id = str(student_id)

    history = student_memory.get(student_id, [])
    if not history:
        return

    subjects = history[0]["knowledge"].keys()

    plt.figure()
    for subject in subjects:
        values = [step["knowledge"][subject] for step in history]
        plt.plot(values, label=subject)

    plt.xlabel("Time Step")
    plt.ylabel("Score")
    plt.title("Subject-wise Progress")
    plt.legend()
    plt.show()

# -------------------------------
# RUN
# -------------------------------
def run_test(student_id):
    print("\n===== DIGITAL TWIN OUTPUT =====\n")

    load_memory()

    profile = build_profile(student_id)
    profile = add_psychology(profile)
    update_memory(student_id, profile)

    print("📊 PROFILE:", profile)

    print("\n📈 PERFORMANCE:", compute_performance(profile))

    prediction = predict_struggle(profile)
    print("\n⚠️ RISK:", prediction["risk"])

    print("\n💡 RECOMMENDATIONS:")
    for r in recommend_actions(profile):
        print("-", r)

    print("\n🧪 MULTI-STRATEGY SIMULATION:")
    sims = run_multiple_simulations(profile)
    print(sims)

    best_action = max(sims, key=sims.get)
    print("🏆 BEST STRATEGY:", best_action)

    profile = simulate_intervention(profile, best_action)
    update_memory(student_id, profile)

    print("\n📘 LLM EXPLANATION:")
    print(call_llm("Explain gradient descent simply"))

    profile = update_from_real_outcome(profile, 0.7)
    update_memory(student_id, profile)
    save_memory()

    print("\n🎯 GOAL ACHIEVED:", check_goal(profile))

    plot_subject_progress(student_id)

# -------------------------------
if __name__ == "__main__":
    run_test(1)