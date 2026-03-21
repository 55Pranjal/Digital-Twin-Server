"""
app.py
------
Streamlit UI for EduTwin.

Two views:
  Teacher View — Class-wide overview, risk heatmap, insights for all students
  Student View — Personal twin, recommendations, LLM explanations, simulations
"""

import io
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from digital_twin import (
    COURSE_TOPICS,
    TARGET_SCORE,
    WEAK_THRESHOLD,
    best_intervention,
    build_profile,
    check_goal,
    compute_performance,
    diagnose_weaknesses_with_llm,
    generate_personalized_explanation,
    generate_study_plan,
    load_dataset,
    load_memory,
    plot_radar,
    predict_performance_llm,
    predict_struggle,
    recommend_actions,
    run_multiple_simulations,
    save_memory,
    simulate_exam_answer,
    simulate_intervention,
    student_memory,
    update_memory,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduTwin",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 4px solid #4c78c8;
        margin-bottom: 10px;
    }
    .risk-high   { border-left-color: #e74c3c !important; }
    .risk-medium { border-left-color: #f39c12 !important; }
    .risk-low    { border-left-color: #27ae60 !important; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
RISK_EMOJI = {"High Risk": "🔴", "Medium Risk": "🟠", "Low Risk": "🟢"}
RISK_COLOR = {"High Risk": "#e74c3c",  "Medium Risk": "#f39c12", "Low Risk": "#27ae60"}


@st.cache_data(show_spinner=False)
def get_all_profiles(num_students: int) -> list[dict]:
    profiles = []
    for sid in range(1, num_students + 1):
        try:
            p = build_profile(sid, force_rebuild=True)
            p["_risk"] = predict_struggle(p)
            profiles.append(p)
        except Exception:
            pass
    return profiles


def radar_chart_bytes(profile: dict) -> bytes:
    buf = io.BytesIO()
    plot_radar(profile, save_path=None)
    plt.figure()
    plot_radar(profile)
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def progress_chart_bytes(student_id: str) -> bytes | None:
    history = student_memory.get(str(student_id), [])
    if len(history) < 2:
        return None
    subjects = list(history[0]["knowledge"].keys())
    fig, ax  = plt.subplots(figsize=(9, 4))
    for s in subjects:
        vals = [step["knowledge"].get(s, 0) for step in history]
        ax.plot(vals, marker="o", label=s)
    ax.axhline(TARGET_SCORE, color="gray", linestyle="--", linewidth=0.8, label="Target")
    ax.set_xlabel("Memory Step")
    ax.set_ylabel("Score")
    ax.set_title(f"Topic Progress — Student {student_id}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130)
    plt.close()
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# INITIALISE STATE
# ─────────────────────────────────────────────
if "memory_loaded" not in st.session_state:
    load_memory()
    st.session_state["memory_loaded"] = True

try:
    df = load_dataset()
    NUM_STUDENTS = df["student_id"].nunique()
except FileNotFoundError:
    st.error("❌ Dataset not found. Please run `python generate_data.py` first.")
    st.stop()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/graduation-cap.png", width=48)
    st.title("EduTwin")
    st.caption("LLM-Powered Student Digital Twin")
    st.divider()

    view = st.radio("Select view", ["🏫 Teacher View", "🎒 Student View"], index=0)
    st.divider()

    if view == "🎒 Student View":
        student_id = st.selectbox(
            "Select student",
            options=list(range(1, NUM_STUDENTS + 1)),
            format_func=lambda x: f"Student {x}",
        )
    else:
        student_id = None

    st.divider()
    st.caption(f"Dataset: {NUM_STUDENTS} students · {len(COURSE_TOPICS)} topics")
    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    llm_status = "✅ Groq API" if groq_configured else "⚠️ Ollama (local)"
    st.caption(f"LLM: {llm_status}")


# ═══════════════════════════════════════════════════════════
# TEACHER VIEW
# ═══════════════════════════════════════════════════════════
if view == "🏫 Teacher View":
    st.title("🏫 Teacher Dashboard")
    st.caption("Class-wide insights, risk monitoring, and cohort analytics.")

    with st.spinner("Loading all student profiles…"):
        all_profiles = get_all_profiles(NUM_STUDENTS)

    # ── KPI row ──────────────────────────────────────
    perfs        = [compute_performance(p) for p in all_profiles]
    risks        = [p["_risk"]["risk"] for p in all_profiles]
    high_risk_n  = risks.count("High Risk")
    med_risk_n   = risks.count("Medium Risk")
    goal_met_n   = sum(1 for p in all_profiles if check_goal(p))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Students", NUM_STUDENTS)
    k2.metric("Avg Score", f"{round(np.mean(perfs) * 100, 1)}%")
    k3.metric("🔴 High Risk", high_risk_n)
    k4.metric("✅ Goal Met", goal_met_n)

    st.divider()

    # ── Two-column layout ─────────────────────────────
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown('<div class="section-header">📊 Topic Performance Heatmap</div>',
                    unsafe_allow_html=True)

        heatmap_data = {
            t: [p["knowledge"].get(t, 0) for p in all_profiles]
            for t in COURSE_TOPICS
        }
        hm_df = pd.DataFrame(heatmap_data)
        hm_df.index = [f"S{p['student_id']}" for p in all_profiles]

        fig, ax = plt.subplots(figsize=(10, max(4, NUM_STUDENTS // 4)))
        im = ax.imshow(hm_df.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(COURSE_TOPICS)))
        ax.set_xticklabels(COURSE_TOPICS, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(all_profiles)))
        ax.set_yticklabels(hm_df.index, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title("Score per student per topic", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        st.markdown('<div class="section-header">⚠️ Risk Distribution</div>',
                    unsafe_allow_html=True)

        risk_counts = {"High Risk": high_risk_n, "Medium Risk": med_risk_n,
                       "Low Risk": risks.count("Low Risk")}
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        colors = [RISK_COLOR[k] for k in risk_counts]
        ax2.pie(risk_counts.values(), labels=risk_counts.keys(),
                colors=colors, autopct="%1.0f%%", startangle=140)
        ax2.set_title("Student Risk Levels")
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.markdown('<div class="section-header">📉 Weakest Topics Class-Wide</div>',
                    unsafe_allow_html=True)

        topic_avg = {t: np.mean([p["knowledge"].get(t, 0) for p in all_profiles])
                     for t in COURSE_TOPICS}
        sorted_topics = sorted(topic_avg.items(), key=lambda x: x[1])
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        topics_sorted = [x[0] for x in sorted_topics]
        avgs_sorted   = [x[1] for x in sorted_topics]
        bars = ax3.barh(topics_sorted, avgs_sorted,
                        color=["#e74c3c" if v < WEAK_THRESHOLD else "#27ae60" for v in avgs_sorted])
        ax3.axvline(WEAK_THRESHOLD, color="gray", linestyle="--", linewidth=0.8)
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Average Score")
        ax3.set_title("Topic averages (class)", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    # ── Student risk table ────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">🗂 All Students — Risk Table</div>',
                unsafe_allow_html=True)

    table_rows = []
    for p in all_profiles:
        r = p["_risk"]
        table_rows.append({
            "ID":         int(p["student_id"]),
            "Name":       p["name"],
            "Branch":     p["branch"],
            "Year":       p["year"],
            "Avg Score":  f"{round(compute_performance(p)*100,1)}%",
            "Risk":       r["risk"],
            "Risk Score": round(r["score"], 3),
            "Weak Topics": ", ".join(p["weak_topics"]) or "—",
            "Goal":       "✅" if check_goal(p) else "❌",
        })

    tdf = pd.DataFrame(table_rows)

    def color_risk(val):
        c = {"High Risk": "#ffe0e0", "Medium Risk": "#fff3cd", "Low Risk": "#d4edda"}
        return f"background-color: {c.get(val, '')}"

    st.dataframe(
        tdf.style.applymap(color_risk, subset=["Risk"]),
        use_container_width=True,
        height=400,
    )

    # ── Class-level LLM insight ───────────────────────
    st.divider()
    st.markdown('<div class="section-header">🤖 Class-Level AI Insight</div>',
                unsafe_allow_html=True)

    if st.button("Generate class-wide insight (LLM)", type="primary"):
        from digital_twin import call_llm
        summary_block = "\n".join(
            f"- {p['name']}: avg {round(compute_performance(p)*100,1)}%, "
            f"risk={p['_risk']['risk']}, weak={','.join(p['weak_topics'][:2]) or 'none'}"
            for p in all_profiles[:20]  # cap to avoid token overflow
        )
        prompt = (
            f"You are a university course coordinator reviewing class performance.\n\n"
            f"Student summaries (sample of up to 20):\n{summary_block}\n\n"
            "Provide 3 concrete, actionable teaching interventions for the class as a whole. "
            "Identify the most at-risk subgroups and suggest targeted strategies. "
            "Keep your response under 250 words."
        )
        with st.spinner("Generating insight…"):
            insight = call_llm(prompt)
        st.info(insight)


# ═══════════════════════════════════════════════════════════
# STUDENT VIEW
# ═══════════════════════════════════════════════════════════
else:
    st.title(f"🎒 Student Digital Twin")

    with st.spinner("Building profile…"):
        profile = build_profile(student_id, force_rebuild=True)
        update_memory(student_id, profile)
        save_memory()

    risk   = predict_struggle(profile)
    perf   = compute_performance(profile)

    # ── Header ─────────────────────────────────────────
    h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
    h1.markdown(f"### {profile['name']}")
    h1.caption(f"Year {profile['year']} · {profile['branch']} · "
               f"Archetype: {profile.get('archetype', 'N/A')}")
    h2.metric("Avg Score",  f"{round(perf * 100, 1)}%")
    h3.metric("Confidence", profile["psychology"]["confidence_level"].capitalize())
    h4.metric("Risk",       f"{RISK_EMOJI.get(risk['risk'],'')} {risk['risk']}")

    st.info(f"**Profile summary:** {profile['summary']}")
    st.divider()

    # ── Tabs ───────────────────────────────────────────
    tab_overview, tab_llm, tab_sim, tab_plan, tab_history = st.tabs([
        "📊 Overview",
        "🤖 AI Capabilities",
        "🧪 Simulations",
        "📅 Study Plan",
        "🕰 History",
    ])

    # ── TAB 1: Overview ─────────────────────────────────
    with tab_overview:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("**📚 Topic Scores**")
            score_df = pd.DataFrame([
                {"Topic": t,
                 "Score": round(v * 100, 1),
                 "Status": "✅ Strong" if v >= TARGET_SCORE else ("⚠️ Weak" if v < WEAK_THRESHOLD else "🔶 Fair")}
                for t, v in sorted(profile["knowledge"].items(), key=lambda x: x[1])
            ])
            st.dataframe(score_df, use_container_width=True, hide_index=True)

            st.markdown("**🧠 Behavior Metrics**")
            beh = profile["behavior"]
            b_df = pd.DataFrame([
                {"Metric": "Avg Time (min)",   "Value": beh["avg_time"]},
                {"Metric": "Avg Attempts",     "Value": beh["avg_attempts"]},
                {"Metric": "Confidence",       "Value": beh["avg_confidence"]},
                {"Metric": "Engagement",       "Value": beh["engagement"]},
                {"Metric": "Fatigue",          "Value": beh["fatigue"]},
            ])
            st.dataframe(b_df, use_container_width=True, hide_index=True)

        with c2:
            st.markdown("**🕸 Knowledge Radar**")
            topics = list(profile["knowledge"].keys())
            scores = [profile["knowledge"][t] for t in topics]
            N      = len(topics)
            angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
            scores_radar = scores + [scores[0]]

            fig_r, ax_r = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
            ax_r.plot(angles, scores_radar, linewidth=2, color="#4c78c8")
            ax_r.fill(angles, scores_radar, alpha=0.25, color="#4c78c8")
            ax_r.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], topics, fontsize=9)
            ax_r.set_ylim(0, 1)
            ax_r.axhline(TARGET_SCORE, color="red", linestyle="--",
                         linewidth=0.7, alpha=0.5, label="Target")
            ax_r.set_title(f"{profile['name']}", pad=18, fontsize=10)
            st.pyplot(fig_r, use_container_width=True)
            plt.close()

            st.markdown("**💡 Recommendations**")
            for r in recommend_actions(profile):
                st.markdown(f"- {r}")

    # ── TAB 2: AI Capabilities ──────────────────────────
    with tab_llm:
        st.markdown("#### 🤖 LLM-Powered Twin Capabilities")

        # Weakness Diagnosis
        with st.expander("🔍 LLM Weakness Diagnosis", expanded=True):
            if st.button("Diagnose weaknesses (LLM)", key="btn_diag"):
                with st.spinner("Analysing profile…"):
                    diag = diagnose_weaknesses_with_llm(profile)
                st.markdown(diag)

        # Personalized Explanation
        with st.expander("📘 Personalized Explanation", expanded=False):
            concept = st.selectbox("Choose a concept to explain:", COURSE_TOPICS, key="explain_sel")
            if st.button("Generate explanation", key="btn_explain"):
                with st.spinner(f"Generating explanation for {concept}…"):
                    exp = generate_personalized_explanation(profile, concept)
                st.markdown(exp)

        # Performance Prediction
        with st.expander("📈 Performance Prediction", expanded=False):
            pred_topic = st.selectbox("Predict performance on:", COURSE_TOPICS, key="pred_sel")
            if st.button("Predict performance", key="btn_pred"):
                with st.spinner("Predicting…"):
                    pred = predict_performance_llm(profile, pred_topic)
                st.markdown(pred)

        # Exam Answer Simulation
        with st.expander("📝 Exam Answer Simulation", expanded=False):
            sim_topic = st.selectbox("Topic:", COURSE_TOPICS, key="sim_topic")
            exam_q    = st.text_area(
                "Enter exam question:",
                value="Explain the concept of gradient descent and why the learning rate matters.",
                key="exam_q",
            )
            if st.button("Simulate student's answer", key="btn_sim"):
                with st.spinner("Simulating…"):
                    answer = simulate_exam_answer(profile, exam_q, sim_topic)
                st.markdown(f"**Simulated Answer:**\n\n{answer}")

    # ── TAB 3: Simulations ──────────────────────────────
    with tab_sim:
        st.markdown("#### 🧪 Intervention Simulations")
        st.caption("Simulate how each intervention strategy would change the student's performance.")

        with st.spinner("Running simulations…"):
            sims     = run_multiple_simulations(profile)
            best_act = best_intervention(profile)

        # Bar chart
        sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
        actions  = [s[0] for s in sorted_sims]
        perf_vals = [round(s[1] * 100, 1) for s in sorted_sims]
        baseline = round(perf * 100, 1)

        fig_s, ax_s = plt.subplots(figsize=(8, 4))
        colors = ["#27ae60" if a == best_act else "#4c78c8" for a in actions]
        bars = ax_s.barh(actions, perf_vals, color=colors)
        ax_s.axvline(baseline, color="gray", linestyle="--", linewidth=1, label=f"Current: {baseline}%")
        ax_s.set_xlabel("Projected Avg Score (%)")
        ax_s.set_title("Projected Performance per Intervention")
        ax_s.legend()
        for bar, val in zip(bars, perf_vals):
            ax_s.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                      f"{val}%", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_s, use_container_width=True)
        plt.close()

        st.success(f"🏆 **Best strategy:** `{best_act}` — projected score: **{round(sims[best_act]*100,1)}%**")

        # Apply intervention
        st.markdown("---")
        chosen = st.selectbox("Apply an intervention:", list(sims.keys()))
        if st.button("Apply & update profile", key="btn_apply"):
            new_profile = simulate_intervention(profile, chosen)
            update_memory(student_id, new_profile)
            save_memory()
            st.success(f"✅ Intervention '{chosen}' applied. Profile updated in memory.")
            st.markdown(f"**New summary:** {new_profile['summary']}")

    # ── TAB 4: Study Plan ───────────────────────────────
    with tab_plan:
        st.markdown("#### 📅 Personalized Study Plan")
        days = st.slider("Plan duration (days):", 3, 14, 7)
        if st.button("Generate study plan (LLM)", key="btn_plan"):
            with st.spinner("Creating personalised plan…"):
                plan = generate_study_plan(profile, days)
            st.markdown(plan)

    # ── TAB 5: History ──────────────────────────────────
    with tab_history:
        st.markdown("#### 🕰 Memory History")
        history = student_memory.get(str(student_id), [])

        if not history:
            st.info("No history yet. Interact with other tabs to build memory.")
        else:
            st.caption(f"{len(history)} memory snapshots recorded.")

            # Progress chart
            if len(history) >= 2:
                topics = list(history[0]["knowledge"].keys())
                fig_h, ax_h = plt.subplots(figsize=(9, 4))
                for topic in topics:
                    vals = [h["knowledge"].get(topic, 0) for h in history]
                    ax_h.plot(vals, marker="o", label=topic)
                ax_h.axhline(TARGET_SCORE, color="gray", linestyle="--", linewidth=0.7, label="Target")
                ax_h.set_xlabel("Snapshot")
                ax_h.set_ylabel("Score")
                ax_h.set_title("Topic Progression Over Time")
                ax_h.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_h, use_container_width=True)
                plt.close()

            # Raw memory viewer
            with st.expander("View raw memory snapshots"):
                for i, snap in enumerate(reversed(history[-5:])):
                    st.markdown(f"**Snapshot {len(history) - i}** — "
                                f"`{pd.Timestamp(snap.get('timestamp', 0), unit='s').strftime('%Y-%m-%d %H:%M')}`")
                    st.json({k: v for k, v in snap.items() if k != "timestamp"}, expanded=False)
