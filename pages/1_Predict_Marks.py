"""
1_Predict_Marks.py
==================
Page 2: Predict Marks

This page provides an interactive form where users enter student details
and receive a predicted final grade (G3) from the trained model.

Author: Student Marks Prediction System
"""

import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st
from src.predict import (
    get_prediction,
    get_grade_percentage,
    predict_dataframe,
)
from src.preprocessing import FEATURE_COLUMNS
from src.train import load_metadata

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Predict Marks", page_icon="🔮", layout="wide")
st.session_state.setdefault("prediction_history", [])

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .result-card {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 2px solid #4caf50;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-grade { font-size: 4rem; font-weight: bold; color: #2e7d32; }
    .result-label { font-size: 1.2rem; color: #555; margin-top: 0.5rem; }
    .result-interpretation { font-size: 1.5rem; margin-top: 1rem; }

    .error-card {
        background: #ffebee;
        border: 2px solid #f44336;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .section-header {
        background: #e8f0fe;
        border-left: 4px solid #1a73e8;
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        color: #1a73e8;
    }

    .disclaimer {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.9rem;
        color: #6d4c41;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔮 Predict Student Marks")
st.markdown("""
Enter the student's information below and click **Predict Grade** to see the
expected final grade (G3) on a scale of **0 to 20**.
""")
st.caption(
    "Inputs reflect the UCI Mathematics dataset. It uses weekly study-time "
    "categories and has no sleep-hours or attendance-percentage field."
)

# ── Check model availability ──────────────────────────────────────────────────
metadata = load_metadata()
best_model_name = metadata.get("best_model", None)

if best_model_name is None:
    st.error(
        "⚠️ **Trained model not found.**\n\n"
        "Please run `python train_model.py` in your terminal before using this page."
    )
    st.stop()

st.info(f"🤖 **Active Model:** {best_model_name} (best performing on the test set)")

# ── Input Form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    # ── Section 1: Exam Grades ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📝 Previous Exam Grades</div>', unsafe_allow_html=True)
    st.caption("G1 and G2 are earlier exam grades in the same academic year. They are the strongest predictors of the final grade (G3).")

    col1, col2 = st.columns(2)
    with col1:
        g1 = st.slider("G1 — First Period Grade (0–20)", min_value=0, max_value=20, value=10,
                        help="The grade the student received in the first exam period.")
    with col2:
        g2 = st.slider("G2 — Second Period Grade (0–20)", min_value=0, max_value=20, value=10,
                        help="The grade the student received in the second exam period.")

    # ── Section 2: Study Habits ───────────────────────────────────────────────
    st.markdown('<div class="section-header">📚 Study Habits & Academics</div>', unsafe_allow_html=True)

    col3, col4, col5 = st.columns(3)
    with col3:
        studytime = st.selectbox(
            "Weekly Study Time",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "< 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "> 10 hours"}[x],
            index=1,
            help="How many hours per week does the student study?"
        )
    with col4:
        failures = st.selectbox(
            "Number of Past Failures",
            options=[0, 1, 2, 3],
            format_func=lambda x: f"{x} failure{'s' if x != 1 else ''}",
            index=0,
            help="How many classes has the student failed before?"
        )
    with col5:
        absences = st.number_input(
            "Number of Absences (0–93)",
            min_value=0, max_value=93, value=4,
            help="Total number of school absences."
        )

    # ── Section 3: Personal Info ───────────────────────────────────────────────
    st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)

    col6, col7, col8 = st.columns(3)
    with col6:
        sex = st.selectbox("Sex", options=["F", "M"],
                           format_func=lambda x: "Female" if x == "F" else "Male")
        age = st.number_input("Age (15–22)", min_value=15, max_value=22, value=16)
    with col7:
        address = st.selectbox("Home Address", options=["U", "R"],
                               format_func=lambda x: "Urban" if x == "U" else "Rural")
        famsize = st.selectbox("Family Size", options=["LE3", "GT3"],
                               format_func=lambda x: "≤ 3 members" if x == "LE3" else "> 3 members")
    with col8:
        higher = st.selectbox("Wants Higher Education?", options=["yes", "no"])
        internet = st.selectbox("Internet Access at Home?", options=["yes", "no"])

    # ── Section 4: Family Background ─────────────────────────────────────────
    st.markdown('<div class="section-header">👨‍👩‍👧 Family Background</div>', unsafe_allow_html=True)

    col9, col10 = st.columns(2)
    with col9:
        medu = st.selectbox("Mother's Education Level", options=[0, 1, 2, 3, 4],
                            format_func=lambda x: {0:"None",1:"Primary (up to 4th grade)",2:"5th–9th grade",3:"Secondary education",4:"Higher education"}[x],
                            index=2)
        mjob = st.selectbox("Mother's Job",
                            options=["teacher", "health", "services", "at_home", "other"],
                            index=4)
    with col10:
        fedu = st.selectbox("Father's Education Level", options=[0, 1, 2, 3, 4],
                            format_func=lambda x: {0:"None",1:"Primary (up to 4th grade)",2:"5th–9th grade",3:"Secondary education",4:"Higher education"}[x],
                            index=2)
        fjob = st.selectbox("Father's Job",
                            options=["teacher", "health", "services", "at_home", "other"],
                            index=4)

    # ── Section 5: Support & Lifestyle ────────────────────────────────────────
    st.markdown('<div class="section-header">🏫 Support & Lifestyle</div>', unsafe_allow_html=True)

    col11, col12, col13 = st.columns(3)
    with col11:
        schoolsup = st.selectbox("Extra School Support?", options=["yes", "no"], index=1)
        famsup    = st.selectbox("Family Educational Support?", options=["yes", "no"])
        paid      = st.selectbox("Extra Paid Classes?", options=["yes", "no"], index=1)
        activities = st.selectbox("Extra-curricular Activities?", options=["yes", "no"])
    with col12:
        romantic  = st.selectbox("In a Romantic Relationship?", options=["yes", "no"], index=1)
        traveltime = st.selectbox("Travel Time to School",
                                  options=[1, 2, 3, 4],
                                  format_func=lambda x: {1:"< 15 min",2:"15–30 min",3:"30 min–1 hr",4:"> 1 hr"}[x])
        famrel    = st.slider("Family Relationship Quality (1=Bad, 5=Excellent)", 1, 5, 4)
        freetime  = st.slider("Free Time After School (1=Low, 5=High)", 1, 5, 3)
    with col13:
        goout  = st.slider("Going Out with Friends (1=Low, 5=High)", 1, 5, 3)
        dalc   = st.slider("Weekday Alcohol Consumption (1=Low, 5=High)", 1, 5, 1)
        walc   = st.slider("Weekend Alcohol Consumption (1=Low, 5=High)", 1, 5, 1)
        health = st.slider("Current Health Status (1=Bad, 5=Good)", 1, 5, 3)

    # ── Submit ────────────────────────────────────────────────────────────────
    submitted = st.form_submit_button("🔮 Predict Grade", use_container_width=True, type="primary")

# ── Prediction Result ─────────────────────────────────────────────────────────
if submitted:
    # Build input dictionary matching exactly what the model expects
    input_data = {
        "G1": g1, "G2": g2,
        "studytime": studytime, "failures": failures, "absences": absences,
        "sex": sex, "age": age, "address": address, "famsize": famsize,
        "Medu": medu, "Fedu": fedu, "Mjob": mjob, "Fjob": fjob,
        "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "higher": higher, "internet": internet,
        "romantic": romantic, "famrel": famrel, "freetime": freetime,
        "goout": goout, "Dalc": dalc, "Walc": walc, "health": health,
        "traveltime": traveltime,
    }

    with st.spinner("Running prediction..."):
        result = get_prediction(input_data)

    if result["success"]:
        grade = result["prediction"]
        pct   = get_grade_percentage(grade)
        interp = result["interpretation"]
        st.session_state["prediction_history"].append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Type": "Single student",
            "Predicted G3": f"{grade:.2f}",
        })

        st.markdown(f"""
        <div class="result-card">
            <div class="result-grade">{grade} / 20</div>
            <div class="result-label">Predicted Final Grade (G3) &nbsp;|&nbsp; Equivalent: {pct}%</div>
            <div class="result-interpretation">{interp}</div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar
        st.progress(int(grade * 5))  # 0-20 scale → 0-100 for progress bar

        # Details expander
        with st.expander("📋 Input Summary"):
            summary_df = pd.Series(input_data, name="Value", dtype="string").to_frame()
            st.dataframe(summary_df, use_container_width=True)

    else:
        st.markdown(f"""
        <div class="error-card">
            <b>⚠️ Prediction Error</b><br>{result['error']}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> This prediction is a statistical estimate based on historical data.
        It is intended for educational and analytical purposes only. Actual examination results
        depend on many factors that this model cannot fully capture.
        Do not use this for high-stakes academic decisions.
    </div>
    """, unsafe_allow_html=True)

# ── Bulk CSV Predictions ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📥 Bulk Predictions from CSV")
st.markdown(
    "Upload a CSV with one row per student and all model input columns. "
    "An optional `G3` column is retained in the output but is never used as an input."
)
st.download_button(
    "Download CSV header template",
    data=pd.DataFrame(columns=FEATURE_COLUMNS).to_csv(index=False),
    file_name="student_prediction_template.csv",
    mime="text/csv",
)

uploaded_file = st.file_uploader("Upload student records CSV", type=["csv"])
if uploaded_file is not None and st.button("Predict uploaded records", type="primary"):
    try:
        uploaded_data = pd.read_csv(uploaded_file, sep=None, engine="python")
        results = predict_dataframe(uploaded_data)
    except (
        FileNotFoundError,
        ValueError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as error:
        st.error(f"Could not predict the uploaded file: {error}")
    else:
        st.success(f"Generated predictions for {len(results)} student records.")
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button(
            "Download predictions",
            data=results.to_csv(index=False),
            file_name="student_marks_predictions.csv",
            mime="text/csv",
        )
        st.session_state["prediction_history"].append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Type": "CSV upload",
            "Predicted G3": f"{len(results)} records",
        })

if st.session_state["prediction_history"]:
    with st.expander("🕘 This session's prediction history"):
        st.dataframe(
            pd.DataFrame(st.session_state["prediction_history"]),
            use_container_width=True,
            hide_index=True,
        )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📖 Grade Scale Reference")
    st.markdown("""
    | Grade | Classification |
    |-------|---------------|
    | 18–20 | 🌟 Excellent |
    | 14–17 | ✅ Good |
    | 10–13 | 📘 Satisfactory |
    | 8–9   | ⚠️ Needs Improvement |
    | 0–7   | ❌ At Risk |
    """)
    st.markdown("---")
    st.markdown("ℹ️ The scale follows the **Portuguese grading system** used in the UCI dataset.")
