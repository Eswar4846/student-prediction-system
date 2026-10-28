"""
2_Data_Analysis.py
==================
Page 3: Data Analysis

Displays charts and statistics about the UCI Student Performance Dataset.

Author: Student Marks Prediction System
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_utils import load_data, get_dataset_summary, COLUMN_DESCRIPTIONS

st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")

st.markdown("# 📊 Data Analysis")
st.markdown("""
Explore the UCI Student Performance Dataset. Charts show relationships between
student characteristics and final grades.

> 🔔 **Note:** Relationships shown here do **not** imply causation. For example,
> fewer absences being associated with higher grades does not mean attendance
> alone causes better performance.
""")

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    try:
        return load_data()
    except FileNotFoundError:
        return None

df = get_data()

if df is None:
    st.error(
        "⚠️ **Dataset not found.**\n\n"
        "Please download `student-mat.csv` and place it in the `data/` folder.\n"
        "See `data/README.md` for instructions."
    )
    st.stop()

summary = get_dataset_summary(df)

# ── Overview Metrics ──────────────────────────────────────────────────────────
st.markdown("## 📋 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Student Records", summary["num_records"])
col2.metric("Number of Features", summary["num_features"])
col3.metric("Missing Values", summary["total_missing"])
col4.metric("Duplicate Rows", summary["duplicate_rows"])

st.markdown("---")

# ── Target Distribution ───────────────────────────────────────────────────────
st.markdown("## 🎯 Distribution of Final Grades (G3)")
st.caption("G3 is the target variable — what the model predicts. Scale: 0 to 20.")

fig_hist = px.histogram(
    df, x="G3",
    nbins=21,
    color_discrete_sequence=["#1a73e8"],
    labels={"G3": "Final Grade (G3)", "count": "Number of Students"},
    title="Distribution of Final Grades (G3)",
)
fig_hist.update_layout(bargap=0.05, xaxis=dict(dtick=1))
st.plotly_chart(fig_hist, use_container_width=True)

# ── Grade Stats ───────────────────────────────────────────────────────────────
ts = summary["target_stats"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Min Grade", ts["min"])
c2.metric("Max Grade", ts["max"])
c3.metric("Mean Grade", round(ts["mean"], 2))
c4.metric("Median Grade", ts["median"])
c5.metric("Std Dev", round(ts["std"], 2))

st.markdown("---")

# ── Study Time vs Grade ───────────────────────────────────────────────────────
st.markdown("## 📚 Study Time vs Final Grade")
st.caption("Students who study more tend to score higher on average — but individual results vary.")

study_map = {1: "< 2 hrs", 2: "2–5 hrs", 3: "5–10 hrs", 4: "> 10 hrs"}
df_plot = df.copy()
df_plot["Study Time Label"] = df_plot["studytime"].map(study_map)

fig_study = px.box(
    df_plot, x="Study Time Label", y="G3",
    category_orders={"Study Time Label": list(study_map.values())},
    color="Study Time Label",
    labels={"G3": "Final Grade (G3)", "Study Time Label": "Weekly Study Time"},
    title="Final Grade Distribution by Study Time",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
st.plotly_chart(fig_study, use_container_width=True)

st.markdown("---")

# ── Absences vs Grade ─────────────────────────────────────────────────────────
st.markdown("## 🏫 Absences vs Final Grade")
st.caption("Higher absences are often associated with lower final grades.")

fig_abs = px.scatter(
    df, x="absences", y="G3",
    labels={"absences": "Number of Absences", "G3": "Final Grade (G3)"},
    title="School Absences vs Final Grade",
    color_discrete_sequence=["#e91e63"],
    opacity=0.6,
)
st.plotly_chart(fig_abs, use_container_width=True)

st.markdown("---")

# ── G1, G2, G3 Correlation ────────────────────────────────────────────────────
st.markdown("## 🔗 Grade Progression: G1 → G2 → G3")
st.caption("Earlier grades are the strongest predictors of the final grade.")

col_a, col_b = st.columns(2)
with col_a:
    fig_g1g3 = px.scatter(
        df, x="G1", y="G3",
        labels={"G1": "First Period Grade (G1)", "G3": "Final Grade (G3)"},
        title="G1 vs G3",
        color_discrete_sequence=["#1a73e8"], opacity=0.6,
    )
    st.plotly_chart(fig_g1g3, use_container_width=True)

with col_b:
    fig_g2g3 = px.scatter(
        df, x="G2", y="G3",
        labels={"G2": "Second Period Grade (G2)", "G3": "Final Grade (G3)"},
        title="G2 vs G3",
        color_discrete_sequence=["#0d47a1"], opacity=0.6,
    )
    st.plotly_chart(fig_g2g3, use_container_width=True)

st.markdown("---")

# ── Gender vs Grade ───────────────────────────────────────────────────────────
st.markdown("## 👥 Final Grade by Gender")
fig_sex = px.box(
    df, x="sex", y="G3",
    color="sex",
    labels={"sex": "Sex (F=Female, M=Male)", "G3": "Final Grade (G3)"},
    title="Grade Distribution by Gender",
    color_discrete_map={"F": "#e91e63", "M": "#1a73e8"},
)
st.plotly_chart(fig_sex, use_container_width=True)

st.markdown("---")

# ── Failures vs Grade ─────────────────────────────────────────────────────────
st.markdown("## ❌ Past Failures vs Final Grade")
st.caption("Students with more past class failures tend to score lower on their final exam.")

fail_avg = df.groupby("failures")["G3"].mean().reset_index()
fail_avg.columns = ["Past Failures", "Average Final Grade"]
fig_fail = px.bar(
    fail_avg, x="Past Failures", y="Average Final Grade",
    color="Average Final Grade",
    color_continuous_scale="RdYlGn",
    title="Average Final Grade by Number of Past Failures",
    labels={"Average Final Grade": "Avg G3 (0–20)"},
)
st.plotly_chart(fig_fail, use_container_width=True)

st.markdown("---")

# ── Higher Education Aspiration ────────────────────────────────────────────────
st.markdown("## 🎓 Higher Education Aspiration vs Final Grade")
fig_higher = px.violin(
    df, x="higher", y="G3",
    color="higher",
    labels={"higher": "Wants Higher Education?", "G3": "Final Grade (G3)"},
    title="Grade Distribution by Higher Education Aspiration",
    color_discrete_map={"yes": "#4caf50", "no": "#f44336"},
    box=True,
)
st.plotly_chart(fig_higher, use_container_width=True)

st.markdown("---")

# ── Correlation Heatmap ───────────────────────────────────────────────────────
st.markdown("## 🔥 Correlation Matrix (Numerical Features)")
st.caption("Shows how strongly each numerical feature is correlated with others. Values close to 1 or -1 indicate strong correlation.")

numerical_cols = ["G1", "G2", "G3", "studytime", "failures", "absences",
                  "age", "Medu", "Fedu", "famrel", "freetime", "goout",
                  "Dalc", "Walc", "health", "traveltime"]
corr_matrix = df[numerical_cols].corr()

fig_corr = px.imshow(
    corr_matrix,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Correlation Matrix",
    aspect="auto",
    text_auto=".2f",
)
fig_corr.update_layout(height=600)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# ── Raw Data Preview ──────────────────────────────────────────────────────────
st.markdown("## 🗃️ Dataset Preview")
st.caption(f"Showing the first 20 rows of the dataset ({len(df)} total rows).")

with st.expander("View raw dataset sample"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("View column descriptions"):
    desc_df = pd.DataFrame(
        [(col, desc) for col, desc in COLUMN_DESCRIPTIONS.items()],
        columns=["Column", "Description"]
    )
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

with st.expander("View missing values report"):
    missing = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"])
    missing["Percentage"] = (missing["Missing Values"] / len(df) * 100).round(2)
    st.dataframe(missing, use_container_width=True)
