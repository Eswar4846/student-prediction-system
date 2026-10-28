"""
app.py
======
Main entry point for the Student Marks Prediction System Streamlit app.

This file defines the Home page. Other pages live in the pages/ directory
and are automatically discovered by Streamlit's multi-page app system.

Run with:  streamlit run app.py

Author: Student Marks Prediction System
"""

import os

import streamlit as st


st.set_page_config(
    page_title="Student Marks Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .main-header p  { font-size: 1.1rem; opacity: 0.9; margin: 0; }

    /* Feature cards */
    .feature-card {
        background: #f8f9fa;
        border-left: 4px solid #1a73e8;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .feature-card h3 { color: #1a73e8; margin-bottom: 0.3rem; font-size: 1.1rem; }
    .feature-card p  { color: #555; margin: 0; font-size: 0.95rem; }

    /* Tech badge */
    .tech-badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    /* Info box */
    .info-box {
        background: #e8f5e9;
        border: 1px solid #a5d6a7;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }

    /* Divider */
    .section-divider {
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 2rem 0;
    }

    /* Sidebar styling */
    .css-1d391kg { background-color: #f0f4ff; }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="main-header">
    <h1>🎓 Student Marks Prediction System</h1>
    <p>Machine Learning-powered academic performance estimation using the UCI Student Performance Dataset</p>
</div>
""", unsafe_allow_html=True)


st.markdown("## 📋 About This Project")
st.markdown("""
This application uses **machine learning** to predict a student's expected final examination grade
based on their personal, family, and academic characteristics.

The predictions are based on the **UCI Student Performance Dataset**, which contains data
from 395 students studying Mathematics in Portuguese schools. The target grade (**G3**) ranges
from **0 to 20** using the Portuguese grading scale.

> ⚠️ **Disclaimer:** Predictions are estimates for educational purposes only.
> They should not be used to make high-stakes decisions about students.
""")
st.warning(
    "This project currently uses the UCI Mathematics dataset. Its target is G3 "
    "(0–20), studytime is a weekly category, and the dataset does not include "
    "sleep hours or attendance percentage."
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Project Objectives ────────────────────────────────────────────────────────
st.markdown("## 🎯 Project Objectives")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🔮 Predict Final Grades</h3>
        <p>Estimate a student's G3 (final grade) based on earlier performance (G1, G2) and other factors.</p>
    </div>
    <div class="feature-card">
        <h3>📊 Explore Data Insights</h3>
        <p>Visualize how study time, absences, and other factors relate to student performance.</p>
    </div>
    <div class="feature-card">
        <h3>🤖 Compare ML Algorithms</h3>
        <p>Compare Linear Regression, Decision Tree, and Random Forest side by side.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>📈 Understand Model Metrics</h3>
        <p>Learn what MAE, RMSE, and R² mean in plain language.</p>
    </div>
    <div class="feature-card">
        <h3>🧪 Reproducible Results</h3>
        <p>A fixed random seed and shared train/test split support reproducible model comparisons.</p>
    </div>
    <div class="feature-card">
        <h3>🎓 Educational Resource</h3>
        <p>Clean, well-commented code designed to help beginners understand ML concepts.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── How It Works ─────────────────────────────────────────────────────────────
st.markdown("## ⚙️ How the Prediction System Works")

steps = [
    ("1️⃣", "Data Collection", "The UCI Student Performance dataset is used. It contains 395 student records with 30+ features."),
    ("2️⃣", "Preprocessing", "Numerical features are imputed with medians. Categorical features are one-hot encoded using sklearn Pipelines."),
    ("3️⃣", "Model Training", "Three regression models are trained on 80% of the data (316 students)."),
    ("4️⃣", "Evaluation", "Each model is evaluated on the remaining 20% (79 students) using MAE, RMSE, and R²."),
    ("5️⃣", "Best Model Selection", "The model with the lowest test-set MAE is automatically selected and saved."),
    ("6️⃣", "Prediction", "Enter student details in the Predict Marks page and the best model generates a grade estimate."),
]

for icon, title, desc in steps:
    st.markdown(f"**{icon} {title}** — {desc}")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Technology Stack ──────────────────────────────────────────────────────────
st.markdown("## 🛠️ Technology Stack")
tech_items = [
    "Python 3.10+", "Streamlit", "Scikit-learn", "Pandas",
    "NumPy", "Plotly", "Matplotlib", "Joblib", "Pytest"
]
badges = "".join(f'<span class="tech-badge">{t}</span>' for t in tech_items)
st.markdown(f'<div style="margin: 0.5rem 0;">{badges}</div>', unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Navigation Guide ──────────────────────────────────────────────────────────
st.markdown("## 🗺️ Navigate the Application")

nav_col1, nav_col2 = st.columns(2)
with nav_col1:
    st.info("**🔮 Predict Marks** — Enter student information to get a grade prediction.")
    st.info("**📊 Data Analysis** — Explore charts and statistics about the dataset.")
with nav_col2:
    st.info("**🏆 Model Comparison** — Compare test-set metrics for all three algorithms.")
    st.info("**📄 Project Guide** — Installation, dataset details, and limitations in README.md.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Student Marks Predictor")
    st.markdown("**Version:** 1.0.0")
    st.markdown("**Dataset:** UCI Student Performance")
    st.markdown("**Target:** G3 (Final Grade, 0–20)")
    st.markdown("---")

    # Model status check
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
    if os.path.exists(model_path):
        st.success("✅ Model is trained and ready")
    else:
        st.warning("⚠️ Model not found. Run `python train_model.py` first.")

    st.markdown("---")
    st.markdown("**Quick Links**")
    st.markdown("- [UCI Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)")
    st.markdown("- [Scikit-learn Docs](https://scikit-learn.org/)")
    st.markdown("- [Streamlit Docs](https://docs.streamlit.io/)")
