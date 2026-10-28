"""Compare regression models using the shared held-out test split."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.train import load_metadata


st.set_page_config(page_title="Model Comparison", page_icon="🏆", layout="wide")
st.markdown("# 🏆 Model Comparison")
st.markdown(
    "All three algorithms are evaluated on the same reproducible 20% test split. "
    "MAE and RMSE are measured in G3 grade points on the UCI 0–20 scale."
)

metadata = load_metadata()
results = metadata.get("all_results", {})
best_model = metadata.get("best_model")

if not results:
    st.warning("No evaluation results are available. Run `python train_model.py` first.")
    st.stop()

comparison = pd.DataFrame.from_dict(results, orient="index")
comparison.index.name = "Algorithm"
comparison = comparison.reset_index().rename(columns={"R2": "R²"})
comparison = comparison.sort_values("MAE").reset_index(drop=True)
comparison["Selection"] = comparison["Algorithm"].map(
    lambda name: "Selected" if name == best_model else ""
)
st.dataframe(
    comparison.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
    use_container_width=True,
    hide_index=True,
)

metric = st.selectbox("Metric to visualize", ["MAE", "RMSE", "R²"])
chart = px.bar(
    comparison,
    x="Algorithm",
    y=metric,
    color="Selection",
    color_discrete_map={"Selected": "#2e7d32", "": "#1a73e8"},
    title=f"{metric} by regression algorithm",
    text_auto=".3f",
)
st.plotly_chart(chart, use_container_width=True)

st.info(
    f"Selected model: **{best_model}** (lowest test-set MAE). The test-set "
    "metrics are useful for this educational comparison, but model selection "
    "also uses that split; use a separate validation or cross-validation study "
    "for an independent estimate of generalization."
)
