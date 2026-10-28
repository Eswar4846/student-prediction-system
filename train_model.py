"""
train_model.py
==============
Main training script for the Student Marks Prediction System.

Run this script to:
1. Load the UCI Student Performance dataset
2. Inspect and validate the data
3. Train Linear Regression, Decision Tree, and Random Forest models
4. Evaluate all models on the same held-out test set
5. Select the best model by lowest MAE
6. Save the best pipeline and metadata

Usage:
    python train_model.py

Output:
    models/best_model.pkl       — Trained pipeline (preprocessing + model)
    models/model_metadata.json  — Training configuration and evaluation results

Author: Student Marks Prediction System
"""

import sys
import os

# Some Windows terminals still use legacy code pages. Reconfigure stdout/stderr
# to UTF-8 so console output can display the project labels safely.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path so 'src' module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_utils import (
    load_data,
    validate_columns,
    validate_training_data,
    inspect_data,
    get_data_path,
)
from src.train import train_all_models, save_best_model


def main():
    print("=" * 60)
    print("  STUDENT MARKS PREDICTION - MODEL TRAINING")
    print("=" * 60)

    # Step 1: Load Dataset
    print("\n[1/5] Loading dataset...")
    try:
        df = load_data()
        print(f"      Loaded {len(df)} student records from: {get_data_path()}")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease follow the instructions in data/README.md to download the dataset.")
        sys.exit(1)

    # Step 2: Validate Columns
    print("\n[2/5] Validating dataset columns...")
    validation = validate_columns(df)
    if not validation["valid"]:
        print(f"\nERROR: {validation['message']}")
        print("Please make sure you are using the correct dataset (student-mat.csv).")
        sys.exit(1)
    print(f"      {validation['message']}")

    validation = validate_training_data(df)
    if not validation["valid"]:
        print(f"\nERROR: {validation['message']}")
        sys.exit(1)
    print(f"      {validation['message']}")

    # Remove duplicates if any
    initial_count = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_count:
        print(f"      Removed {initial_count - len(df)} duplicate rows.")

    # Step 3: Inspect Data
    print("\n[3/5] Inspecting dataset...")
    inspect_data(df)

    # ── Step 4: Train All Models ─────────────────────────────────
    print("\n[4/5] Training all models...")
    training_output = train_all_models(df)

    results        = training_output["results"]
    best_name      = training_output["best_model_name"]
    pipelines      = training_output["pipelines"]

    # ── Step 5: Save Best Model ───────────────────────────────────
    print("\n[5/5] Saving best model...")
    best_pipeline = pipelines[best_name]
    save_best_model(best_pipeline, best_name, results, df)

    # ── Print Final Summary ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<22} {'MAE':>7} {'RMSE':>8} {'R²':>8}")
    print("-" * 50)
    for model_name, metrics in results.items():
        marker = " ← BEST" if model_name == best_name else ""
        print(
            f"{model_name:<22} "
            f"{metrics['MAE']:>7.4f} "
            f"{metrics['RMSE']:>8.4f} "
            f"{metrics['R2']:>8.4f}"
            f"{marker}"
        )
    print("=" * 60)
    print("\nNote: G1 and G2 (earlier exam grades) are used as features.")
    print("      They are NOT data leakage — they precede G3 in time.")
    print("\nNext step: Run the app with:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
