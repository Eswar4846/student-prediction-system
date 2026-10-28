"""
train.py
========
Model training functions for the Student Marks Prediction System.

This module:
- Builds complete sklearn Pipelines (preprocessing + model)
- Trains Linear Regression, Decision Tree, and Random Forest models
- Evaluates each model on the held-out test set
- Selects the best model based on lowest MAE
- Saves the best pipeline and metadata

Author: Student Marks Prediction System
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import build_preprocessor, prepare_features_and_target, get_feature_columns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Random seed for reproducibility — always use the same seed so results don't change
RANDOM_SEED = 42

# Fraction of data to hold out for testing
TEST_SIZE = 0.2


def get_model_definitions() -> dict:
    """
    Return a dictionary of model name → unfitted sklearn estimator.

    Three models are compared:
    1. Linear Regression  — Simple, fast, assumes linear relationships
    2. Decision Tree      — Learns if/else rules from the data
    3. Random Forest      — Many decision trees combined (ensemble)

    Returns
    -------
    dict
        Keys are model names, values are unfitted estimator objects.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_SEED,
            max_depth=10,        # Limit depth to avoid overfitting
            min_samples_leaf=5,  # Each leaf must have at least 5 samples
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,    # 100 decision trees in the forest
            random_state=RANDOM_SEED,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,           # Use all CPU cores for speed
        ),
    }


def build_full_pipeline(model) -> Pipeline:
    """
    Build a complete sklearn Pipeline: preprocessing → model.

    Using a Pipeline ensures that:
    - Preprocessing (imputation, encoding) is fit only on training data
    - The same transformations are automatically applied during prediction

    Parameters
    ----------
    model : sklearn estimator
        An unfitted sklearn regression model.

    Returns
    -------
    Pipeline
        A complete pipeline ready to be fitted.
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def split_data(df: pd.DataFrame) -> tuple:
    """
    Split the dataset into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X, y = prepare_features_and_target(df)

    # train_test_split randomly splits the data
    # test_size=0.2 means 20% goes to testing, 80% to training
    # random_state ensures we get the same split every time
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    return X_train, X_test, y_train, y_test


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a fitted pipeline on the test set.

    Metrics:
    - MAE  : Average of |actual - predicted|. Easy to interpret (same units as G3).
    - RMSE : Square root of average squared errors. Penalises large errors more.
    - R²   : How well the model explains variation (1.0 = perfect, 0.0 = just predicting the mean).

    Parameters
    ----------
    pipeline : Pipeline
        A fitted sklearn Pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Actual target values.

    Returns
    -------
    dict
        Dictionary containing MAE, RMSE, and R² scores.
    """
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
    }


def train_all_models(df: pd.DataFrame) -> dict:
    """
    Train all three models and evaluate them on the same test set.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset.

    Returns
    -------
    dict
        Results dictionary containing:
        - 'results': dict of model name → metrics
        - 'best_model_name': str
        - 'pipelines': dict of model name → fitted pipeline
        - 'X_test': test features
        - 'y_test': actual test labels
    """
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    model_definitions = get_model_definitions()
    results = {}
    pipelines = {}

    print("\nTraining models...")
    print("-" * 50)

    for model_name, model in model_definitions.items():
        print(f"\n  Training: {model_name}...")

        # Build and train the full pipeline
        pipeline = build_full_pipeline(model)
        pipeline.fit(X_train, y_train)

        # Evaluate on the held-out test set
        metrics = evaluate_model(pipeline, X_test, y_test)

        results[model_name] = metrics
        pipelines[model_name] = pipeline

        print(f"    MAE  = {metrics['MAE']:.4f}")
        print(f"    RMSE = {metrics['RMSE']:.4f}")
        print(f"    R²   = {metrics['R2']:.4f}")

    # Select the best model: lowest MAE on the test set
    best_model_name = min(results, key=lambda name: results[name]["MAE"])

    print("\n" + "=" * 50)
    print(f"BEST MODEL: {best_model_name}")
    print(f"  (Selected by lowest test-set MAE = {results[best_model_name]['MAE']:.4f})")
    print("=" * 50)

    return {
        "results": results,
        "best_model_name": best_model_name,
        "pipelines": pipelines,
        "X_test": X_test,
        "y_test": y_test,
    }


def save_best_model(pipeline: Pipeline, model_name: str, results: dict, df: pd.DataFrame) -> None:
    """
    Save the best fitted pipeline to disk using Joblib.
    Also save a JSON metadata file with model information.

    Parameters
    ----------
    pipeline : Pipeline
        The best fitted pipeline.
    model_name : str
        Name of the best model.
    results : dict
        All model evaluation results.
    df : pd.DataFrame
        The training dataset (used to record feature info).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save the entire pipeline (preprocessing + model)
    # This ensures predictions use the same transformations as training
    joblib.dump(pipeline, BEST_MODEL_PATH)
    print(f"\nModel saved to: {BEST_MODEL_PATH}")

    # Save metadata as JSON for the UI to read
    metadata = {
        "dataset_source": "UCI Machine Learning Repository — Student Performance Dataset",
        "dataset_url": "https://archive.ics.uci.edu/dataset/320/student+performance",
        "target_column": "G3",
        "target_scale": "0 to 20 (Portuguese grading scale)",
        "best_model": model_name,
        "selection_criterion": "Lowest test-set MAE",
        "all_results": results,
        "features": get_feature_columns(),
        "num_training_samples": len(df),
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
        "note_on_g1_g2": (
            "G1 and G2 are first and second period grades. "
            "They are used as features because they precede G3 in time. "
            "This is NOT data leakage — it simulates predicting the final "
            "grade based on earlier exam performance within the same school year."
        ),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {METADATA_PATH}")


def load_best_pipeline() -> Pipeline:
    """
    Load the saved best pipeline from disk.

    Returns
    -------
    Pipeline
        The fitted sklearn Pipeline ready for predictions.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist (train_model.py has not been run).
    """
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at: {BEST_MODEL_PATH}\n"
            "Please run 'python train_model.py' first to train the model."
        )
    return joblib.load(BEST_MODEL_PATH)


def load_metadata() -> dict:
    """
    Load model metadata from the JSON file.

    Returns
    -------
    dict
        Metadata dictionary, or a default dict if file is not found.
    """
    if not os.path.exists(METADATA_PATH):
        return {
            "note": "Metadata not available. Please run train_model.py first.",
            "best_model": None,
            "all_results": {},
        }

    with open(METADATA_PATH, "r") as f:
        return json.load(f)
