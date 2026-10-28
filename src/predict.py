"""
predict.py
==========
Prediction interface for the Student Marks Prediction System.

This module provides a clean API for making predictions using the
saved trained pipeline. The Streamlit UI calls functions from this module.

Author: Student Marks Prediction System
"""

import numpy as np
import pandas as pd

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    validate_input_for_prediction,
    validate_prediction_dataframe,
    dict_to_dataframe,
)
from src.train import load_best_pipeline


def predict_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Validate a batch, predict G3, and return a copy with predicted_G3 added."""
    error = validate_prediction_dataframe(data)
    if error:
        raise ValueError(error)

    features = data.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERICAL_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="raise")
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].str.strip()

    pipeline = load_best_pipeline()
    predictions = np.clip(pipeline.predict(features), 0.0, 20.0)
    result = data.copy()
    result["predicted_G3"] = np.round(predictions, 2)
    return result


def get_prediction(input_dict: dict) -> dict:
    """
    Make a final grade prediction for a single student.

    This function:
    1. Validates the input dictionary
    2. Converts it to a DataFrame
    3. Loads the saved pipeline
    4. Makes a prediction
    5. Returns the result with interpretation

    Parameters
    ----------
    input_dict : dict
        Dictionary of feature name → value for one student.

    Returns
    -------
    dict
        {
          'success': bool,
          'prediction': float or None,
          'grade_out_of': 20,
          'interpretation': str,
          'error': str or None,
        }
    """
    # Step 1: Validate inputs
    is_valid, error_message = validate_input_for_prediction(input_dict)
    if not is_valid:
        return {
            "success": False,
            "prediction": None,
            "grade_out_of": 20,
            "interpretation": "",
            "error": error_message,
        }

    try:
        prediction = predict_dataframe(dict_to_dataframe(input_dict))
        predicted_grade = float(prediction["predicted_G3"].iloc[0])
    except FileNotFoundError as e:
        return {
            "success": False,
            "prediction": None,
            "grade_out_of": 20,
            "interpretation": "",
            "error": str(e),
        }

    return {
        "success": True,
        "prediction": predicted_grade,
        "grade_out_of": 20,
        "interpretation": interpret_grade(predicted_grade),
        "error": None,
    }


def interpret_grade(grade: float) -> str:
    """
    Return a human-readable interpretation of a grade on the 0–20 scale.

    This follows the Portuguese grading system used in the UCI dataset.

    Parameters
    ----------
    grade : float
        Predicted grade between 0 and 20.

    Returns
    -------
    str
        A descriptive label for the grade.
    """
    if grade >= 18:
        return "🌟 Excellent (18–20)"
    elif grade >= 14:
        return "✅ Good (14–17)"
    elif grade >= 10:
        return "📘 Satisfactory (10–13)"
    elif grade >= 8:
        return "⚠️ Needs Improvement (8–9)"
    else:
        return "❌ At Risk of Failing (0–7)"


def get_grade_percentage(grade: float) -> float:
    """
    Convert a 0–20 grade to a 0–100 percentage for display purposes only.
    The model always works in the 0–20 scale internally.

    Parameters
    ----------
    grade : float
        Grade on the 0–20 scale.

    Returns
    -------
    float
        Equivalent percentage.
    """
    return round((grade / 20.0) * 100, 1)
