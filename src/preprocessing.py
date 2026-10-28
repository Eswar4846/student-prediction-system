"""
preprocessing.py
================
Data preprocessing pipeline for the Student Marks Prediction System.

This module builds a scikit-learn preprocessing pipeline that:
- Separates numerical and categorical columns
- Handles missing values (imputation)
- Encodes categorical columns using OneHotEncoder
- Uses ColumnTransformer to apply different steps to different columns

IMPORTANT: The pipeline is fitted ONLY on training data to prevent data leakage.
           It is then applied (transformed) to test data.

Author: Student Marks Prediction System
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from src.data_utils import TARGET_COLUMN


# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------

# These are the features the model will use for prediction.
# G1 and G2 are included intentionally — they are earlier exam grades that
# help predict the final grade. This is NOT data leakage because G1 and G2
# happen before G3 (the final exam) in the academic calendar.
FEATURE_COLUMNS = [
    # Earlier exam grades (strong predictors)
    "G1", "G2",
    # Study habits
    "studytime", "failures", "absences",
    # Personal / demographic
    "sex", "age", "address", "famsize",
    # Family background
    "Medu", "Fedu", "Mjob", "Fjob",
    # School support
    "schoolsup", "famsup", "paid",
    # Lifestyle
    "activities", "higher", "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    # Travel
    "traveltime",
]

# Numerical columns that need imputation with median
NUMERICAL_FEATURES = [
    "G1", "G2", "studytime", "failures", "absences",
    "age", "Medu", "Fedu", "famrel", "freetime",
    "goout", "Dalc", "Walc", "health", "traveltime",
]

# Categorical columns that need encoding
CATEGORICAL_FEATURES = [
    "sex", "address", "famsize", "Mjob", "Fjob",
    "schoolsup", "famsup", "paid", "activities",
    "higher", "internet", "romantic",
]


def get_feature_columns() -> list:
    """Return the list of feature columns used by the model."""
    return FEATURE_COLUMNS.copy()


def get_numerical_features() -> list:
    """Return the list of numerical feature column names."""
    return NUMERICAL_FEATURES.copy()


def get_categorical_features() -> list:
    """Return the list of categorical feature column names."""
    return CATEGORICAL_FEATURES.copy()


def build_preprocessor() -> ColumnTransformer:
    """
    Build and return a scikit-learn ColumnTransformer for preprocessing.

    What this does:
    - For numerical columns: fills missing values with the median, then passes them through.
    - For categorical columns: fills missing values with 'unknown', then one-hot encodes them.

    One-Hot Encoding converts categorical strings (like 'yes'/'no') into
    numerical 0/1 columns so machine learning algorithms can work with them.

    Returns
    -------
    ColumnTransformer
        An unfitted preprocessing transformer ready to be placed in a Pipeline.
    """

    # Numerical pipeline:
    # Step 1: Fill missing values with the median (robust to outliers)
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ])

    # Categorical pipeline:
    # Step 1: Fill missing values with the string 'unknown'
    # Step 2: One-hot encode — turn each category into a 0/1 column
    #         handle_unknown='ignore' prevents errors if a new category appears during prediction
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # ColumnTransformer applies different pipelines to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # Drop any columns not listed above
    )

    return preprocessor


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate input features from the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset including target column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X (features) and y (target values).
    """
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    X = df[FEATURE_COLUMNS].copy()

    # Extract the target column (G3 = final grade, 0–20)
    y = df[TARGET_COLUMN].copy()

    return X, y


def validate_prediction_dataframe(data: pd.DataFrame) -> str | None:
    """Return a validation error for prediction inputs, or None when valid."""
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        return f"Missing required feature columns: {', '.join(missing)}."
    if data.empty:
        return "The uploaded CSV does not contain any student records."

    errors = []
    numerical_ranges = {
        "G1": (0, 20),
        "G2": (0, 20),
        "studytime": (1, 4),
        "failures": (0, 3),
        "absences": (0, 93),
        "age": (15, 22),
        "Medu": (0, 4),
        "Fedu": (0, 4),
        "famrel": (1, 5),
        "freetime": (1, 5),
        "goout": (1, 5),
        "Dalc": (1, 5),
        "Walc": (1, 5),
        "health": (1, 5),
        "traveltime": (1, 4),
    }

    for column in NUMERICAL_FEATURES:
        values = pd.to_numeric(data[column], errors="coerce")
        low, high = numerical_ranges[column]
        invalid = values.isna() | ~np.isfinite(values) | (values < low) | (values > high)
        invalid |= values.notna() & (values % 1 != 0)
        for position in np.flatnonzero(invalid.to_numpy())[:5]:
            errors.append(
                f"Row {position + 2}, '{column}' must be a whole number between "
                f"{low} and {high}."
            )

    allowed_categories = {
        "sex": {"F", "M"},
        "address": {"U", "R"},
        "famsize": {"LE3", "GT3"},
        "Mjob": {"teacher", "health", "services", "at_home", "other"},
        "Fjob": {"teacher", "health", "services", "at_home", "other"},
        "schoolsup": {"yes", "no"},
        "famsup": {"yes", "no"},
        "paid": {"yes", "no"},
        "activities": {"yes", "no"},
        "higher": {"yes", "no"},
        "internet": {"yes", "no"},
        "romantic": {"yes", "no"},
    }
    for column in CATEGORICAL_FEATURES:
        values = data[column].astype("string").str.strip()
        invalid = values.isna() | ~values.isin(allowed_categories[column])
        for position in np.flatnonzero(invalid.to_numpy())[:5]:
            errors.append(
                f"Row {position + 2}, '{column}' must be one of "
                f"{', '.join(sorted(allowed_categories[column]))}."
            )

    if errors:
        suffix = " Additional invalid values may be present." if len(errors) == 5 else ""
        return " ".join(errors[:5]) + suffix
    return None


def validate_input_for_prediction(input_dict: dict) -> tuple[bool, str]:
    """
    Validate a dictionary of user inputs before making a prediction.

    Parameters
    ----------
    input_dict : dict
        Dictionary of feature name → value from user input.

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    error = validate_prediction_dataframe(pd.DataFrame([input_dict]))
    return error is None, error or ""


def dict_to_dataframe(input_dict: dict) -> pd.DataFrame:
    """
    Convert a user input dictionary into a single-row DataFrame
    suitable for model prediction.

    Parameters
    ----------
    input_dict : dict
        A dictionary where keys are feature names.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row and columns matching FEATURE_COLUMNS.
    """
    # Only keep the columns the model expects
    filtered = {col: [input_dict[col]] for col in FEATURE_COLUMNS if col in input_dict}
    return pd.DataFrame(filtered)
