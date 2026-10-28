"""
data_utils.py
=============
Utility functions for loading and validating the UCI Student Performance Dataset.

This module handles:
- Downloading or loading the dataset
- Validating required columns
- Providing basic dataset statistics

Author: Student Marks Prediction System
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to the dataset file (relative to project root)
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")

# Target column we want to predict
TARGET_COLUMN = "G3"

# Columns that could cause data leakage if used carelessly
LEAKAGE_COLUMNS = ["G3"]  # G1 and G2 are intentionally used as features

# Required columns that must be present in the dataset
REQUIRED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
    "health", "absences", "G1", "G2", "G3"
]

# Column descriptions for display in the UI
COLUMN_DESCRIPTIONS = {
    "school": "Student's school (GP = Gabriel Pereira, MS = Mousinho da Silveira)",
    "sex": "Student's sex (F = Female, M = Male)",
    "age": "Student's age (15 to 22 years)",
    "address": "Home address type (U = Urban, R = Rural)",
    "famsize": "Family size (LE3 = ≤3 members, GT3 = >3 members)",
    "Pstatus": "Parents' cohabitation status (T = Together, A = Apart)",
    "Medu": "Mother's education level (0=None, 1=Primary, 2=5th-9th grade, 3=Secondary, 4=Higher)",
    "Fedu": "Father's education level (0=None, 1=Primary, 2=5th-9th grade, 3=Secondary, 4=Higher)",
    "Mjob": "Mother's job",
    "Fjob": "Father's job",
    "reason": "Reason to choose this school",
    "guardian": "Student's guardian",
    "traveltime": "Home to school travel time (1=<15min, 2=15-30min, 3=30min-1hr, 4=>1hr)",
    "studytime": "Weekly study time (1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs)",
    "failures": "Number of past class failures (0 to 3)",
    "schoolsup": "Extra educational school support (yes/no)",
    "famsup": "Family educational support (yes/no)",
    "paid": "Extra paid classes within the course subject (yes/no)",
    "activities": "Extra-curricular activities (yes/no)",
    "nursery": "Attended nursery school (yes/no)",
    "higher": "Wants to take higher education (yes/no)",
    "internet": "Internet access at home (yes/no)",
    "romantic": "With a romantic relationship (yes/no)",
    "famrel": "Quality of family relationships (1=Very Bad to 5=Excellent)",
    "freetime": "Free time after school (1=Very Low to 5=Very High)",
    "goout": "Going out with friends (1=Very Low to 5=Very High)",
    "Dalc": "Workday alcohol consumption (1=Very Low to 5=Very High)",
    "Walc": "Weekend alcohol consumption (1=Very Low to 5=Very High)",
    "health": "Current health status (1=Very Bad to 5=Very Good)",
    "absences": "Number of school absences (0 to 93)",
    "G1": "First period grade (0 to 20) — used as a feature",
    "G2": "Second period grade (0 to 20) — used as a feature",
    "G3": "Final grade (0 to 20) — THIS IS WHAT WE PREDICT",
}


def get_data_path() -> str:
    """Return the absolute path to the dataset file."""
    return os.path.abspath(DATA_FILE)


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the UCI Student Performance dataset from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to the standard data directory.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    if filepath is None:
        filepath = get_data_path()

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n\n"
            "Please download the dataset:\n"
            "1. Visit: https://archive.ics.uci.edu/dataset/320/student+performance\n"
            "2. Download and extract the ZIP file.\n"
            "3. Copy 'student-mat.csv' into the 'data/' folder of this project."
        )

    # The UCI dataset uses semicolons as delimiters
    df = pd.read_csv(filepath, sep=";")
    return df


def validate_columns(df: pd.DataFrame) -> dict:
    """
    Validate that the dataset contains the required columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to validate.

    Returns
    -------
    dict
        A dictionary with 'valid' (bool), 'missing_columns' (list), 'message' (str).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        return {
            "valid": False,
            "missing_columns": missing,
            "message": f"Missing required columns: {missing}"
        }

    return {
        "valid": True,
        "missing_columns": [],
        "message": "All required columns are present."
    }


def validate_training_data(df: pd.DataFrame) -> dict:
    """Check that model inputs and target contain values in the dataset's ranges."""
    ranges = {
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
        TARGET_COLUMN: (0, 20),
    }
    errors = []

    for column, (low, high) in ranges.items():
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = (
            (values.isna() & (df[column].notna() | (column == TARGET_COLUMN)))
            | ~np.isfinite(values.fillna(0))
            | (values < low)
            | (values > high)
        )
        if invalid.any():
            rows = (np.flatnonzero(invalid.to_numpy())[:5] + 2).tolist()
            errors.append(f"{column} contains invalid values at CSV row(s) {rows}.")

    if errors:
        return {"valid": False, "message": " ".join(errors)}
    return {"valid": True, "message": "Training values are within the expected ranges."}


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the dataset for display in the UI.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.

    Returns
    -------
    dict
        Summary statistics and information.
    """
    summary = {
        "num_records": len(df),
        "num_features": len(df.columns) - 1,  # Exclude target
        "num_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": df.isnull().sum().sum(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "target_stats": {
            "min": float(df[TARGET_COLUMN].min()),
            "max": float(df[TARGET_COLUMN].max()),
            "mean": float(df[TARGET_COLUMN].mean()),
            "median": float(df[TARGET_COLUMN].median()),
            "std": float(df[TARGET_COLUMN].std()),
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return summary


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a detailed inspection of the dataset to the console.
    Useful during training to understand the data before processing.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect.
    """
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"Shape             : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Duplicate rows    : {df.duplicated().sum()}")
    print(f"Total missing     : {df.isnull().sum().sum()}")
    print()

    print("── Target Variable (G3) ──")
    print(df[TARGET_COLUMN].describe())
    print()

    missing_cols = df.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    if len(missing_cols) == 0:
        print("No missing values found — dataset is complete.")
    else:
        print("Columns with missing values:")
        print(missing_cols)
    print("=" * 60)
