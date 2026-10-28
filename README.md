# Student Marks Prediction System

A Streamlit application that trains and compares regression models on the UCI
Student Performance Mathematics dataset, predicts final grades, and explores
the dataset through interactive charts.

## Dataset and scope

The included `data/student-mat.csv` contains 395 Mathematics student records
from the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance).
Its target is `G3`, the final Portuguese grade on a **0–20** scale. The project
uses earlier grades (`G1`, `G2`), study-time category, absences, and selected
academic, demographic, and lifestyle fields.

This source dataset does **not** contain daily study hours, attendance
percentage, or sleep duration. In particular, `studytime` is a weekly category
(less than 2 hours, 2–5, 5–10, or more than 10 hours), and `absences` is a count.
The application does not fabricate or infer the missing fields or represent
the UCI grade as a 0–100 model target. Use an appropriately documented dataset
if those exact features and scale are required.

The dataset is public and contains no direct student identifiers. Predictions
are educational estimates, not guaranteed results or suitable inputs to
high-stakes decisions.

## Requirements

- Python 3.10 or later
- pip

## Install and run

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Training creates `models/best_model.pkl` and updates
`models/model_metadata.json`. The saved Joblib pipeline includes both
preprocessing and the selected estimator. Training uses a reproducible 80/20
train/test split and compares Linear Regression, Decision Tree Regressor, and
Random Forest Regressor using MAE, RMSE, and R².

The pipeline keeps imputation and categorical encoding inside the estimator,
so those transformations are fitted on training records only. The model with
the lowest test-set MAE is saved. Because the same test split is used for model
selection and reporting, its metrics are an educational comparison rather than
an independent final generalization estimate.

## Application pages

- **Home:** project overview and dataset limitations.
- **Predict Marks:** single-student form and bulk CSV prediction. Download the
  header template on the page; include all model input columns, with one
  student per row. An optional `G3` column is carried through but is not used
  as a model input. Predictions can be downloaded as CSV.
- **Data Analysis:** dataset summaries, grade distributions, and feature
  visualizations.
- **Model Comparison:** held-out metrics for each estimator. Train the model
  first to populate the results.

Prediction history is held only in the current Streamlit session and is not
written to disk.

## Validate

Run the test suite with:

```powershell
python -m pytest
```

See [`data/README.md`](data/README.md) for dataset schema and source details.
