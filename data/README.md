# Data Directory

## UCI Student Performance Dataset

### Download Instructions

This project uses the **UCI Student Performance Dataset**.

**Official Source:** https://archive.ics.uci.edu/dataset/320/student+performance

### How to Download

1. Visit: https://archive.ics.uci.edu/dataset/320/student+performance
2. Click the **Download** button to get the ZIP file.
3. Extract the ZIP file.
4. Inside, you will find two CSV files:
   - `student-mat.csv` — Mathematics course data (395 students)
   - `student-por.csv` — Portuguese course data (649 students)
5. Copy **`student-mat.csv`** into this `data/` folder.
6. Rename it to `student-mat.csv` (it should already have this name).

### Dataset Overview

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Dataset ID | 320 |
| Number of Students (Math) | 395 |
| Number of Students (Portuguese) | 649 |
| Number of Features | 30+ |
| Target Variable | G3 (Final Grade, scale 0–20) |

### Important Columns

| Column | Description |
|---|---|
| school | Student's school (GP or MS) |
| sex | Student's sex (F or M) |
| age | Student's age (15–22) |
| address | Urban (U) or Rural (R) |
| famsize | Family size (LE3 = ≤3, GT3 = >3) |
| Pstatus | Parents cohabitation (T = together, A = apart) |
| Medu | Mother's education (0–4) |
| Fedu | Father's education (0–4) |
| studytime | Weekly study time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h) |
| failures | Number of past class failures (0–3) |
| activities | Extra-curricular activities (yes/no) |
| higher | Wants higher education (yes/no) |
| internet | Internet access at home (yes/no) |
| absences | Number of school absences (0–93) |
| G1 | First period grade (0–20) |
| G2 | Second period grade (0–20) |
| **G3** | **Final grade (0–20) — PREDICTION TARGET** |

### Note on G1, G2, and G3

- G1 = Grade from the 1st exam period
- G2 = Grade from the 2nd exam period  
- G3 = Final grade (what we predict)

The model **uses G1 and G2 as features** because they are strong predictors of the final grade.
This simulates a real scenario where a teacher wants to estimate a student's final score
based on their earlier performance in the same academic year.

### Feature availability

This source dataset does not contain daily study hours, sleep duration, or
attendance percentage. `studytime` is a four-level weekly study-time category,
and `absences` is an absence count rather than an attendance percentage. The
application uses the fields that are present and predicts G3 on its original
0–20 scale; it does not fabricate replacements for unavailable columns.

### Privacy Notice

This dataset is publicly available from the UCI Repository and contains no personally
identifiable information about individual students. Do not replace it with real student
data from your institution.
