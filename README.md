# Employee Attrition Prediction & HR Intelligence System

An end-to-end data science system that predicts which employees are at risk of
leaving, quantifies that risk, and **explains why** — built as a portfolio
project demonstrating the full lifecycle from raw data to a deployed
application.

---

## 1. Project Overview

Employee turnover is expensive: recruiting, onboarding, and lost productivity
can cost well over an employee's annual salary. This project builds a
classification model that estimates an individual employee's probability of
attrition, converts that probability into an HR-friendly risk category (LOW /
MEDIUM / HIGH), and uses SHAP to explain the top factors behind each
prediction — all surfaced through an interactive Streamlit application.

## 2. Business Problem

> Can we identify, ahead of time, which employees are likely to leave, and
> tell HR *why*, so retention efforts can be targeted rather than blanket?

This is framed as a **binary classification** problem (`Attrition`: Yes/No)
where the minority class (Yes, ~16% of employees) is the one that matters
most operationally — missing a true leaver (a false negative) is the costliest
error, since no retention action is taken for that employee.

## 3. Objectives

- Clean and understand the dataset without discarding signal.
- Run EDA and formal statistical tests to ground business claims in evidence.
- Engineer a small set of interpretable, HR-meaningful features.
- Train and fairly compare four ML models using a leak-free pipeline.
- Select a final model on a metric appropriate for imbalanced classification.
- Explain every prediction dynamically (no hard-coded explanations).
- Ship a working Streamlit application for both business and technical users.

## 4. Dataset

**IBM HR Analytics Employee Attrition & Performance** — 1,470 employees, 35
attributes, publicly available (e.g. via Kaggle or IBM's sample-data
repositories). Expected file: `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`.
The loader (`src/preprocessing.py::load_raw_data`) is configurable — any CSV
with the same schema can be substituted at that path.

- Target: `Attrition` (`Yes` → 1, `No` → 0)
- No missing values, no duplicate rows
- ~16% positive class (moderately imbalanced)

## 5. Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | pandas, numpy, scikit-learn, XGBoost |
| Stats | scipy |
| Explainability | SHAP (with a documented importance/coefficient fallback) |
| Visualization | matplotlib, seaborn, Plotly |
| App | Streamlit |
| Persistence | joblib |

## 6. Architecture / Pipeline

```text
Raw CSV -> Data Understanding -> Cleaning -> EDA -> Statistical Tests
        -> Feature Engineering -> ColumnTransformer Preprocessing
        -> Train/Test Split (stratified) -> Baseline -> 4 ML Models
        -> Stratified Cross-Validation -> Hyperparameter Tuning (top 2)
        -> Model Comparison -> Final Model -> SHAP Explainability
        -> Streamlit Application
```

All pipeline logic lives once in `src/` and is imported by both the notebooks
(for documented analysis) and the Streamlit app (for serving) — there is a
single source of truth, so the app's numbers always match the notebooks.

## 7. Data Preprocessing

- Dropped constant/zero-variance columns (`EmployeeCount`, `StandardHours`,
  `Over18`) and the row identifier (`EmployeeNumber`) — no signal, only noise
  or leakage risk.
- No missing-value imputation was strictly necessary (0 missing cells), but
  the `ColumnTransformer` still includes median/most-frequent imputers so the
  pipeline is robust to any future data with gaps.
- Outliers were **inspected, not blindly removed** — high-tenure, high-income
  values are legitimate, not data errors, and tree models tolerate them well.
- All preprocessing (imputers, scaler, one-hot encoder) is fit **only on the
  training split**, inside a scikit-learn `Pipeline`, to prevent train/test
  leakage.

## 8. EDA — Key Findings

- Attrition is concentrated in the **first few years of tenure**.
- **Overtime** shows a markedly higher attrition rate than no-overtime.
- Lower **job satisfaction**, **environment satisfaction**, and **work-life
  balance** scores correlate with higher attrition.
- **Sales Representative** and **Laboratory Technician** roles have the
  highest attrition rates among job roles.
- Leavers skew younger and have lower monthly income than stayers.

See `notebooks/02_eda_and_statistics.ipynb` for the full set of charts.

## 9. Statistical Analysis

Hypothesis tests (α = 0.05) formalize the EDA observations:

| Question | Test | Result |
|---|---|---|
| Is overtime associated with attrition? | Chi-square | Significant |
| Is business travel frequency associated with attrition? | Chi-square | Significant |
| Is marital status associated with attrition? | Chi-square | Significant |
| Does income differ between leavers/stayers? | Mann-Whitney U | Significant |
| Does age differ between leavers/stayers? | Welch's t-test | Significant |
| Does tenure differ between leavers/stayers? | Mann-Whitney U | Significant |

**These are associations, not causal claims** — see Limitations.

## 10. Feature Engineering

| Feature | Rationale |
|---|---|
| `IncomePerYearExperience` | Pay relative to career experience |
| `YearsWithoutPromotionRatio` | Share of tenure spent without advancement (stagnation) |
| `TenureRatio` | Company tenure relative to total career length |
| `AvgSatisfactionScore` | Composite of four noisy satisfaction survey items |
| `JobStabilityScore` | Average time in a role before moving (restlessness proxy) |

## 11. ML Models

Baseline (majority-class dummy) plus four candidates, all imbalance-aware via
`class_weight='balanced'` (XGBoost: `scale_pos_weight`) rather than SMOTE —
chosen because the imbalance is moderate (not extreme) and this avoids
synthesizing artificial employee records:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

## 12. Model Comparison

*(Generated by `src/train.py`; see `reports/model_results.csv` for the exact
numbers reproduced by this run.)*

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline (Majority Class) | 0.840 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.772 | 0.378 | 0.660 | 0.481 | 0.805 |
| Decision Tree | 0.786 | 0.333 | 0.340 | 0.337 | 0.605 |
| Random Forest | 0.847 | 0.625 | 0.106 | 0.182 | 0.789 |
| XGBoost | 0.837 | 0.482 | 0.277 | 0.351 | 0.774 |
| **XGBoost (Tuned) — final** | 0.833 | 0.474 | 0.383 | **0.424** | 0.751 |
| Logistic Regression (Tuned) | 0.742 | 0.351 | 0.723 | 0.472 | 0.794 |

**Final model: XGBoost (Tuned)** — selected by mean 5-fold CV F1; the held-out
test set is reserved for final evaluation. Accuracy alone is misleading here:
the baseline scores 84% accuracy while catching zero actual leavers.
Cross-validation (5-fold stratified, scoring = F1) was used both to rank
candidates before tuning and to drive `RandomizedSearchCV` during tuning,
which reduces the risk of overfitting to one particular train/test split.

## 13. Explainable AI

SHAP (`TreeExplainer` for tree-based models) provides:

- **Global feature importance** — which features matter most across all
  employees (Overtime, satisfaction scores, income, stock options, and age
  are consistently among the top drivers).
- **Local, per-employee explanations** — the top contributing factors and
  their direction (increases / decreases risk) for one specific prediction,
  computed live and never hard-coded.

If SHAP cannot be applied to a given model, the app automatically falls back
to native `feature_importances_` / coefficients, clearly labeled as an
approximation in the UI.

## 14. Key Business Insights

*(Correlational, not causal — see Limitations.)*

- Overtime is associated with a higher attrition rate; possibly linked to
  burnout or reduced work-life balance.
- Lower job/environment satisfaction correlates with higher exit rates —
  engagement surveys could serve as an early-warning signal.
- Shorter tenure (first 2–3 years) is linked to elevated attrition risk,
  suggesting onboarding and early-career support as a retention lever.
- Certain job roles (Sales Representative, Laboratory Technician) show
  consistently higher attrition and may warrant role-specific programs.

## 15. Streamlit Application

Six pages: **Home**, **HR Dashboard**, **Employee Risk Prediction**,
**Model Performance**, **Employee Insights**, and **About Project**. The
prediction page takes a full employee profile and returns a live probability,
risk band, and top 4 contributing factors.

## 16. Installation

```bash
git clone https://github.com/RuchiK353/hr-attrition-intelligence-system.git
cd hr-attrition-intelligence-system
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Place the dataset at `data/WA_Fn-UseC_-HR-Employee-Attrition.csv` if it is not
already present (see Dataset section for the source).

## 17. How to Run

```bash
# 1. Train the model (creates models/final_model.pkl and reports/model_results.csv)
python -m src.train

# 2. Launch the Streamlit app
streamlit run app/app.py
```

Notebooks in `notebooks/` can be run independently (`jupyter lab` /
`jupyter notebook`) for the full documented analysis — they import the same
`src/` modules used by the app.

## 18. Results

- **Final model:** XGBoost (Tuned)
- **Test F1 (Attrition class):** 0.423
- **Test ROC-AUC:** 0.751
- Full comparison table: `reports/model_results.csv`

## 19. Limitations

- Single-company snapshot dataset — findings may not generalize to other
  organizations or time periods.
- All relationships are **correlational**; no causal claims are made or
  should be inferred from this analysis.
- The model predicts *likelihood*, not *timing*, of attrition.
- Risk thresholds (LOW/MEDIUM/HIGH at 40%/70%) are **configurable business
  choices**, not a validated industry standard.
- Recall on the minority class (~0.49) means roughly half of true leavers are
  still missed — this is a decision-support tool, not a substitute for
  ongoing HR judgment.

## 20. Future Improvements

- Collect longitudinal data to model *time-to-attrition* (survival analysis).
- Add fairness auditing across gender/age/marital-status subgroups.
- Explore cost-sensitive thresholding tuned to actual replacement-cost data.
- Add model monitoring for feature and prediction drift in production.

## 21. Author

Built as a Data Science portfolio project.
