"""
Central configuration for the Employee Attrition Prediction project.
Keeping paths, seeds, and thresholds here avoids magic numbers scattered
across notebooks, src modules, and the Streamlit app.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "final_model.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "model_results.csv"

# --- Reproducibility -----------------------------------------------------
RANDOM_SEED = 42

# --- Target --------------------------------------------------------------
TARGET_COL = "Attrition"
TARGET_MAP = {"Yes": 1, "No": 0}

# Columns that are constant/zero-variance or are identifiers and therefore
# carry no predictive signal or risk leaking a row-index-like ID into the model.
LEAKAGE_OR_CONSTANT_COLS = [
    "EmployeeCount",     # constant = 1 for every row
    "StandardHours",     # constant = 80 for every row
    "Over18",             # constant = 'Y' for every row
    "EmployeeNumber",    # unique identifier, not a real feature
]

# --- Risk thresholds (configurable business thresholds, NOT a proven
# industry standard -- see README for the caveat) -------------------------
RISK_THRESHOLDS = {"LOW": (0.0, 0.40), "MEDIUM": (0.40, 0.70), "HIGH": (0.70, 1.01)}


def probability_to_risk(prob: float) -> str:
    """Map an attrition probability in [0, 1] to a LOW/MEDIUM/HIGH label."""
    for label, (low, high) in RISK_THRESHOLDS.items():
        if low <= prob < high:
            return label
    return "HIGH"  # prob == 1.0 edge case


# --- Feature groups used throughout preprocessing/EDA ---------------------
NUMERICAL_FEATURES = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
]

CATEGORICAL_FEATURES = [
    "BusinessTravel", "Department", "EducationField", "Gender", "JobRole",
    "MaritalStatus", "OverTime",
]

# Engineered features appended after feature engineering (see features.py)
ENGINEERED_FEATURES = [
    "IncomePerYearExperience", "YearsWithoutPromotionRatio", "TenureRatio",
    "AvgSatisfactionScore", "JobStabilityScore",
]
