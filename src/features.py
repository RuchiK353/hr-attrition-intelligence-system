"""
Feature engineering.

Every engineered feature has an explicit HR/business rationale (see comments).
Applied identically to train and test / single-employee inference input, and
built only from raw columns available at prediction time, so there is no leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # IncomePerYearExperience: pay relative to career experience. Employees who
    # feel underpaid *relative to their experience* are a classic attrition driver.
    out["IncomePerYearExperience"] = out["MonthlyIncome"] / (out["TotalWorkingYears"] + 1)

    # YearsWithoutPromotionRatio: share of the employee's tenure spent without a
    # promotion. High ratio -> stagnation, a known driver of voluntary exit.
    out["YearsWithoutPromotionRatio"] = out["YearsSinceLastPromotion"] / (out["YearsAtCompany"] + 1)

    # TenureRatio: how much of total career has been spent at THIS company.
    # Low ratio despite long total experience can signal a "job hopper" profile.
    out["TenureRatio"] = out["YearsAtCompany"] / (out["TotalWorkingYears"] + 1)

    # AvgSatisfactionScore: single composite of the four satisfaction-style
    # survey questions, since they are individually noisy but jointly meaningful.
    satisfaction_cols = [
        "JobSatisfaction", "EnvironmentSatisfaction",
        "RelationshipSatisfaction", "WorkLifeBalance",
    ]
    out["AvgSatisfactionScore"] = out[satisfaction_cols].mean(axis=1)

    # JobStabilityScore: how long, on average, the employee stays in a role
    # before moving (internally or externally). Low value -> restless profile.
    out["JobStabilityScore"] = out["YearsInCurrentRole"] / (out["NumCompaniesWorked"] + 1)

    # Guard against inf/nan from any unexpected zero-division edge cases.
    engineered = [
        "IncomePerYearExperience", "YearsWithoutPromotionRatio",
        "TenureRatio", "AvgSatisfactionScore", "JobStabilityScore",
    ]
    out[engineered] = out[engineered].replace([np.inf, -np.inf], np.nan)
    out[engineered] = out[engineered].fillna(out[engineered].median())

    return out
