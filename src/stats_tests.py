"""
Statistical hypothesis testing helpers used in notebook 02.
Each function returns a small, printable dict: null hyp, alt hyp, statistic,
p-value, and a plain-language conclusion at alpha = 0.05.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

ALPHA = 0.05


def chi_square_test(df: pd.DataFrame, categorical_col: str, target_col: str = "Attrition") -> dict:
    """Chi-square test of independence between a categorical feature and attrition."""
    contingency = pd.crosstab(df[categorical_col], df[target_col])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    return {
        "test": "Chi-square test of independence",
        "feature": categorical_col,
        "null_hypothesis": f"{categorical_col} is independent of Attrition.",
        "alt_hypothesis": f"{categorical_col} is associated with Attrition.",
        "statistic": round(chi2, 3),
        "dof": dof,
        "p_value": round(p, 5),
        "conclusion": (
            f"Reject H0 (p < {ALPHA}): {categorical_col} is significantly associated with Attrition."
            if p < ALPHA else
            f"Fail to reject H0 (p >= {ALPHA}): no significant association found."
        ),
    }


def numeric_group_test(df: pd.DataFrame, numeric_col: str, target_col: str = "Attrition") -> dict:
    """
    Compare a numeric feature between Attrition=Yes (1) and Attrition=No (0) groups.
    Uses Welch's t-test (unequal variances assumed) as the default; falls back to
    Mann-Whitney U (non-parametric) when the normality assumption looks unreasonable
    (small group size or heavy skew), and reports which test was used.
    """
    leavers = df.loc[df[target_col] == 1, numeric_col].dropna()
    stayers = df.loc[df[target_col] == 0, numeric_col].dropna()

    skew_leavers = abs(stats.skew(leavers))
    use_nonparametric = skew_leavers > 1.5 or len(leavers) < 30

    if use_nonparametric:
        stat, p = stats.mannwhitneyu(leavers, stayers, alternative="two-sided")
        test_name = "Mann-Whitney U test (non-parametric)"
    else:
        stat, p = stats.ttest_ind(leavers, stayers, equal_var=False)
        test_name = "Welch's t-test"

    return {
        "test": test_name,
        "feature": numeric_col,
        "null_hypothesis": f"Mean/distribution of {numeric_col} is the same for leavers and stayers.",
        "alt_hypothesis": f"{numeric_col} differs significantly between leavers and stayers.",
        "statistic": round(float(stat), 3),
        "p_value": round(float(p), 5),
        "leavers_mean": round(leavers.mean(), 2),
        "stayers_mean": round(stayers.mean(), 2),
        "conclusion": (
            f"Reject H0 (p < {ALPHA}): {numeric_col} differs significantly between leavers and stayers."
            if p < ALPHA else
            f"Fail to reject H0 (p >= {ALPHA}): no significant difference found."
        ),
    }
