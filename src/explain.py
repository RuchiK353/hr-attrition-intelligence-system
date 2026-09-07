"""
Explainability layer: SHAP-based global and local explanations, with an
automatic fallback to model feature_importances_ / coefficients if SHAP
cannot be applied to the loaded model (e.g. an unsupported estimator type).

All outputs are computed dynamically from the fitted pipeline -- nothing
here is hard-coded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False


def _get_feature_names(preprocessor) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def _transform(pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.named_steps["prep"].transform(X)


def build_explainer(pipeline, X_background: pd.DataFrame):
    """
    Build a SHAP explainer for the model inside `pipeline`, on already-raw
    (untransformed) background data. Returns (explainer, feature_names, mode)
    where mode is 'shap' or 'fallback'.
    """
    clf = pipeline.named_steps["clf"]
    prep = pipeline.named_steps["prep"]
    feature_names = _get_feature_names(prep)

    if not _SHAP_AVAILABLE:
        return None, feature_names, "fallback"

    try:
        X_bg_transformed = prep.transform(X_background.sample(
            min(100, len(X_background)), random_state=42
        ))
        # TreeExplainer for tree models (fast, exact); generic Explainer otherwise.
        tree_based = clf.__class__.__name__ in (
            "RandomForestClassifier", "DecisionTreeClassifier", "XGBClassifier"
        )
        if tree_based:
            explainer = shap.TreeExplainer(clf)
        else:
            explainer = shap.Explainer(clf, X_bg_transformed, feature_names=feature_names)
        return explainer, feature_names, "shap"
    except Exception:
        # Any SHAP incompatibility (version mismatch, unsupported model, etc.)
        # falls back gracefully rather than crashing the app.
        return None, feature_names, "fallback"


def global_feature_importance(pipeline, explainer, feature_names, X_sample: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return a DataFrame of feature -> mean |impact| on the attrition prediction, sorted desc."""
    clf = pipeline.named_steps["clf"]
    prep = pipeline.named_steps["prep"]

    if mode == "shap" and explainer is not None:
        X_t = prep.transform(X_sample)
        shap_values = explainer.shap_values(X_t)
        if isinstance(shap_values, list):  # binary-classification list output
            shap_values = shap_values[1]
        if shap_values.ndim == 3:  # (n_samples, n_features, n_classes)
            shap_values = shap_values[:, :, 1]
        importance = np.abs(shap_values).mean(axis=0)
    else:
        if hasattr(clf, "feature_importances_"):
            importance = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importance = np.abs(clf.coef_[0])
        else:
            raise RuntimeError("Model provides neither SHAP support nor importances/coefficients.")

    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def explain_single_prediction(pipeline, explainer, feature_names, X_row: pd.DataFrame, mode: str, top_n: int = 4) -> list[dict]:
    """
    Return the top_n factors driving ONE employee's prediction, each with the
    direction of effect (increases / decreases risk). X_row must be a single-row
    raw (untransformed) DataFrame.
    """
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    X_t = prep.transform(X_row)

    if mode == "shap" and explainer is not None:
        shap_values = explainer.shap_values(X_t)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        contributions = shap_values[0]
    else:
        # Fallback: approximate a per-instance "contribution" as
        # (standardized feature value) * (global importance / coefficient),
        # clearly documented as an approximation, not a true Shapley value.
        if hasattr(clf, "coef_"):
            weights = clf.coef_[0]
        elif hasattr(clf, "feature_importances_"):
            weights = clf.feature_importances_
        else:
            raise RuntimeError("No explainability method available for this model.")
        x_vals = X_t.toarray()[0] if hasattr(X_t, "toarray") else np.asarray(X_t)[0]
        contributions = weights * x_vals

    ranked_idx = np.argsort(-np.abs(contributions))[:top_n]
    factors = []
    for idx in ranked_idx:
        factors.append({
            "feature": feature_names[idx],
            "effect": "increases risk" if contributions[idx] > 0 else "decreases risk",
            "magnitude": float(abs(contributions[idx])),
        })
    return factors


def humanize_feature_name(raw_name: str, X_row: pd.DataFrame | None = None, categorical_features: list[str] | None = None) -> str:
    """
    Turn a ColumnTransformer output name into a readable, employee-specific label.
    For one-hot encoded categoricals (e.g. 'cat__OverTime_Yes'), if the original
    raw value is available in X_row, show 'OverTime: Yes' (the employee's actual
    category) rather than the ambiguous dummy-variable name.
    """
    name = raw_name.split("__", 1)[-1]

    if categorical_features and X_row is not None:
        for col in categorical_features:
            if name.startswith(col + "_"):
                actual_value = X_row.iloc[0][col]
                return f"{col}: {actual_value}"

    return name.replace("_", " ")
