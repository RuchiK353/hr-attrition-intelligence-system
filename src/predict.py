"""
Single-employee prediction utilities used by the Streamlit app.
Loads the saved pipeline once and exposes a simple predict_employee() function
that returns probability, risk label, and dynamically computed top factors.
"""

from __future__ import annotations

import joblib
import pandas as pd

from . import config, explain, features


def load_model_bundle(path=None):
    path = path or config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def prepare_input_row(raw_input: dict, feature_columns: list[str]) -> pd.DataFrame:
    """
    Turn a dict of raw employee attributes (as collected from the Streamlit form)
    into a single-row DataFrame with engineered features, matching training schema.
    """
    row = pd.DataFrame([raw_input])
    row = features.engineer_features(row)
    missing = [c for c in feature_columns if c not in row.columns]
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")
    return row[feature_columns]


def predict_employee(raw_input: dict, bundle: dict, explainer_cache: dict, background_df: pd.DataFrame) -> dict:
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    X_row = prepare_input_row(raw_input, feature_columns)
    proba = float(pipeline.predict_proba(X_row)[0, 1])
    risk = config.probability_to_risk(proba)

    if "explainer" not in explainer_cache:
        explainer, feature_names, mode = explain.build_explainer(pipeline, background_df[feature_columns])
        explainer_cache["explainer"] = explainer
        explainer_cache["feature_names"] = feature_names
        explainer_cache["mode"] = mode

    factors = explain.explain_single_prediction(
        pipeline, explainer_cache["explainer"], explainer_cache["feature_names"],
        X_row, explainer_cache["mode"], top_n=4,
    )
    categorical_features = bundle.get("categorical_features", [])
    for f in factors:
        f["feature"] = explain.humanize_feature_name(f["feature"], X_row, categorical_features)

    return {
        "probability": proba,
        "risk": risk,
        "top_factors": factors,
        "explain_mode": explainer_cache["mode"],
    }
