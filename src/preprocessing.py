"""
Data loading, cleaning, and the scikit-learn preprocessing pipeline.

Design notes
------------
* `load_raw_data` / `clean_data` are pure functions (no fitting) so they are
  safe to call on train and test data identically -> no leakage.
* `build_preprocessor` returns an *unfitted* ColumnTransformer. It must only
  ever be `.fit()` on the training split (enforced in train.py).
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw IBM HR Attrition CSV."""
    path = path or config.DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Expected file: WA_Fn-UseC_-HR-Employee-Attrition.csv\n"
            "Source: IBM HR Analytics Employee Attrition & Performance dataset "
            "(publicly available, e.g. via Kaggle or IBM's sample data repos).\n"
            "You may also supply any compatible CSV with the same schema by "
            "placing it at this path."
        )
    return pd.read_csv(path)


def inspect_data(df: pd.DataFrame) -> dict:
    """Return a summary dict used for the 'Data Understanding' step. No mutation."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "constant_columns": [c for c in df.columns if df[c].nunique() <= 1],
        "target_distribution": df[config.TARGET_COL].value_counts().to_dict(),
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
      * drop exact duplicate rows
      * drop constant / identifier / leakage columns
      * binary-encode the target
    Outliers are intentionally NOT removed here -- see EDA notebook 02 for the
    rationale (tree-based models are robust to them, and the extreme values
    are legitimate, not data-entry errors).
    """
    out = df.copy()
    out = out.drop_duplicates()

    cols_to_drop = [c for c in config.LEAKAGE_OR_CONSTANT_COLS if c in out.columns]
    out = out.drop(columns=cols_to_drop)

    out[config.TARGET_COL] = out[config.TARGET_COL].map(config.TARGET_MAP)
    if out[config.TARGET_COL].isnull().any():
        raise ValueError("Unexpected value in Attrition column outside {Yes, No}.")

    return out.reset_index(drop=True)


def build_preprocessor(numerical_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    Build an UNFITTED ColumnTransformer.
    Numerical: median impute (robust to outliers) + standard scale.
    Categorical: most-frequent impute + one-hot encode (ignore unseen categories
    at inference time so the Streamlit app never crashes on an edge-case input).
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return preprocessor
