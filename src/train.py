"""
End-to-end training script:
  baseline -> 4 models -> stratified CV -> tuning (best 2) -> comparison table
  -> final model selection -> save pipeline (preprocessing + model) to disk.

Run: python -m src.train
"""

from __future__ import annotations

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from . import config, features, preprocessing

warnings.filterwarnings("ignore")


def load_and_prepare():
    raw = preprocessing.load_raw_data()
    clean = preprocessing.clean_data(raw)
    fe = features.engineer_features(clean)

    numerical = config.NUMERICAL_FEATURES + config.ENGINEERED_FEATURES
    categorical = config.CATEGORICAL_FEATURES

    X = fe[numerical + categorical]
    y = fe[config.TARGET_COL]
    return X, y, numerical, categorical


def score_row(model_name: str, y_true, y_pred, y_proba) -> dict:
    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_proba), 4),
    }


def main():
    X, y, numerical, categorical = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train attrition rate: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

    preprocessor = preprocessing.build_preprocessor(numerical, categorical)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    results = []
    fitted_pipelines = {}

    # --- Baseline: majority-class dummy classifier ------------------------
    baseline = Pipeline([("prep", preprocessor), ("clf", DummyClassifier(strategy="most_frequent"))])
    baseline.fit(X_train, y_train)
    proba = baseline.predict_proba(X_test)[:, 1]
    results.append(score_row("Baseline (Majority Class)", y_test, baseline.predict(X_test), proba))

    # --- Candidate models ---------------------------------------------------
    # class_weight='balanced' chosen over SMOTE: the imbalance (~16% positive)
    # is moderate, not extreme, and class_weight avoids inventing synthetic
    # employee records -- see README for the justification.
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=config.RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=config.RANDOM_SEED, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=config.RANDOM_SEED, eval_metric="logloss", n_jobs=-1,
        ),
    }

    cv_summary = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        t0 = time.time()
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        cv_summary[name] = {"mean_f1": round(cv_scores.mean(), 4), "std_f1": round(cv_scores.std(), 4)}

        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        results.append(score_row(name, y_test, pipe.predict(X_test), proba))
        fitted_pipelines[name] = pipe
        print(f"{name}: CV F1={cv_scores.mean():.4f} (+/-{cv_scores.std():.4f}), "
              f"trained in {time.time()-t0:.1f}s")

    print("\nCross-validation summary (5-fold stratified, scoring=F1):")
    for k, v in cv_summary.items():
        print(f"  {k}: mean={v['mean_f1']}, std={v['std_f1']}")

    # --- Hyperparameter tuning: tune only the 2 strongest models by CV F1 --
    ranked = sorted(cv_summary.items(), key=lambda kv: kv[1]["mean_f1"], reverse=True)
    top2 = [name for name, _ in ranked[:2]]
    print(f"\nTuning top 2 models by CV F1: {top2}")

    param_grids = {
        "Random Forest": {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 6, 10, 16],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2"],
        },
        "XGBoost": {
            "clf__n_estimators": [150, 300, 500],
            "clf__max_depth": [3, 4, 5, 6],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.7, 0.85, 1.0],
        },
        "Logistic Regression": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l2"],
        },
        "Decision Tree": {
            "clf__max_depth": [3, 5, 8, 12, None],
            "clf__min_samples_leaf": [1, 2, 4, 8],
        },
    }

    tuned_results = {}
    for name in top2:
        grid = param_grids[name]
        pipe = fitted_pipelines[name]
        search = RandomizedSearchCV(
            pipe, param_distributions=grid, n_iter=15, scoring="f1", cv=cv,
            random_state=config.RANDOM_SEED, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        proba = best_pipe.predict_proba(X_test)[:, 1]
        row = score_row(f"{name} (Tuned)", y_test, best_pipe.predict(X_test), proba)
        results.append(row)
        tuned_results[name] = {
            "best_params": search.best_params_,
            "best_cv_f1": round(search.best_score_, 4),
            "test_row": row,
            "pipeline": best_pipe,
        }
        print(f"{name} tuned -> best CV F1={search.best_score_:.4f}, params={search.best_params_}")

    # --- Final model selection: CV only; holdout test is used once for final evaluation ---
    # IMPORTANT: never choose the final model using the holdout test set.
    final_name = max(tuned_results, key=lambda n: tuned_results[n]["best_cv_f1"])
    final_pipeline = tuned_results[final_name]["pipeline"]
    print(f"\nFinal model selected: {final_name} (Tuned) -- highest mean CV F1. "
          "The held-out test set is reserved for final evaluation.")

    # --- Persist artifacts ---------------------------------------------------
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "pipeline": final_pipeline,
            "model_name": final_name,
            "numerical_features": numerical,
            "categorical_features": categorical,
            "feature_columns": list(X.columns),
        },
        config.MODEL_PATH,
    )
    print(f"Saved final model pipeline to {config.MODEL_PATH}")

    config.REPORTS_DIR.mkdir(exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.RESULTS_PATH, index=False)
    print(f"Saved model comparison table to {config.RESULTS_PATH}")
    print("\n" + results_df.to_string(index=False))

    # Save a small json of extra artifacts the app / notebooks might want
    joblib.dump(
        {"cv_summary": cv_summary, "tuned": {k: {kk: vv for kk, vv in v.items() if kk != "pipeline"}
                                              for k, v in tuned_results.items()},
         "final_model": final_name, "X_test": X_test, "y_test": y_test},
        config.REPORTS_DIR / "eval_artifacts.pkl",
    )

    return results_df, final_name


if __name__ == "__main__":
    main()
