"""CLI training script for the destination travel-style classifier.

Run with:
    uv run python -m ml.train

Outputs:
    ml/model.joblib        — best sklearn Pipeline (refit on full training set)
    ml/model_meta.json     — threshold, classes, feature columns, timestamp
    ml/results.csv         — one row appended per experiment

Three models are compared with 5-fold stratified CV:
    1. LogisticRegression   (strong linear baseline)
    2. RandomForestClassifier
    3. LGBMClassifier

LightGBM is then tuned with RandomizedSearchCV(n_iter=30).
The model with the highest macro-F1 mean wins.
"""

from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Constants ─────────────────────────────────────────────────────────────────

ML_DIR = Path(__file__).parent
DATA_PATH = ML_DIR / "destinations.csv"
MODEL_PATH = ML_DIR / "model.joblib"
META_PATH = ML_DIR / "model_meta.json"
RESULTS_PATH = ML_DIR / "results.csv"
CONFUSION_PNG = ML_DIR / "confusion_matrix.png"

TARGET_COL = "travel_style"
# Columns to drop — destination name and country are identifiers that leak
# label information (a model trained with them can't generalise to new places).
DROP_COLS = ["destination", "country"]

NUMERIC_COLS = [
    "avg_temp_peak_season_c",
    "peak_season_length_months",
    "unesco_sites_count",
    "outdoor_activity_score",
    "daily_cost_bucket",
    "coastal_access",
    "visa_difficulty",
    "english_prevalence",
]
CATEGORICAL_COLS = [
    "climate_zone",
    "terrain_primary",
    "accommodation_range",
    "tourism_maturity",
]

CONFIDENCE_THRESHOLD = 0.60
CV_FOLDS = 5
RANDOM_STATE = 42

# ── Reproducibility ───────────────────────────────────────────────────────────

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def _make_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("preprocess", _make_preprocessor()),
        ("clf", clf),
    ])


def _append_results(row: dict) -> None:
    write_header = not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0
    with RESULTS_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp", "model", "params",
                "cv_folds", "accuracy_mean", "accuracy_std",
                "macro_f1_mean", "macro_f1_std",
                "per_class_f1_json", "seed",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _cv_evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, model_name: str, params: dict) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=False,
    )
    acc_mean = float(np.mean(scores["test_accuracy"]))
    acc_std = float(np.std(scores["test_accuracy"]))
    f1_mean = float(np.mean(scores["test_f1_macro"]))
    f1_std = float(np.std(scores["test_f1_macro"]))

    # Per-class F1: refit once on a fresh split to get the report
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    per_class_f1 = {k: round(v["f1-score"], 4) for k, v in report.items() if isinstance(v, dict) and k not in ("accuracy", "macro avg", "weighted avg")}

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"  CV accuracy : {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  CV macro-F1 : {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  Per-class F1: {per_class_f1}")

    row = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "model": model_name,
        "params": json.dumps(params),
        "cv_folds": CV_FOLDS,
        "accuracy_mean": round(acc_mean, 4),
        "accuracy_std": round(acc_std, 4),
        "macro_f1_mean": round(f1_mean, 4),
        "macro_f1_std": round(f1_std, 4),
        "per_class_f1_json": json.dumps(per_class_f1),
        "seed": RANDOM_STATE,
    }
    _append_results(row)
    return {"model_name": model_name, "pipeline": pipeline, "macro_f1_mean": f1_mean, "row": row}


def _save_confusion_matrix(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, classes: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        pipeline.fit(X_tr, y_tr)
        cm = confusion_matrix(y_val, pipeline.predict(X_val), labels=classes)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, ax=ax, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix — Best Model")
        fig.tight_layout()
        fig.savefig(CONFUSION_PNG, dpi=150)
        plt.close(fig)
        print(f"\nConfusion matrix saved to {CONFUSION_PNG}")
    except ImportError:
        print("matplotlib/seaborn not available — skipping confusion matrix plot")


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading data from {DATA_PATH}")
    X, y = _load_data()
    print(f"  {len(X)} rows, {len(X.columns)} features, {y.nunique()} classes")
    print(f"  Class distribution:\n{y.value_counts().to_string()}")

    classes = sorted(y.unique().tolist())

    # ── Baseline models ───────────────────────────────────────────────────────
    candidates = [
        (
            "LogisticRegression",
            _make_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
            {"class_weight": "balanced", "max_iter": 2000},
        ),
        (
            "RandomForest",
            _make_pipeline(RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
            {"class_weight": "balanced", "n_estimators": 100},
        ),
        (
            "LightGBM",
            _make_pipeline(LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)),
            {"class_weight": "balanced"},
        ),
    ]

    results = []
    for name, pipeline, params in candidates:
        res = _cv_evaluate(pipeline, X, y, name, params)
        results.append(res)

    # ── LightGBM hyperparameter tuning ────────────────────────────────────────
    print("\nTuning LightGBM with RandomizedSearchCV(n_iter=30) …")
    param_distributions = {
        "clf__num_leaves":        [15, 31, 63],
        "clf__max_depth":         [-1, 4, 6, 8],
        "clf__learning_rate":     [0.01, 0.05, 0.1],
        "clf__n_estimators":      [100, 200, 400],
        "clf__min_child_samples": [5, 10, 20],
        "clf__reg_alpha":         [0.0, 0.1, 1.0],
    }
    lgbm_base = _make_pipeline(
        LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        lgbm_base, param_distributions,
        n_iter=30, cv=cv, scoring="f1_macro",
        random_state=RANDOM_STATE, n_jobs=-1, refit=True,
    )
    search.fit(X, y)
    best_params = search.best_params_
    best_f1 = float(search.best_score_)
    print(f"  Best params : {best_params}")
    print(f"  Best macro-F1: {best_f1:.4f}")

    tuned_row = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "model": "LightGBM_tuned",
        "params": json.dumps(best_params),
        "cv_folds": CV_FOLDS,
        "accuracy_mean": round(best_f1, 4),  # search scored by f1_macro
        "accuracy_std": 0.0,
        "macro_f1_mean": round(best_f1, 4),
        "macro_f1_std": 0.0,
        "per_class_f1_json": "{}",
        "seed": RANDOM_STATE,
    }
    _append_results(tuned_row)
    results.append({"model_name": "LightGBM_tuned", "pipeline": search.best_estimator_, "macro_f1_mean": best_f1})

    # ── Pick winner ───────────────────────────────────────────────────────────
    winner = max(results, key=lambda r: r["macro_f1_mean"])
    print(f"\nWinner: {winner['model_name']} (macro-F1 = {winner['macro_f1_mean']:.4f})")

    # Refit winner on the FULL dataset — CV is done, we want maximum signal now
    winner_pipeline = winner["pipeline"]
    winner_pipeline.fit(X, y)

    # ── Save model ────────────────────────────────────────────────────────────
    joblib.dump(winner_pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    feature_columns = NUMERIC_COLS + CATEGORICAL_COLS
    meta = {
        "model": winner["model_name"],
        "classes": classes,
        "feature_columns": feature_columns,
        "threshold": CONFIDENCE_THRESHOLD,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "git_sha": _get_git_sha(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved to {META_PATH}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    _save_confusion_matrix(winner_pipeline, X, y, classes)

    print("\nDone. Summary:")
    print(f"  Winner        : {winner['model_name']}")
    print(f"  Macro-F1 mean : {winner['macro_f1_mean']:.4f}")
    print(f"  Threshold     : {CONFIDENCE_THRESHOLD}")
    print(f"  Classes       : {classes}")


if __name__ == "__main__":
    main()
