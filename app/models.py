from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def train_models(X, y):
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=250, max_depth=16, n_jobs=-1, random_state=42, class_weight="balanced"
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.06,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=4,
            random_state=42,
        )

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )
        results[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        trained[name] = model
    best_name = max(results.keys(), key=lambda n: results[n]["f1"])
    return trained, results, best_name


def save_artifacts(model, feature_columns, anomaly_model, artifact_dir: Path):
    artifact_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, artifact_dir / "classifier.joblib")
    joblib.dump(list(feature_columns), artifact_dir / "feature_columns.joblib")
    joblib.dump(anomaly_model, artifact_dir / "anomaly_model.joblib")


def save_artifacts_with_preprocessors(
    model,
    feature_columns,
    artifact_dir: Path,
    imputer,
    scaler,
):
    artifact_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, artifact_dir / "classifier.joblib")
    joblib.dump(list(feature_columns), artifact_dir / "feature_columns.joblib")
    joblib.dump(imputer, artifact_dir / "imputer.joblib")
    joblib.dump(scaler, artifact_dir / "scaler.joblib")
    joblib.dump(imputer, artifact_dir / "imputer.joblib")
    joblib.dump(scaler, artifact_dir / "scaler.joblib")


def load_artifacts(artifact_dir: Path):
    model = joblib.load(artifact_dir / "classifier.joblib")
    feature_columns = joblib.load(artifact_dir / "feature_columns.joblib")
    imputer = joblib.load(artifact_dir / "imputer.joblib")
    scaler = joblib.load(artifact_dir / "scaler.joblib")
    return model, feature_columns, imputer, scaler


def align_features(df, feature_columns):
    out = df.copy()
    for col in feature_columns:
        if col not in out.columns:
            out[col] = 0.0
    out = out[feature_columns]
    return out


def risk_score(probability: np.ndarray) -> np.ndarray:
    return np.clip(probability * 100, 0, 100)

