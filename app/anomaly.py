from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def train_anomaly_detector(X):
    model = IsolationForest(
        n_estimators=150,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def get_anomaly_score(model: IsolationForest, X) -> np.ndarray:
    # More positive means normal; invert to turn into anomaly intensity.
    score = -model.decision_function(X)
    return np.clip(score, 0, None)

