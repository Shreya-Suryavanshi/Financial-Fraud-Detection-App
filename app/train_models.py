from __future__ import annotations

import json
import numpy as np

import pandas as pd

from app.config import settings
from app.etl import fit_preprocessor, load_data
from app.models import save_artifacts_with_preprocessors, train_models


def main():
    # Use a large but bounded training sample for speed/memory balance.
    if settings.data_file.exists():
        df = load_data(str(settings.data_file), nrows=300000)
    else:
        print(
            f"Training dataset not found at {settings.data_file}. "
            "Generating synthetic training data instead."
        )
        df = generate_synthetic_training_data(nrows=30000)
    X, y, feature_columns, imputer, scaler = fit_preprocessor(df)
    trained, metrics, best_name = train_models(X, y)
    save_artifacts_with_preprocessors(
        trained[best_name],
        feature_columns,
        settings.artifacts_dir,
        imputer=imputer,
        scaler=scaler,
    )

    metrics_file = settings.artifacts_dir / "metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "metrics": metrics}, f, indent=2)

    print(f"Best model: {best_name}")
    print(pd.DataFrame(metrics).T)


def generate_synthetic_training_data(nrows: int = 30000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    tx_type = rng.choice(types, size=nrows, p=[0.45, 0.18, 0.18, 0.12, 0.07])
    amount = np.clip(rng.normal(55000, 38000, size=nrows), 100.0, 250000.0)
    oldbalance_org = np.maximum(amount + rng.normal(22000, 26000, size=nrows), 0.0)
    newbalance_orig = np.clip(oldbalance_org - amount * rng.uniform(0.45, 0.95, size=nrows), 0.0, None)
    oldbalance_dest = np.maximum(rng.normal(14000, 9000, size=nrows), 0.0)
    newbalance_dest = np.clip(oldbalance_dest + amount * rng.uniform(0.18, 0.96, size=nrows), 0.0, None)
    step = rng.integers(1, 741, size=nrows)

    df = pd.DataFrame({
        "step": step,
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_orig,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
    })

    base_prob = 0.02 + (amount / 250000.0) * 0.08
    type_risk = np.where(np.isin(tx_type, ["TRANSFER", "CASH_OUT"]), 0.06, 0.0)
    amount_risk = np.where(amount > 100000, 0.09, 0.0)
    balance_risk = np.where(
        (np.isin(tx_type, ["TRANSFER", "CASH_OUT"])) & (oldbalance_org < amount * 1.5),
        0.12,
        0.0,
    )
    dest_risk = np.where((oldbalance_dest < 5000) & (amount > 50000), 0.07, 0.0)
    fraud_prob = np.clip(base_prob + type_risk + amount_risk + balance_risk + dest_risk, 0.01, 0.65)
    df["isFraud"] = (rng.random(nrows) < fraud_prob).astype(int)
    return df


if __name__ == "__main__":
    main()

