from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

RAW_FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

TARGET = "isFraud"


def load_data(path: str, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows)
    return df[RAW_FEATURES + [TARGET]].copy()


def fit_preprocessor(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], SimpleImputer, MinMaxScaler]:
    """
    Fit preprocessing (one-hot + missing handling + normalization) and transform X.
    Returns the transformed feature matrix and the fitted transformers so they can be reused at inference time.
    """
    local = df.copy()
    local["type"] = local["type"].astype("category")
    local = pd.get_dummies(local, columns=["type"], drop_first=False)

    y = local[TARGET].astype(int)
    X = local.drop(columns=[TARGET])
    feature_columns = X.columns.tolist()

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
    return X_scaled, y, feature_columns, imputer, scaler


def transform_preprocessor(
    df: pd.DataFrame, imputer: SimpleImputer, scaler: MinMaxScaler, feature_columns: list[str]
) -> pd.DataFrame:
    """
    Apply preprocessing using fitted transformers from training.
    """
    local = df.copy()
    if "type" in local.columns:
        local["type"] = local["type"].fillna("PAYMENT").astype("category")
    else:  # pragma: no cover
        raise ValueError("Expected column 'type' in input data.")

    local = pd.get_dummies(local, columns=["type"], drop_first=False)
    if TARGET in local.columns:
        local = local.drop(columns=[TARGET])

    for col in feature_columns:
        if col not in local.columns:
            local[col] = 0.0
    X = local[feature_columns]

    X_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns)
    X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=X_imputed.columns)
    return X_scaled


def etl_preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    local = df.copy()
    local["type"] = local["type"].astype("category")
    local = pd.get_dummies(local, columns=["type"], drop_first=False)

    y = local[TARGET].astype(int)
    X = local.drop(columns=[TARGET])

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
    return X_scaled, y

