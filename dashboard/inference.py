# dashboard/inference.py

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

# Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


REQUIRED_COLUMNS = [
    "TransactionID", "AccountID", "TransactionAmount", "TransactionDate",
    "TransactionType", "Location", "DeviceID", "IP Address", "MerchantID",
    "Channel", "CustomerAge", "CustomerOccupation", "TransactionDuration",
    "LoginAttempts", "AccountBalance", "PreviousTransactionDate"
]


def validate_columns(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return missing


def preprocess(df):
    df = df.copy()
    raw = df.copy()

    # Drop ID columns
    drop_cols = ["TransactionID", "AccountID", "DeviceID", "IP Address", "MerchantID"]
    df.drop(columns=drop_cols, inplace=True)

    # Parse dates
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], dayfirst=True, errors="coerce")
    df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"], dayfirst=True, errors="coerce")

    df["transaction_hour"] = df["TransactionDate"].dt.hour
    df["transaction_day"] = df["TransactionDate"].dt.day_of_week
    df["transaction_month"] = df["TransactionDate"].dt.month
    df["days_since_last_transaction"] = (
        df["TransactionDate"] - df["PreviousTransactionDate"]
    ).dt.days.abs().fillna(0)

    df.drop(columns=["TransactionDate", "PreviousTransactionDate"], inplace=True)

    # Encode categorical
    df["TransactionType"] = (df["TransactionType"] == "Debit").astype(int)
    df = pd.get_dummies(df, columns=["Channel", "CustomerOccupation"], drop_first=False)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Apply saved location frequency mapping
    location_freq = joblib.load("saved_scalers/location_frequency.pkl")
    df["Location"] = df["Location"].map(location_freq).fillna(0)

    # Ensuring that all expected columns exist in a new file
    expected_cols = [
        "TransactionAmount", "TransactionType", "Location", "CustomerAge",
        "TransactionDuration", "LoginAttempts", "AccountBalance",
        "transaction_hour", "transaction_day", "transaction_month",
        "days_since_last_transaction",
        "Channel_ATM", "Channel_Branch", "Channel_Online",
        "CustomerOccupation_Doctor", "CustomerOccupation_Engineer",
        "CustomerOccupation_Retired", "CustomerOccupation_Student"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    return df, raw


def feature_engineer(df):
    df = df.copy()

    df["amount_balance_ratio"] = df["TransactionAmount"] / (df["AccountBalance"] + 1)
    df["night_transaction"] = (df["transaction_hour"] <= 5).astype(int)
    df["high_login_attempts"] = (df["LoginAttempts"] > 3).astype(int)
    df["quick_transactions"] = df["TransactionAmount"] / (df["TransactionDuration"] + 1)
    df["repeat_transactions"] = (df["days_since_last_transaction"] == 0).astype(int)

    #Applying saved scalers
    robust_scaler = joblib.load("saved_scalers/robust_scl.pkl")
    standard_scaler = joblib.load("saved_scalers/standard_scl.pkl")

    robust_cols = ["TransactionAmount", "LoginAttempts", "amount_balance_ratio", "quick_transactions"]
    standard_cols = ["CustomerAge", "TransactionDuration", "AccountBalance",
                     "transaction_hour", "transaction_day", "transaction_month",
                     "days_since_last_transaction"]

    df[robust_cols] = robust_scaler.transform(df[robust_cols])
    df[standard_cols] = standard_scaler.transform(df[standard_cols])

    return df


def run_isolation_forest(df):
    model = joblib.load("models/saved_models/isolation_forest.pkl")
    predictions = model.predict(df)
    scores = model.decision_function(df)
    df = df.copy()
    df["anomaly_score"] = scores
    df["is_fraud"] = (predictions == -1).astype(int)
    return df


def run_autoencoder(df):
    input_dim = df.shape[1]
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load("models/saved_models/autoencoder.pth"))
    model.eval()

    tensor = torch.tensor(df.values, dtype=torch.float32)
    with torch.no_grad():
        reconstructed = model(tensor)
        per_feature_errors = ((tensor - reconstructed) ** 2).numpy()
        errors = per_feature_errors.mean(axis=1)

    threshold = np.percentile(errors, 95)
    df = df.copy()
    df["reconstruction_error"] = errors
    df["is_fraud"] = (errors > threshold).astype(int)

    feature_names = df.columns[:input_dim].tolist()
    for i, name in enumerate(feature_names):
        df[f"err_{name}"] = per_feature_errors[:, i]

    return df


def run_pipeline(df, model_name):
    processed, raw = preprocess(df)
    engineered = feature_engineer(processed)

    if model_name == "Isolation Forest":
        results = run_isolation_forest(engineered)
    else:
        results = run_autoencoder(engineered)

    results = results.reset_index(drop=True)
    results["original_index"] = results.index
    
    raw = raw.reset_index(drop=True)
    merged = results.merge(raw, left_on="original_index", right_index=True, suffixes=("_scaled", ""))
    return merged