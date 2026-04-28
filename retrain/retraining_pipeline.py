import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os
import shutil
from datetime import datetime
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from logger import log_info, log_error


# Autoencoder architecture which matches the original architecture (copy pasted from autoencoder.py file)
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


def preprocess_data(df):
    """Applies the same preprocessing as the original one."""
    df = df.copy()
    drop_cols = ["TransactionID", "AccountID", "DeviceID", "IP Address", "MerchantID"]
    df.drop(columns=drop_cols, inplace=True)

    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], dayfirst=True, errors="coerce")
    df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"], dayfirst=True, errors="coerce")

    df["transaction_hour"] = df["TransactionDate"].dt.hour
    df["transaction_day"] = df["TransactionDate"].dt.day_of_week
    df["transaction_month"] = df["TransactionDate"].dt.month
    df["days_since_last_transaction"] = (
        df["TransactionDate"] - df["PreviousTransactionDate"]
    ).dt.days.abs().fillna(0)

    df.drop(columns=["TransactionDate", "PreviousTransactionDate"], inplace=True)

    df["TransactionType"] = (df["TransactionType"] == "Debit").astype(int)
    df = pd.get_dummies(df, columns=["Channel", "CustomerOccupation"], drop_first=False)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def feature_engineer(df):
    """Applies the same feature engineering as the original one."""
    df = df.copy()
    df["amount_balance_ratio"] = df["TransactionAmount"] / (df["AccountBalance"] + 1)
    df["night_transaction"] = (df["transaction_hour"] <= 5).astype(int)
    df["high_login_attempts"] = (df["LoginAttempts"] > 3).astype(int)
    df["quick_transactions"] = df["TransactionAmount"] / (df["TransactionDuration"] + 1)
    df["repeat_transactions"] = (df["days_since_last_transaction"] == 0).astype(int)
    return df


def retrain_models(new_data_df):
    """Full retraining pipeline. Returns comparison dict."""
    log_info("Retraining started")

    #Combines old +  new data 
    old_data = pd.read_csv("data/bank_transactions_data_2.csv")
    combined = pd.concat([old_data, new_data_df], ignore_index=True)
    log_info(f"Combined dataset: {len(old_data)} old + {len(new_data_df)} new = {len(combined)} total")

    #Preprocess both 
    processed = preprocess_data(combined)

    #Train/test split
    X_train, X_test = train_test_split(processed, test_size=0.2, random_state=42)

    #Location frequency encoding
    location_freq = X_train["Location"].value_counts(normalize=True)
    X_train["Location"] = X_train["Location"].map(location_freq)
    X_test["Location"] = X_test["Location"].map(location_freq).fillna(0)

    #Feature engineering
    X_train = feature_engineer(X_train)
    X_test = feature_engineer(X_test)

    #Scaling
    robust_cols = ["TransactionAmount", "LoginAttempts", "amount_balance_ratio", "quick_transactions"]
    standard_cols = ["CustomerAge", "TransactionDuration", "AccountBalance",
                     "transaction_hour", "transaction_day", "transaction_month",
                     "days_since_last_transaction"]

    robust_scaler = RobustScaler()
    standard_scaler = StandardScaler()

    X_train[robust_cols] = robust_scaler.fit_transform(X_train[robust_cols])
    X_test[robust_cols] = robust_scaler.transform(X_test[robust_cols])
    X_train[standard_cols] = standard_scaler.fit_transform(X_train[standard_cols])
    X_test[standard_cols] = standard_scaler.transform(X_test[standard_cols])

    #Training Isolation Forest
    log_info("Training new Isolation Forest...")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05,
                                  max_samples="auto", random_state=42)
    iso_forest.fit(X_train)
    iso_predictions = iso_forest.predict(X_test)
    iso_flagged = (iso_predictions == -1).sum()

    #Training Autoencoder
    log_info("Training new Autoencoder...")
    input_dim = X_train.shape[1]
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=32, shuffle=True)

    ae_model = Autoencoder(input_dim)
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(50):
        ae_model.train()
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            reconstructed = ae_model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()

    ae_model.eval()
    with torch.no_grad():
        reconstructed = ae_model(X_test_tensor)
        errors = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1).numpy()

    threshold = np.percentile(errors, 95)
    ae_flagged = (errors > threshold).sum()

    #Save new models temporarily
    os.makedirs("retrain/temp", exist_ok=True)
    joblib.dump(iso_forest, "retrain/temp/isolation_forest_new.pkl")
    torch.save(ae_model.state_dict(), "retrain/temp/autoencoder_new.pth")
    joblib.dump(robust_scaler, "retrain/temp/robust_scl_new.pkl")
    joblib.dump(standard_scaler, "retrain/temp/standard_scl_new.pkl")
    joblib.dump(location_freq, "retrain/temp/location_frequency_new.pkl")

    # Compare with old models on the same test set
    old_iso = joblib.load("models/saved_models/isolation_forest.pkl")
    old_iso_predictions = old_iso.predict(X_test)
    old_iso_flagged = (old_iso_predictions == -1).sum()

    comparison = {
        "test_size": len(X_test),
        "old_iso_flagged": int(old_iso_flagged),
        "new_iso_flagged": int(iso_flagged),
        "old_iso_pct": round(old_iso_flagged / len(X_test) * 100, 2),
        "new_iso_pct": round(iso_flagged / len(X_test) * 100, 2),
        "new_ae_flagged": int(ae_flagged),
        "new_ae_pct": round(ae_flagged / len(X_test) * 100, 2),
        "training_samples": len(X_train),
    }

    log_info(f"Retraining completed: {comparison}")
    return comparison


def commit_new_models():
    """Replaces the old models with newly trained ones. Old models are backed up."""
    log_info("Committing new models...")

    # Backup current models with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_dir = f"models/saved_models/backups/{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)

    # Backup
    if os.path.exists("models/saved_models/isolation_forest.pkl"):
        shutil.copy("models/saved_models/isolation_forest.pkl", f"{backup_dir}/isolation_forest.pkl")
    if os.path.exists("models/saved_models/autoencoder.pth"):
        shutil.copy("models/saved_models/autoencoder.pth", f"{backup_dir}/autoencoder.pth")
    if os.path.exists("saved_scalers/robust_scl.pkl"):
        shutil.copy("saved_scalers/robust_scl.pkl", f"{backup_dir}/robust_scl.pkl")
    if os.path.exists("saved_scalers/standard_scl.pkl"):
        shutil.copy("saved_scalers/standard_scl.pkl", f"{backup_dir}/standard_scl.pkl")
    if os.path.exists("saved_scalers/location_frequency.pkl"):
        shutil.copy("saved_scalers/location_frequency.pkl", f"{backup_dir}/location_frequency.pkl")
        
#Replaces with new models
    shutil.copy("retrain/temp/isolation_forest_new.pkl", "models/saved_models/isolation_forest.pkl")
    shutil.copy("retrain/temp/autoencoder_new.pth", "models/saved_models/autoencoder.pth")
    shutil.copy("retrain/temp/robust_scl_new.pkl", "saved_scalers/robust_scl.pkl")
    shutil.copy("retrain/temp/standard_scl_new.pkl", "saved_scalers/standard_scl.pkl")
    shutil.copy("retrain/temp/location_frequency_new.pkl", "saved_scalers/location_frequency.pkl")

    log_info(f"New models deployed. Old models backed up to {backup_dir}")
    return backup_dir


def discard_new_models():
    """Delete temporary models, keep current ones."""
    temp_dir = "retrain/temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    log_info("New models removed.")