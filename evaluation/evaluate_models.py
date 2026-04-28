import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


# Autoencoder must match training architecture
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


def preprocess(df):
    df = df.copy()
    df.drop(columns=["TransactionID", "AccountID", "DeviceID", "IP Address", "MerchantID"], inplace=True)

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
    df = df.copy()
    df["amount_balance_ratio"] = df["TransactionAmount"] / (df["AccountBalance"] + 1)
    df["night_transaction"] = (df["transaction_hour"] <= 5).astype(int)
    df["high_login_attempts"] = (df["LoginAttempts"] > 3).astype(int)
    df["quick_transactions"] = df["TransactionAmount"] / (df["TransactionDuration"] + 1)
    df["repeat_transactions"] = (df["days_since_last_transaction"] == 0).astype(int)
    return df


def print_metrics(model_name, y_true, y_pred):
    """Print all metrics for a model in a formatted way."""
    print(f"\n{'='*60}")
    print(f"  {model_name} EVALUATION")
    print('='*60)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted Normal  Predicted Fraud")
    print(f"  Actual Normal       {cm[0][0]:>6}            {cm[0][1]:>6}")
    print(f"  Actual Fraud        {cm[1][0]:>6}            {cm[1][1]:>6}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


#Evaluation
print("Loading labelled dataset...")
data = pd.read_csv("evaluation/bank_transactions_labelled.csv")

# Separate labels from features
y = data["is_fraud_label"]
X = data.drop(columns=["is_fraud_label"])

print(f"Total: {len(X)} transactions, {y.sum()} labelled as fraud ({y.mean()*100:.2f}%)")

# Preprocess
X_processed = preprocess(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42)

# Apply location frequency from train only
location_freq = X_train["Location"].value_counts(normalize=True)
X_train["Location"] = X_train["Location"].map(location_freq)
X_test["Location"] = X_test["Location"].map(location_freq).fillna(0)

# Feature engineering
X_train = feature_engineer(X_train)
X_test = feature_engineer(X_test)

# Scaling
robust_cols = ["TransactionAmount", "LoginAttempts", "amount_balance_ratio", "quick_transactions"]
standard_cols = ["CustomerAge", "TransactionDuration", "AccountBalance", "transaction_hour", "transaction_day", "transaction_month",
                 "days_since_last_transaction"]

robust_scaler = RobustScaler()
standard_scaler = StandardScaler()

X_train[robust_cols] = robust_scaler.fit_transform(X_train[robust_cols])
X_test[robust_cols] = robust_scaler.transform(X_test[robust_cols])
X_train[standard_cols] = standard_scaler.fit_transform(X_train[standard_cols])
X_test[standard_cols] = standard_scaler.transform(X_test[standard_cols])

print(f"\nTest set: {len(X_test)} transactions, {y_test.sum()} labelled fraud")

#ISOLATION FOREST model
print("\nLoading Isolation Forest...")
iso_forest = joblib.load("models/saved_models/isolation_forest.pkl")
iso_predictions = iso_forest.predict(X_test)
iso_pred_binary = (iso_predictions == -1).astype(int)

iso_metrics = print_metrics("ISOLATION FOREST", y_test, iso_pred_binary)

#AUTOENCODER model
print("\nLoading Autoencoder...")
input_dim = X_test.shape[1]
ae_model = Autoencoder(input_dim)
ae_model.load_state_dict(torch.load("models/saved_models/autoencoder.pth"))
ae_model.eval()

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
with torch.no_grad():
    reconstructed = ae_model(X_test_tensor)
    errors = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1).numpy()

threshold = np.percentile(errors, 95)
ae_pred_binary = (errors > threshold).astype(int)

ae_metrics = print_metrics("AUTOENCODER", y_test, ae_pred_binary)

#Summary of comparison of both models 
print("\n" + "="*60)
print("  MODEL COMPARISON SUMMARY")
print("="*60)
print(f"  {'Metric':<12} {'Isolation Forest':<20} {'Autoencoder':<15}")
print(f"  {'-'*12} {'-'*20} {'-'*15}")
print(f"  {'Accuracy':<12} {iso_metrics['accuracy']*100:<20.2f} {ae_metrics['accuracy']*100:<15.2f}")
print(f"  {'Precision':<12} {iso_metrics['precision']*100:<20.2f} {ae_metrics['precision']*100:<15.2f}")
print(f"  {'Recall':<12} {iso_metrics['recall']*100:<20.2f} {ae_metrics['recall']*100:<15.2f}")
print(f"  {'F1-Score':<12} {iso_metrics['f1']*100:<20.2f} {ae_metrics['f1']*100:<15.2f}")

#Save results
results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Isolation Forest": [
        f"{iso_metrics['accuracy']*100:.2f}%",
        f"{iso_metrics['precision']*100:.2f}%",
        f"{iso_metrics['recall']*100:.2f}%",
        f"{iso_metrics['f1']*100:.2f}%"
    ],
    "Autoencoder": [
        f"{ae_metrics['accuracy']*100:.2f}%",
        f"{ae_metrics['precision']*100:.2f}%",
        f"{ae_metrics['recall']*100:.2f}%",
        f"{ae_metrics['f1']*100:.2f}%"
    ]
})

results_df.to_csv("evaluation/evaluation_results.csv", index=False)
print(f"\nResults saved to: evaluation/evaluation_results.csv")