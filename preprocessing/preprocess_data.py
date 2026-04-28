#This file follows 6 steps to complete preprocessing, 
#feature engineering is separated to a different file

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

#Step 1:  Load dataset from data file
data = pd.read_csv("data/bank_transactions_data_2.csv")

#Step 2: drop unnecessary columns like IDs, high cardinality columns
drop_cols = ["TransactionID", "AccountID", "DeviceID", "IP Address", "MerchantID"]
data.drop(columns=drop_cols, inplace=True)

#Step 3: Parse TransactionDate and previous transaction dates

data["TransactionDate"] = pd.to_datetime(data["TransactionDate"], dayfirst=True, errors="coerce")
data["PreviousTransactionDate"] = pd.to_datetime(data["PreviousTransactionDate"], dayfirst=True, errors="coerce")

data["transaction_hour"] = data["TransactionDate"].dt.hour
data["transaction_day"] = data["TransactionDate"].dt.day_of_week
data["transaction_month"] = data["TransactionDate"].dt.month
data["days_since_last_transaction"] = (
    data["TransactionDate"] - data["PreviousTransactionDate"]
).dt.days.abs()

data.drop(columns=["TransactionDate", "PreviousTransactionDate"], inplace=True)
data["days_since_last_transaction"] = data["days_since_last_transaction"].fillna(0)


#Step 4: Encode categorical columns 
data["TransactionType"] = (data["TransactionType"] == "Debit").astype(int)
data = pd.get_dummies(data, columns = ["Channel", "CustomerOccupation"], drop_first=False)

bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype(int)

#Step 5: Train/Test Split - 80% training 20% testing
X = data.copy()
X_train, X_test = train_test_split (X, test_size=0.2, random_state=42)

location_frequency = X_train["Location"].value_counts(normalize=True)
X_train["Location"] = X_train["Location"].map(location_frequency)
X_test["Location"] = X_test["Location"].map(location_frequency).fillna(0)


#Step 6: Save the preprocessed data for later use 
os.makedirs("preprocessing", exist_ok=True)
X_train.to_csv("preprocessing/train_set.csv", index=False)
X_test.to_csv("preprocessing/test_set.csv", index=False)
joblib.dump(location_frequency, "saved_scalers/location_frequency.pkl")

print("Preprocessing completed")
print("Train shape:", X_train.shape)
print("Test shape: ", X_test.shape)
print("\nMissing values in train set:\n", X_train.isnull().sum()[X_train.isnull().sum()>0])
print("\nfirst 3 rows:\n", X_train.head(3).to_string())
