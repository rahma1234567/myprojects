import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import os

#load preprocessed data

X_train = pd.read_csv("preprocessing/train_set.csv")
X_test = pd.read_csv("preprocessing/test_set.csv")

#Feature Engineering Start:

#1. Ratio of TA to AB
X_train["amount_balance_ratio"] = X_train["TransactionAmount"] / (X_train["AccountBalance"] + 1)
X_test["amount_balance_ratio"] = X_test["TransactionAmount"] / (X_test["AccountBalance"] + 1)

#2. Transactions during midnight
X_train ["night_transaction"] = (X_train["transaction_hour"] <= 5).astype(int)
X_test ["night_transaction"] = (X_test["transaction_hour"] <= 5).astype(int)

#3. Login attempts > 3 

X_train ["high_login_attempts"] = (X_train["LoginAttempts"] > 3).astype(int)
X_test ["high_login_attempts"] = (X_test["LoginAttempts"] > 3).astype(int)

#4. Speed of Transaction
X_train ["quick_transactions"] = X_train["TransactionAmount"] / (X_train["TransactionDuration"] + 1)
X_test ["quick_transactions"] = X_test["TransactionAmount"] / (X_test["TransactionDuration"] + 1)


#5. Same day transactions
X_train ["repeat_transactions"] = (X_train["days_since_last_transaction"] == 0).astype(int)
X_test ["repeat_transactions"] = (X_test["days_since_last_transaction"] == 0).astype(int)

print("Features after feature engineering:", X_train.shape[1])
print ("New features added: amount_balance_ratio, night_transaction, high_login_attempts, quick_transactions, repeat_transactions")


#Now all those numeric columns will be scaled
robust_cols = ["TransactionAmount", "LoginAttempts", "amount_balance_ratio", "quick_transactions"]
robust_scaler = RobustScaler()

standard_cols = ["CustomerAge", "TransactionDuration","AccountBalance",
                 "transaction_hour", "transaction_day", "transaction_month", 
                 "days_since_last_transaction"]
standard_scaler = StandardScaler()

X_train[robust_cols]   = robust_scaler.fit_transform(X_train[robust_cols])
X_test[robust_cols]    = robust_scaler.transform(X_test[robust_cols])
 
X_train[standard_cols] = standard_scaler.fit_transform(X_train[standard_cols])
X_test[standard_cols]  = standard_scaler.transform(X_test[standard_cols])

#Save these scalers which which will be used later for retraining
os.makedirs("saved_scalers", exist_ok=True)
joblib.dump(robust_scaler, "saved_scalers/robust_scl.pkl")
joblib.dump(standard_scaler, "saved_scalers/standard_scl.pkl")

#Save final preprocessed + feature engineered data - ready for training models
X_train.to_csv ("feature_engineering/X_train_final.csv", index=False)
X_test.to_csv ("feature_engineering/X_test_final.csv", index=False)

print("FEATURE ENGINEERING COMPLETED")
print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("\nFirst 3 rows:\n", X_train.head(3).to_string())

#next folder - models/training