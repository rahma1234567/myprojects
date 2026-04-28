import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

# Load the preprocessed + engineered data
X_train = pd.read_csv("feature_engineering/X_train_final.csv")
X_test = pd.read_csv("feature_engineering/X_test_final.csv")

# Training Starts - define parameters
n_estimators = 100
contamination = 0.05
sample_size = "auto"

isolation_forest = IsolationForest(
    n_estimators=n_estimators,
    contamination=contamination,
    max_samples=sample_size,
    random_state=42
)
isolation_forest.fit(X_train)
print("Training completed")

# Predict on test set
predictions = isolation_forest.predict(X_test)
anomaly_scores = isolation_forest.decision_function(X_test)

X_test = X_test.copy()
X_test["anomaly_score"] = anomaly_scores
X_test["is_fraud"] = (predictions == -1).astype(int)

# Evaluate
total = len(X_test)
flagged = X_test["is_fraud"].sum()
print("TOTAL TRANSACTIONS", total)
print("FLAGGED AS FRAUD", flagged)
print("FRAUD PERCENTAGE: {:.2f}%".format(flagged / total * 100))
print("\nAnomaly Score Stats:")
print(X_test["anomaly_score"].describe())

# Save model and results
os.makedirs("models/training", exist_ok=True)
os.makedirs("models/saved_models", exist_ok=True)
joblib.dump(isolation_forest, "models/saved_models/isolation_forest.pkl")
X_test.to_csv("models/training/isolation_forest_results.csv", index=True, index_label="original_index")
print("\nModel saved and results exported.")
