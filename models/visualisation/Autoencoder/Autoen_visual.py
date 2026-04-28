import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load results
X_test = pd.read_csv("models/training/autoencoder_results.csv")
X_test["is_fraud"] = X_test["is_fraud"].astype(str)

os.makedirs("models/visualisation/Autoencoder/plots", exist_ok=True)

# 1. High Login Attempts vs Normal Transactions
plt.figure(figsize=(7, 5))
sns.countplot(data=X_test, x="high_login_attempts", hue="is_fraud",
              palette={"0": "steelblue", "1": "red"})
plt.title("High Login Attempts vs Normal Transactions - Autoencoder")
plt.xlabel("High Login Attempts (0 = 3 or below, 1 = more than 3)")
plt.ylabel("Count")
plt.legend(title="Is Fraud", labels=["Normal", "Fraud"])
plt.tight_layout()
plt.savefig("models/visualisation/Autoencoder/plots/ae_login_attempts.png")
plt.show()

# 2. Transaction Amount vs Account Balance
plt.figure(figsize=(10, 5))
sns.scatterplot(data=X_test, x="TransactionAmount", y="AccountBalance",
                hue="is_fraud", palette={"0": "steelblue", "1": "red"}, alpha=0.6)
plt.title("Transaction Amount vs Account Balance - Autoencoder")
plt.xlabel("Transaction Amount (scaled)")
plt.ylabel("Account Balance (scaled)")
plt.legend(title="Is Fraud", labels=["Normal", "Fraud"])
plt.tight_layout()
plt.savefig("models/visualisation/Autoencoder/plots/ae_amount_vs_balance.png")
plt.show()

# 3. Same Day Transactions
plt.figure(figsize=(7, 5))
sns.countplot(data=X_test, x="repeat_transactions", hue="is_fraud",
              palette={"0": "steelblue", "1": "red"})
plt.title("Same Day Transactions vs Fraud - Autoencoder")
plt.xlabel("Same Day Transaction (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend(title="Is Fraud", labels=["Normal", "Fraud"])
plt.tight_layout()
plt.savefig("models/visualisation/Autoencoder/plots/ae_same_day_transactions.png")
plt.show()

# 4. Days Since Last Transaction
plt.figure(figsize=(8, 5))
sns.boxplot(data=X_test, x="is_fraud", y="days_since_last_transaction",
            hue="is_fraud", palette={"0": "steelblue", "1": "red"}, legend=False)
plt.title("Days Since Last Transaction vs Fraud - Autoencoder")
plt.xlabel("Is Fraud (0 = Normal, 1 = Fraud)")
plt.ylabel("Days Since Last Transaction (scaled)")
plt.tight_layout()
plt.savefig("models/visualisation/Autoencoder/plots/ae_days_since_last.png")
plt.show()

# 5. Reconstruction Error Distribution
plt.figure(figsize=(10, 5))
sns.histplot(data=X_test, x="reconstruction_error", hue="is_fraud",
             bins=50, palette={"0": "steelblue", "1": "red"})
plt.axvline(x=X_test[X_test["is_fraud"] == "1"]["reconstruction_error"].min(),
            color="black", linestyle="--", label="Threshold")
plt.title("Reconstruction Error Distribution - Autoencoder")
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.legend(["Threshold", "Normal", "Fraud"])
plt.tight_layout()
plt.savefig("models/visualisation/Autoencoder/plots/ae_error_distribution.png")
plt.show()

# 6. Reconstruction Error vs Transaction Amount
plt.figure(figsize=(10, 5))
sns.scatterplot(data=X_test, x="TransactionAmount", y="reconstruction_error",
                hue="is_fraud", palette={"0": "steelblue", "1": "red"}, alpha=0.6)
plt.axhline(y=X_test[X_test["is_fraud"] == "1"]["reconstruction_error"].min(),
            color="black", linestyle="--", label="Threshold")
plt.title("Reconstruction Error vs Transaction Amount - Autoencoder")
plt.xlabel("Transaction Amount (scaled)")
plt.ylabel("Reconstruction Error")
plt.legend(title="Is Fraud", labels=["Threshold", "Normal", "Fraud"])
plt.tight_layout()
plt.savefig("models/visualisation/Autoencoder/plots/ae_error_vs_amount.png")
plt.show()

print("All plots saved to models/visualisation/Autoencoder/plots/")