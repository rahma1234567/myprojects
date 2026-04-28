import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

X_test = pd.read_csv("models/training/isolation_forest_results.csv")
X_test["is_fraud"] = X_test["is_fraud"].astype(str)

#High Login Attempts vs Normal Transactions
plt.figure(figsize=(7, 5))
sns.countplot(data=X_test, x="high_login_attempts", hue="is_fraud",
              palette={"0": "steelblue", "1": "red"})
plt.title("High Login Attempts vs Normal Transactions - Isolation Forest")
plt.xlabel("High Login Attempts (0 = 3 or below, 1 = more than 3)")
plt.ylabel("Count")
plt.legend(title="Is Fraud", labels=["Normal", "Fraud"])
plt.tight_layout()
plt.savefig("models/visualisation/IsolationForest/plots/if_login_attempts.png")
plt.show()

#Transaction amount vs balance
plt.figure(figsize=(10, 5))
sns.scatterplot(data=X_test, x="TransactionAmount", y="AccountBalance",
                hue="is_fraud", palette={"0": "steelblue", "1": "red"}, alpha=0.6)
plt.title("Transaction Amount vs Account Balance - Isolation Forest")
plt.xlabel("Transaction Amount (scaled)")
plt.ylabel("Account Balance (scaled)")
plt.legend(title="Is Fraud", labels=["Normal", "Fraud"])
plt.tight_layout()
plt.savefig("models/visualisation/IsolationForest/plots/if_amount_vs_balance.png")
plt.show()

#Same day transactions
plt.figure(figsize=(7, 5))
sns.countplot(data=X_test, x="repeat_transactions", hue="is_fraud",
              palette={"0": "steelblue", "1": "red"})
plt.title("Same Day Transactions vs Fraud - Isolation Forest")
plt.xlabel("Same Day Transaction (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend(title="Is Fraud", labels=["Normal", "Fraud"])
plt.tight_layout()
plt.savefig("models/visualisation/IsolationForest/plots/if_same_day_transactions.png")
plt.show()

#Days since the last transaction
plt.figure(figsize=(8, 5))
sns.boxplot(data=X_test, x="is_fraud", y="days_since_last_transaction",
            hue="is_fraud", palette={"0": "steelblue", "1": "red"}, legend=False)
plt.title("Days Since Last Transaction vs Fraud - Isolation Forest")
plt.xlabel("Is Fraud (0 = Normal, 1 = Fraud)")
plt.ylabel("Days Since Last Transaction (scaled)")
plt.tight_layout()
plt.savefig("models/visualisation/IsolationForest/plots/if_days_since_last.png")
plt.show()

#Anomaly score distribution

plt.figure(figsize=(10, 5))
sns.histplot(data=X_test, x="anomaly_score", hue="is_fraud",
             bins=50, palette={"0": "steelblue", "1": "red"})
plt.axvline(x=0, color="black", linestyle="--", label="Decision Boundary (0)")
plt.title("Anomaly Score Distribution - Isolation Forest")
plt.xlabel("Anomaly Score (lower = more anomalous)")
plt.ylabel("Count")
plt.legend(["Decision Boundary", "Normal", "Fraud"])
plt.tight_layout()
plt.savefig("models/visualisation/IsolationForest/plots/if_score_distribution.png")
plt.show()

#Anomaly score vs Transaction amount
plt.figure(figsize=(10, 5))
sns.scatterplot(data=X_test, x="TransactionAmount", y="anomaly_score",
                hue="is_fraud", palette={"0": "steelblue", "1": "red"}, alpha=0.6)
plt.axhline(y=0, color="black", linestyle="--", label="Decision Boundary (0)")
plt.title("Anomaly Score vs Transaction Amount - Isolation Forest")
plt.xlabel("Transaction Amount (scaled)")
plt.ylabel("Anomaly Score")
plt.legend(title="Is Fraud", labels=["Decision Boundary", "Normal", "Fraud"])
plt.tight_layout()
plt.savefig("models/visualisation/IsolationForest/plots/if_score_vs_amount.png")
plt.show()

print("All graphs are saved to models/visualisation/IsolationForest/plots/")