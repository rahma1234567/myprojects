import pandas as pd
import os

#Load raw data
data = pd.read_csv("data/bank_transactions_data_2.csv")
print(f"Loaded {len(data)} transactions")

#Parsing the dates to extract extract hour
data["TransactionDate"] = pd.to_datetime(data["TransactionDate"], dayfirst=True, errors="coerce")
data["PreviousTransactionDate"] = pd.to_datetime(data["PreviousTransactionDate"], dayfirst=True, errors="coerce")

#Columns used to create rules
data["transaction_hour"] = data["TransactionDate"].dt.hour
data["days_since_last"] = (data["TransactionDate"] - data["PreviousTransactionDate"]).dt.days.abs().fillna(999)

#Compute the thresholds
high_amount_threshold = data["TransactionAmount"].quantile(0.95)
print(f"High amount threshold (95th percentile): £{high_amount_threshold:.2f}")

#Create fraud rules and returns true/false valules 
rule_high_amount = data["TransactionAmount"] > high_amount_threshold
rule_exceeds_balance = data["TransactionAmount"] > data["AccountBalance"]
rule_high_login = data["LoginAttempts"] > 3
rule_night = data["transaction_hour"] <= 5
rule_high_ratio = (data["TransactionAmount"] / (data["AccountBalance"] + 1)) > 0.7
rule_quick_high = (data["days_since_last"] == 0) & (data["TransactionAmount"] > high_amount_threshold)

#sum of all the rules that were triggered per transaction
rules_triggered = (
    rule_high_amount.astype(int) +
    rule_exceeds_balance.astype(int) +
    rule_high_login.astype(int) +
    rule_night.astype(int) +
    rule_high_ratio.astype(int) +
    rule_quick_high.astype(int)
)

#IF 2+ rules triggered, labels it fraud
data["is_fraud_label"] = (rules_triggered >= 2).astype(int)

#Printing all the summary
total = len(data)
fraud_count = data["is_fraud_label"].sum()
print("\n--- LABELLING SUMMARY ---")
print(f"Total transactions: {total}")
print(f"Labelled as fraud: {fraud_count}")
print(f"Fraud percentage: {fraud_count/total*100:.2f}%")

print("\nRule trigger frequency:")
print(f"  High amount:       {rule_high_amount.sum()}")
print(f"  Exceeds balance:   {rule_exceeds_balance.sum()}")
print(f"  High login:        {rule_high_login.sum()}")
print(f"  Night-time:        {rule_night.sum()}")
print(f"  High ratio:        {rule_high_ratio.sum()}")
print(f"  Quick high amount: {rule_quick_high.sum()}")

#Drop the columns extracted from data and save
data = data.drop(columns=["transaction_hour", "days_since_last"])

# Convert dates back to original string format to keep consistency
data["TransactionDate"] = data["TransactionDate"].dt.strftime("%d/%m/%Y %H:%M")
data["PreviousTransactionDate"] = data["PreviousTransactionDate"].dt.strftime("%d/%m/%Y %H:%M")

data.to_csv("evaluation/bank_transactions_labelled.csv", index=False)

print("\nLabelled dataset saved to: evaluation/bank_transactions_labelled.csv")