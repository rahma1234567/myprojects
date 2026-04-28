import streamlit as st
import pandas as pd


def show_anomaly_details(merged, raw_data, model_choice):
    """Display the anomaly detail panel with reasons."""
    st.subheader("Anomaly Detected")
    anomalies = merged[merged["is_fraud"] == 1].sort_values("risk_pct", ascending=False)

    if len(anomalies) == 0:
        st.info("No anomalies detected.")
        return

    selected_index = st.selectbox(
        "Select the anomaly to view",
        anomalies.index,
        format_func=lambda x: f"Transaction {anomalies.loc[x, 'TransactionID']} - Risk {anomalies.loc[x, 'risk_pct']}%"
    )
    anomaly = anomalies.loc[selected_index]

    risks_label = {"High": "🟥", "Medium": "🟨", "Low": "🟩"}
    st.markdown(f"### Risk: {risks_label[anomaly['risk_level']]}  {anomaly['risk_pct']}%")
    st.write(f"**Customer ID:** {anomaly['AccountID']}")
    st.write(f"**Time:** {anomaly['TransactionDate']}")
    st.write(f"**Amount:** £{anomaly['TransactionAmount']:.2f}")
    st.write(f"**Location:** {anomaly['Location']}")
    st.write(f"**Balance:** £{anomaly['AccountBalance']:.2f}")

    _show_reasons(anomaly, raw_data)
    _show_autoencoder_features(anomaly, merged, model_choice)


def _show_reasons(anomaly, raw_data):
    """Build and display reason / explanation list."""
    st.markdown("**Reason / Explanation:**")
    reasons = []

    if anomaly['TransactionAmount'] > raw_data['TransactionAmount'].quantile(0.95):
        reasons.append(f"High transaction amount (£{anomaly['TransactionAmount']:.2f})")

    if anomaly['TransactionAmount'] > anomaly['AccountBalance']:
        reasons.append(f"Transaction amount (£{anomaly['TransactionAmount']:.2f}) exceeds account balance (£{anomaly['AccountBalance']:.2f})")

    ratio = anomaly['TransactionAmount'] / (anomaly['AccountBalance'] + 1)
    if ratio > 0.5 and anomaly['TransactionAmount'] <= anomaly['AccountBalance']:
        reasons.append(f"Transaction uses {ratio*100:.1f}% of account balance")

    if anomaly['LoginAttempts'] > 3:
        reasons.append(f"Suspicious login activity ({int(anomaly['LoginAttempts'])} login attempts)")

    try:
        hour = pd.to_datetime(anomaly['TransactionDate'], dayfirst=True).hour
        if hour <= 5:
            reasons.append(f"Late-night transaction at {hour:02d}:00")
    except:
        pass

    try:
        day_of_week = pd.to_datetime(anomaly['TransactionDate'], dayfirst=True).day_of_week
        if day_of_week >= 5:
            day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day_of_week]
            reasons.append(f"Weekend transaction ({day_name})")
    except:
        pass

    if anomaly['TransactionDuration'] < 10:
        reasons.append(f"Very fast transaction ({int(anomaly['TransactionDuration'])} seconds)")
    elif anomaly['TransactionDuration'] > raw_data['TransactionDuration'].quantile(0.95):
        reasons.append(f"Unusually long transaction duration ({int(anomaly['TransactionDuration'])} seconds)")

    if reasons:
        for r in reasons:
            st.write(r)
    else:
        st.write("Flagged based on known patterns")


def _show_autoencoder_features(anomaly, merged, model_choice):
    """Show top contributing features for autoencoder model."""
    if model_choice != "Autoencoder":
        return

    err_cols = [c for c in merged.columns if c.startswith("err_")]
    if not err_cols:
        return

    feature_errors = anomaly[err_cols].sort_values(ascending=False).head(3)
    st.markdown("**Top contributing features (Autoencoder):**")
    for col, err in feature_errors.items():
        feature_name = col.replace("err_", "")
        st.write(f"`{feature_name}` — error: {err:.4f}")