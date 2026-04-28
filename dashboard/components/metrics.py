import streamlit as st
import pandas as pd


def show_metrics(merged):
    """Display the 4 top metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", len(merged))
    with col2:
        st.metric("Anomalies Detected", int(merged["is_fraud"].sum()))
    with col3:
        avg_score = merged[merged["is_fraud"] == 1]["risk_pct"].mean()
        st.metric("Avg Anomaly Risk", f"{avg_score:.1f}%" if not pd.isna(avg_score) else "0%")
    with col4:
        st.metric("High Risk Count", (merged["risk_level"] == "High").sum())