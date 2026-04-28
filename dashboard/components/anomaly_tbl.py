import streamlit as st


def show_anomalies_table(merged):
    """Display the bottom flagged anomalies table."""
    st.markdown("---")
    st.subheader("All Flagged Anomalies")

    display_cols = ["TransactionID", "AccountID", "TransactionDate",
                    "TransactionAmount", "Location", "AccountBalance",
                    "risk_pct", "risk_level"]
    st.dataframe(
        merged[merged["is_fraud"] == 1][display_cols].sort_values("risk_pct", ascending=False),
        width="stretch"
    )