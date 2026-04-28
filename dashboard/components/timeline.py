import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def show_timeline(merged):
    """Display the anomalies timeline chart by hour."""
    st.subheader("Anomalies Timeline")
    merged["hour"] = pd.to_datetime(merged["TransactionDate"], dayfirst=True, errors="coerce").dt.hour
    hourly = merged[merged["is_fraud"] == 1].groupby("hour").size().reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly.index, hourly.values, color="cyan", marker="o", linewidth=2)
    ax.fill_between(hourly.index, hourly.values, alpha=0.3, color="cyan")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Anomaly Count")
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)