import streamlit as st
import pandas as pd
import os


def show_logs_viewer():
    """Display the Investigation Logs tab."""
    st.subheader("Investigation Logs")
    st.write("Review system activity and flagged anomalies for investigation purposes.")

    log_type = st.radio("Select log type", ["System Logs", "Anomaly Logs"], horizontal=True)

    log_files = sorted(os.listdir("logs")) if os.path.exists("logs") else []
    available_dates = sorted(set(
        f.split("_")[1].replace(".log", "")
        for f in log_files if f.endswith(".log") and "_" in f
    ), reverse=True)

    if not available_dates:
        st.warning("No logs found yet. Interact with the dashboard to generate logs.")
        return

    selected_date = st.selectbox("Select date", available_dates)

    prefix = "system" if log_type == "System Logs" else "anomalies"
    log_file = f"logs/{prefix}_{selected_date}.log"

    if not os.path.exists(log_file):
        st.info(f"No {log_type.lower()} for {selected_date}.")
        return

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    log_data = []
    for line in lines:
        parts = line.strip().split(" | ", 2)
        if len(parts) == 3:
            log_data.append({
                "Timestamp": parts[0],
                "Level": parts[1],
                "Message": parts[2]
            })

    if not log_data:
        st.info("Log file is empty.")
        return

    logs_df = pd.DataFrame(log_data)

    # Filters
    col_a, col_b = st.columns([1, 2])
    with col_a:
        level_filter = st.multiselect(
            "Filter by level",
            options=logs_df["Level"].unique().tolist(),
            default=logs_df["Level"].unique().tolist()
        )
    with col_b:
        search = st.text_input("Search messages", "")

    filtered = logs_df[logs_df["Level"].isin(level_filter)]
    if search:
        filtered = filtered[filtered["Message"].str.contains(search, case=False, na=False)]

    st.write(f"Showing {len(filtered)} of {len(logs_df)} entries")
    st.dataframe(filtered, width="stretch", height=400)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download logs as CSV",
        data=csv,
        file_name=f"{prefix}_logs_{selected_date}.csv",
        mime="text/csv"
    )