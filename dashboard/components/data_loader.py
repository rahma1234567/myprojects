import pandas as pd
import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from inference import run_pipeline, validate_columns, REQUIRED_COLUMNS
from logger import log_info, log_error


@st.cache_data
def load_trained_data(model):
    """Load pre-computed results for demo mode."""
    raw_data = pd.read_csv("data/bank_transactions_data_2.csv")
    if model == "Isolation Forest":
        results = pd.read_csv("models/training/isolation_forest_results.csv")
        score = "anomaly_score"
    else:
        results = pd.read_csv("models/training/autoencoder_results.csv")
        score = "reconstruction_error"
    merged = results.merge(raw_data, left_on="original_index", right_index=True, suffixes=("_scaled", ""))
    return merged, score, raw_data


def handle_upload_mode(model_choice):
    """Handle file upload + run pipeline. Returns merged, score, raw_data or stops execution."""
    uploaded_file = st.sidebar.file_uploader("Upload a new CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV file that matches the required columns mentioned below.")
        with st.expander("Required columns"):
            st.write(REQUIRED_COLUMNS)
        st.stop()

    data = pd.read_csv(uploaded_file)
    log_info(f"File uploaded: {uploaded_file.name} ({len(data)} rows)")

    missing = validate_columns(data)
    if missing:
        log_error(f"Upload failed - missing columns: {missing}")
        st.error(f"The required columns are missing from the file: {missing}")
        st.stop()

    with st.spinner("Running pipeline..."):
        merged = run_pipeline(data, model_choice)
        raw_data = data.copy()
        score = "anomaly_score" if model_choice == "Isolation Forest" else "reconstruction_error"

    log_info(f"Pipeline executed: {len(data)} transactions processed using {model_choice}")
    st.success(f"Processed {len(data)} transactions using {model_choice}")
    return merged, score, raw_data


def calculate_risk(merged, score, model_choice):
    """Calculate risk percentage and risk level."""
    if model_choice == "Isolation Forest":
        min_score = merged[score].min()
        max_score = merged[score].max()
        merged["risk_pct"] = ((max_score - merged[score]) / (max_score - min_score) * 100).round(1)
    else:
        min_err = merged[score].min()
        max_err = merged[score].max()
        merged["risk_pct"] = ((merged[score] - min_err) / (max_err - min_err) * 100).round(1)

    def risk_level(pct):
        if pct >= 80:
            return "High"
        elif pct >= 50:
            return "Medium"
        else:
            return "Low"

    merged["risk_level"] = merged["risk_pct"].apply(risk_level)
    return merged