import streamlit as st
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from components.authentication.login import authenticate
from logger import log_info, log_anomaly
from components.data_loader import load_trained_data, handle_upload_mode, calculate_risk
from components.metrics import show_metrics
from components.anomaly_info import show_anomaly_details
from components.timeline import show_timeline
from components.anomaly_tbl import show_anomalies_table
from components.log_view import show_logs_viewer
from components.retrain import show_retraining

# Page setup
st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")

#Login Page
authenticator, name, username, role = authenticate()

# Show dashboard after login
st.title("Anomaly Detection System Dashboard")
st.sidebar.write(f"Logged in as: **{name}** ({role})")
authenticator.logout("Logout", "sidebar")
st.markdown("---")

# Sidebar
st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode", ["Trained data", "Upload"])
model_choice = st.sidebar.selectbox("Select Model", ["Isolation Forest", "Autoencoder"])

log_info(f"[{username}] selected mode: {mode}")
log_info(f"[{username}] selected model: {model_choice}")

#Load the data
if mode == "Trained data":
    merged, score, raw_data = load_trained_data(model_choice)
    st.success(f"This section shows the original trained data results from {model_choice}")
else:
    merged, score, raw_data = handle_upload_mode(model_choice)

# Calculate risk
merged = calculate_risk(merged, score, model_choice)

# Log flagged anomalies 
if "anomalies_logged" not in st.session_state:
    flagged = merged[merged["is_fraud"] == 1]
    for _, row in flagged.iterrows():
        log_anomaly(
            transaction_id=row["TransactionID"],
            customer_id=row["AccountID"],
            risk_pct=row["risk_pct"],
            model=model_choice,
            location=row["Location"],
            amount=row["TransactionAmount"]
        )
    st.session_state["anomalies_logged"] = True


# Viewing tabs - role-based access 
if role == "Admin":
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Investigation Logs", "Model Retraining"])
else:
    tab1, = st.tabs(["Dashboard"])
    tab2 = None
    tab3 = None

with tab1:
    show_metrics(merged)
    st.markdown("---")
    left_col, right_col = st.columns([1, 1])
    with left_col:
        show_anomaly_details(merged, raw_data, model_choice)
    with right_col:
        show_timeline(merged)
    show_anomalies_table(merged)

# Admin-only tabs
if role == "Admin":
    with tab2:
        show_logs_viewer()
    with tab3:
        show_retraining(username)