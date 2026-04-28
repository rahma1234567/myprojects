import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from retrain.retraining_pipeline import retrain_models, commit_new_models, discard_new_models
from logger import log_info, log_error


REQUIRED_COLUMNS = [
    "TransactionID", "AccountID", "TransactionAmount", "TransactionDate",
    "TransactionType", "Location", "DeviceID", "IP Address", "MerchantID",
    "Channel", "CustomerAge", "CustomerOccupation", "TransactionDuration",
    "LoginAttempts", "AccountBalance", "PreviousTransactionDate"
]


def show_retraining(username):
    """Display the Retraining tab. Admin-only."""
    st.subheader("Model Retraining")
    st.write("Upload new transaction data to retrain the models with the combined dataset.")

    # Step 1: File upload
    uploaded_file = st.file_uploader("Upload new training data CSV", type=["csv"], key="retrain_upload")

    if uploaded_file is None:
        st.info("Upload a new CSV file to begin retraining. The file must match the required schema.")
        with st.expander("Required columns"):
            st.write(REQUIRED_COLUMNS)
        return

    # Validate
    new_data = pd.read_csv(uploaded_file)
    missing = [col for col in REQUIRED_COLUMNS if col not in new_data.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    st.success(f"File uploaded: {uploaded_file.name} ({len(new_data)} new rows)")

    # Step 2: Trigger retraining
    if st.button("Retrain Models", type="primary"):
        with st.spinner("Retraining in progress... This may take a minute."):
            try:
                comparison = retrain_models(new_data)
                st.session_state["retrain_comparison"] = comparison
                st.session_state["retrain_complete"] = True
                log_info(f"[{username}] triggered retraining successfully")
            except Exception as e:
                log_error(f"Retraining failed: {str(e)}")
                st.error(f"Retraining failed: {str(e)}")
                return

    # Step 3: Show comparison results
    if st.session_state.get("retrain_complete"):
        comp = st.session_state["retrain_comparison"]
        st.markdown("---")
        st.subheader("Comparison: Old vs New Models")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Set Size", comp["test_size"])
        with col2:
            st.metric("Training Samples", comp["training_samples"])
        with col3:
            st.metric("Total Records Used", comp["test_size"] + comp["training_samples"])

        st.markdown("### Isolation Forest")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Old Model - Flagged", f"{comp['old_iso_flagged']} ({comp['old_iso_pct']}%)")
        with c2:
            st.metric("New Model - Flagged", f"{comp['new_iso_flagged']} ({comp['new_iso_pct']}%)",
                      delta=f"{comp['new_iso_flagged'] - comp['old_iso_flagged']}")

        st.markdown("### Autoencoder")
        st.metric("New Model - Flagged", f"{comp['new_ae_flagged']} ({comp['new_ae_pct']}%)")
        st.caption("Note: Autoencoder uses dynamic threshold so direct old/new comparison is not applicable.")

        st.markdown("---")
        st.subheader("Decision")
        st.write("Review the comparison above. Save the new models to deploy them, or discard to keep the current ones.")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Save New Models", type="primary"):
                backup_dir = commit_new_models()
                log_info(f"[{username}] saved new models. Backup: {backup_dir}")
                st.success(f"New models deployed! Old models backed up to: {backup_dir}")
                st.session_state["retrain_complete"] = False
                st.rerun()
        with col_b:
            if st.button("Discard New Models"):
                discard_new_models()
                log_info(f"[{username}] discarded new models")
                st.warning("New models discarded. Current models retained.")
                st.session_state["retrain_complete"] = False
                st.rerun()