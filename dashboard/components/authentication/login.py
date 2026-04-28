import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from logger import log_info, log_warning


def authenticate():
    """Display login form. Returns (authenticator, name, username, role) if successful."""
    
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials.yaml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"]
    )

    authenticator.login(location="main")

    auth_status = st.session_state.get("authentication_status")
    name = st.session_state.get("name")
    username = st.session_state.get("username")

    if auth_status is False:
        st.error("Username/password is incorrect.")
        log_warning(f"Failed login attempt for username: {username}")
        st.stop()
    elif auth_status is None:
        st.warning("Please enter your username and password.")
        st.stop()

    role = config["credentials"]["usernames"][username]["role"]
    log_info(f"User '{username}' logged in as {role}")
    return authenticator, name, username, role