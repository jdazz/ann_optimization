# FILE: app.py (Updated to save config on button click)

import streamlit as st
import os
import optuna

# Custom module imports
from utils.config_utils import load_config, save_config
from ui.sidebar import render_sidebar
from ui.main_page import render_results, render_log, render_footer
from core.pipeline import run_training_pipeline

# --- Constants ---
CONFIG_PATH = os.path.join(os.getcwd(), "config.yaml")

# --- Page Config & Initial Setup ---
st.set_page_config(layout="wide")

# Load default config *once*
try:
    default_config = load_config(CONFIG_PATH)
except FileNotFoundError:
    st.error(f"FATAL: config.yaml not found at {CONFIG_PATH}")
    st.stop()

# --- Session State Initialization ---
if 'default_config' not in st.session_state:
    st.session_state.default_config = default_config
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'log_output' not in st.session_state:
    st.session_state.log_output = ""
if 'uploaded_train_file' not in st.session_state:
    st.session_state.uploaded_train_file = None
if 'uploaded_test_file' not in st.session_state:
    st.session_state.uploaded_test_file = None
# NEW: Store the mutable config from the sidebar
if 'current_ui_config' not in st.session_state:
    st.session_state.current_ui_config = default_config


# --- Render Sidebar ---
# This function updates st.session_state.current_ui_config with the latest widget values
render_sidebar(st.session_state.default_config, CONFIG_PATH)

# --- Main App Interface ---
st.title("ANN Optimization Dashboard")

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# --- Training Button Logic ---
if st.button("Start Training and Testing", type="primary"):
    
    # 1. PRE-TRAINING STEP: SAVE THE CURRENT UI CONFIG
    try:
        # Use the configuration dict updated by the sidebar widgets
        save_config(st.session_state.current_ui_config, CONFIG_PATH)
        st.toast("Configuration saved successfully!")
    except Exception as e:
        st.error(f"Error saving configuration: {e}")
        st.stop()
    
    st.session_state.training_results = None
    st.session_state.log_output = ""
    
    # 2. FILE VALIDATION
    if not st.session_state.uploaded_train_file or not st.session_state.uploaded_test_file:
        st.error("Please upload both Training and Testing data files in the sidebar.")
    else:
        # 3. RUN PIPELINE
        with st.spinner("Running optimization and testing..."):
            
            # Re-load config (which was just saved) to ensure core/pipeline.py
            # and src files read the absolute latest version from disk.
            current_config = load_config(CONFIG_PATH) 

            results, log, study, error = run_training_pipeline(
                st.session_state.uploaded_train_file,
                st.session_state.uploaded_test_file
            )
            
            # 4. STORE RESULTS
            st.session_state.log_output = log
            if error:
                st.error(f"An error occurred during training: {error}")
            if results:
                st.session_state.training_results = results

# --- Display Results ---
if st.session_state.training_results:
    # Use the configuration that was just saved/used for training
    try:
        # We need the current config for the plot option
        current_config = load_config(CONFIG_PATH) 
        plot_config = current_config.get("display", {})
        render_results(st.session_state.training_results, plot_config)
    except FileNotFoundError:
        st.error("Could not load config file to display results.")

# --- Display Log ---
render_log(st.session_state.log_output)

# --- Display Footer ---
render_footer()