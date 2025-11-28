# app.py - REVISED MODULAR STRUCTURE

import streamlit as st
import pandas as pd
import threading
import time
import os
import yaml
import json
import io
import zipfile
import numpy as np
from collections import deque
import queue
from sklearn.preprocessing import StandardScaler
import optuna.visualization as ov

# --- Import Project Modules (Keep these) ---
from src.dataset import Dataset # <-- We need this for read_columns_from_source
from core.pipeline import run_training_pipeline
from src.model_test import test
from src.plot import make_plot, make_plotly_figure

# --- Import Utilities ---
# NOTE: render_config_ui must be updated to handle column selection
from ui.config_ui import render_config_ui 
from utils.config_utils import load_config, save_config
from utils.initialize_session import initialize_session_state
from utils.queue_utils import process_queue_updates
from utils.state_manager import handle_thread_reattachment
from utils.data_utils import detect_and_handle_data_input

# --- Import UI Components ---
from ui.control_panel import render_control_panel
from ui.results_panel import render_final_results


CONFIG_PATH = "config.yaml"
DEFAULT_CONFIG = load_config(CONFIG_PATH)


# ----------------------------------------------------------------------------------
## 3. Main App Execution (The Controller)
# ----------------------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("ANN Optimization Dashboard")

initialize_session_state()
# print(st.session_state.is_resumable, "is resumable")
# print(st.session_state.is_running, "is running")


# --- CRITICAL THREAD RE-ATTACHMENT LOGIC ---
handle_thread_reattachment()


# --- 4. Sidebar Integration & Data Input Handling ---

# Get the ephemeral file objects from the last upload (which will be None on reload)
ephemeral_train_file = st.session_state.get('uploaded_train_file')
ephemeral_test_file = st.session_state.get('uploaded_test_file')

# --- CRITICAL: FILE PERSISTENCE LOGIC ---
# If a NEW file is uploaded, update the persistent copy.
if ephemeral_train_file is not None:
    st.session_state.persistent_train_file_obj = ephemeral_train_file
elif 'persistent_train_file_obj' not in st.session_state:
    st.session_state.persistent_train_file_obj = None

if ephemeral_test_file is not None:
    st.session_state.persistent_test_file_obj = ephemeral_test_file
elif 'persistent_test_file_obj' not in st.session_state:
    st.session_state.persistent_test_file_obj = None
# ------------------------------------------

# Use the PERSISTENT file objects for the rest of the application logic
uploaded_train_file_to_use = st.session_state.persistent_train_file_obj
uploaded_test_file_to_use = st.session_state.persistent_test_file_obj


# --- NEW: COLUMN DETECTION AND WIDGET LOGIC ---
# This runs immediately after file persistence is resolved.

detected_columns = []
if uploaded_train_file_to_use is not None:
    # Call the new static method to read columns
    detected_columns = Dataset.read_columns_from_source(uploaded_train_file_to_use)
    
# Store the detected columns in session state for the UI
st.session_state.available_columns = detected_columns

# Pass the columns to the config UI so it can render the selectors
render_config_ui(DEFAULT_CONFIG, CONFIG_PATH, available_columns=detected_columns)
st.markdown("---") 

# Continue with the rest of data processing...
user_set_split_ratio = st.session_state.current_ui_config.get("cross_validation", {}).get("test_split_ratio", 0.2)

# Pass the persistent copies to the detection utility
# NOTE: This step assumes target/feature columns are now selected and updated in st.session_state.config
detect_and_handle_data_input(
    uploaded_train_file_to_use, 
    uploaded_test_file_to_use, 
    user_set_split_ratio
)


# --- Scaling Warning ---
should_standardize_config = st.session_state.config.get("cross_validation", {}).get("standardize_features", False)
if should_standardize_config:
    st.warning("⚠️ **Model Scaling Active**")
    st.markdown(
        "Your input features were scaled. To ensure correct predictions with the downloaded model, use the **`scaler_params.json`** values to transform new input data (`x`) before feeding it to the model."
    )
    st.code("x_scaled = (x - mean) / std")
    
# ----------------------------------------------------------------------------------
## 5. Main Panel: Control and Live Status (Delegated to ui/control_panel.py)
# ----------------------------------------------------------------------------------
render_control_panel()


# ----------------------------------------------------------------------------------
## 6. Final Results Section (Delegated to ui/results_panel.py)
# ----------------------------------------------------------------------------------
render_final_results()


# ----------------------------------------------------------------------------------
## 7. Auto-Rerun and Thread Completion Handler (FIXED POLLING LOGIC)
# ----------------------------------------------------------------------------------

# 1. Process Queue
queue_data_received = process_queue_updates()

# 2. Polling Logic: Rerun immediately if data was received, or if we expect data soon.
should_poll = st.session_state.get('is_running', False) or st.session_state.get('is_resumable', False)

if queue_data_received:
    st.rerun() 
elif should_poll:
    time.sleep(0.5) # Reduced sleep time for better responsiveness
    st.rerun()
    