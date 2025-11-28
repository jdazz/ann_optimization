# app.py - ANN Optimization Dashboard (Rewritten)

import os
import time
import json
import io
import zipfile
import threading
from collections import deque
import queue

import numpy as np
import pandas as pd
import streamlit as st
import optuna.visualization as ov
from sklearn.preprocessing import StandardScaler

# --- Project Modules ---
from src.dataset import Dataset
from core.pipeline import run_training_pipeline
from src.model_test import test
from src.plot import make_plot, make_plotly_figure

# --- Utilities ---
from ui.config_ui import render_config_ui
from utils.config_utils import load_config, save_config
from utils.initialize_session import initialize_session_state
from utils.queue_utils import process_queue_updates
from utils.state_manager import handle_thread_reattachment
from utils.data_utils import detect_and_handle_data_input

# --- UI Components ---
from ui.control_panel import render_control_panel
from ui.results_panel import render_final_results


# ----------------------------------------------------------------------------------
# 1. Config Loading
# ----------------------------------------------------------------------------------
CONFIG_PATH = "config.yaml"
DEFAULT_CONFIG = load_config(CONFIG_PATH)


# ----------------------------------------------------------------------------------
# 2. App Setup
# ----------------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("ANN Optimization Dashboard")

# Initialize base session state (may already set current_ui_config, etc.)
initialize_session_state()

# Ensure current_ui_config exists and is a dict
if "current_ui_config" not in st.session_state or st.session_state.current_ui_config is None:
    # Use a copy of DEFAULT_CONFIG as working config in the UI
    st.session_state.current_ui_config = DEFAULT_CONFIG.copy()

# OPTIONAL: Initialize selected features/target from config if not already set
variables_cfg = st.session_state.current_ui_config.get("variables", {})
if "selected_features" not in st.session_state:
    input_names = variables_cfg.get("input_names", "")
    st.session_state.selected_features = [
        col.strip() for col in str(input_names).split("\n") if col.strip()
    ]

if "selected_targets" not in st.session_state:
    st.session_state.selected_targets = variables_cfg.get("output_names", None)

# Critical thread re-attachment (if a training thread is already running in session_state)
handle_thread_reattachment()


# ----------------------------------------------------------------------------------
# 3. Data Input Handling, Persistence, and Config Update
# ----------------------------------------------------------------------------------
st.subheader("Upload Training & Testing Files")

# Upload widgets (ephemeral file objects on each rerun)
train_file = st.file_uploader(
    "Upload Training File",
    type=["csv", "xlsx", "json"],
    key="train_file_ephemeral",
)
test_file = st.file_uploader(
    "Upload Test File (optional)",
    type=["csv", "xlsx", "json"],
    key="test_file_ephemeral",
)

# --- Auto detect columns when train file is provided ---
if train_file is not None:
    try:
        detected_columns = Dataset.read_columns_from_source(train_file)
    except Exception as e:
        st.error(f"❌ Failed to read columns: {e}")
        detected_columns = []

    st.session_state.available_columns = detected_columns
    # persist file objects in session_state to survive reruns
    st.session_state.persistent_train_file_obj = train_file
    st.session_state.persistent_test_file_obj = test_file
else:
    st.session_state.available_columns = []
    st.session_state.persistent_train_file_obj = None
    st.session_state.persistent_test_file_obj = None


# ----------------------------------------------------------------------------------
# 4. Feature + Target Selectors (only when columns available)
# ----------------------------------------------------------------------------------
if st.session_state.available_columns:

    st.subheader("Select Input and Output Columns")

    available_cols = st.session_state.available_columns
    cfg = st.session_state.current_ui_config
    vars_cfg = cfg.setdefault("variables", {})

    # --------------------------------------------------------------------------
    # 4.1 Feature Multiselect
    # --------------------------------------------------------------------------
    # Keep only features that still exist in the uploaded file
    current_features = [
        f for f in st.session_state.selected_features
        if f in available_cols
    ]

    # If nothing selected yet, fall back to config.yaml > variables.input_names
    if not current_features:
        input_names_from_cfg = vars_cfg.get("input_names", "")
        if input_names_from_cfg:
            from_cfg_list = [
                c.strip()
                for c in str(input_names_from_cfg).split("\n")
                if c.strip() and c.strip() in available_cols
            ]
            current_features = from_cfg_list

    # If still empty, default to all columns except the last one
    if not current_features and available_cols:
        current_features = available_cols[:-1]

    st.multiselect(
        "Select feature columns (X)",
        options=available_cols,
        default=current_features,
        key="selected_features",  # stored in session_state
    )

    # --------------------------------------------------------------------------
    # 4.2 Target Selectbox
    # --------------------------------------------------------------------------
    # Target options exclude selected features
    target_options = [
        col for col in available_cols
        if col not in st.session_state.selected_features
    ]

    # Determine desired target: session_state → config → fallback to last column
    desired_target_name = st.session_state.selected_targets
    if desired_target_name is None:
        desired_target_name = vars_cfg.get("output_names", None)
    if desired_target_name is None and available_cols:
        desired_target_name = available_cols[-1]

    # Safe index computation
    if target_options:
        if desired_target_name in target_options:
            default_target_index = target_options.index(desired_target_name)
        else:
            default_target_index = 0
    else:
        target_options = ["(No Target Available)"]
        default_target_index = 0

    st.selectbox(
        "Select target column (Y)",
        options=target_options,
        index=default_target_index,
        key="selected_targets",  # stored in session_state
    )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 4.3 Sync into current_ui_config + advanced textareas + SAVE TO YAML
    # --------------------------------------------------------------------------
    # Features → newline-separated string
    feature_list = st.session_state.get("selected_features", [])
    feature_string = "\n".join(feature_list)
    vars_cfg["input_names"] = feature_string

    # Target
    target_value = st.session_state.get("selected_targets", "")
    if target_value == "(No Target Available)":
        target_string = ""
    else:
        target_string = str(target_value) if target_value else ""
    vars_cfg["output_names"] = target_string

    # These are used by config_ui.py (advanced textareas)
    st.session_state["input_vars_advanced"] = feature_string
    st.session_state["output_vars_global"] = target_string

    # Persist to config.yaml on every change
    # (Assumes save_config(config_dict, path) signature)
    save_config(st.session_state.current_ui_config, CONFIG_PATH)


# ----------------------------------------------------------------------------------
# 5. Config UI (model / HPO / CV settings)
# ----------------------------------------------------------------------------------
# render_config_ui should read/write st.session_state.current_ui_config internally
render_config_ui(
    DEFAULT_CONFIG,
    CONFIG_PATH,
)

st.markdown("---")


# ----------------------------------------------------------------------------------
# 6. Data Processing, Splitting, and Standardization Setup
# ----------------------------------------------------------------------------------
user_set_split_ratio = st.session_state.current_ui_config.get(
    "cross_validation", {}
).get("test_split_ratio", 0.2)

# Use persistent file objects for actual data handling
detect_and_handle_data_input(
    st.session_state.persistent_train_file_obj,
    st.session_state.persistent_test_file_obj,
    user_set_split_ratio,
)

# Scaling warning based on current_ui_config
should_standardize_config = st.session_state.current_ui_config.get(
    "cross_validation", {}
).get("standardize_features", False)

if should_standardize_config:
    st.warning("⚠️ **Model Scaling Active**")
    st.markdown(
        "Your input features were scaled. To ensure correct predictions with the "
        "downloaded model, use the **`scaler_params.json`** values to transform "
        "new input data (`x`) before feeding it to the model."
    )
    st.code("x_scaled = (x - mean) / std")


# ----------------------------------------------------------------------------------
# 7. Main Panel: Control & Live Status
# ----------------------------------------------------------------------------------
render_control_panel()


# ----------------------------------------------------------------------------------
# 8. Final Results Section
# ----------------------------------------------------------------------------------
render_final_results()


# ----------------------------------------------------------------------------------
# 9. Auto-Rerun and Thread Completion Handler (Polling Logic)
# ----------------------------------------------------------------------------------
# 1. Process queue messages from backend thread
queue_data_received = process_queue_updates()

# 2. Decide whether to poll again
should_poll = st.session_state.get("is_running", False) or st.session_state.get(
    "is_resumable", False
)

if queue_data_received:
    st.rerun()
elif should_poll:
    # Gentle polling to keep UI responsive but not hammer the CPU
    time.sleep(1)
    st.rerun()