# app.py - ANN Optimization Dashboard (Filesystem-driven reloads)

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
from utils.run_manager import (
    make_config_hash,
    derive_dataset_name,
    find_existing_run,
    find_any_in_progress,
    read_value_file,
    zip_run_dir,
)

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

# Initialize base session state (may already set current_ui_config, log_messages, etc.)
initialize_session_state()
handle_thread_reattachment()

# Ensure log_messages exists and is a deque
if "log_messages" not in st.session_state or st.session_state.log_messages is None:
    st.session_state.log_messages = deque(maxlen=500)

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


# ----------------------------------------------------------------------------------
# 2.1 Hydrate run state + metrics/logs from disk (no thread reattachment)
# ----------------------------------------------------------------------------------
def hydrate_from_disk():
    """
    Use the filesystem as the source of truth for the current run.
    - Detect any __IN_PROGRESS run via find_any_in_progress("runs")
    - Update run_status, current_run_dir, is_running
    - Hydrate metrics and summary logs from the run folder
    """

    status, path, run_id = find_any_in_progress("runs")

    # Persist basic run info in session_state
    st.session_state.run_status = status
    st.session_state.current_run_dir = path

    # For the UI, treat "IN_PROGRESS" as running; otherwise not running
    st.session_state.is_running = (status == "IN_PROGRESS")

    # If we didn't find an in-progress run, we still might want the last completed run
    # (optional: use find_existing_run based on config+dataset)
    if not path:
        try:
            cfg_hash = make_config_hash(st.session_state.config)
            dataset_name = st.session_state.get("dataset_name")
            if not dataset_name:
                train_file = st.session_state.get("persistent_train_file_obj")
                if train_file and hasattr(train_file, "name"):
                    dataset_name = derive_dataset_name(
                        type("Tmp", (), {"name": train_file.name})()
                    )
            if not dataset_name:
                dataset_name = "dataset"

            status_existing, path_existing = find_existing_run("runs", dataset_name, cfg_hash)
            if path_existing:
                status = status_existing
                path = path_existing
                st.session_state.run_status = status
                st.session_state.current_run_dir = path
        except Exception:
            # If anything goes wrong here, just skip and leave whatever we had
            pass

    # If we still don't have a path, nothing to hydrate
    if not path:
        return

    # -----------------------------
    # Hydrate metrics from files
    # -----------------------------
    st.session_state.best_loss_so_far = (
        read_value_file(path, "best_cv_loss.txt") or float("inf")
    )
    st.session_state.best_intermediate_r2 = read_value_file(
        path, "best_intermediate_r2.txt"
    )
    st.session_state.best_intermediate_nmae = read_value_file(
        path, "best_intermediate_nmae.txt"
    )
    st.session_state.best_intermediate_accuracy = read_value_file(
        path, "best_intermediate_accuracy.txt"
    )

    best_metrics = read_value_file(path, "best_metrics.json")
    if best_metrics:
        # Allow either already-parsed or raw JSON string
        if isinstance(best_metrics, str):
            try:
                best_metrics = json.loads(best_metrics)
            except Exception:
                pass
        st.session_state.live_best_test_metrics = best_metrics

    # -----------------------------
    # Hydrate logs from summary.txt
    # -----------------------------
    summary_path = os.path.join(path, "summary.txt")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                lines = f.readlines()

            # Reset log_messages and load from file (newest first)
            st.session_state.log_messages.clear()
            for line in reversed(lines):
                st.session_state.log_messages.append(line.rstrip("\n"))

        except Exception:
            # If summary fails, we just keep whatever logs we had
            pass

    # Add warning for in-progress runs
    if status == "IN_PROGRESS" and run_id:
        st.session_state.log_messages.appendleft(
            f"⚠️ Found in-progress run on disk ({run_id}). "
            f"Live updates are being read from the run folder."
        )


# Run hydration on every script execution (i.e., on every rerun)
hydrate_from_disk()

# ----------------------------------------------------------------------------------
# 2.2 Completed run archives listing
# ----------------------------------------------------------------------------------
def list_run_archives(base_dir="runs"):
    archives = []
    if not os.path.isdir(base_dir):
        return archives
    for name in os.listdir(base_dir):
        if name.endswith(".zip"):
            path = os.path.join(base_dir, name)
            if os.path.isfile(path):
                archives.append((name, path))
    return sorted(archives, reverse=True)


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
    save_config(st.session_state.current_ui_config, CONFIG_PATH)


# ----------------------------------------------------------------------------------
# 5. Config UI (model / HPO / CV settings)
# ----------------------------------------------------------------------------------
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
# 8. Final Results Section + Completed Archives
# ----------------------------------------------------------------------------------
render_final_results()

# Completed run archives display
archives = list_run_archives()
if archives:
    st.subheader("Completed Runs (ZIP)")
    for name, path in archives:
        try:
            with open(path, "rb") as f:
                data = f.read()
            st.download_button(
                label=f"Download {name}",
                data=data,
                file_name=name,
                mime="application/zip",
            )
        except Exception:
            st.write(f"{name} (unreadable)")
else:
    st.info("No completed run archives found yet.")


# ----------------------------------------------------------------------------------
# 9. Auto-Rerun and Filesystem-based Polling Logic
# ----------------------------------------------------------------------------------
# 1. Process queue messages from backend thread (still used when the thread exists
#    in the same Streamlit process; but reloads no longer depend on reattaching)
queue_data_received = process_queue_updates()

# 2. Decide whether to poll again
# Treat "IN_PROGRESS" status from the run folder as the source of truth
run_status = st.session_state.get("run_status")
should_poll = (run_status == "IN_PROGRESS") or st.session_state.get("is_resumable", False)

if queue_data_received:
    st.rerun()
elif should_poll:
    # Gentle polling to keep UI responsive but not hammer the CPU
    time.sleep(1)
    st.rerun()
